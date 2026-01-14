"""
TRAINING SCRIPT FOR CAUSALLY-CONSTRAINED DISENTANGLED BDI VAE (C²D-BDI)

This is the training script for the revolutionary C²D-BDI model that properly
implements Theory of Mind with:

1. SEPARATE FEATURE BRANCHES for Belief (spatial) and Desire (preferences)
2. β-TCVAE for disentanglement within each VAE
3. MUTUAL INFORMATION MINIMIZATION between z_b and z_d
4. PATH PROGRESS CONDITIONING for temporal awareness
5. AUXILIARY RECONSTRUCTION TASKS (transitions for belief, categories for desire)

TRAINING PHASES:
1. Warmup: Focus on reconstruction, low β weights
2. Disentanglement: Gradually increase β for TC penalty
3. Refinement: Full loss with MI minimization

MONITORING:
- Track disentanglement metrics (TC loss, MI between z_b/z_d)
- Track prediction accuracy by progress percentile
- Track VAE quality metrics (reconstruction, KL)

Usage:
    python -m models.vae_bdi_simple.train_bdi_vae_v2 [OPTIONS]

Example:
    python -m models.vae_bdi_simple.train_bdi_vae_v2 --num_epochs 100 --batch_size 256
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

# Set environment
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.vae_bdi_simple.bdi_vae_v2_model import CausallyConstrainedBDIVAE
from models.vae_bdi_simple.bdi_dataset_v2 import (
    BDIVAEDatasetV2, 
    collate_bdi_samples_v2,
    TemporalConsistencyBatchSampler,
)
from models.utils.data_loader import load_simulation_data
from models.utils.utils import get_device, set_seed, save_checkpoint, AverageMeter

# W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


# =============================================================================
# ENHANCED LOGGING
# =============================================================================

class EnhancedWandBLogger:
    """Enhanced W&B logger with disentanglement metrics and progress tracking."""
    
    def __init__(
        self, 
        project_name: str = "bdi-tom-v2",
        config: Dict = None,
        run_name: str = None,
    ):
        self.enabled = WANDB_AVAILABLE
        self.global_step = 0
        
        if self.enabled:
            wandb.init(
                project=project_name,
                entity="nigeldoering-uc-san-diego",
                config=config or {},
                name=run_name,
            )
            print(f"✅ W&B initialized (run: {run_name or 'auto'})!")
        else:
            print("⚠️  W&B disabled")
    
    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        metrics: Dict[str, float],
        lr: float,
    ):
        """Log batch-level metrics."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'learning_rate': lr,
            'global_step': self.global_step,
        }
        
        for key, value in metrics.items():
            log_dict[f'batch/{key}'] = value
        
        wandb.log(log_dict, step=self.global_step)
        self.global_step += 1
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        disentanglement_metrics: Dict[str, float] = None,
        progress_metrics: Dict[str, Dict[str, float]] = None,
    ):
        """Log epoch-level metrics with disentanglement tracking."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'learning_rate': lr,
        }
        
        # Standard metrics
        for key, value in train_metrics.items():
            log_dict[f'train/{key}'] = value
        
        for key, value in val_metrics.items():
            log_dict[f'val/{key}'] = value
        
        # Disentanglement metrics
        if disentanglement_metrics:
            for key, value in disentanglement_metrics.items():
                log_dict[f'disentangle/{key}'] = value
        
        # Progress-based metrics (accuracy by path completion)
        if progress_metrics:
            for progress_bin, metrics in progress_metrics.items():
                for key, value in metrics.items():
                    log_dict[f'progress_{progress_bin}/{key}'] = value
        
        wandb.log(log_dict, step=self.global_step)
        self.global_step += 1
    
    def log_model_info(self, total_params: int, trainable_params: int, config: Dict):
        """Log model architecture information."""
        if not self.enabled:
            return
        
        info = {
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
        }
        
        for key, value in config.items():
            info[f'config/{key}'] = value
        
        wandb.log(info, step=self.global_step)
        self.global_step += 1
    
    def log_summary(
        self,
        best_epoch: int,
        best_val_acc: float,
        total_epochs: int,
        total_time_hours: float,
    ):
        """Log training summary."""
        if not self.enabled:
            return
        
        wandb.log({
            'summary/best_epoch': best_epoch,
            'summary/best_val_goal_acc': best_val_acc,
            'summary/total_epochs': total_epochs,
            'summary/total_time_hours': total_time_hours,
        })
    
    def finish(self):
        if self.enabled:
            wandb.finish()


# =============================================================================
# LOSS COMPUTATION
# =============================================================================

def compute_temporal_consistency_loss(
    batch_belief_z: torch.Tensor,
    batch_desire_z: torch.Tensor,
    batch_traj_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute temporal consistency losses for samples from the same trajectory.
    
    - Desire should be HIGHLY consistent within a trajectory (same goal)
    - Belief should evolve smoothly (cosine similarity with neighbors)
    
    Args:
        batch_belief_z: [batch, belief_dim] belief latents
        batch_desire_z: [batch, desire_dim] desire latents
        batch_traj_ids: [batch] trajectory IDs
        device: torch device
    
    Returns:
        desire_consistency: Scalar loss (lower = more consistent desires)
        belief_smoothness: Scalar loss (lower = smoother belief evolution)
    """
    unique_trajs = torch.unique(batch_traj_ids)
    
    desire_losses = []
    belief_losses = []
    
    for traj_id in unique_trajs:
        mask = batch_traj_ids == traj_id
        if mask.sum() < 2:
            continue
        
        traj_desire_z = batch_desire_z[mask]  # [n_samples, desire_dim]
        traj_belief_z = batch_belief_z[mask]  # [n_samples, belief_dim]
        
        # Desire consistency: All desires in same trajectory should be similar
        # Use variance across samples as loss (minimize variance)
        desire_mean = traj_desire_z.mean(dim=0, keepdim=True)
        desire_variance = ((traj_desire_z - desire_mean) ** 2).mean()
        desire_losses.append(desire_variance)
        
        # Belief smoothness: Consecutive beliefs should be similar
        # Use cosine similarity between consecutive samples
        if traj_belief_z.shape[0] >= 2:
            belief_sim = nn.functional.cosine_similarity(
                traj_belief_z[:-1], traj_belief_z[1:], dim=-1
            )
            # Loss = 1 - similarity (want high similarity)
            belief_smoothness = (1 - belief_sim).mean()
            belief_losses.append(belief_smoothness)
    
    if desire_losses:
        desire_consistency = torch.stack(desire_losses).mean()
    else:
        desire_consistency = torch.tensor(0.0, device=device)
    
    if belief_losses:
        belief_smoothness = torch.stack(belief_losses).mean()
    else:
        belief_smoothness = torch.tensor(0.0, device=device)
    
    return desire_consistency, belief_smoothness


# =============================================================================
# TRAINING EPOCH
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    epoch: int,
    config: argparse.Namespace,
    wandb_logger: Optional[EnhancedWandBLogger] = None,
    scheduler = None,  # Added scheduler parameter
) -> Dict[str, float]:
    """
    Train for one epoch with C²D-BDI model.
    
    Losses:
    1. VAE losses (belief TCVAE, desire TCVAE, intention VAE)
    2. Mutual information minimization (z_b, z_d)
    3. Prediction losses (goal, nextstep, category)
    4. Temporal consistency (optional, if batch sampler supports it)
    5. Progress prediction (optional)
    """
    model.train()
    
    # Initialize meters
    meters = defaultdict(AverageMeter)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        agent_id = batch['agent_id'].to(device)
        path_progress = batch['path_progress'].to(device)
        traj_id = batch['traj_id'].to(device)
        
        # Forward pass
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            path_progress=path_progress,
            compute_loss=True,
            next_node_idx=next_node_idx,  # For belief auxiliary
            goal_cat_idx=goal_cat_idx,  # For desire auxiliary
        )
        
        # ================================================================
        # NaN DETECTION (EARLY WARNING)
        # ================================================================
        nan_detected = False
        nan_source = None
        
        # Check for NaN in outputs
        for key in ['belief_z', 'desire_z', 'intention_z', 'goal', 'nextstep']:
            if key in outputs and torch.isnan(outputs[key]).any():
                nan_detected = True
                nan_source = f"output[{key}]"
                break
        
        # Check for NaN in losses
        if 'total_vae_loss' in outputs and torch.isnan(outputs['total_vae_loss']):
            nan_detected = True
            nan_source = "total_vae_loss"
        
        if nan_detected:
            print(f"\n⚠️  NaN detected in {nan_source} at batch {batch_idx}!")
            print(f"   Skipping this batch...")
            continue
        
        # ================================================================
        # COMPUTE LOSSES
        # ================================================================
        
        # 1. Prediction losses
        loss_goal = criterion['goal'](outputs['goal'], goal_idx)
        loss_nextstep = criterion['nextstep'](outputs['nextstep'], next_node_idx)
        loss_category = criterion['category'](outputs['category'], goal_cat_idx)
        
        # Weighted prediction loss
        pred_loss = (
            config.goal_weight * loss_goal +
            config.nextstep_weight * loss_nextstep +
            config.category_weight * loss_category
        )
        
        # 2. VAE losses (already computed in forward)
        vae_loss = outputs['total_vae_loss']
        
        # 3. Temporal consistency loss (optional)
        temporal_loss = torch.tensor(0.0, device=device)
        if config.use_temporal_consistency:
            desire_consistency, belief_smoothness = compute_temporal_consistency_loss(
                outputs['belief_z'], outputs['desire_z'], traj_id, device
            )
            temporal_loss = (
                config.desire_consistency_weight * desire_consistency +
                config.belief_smoothness_weight * belief_smoothness
            )
        
        # 4. Progress prediction loss (optional)
        progress_loss = torch.tensor(0.0, device=device)
        if model.use_progress and 'progress_loss' in outputs:
            progress_loss = config.progress_weight * outputs['progress_loss']
        
        # Total loss
        total_loss = pred_loss + config.vae_weight * vae_loss + temporal_loss + progress_loss
        
        # NaN check before backward
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n⚠️  NaN/Inf in total_loss at batch {batch_idx}!")
            print(f"   pred_loss: {pred_loss.item() if not torch.isnan(pred_loss) else 'NaN'}")
            print(f"   vae_loss: {vae_loss.item() if not torch.isnan(vae_loss) else 'NaN'}")
            print(f"   Skipping this batch...")
            continue
        
        # ================================================================
        # BACKWARD
        # ================================================================
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step scheduler (OneCycleLR needs per-batch stepping)
        if scheduler is not None:
            scheduler.step()
        
        # ================================================================
        # COMPUTE ACCURACIES
        # ================================================================
        goal_acc = (outputs['goal'].argmax(dim=1) == goal_idx).float().mean().item() * 100
        nextstep_acc = (outputs['nextstep'].argmax(dim=1) == next_node_idx).float().mean().item() * 100
        category_acc = (outputs['category'].argmax(dim=1) == goal_cat_idx).float().mean().item() * 100
        
        # ================================================================
        # UPDATE METERS
        # ================================================================
        meters['loss'].update(total_loss.item(), batch_size)
        meters['pred_loss'].update(pred_loss.item(), batch_size)
        meters['vae_loss'].update(vae_loss.item(), batch_size)
        meters['goal_acc'].update(goal_acc, batch_size)
        meters['nextstep_acc'].update(nextstep_acc, batch_size)
        meters['category_acc'].update(category_acc, batch_size)
        
        # VAE component losses
        meters['belief_loss'].update(outputs['belief_loss'].item(), batch_size)
        meters['belief_recon'].update(outputs['belief_recon_loss'].item(), batch_size)
        meters['belief_tc'].update(outputs['belief_tc_loss'].item(), batch_size)
        meters['desire_loss'].update(outputs['desire_loss'].item(), batch_size)
        meters['desire_recon'].update(outputs['desire_recon_loss'].item(), batch_size)
        meters['desire_tc'].update(outputs['desire_tc_loss'].item(), batch_size)
        meters['intention_loss'].update(outputs['intention_loss'].item(), batch_size)
        meters['bd_mi'].update(outputs['bd_mi_loss'].item(), batch_size)
        
        if config.use_temporal_consistency:
            meters['temporal_loss'].update(temporal_loss.item(), batch_size)
        
        # Log to W&B
        if wandb_logger is not None and batch_idx % 10 == 0:
            batch_metrics = {
                'loss': total_loss.item(),
                'goal_acc': goal_acc,
                'belief_tc': outputs['belief_tc_loss'].item(),
                'desire_tc': outputs['desire_tc_loss'].item(),
                'bd_mi': outputs['bd_mi_loss'].item(),
            }
            lr = optimizer.param_groups[0]['lr']
            wandb_logger.log_batch(epoch, batch_idx, batch_metrics, lr)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{meters['loss'].avg:.4f}",
            'goal': f"{meters['goal_acc'].avg:.1f}%",
            'MI': f"{meters['bd_mi'].avg:.4f}",
        })
    
    return {k: v.avg for k, v in meters.items()}


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    config: argparse.Namespace,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Validate C²D-BDI model with progress-stratified metrics.
    
    Returns:
        overall_metrics: Dict of averaged metrics
        progress_metrics: Dict of metrics by progress bin (0-25%, 25-50%, etc.)
    """
    model.eval()
    
    meters = defaultdict(AverageMeter)
    
    # Progress-stratified tracking
    progress_bins = ['0-25', '25-50', '50-75', '75-100']
    progress_meters = {
        bin_name: defaultdict(AverageMeter) for bin_name in progress_bins
    }
    
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch in pbar:
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        agent_id = batch['agent_id'].to(device)
        path_progress = batch['path_progress'].to(device)
        
        # Forward
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            path_progress=path_progress,
            compute_loss=True,
            next_node_idx=next_node_idx,
            goal_cat_idx=goal_cat_idx,
        )
        
        # Prediction losses
        loss_goal = criterion['goal'](outputs['goal'], goal_idx)
        loss_nextstep = criterion['nextstep'](outputs['nextstep'], next_node_idx)
        loss_category = criterion['category'](outputs['category'], goal_cat_idx)
        
        pred_loss = (
            config.goal_weight * loss_goal +
            config.nextstep_weight * loss_nextstep +
            config.category_weight * loss_category
        )
        
        vae_loss = outputs['total_vae_loss']
        total_loss = pred_loss + config.vae_weight * vae_loss
        
        # Accuracies
        goal_preds = outputs['goal'].argmax(dim=1)
        nextstep_preds = outputs['nextstep'].argmax(dim=1)
        category_preds = outputs['category'].argmax(dim=1)
        
        goal_correct = (goal_preds == goal_idx)
        nextstep_correct = (nextstep_preds == next_node_idx)
        category_correct = (category_preds == goal_cat_idx)
        
        goal_acc = goal_correct.float().mean().item() * 100
        nextstep_acc = nextstep_correct.float().mean().item() * 100
        category_acc = category_correct.float().mean().item() * 100
        
        # Update overall meters
        meters['loss'].update(total_loss.item(), batch_size)
        meters['pred_loss'].update(pred_loss.item(), batch_size)
        meters['vae_loss'].update(vae_loss.item(), batch_size)
        meters['goal_acc'].update(goal_acc, batch_size)
        meters['nextstep_acc'].update(nextstep_acc, batch_size)
        meters['category_acc'].update(category_acc, batch_size)
        meters['belief_tc'].update(outputs['belief_tc_loss'].item(), batch_size)
        meters['desire_tc'].update(outputs['desire_tc_loss'].item(), batch_size)
        meters['bd_mi'].update(outputs['bd_mi_loss'].item(), batch_size)
        
        # Progress-stratified metrics
        for i in range(batch_size):
            prog = path_progress[i].item()
            
            if prog < 0.25:
                bin_name = '0-25'
            elif prog < 0.5:
                bin_name = '25-50'
            elif prog < 0.75:
                bin_name = '50-75'
            else:
                bin_name = '75-100'
            
            progress_meters[bin_name]['goal_acc'].update(
                goal_correct[i].float().item() * 100, 1
            )
            progress_meters[bin_name]['nextstep_acc'].update(
                nextstep_correct[i].float().item() * 100, 1
            )
        
        pbar.set_postfix({'loss': f"{meters['loss'].avg:.4f}"})
    
    overall = {k: v.avg for k, v in meters.items()}
    
    progress = {
        bin_name: {k: v.avg for k, v in bin_meters.items()}
        for bin_name, bin_meters in progress_meters.items()
    }
    
    return overall, progress


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_with_splits(
    data_dir: str,
    graph_path: str,
    split_indices_path: str,
    trajectory_filename: str = 'all_trajectories.json',
) -> Tuple:
    """Load data with pre-defined splits."""
    print("\n" + "=" * 100)
    print("📂 LOADING DATA WITH PRE-DEFINED SPLITS")
    print("=" * 100)
    
    trajectories, graph, poi_nodes = load_simulation_data(
        data_dir, graph_path, trajectory_filename
    )
    print(f"✅ Loaded {len(trajectories)} trajectories")
    print(f"✅ Graph: {len(graph.nodes)} nodes, {len(poi_nodes)} POIs")
    
    print(f"\n📊 Loading splits from: {split_indices_path}")
    with open(split_indices_path, 'r') as f:
        split_data = json.load(f)
    
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']
    test_indices = split_data['test_indices']
    
    print(f"   Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    train_trajs = [trajectories[i] for i in train_indices]
    val_trajs = [trajectories[i] for i in val_indices]
    test_trajs = [trajectories[i] for i in test_indices]
    
    return train_trajs, val_trajs, test_trajs, graph, poi_nodes


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train C²D-BDI VAE Model')
    
    # Data paths
    parser.add_argument('--data_dir', type=str,
                       default='data/simulation_data/run_8')
    parser.add_argument('--graph_path', type=str,
                       default='data/processed/ucsd_walk_full.graphml')
    parser.add_argument('--split_indices_path', type=str,
                       default='data/simulation_data/run_8/split_data/split_indices_seed42.json')
    parser.add_argument('--trajectory_filename', type=str,
                       default='all_trajectories.json')
    
    # Model - Embedding
    parser.add_argument('--node_embedding_dim', type=int, default=64)
    parser.add_argument('--temporal_dim', type=int, default=64)
    parser.add_argument('--agent_dim', type=int, default=64)
    parser.add_argument('--fusion_dim', type=int, default=128)
    
    # Model - VAE
    parser.add_argument('--belief_latent_dim', type=int, default=32)
    parser.add_argument('--desire_latent_dim', type=int, default=32)
    parser.add_argument('--intention_latent_dim', type=int, default=64)
    parser.add_argument('--vae_hidden_dim', type=int, default=128)
    parser.add_argument('--vae_num_layers', type=int, default=2)
    
    # Model - Prediction
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=4)
    
    # VAE Loss weights (β-TCVAE)
    parser.add_argument('--beta_belief', type=float, default=1.0,
                       help='β for belief Total Correlation penalty')
    parser.add_argument('--beta_desire', type=float, default=1.0,
                       help='β for desire Total Correlation penalty')
    parser.add_argument('--beta_intention', type=float, default=0.5,
                       help='β for intention KL')
    parser.add_argument('--mi_weight', type=float, default=0.1,
                       help='Weight for MI(z_b, z_d) minimization')
    parser.add_argument('--transition_weight', type=float, default=0.1,
                       help='Weight for belief transition auxiliary')
    parser.add_argument('--category_weight_aux', type=float, default=0.1,
                       help='Weight for desire category auxiliary')
    
    # Training loss weights
    parser.add_argument('--vae_weight', type=float, default=0.01,
                       help='Weight for total VAE loss (start small)')
    parser.add_argument('--goal_weight', type=float, default=1.0)
    parser.add_argument('--nextstep_weight', type=float, default=0.5)
    parser.add_argument('--category_weight', type=float, default=0.5)
    parser.add_argument('--progress_weight', type=float, default=0.1)
    
    # Temporal consistency
    parser.add_argument('--use_temporal_consistency', action='store_true',
                       help='Enable temporal consistency loss')
    parser.add_argument('--desire_consistency_weight', type=float, default=0.1)
    parser.add_argument('--belief_smoothness_weight', type=float, default=0.05)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Data
    parser.add_argument('--min_traj_length', type=int, default=2)
    parser.add_argument('--use_temporal_sampler', action='store_true',
                       help='Use temporal consistency batch sampler')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints/bdi_vae_v2')
    parser.add_argument('--save_every', type=int, default=10)
    
    # Logging
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"🖥️  Device: {device}")
    
    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    train_trajs, val_trajs, test_trajs, graph, poi_nodes = load_data_with_splits(
        args.data_dir,
        args.graph_path,
        args.split_indices_path,
        args.trajectory_filename,
    )
    
    # Create datasets
    print("\n" + "=" * 100)
    print("📊 CREATING C²D-BDI DATASETS (V2)")
    print("=" * 100)
    
    print("\n🔹 Creating training dataset...")
    train_dataset = BDIVAEDatasetV2(
        train_trajs,
        graph,
        poi_nodes,
        min_traj_length=args.min_traj_length,
        include_progress=True,
    )
    
    print("\n🔹 Creating validation dataset...")
    val_dataset = BDIVAEDatasetV2(
        val_trajs,
        graph,
        poi_nodes,
        node_to_idx_map=train_dataset.node_to_idx,
        min_traj_length=args.min_traj_length,
        include_progress=True,
    )
    
    print(f"\n✅ Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Create data loaders
    if args.use_temporal_sampler:
        print("🔄 Using temporal consistency batch sampler")
        train_sampler = TemporalConsistencyBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            samples_per_trajectory=4,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_bdi_samples_v2,
            pin_memory=True if device.type == 'cuda' else False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_bdi_samples_v2,
            pin_memory=True if device.type == 'cuda' else False,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bdi_samples_v2,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model
    print("\n" + "=" * 100)
    print("🏗️  CREATING C²D-BDI VAE MODEL")
    print("=" * 100)
    
    model = CausallyConstrainedBDIVAE(
        num_nodes=len(graph.nodes),
        num_agents=100,
        num_poi_nodes=len(poi_nodes),
        num_categories=7,
        node_embedding_dim=args.node_embedding_dim,
        temporal_dim=args.temporal_dim,
        agent_dim=args.agent_dim,
        fusion_dim=args.fusion_dim,
        belief_latent_dim=args.belief_latent_dim,
        desire_latent_dim=args.desire_latent_dim,
        intention_latent_dim=args.intention_latent_dim,
        vae_hidden_dim=args.vae_hidden_dim,
        vae_num_layers=args.vae_num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads,
        beta_belief=args.beta_belief,
        beta_desire=args.beta_desire,
        beta_intention=args.beta_intention,
        mi_weight=args.mi_weight,
        transition_weight=args.transition_weight,
        category_weight=args.category_weight_aux,
        use_progress=True,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ C²D-BDI Model created!")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   β_belief={args.beta_belief}, β_desire={args.beta_desire}")
    print(f"   MI weight={args.mi_weight}")
    print("=" * 100)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Ensure pct_start is valid (between 0 and 1)
    pct_start = min(args.warmup_epochs / args.num_epochs, 0.3)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        anneal_strategy='cos',
    )
    
    # Loss functions
    criterion = {
        'goal': nn.CrossEntropyLoss(),
        'nextstep': nn.CrossEntropyLoss(),
        'category': nn.CrossEntropyLoss(),
    }
    
    # W&B
    wandb_logger = None
    if not args.no_wandb and WANDB_AVAILABLE:
        config = vars(args)
        config.update({
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_nodes': len(graph.nodes),
            'num_pois': len(poi_nodes),
            'model_type': 'c2d_bdi_vae',
        })
        wandb_logger = EnhancedWandBLogger(
            project_name="bdi-tom-v2",
            config=config,
            run_name=args.wandb_run_name,
        )
        wandb_logger.log_model_info(total_params, trainable_params, vars(args))
    
    # Training loop
    print("\n" + "=" * 100)
    print("🚀 STARTING C²D-BDI TRAINING")
    print("=" * 100 + "\n")
    
    best_val_goal_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*100}")
        print(f"EPOCH {epoch}/{args.num_epochs}")
        print(f"{'='*100}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args, wandb_logger, scheduler
        )
        
        # Validate
        val_metrics, progress_metrics = validate(
            model, val_loader, criterion, device, args
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to W&B
        if wandb_logger is not None:
            disentangle_metrics = {
                'belief_tc': train_metrics.get('belief_tc', 0),
                'desire_tc': train_metrics.get('desire_tc', 0),
                'bd_mi': train_metrics.get('bd_mi', 0),
            }
            wandb_logger.log_epoch(
                epoch, train_metrics, val_metrics, current_lr,
                disentangle_metrics, progress_metrics
            )
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train - Loss: {train_metrics['loss']:.4f} | Goal: {train_metrics['goal_acc']:.1f}% | MI(B,D): {train_metrics['bd_mi']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f} | Goal: {val_metrics['goal_acc']:.1f}%")
        print(f"   Progress Metrics:")
        for bin_name, metrics in progress_metrics.items():
            if 'goal_acc' in metrics:
                print(f"      {bin_name}%: Goal={metrics['goal_acc']:.1f}%")
        
        # Check improvement
        if val_metrics['goal_acc'] > best_val_goal_acc:
            best_val_goal_acc = val_metrics['goal_acc']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch,
                val_metrics['goal_acc'], is_best=True
            )
            print(f"   ✅ New best! Val Goal Acc: {best_val_goal_acc:.1f}%")
        else:
            patience_counter += 1
            print(f"   No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'
            )
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch,
                val_metrics['goal_acc'], is_best=False
            )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏸️  Early stopping at epoch {epoch}")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n" + "=" * 100)
    print("🎉 C²D-BDI TRAINING COMPLETE!")
    print("=" * 100)
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Val Goal Acc: {best_val_goal_acc:.1f}%")
    print(f"   Total Time: {total_time/3600:.2f} hours")
    print("=" * 100 + "\n")
    
    if wandb_logger is not None:
        wandb_logger.log_summary(best_epoch, best_val_goal_acc, epoch, total_time/3600)
        wandb_logger.finish()


if __name__ == '__main__':
    main()
