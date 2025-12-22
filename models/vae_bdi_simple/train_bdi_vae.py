"""
BDI VAE TRAINING SCRIPT

This script trains the hierarchical BDI (Belief-Desire-Intention) VAE model
on the UCSD campus navigation dataset with pre-defined train/val/test splits.

ARCHITECTURE:
- Unified Embedding Pipeline → [Belief VAE, Desire VAE] → Intention VAE → Predictions

LOSS COMPONENTS:
- VAE Reconstruction Losses (belief, desire, intention)
- KL Divergence Losses (β-VAE for disentanglement)
- Prediction Losses (goal, next step, category)

Usage:
    python -m models.vae_bdi_simple.train_bdi_vae [OPTIONS]

Example:
    python -m models.vae_bdi_simple.train_bdi_vae --num_epochs 50 --batch_size 128

Data:
    - Trajectories: data/simulation_data/run_8/trajectories/all_trajectories.json
    - Split indices: data/simulation_data/run_8/split_data/split_indices_seed42.json
    - Graph: data/processed/ucsd_walk_full.graphml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import time

# Set MPS fallback for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.vae_bdi_simple.bdi_vae_model import BDIVAEPredictor
from models.vae_bdi_simple.bdi_dataset import BDIVAEDataset, collate_bdi_samples
from models.utils.data_loader import load_simulation_data
from models.utils.utils import get_device, set_seed, save_checkpoint, AverageMeter

# W&B imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


# ============================================================================
# W&B LOGGER (SAME AS TRANSFORMER)
# ============================================================================

class WandBLogger:
    """Handles W&B logging during training with proper accuracy averaging."""
    
    def __init__(self, project_name: str = "bdi-tom", config: Dict = None, run_name: str = None):
        self.enabled = WANDB_AVAILABLE
        self.global_step = 0
        
        if self.enabled:
            wandb.init(project=project_name, config=config or {}, name=run_name)
            
            # Define metric relationships to tell W&B which metrics use which x-axis
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("batch/*", step_metric="global_step")
            
            print(f"✅ W&B initialized{f' (run: {run_name})' if run_name else ''}!")
        else:
            print("⚠️  W&B disabled")
    
    def log_batch(self, epoch: int, batch_idx: int, batch_size: int, metrics: Dict[str, float], lr: float):
        """Log batch-level metrics."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'learning_rate': lr,
            'global_step': self.global_step,
        }
        
        # Prefix batch metrics with 'batch/'
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
        train_percentile_metrics: Dict[str, Dict[str, float]] = None,
        val_percentile_metrics: Dict[str, Dict[str, float]] = None
    ):
        """
        Log epoch-level metrics: accuracies, losses, VAE losses, and percentiles.
        
        Organizes metrics into subcategories:
        - train/accuracy/* and val/accuracy/* for all accuracy metrics
        - train/loss/* and val/loss/* for all loss components
        - train/vae/* and val/vae/* for VAE-specific losses
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics (averaged)
            val_metrics: Validation metrics (averaged)
            lr: Current learning rate
            train_percentile_metrics: Training percentile metrics (15%, 50%, 85%)
            val_percentile_metrics: Validation percentile metrics (15%, 50%, 85%)
        """
        if not self.enabled:
            return
        
        # First log the epoch value itself (required for step_metric to work)
        wandb.log({'epoch': epoch})
        
        log_dict = {
            'learning_rate': lr,
        }
        
        # === ACCURACY METRICS ===
        # Train accuracies
        log_dict['train/accuracy/goal_top1'] = train_metrics.get('goal_acc', 0)
        log_dict['train/accuracy/goal_top5'] = train_metrics.get('goal_top5_acc', 0)
        log_dict['train/accuracy/next_node'] = train_metrics.get('nextstep_acc', 0)
        log_dict['train/accuracy/goal_category'] = train_metrics.get('category_acc', 0)
        
        # Val accuracies
        log_dict['val/accuracy/goal_top1'] = val_metrics.get('goal_acc', 0)
        log_dict['val/accuracy/goal_top5'] = val_metrics.get('goal_top5_acc', 0)
        log_dict['val/accuracy/next_node'] = val_metrics.get('nextstep_acc', 0)
        log_dict['val/accuracy/goal_category'] = val_metrics.get('category_acc', 0)
        
        # === LOSS METRICS ===
        # Train losses
        log_dict['train/loss/total'] = train_metrics.get('loss', 0)
        log_dict['train/loss/prediction'] = train_metrics.get('total_pred_loss', 0)
        log_dict['train/loss/vae'] = train_metrics.get('total_vae_loss', 0)
        log_dict['train/loss/goal'] = train_metrics.get('loss_goal', 0)
        log_dict['train/loss/next_node'] = train_metrics.get('loss_nextstep', 0)
        log_dict['train/loss/category'] = train_metrics.get('loss_category', 0)
        
        # Val losses
        log_dict['val/loss/total'] = val_metrics.get('loss', 0)
        log_dict['val/loss/prediction'] = val_metrics.get('total_pred_loss', 0)
        log_dict['val/loss/vae'] = val_metrics.get('total_vae_loss', 0)
        log_dict['val/loss/goal'] = val_metrics.get('loss_goal', 0)
        log_dict['val/loss/next_node'] = val_metrics.get('loss_nextstep', 0)
        log_dict['val/loss/category'] = val_metrics.get('loss_category', 0)
        
        # === VAE BREAKDOWN ===
        vae_keys = ['belief_loss', 'belief_recon_loss', 'belief_kl_loss',
                    'desire_loss', 'desire_recon_loss', 'desire_kl_loss',
                    'intention_loss', 'intention_recon_loss', 'intention_kl_loss']
        
        for key in vae_keys:
            if key in train_metrics:
                log_dict[f'train/vae/{key}'] = train_metrics[key]
            if key in val_metrics:
                log_dict[f'val/vae/{key}'] = val_metrics[key]
        
        # === PERCENTILE METRICS ===
        # Log train percentiles
        if train_percentile_metrics is not None:
            for pct in ['15%', '50%', '85%']:
                if pct in train_percentile_metrics:
                    log_dict[f'train/accuracy/goal_top1_{pct}'] = train_percentile_metrics[pct].get('goal_acc', 0)
        
        # Log val percentiles
        if val_percentile_metrics is not None:
            for pct in ['15%', '50%', '85%']:
                if pct in val_percentile_metrics:
                    log_dict[f'val/accuracy/goal_top1_{pct}'] = val_percentile_metrics[pct].get('goal_acc', 0)
        
        # Log all metrics (will automatically use epoch as step due to define_metric)
        wandb.log(log_dict)
    
    def log_model_info(self, total_params: int, trainable_params: int, model_config: Dict):
        """Log model architecture information."""
        if not self.enabled:
            return
        
        info_dict = {
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/parameter_ratio': trainable_params / (total_params + 1e-8),
        }
        
        # Log architecture details
        for key, value in model_config.items():
            info_dict[f'model_config/{key}'] = value
        
        wandb.log(info_dict)
    
    def log_summary(self, best_epoch: int, best_val_acc: float, total_epochs: int, total_time_hours: float):
        """Log training summary statistics."""
        if not self.enabled:
            return
        
        summary_dict = {
            'summary/best_epoch': best_epoch,
            'summary/best_val_goal_acc': best_val_acc,
            'summary/total_epochs_trained': total_epochs,
            'summary/total_training_time_hours': total_time_hours,
            'summary/avg_time_per_epoch': total_time_hours / (total_epochs + 1e-8),
        }
        
        wandb.log(summary_dict)
    
    def log_checkpoint(self, epoch: int, checkpoint_path: str, best: bool = False):
        """Log checkpoint information."""
        if not self.enabled:
            return
        
        checkpoint_dict = {
            'checkpoint/epoch': epoch,
            'checkpoint/path': checkpoint_path,
            'checkpoint/is_best': best,
        }
        
        wandb.log(checkpoint_dict)
    
    def finish(self):
        """Finish logging."""
        if self.enabled:
            wandb.finish()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    epoch: int,
    wandb_logger: WandBLogger = None,
    vae_loss_weight: float = 0.1,
    pred_loss_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Train for one epoch with BDI VAE model.
    
    Computes:
    - VAE losses (belief, desire, intention)
    - Prediction losses (goal, nextstep, category)
    - Total loss = VAE losses + Prediction losses
    """
    model.train()
    metrics = {
        'loss': AverageMeter(),
        'total_vae_loss': AverageMeter(),
        'total_pred_loss': AverageMeter(),
        'belief_loss': AverageMeter(),
        'belief_recon_loss': AverageMeter(),
        'belief_kl_loss': AverageMeter(),
        'desire_loss': AverageMeter(),
        'desire_recon_loss': AverageMeter(),
        'desire_kl_loss': AverageMeter(),
        'intention_loss': AverageMeter(),
        'intention_recon_loss': AverageMeter(),
        'intention_kl_loss': AverageMeter(),
        'loss_goal': AverageMeter(),
        'loss_nextstep': AverageMeter(),
        'loss_category': AverageMeter(),
        'goal_acc': AverageMeter(),
        'goal_top5_acc': AverageMeter(),
        'nextstep_acc': AverageMeter(),
        'category_acc': AverageMeter(),
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)  # [batch, seq_len]
        history_lengths = batch['history_lengths'].to(device)            # [batch]
        next_node_idx = batch['next_node_idx'].to(device)                # [batch]
        goal_idx = batch['goal_idx'].to(device)                          # [batch]
        goal_cat_idx = batch['goal_cat_idx'].to(device)                  # [batch]
        agent_id = batch['agent_id'].to(device)                          # [batch]
        
        # Forward pass
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            compute_loss=True,
        )
        
        # Extract predictions
        goal_logits = outputs['goal']              # [batch, num_poi_nodes]
        nextstep_logits = outputs['nextstep']      # [batch, num_nodes]
        category_logits = outputs['category']      # [batch, num_categories]
        
        # Compute prediction losses
        loss_goal = criterion['goal'](goal_logits, goal_idx)
        loss_nextstep = criterion['nextstep'](nextstep_logits, next_node_idx)
        loss_category = criterion['category'](category_logits, goal_cat_idx)
        
        # Weighted prediction loss
        total_pred_loss = 1.0 * loss_goal + 0.5 * loss_nextstep + 0.5 * loss_category
        
        # Extract VAE losses
        belief_loss = outputs['belief_loss']
        belief_recon = outputs['belief_recon_loss']
        belief_kl = outputs['belief_kl_loss']
        desire_loss = outputs['desire_loss']
        desire_recon = outputs['desire_recon_loss']
        desire_kl = outputs['desire_kl_loss']
        intention_loss = outputs['intention_loss']
        intention_recon = outputs['intention_recon_loss']
        intention_kl = outputs['intention_kl_loss']
        total_vae_loss = outputs['total_vae_loss']
        
        # Total loss = Weighted VAE losses + Weighted Prediction losses
        loss = (vae_loss_weight * total_vae_loss) + (pred_loss_weight * total_pred_loss)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute accuracies
        # Top-1 accuracy
        goal_acc = (goal_logits.argmax(dim=1) == goal_idx).float().mean().item() * 100
        nextstep_acc = (nextstep_logits.argmax(dim=1) == next_node_idx).float().mean().item() * 100
        category_acc = (category_logits.argmax(dim=1) == goal_cat_idx).float().mean().item() * 100
        
        # Top-5 accuracy for goal prediction
        _, top5_preds = goal_logits.topk(5, dim=1)
        goal_top5_acc = (top5_preds == goal_idx.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['total_vae_loss'].update(total_vae_loss.item(), batch_size)
        metrics['total_pred_loss'].update(total_pred_loss.item(), batch_size)
        metrics['belief_loss'].update(belief_loss.item(), batch_size)
        metrics['belief_recon_loss'].update(belief_recon.item(), batch_size)
        metrics['belief_kl_loss'].update(belief_kl.item(), batch_size)
        metrics['desire_loss'].update(desire_loss.item(), batch_size)
        metrics['desire_recon_loss'].update(desire_recon.item(), batch_size)
        metrics['desire_kl_loss'].update(desire_kl.item(), batch_size)
        metrics['intention_loss'].update(intention_loss.item(), batch_size)
        metrics['intention_recon_loss'].update(intention_recon.item(), batch_size)
        metrics['intention_kl_loss'].update(intention_kl.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['goal_top5_acc'].update(goal_top5_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
        
        # Log batch metrics to W&B
        if wandb_logger is not None:
            batch_metrics = {
                'loss': loss.item(),
                'total_vae_loss': total_vae_loss.item(),
                'total_pred_loss': total_pred_loss.item(),
                'belief_loss': belief_loss.item(),
                'desire_loss': desire_loss.item(),
                'intention_loss': intention_loss.item(),
                'loss_goal': loss_goal.item(),
                'goal_acc': goal_acc,
            }
            lr = optimizer.param_groups[0]['lr']
            wandb_logger.log_batch(epoch, batch_idx, batch_size, batch_metrics, lr)
        
        pbar.set_postfix({
            'loss': f"{metrics['loss'].avg:.4f}",
            'vae': f"{metrics['total_vae_loss'].avg:.4f}",
            'goal_acc': f"{metrics['goal_acc'].avg:.1f}%",
        })
    
    # Return AVERAGE metrics across all batches
    return {k: v.avg for k, v in metrics.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    vae_loss_weight: float = 0.1,
    pred_loss_weight: float = 1.0,
    compute_percentiles: bool = False,
) -> tuple:
    """
    Validate the BDI VAE model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss functions
        device: Device to use
        vae_loss_weight: Weight for VAE losses
        pred_loss_weight: Weight for prediction losses
        compute_percentiles: If True, compute goal accuracy at 15%, 50%, 85% history lengths
    
    Returns:
        Tuple of (average_metrics, percentile_metrics)
    """
    model.eval()
    metrics = {
        'loss': AverageMeter(),
        'total_vae_loss': AverageMeter(),
        'total_pred_loss': AverageMeter(),
        'belief_loss': AverageMeter(),
        'belief_recon_loss': AverageMeter(),
        'belief_kl_loss': AverageMeter(),
        'desire_loss': AverageMeter(),
        'desire_recon_loss': AverageMeter(),
        'desire_kl_loss': AverageMeter(),
        'intention_loss': AverageMeter(),
        'intention_recon_loss': AverageMeter(),
        'intention_kl_loss': AverageMeter(),
        'loss_goal': AverageMeter(),
        'loss_nextstep': AverageMeter(),
        'loss_category': AverageMeter(),
        'goal_acc': AverageMeter(),
        'goal_top5_acc': AverageMeter(),
        'nextstep_acc': AverageMeter(),
        'category_acc': AverageMeter(),
    }
    
    # For percentile tracking (by history length)
    all_history_lengths = []
    all_goal_correct = []
    
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
        
        # Forward pass
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            compute_loss=True,
        )
        
        # Extract predictions
        goal_logits = outputs['goal']
        nextstep_logits = outputs['nextstep']
        category_logits = outputs['category']
        
        # Compute prediction losses
        loss_goal = criterion['goal'](goal_logits, goal_idx)
        loss_nextstep = criterion['nextstep'](nextstep_logits, next_node_idx)
        loss_category = criterion['category'](category_logits, goal_cat_idx)
        total_pred_loss = 1.0 * loss_goal + 0.5 * loss_nextstep + 0.5 * loss_category
        
        # Extract VAE losses
        total_vae_loss = outputs['total_vae_loss']
        loss = (vae_loss_weight * total_vae_loss) + (pred_loss_weight * total_pred_loss)
        
        # Compute accuracies
        # Top-1 accuracy
        goal_acc = (goal_logits.argmax(dim=1) == goal_idx).float().mean().item() * 100
        nextstep_acc = (nextstep_logits.argmax(dim=1) == next_node_idx).float().mean().item() * 100
        category_acc = (category_logits.argmax(dim=1) == goal_cat_idx).float().mean().item() * 100
        
        # Top-5 accuracy for goal prediction
        _, top5_preds = goal_logits.topk(5, dim=1)
        goal_top5_acc = (top5_preds == goal_idx.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['total_vae_loss'].update(total_vae_loss.item(), batch_size)
        metrics['total_pred_loss'].update(total_pred_loss.item(), batch_size)
        metrics['belief_loss'].update(outputs['belief_loss'].item(), batch_size)
        metrics['belief_recon_loss'].update(outputs['belief_recon_loss'].item(), batch_size)
        metrics['belief_kl_loss'].update(outputs['belief_kl_loss'].item(), batch_size)
        metrics['desire_loss'].update(outputs['desire_loss'].item(), batch_size)
        metrics['desire_recon_loss'].update(outputs['desire_recon_loss'].item(), batch_size)
        metrics['desire_kl_loss'].update(outputs['desire_kl_loss'].item(), batch_size)
        metrics['intention_loss'].update(outputs['intention_loss'].item(), batch_size)
        metrics['intention_recon_loss'].update(outputs['intention_recon_loss'].item(), batch_size)
        metrics['intention_kl_loss'].update(outputs['intention_kl_loss'].item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['goal_top5_acc'].update(goal_top5_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
        
        # Track for percentile computation
        if compute_percentiles:
            goal_correct = (goal_logits.argmax(dim=1) == goal_idx).cpu().numpy()
            lengths = history_lengths.cpu().numpy()
            all_history_lengths.extend(lengths)
            all_goal_correct.extend(goal_correct)
        
        pbar.set_postfix({'loss': f"{metrics['loss'].avg:.4f}"})
    
    # Compute percentile metrics
    percentile_metrics = None
    if compute_percentiles and len(all_history_lengths) > 0:
        import numpy as np
        all_history_lengths = np.array(all_history_lengths)
        all_goal_correct = np.array(all_goal_correct)
        
        # Get 15th, 50th, 85th percentiles of history lengths
        p15 = np.percentile(all_history_lengths, 15)
        p50 = np.percentile(all_history_lengths, 50)
        p85 = np.percentile(all_history_lengths, 85)
        
        percentile_metrics = {}
        for pct_name, pct_value in [('15%', p15), ('50%', p50), ('85%', p85)]:
            # Find samples within ±10% of this percentile
            lower = pct_value * 0.9
            upper = pct_value * 1.1
            mask = (all_history_lengths >= lower) & (all_history_lengths <= upper)
            
            if mask.sum() > 0:
                percentile_metrics[pct_name] = {
                    'goal_acc': all_goal_correct[mask].mean() * 100,
                }
    
    return {k: v.avg for k, v in metrics.items()}, percentile_metrics


# ============================================================================
# DATA LOADING WITH PRE-DEFINED SPLITS
# ============================================================================

def load_data_with_splits(
    data_dir: str,
    graph_path: str,
    split_indices_path: str,
    trajectory_filename: str = 'all_trajectories.json',
) -> tuple:
    """
    Load data with pre-defined train/val/test splits.
    
    Args:
        data_dir: Directory containing trajectory data
        graph_path: Path to graph file
        split_indices_path: Path to JSON file with split indices
        trajectory_filename: Name of trajectory file
    
    Returns:
        Tuple of (train_trajs, val_trajs, test_trajs, graph, poi_nodes)
    """
    print("\n" + "=" * 100)
    print("📂 LOADING DATA WITH PRE-DEFINED SPLITS")
    print("=" * 100)
    
    # Load full dataset
    trajectories, graph, poi_nodes = load_simulation_data(data_dir, graph_path, trajectory_filename)
    print(f"✅ Loaded {len(trajectories)} trajectories")
    print(f"✅ Loaded graph with {len(graph.nodes)} nodes ({len(poi_nodes)} POI nodes)")
    
    # Load split indices
    print(f"\n📊 Loading split indices from: {split_indices_path}")
    with open(split_indices_path, 'r') as f:
        split_data = json.load(f)
    
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']
    test_indices = split_data['test_indices']
    
    print(f"   Train: {len(train_indices)} trajectories")
    print(f"   Val:   {len(val_indices)} trajectories")
    print(f"   Test:  {len(test_indices)} trajectories")
    
    # Split trajectories
    train_trajs = [trajectories[i] for i in train_indices]
    val_trajs = [trajectories[i] for i in val_indices]
    test_trajs = [trajectories[i] for i in test_indices]
    
    print("\n✅ Data split complete!")
    print("=" * 100 + "\n")
    
    return train_trajs, val_trajs, test_trajs, graph, poi_nodes


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train BDI VAE Model')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, 
                       default='data/simulation_data/run_8',
                       help='Directory containing trajectory data')
    parser.add_argument('--graph_path', type=str,
                       default='data/processed/ucsd_walk_full.graphml',
                       help='Path to graph file')
    parser.add_argument('--split_indices_path', type=str,
                       default='data/simulation_data/run_8/split_data/split_indices_seed42.json',
                       help='Path to split indices file')
    parser.add_argument('--trajectory_filename', type=str,
                       default='all_trajectories.json',
                       help='Name of trajectory file')
    
    # Model hyperparameters - Embedding
    parser.add_argument('--node_embedding_dim', type=int, default=64,
                       help='Node embedding dimension')
    parser.add_argument('--temporal_dim', type=int, default=64,
                       help='Temporal embedding dimension')
    parser.add_argument('--agent_dim', type=int, default=64,
                       help='Agent embedding dimension')
    parser.add_argument('--fusion_dim', type=int, default=128,
                       help='Fusion dimension (unified embedding output)')
    parser.add_argument('--freeze_embedding', action='store_true',
                       help='Freeze embedding pipeline (use with --pretrained_embedding)')
    parser.add_argument('--pretrained_embedding', type=str, default=None,
                       help='Path to pretrained unified embedding pipeline (e.g., checkpoints/keepers/unified_embedding.pt)')
    
    # Model hyperparameters - VAE
    parser.add_argument('--belief_latent_dim', type=int, default=32,
                       help='Belief VAE latent dimension')
    parser.add_argument('--desire_latent_dim', type=int, default=32,
                       help='Desire VAE latent dimension')
    parser.add_argument('--intention_latent_dim', type=int, default=64,
                       help='Intention VAE latent dimension')
    parser.add_argument('--vae_hidden_dim', type=int, default=128,
                       help='VAE hidden layer dimension')
    parser.add_argument('--vae_num_layers', type=int, default=2,
                       help='Number of VAE hidden layers')
    
    # Model hyperparameters - Prediction heads
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Prediction head hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads (for fusion)')
    
    # β-VAE weights
    parser.add_argument('--beta_belief', type=float, default=1.0,
                       help='β weight for belief VAE KL loss')
    parser.add_argument('--beta_desire', type=float, default=1.0,
                       help='β weight for desire VAE KL loss')
    parser.add_argument('--beta_intention', type=float, default=1.0,
                       help='β weight for intention VAE KL loss')
    
    # Loss weighting
    parser.add_argument('--vae_loss_weight', type=float, default=0.1,
                       help='Weight for VAE losses (lower = focus more on predictions)')
    parser.add_argument('--pred_loss_weight', type=float, default=1.0,
                       help='Weight for prediction losses')
    parser.add_argument('--use_kl_annealing', action='store_true',
                       help='Use KL annealing: start with low VAE weight, gradually increase')
    parser.add_argument('--kl_anneal_epochs', type=int, default=10,
                       help='Number of epochs to anneal VAE loss weight from 0 to target value')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (per-node training)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data processing
    parser.add_argument('--min_traj_length', type=int, default=2,
                       help='Minimum trajectory length')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/bdi_vae_simple',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Name for W&B run (default: auto-generated)')
    parser.add_argument('--log_percentiles', action='store_true',
                       help='Log accuracy at 15%%, 50%%, 85%% history length percentiles')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data with pre-defined splits
    train_trajs, val_trajs, test_trajs, graph, poi_nodes = load_data_with_splits(
        args.data_dir,
        args.graph_path,
        args.split_indices_path,
        args.trajectory_filename
    )
    
    # Create datasets (WITH per-node expansion!)
    print("\n" + "=" * 100)
    print("📊 CREATING BDI VAE DATASETS (PER-NODE EXPANSION)")
    print("=" * 100)
    
    print("\n🔹 Creating training dataset...")
    train_dataset = BDIVAEDataset(
        train_trajs,
        graph,
        poi_nodes,
        min_traj_length=args.min_traj_length,
    )
    
    print("\n🔹 Creating validation dataset...")
    val_dataset = BDIVAEDataset(
        val_trajs,
        graph,
        poi_nodes,
        node_to_idx_map=train_dataset.node_to_idx,  # Use same mapping
        min_traj_length=args.min_traj_length,
    )
    
    print("\n🔹 Creating test dataset...")
    test_dataset = BDIVAEDataset(
        test_trajs,
        graph,
        poi_nodes,
        node_to_idx_map=train_dataset.node_to_idx,  # Use same mapping
        min_traj_length=args.min_traj_length,
    )
    
    print(f"\n✅ Dataset sizes:")
    print(f"   Train: {len(train_dataset)} per-node samples")
    print(f"   Val:   {len(val_dataset)} per-node samples")
    print(f"   Test:  {len(test_dataset)} per-node samples")
    print("=" * 100 + "\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_bdi_samples,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bdi_samples,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model
    print("\n" + "=" * 100)
    print("🏗️  CREATING BDI VAE MODEL")
    print("=" * 100)
    
    model = BDIVAEPredictor(
        num_nodes=len(graph.nodes),
        num_agents=100,  # 100 agents from simulation
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
        freeze_embedding=args.freeze_embedding,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ BDI VAE Model created!")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters:    {total_params - trainable_params:,}")
    print(f"   Belief latent dim:    {args.belief_latent_dim}")
    print(f"   Desire latent dim:    {args.desire_latent_dim}")
    print(f"   Intention latent dim: {args.intention_latent_dim}")
    print("=" * 100 + "\n")
    
    # Load pretrained embedding pipeline if provided
    if args.pretrained_embedding is not None:
        print(f"\n📥 Loading pretrained embedding pipeline from: {args.pretrained_embedding}")
        try:
            checkpoint = torch.load(args.pretrained_embedding, map_location=device)
            
            # Check config for debugging (but ignore incorrect values)
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                print(f"   Saved config: {saved_config}")
                if saved_config.get('num_agents', 1) == 1:
                    print(f"   ⚠️  Config shows 1 agent, but actual weights support 100 agents (config error, weights are correct)")
            
            # Load the actual state dict (this has the correct 100-agent weights)
            model.embedding_pipeline.load_state_dict(checkpoint['embedding_pipeline_state_dict'])
            print(f"✅ Pretrained embedding pipeline loaded successfully!")
            if 'epoch' in checkpoint:
                print(f"   Pre-trained for {checkpoint['epoch']} epochs")
            
            # Verify the actual loaded weights support 100 agents
            if hasattr(model.embedding_pipeline, 'agent_encoder') and model.embedding_pipeline.use_agent:
                try:
                    agent_emb_weight = model.embedding_pipeline.agent_encoder.agent_context.agent_emb.agent_embedding.weight
                    actual_num_agents = agent_emb_weight.shape[0]
                    print(f"   ✅ Verified: Agent embeddings support {actual_num_agents} agents")
                except AttributeError:
                    print(f"   ✅ Agent encoder loaded (verification skipped)")
            
            # Freeze if requested
            if args.freeze_embedding:
                model._freeze_embeddings()
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"   Trainable parameters (after freeze): {trainable_params:,}")
        except FileNotFoundError:
            print(f"⚠️  Warning: Pretrained embedding file not found: {args.pretrained_embedding}")
            print(f"   Training from scratch instead.")
        except Exception as e:
            print(f"⚠️  Error loading pretrained embedding: {e}")
            print(f"   Training from scratch instead.")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Loss functions (for predictions only; VAE losses computed in model)
    criterion = {
        'goal': nn.CrossEntropyLoss(),
        'nextstep': nn.CrossEntropyLoss(),
        'category': nn.CrossEntropyLoss(),
    }
    
    # Initialize W&B
    wandb_logger = None
    if not args.no_wandb and WANDB_AVAILABLE:
        config = vars(args)
        config.update({
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_nodes': len(graph.nodes),
            'num_pois': len(poi_nodes),
            'model_type': 'bdi_vae',
        })
        wandb_logger = WandBLogger(
            project_name="bdi-tom",  # Same project as transformer
            config=config,
            run_name=args.wandb_run_name
        )
        
        # Log model info
        model_config = {
            'belief_latent_dim': args.belief_latent_dim,
            'desire_latent_dim': args.desire_latent_dim,
            'intention_latent_dim': args.intention_latent_dim,
            'fusion_dim': args.fusion_dim,
            'vae_hidden_dim': args.vae_hidden_dim,
            'beta_belief': args.beta_belief,
            'beta_desire': args.beta_desire,
            'beta_intention': args.beta_intention,
        }
        wandb_logger.log_model_info(total_params, trainable_params, model_config)
    
    # Training loop
    print("\n" + "=" * 100)
    print("🚀 STARTING BDI VAE TRAINING")
    print("=" * 100 + "\n")
    
    if args.use_kl_annealing:
        print(f"📈 Using KL annealing: VAE loss weight will increase from 0.0 to {args.vae_loss_weight} over {args.kl_anneal_epochs} epochs")
    
    best_val_goal_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, args.num_epochs + 1):
        # KL annealing: gradually increase VAE loss weight
        if args.use_kl_annealing:
            if epoch <= args.kl_anneal_epochs:
                current_vae_weight = args.vae_loss_weight * (epoch / args.kl_anneal_epochs)
            else:
                current_vae_weight = args.vae_loss_weight
        else:
            current_vae_weight = args.vae_loss_weight
        
        print(f"\n{'='*100}")
        print(f"EPOCH {epoch}/{args.num_epochs}")
        if args.use_kl_annealing and epoch <= args.kl_anneal_epochs:
            print(f"VAE Loss Weight (annealed): {current_vae_weight:.4f}")
        print(f"{'='*100}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, wandb_logger,
            vae_loss_weight=current_vae_weight,
            pred_loss_weight=args.pred_loss_weight
        )
        
        # Validate
        val_metrics, val_percentile_metrics = validate(
            model, val_loader, criterion, device,
            vae_loss_weight=current_vae_weight,
            pred_loss_weight=args.pred_loss_weight,
            compute_percentiles=args.log_percentiles
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to W&B
        if wandb_logger is not None:
            # Add current VAE weight to log
            log_dict_extra = {'vae_loss_weight': current_vae_weight}
            wandb.log(log_dict_extra, step=epoch)
            
            wandb_logger.log_epoch(
                epoch, train_metrics, val_metrics, current_lr,
                val_percentile_metrics=val_percentile_metrics
            )
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train - Loss: {train_metrics['loss']:.4f} | VAE: {train_metrics['total_vae_loss']:.4f} | Pred: {train_metrics['total_pred_loss']:.4f}")
        print(f"           Goal Top-1: {train_metrics['goal_acc']:.1f}% | Top-5: {train_metrics['goal_top5_acc']:.1f}% | Next: {train_metrics['nextstep_acc']:.1f}% | Cat: {train_metrics['category_acc']:.1f}%")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f} | VAE: {val_metrics['total_vae_loss']:.4f} | Pred: {val_metrics['total_pred_loss']:.4f}")
        print(f"           Goal Top-1: {val_metrics['goal_acc']:.1f}% | Top-5: {val_metrics['goal_top5_acc']:.1f}% | Next: {val_metrics['nextstep_acc']:.1f}% | Cat: {val_metrics['category_acc']:.1f}%")
        
        if val_percentile_metrics is not None:
            print(f"\n   📈 Validation Goal Accuracy by History Length Percentile:")
            print(f"      15%: {val_percentile_metrics['15%']['goal_acc']:.1f}%")
            print(f"      50%: {val_percentile_metrics['50%']['goal_acc']:.1f}%")
            print(f"      85%: {val_percentile_metrics['85%']['goal_acc']:.1f}%")
        
        print(f"   VAE Breakdown:")
        print(f"      Belief:    recon={train_metrics['belief_recon_loss']:.4f}, kl={train_metrics['belief_kl_loss']:.4f}")
        print(f"      Desire:    recon={train_metrics['desire_recon_loss']:.4f}, kl={train_metrics['desire_kl_loss']:.4f}")
        print(f"      Intention: recon={train_metrics['intention_recon_loss']:.4f}, kl={train_metrics['intention_kl_loss']:.4f}")
        
        # Check for improvement
        if val_metrics['goal_acc'] > best_val_goal_acc:
            best_val_goal_acc = val_metrics['goal_acc']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                val_metrics['goal_acc'],
                is_best=True
            )
            print(f"   ✅ New best model! Val Goal Acc: {best_val_goal_acc:.1f}%")
            
            if wandb_logger is not None:
                wandb_logger.log_checkpoint(epoch, checkpoint_path, best=True)
        else:
            patience_counter += 1
            print(f"   No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                val_metrics['goal_acc'],
                is_best=False
            )
            
            if wandb_logger is not None:
                wandb_logger.log_checkpoint(epoch, checkpoint_path, best=False)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏸️  Early stopping triggered after {epoch} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n" + "=" * 100)
    print("🎉 BDI VAE TRAINING COMPLETE!")
    print("=" * 100)
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Val Goal Acc: {best_val_goal_acc:.1f}%")
    print(f"   Total Time: {total_time/3600:.2f} hours")
    print("=" * 100 + "\n")
    
    # Log summary
    if wandb_logger is not None:
        wandb_logger.log_summary(
            best_epoch, best_val_goal_acc, epoch, total_time/3600
        )
        wandb_logger.finish()


if __name__ == '__main__':
    main()
