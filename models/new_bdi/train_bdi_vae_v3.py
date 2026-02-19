"""
TRAINING SCRIPT FOR SEQUENTIAL CONDITIONAL BDI-VAE (SC-BDI) V3

This training script properly implements ALL the improvements in V3:

CRITICAL FIXES OVER V2 TRAINING:
1. PASSES goal_idx TO MODEL - enables InfoNCE and direct goal prediction!
2. IMPLEMENTS KL ANNEALING - prevents posterior collapse
3. MONITORS DESIRE-GOAL ACCURACY - tracks if desires predict goals
4. FREE-BITS WARMUP - gradual introduction of free-bits
5. PROPER LOSS LOGGING - all components tracked

KL ANNEALING SCHEDULE:
- Epochs 0-10: Linear warmup from 0.0 → 1.0
- Epochs 10+: Full KL weight (1.0)

This is based on "Generating Diverse High-Fidelity Images with VQ-VAE-2"
and common β-VAE training practices.

Usage:
    python -m models.vae_bdi_simple.train_bdi_vae_v3 [OPTIONS]

Example:
    python -m models.vae_bdi_simple.train_bdi_vae_v3 --num_epochs 100 --batch_size 256

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

from models.new_bdi.bdi_vae_v3_model import SequentialConditionalBDIVAE, create_sc_bdi_vae_v3
from models.new_bdi.bdi_dataset_v3 import (
    BDIVAEDatasetV3, 
    collate_bdi_samples_v3,
    TemporalConsistencyBatchSampler,
)
from models.utils.data_loader import load_simulation_data
from models.utils.utils import get_device, set_seed, save_checkpoint, AverageMeter

# Visualization
try:
    from models.new_bdi.visualize_training import TrainingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization module not available")

# W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


# =============================================================================
# KL ANNEALING SCHEDULE
# =============================================================================

class KLAnnealingSchedule:
    """
    KL annealing schedule to prevent posterior collapse.
    
    Strategies:
    - 'linear': Linear warmup from 0 to 1
    - 'cosine': Cosine warmup
    - 'cyclical': Cyclical annealing (Bowman et al.)
    - 'monotonic': Linear then hold
    """
    
    def __init__(
        self,
        strategy: str = 'monotonic',
        warmup_epochs: int = 10,
        cycle_epochs: int = 20,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def get_weight(self, epoch: int) -> float:
        """Get KL weight for given epoch."""
        if self.strategy == 'monotonic':
            # Linear warmup, then hold
            if epoch < self.warmup_epochs:
                return self.min_weight + (self.max_weight - self.min_weight) * (epoch / self.warmup_epochs)
            return self.max_weight
        
        elif self.strategy == 'linear':
            # Linear warmup only
            return min(self.max_weight, self.min_weight + (self.max_weight - self.min_weight) * (epoch / self.warmup_epochs))
        
        elif self.strategy == 'cosine':
            # Cosine warmup
            if epoch < self.warmup_epochs:
                return self.min_weight + 0.5 * (self.max_weight - self.min_weight) * (1 - np.cos(np.pi * epoch / self.warmup_epochs))
            return self.max_weight
        
        elif self.strategy == 'cyclical':
            # Cyclical annealing
            cycle_position = epoch % self.cycle_epochs
            return self.min_weight + (self.max_weight - self.min_weight) * (cycle_position / self.cycle_epochs)
        
        else:
            return self.max_weight


# =============================================================================
# ENHANCED LOGGER
# =============================================================================

class EnhancedWandBLogger:
    """Enhanced W&B logger with V3-specific metrics."""
    
    def __init__(
        self, 
        project_name: str = "tom-compare-v1",
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
        kl_weight: float,
    ):
        """Log batch-level metrics."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'learning_rate': lr,
            'kl_weight': kl_weight,
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
        kl_weight: float,
        progress_metrics: Dict[str, Dict[str, float]] = None,
    ):
        """Log epoch-level metrics."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'learning_rate': lr,
            'kl_weight': kl_weight,
        }
        
        for key, value in train_metrics.items():
            log_dict[f'train/{key}'] = value
        
        for key, value in val_metrics.items():
            log_dict[f'val/{key}'] = value
        
        if progress_metrics:
            for progress_bin, metrics in progress_metrics.items():
                for key, value in metrics.items():
                    log_dict[f'progress_{progress_bin}/{key}'] = value
        
        wandb.log(log_dict, step=self.global_step)
        self.global_step += 1
    
    def finish(self):
        if self.enabled:
            wandb.finish()


# =============================================================================
# TRAINING EPOCH
# =============================================================================

def print_epoch_diagnostics(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    num_poi_nodes: int,
    num_categories: int,
    kl_weight: float,
):
    """
    Print comprehensive diagnostics after each epoch.
    
    Explains WHY accuracy is what it is and what to expect.
    """
    random_goal_acc = 100.0 / num_poi_nodes  # e.g., 100/735 = 0.136%
    random_cat_acc = 100.0 / num_categories  # e.g., 100/7 = 14.3%
    
    goal_acc = val_metrics.get('goal_acc', 0)
    desire_goal_acc = val_metrics.get('desire_goal_acc', 0)
    infonce = train_metrics.get('infonce_loss', 0)
    
    # Calculate how much better than random
    goal_improvement = goal_acc / random_goal_acc if random_goal_acc > 0 else 0
    desire_improvement = desire_goal_acc / random_goal_acc if random_goal_acc > 0 else 0
    
    print("\n" + "─" * 60)
    print("📋 EPOCH DIAGNOSTICS")
    print("─" * 60)
    
    # Random baseline context
    print(f"\n🎯 Random Baseline:")
    print(f"   • Goal prediction ({num_poi_nodes} POIs): {random_goal_acc:.2f}%")
    print(f"   • Category prediction ({num_categories} cats): {random_cat_acc:.1f}%")
    
    # Current performance relative to random
    print(f"\n📈 Current Performance vs Random:")
    print(f"   • Goal accuracy:        {goal_acc:.2f}% ({goal_improvement:.1f}x random)")
    print(f"   • Desire→Goal accuracy: {desire_goal_acc:.2f}% ({desire_improvement:.1f}x random)")
    
    # Interpretation
    print(f"\n🔍 Interpretation:")
    if epoch == 0:
        print(f"   ✅ Epoch 1 is expected to be near-random")
        print(f"   • KL weight = {kl_weight:.2f} (annealing not yet engaged)")
        print(f"   • InfoNCE loss = {infonce:.2f} (still learning alignment)")
        print(f"   • This is NORMAL! Give it 10-20 epochs.")
    elif epoch < 10:
        if goal_acc > random_goal_acc * 2:
            print(f"   ✅ Learning! Already {goal_improvement:.1f}x better than random")
        else:
            print(f"   ⚠️  Still near random. KL annealing in progress.")
        print(f"   • KL weight = {kl_weight:.2f} (will reach 1.0 at epoch 10)")
    else:
        if goal_acc < 5:
            print(f"   ⚠️  Accuracy still low after KL warmup!")
            print(f"   • Check if goal_idx is being used correctly")
            print(f"   • Consider increasing learning rate or infonce_weight")
        elif goal_acc < 15:
            print(f"   🔵 Moderate learning. This is a hard problem with {num_poi_nodes} goals.")
        else:
            print(f"   ✅ Good progress! {goal_improvement:.1f}x better than random")
    
    # InfoNCE interpretation
    print(f"\n📊 InfoNCE Loss Analysis:")
    if infonce > 5:
        print(f"   • InfoNCE = {infonce:.2f}: High (desire-goal alignment weak)")
        print(f"   • Expected to decrease to ~2-3 as training progresses")
    elif infonce > 3:
        print(f"   • InfoNCE = {infonce:.2f}: Medium (alignment improving)")
    else:
        print(f"   • InfoNCE = {infonce:.2f}: Low (good desire-goal alignment!)")
    
    print("─" * 60 + "\n")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    epoch: int,
    config: argparse.Namespace,
    kl_weight: float,
    wandb_logger: Optional[EnhancedWandBLogger] = None,
    scheduler = None,
) -> Dict[str, float]:
    """
    Train for one epoch with SC-BDI V3 model.
    
    CRITICAL: Passes goal_idx to model for InfoNCE and direct goal prediction!
    """
    model.train()
    model.set_kl_weight(kl_weight)  # Set KL annealing weight
    
    meters = defaultdict(AverageMeter)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)  # CRITICAL: This is now used!
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        agent_id = batch['agent_id'].to(device)
        path_progress = batch['path_progress'].to(device)
        traj_id = batch['traj_id'].to(device)

        # Temporal features (from enriched trajectories)
        hours = batch['hour'].to(device) if 'hour' in batch else None
        days = batch['day_of_week'].to(device) if 'day_of_week' in batch else None
        deltas = batch['history_temporal_deltas'].to(device) if 'history_temporal_deltas' in batch else None
        velocities = batch['history_velocities'].to(device) if 'history_velocities' in batch else None
        
        # Forward pass - PASS GOAL_IDX FOR INFONCE!
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            path_progress=path_progress,
            compute_loss=True,
            next_node_idx=next_node_idx,
            goal_idx=goal_idx,  # CRITICAL: Enable InfoNCE + direct goal prediction!
            goal_cat_idx=goal_cat_idx,
            hours=hours,
            days=days,
            deltas=deltas,
            velocities=velocities,
        )
        
        # NaN detection
        nan_detected = False
        for key in ['belief_z', 'desire_z', 'intention_z', 'goal', 'nextstep']:
            if key in outputs and torch.isnan(outputs[key]).any():
                nan_detected = True
                print(f"\n⚠️  NaN in {key} at batch {batch_idx}, skipping...")
                break
        
        if nan_detected:
            continue
        
        # Prediction losses (from intention head)
        loss_goal = criterion['goal'](outputs['goal'], goal_idx)
        loss_nextstep = criterion['nextstep'](outputs['nextstep'], next_node_idx)
        loss_category = criterion['category'](outputs['category'], goal_cat_idx)
        
        pred_loss = (
            config.goal_weight * loss_goal +
            config.nextstep_weight * loss_nextstep +
            config.category_weight * loss_category
        )
        
        # VAE losses (already includes InfoNCE, conditional KL, free-bits)
        vae_loss = outputs['total_vae_loss']
        
        # Progress prediction loss
        progress_loss = torch.tensor(0.0, device=device)
        if model.use_progress and 'progress_loss' in outputs:
            progress_loss = config.progress_weight * outputs['progress_loss']
        
        # Total loss
        total_loss = pred_loss + config.vae_weight * vae_loss + progress_loss
        
        # NaN check before backward
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n⚠️  NaN/Inf loss at batch {batch_idx}, skipping...")
            continue
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Compute accuracies
        # 1. Main goal prediction (from intention)
        goal_acc = (outputs['goal'].argmax(dim=1) == goal_idx).float().mean().item() * 100
        nextstep_acc = (outputs['nextstep'].argmax(dim=1) == next_node_idx).float().mean().item() * 100
        category_acc = (outputs['category'].argmax(dim=1) == goal_cat_idx).float().mean().item() * 100
        
        # 2. Desire-direct goal prediction (NEW! Critical metric!)
        desire_goal_acc = 0.0
        if 'desire_goal_logits' in outputs:
            desire_goal_acc = (outputs['desire_goal_logits'].argmax(dim=1) == goal_idx).float().mean().item() * 100
        
        # Update meters
        meters['loss'].update(total_loss.item(), batch_size)
        meters['pred_loss'].update(pred_loss.item(), batch_size)
        meters['vae_loss'].update(vae_loss.item(), batch_size)
        meters['goal_acc'].update(goal_acc, batch_size)
        meters['nextstep_acc'].update(nextstep_acc, batch_size)
        meters['category_acc'].update(category_acc, batch_size)
        meters['desire_goal_acc'].update(desire_goal_acc, batch_size)  # NEW!
        
        # VAE component losses
        meters['belief_loss'].update(outputs['belief_loss'].item(), batch_size)
        meters['belief_recon'].update(outputs['belief_recon_loss'].item(), batch_size)
        meters['belief_kl'].update(outputs['belief_kl'].item(), batch_size)
        meters['desire_loss'].update(outputs['desire_loss'].item(), batch_size)
        meters['desire_recon'].update(outputs['desire_recon_loss'].item(), batch_size)
        meters['desire_kl'].update(outputs['desire_kl'].item(), batch_size)
        meters['intention_loss'].update(outputs['intention_loss'].item(), batch_size)
        meters['intention_kl'].update(outputs['intention_kl'].item(), batch_size)
        meters['mi_loss'].update(outputs['mi_loss'].item(), batch_size)
        meters['infonce_loss'].update(outputs['infonce_loss'].item(), batch_size)  # NEW!
        
        # Desire auxiliary losses
        if 'desire_goal_loss' in outputs:
            meters['desire_goal_loss'].update(outputs['desire_goal_loss'].item(), batch_size)
        if 'desire_category_loss' in outputs:
            meters['desire_category_loss'].update(outputs['desire_category_loss'].item(), batch_size)
        
        # Log to W&B periodically
        if wandb_logger is not None and batch_idx % 10 == 0:
            batch_metrics = {
                'loss': total_loss.item(),
                'goal_acc': goal_acc,
                'desire_goal_acc': desire_goal_acc,
                'infonce_loss': outputs['infonce_loss'].item(),
                'mi_loss': outputs['mi_loss'].item(),
            }
            lr = optimizer.param_groups[0]['lr']
            wandb_logger.log_batch(epoch, batch_idx, batch_metrics, lr, kl_weight)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{meters['loss'].avg:.4f}",
            'goal': f"{meters['goal_acc'].avg:.1f}%",
            'd_goal': f"{meters['desire_goal_acc'].avg:.1f}%",  # Desire→Goal
            'nce': f"{meters['infonce_loss'].avg:.3f}",
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
    """Validate SC-BDI V3 model with progress-stratified metrics."""
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

        # Temporal features
        hours = batch['hour'].to(device) if 'hour' in batch else None
        days = batch['day_of_week'].to(device) if 'day_of_week' in batch else None
        deltas = batch['history_temporal_deltas'].to(device) if 'history_temporal_deltas' in batch else None
        velocities = batch['history_velocities'].to(device) if 'history_velocities' in batch else None
        
        # Forward - PASS GOAL_IDX
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            path_progress=path_progress,
            compute_loss=True,
            next_node_idx=next_node_idx,
            goal_idx=goal_idx,  # Enable InfoNCE
            goal_cat_idx=goal_cat_idx,
            hours=hours,
            days=days,
            deltas=deltas,
            velocities=velocities,
        )
        
        # Losses
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
        
        # Desire→Goal accuracy
        desire_goal_acc = 0.0
        desire_goal_correct = torch.zeros_like(goal_correct)
        if 'desire_goal_logits' in outputs:
            desire_goal_preds = outputs['desire_goal_logits'].argmax(dim=1)
            desire_goal_correct = (desire_goal_preds == goal_idx)
            desire_goal_acc = desire_goal_correct.float().mean().item() * 100
        
        # Update meters
        meters['loss'].update(total_loss.item(), batch_size)
        meters['pred_loss'].update(pred_loss.item(), batch_size)
        meters['vae_loss'].update(vae_loss.item(), batch_size)
        meters['goal_acc'].update(goal_acc, batch_size)
        meters['nextstep_acc'].update(nextstep_acc, batch_size)
        meters['category_acc'].update(category_acc, batch_size)
        meters['desire_goal_acc'].update(desire_goal_acc, batch_size)
        meters['infonce_loss'].update(outputs['infonce_loss'].item(), batch_size)
        meters['mi_loss'].update(outputs['mi_loss'].item(), batch_size)
        
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
            progress_meters[bin_name]['desire_goal_acc'].update(
                desire_goal_correct[i].float().item() * 100, 1
            )
        
        pbar.set_postfix({
            'loss': f"{meters['loss'].avg:.4f}",
            'goal': f"{meters['goal_acc'].avg:.1f}%",
        })
    
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
    """Load trajectory data with pre-defined train/val/test splits.

    Supports two trajectory formats:

    1. **Flat list** (run_8_enriched/enriched_trajectories.json):
       A JSON list of 100 000 trajectory dicts.  Trajectories are ordered
       by agent (1000 per agent for 100 agents).  ``agent_id`` is inferred
       from position: ``agent_id = traj_index // 1000``.

    2. **Nested dict** (run_8/trajectories/all_trajectories.json):
       ``{ "agent_000": [traj, …], "agent_001": [traj, …], … }``
       ``agent_id`` is derived from the sorted key order.

    In both cases every trajectory gets an integer ``agent_id`` field before
    it is returned.
    """
    print("\n" + "=" * 100)
    print("📂 LOADING DATA WITH PRE-DEFINED SPLITS")
    print("=" * 100)

    import networkx as nx
    from graph_controller.world_graph import WorldGraph

    # Load graph
    print(f"   📊 Loading graph from {graph_path}...")
    graph = nx.read_graphml(graph_path)
    print(f"   ✅ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Load trajectories
    traj_path = Path(data_dir) / trajectory_filename
    print(f"   📂 Loading trajectories from {traj_path}...")
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)

    # -----------------------------------------------------------
    # Handle nested dict format (old style)
    # -----------------------------------------------------------
    if isinstance(traj_data, dict):
        print(f"   📊 Detected nested trajectory format (by agent)")
        trajectories = []
        sorted_agents = sorted(traj_data.keys())

        for agent_idx, agent_key in enumerate(sorted_agents):
            for traj in traj_data[agent_key]:
                traj['agent_id'] = agent_idx
                trajectories.append(traj)

        num_agents = len(sorted_agents)
        print(f"   ✅ Flattened {len(trajectories)} trajectories from {num_agents} agents")

    # -----------------------------------------------------------
    # Handle flat list format (enriched / run_8_enriched)
    # -----------------------------------------------------------
    elif isinstance(traj_data, list):
        trajectories = traj_data
        print(f"   📊 Detected flat trajectory list ({len(trajectories)} items)")

        # Load the agent manifest to know how many agents exist
        agents_path = Path(data_dir).parent / 'agents' / 'all_agents.json'
        if agents_path.exists():
            with open(agents_path, 'r') as f:
                agents_meta = json.load(f)
            num_agents = len(agents_meta)
            print(f"   📊 Loaded agent manifest: {num_agents} agents")
        else:
            # Infer: 100 agents, 1000 trajs each by default
            num_agents = max(1, len(trajectories) // 1000)
            print(f"   ⚠️  No agent manifest found – inferring {num_agents} agents")

        trajs_per_agent = len(trajectories) // num_agents

        # Assign agent_id based on position (block-ordered)
        for idx, traj in enumerate(trajectories):
            if 'agent_id' not in traj:
                traj['agent_id'] = idx // trajs_per_agent

        print(f"   ✅ Assigned agent_id (block size {trajs_per_agent}) to {len(trajectories)} trajectories")
    else:
        raise ValueError(f"Unexpected trajectory data type: {type(traj_data)}")

    # Verify temporal enrichment
    sample_t = trajectories[0]
    has_temporal = all(k in sample_t for k in ['temporal_deltas', 'velocities', 'hour', 'day_of_week'])
    print(f"   🕐 Temporal enrichment: {'✅ present' if has_temporal else '❌ absent'}")

    # Get POI nodes using WorldGraph
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    print(f"   📍 Found {len(poi_nodes)} POI nodes (categories: {world_graph.relevant_categories})")

    # Load split indices
    print(f"   📊 Loading split indices from {split_indices_path}...")
    with open(split_indices_path, 'r') as f:
        splits = json.load(f)

    train_indices = splits['train_indices']
    val_indices = splits['val_indices']
    test_indices = splits.get('test_indices', [])

    print(f"   ✅ Splits: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    # Validate indices
    max_idx = max(max(train_indices), max(val_indices))
    if max_idx >= len(trajectories):
        print(f"   ⚠️  WARNING: Max split index ({max_idx}) >= trajectory count ({len(trajectories)})")
        print(f"   Filtering to valid indices...")
        train_indices = [i for i in train_indices if i < len(trajectories)]
        val_indices = [i for i in val_indices if i < len(trajectories)]
        test_indices = [i for i in test_indices if i < len(trajectories)]
        print(f"   ✅ Filtered: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    return graph, trajectories, poi_nodes, train_indices, val_indices, test_indices, num_agents


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SC-BDI V3 Model")
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                        default='data/simulation_data/run_8/trajectories',
                        help='Directory with all trajectories')
    parser.add_argument('--trajectory_filename', type=str,
                        default='all_trajectories.json',
                        help='Name of the trajectory JSON file inside data_dir')
    parser.add_argument('--graph_path', type=str,
                        default='data/processed/ucsd_walk_full.graphml',
                        help='Path to graph file')
    parser.add_argument('--split_indices_path', type=str,
                        default='data/simulation_data/run_8/split_data/split_indices_seed42.json',
                        help='Path to split indices')
    
    # Model architecture
    parser.add_argument('--node_embedding_dim', type=int, default=64)
    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--belief_latent_dim', type=int, default=32)
    parser.add_argument('--desire_latent_dim', type=int, default=16)  # Smaller for abstraction
    parser.add_argument('--intention_latent_dim', type=int, default=32)
    parser.add_argument('--vae_hidden_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # VAE loss weights
    parser.add_argument('--beta_belief', type=float, default=1.0)
    parser.add_argument('--beta_desire', type=float, default=1.0)
    parser.add_argument('--beta_intention', type=float, default=1.0)
    parser.add_argument('--mi_weight', type=float, default=0.1)
    parser.add_argument('--infonce_weight', type=float, default=1.0)  # NEW!
    parser.add_argument('--desire_goal_weight', type=float, default=0.5)  # NEW!
    parser.add_argument('--free_bits', type=float, default=0.5)
    
    # Prediction loss weights
    parser.add_argument('--goal_weight', type=float, default=1.0)
    parser.add_argument('--nextstep_weight', type=float, default=0.5)
    parser.add_argument('--category_weight', type=float, default=0.3)
    parser.add_argument('--vae_weight', type=float, default=0.1)
    parser.add_argument('--progress_weight', type=float, default=0.1)
    
    # KL annealing
    parser.add_argument('--kl_annealing_strategy', type=str, default='monotonic',
                        choices=['monotonic', 'linear', 'cosine', 'cyclical'])
    parser.add_argument('--kl_warmup_epochs', type=int, default=10)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=16)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/sc_bdi_v3')
    parser.add_argument('--save_every', type=int, default=10)
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='tom-compare-v1')
    parser.add_argument('--run_name', type=str, default=None)
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Enable training visualizations')
    parser.add_argument('--visualize_every', type=int, default=5, help='Visualize every N epochs')
    parser.add_argument('--viz_samples', type=int, default=5000, help='Max samples for visualization')
    
    config = parser.parse_args()
    
    # Setup
    set_seed(config.seed)
    device = get_device()
    print(f"\n🖥️  Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    graph, trajectories, poi_nodes, train_idx, val_idx, test_idx, num_agents = load_data_with_splits(
        config.data_dir,
        config.graph_path,
        config.split_indices_path,
        trajectory_filename=config.trajectory_filename,
    )
    
    # Create node-to-idx mapping
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]
    
    train_dataset = BDIVAEDatasetV3(
        trajectories=train_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=node_to_idx,
        include_progress=True,
        include_temporal=True,
    )
    
    val_dataset = BDIVAEDatasetV3(
        trajectories=val_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=node_to_idx,
        include_progress=True,
        include_temporal=True,
    )
    
    print(f"   ✅ Train samples: {len(train_dataset)}")
    print(f"   ✅ Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_bdi_samples_v3,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_bdi_samples_v3,
        pin_memory=True,
    )
    
    # Get counts
    num_nodes = graph.number_of_nodes()
    num_poi_nodes = len(poi_nodes)
    num_categories = len(train_dataset.CATEGORY_TO_IDX)
    
    print(f"\n📊 Dataset stats:")
    print(f"   Nodes: {num_nodes}")
    print(f"   Agents: {num_agents}")
    print(f"   POI nodes: {num_poi_nodes}")
    print(f"   Categories: {num_categories}")
    
    # Create model
    print("\n🏗️  Creating SC-BDI V3 model...")
    model = create_sc_bdi_vae_v3(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=num_categories,
        node_embedding_dim=config.node_embedding_dim,
        fusion_dim=config.fusion_dim,
        belief_latent_dim=config.belief_latent_dim,
        desire_latent_dim=config.desire_latent_dim,
        intention_latent_dim=config.intention_latent_dim,
        vae_hidden_dim=config.vae_hidden_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        beta_belief=config.beta_belief,
        beta_desire=config.beta_desire,
        beta_intention=config.beta_intention,
        mi_weight=config.mi_weight,
        infonce_weight=config.infonce_weight,
        desire_goal_weight=config.desire_goal_weight,
        free_bits=config.free_bits,
        use_progress=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Total parameters: {total_params:,}")
    print(f"   ✅ Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    
    # KL annealing schedule
    kl_scheduler = KLAnnealingSchedule(
        strategy=config.kl_annealing_strategy,
        warmup_epochs=config.kl_warmup_epochs,
    )
    
    # Criteria
    criterion = {
        'goal': nn.CrossEntropyLoss(),
        'nextstep': nn.CrossEntropyLoss(),
        'category': nn.CrossEntropyLoss(),
    }
    
    # W&B
    wandb_logger = None
    if config.use_wandb:
        wandb_logger = EnhancedWandBLogger(
            project_name=config.wandb_project,
            config=vars(config),
            run_name=config.run_name,
        )
    
    # Training loop
    print("\n" + "=" * 100)
    print("🚀 STARTING TRAINING")
    print("=" * 100)
    
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        # Get KL weight for this epoch
        kl_weight = kl_scheduler.get_weight(epoch)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config.num_epochs} | KL Weight: {kl_weight:.3f}")
        print('='*80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            config, kl_weight, wandb_logger, scheduler
        )
        
        # Validate
        val_metrics, progress_metrics = validate(
            model, val_loader, criterion, device, config
        )
        
        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"\n📊 Train: loss={train_metrics['loss']:.4f}, "
              f"goal={train_metrics['goal_acc']:.1f}%, "
              f"desire_goal={train_metrics['desire_goal_acc']:.1f}%, "
              f"InfoNCE={train_metrics['infonce_loss']:.4f}")
        print(f"📊 Val: loss={val_metrics['loss']:.4f}, "
              f"goal={val_metrics['goal_acc']:.1f}%, "
              f"desire_goal={val_metrics['desire_goal_acc']:.1f}%")
        
        # Progress-stratified results
        print("\n📈 Goal Accuracy by Path Progress:")
        for bin_name, metrics in progress_metrics.items():
            print(f"   {bin_name}%: goal={metrics.get('goal_acc', 0):.1f}%, "
                  f"desire_goal={metrics.get('desire_goal_acc', 0):.1f}%")
        
        # Print diagnostics (especially helpful for first few epochs)
        if epoch < 5 or epoch % 10 == 0:
            print_epoch_diagnostics(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                num_poi_nodes=num_poi_nodes,
                num_categories=num_categories,
                kl_weight=kl_weight,
            )
        
        # W&B epoch logging
        if wandb_logger:
            wandb_logger.log_epoch(
                epoch, train_metrics, val_metrics, lr, kl_weight, progress_metrics
            )
        
        # Save best model
        if val_metrics['goal_acc'] > best_val_acc:
            best_val_acc = val_metrics['goal_acc']
            best_epoch = epoch
            save_checkpoint(
                filepath=str(checkpoint_dir / 'best_model.pt'),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metric=best_val_acc,
                is_best=True,
            )
            print(f"\n🎯 New best model! Val goal accuracy: {best_val_acc:.2f}%")
        
        # Periodic save
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                filepath=str(checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metric=val_metrics['goal_acc'],
                is_best=False,
            )
        
        # Visualization
        if config.visualize and VISUALIZATION_AVAILABLE and (epoch + 1) % config.visualize_every == 0:
            visualizer = TrainingVisualizer(save_dir='artifacts/diagnostics')
            visualizer.visualize_latent_space(
                model, val_loader, device, epoch + 1,
                max_samples=config.viz_samples,
            )
            visualizer.visualize_desire_goal_alignment(
                model, val_loader, device, epoch + 1,
                max_samples=config.viz_samples,
            )
        
        # Diagnostics - explain accuracy
        if epoch == 0:
            print_epoch_diagnostics(
                epoch,
                train_metrics,
                val_metrics,
                num_poi_nodes,
                num_categories,
                kl_weight,
            )
    
    # Final summary
    total_time = (time.time() - start_time) / 3600
    print("\n" + "=" * 100)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 100)
    print(f"   Best epoch: {best_epoch + 1}")
    print(f"   Best val goal accuracy: {best_val_acc:.2f}%")
    print(f"   Total time: {total_time:.2f} hours")
    
    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    main()
