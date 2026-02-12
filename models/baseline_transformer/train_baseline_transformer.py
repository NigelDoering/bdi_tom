"""
BASELINE TRANSFORMER TRAINING SCRIPT

This script trains the baseline Transformer model (PerNodeTransformerPredictor) on the
UCSD campus navigation dataset with pre-defined train/val/test splits.

KEY DIFFERENCE FROM LSTM:
- Uses full trajectories (70k samples vs 700k per-node samples)
- Transformer processes all positions in parallel with causal masking
- 10-30x faster training expected

Usage:
    python -m models.baseline_transformer.train_baseline_transformer [OPTIONS]

Example:
    python -m models.baseline_transformer.train_baseline_transformer --num_epochs 50 --batch_size 64

Data:
    - Trajectories: data/simulation_data/run_8/trajectories/trajectories.json
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
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.baseline_transformer.baseline_transformer_model import PerNodeTransformerPredictor
from models.baseline_transformer.baseline_transformer_dataset import TransformerTrajectoryDataset, collate_transformer_trajectories
from models.utils.data_loader import load_simulation_data
from models.utils.utils import get_device, set_seed, save_checkpoint, save_embedding_pipeline, AverageMeter

# W&B imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


# ============================================================================
# W&B LOGGER (FIXED)
# ============================================================================

class WandBLogger:
    """W&B logger with progress-stratified metrics matching BDI-VAE V3 format."""
    
    def __init__(self, project_name: str = "tom-compare-v1", config: Dict = None, run_name: str = None):
        self.enabled = WANDB_AVAILABLE
        self.global_step = 0
        
        if self.enabled:
            wandb.init(project=project_name, entity="nigeldoering-uc-san-diego", config=config or {}, name=run_name)
            print(f"✅ W&B initialized (project: {project_name}){f' (run: {run_name})' if run_name else ''}!")
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
        progress_metrics: Dict[str, Dict[str, float]] = None,
    ):
        """
        Log epoch-level metrics with progress-stratified accuracy.
        
        Matches the BDI-VAE V3 logging format:
          train/<key>, val/<key>, progress_<bin>/<key>
        """
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'learning_rate': lr,
        }
        
        for key, value in train_metrics.items():
            log_dict[f'train/{key}'] = value
        
        for key, value in val_metrics.items():
            log_dict[f'val/{key}'] = value
        
        # Log progress-stratified metrics (matching BDI-VAE V3 format)
        if progress_metrics is not None:
            for progress_bin, metrics in progress_metrics.items():
                for key, value in metrics.items():
                    log_dict[f'progress_{progress_bin}/{key}'] = value
        
        wandb.log(log_dict, step=self.global_step)
        self.global_step += 1
    
    def log_model_info(self, total_params: int, trainable_params: int, model_config: Dict):
        """Log model architecture information."""
        if not self.enabled:
            return
        
        info_dict = {
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/parameter_ratio': trainable_params / (total_params + 1e-8),
        }
        for key, value in model_config.items():
            info_dict[f'model_config/{key}'] = value
        wandb.log(info_dict, step=self.global_step)
        self.global_step += 1
    
    def log_summary(self, best_epoch: int, best_val_acc: float, total_epochs: int, total_time_hours: float):
        """Log training summary statistics."""
        if not self.enabled:
            return
        
        wandb.log({
            'summary/best_epoch': best_epoch,
            'summary/best_val_goal_acc': best_val_acc,
            'summary/total_epochs_trained': total_epochs,
            'summary/total_training_time_hours': total_time_hours,
            'summary/avg_time_per_epoch': total_time_hours / (total_epochs + 1e-8),
        })
    
    def log_checkpoint(self, epoch: int, checkpoint_path: str, best: bool = False):
        """Log checkpoint information."""
        if not self.enabled:
            return
        
        wandb.log({
            'checkpoint/epoch': epoch,
            'checkpoint/path': checkpoint_path,
            'checkpoint/is_best': best,
        })
    
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
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Transformer processes full trajectories with causal masking, predicting at all positions.
    """
    model.train()
    metrics = {
        'loss': AverageMeter(),
        'loss_goal': AverageMeter(),
        'loss_nextstep': AverageMeter(),
        'loss_category': AverageMeter(),
        'goal_acc': AverageMeter(),
        'nextstep_acc': AverageMeter(),
        'category_acc': AverageMeter(),
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        batch_size = batch['node_indices'].size(0)
        
        # Move to device
        node_indices = batch['node_indices'].to(device)          # [batch, max_len]
        next_indices = batch['next_indices'].to(device)          # [batch, max_len]
        goal_idx = batch['goal_idx'].to(device)                  # [batch]
        goal_cat_idx = batch['goal_cat_idx'].to(device)          # [batch]
        agent_ids = batch['agent_ids'].to(device)                # [batch]
        hours = batch['hours'].to(device)                        # [batch]
        padding_mask = batch['padding_mask'].to(device)          # [batch, max_len] (bool)
        seq_lengths = batch['seq_lengths'].to(device)            # [batch]
        
        # Forward pass - returns predictions at ALL positions
        predictions = model(node_indices, agent_ids, hours, padding_mask)
        # predictions: {'goal': [batch, seq_len, num_goals], 
        #               'nextstep': [batch, seq_len, num_nodes],
        #               'category': [batch, seq_len, num_cats]}
        
        # Flatten predictions and targets, ignoring padding positions
        # We need to compute loss only on non-padded positions
        
        batch_loss_goal = 0
        batch_loss_nextstep = 0
        batch_loss_category = 0
        total_valid_positions = 0
        
        # Track accuracies across all valid positions
        batch_goal_correct = 0
        batch_nextstep_correct = 0
        batch_category_correct = 0
        
        for i in range(batch_size):
            seq_len = seq_lengths[i].item()
            
            # Extract predictions for valid positions [seq_len, num_classes]
            pred_goal_i = predictions['goal'][i, :seq_len]           # [seq_len, num_goals]
            pred_nextstep_i = predictions['nextstep'][i, :seq_len]   # [seq_len, num_nodes]
            pred_category_i = predictions['category'][i, :seq_len]   # [seq_len, num_cats]
            
            # Extract targets for valid positions [seq_len]
            target_nextstep_i = next_indices[i, :seq_len]            # [seq_len]
            
            # Goal and category are constant for all positions
            target_goal_i = goal_idx[i].expand(seq_len)              # [seq_len]
            target_category_i = goal_cat_idx[i].expand(seq_len)      # [seq_len]
            
            # Compute losses
            batch_loss_goal += criterion['goal'](pred_goal_i, target_goal_i)
            batch_loss_nextstep += criterion['nextstep'](pred_nextstep_i, target_nextstep_i)
            batch_loss_category += criterion['category'](pred_category_i, target_category_i)
            
            # Compute accuracies
            batch_goal_correct += (pred_goal_i.argmax(dim=1) == target_goal_i).sum().item()
            batch_nextstep_correct += (pred_nextstep_i.argmax(dim=1) == target_nextstep_i).sum().item()
            batch_category_correct += (pred_category_i.argmax(dim=1) == target_category_i).sum().item()
            
            total_valid_positions += seq_len
        
        # Average losses across all valid positions
        loss_goal = batch_loss_goal / batch_size
        loss_nextstep = batch_loss_nextstep / batch_size
        loss_category = batch_loss_category / batch_size
        
        # Weighted loss
        loss = 1.0 * loss_goal + 0.5 * loss_nextstep + 0.5 * loss_category
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute average accuracies across all valid positions (as percentages)
        goal_acc = (batch_goal_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        nextstep_acc = (batch_nextstep_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        category_acc = (batch_category_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        
        # Update metrics (average across batches)
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, total_valid_positions)
        metrics['nextstep_acc'].update(nextstep_acc, total_valid_positions)
        metrics['category_acc'].update(category_acc, total_valid_positions)
        
        # Log batch metrics to W&B
        if wandb_logger is not None:
            batch_metrics = {
                'loss': loss.item(),
                'loss_goal': loss_goal.item(),
                'loss_nextstep': loss_nextstep.item(),
                'loss_category': loss_category.item(),
                'goal_acc': goal_acc,
                'nextstep_acc': nextstep_acc,
                'category_acc': category_acc,
            }
            lr = optimizer.param_groups[0]['lr']
            wandb_logger.log_batch(epoch, batch_idx, batch_size, batch_metrics, lr)
        
        pbar.set_postfix({
            'loss': f"{metrics['loss'].avg:.4f}",
            'goal_acc': f"{metrics['goal_acc'].avg:.1f}%",
            'next_acc': f"{metrics['nextstep_acc'].avg:.1f}%",
            'cat_acc': f"{metrics['category_acc'].avg:.1f}%",
        })
    
    # Return AVERAGE metrics across all batches
    return {k: v.avg for k, v in metrics.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    device: torch.device,
) -> tuple:
    """
    Validate the model with progress-stratified metrics.
    
    For the transformer, each position in the trajectory has an implicit progress
    value: position / seq_len. We bin these into 0-25%, 25-50%, 50-75%, 75-100%
    to match the BDI-VAE V3 progress-stratified format.
    
    Returns:
        Tuple of (average_metrics, progress_metrics)
        - average_metrics: Dict with averaged metrics across all positions
        - progress_metrics: Dict with metrics at 0-25%, 25-50%, 50-75%, 75-100% progress bins
    """
    model.eval()
    metrics = {
        'loss': AverageMeter(),
        'loss_goal': AverageMeter(),
        'loss_nextstep': AverageMeter(),
        'loss_category': AverageMeter(),
        'goal_acc': AverageMeter(),
        'nextstep_acc': AverageMeter(),
        'category_acc': AverageMeter(),
    }
    
    # Progress-stratified tracking (matching BDI-VAE V3 format)
    progress_bins = ['0-25', '25-50', '50-75', '75-100']
    progress_meters = {
        bin_name: defaultdict(AverageMeter) for bin_name in progress_bins
    }
    
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch in pbar:
        batch_size = batch['node_indices'].size(0)
        
        # Move to device
        node_indices = batch['node_indices'].to(device)
        next_indices = batch['next_indices'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        agent_ids = batch['agent_ids'].to(device)
        hours = batch['hours'].to(device)
        padding_mask = batch['padding_mask'].to(device)  # [batch, max_len] (bool)
        seq_lengths = batch['seq_lengths'].to(device)
        
        # Forward pass
        predictions = model(node_indices, agent_ids, hours, padding_mask)
        
        # Compute losses and accuracies
        batch_loss_goal = 0
        batch_loss_nextstep = 0
        batch_loss_category = 0
        total_valid_positions = 0
        
        batch_goal_correct = 0
        batch_nextstep_correct = 0
        batch_category_correct = 0
        
        for i in range(batch_size):
            seq_len = seq_lengths[i].item()
            
            # Extract predictions for valid positions
            pred_goal_i = predictions['goal'][i, :seq_len]
            pred_nextstep_i = predictions['nextstep'][i, :seq_len]
            pred_category_i = predictions['category'][i, :seq_len]
            
            # Extract targets
            target_nextstep_i = next_indices[i, :seq_len]
            target_goal_i = goal_idx[i].expand(seq_len)
            target_category_i = goal_cat_idx[i].expand(seq_len)
            
            # Compute losses
            batch_loss_goal += criterion['goal'](pred_goal_i, target_goal_i)
            batch_loss_nextstep += criterion['nextstep'](pred_nextstep_i, target_nextstep_i)
            batch_loss_category += criterion['category'](pred_category_i, target_category_i)
            
            # Compute accuracies
            goal_correct_i = (pred_goal_i.argmax(dim=1) == target_goal_i)
            nextstep_correct_i = (pred_nextstep_i.argmax(dim=1) == target_nextstep_i)
            category_correct_i = (pred_category_i.argmax(dim=1) == target_category_i)
            
            batch_goal_correct += goal_correct_i.sum().item()
            batch_nextstep_correct += nextstep_correct_i.sum().item()
            batch_category_correct += category_correct_i.sum().item()
            
            total_valid_positions += seq_len
            
            # Progress-stratified per-position metrics
            for pos in range(seq_len):
                # Progress: how far through the trajectory (0.0 to 1.0)
                progress = pos / (seq_len - 1) if seq_len > 1 else 0.0
                
                if progress < 0.25:
                    bin_name = '0-25'
                elif progress < 0.5:
                    bin_name = '25-50'
                elif progress < 0.75:
                    bin_name = '50-75'
                else:
                    bin_name = '75-100'
                
                progress_meters[bin_name]['goal_acc'].update(
                    goal_correct_i[pos].float().item() * 100, 1
                )
                progress_meters[bin_name]['nextstep_acc'].update(
                    nextstep_correct_i[pos].float().item() * 100, 1
                )
        
        # Average losses
        loss_goal = batch_loss_goal / batch_size
        loss_nextstep = batch_loss_nextstep / batch_size
        loss_category = batch_loss_category / batch_size
        loss = 1.0 * loss_goal + 0.5 * loss_nextstep + 0.5 * loss_category
        
        # Compute average accuracies (as percentages)
        goal_acc = (batch_goal_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        nextstep_acc = (batch_nextstep_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        category_acc = (batch_category_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, total_valid_positions)
        metrics['nextstep_acc'].update(nextstep_acc, total_valid_positions)
        metrics['category_acc'].update(category_acc, total_valid_positions)
        
        pbar.set_postfix({'loss': f"{metrics['loss'].avg:.4f}"})
    
    overall = {k: v.avg for k, v in metrics.items()}
    progress = {
        bin_name: {k: v.avg for k, v in bin_meters.items()}
        for bin_name, bin_meters in progress_meters.items()
    }
    
    return overall, progress


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
    parser = argparse.ArgumentParser(description='Train Baseline Transformer Model')
    
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
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Embedding configuration
    parser.add_argument('--node_embedding_dim', type=int, default=128,
                       help='Node embedding dimension')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (can be larger than LSTM since no expansion)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Data processing
    parser.add_argument('--min_traj_length', type=int, default=2,
                       help='Minimum trajectory length')
    parser.add_argument('--max_traj_length', type=int, default=50,
                       help='Maximum trajectory length')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/baseline_transformer',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--save_embedding_pipeline', action='store_true',
                       help='Save embedding pipeline separately for transfer learning')
    
    # Logging
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Name for W&B run (default: auto-generated)')
    
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
    
    # Create datasets (NO per-node expansion!)
    print("\n" + "=" * 100)
    print("📊 CREATING DATASETS (NO EXPANSION - FULL TRAJECTORIES)")
    print("=" * 100)
    
    print("\n🔹 Creating training dataset...")
    train_dataset = TransformerTrajectoryDataset(
        train_trajs,
        graph,
        poi_nodes,
        min_traj_length=args.min_traj_length,
        max_traj_length=args.max_traj_length,
    )
    
    print("\n🔹 Creating validation dataset...")
    val_dataset = TransformerTrajectoryDataset(
        val_trajs,
        graph,
        poi_nodes,
        node_to_idx_map=train_dataset.node_to_idx,  # Use same mapping
        min_traj_length=args.min_traj_length,
        max_traj_length=args.max_traj_length,
    )
    
    print("\n🔹 Creating test dataset...")
    test_dataset = TransformerTrajectoryDataset(
        test_trajs,
        graph,
        poi_nodes,
        node_to_idx_map=train_dataset.node_to_idx,  # Use same mapping
        min_traj_length=args.min_traj_length,
        max_traj_length=args.max_traj_length,
    )
    
    print(f"\n✅ Dataset sizes:")
    print(f"   Train: {len(train_dataset)} trajectories (NO expansion!)")
    print(f"   Val:   {len(val_dataset)} trajectories")
    print(f"   Test:  {len(test_dataset)} trajectories")
    print("=" * 100 + "\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_transformer_trajectories,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_transformer_trajectories,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model
    print("\n" + "=" * 100)
    print("🏗️  CREATING MODEL")
    print("=" * 100)
    
    model = PerNodeTransformerPredictor(
        num_nodes=len(graph.nodes),
        num_agents=100,  # 100 agents from simulation
        num_poi_nodes=len(poi_nodes),
        num_categories=7,
        node_embedding_dim=args.node_embedding_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created!")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print("=" * 100 + "\n")
    
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
    
    # Loss functions
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
        })
        wandb_logger = WandBLogger(
            project_name="tom-compare-v1", 
            config=config,
            run_name=args.wandb_run_name
        )
        
        # Log model info
        model_config = {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'node_embedding_dim': args.node_embedding_dim,
        }
        wandb_logger.log_model_info(total_params, trainable_params, model_config)
    
    # Training loop
    print("\n" + "=" * 100)
    print("🚀 STARTING TRAINING")
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
            model, train_loader, optimizer, criterion, device, epoch, wandb_logger
        )
        
        # Validate with progress-stratified metrics
        val_metrics, progress_metrics = validate(
            model, val_loader, criterion, device,
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to W&B with progress-stratified metrics
        if wandb_logger is not None:
            wandb_logger.log_epoch(
                epoch, train_metrics, val_metrics, current_lr,
                progress_metrics=progress_metrics
            )
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train - Loss: {train_metrics['loss']:.4f} | Goal: {train_metrics['goal_acc']:.3f} | Next: {train_metrics['nextstep_acc']:.3f} | Cat: {train_metrics['category_acc']:.3f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f} | Goal: {val_metrics['goal_acc']:.3f} | Next: {val_metrics['nextstep_acc']:.3f} | Cat: {val_metrics['category_acc']:.3f}")
        
        # Print progress-stratified results
        print(f"\n   📈 Goal Accuracy by Path Progress:")
        for bin_name, metrics in progress_metrics.items():
            print(f"      {bin_name}%: goal={metrics.get('goal_acc', 0):.1f}%")
        
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
            print(f"   ✅ New best model! Val Goal Acc: {best_val_goal_acc:.3f}")
            
            # Save embedding pipeline separately if requested
            if args.save_embedding_pipeline:
                embedding_path = os.path.join(args.checkpoint_dir, 'best_embedding_pipeline.pt')
                embedding_config = {
                    'node_embedding_dim': args.node_embedding_dim,
                    'num_nodes': len(graph.nodes),
                    'num_categories': 7,
                    'num_agents': 100,
                }
                save_embedding_pipeline(embedding_path, model, epoch, config=embedding_config)
            
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
            
            # Save embedding pipeline separately if requested
            if args.save_embedding_pipeline:
                embedding_path = os.path.join(args.checkpoint_dir, f'embedding_pipeline_epoch_{epoch}.pt')
                embedding_config = {
                    'node_embedding_dim': args.node_embedding_dim,
                    'num_nodes': len(graph.nodes),
                    'num_categories': 7,
                    'num_agents': 100,
                }
                save_embedding_pipeline(embedding_path, model, epoch, config=embedding_config)
            
            if wandb_logger is not None:
                wandb_logger.log_checkpoint(epoch, checkpoint_path, best=False)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏸️  Early stopping triggered after {epoch} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n" + "=" * 100)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 100)
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Val Goal Acc: {best_val_goal_acc:.3f}")
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
