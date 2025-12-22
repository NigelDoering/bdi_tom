"""
TRAIN TRANSFORMER + GAT MODEL

This script trains the Transformer+GAT model on campus navigation data.
The model combines:
1. GAT encoder (captures global graph structure)
2. Unified embeddings (trajectory-specific features)
3. Fusion layer (combines both perspectives)
4. Transformer (sequence modeling with causal masking)
5. Multi-task heads (goal, next step, category prediction)

COMPARISON TO BASELINE:
- Baseline: Uses only Node2Vec static embeddings
- This model: Adds GAT for learned, attention-based graph embeddings
- Both use same dataset, loss functions, and training procedure
- GAT should provide better graph structure awareness

USAGE:
    python models/transformer_gat/train_transformer_gat.py [--options]

OPTIONS:
    --epochs: Number of training epochs (default: 50)
    --batch_size: Batch size (default: 256)
    --lr: Learning rate (default: 0.0001)
    --weight_decay: Weight decay (default: 0.0001)
    --dropout: Dropout rate (default: 0.1)
    --freeze_gat: Freeze GAT encoder (default: False)
    --freeze_unified: Freeze unified embeddings (default: False)
    --use_wandb: Enable W&B logging (default: True)
    --project: W&B project name (default: "transformer-gat-campus-nav")
    --run_name: W&B run name (default: auto-generated)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import networkx as nx
import wandb
from tqdm import tqdm
import numpy as np

# Import model
from models.transformer_gat.transformer_gat_model import TransformerGATPredictor

# Import dataset (reuse transformer dataset)
from models.baseline_transformer.baseline_transformer_dataset import (
    TransformerTrajectoryDataset,
    collate_transformer_trajectories
)

# Import utilities
from models.utils.data_loader import load_simulation_data
from models.utils.utils import get_device, set_seed, AverageMeter
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Transformer+GAT model")
    
    # Data paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/simulation_data/run_8',
        help='Directory containing trajectory data'
    )
    parser.add_argument(
        '--graph_path',
        type=str,
        default='data/processed/ucsd_walk_full.graphml',
        help='Path to graph file'
    )
    parser.add_argument(
        '--split_indices_path',
        type=str,
        default='data/simulation_data/run_8/split_data/split_indices_seed42.json',
        help='Path to JSON file with pre-defined train/val/test split indices'
    )
    parser.add_argument(
        '--trajectory_filename',
        type=str,
        default='trajectories/all_trajectories.json',
        help='Path to trajectory file relative to data_dir'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/transformer_gat',
        help='Directory to save model checkpoints'
    )
    
    # Model architecture
    parser.add_argument('--gat_hidden_dim', type=int, default=128, help='GAT hidden dimension')
    parser.add_argument('--gat_output_dim', type=int, default=128, help='GAT output dimension')
    parser.add_argument('--gat_num_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--gat_num_heads', type=int, default=4, help='Number of GAT attention heads')
    parser.add_argument('--node_embedding_dim', type=int, default=128, help='Node embedding dimension')
    parser.add_argument('--temporal_dim', type=int, default=128, help='Temporal embedding dimension')
    parser.add_argument('--agent_dim', type=int, default=128, help='Agent embedding dimension')
    parser.add_argument('--fusion_dim', type=int, default=256, help='Unified fusion dimension')
    parser.add_argument('--combined_fusion_dim', type=int, default=384, help='Combined fusion dimension')
    parser.add_argument('--d_model', type=int, default=512, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of transformer attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Transformer feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Loss weights
    parser.add_argument('--goal_weight', type=float, default=1.0, help='Goal loss weight')
    parser.add_argument('--nextstep_weight', type=float, default=1.0, help='Next step loss weight')
    parser.add_argument('--category_weight', type=float, default=0.5, help='Category loss weight')
    
    # Control flags
    parser.add_argument('--freeze_gat', action='store_true', help='Freeze GAT encoder')
    parser.add_argument('--freeze_unified', action='store_true', help='Freeze unified embeddings')
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')
    
    # W&B settings
    parser.add_argument('--project', type=str, default='transformer-gat-campus-nav', help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
    
    # Data settings
    parser.add_argument('--min_traj_length', type=int, default=2, help='Minimum trajectory length')
    parser.add_argument('--max_traj_length', type=int, default=60, help='Maximum trajectory length')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class WandBLogger:
    """W&B logger with proper metric definitions."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.global_step = 0
    
    def init(self, project: str, run_name: Optional[str], config: Dict):
        """Initialize W&B."""
        if not self.enabled:
            return
        
        wandb.init(project=project, name=run_name, config=config)
        
        # Define metrics and their axes
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        
        # Epoch metrics
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("learning_rate", step_metric="epoch")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        val_percentile_metrics: Dict[str, Dict[str, float]] = None
    ):
        """Log epoch-level metrics."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'learning_rate': lr,
            'train/loss': train_metrics.get('loss', 0),
            'train/goal_acc': train_metrics.get('goal_acc', 0),
            'val/loss': val_metrics.get('loss', 0),
            'val/goal_acc': val_metrics.get('goal_acc', 0),
        }
        
        # Log val percentiles (already in percentage format)
        if val_percentile_metrics is not None:
            for pct in ['15%', '50%', '85%']:
                if pct in val_percentile_metrics:
                    log_dict[f'val/goal_acc_{pct}'] = val_percentile_metrics[pct].get('goal_acc', 0)
        
        wandb.log(log_dict)
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion: Dict[str, nn.Module],
    device: torch.device,
    loss_weights: Dict[str, float],
    grad_clip: float,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary of training metrics (averaged)
    """
    model.train()
    
    metrics = {
        'loss': AverageMeter(),
        'loss_goal': AverageMeter(),
        'loss_nextstep': AverageMeter(),
        'loss_category': AverageMeter(),
        'goal_acc': AverageMeter(),
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch in pbar:
        batch_size = batch['node_indices'].size(0)
        
        # Move batch to device
        node_indices = batch['node_indices'].to(device)
        next_indices = batch['next_indices'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        agent_ids = batch['agent_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        
        # Forward pass
        predictions = model(
            node_indices=node_indices,
            agent_ids=agent_ids,
            padding_mask=padding_mask
        )
        
        # Compute losses and accuracies (per-position like baseline)
        batch_loss_goal = 0
        batch_loss_nextstep = 0
        batch_loss_category = 0
        total_valid_positions = 0
        batch_goal_correct = 0
        
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
            
            # Compute goal accuracy
            goal_correct_i = (pred_goal_i.argmax(dim=1) == target_goal_i)
            batch_goal_correct += goal_correct_i.sum().item()
            
            total_valid_positions += seq_len
        
        # Average losses
        loss_goal = batch_loss_goal / batch_size
        loss_nextstep = batch_loss_nextstep / batch_size
        loss_category = batch_loss_category / batch_size
        loss = (
            loss_weights['goal'] * loss_goal +
            loss_weights['nextstep'] * loss_nextstep +
            loss_weights['category'] * loss_category
        )
        
        # Compute goal accuracy as percentage
        goal_acc = (batch_goal_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, total_valid_positions)
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{metrics['loss'].avg:.4f}", 'goal_acc': f"{metrics['goal_acc'].avg:.2f}%"})
    
    return {k: v.avg for k, v in metrics.items()}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    loss_weights: Dict[str, float],
    epoch: int,
    compute_percentiles: bool = False
) -> Tuple[Dict[str, float], Optional[Dict[str, Dict[str, float]]]]:
    """
    Validate for one epoch.
    
    Returns:
        Tuple of (average_metrics, percentile_metrics)
    """
    model.eval()
    
    metrics = {
        'loss': AverageMeter(),
        'loss_goal': AverageMeter(),
        'loss_nextstep': AverageMeter(),
        'loss_category': AverageMeter(),
        'goal_acc': AverageMeter(),
    }
    
    # For percentile tracking
    percentile_data = {
        '15%': {'goal_correct': 0, 'count': 0},
        '50%': {'goal_correct': 0, 'count': 0},
        '85%': {'goal_correct': 0, 'count': 0},
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for batch in pbar:
        batch_size = batch['node_indices'].size(0)
        
        # Move to device
        node_indices = batch['node_indices'].to(device)
        next_indices = batch['next_indices'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        agent_ids = batch['agent_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        
        # Forward pass
        predictions = model(
            node_indices=node_indices,
            agent_ids=agent_ids,
            padding_mask=padding_mask
        )
        
        # Compute losses and accuracies
        batch_loss_goal = 0
        batch_loss_nextstep = 0
        batch_loss_category = 0
        total_valid_positions = 0
        batch_goal_correct = 0
        
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
            
            # Compute goal accuracy
            goal_correct_i = (pred_goal_i.argmax(dim=1) == target_goal_i)
            batch_goal_correct += goal_correct_i.sum().item()
            
            total_valid_positions += seq_len
            
            # Compute percentile-specific accuracies
            if compute_percentiles and seq_len > 0:
                # Get positions at 15%, 50%, 85% of trajectory
                pos_15 = max(0, int(seq_len * 0.15) - 1)
                pos_50 = max(0, int(seq_len * 0.50) - 1)
                pos_85 = max(0, int(seq_len * 0.85) - 1)
                
                # Track goal accuracy at each percentile
                percentile_data['15%']['goal_correct'] += goal_correct_i[pos_15].item()
                percentile_data['15%']['count'] += 1
                
                percentile_data['50%']['goal_correct'] += goal_correct_i[pos_50].item()
                percentile_data['50%']['count'] += 1
                
                percentile_data['85%']['goal_correct'] += goal_correct_i[pos_85].item()
                percentile_data['85%']['count'] += 1
        
        # Average losses
        loss_goal = batch_loss_goal / batch_size
        loss_nextstep = batch_loss_nextstep / batch_size
        loss_category = batch_loss_category / batch_size
        loss = (
            loss_weights['goal'] * loss_goal +
            loss_weights['nextstep'] * loss_nextstep +
            loss_weights['category'] * loss_category
        )
        
        # Compute goal accuracy as percentage
        goal_acc = (batch_goal_correct / total_valid_positions * 100) if total_valid_positions > 0 else 0
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, total_valid_positions)
        
        pbar.set_postfix({'loss': f"{metrics['loss'].avg:.4f}", 'goal_acc': f"{metrics['goal_acc'].avg:.2f}%"})
    
    # Compute percentile metrics (as percentages)
    percentile_metrics = None
    if compute_percentiles:
        percentile_metrics = {}
        for pct, data in percentile_data.items():
            if data['count'] > 0:
                percentile_metrics[pct] = {
                    'goal_acc': (data['goal_correct'] / data['count']) * 100,  # Convert to percentage
                }
    
    return {k: v.avg for k, v in metrics.items()}, percentile_metrics


def main():
    """Main training loop."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # ================================================================
    # LOAD DATA
    # ================================================================
    print(f"\n📊 Loading data...")
    
    # Load graph
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    # Load trajectories with pre-defined splits
    print(f"\n📂 Loading trajectories from: {args.data_dir}")
    trajectories, graph, poi_nodes = load_simulation_data(
        args.data_dir,
        args.graph_path,
        args.trajectory_filename
    )
    print(f"   ✅ Loaded {len(trajectories)} trajectories")
    print(f"   ✅ Loaded graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"   ✅ Found {len(poi_nodes)} POI nodes")
    
    # Load split indices
    print(f"\n📊 Loading split indices from: {args.split_indices_path}")
    with open(args.split_indices_path, 'r') as f:
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
    
    # Create datasets
    print(f"\n📦 Creating datasets...")
    train_dataset = TransformerTrajectoryDataset(
        trajectories=train_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        min_traj_length=args.min_traj_length,
        max_traj_length=args.max_traj_length
    )
    
    val_dataset = TransformerTrajectoryDataset(
        trajectories=val_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=train_dataset.node_to_idx,
        min_traj_length=args.min_traj_length,
        max_traj_length=args.max_traj_length
    )
    
    print(f"   ✅ Train dataset: {len(train_dataset)} samples")
    print(f"   ✅ Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_transformer_trajectories,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_transformer_trajectories,
        pin_memory=True
    )
    
    # ================================================================
    # CREATE MODEL
    # ================================================================
    print(f"\n🏗️  Creating model...")
    
    num_nodes = len(graph.nodes())
    num_poi_nodes = len(poi_nodes)
    num_agents = 10  # Fixed for this dataset
    num_categories = 7
    
    model = TransformerGATPredictor(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=num_categories,
        # GAT params
        gat_hidden_dim=args.gat_hidden_dim,
        gat_output_dim=args.gat_output_dim,
        gat_num_layers=args.gat_num_layers,
        gat_num_heads=args.gat_num_heads,
        # Unified params
        node_embedding_dim=args.node_embedding_dim,
        temporal_dim=args.temporal_dim,  # Use args instead of hardcoded value
        agent_dim=args.agent_dim,
        fusion_dim=args.fusion_dim,
        # Combined fusion
        combined_fusion_dim=args.combined_fusion_dim,
        # Transformer params
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        # Control flags
        freeze_gat=args.freeze_gat,
        freeze_unified=args.freeze_unified,
    ).to(device)
    
    # Set graph structure
    node_to_idx = train_dataset.node_to_idx
    model.set_graph_structure(graph, node_to_idx)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model created")
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   📊 Trainable parameters: {trainable_params:,}")
    
    # ================================================================
    # SETUP TRAINING
    # ================================================================
    print(f"\n⚙️  Setting up training...")
    
    # Loss weights
    loss_weights = {
        'goal': args.goal_weight,
        'nextstep': args.nextstep_weight,
        'category': args.category_weight
    }
    print(f"   Loss weights: {loss_weights}")
    
    # Loss functions
    criterion = {
        'goal': nn.CrossEntropyLoss(),
        'nextstep': nn.CrossEntropyLoss(),
        'category': nn.CrossEntropyLoss(),
    }
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler (cosine with warmup)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
        anneal_strategy='cos'
    )
    
    # ================================================================
    # INITIALIZE W&B
    # ================================================================
    use_wandb = not args.no_wandb
    logger = WandBLogger(enabled=use_wandb)
    
    if use_wandb:
        logger.init(
            project=args.project,
            run_name=args.run_name,
            config=vars(args)
        )
        print(f"   ✅ W&B initialized: {args.project}/{args.run_name or 'auto'}")
    
    # ================================================================
    # TRAINING LOOP
    # ================================================================
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    best_val_loss = float('inf')
    best_val_goal_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            loss_weights=loss_weights,
            grad_clip=args.grad_clip,
            epoch=epoch
        )
        
        # Validate (with percentiles)
        val_metrics, val_percentile_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            loss_weights=loss_weights,
            epoch=epoch,
            compute_percentiles=True
        )
        
        # Get current LR
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Goal Acc: {train_metrics['goal_acc']:.2f}%")
        print(f"   Val Loss: {val_metrics['loss']:.4f} | Goal Acc: {val_metrics['goal_acc']:.2f}%")
        
        if val_percentile_metrics:
            print(f"   Val Goal Acc @ 15%: {val_percentile_metrics.get('15%', {}).get('goal_acc', 0):.2f}%")
            print(f"   Val Goal Acc @ 50%: {val_percentile_metrics.get('50%', {}).get('goal_acc', 0):.2f}%")
            print(f"   Val Goal Acc @ 85%: {val_percentile_metrics.get('85%', {}).get('goal_acc', 0):.2f}%")
        
        # Log to W&B
        logger.log_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=current_lr,
            val_percentile_metrics=val_percentile_metrics
        )
        
        # Save best model (by validation loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, save_path)
            print(f"   💾 Saved best model (val_loss={best_val_loss:.4f})")
        
        # Save best model (by goal accuracy)
        val_goal_acc = val_metrics['goal_acc']
        if val_goal_acc > best_val_goal_acc:
            best_val_goal_acc = val_goal_acc
            save_path = output_dir / 'best_model_goal_acc.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, save_path)
            print(f"   💾 Saved best model (goal_acc={best_val_goal_acc:.2f}%)")
    
    # ================================================================
    # FINISH
    # ================================================================
    print(f"\n{'='*80}")
    print(f"✅ Training complete!")
    print(f"{'='*80}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Best val goal accuracy: {best_val_goal_acc:.2f}%")
    print(f"   Models saved to: {output_dir}")
    
    logger.finish()


if __name__ == '__main__':
    main()
