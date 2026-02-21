"""
BASELINE LSTM TRAINING SCRIPT

This script trains the baseline LSTM model (PerNodeToMPredictor) on the
UCSD campus navigation dataset with pre-defined train/val/test splits.

Usage:
    python -m models.baseline_lstm.train_baseline_lstm [OPTIONS]

Example:
    python -m models.baseline_lstm.train_baseline_lstm --num_epochs 50 --batch_size 32

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
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.baseline_lstm import PerNodeToMPredictor, PerNodeTrajectoryDataset, collate_per_node_samples
from models.utils.data_loader import load_simulation_data
from models.utils.utils import get_device, set_seed, save_checkpoint, save_embedding_pipeline, compute_accuracy, AverageMeter

# W&B imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


# ============================================================================
# W&B LOGGER
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
        wandb.log(info_dict)
    
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
    """Train for one epoch."""
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
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        agent_ids = batch['agent_id'].to(device)
        hours = batch['hours'].to(device)
        
        # Forward pass
        predictions = model(history_node_indices, history_lengths, agent_ids, hours)
        
        # Compute losses
        loss_goal = criterion['goal'](predictions['goal'], goal_idx)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node_idx)
        loss_category = criterion['category'](predictions['category'], goal_cat_idx)
        
        # Weighted loss
        loss = 1.0 * loss_goal + 0.5 * loss_nextstep + 0.5 * loss_category
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute accuracies
        goal_acc = compute_accuracy(predictions['goal'], goal_idx)
        nextstep_acc = compute_accuracy(predictions['nextstep'], next_node_idx)
        category_acc = compute_accuracy(predictions['category'], goal_cat_idx)
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
        
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
            'goal_acc': f"{metrics['goal_acc'].avg:.3f}",
        })
    
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
    
    Returns:
        Tuple of (overall_metrics, progress_metrics)
        - overall_metrics: Dict with averaged metrics
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
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        agent_ids = batch['agent_id'].to(device)
        hours = batch['hours'].to(device)
        path_progress = batch['path_progress']  # keep on CPU for binning
        
        # Forward pass
        predictions = model(history_node_indices, history_lengths, agent_ids, hours)
        
        # Compute losses
        loss_goal = criterion['goal'](predictions['goal'], goal_idx)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node_idx)
        loss_category = criterion['category'](predictions['category'], goal_cat_idx)
        
        # Weighted loss
        loss = 1.0 * loss_goal + 0.5 * loss_nextstep + 0.5 * loss_category
        
        # Compute accuracies
        goal_acc = compute_accuracy(predictions['goal'], goal_idx)
        nextstep_acc = compute_accuracy(predictions['nextstep'], next_node_idx)
        category_acc = compute_accuracy(predictions['category'], goal_cat_idx)
        
        # Update overall metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
        
        # Progress-stratified per-sample metrics
        goal_preds = predictions['goal'].argmax(dim=1)
        nextstep_preds = predictions['nextstep'].argmax(dim=1)
        goal_correct = (goal_preds == goal_idx)
        nextstep_correct = (nextstep_preds == next_node_idx)
        
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
    trajectory_filename: str = 'trajectories.json',
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
    
    # Load all trajectories
    trajectories, graph, poi_nodes = load_simulation_data(
        data_dir,
        graph_path,
        trajectory_filename=trajectory_filename
    )
    
    print(f"\n✅ Loaded {len(trajectories)} total trajectories")
    print(f"✅ Graph has {len(graph.nodes())} nodes")
    print(f"✅ Found {len(poi_nodes)} POI nodes")
    
    # Load split indices
    print(f"\n📂 Loading split indices from {split_indices_path}")
    with open(split_indices_path, 'r') as f:
        split_data = json.load(f)
    
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']
    test_indices = split_data['test_indices']
    
    # Create splits
    train_trajs = [trajectories[i] for i in train_indices]
    val_trajs = [trajectories[i] for i in val_indices]
    test_trajs = [trajectories[i] for i in test_indices]
    
    print(f"\n📊 Data Split (seed={split_data.get('seed', 'unknown')}):")
    print(f"   Train: {len(train_trajs)} ({len(train_trajs)/len(trajectories)*100:.1f}%)")
    print(f"   Val:   {len(val_trajs)} ({len(val_trajs)/len(trajectories)*100:.1f}%)")
    print(f"   Test:  {len(test_trajs)} ({len(test_trajs)/len(trajectories)*100:.1f}%)")
    
    return train_trajs, val_trajs, test_trajs, graph, poi_nodes


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main(args):
    """Main training orchestration."""
    
    print("\n" + "=" * 100)
    print("🧠 BASELINE LSTM TRAINING")
    print("=" * 100)
    
    device = get_device()
    set_seed(args.seed)
    
    print(f"\n📍 Device: {device}")
    print(f"📍 Seed: {args.seed}")
    
    # ================================================================
    # STEP 1: LOAD DATA WITH PRE-DEFINED SPLITS
    # ================================================================
    train_trajs, val_trajs, test_trajs, graph, poi_nodes = load_data_with_splits(
        data_dir=args.data_dir,
        graph_path=args.graph_path,
        split_indices_path=args.split_indices_path,
        trajectory_filename=args.trajectory_filename,
    )
    
    # ================================================================
    # STEP 2: CREATE DATASETS AND LOADERS
    # ================================================================
    print("\n" + "=" * 100)
    print("📊 CREATING DATASETS")
    print("=" * 100)
    
    # Create node-to-index mapping (shared across all datasets)
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    
    print("\n🏋️  Creating training dataset...")
    train_dataset = PerNodeTrajectoryDataset(train_trajs, graph, poi_nodes, node_to_idx)
    
    print("\n🏋️  Creating validation dataset...")
    val_dataset = PerNodeTrajectoryDataset(val_trajs, graph, poi_nodes, node_to_idx)
    
    print("\n🏋️  Creating test dataset...")
    test_dataset = PerNodeTrajectoryDataset(test_trajs, graph, poi_nodes, node_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_per_node_samples,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_per_node_samples,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_per_node_samples,
        num_workers=0,
    )
    
    print(f"\n✅ Train loader: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"✅ Val loader: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"✅ Test loader: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    # ================================================================
    # STEP 3: CREATE MODEL
    # ================================================================
    print("\n" + "=" * 100)
    print("🏗️  CREATING MODEL")
    print("=" * 100)
    
    model = PerNodeToMPredictor(
        num_nodes=len(graph.nodes()),
        num_agents=100,  # 100 agents to match transformer pretrained embeddings
        num_poi_nodes=len(poi_nodes),
        num_categories=7,
        node_embedding_dim=args.node_embedding_dim,
        temporal_dim=args.temporal_dim,
        agent_dim=args.agent_dim,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads,
        freeze_embedding=args.freeze_embedding,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Model created: PerNodeToMPredictor")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
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
    
    # Store config for checkpoint saving
    model_config = {
        'num_nodes': len(graph.nodes()),
        'num_agents': 1,
        'num_poi_nodes': len(poi_nodes),
        'num_categories': 7,
        'node_embedding_dim': args.node_embedding_dim,
        'temporal_dim': args.temporal_dim,
        'agent_dim': args.agent_dim,
        'fusion_dim': args.fusion_dim,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_heads': args.num_heads,
        'freeze_embedding': args.freeze_embedding,
    }
    
    # ================================================================
    # STEP 4: SETUP OPTIMIZATION
    # ================================================================
    print("\n" + "=" * 100)
    print("⚙️  SETTING UP OPTIMIZATION")
    print("=" * 100)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    criterion = {
        'goal': nn.CrossEntropyLoss(),
        'nextstep': nn.CrossEntropyLoss(),
        'category': nn.CrossEntropyLoss(),
    }
    
    print(f"\n✅ Optimizer: AdamW (lr={args.learning_rate}, weight_decay={args.weight_decay})")
    print(f"✅ Scheduler: CosineAnnealingLR (T_max={args.num_epochs})")
    print(f"✅ Loss functions: CrossEntropyLoss (goal, nextstep, category)")
    
    # ================================================================
    # STEP 5: INITIALIZE W&B LOGGING
    # ================================================================
    logger = WandBLogger(
        project_name="tom-compare-v1",
        run_name=args.wandb_run_name,
        config={
            'model': 'PerNodeToMPredictor',
            'architecture': 'baseline_lstm',
            'num_nodes': len(graph.nodes()),
            'num_poi_nodes': len(poi_nodes),
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_dim': args.hidden_dim,
            'num_epochs': args.num_epochs,
            'weight_decay': args.weight_decay,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'data_dir': args.data_dir,
            'graph_path': args.graph_path,
            'seed': args.seed,
        }
    )
    
    # Log model information
    logger.log_model_info(total_params, trainable_params, model_config)
    
    # ================================================================
    # STEP 6: TRAINING LOOP
    # ================================================================
    print("\n" + "=" * 100)
    print("🚀 STARTING TRAINING")
    print("=" * 100)
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        val_metrics, progress_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_metrics, val_metrics, lr, progress_metrics)
        
        print(f"\n📊 Epoch {epoch+1}/{args.num_epochs}")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Train Goal Acc: {train_metrics['goal_acc']:.3f} | Val Goal Acc: {val_metrics['goal_acc']:.3f}")
        print(f"   Train Cat Acc: {train_metrics['category_acc']:.3f} | Val Cat Acc: {val_metrics['category_acc']:.3f}")
        
        # Print progress-stratified results
        print(f"\n   📈 Goal Accuracy by Path Progress:")
        for bin_name, metrics in progress_metrics.items():
            print(f"      {bin_name}%: goal={metrics.get('goal_acc', 0):.1f}%")
        
        # Save checkpoint
        if val_metrics['goal_acc'] > best_val_acc:
            best_val_acc = val_metrics['goal_acc']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(
                str(checkpoint_path),
                model,
                optimizer,
                epoch,
                val_metrics['goal_acc'],
                is_best=True,
                config=model_config
            )
            
            # Save embedding pipeline separately if requested
            if args.save_embedding_pipeline:
                embedding_path = checkpoint_dir / "best_embedding_pipeline.pt"
                embedding_config = {
                    'node_embedding_dim': args.node_embedding_dim,
                    'use_temporal': False,  # LSTM doesn't use temporal/agent encoders
                    'use_agent': False,
                    'num_nodes': len(graph.nodes),
                    'num_categories': 7,
                }
                save_embedding_pipeline(str(embedding_path), model, epoch, config=embedding_config)
            
            logger.log_checkpoint(epoch, str(checkpoint_path), best=True)
            print(f"   ✨ New best model! Val Goal Acc: {best_val_acc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏸️  Early stopping (patience={args.patience})")
                break
    
    # ================================================================
    # STEP 7: FINAL SUMMARY
    # ================================================================
    elapsed_time = time.time() - start_time
    elapsed_hours = elapsed_time / 3600
    
    print("\n" + "=" * 100)
    print("✅ TRAINING COMPLETE!")
    print("=" * 100)
    print(f"   Best Val Goal Accuracy: {best_val_acc:.3f}")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Total Time: {elapsed_hours:.2f} hours")
    print(f"   Checkpoint saved: {checkpoint_dir / 'best_model.pt'}")
    print("=" * 100)
    
    # Log training summary
    logger.log_summary(best_epoch, best_val_acc, epoch + 1, elapsed_hours)
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Baseline LSTM Model")
    
    # Data paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/simulation_data/run_8',
        help='Directory containing trajectory data'
    )
    parser.add_argument(
        '--trajectory_filename',
        type=str,
        default='trajectories/all_trajectories.json',
        help='Trajectory file name (relative to data_dir)'
    )
    parser.add_argument(
        '--split_indices_path',
        type=str,
        default='data/simulation_data/run_8/split_data/split_indices_seed42.json',
        help='Path to JSON file with train/val/test split indices'
    )
    parser.add_argument(
        '--graph_path',
        type=str,
        default='data/processed/ucsd_walk_full.graphml',
        help='Path to graph file'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints/baseline_lstm',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--save_embedding_pipeline',
        action='store_true',
        help='Save embedding pipeline separately for transfer learning'
    )
    
    # Model - Embedding
    parser.add_argument('--node_embedding_dim', type=int, default=64, help='Node embedding dimension')
    parser.add_argument('--temporal_dim', type=int, default=64, help='Temporal encoding dimension')
    parser.add_argument('--agent_dim', type=int, default=64, help='Agent encoding dimension')
    parser.add_argument('--fusion_dim', type=int, default=128, help='Fusion layer dimension')
    parser.add_argument('--freeze_embedding', action='store_true', help='Freeze embedding pipeline')
    parser.add_argument('--pretrained_embedding', type=str, default=None, 
                        help='Path to pretrained unified embedding pipeline (e.g., checkpoints/keepers/unified_embedding.pt)')
    
    # Model - Prediction
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    
    args = parser.parse_args()
    main(args)
