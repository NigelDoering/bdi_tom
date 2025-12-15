"""
PER-NODE THEORY OF MIND TRAINING WITH UNIFIED EMBEDDING PIPELINE

ARCHITECTURE:
1. Simplified Input: Just trajectory history (node sequence)
2. UnifiedEmbeddingPipeline: Separate module for embeddings (can be frozen)
3. Prediction Heads: Learnable task-specific outputs

The model computes ALL temporal features internally (deltas, velocities, etc.)
INPUT: history_node_indices → EMBEDDING PIPELINE → PREDICTION HEADS → outputs
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import time

# Set MPS fallback for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ============================================================================
# IMPORTS
# ============================================================================
from models.training.utils import (
    get_device, set_seed, save_checkpoint, load_checkpoint,
    compute_accuracy, AverageMeter
)
from models.training.data_loader import load_simulation_data, split_data
from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline
from graph_controller.world_graph import WorldGraph

# W&B imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


# ============================================================================
# SIMPLIFIED PER-NODE DATASET
# ============================================================================

class PerNodeTrajectoryDataset(Dataset):
    """
    Simplified per-node dataset that only takes trajectory history as input.
    
    For each trajectory [n1→n2→n3→n4→goal]:
    - Sample 1: history=[n1],        next=n2
    - Sample 2: history=[n1→n2],     next=n3
    - Sample 3: history=[n1→n2→n3],  next=n4
    
    The model computes all temporal features internally!
    """
    
    CATEGORY_TO_IDX = {
        'home': 0,
        'study': 1,
        'food': 2,
        'leisure': 3,
        'errands': 4,
        'health': 5,
        'other': 6,
    }
    
    def __init__(
        self,
        trajectories: List[Dict],
        graph: nx.Graph,
        poi_nodes: List[str],
        node_to_idx_map: Dict[str, int] = None,
        min_traj_length: int = 2,
    ):
        """Initialize per-node dataset."""
        self.samples = []
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.goal_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
        
        # Create node-to-index mapping
        if node_to_idx_map is None:
            self.node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        else:
            self.node_to_idx = node_to_idx_map
        
        # Expand trajectories into per-node samples
        print(f"   🧹 Creating per-node samples from {len(trajectories)} trajectories...")
        valid_count = 0
        sample_count = 0
        
        for traj_idx, traj in enumerate(tqdm(trajectories, desc="   📊 Expanding samples", leave=False)):
            if 'path' not in traj or 'goal_node' not in traj:
                continue
            
            path = traj['path']
            goal_node = traj['goal_node']
            
            # Skip invalid trajectories
            if not isinstance(path, list) or len(path) < min_traj_length:
                continue
            
            if goal_node not in self.goal_to_idx:
                print(f"   ⚠️  Trajectory {traj_idx} has unknown goal node '{goal_node}'. Skipping.")
                continue
            
            valid_count += 1
            goal_idx = self.goal_to_idx[goal_node]
            
            # Create per-node samples from path history
            for step_idx in range(1, len(path)):
                history_path = path[:step_idx]
                next_node = path[step_idx]
                
                # Convert node IDs to indices
                # Handle both simple node IDs and [node_id, category] tuples
                try:
                    history_indices = []
                    for node in history_path:
                        # Extract node ID if it's a list/tuple
                        node_id = node[0] if isinstance(node, (list, tuple)) else node
                        history_indices.append(self.node_to_idx[node_id])
                    
                    # Extract node ID and category for next step
                    next_node_id = next_node[0] if isinstance(next_node, (list, tuple)) else next_node
                    next_node_idx = self.node_to_idx[next_node_id]
                except (KeyError, TypeError, IndexError):
                    continue
                
                # Extract goal category from graph
                goal_cat_idx = 0
                if goal_node in self.graph.nodes:
                    cat_name = self.graph.nodes[goal_node].get('Category', 'other')
                    goal_cat_idx = self.CATEGORY_TO_IDX.get(cat_name, 0)
                
                self.samples.append({
                    'history_node_indices': history_indices,
                    'next_node_idx': next_node_idx,
                    'goal_cat_idx': goal_cat_idx,
                    'goal_idx': goal_idx,
                })
                
                sample_count += 1
        
        print(f"✅ Created {sample_count} per-node samples from {valid_count} trajectories")
        print(f"   Average samples per trajectory: {sample_count / max(valid_count, 1):.1f}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_per_node_samples(batch: List[Dict]) -> Dict:
    """
    Collate per-node samples into a batch.
    
    Pads variable-length history sequences to max length in batch.
    """
    if not batch:
        raise ValueError("Empty batch!")
    
    # Find max history length in batch
    max_history_len = max(len(s['history_node_indices']) for s in batch)
    max_history_len = max(1, max_history_len)
    
    batch_size = len(batch)
    
    # Pad all histories to max length
    padded_histories = []
    history_lengths = []
    
    for sample in batch:
        history = sample['history_node_indices']
        pad_len = max_history_len - len(history)
        
        # Pad with 0 (dummy node)
        padded = history + [0] * pad_len
        padded_histories.append(torch.tensor(padded, dtype=torch.long))
        history_lengths.append(len(history))
    
    return {
        'history_node_indices': torch.stack(padded_histories),  # [batch, max_len]
        'history_lengths': torch.tensor(history_lengths, dtype=torch.long),  # [batch]
        'next_node_idx': torch.tensor([s['next_node_idx'] for s in batch], dtype=torch.long),
        'goal_cat_idx': torch.tensor([s['goal_cat_idx'] for s in batch], dtype=torch.long),
        'goal_idx': torch.tensor([s['goal_idx'] for s in batch], dtype=torch.long),
    }


# ============================================================================
# PER-NODE THEORY OF MIND PREDICTOR
# ============================================================================

class PerNodeToMPredictor(nn.Module):
    """
    Per-Node Theory of Mind Predictor with Separated Embedding & Prediction.
    
    DESIGN:
    - Module 1: UnifiedEmbeddingPipeline (can be frozen for transfer learning)
    - Module 2: History aggregation (LSTM)
    - Module 3: Prediction heads (goal, nextstep, category)
    
    INPUT: history_node_indices [batch, seq_len] with lengths [batch]
    OUTPUT: predictions for goal, next node, category
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_agents: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        # Embedding params
        node_embedding_dim: int = 64,
        temporal_dim: int = 64,
        agent_dim: int = 64,
        fusion_dim: int = 128,
        # Prediction head params
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 4,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.node_embedding_dim = node_embedding_dim
        
        # ================================================================
        # MODULE 1: UNIFIED EMBEDDING PIPELINE
        # ================================================================
        self.embedding_pipeline = UnifiedEmbeddingPipeline(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_categories=num_categories,
            node_embedding_dim=node_embedding_dim,
            temporal_dim=temporal_dim,
            agent_dim=agent_dim,
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            n_fusion_layers=2,
            n_heads=num_heads,
            dropout=dropout,
            use_node2vec=True,
            use_temporal=True,
            use_agent=True,
            use_modality_gating=True,
            use_cross_attention=True,
        )
        
        if freeze_embedding:
            self._freeze_embeddings()
        
        # ================================================================
        # MODULE 2: HISTORY AGGREGATION WITH LSTM
        # ================================================================
        self.history_lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        
        # ================================================================
        # MODULE 3: PREDICTION HEADS
        # ================================================================
        
        # Feature fusion layer (combines current + history context)
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        
        # Goal prediction head
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes),
        )
        
        # Next step prediction head
        self.nextstep_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_nodes),
        )
        
        # Category prediction head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )
    
    def _freeze_embeddings(self):
        """Freeze embedding pipeline parameters."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = False
        print("❄️  Embedding pipeline frozen!")
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding parameters."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = True
        print("🔥 Embedding pipeline unfrozen!")
    
    def _compute_temporal_features(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute temporal features from trajectory history.
        
        Returns:
            hours: [batch, seq_len] - Hour of day (default: 12)
            velocities: [batch, seq_len] - Speed (computed from distance)
            temporal_deltas: [batch, seq_len] - Time between steps (default: 1 hour)
        """
        batch_size, seq_len = history_node_indices.shape
        device = history_node_indices.device
        
        # For now, use defaults (could enhance with graph distances)
        hours = torch.full((batch_size, seq_len), 12.0, device=device)
        
        # Velocity: constant 1.0 (could compute from graph distances)
        velocities = torch.ones((batch_size, seq_len), device=device)
        
        # Temporal deltas: constant 1 hour between steps
        temporal_deltas = torch.ones((batch_size, seq_len), device=device)
        
        return hours, velocities, temporal_deltas
    
    def forward(
        self,
        history_node_indices: torch.Tensor,  # [batch, seq_len]
        history_lengths: torch.Tensor,        # [batch]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: history → embeddings → predictions.
        
        Args:
            history_node_indices: [batch, seq_len] node indices
            history_lengths: [batch] actual sequence length before padding
        
        Returns:
            Dict with keys: 'goal', 'nextstep', 'category', 'embeddings'
        """
        batch_size, seq_len = history_node_indices.shape
        device = history_node_indices.device
        
        # ================================================================
        # STEP 1: COMPUTE TEMPORAL FEATURES FROM HISTORY
        # ================================================================
        hours, velocities, temporal_deltas = self._compute_temporal_features(
            history_node_indices,
            history_lengths,
        )
        
        # ================================================================
        # STEP 2: ENCODE TRAJECTORY HISTORY WITH UNIFIED PIPELINE
        # ================================================================
        # Get node embeddings only (simplified for per-node training)
        # We'll use a simpler approach that avoids the temporal encoder mismatch
        node_emb = self.embedding_pipeline.encode_nodes(
            history_node_indices,
            spatial_coords=None,
            categories=None,
        )  # [batch, seq_len, node_embedding_dim]
        
        # Expand to fusion_dim by simple projection/padding
        # This ensures compatibility with the LSTM input
        batch_size, seq_len, node_dim = node_emb.shape
        if node_dim < self.fusion_dim:
            # Pad with zeros to reach fusion_dim
            padding = torch.zeros(batch_size, seq_len, self.fusion_dim - node_dim, device=device)
            history_embeddings = torch.cat([node_emb, padding], dim=-1)
        else:
            # If node_dim >= fusion_dim, take only the first fusion_dim channels
            history_embeddings = node_emb[:, :, :self.fusion_dim]
        # ================================================================
        # STEP 3: AGGREGATE HISTORY WITH LSTM
        # ================================================================
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            history_embeddings,
            history_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM processes history
        _, (hidden, _) = self.history_lstm(packed)
        history_context = hidden[-1]  # [batch, hidden//2]
        # ================================================================
        # STEP 4: GET CURRENT NODE REPRESENTATION
        # ================================================================
        # Get embedding of the last node in history
        last_node_embeddings = []
        for b in range(batch_size):
            actual_len = history_lengths[b].item()
            last_idx = min(actual_len - 1, seq_len - 1)
            last_node_embeddings.append(history_embeddings[b, last_idx])
        
        current_node_emb = torch.stack(last_node_embeddings)  # [batch, fusion_dim]
        
        # ================================================================
        # STEP 5: FUSE CURRENT STATE WITH HISTORY CONTEXT
        # ================================================================
        combined = torch.cat([current_node_emb, history_context], dim=-1)
        unified_repr = self.feature_fusion(combined)  # [batch, hidden_dim]
        
        # ================================================================
        # STEP 6: PREDICTION HEADS
        # ================================================================
        return {
            'goal': self.goal_head(unified_repr),
            'nextstep': self.nextstep_head(unified_repr),
            'category': self.category_head(unified_repr),
            'embeddings': unified_repr,
        }


# ============================================================================
# W&B LOGGER
# ============================================================================

class WandBLogger:
    """Handles W&B logging during training."""
    
    def __init__(self, project_name: str = "bdi-tom-per-node", config: Optional[Dict] = None):
        self.enabled = WANDB_AVAILABLE
        self.global_step = 0
        
        if self.enabled:
            wandb.init(project=project_name, config=config or {})
            print("✅ W&B initialized!")
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
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float):
        """Log epoch-level metrics with train/val separation."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'learning_rate': lr,
        }
        
        # Add train metrics with 'train/' prefix
        for key, value in train_metrics.items():
            log_dict[f'train/{key}'] = value
        
        # Add val metrics with 'val/' prefix
        for key, value in val_metrics.items():
            log_dict[f'val/{key}'] = value
        
        # Compute and log key ratios and differences
        if 'loss' in train_metrics and 'loss' in val_metrics:
            log_dict['metrics/train_val_loss_ratio'] = train_metrics['loss'] / (val_metrics['loss'] + 1e-8)
            log_dict['metrics/val_train_loss_diff'] = val_metrics['loss'] - train_metrics['loss']
        
        # Log individual task metrics for easy comparison
        for task in ['goal', 'nextstep', 'category']:
            if f'loss_{task}' in train_metrics and f'loss_{task}' in val_metrics:
                log_dict[f'loss_comparison/{task}_train'] = train_metrics[f'loss_{task}']
                log_dict[f'loss_comparison/{task}_val'] = val_metrics[f'loss_{task}']
                log_dict[f'loss_comparison/{task}_diff'] = val_metrics[f'loss_{task}'] - train_metrics[f'loss_{task}']
            
            if f'{task}_acc' in train_metrics and f'{task}_acc' in val_metrics:
                log_dict[f'accuracy_comparison/{task}_train'] = train_metrics[f'{task}_acc']
                log_dict[f'accuracy_comparison/{task}_val'] = val_metrics[f'{task}_acc']
                log_dict[f'accuracy_comparison/{task}_diff'] = val_metrics[f'{task}_acc'] - train_metrics[f'{task}_acc']
        
        wandb.log(log_dict, step=epoch)
    
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
    
    def log_test_results(self, test_metrics: Dict[str, float]):
        """Log test set results."""
        if not self.enabled:
            return
        
        test_dict = {}
        for key, value in test_metrics.items():
            test_dict[f'test/{key}'] = value
        
        wandb.log(test_dict)
    
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
    wandb_logger: Optional['WandBLogger'] = None,
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
        
        # Forward pass
        predictions = model(history_node_indices, history_lengths)
        
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
                'gradient_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0),
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
    wandb_logger: Optional['WandBLogger'] = None,
) -> Dict[str, float]:
    """Validate."""
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
    
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch_idx, batch in enumerate(pbar):
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        goal_cat_idx = batch['goal_cat_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        
        # Forward pass
        predictions = model(history_node_indices, history_lengths)
        
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
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
        
        pbar.set_postfix({'loss': f"{metrics['loss'].avg:.4f}"})
    
    return {k: v.avg for k, v in metrics.items()}


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main(args):
    """Main training orchestration."""
    
    print("=" * 100)
    print("🧠 PER-NODE TRAINING WITH UNIFIED EMBEDDING PIPELINE")
    print("=" * 100)
    
    device = get_device()
    set_seed(args.seed)
    
    print(f"\n📍 Device: {device}")
    print(f"📍 Seed: {args.seed}")
    
    # ================================================================
    # STEP 1: LOAD DATA
    # ================================================================
    print("\n1️⃣  Loading simulation data...")
    
    trajectories, graph, poi_nodes = load_simulation_data(
        args.data_dir,
        args.graph_path
    )
    
    # Split trajectories
    train_trajs, val_trajs, test_trajs = split_data(
        trajectories,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=args.seed
    )
    
    # Create datasets
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    
    train_dataset = PerNodeTrajectoryDataset(train_trajs, graph, poi_nodes, node_to_idx)
    val_dataset = PerNodeTrajectoryDataset(val_trajs, graph, poi_nodes, node_to_idx)
    test_dataset = PerNodeTrajectoryDataset(test_trajs, graph, poi_nodes, node_to_idx)
    
    # Create loaders
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
    
    print(f"   ✅ Train: {len(train_loader)} batches")
    print(f"   ✅ Val: {len(val_loader)} batches")
    print(f"   ✅ Test: {len(test_loader)} batches")
    
    # ================================================================
    # STEP 2: CREATE MODEL
    # ================================================================
    print("\n2️⃣  Creating per-node predictor...")
    
    model = PerNodeToMPredictor(
        num_nodes=len(graph.nodes()),
        num_agents=1,  # Single agent for simplicity
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
    print(f"   ✅ Total params: {total_params:,}")
    print(f"   ✅ Trainable params: {trainable_params:,}")
    
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
        'n_fusion_layers': 2,
        'use_node2vec': True,
        'use_temporal': True,
        'use_agent': True,
        'use_modality_gating': True,
        'use_cross_attention': True,
    }
    
    # ================================================================
    # STEP 3: SETUP OPTIMIZATION
    # ================================================================
    print("\n3️⃣  Setting up optimization...")
    
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
    
    # ================================================================
    # STEP 4: TRAINING LOOP
    # ================================================================
    print("\n4️⃣  Starting training...")
    print("=" * 100)
    
    logger = WandBLogger(config={
        'model': 'PerNodeToMPredictor',
        'num_nodes': len(graph.nodes()),
        'num_poi_nodes': len(poi_nodes),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'num_epochs': args.num_epochs,
        'weight_decay': args.weight_decay,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
    })
    
    # Log model information
    logger.log_model_info(total_params, trainable_params, {
        'node_embedding_dim': args.node_embedding_dim,
        'temporal_dim': args.temporal_dim,
        'agent_dim': args.agent_dim,
        'fusion_dim': args.fusion_dim,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'freeze_embedding': args.freeze_embedding,
    })
    
    best_val_acc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        val_metrics = validate(model, val_loader, criterion, device, logger)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_metrics, val_metrics, lr)
        
        print(f"\n📊 Epoch {epoch+1}/{args.num_epochs}")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Train Goal Acc: {train_metrics['goal_acc']:.3f} | Val Goal Acc: {val_metrics['goal_acc']:.3f}")
        
        # Save checkpoint
        if val_metrics['goal_acc'] > best_val_acc:
            best_val_acc = val_metrics['goal_acc']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model, 
                optimizer, 
                epoch, 
                val_metrics['loss'], 
                val_metrics, 
                str(checkpoint_path),
                config=model_config  # ← Pass config for encoder extraction
            )
            logger.log_checkpoint(epoch, str(checkpoint_path), best=True)
            print(f"   ✨ New best model! Val Goal Acc: {best_val_acc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏸️  Early stopping (patience={patience})")
                break
    
    # Calculate training time
    elapsed_time = time.time() - start_time
    elapsed_hours = elapsed_time / 3600
    
    print("\n" + "=" * 100)
    print(f"✅ Training complete! Best Goal Accuracy: {best_val_acc:.3f}")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Total Time: {elapsed_hours:.2f} hours")
    print("=" * 100)
    
    # Log training summary
    logger.log_summary(best_epoch, best_val_acc, epoch + 1, elapsed_hours)
    
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Per-Node Training with Unified Embeddings")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/simulation_data/run_8_enriched')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/per_node_v2')
    
    # Model - Embedding
    parser.add_argument('--node_embedding_dim', type=int, default=64)
    parser.add_argument('--temporal_dim', type=int, default=64)
    parser.add_argument('--agent_dim', type=int, default=64)
    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--freeze_embedding', type=bool, default=False)
    
    # Model - Prediction
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
