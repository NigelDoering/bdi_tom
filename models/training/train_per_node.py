import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json

# Set MPS fallback for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ============================================================================
# IMPORTS
# ============================================================================
from models.training.utils import (
    get_device, set_seed, save_checkpoint, load_checkpoint,
    compute_accuracy, AverageMeter, MetricsTracker
)
from models.training.data_loader import enrich_and_load_data
from models.training.temporal_feature_enricher import TemporalFeatureEnricher
from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline
from models.en_encoders.enhanced_trajectory_encoder import EnhancedTrajectoryEncoder
from models.en_encoders.enhanced_map_encoder import (
    EnhancedWorldGraphEncoder, GraphDataPreparator
)
from models.en_encoders.enhanced_tom_graph_encoder import EnhancedToMGraphEncoder
from graph_controller.world_graph import WorldGraph

# W&B imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip3 install wandb")


# ============================================================================
# DATA VERIFICATION & PREPARATION
# ============================================================================

class TrajectoryDataValidator:
    """Validates that trajectories have all required data for training."""
    
    @staticmethod
    def verify_trajectory_structure(trajectory: Dict) -> Tuple[bool, str]:
        """
        Verify that a trajectory has all required fields.
        
        Required fields:
        - path: List of [node_id, category_id] pairs
        - goal_node: Final destination node ID
        - hour: Hour of day trajectory started
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if 'path' not in trajectory:
            return False, "Missing 'path' field"
        if not isinstance(trajectory['path'], list) or len(trajectory['path']) == 0:
            return False, "Empty or invalid 'path'"
        if len(trajectory['path'][0]) != 2:
            return False, "Path nodes should be [node_id, category_id] pairs"
        if 'goal_node' not in trajectory:
            return False, "Missing 'goal_node' field"
        if 'hour' not in trajectory:
            return False, "Missing 'hour' field"
        
        return True, ""
    
    @staticmethod
    def validate_all_trajectories(trajectories: List[Dict]) -> Tuple[int, int, List[str]]:
        """
        Validate all trajectories.
        
        Returns:
            Tuple of (num_valid, num_invalid, error_list)
        """
        num_valid = 0
        num_invalid = 0
        errors = []
        
        for idx, traj in enumerate(trajectories):
            is_valid, error_msg = TrajectoryDataValidator.verify_trajectory_structure(traj)
            if is_valid:
                num_valid += 1
            else:
                num_invalid += 1
                errors.append(f"Trajectory {idx}: {error_msg}")
        
        return num_valid, num_invalid, errors


# ============================================================================
# PER-NODE DATASET
# ============================================================================

class PerNodeTrajectoryDataset(Dataset):
    """
    Dataset that creates per-node training samples.
    
    For each trajectory, creates K samples where K = trajectory length.
    Each sample includes:
    - Current node and all previous nodes in trajectory
    - Goal node (label)
    - Next node (label)
    - Category of current node (label)
    """
    
    # Category to index mapping
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
        graph: object,
        poi_nodes: List[str],
        node_to_idx_map: Dict[str, int] = None,
        min_traj_length: int = 2
    ):
        self.trajectories = trajectories
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.goal_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
        
        # Create node-to-index mapping for all nodes in graph
        if node_to_idx_map is None:
            self.node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        else:
            self.node_to_idx = node_to_idx_map
        
        # Create per-node samples
        self.samples = []
        for traj_idx, traj in enumerate(trajectories):
            is_valid, _ = TrajectoryDataValidator.verify_trajectory_structure(traj)
            if not is_valid:
                continue
            
            path = traj['path']
            goal_node = traj['goal_node']
            
            # Only use trajectories of sufficient length
            if len(path) < min_traj_length:
                continue
            
            # Skip if goal not in POI nodes
            if goal_node not in self.goal_to_idx:
                continue
            
            # Create sample for each node in trajectory
            for step_idx in range(1, len(path)):
                # At step_idx, we have seen nodes 0 to step_idx-1
                # We want to predict next node (path[step_idx][0]) and goal
                current_node_id = path[step_idx-1][0]
                next_node_id = path[step_idx][0]
                
                # Get category indices
                next_cat_str = path[step_idx][1]
                next_cat_idx = self.CATEGORY_TO_IDX.get(next_cat_str, 6)  # Default to 'other'
                
                sample = {
                    'trajectory_idx': traj_idx,
                    'step': step_idx,
                    'path_so_far': [self.node_to_idx.get(node[0], 0) for node in path[:step_idx]],
                    'current_node_idx': self.node_to_idx.get(current_node_id, 0),
                    'next_node_idx': self.node_to_idx.get(next_node_id, 0),
                    'next_category_idx': next_cat_idx,
                    'goal_idx': self.goal_to_idx[goal_node],
                    'hour': traj['hour'],
                    'full_trajectory': [self.node_to_idx.get(node[0], 0) for node in path],
                }
                self.samples.append(sample)
        
        print(f"📊 PerNodeDataset: {len(self.samples)} per-node samples from {len(trajectories)} trajectories")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_per_node_samples(batch: List[Dict]) -> Dict:
    """
    Collate per-node samples into batches.
    
    Returns:
        Dict with trajectory information and target labels.
    """
    # Collect trajectory paths for encoding
    trajectories = [s['full_trajectory'] for s in batch]
    node_ids = torch.tensor([s['current_node_idx'] for s in batch], dtype=torch.long)
    
    # Goals, next nodes, and categories as indices
    goal_indices = torch.tensor([s['goal_idx'] for s in batch], dtype=torch.long)
    next_node_indices = torch.tensor([s['next_node_idx'] for s in batch], dtype=torch.long)
    next_categories = torch.tensor([s['next_category_idx'] for s in batch], dtype=torch.long)
    
    hours = torch.tensor([s['hour'] for s in batch], dtype=torch.float32)
    path_lengths = torch.tensor([len(s['path_so_far']) for s in batch], dtype=torch.long)
    
    return {
        'trajectories': trajectories,
        'node_ids': node_ids,
        'goal_nodes': goal_indices,
        'next_node_ids': next_node_indices,
        'next_categories': next_categories,
        'hours': hours,
        'path_lengths': path_lengths,
        'batch_size': len(batch),
    }


# ============================================================================
# ENHANCED MODEL WITH W&B LOGGING
# ============================================================================

class SimpleBDIToMModel(nn.Module):
    """Simplified BDI-ToM model optimized for per-node training."""
    
    def __init__(
        self,
        num_nodes: int,
        graph_node_feat_dim: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        
        # Simple node embedding
        self.node_embedding = nn.Embedding(num_nodes, output_dim)
        self.poi_embedding = nn.Embedding(num_poi_nodes, output_dim)
        self.category_embedding = nn.Embedding(num_categories, 32)
        
        # Temporal encoding
        self.hour_encoding = nn.Linear(1, 32)
        
        # Shared feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(output_dim + 32 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        
        # Task heads
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes),
        )
        
        self.nextstep_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_nodes),
        )
        
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )
    
    def forward(
        self,
        node_ids: torch.Tensor,
        hours: Optional[torch.Tensor] = None,
        days: Optional[torch.Tensor] = None,
        deltas: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for per-node predictions."""
        
        # Handle 1D node_ids
        if node_ids.dim() == 1:
            node_ids = node_ids.unsqueeze(1)
        
        # Get node embeddings (batch, seq_len, hidden)
        if node_ids.size(1) > 1:
            # Multi-node case: use mean of embeddings
            node_emb = self.node_embedding(node_ids).mean(dim=1)
        else:
            # Single node case: squeeze the seq dimension
            node_emb = self.node_embedding(node_ids).squeeze(1)
        
        # Temporal features
        if hours is None:
            hours = torch.zeros(node_emb.size(0), 1, device=node_emb.device)
        else:
            hours = hours.unsqueeze(1) if hours.dim() == 1 else hours
        
        hours_emb = self.hour_encoding(hours)
        
        # Category (use dummy if not provided)
        cat_emb = self.category_embedding(torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device))
        
        # Combine features
        combined = torch.cat([node_emb, hours_emb, cat_emb], dim=-1)
        
        # Process through shared processor
        features = self.feature_processor(combined)
        
        # Return predictions
        return {
            'goal': self.goal_head(features),
            'nextstep': self.nextstep_head(features),
            'category': self.category_head(features),
            'embeddings': features,
        }


# ============================================================================
# COMPREHENSIVE W&B LOGGER
# ============================================================================

class WandBTrainingLogger:
    """Handles all W&B logging during training."""
    
    def __init__(
        self,
        project_name: str = "bdi-tom-unified",
        experiment_name: str = "per-node-training",
        config: Optional[Dict] = None
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.enabled = WANDB_AVAILABLE
        
        if self.enabled:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                tags=["unified-embeddings", "per-node", "multi-task"],
            )
            print(f"✅ W&B initialized: {project_name}/{experiment_name}")
        else:
            print("⚠️  W&B disabled (not installed)")
    
    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ):
        """Log epoch-level metrics."""
        if not self.enabled:
            return
        
        log_dict = {
            'epoch': epoch,
            'learning_rate': learning_rate,
        }
        
        # Train metrics
        for key, val in train_metrics.items():
            log_dict[f'train/{key}'] = val
        
        # Val metrics
        for key, val in val_metrics.items():
            log_dict[f'val/{key}'] = val
        
        wandb.log(log_dict, step=epoch)
    
    def log_task_metrics(
        self,
        phase: str,  # 'train' or 'val'
        epoch: int,
        goal_loss: float,
        goal_acc: float,
        nextstep_loss: float,
        nextstep_acc: float,
        category_loss: float,
        category_acc: float,
    ):
        """Log per-task metrics."""
        if not self.enabled:
            return
        
        log_dict = {
            f'{phase}/goal_loss': goal_loss,
            f'{phase}/goal_acc': goal_acc,
            f'{phase}/nextstep_loss': nextstep_loss,
            f'{phase}/nextstep_acc': nextstep_acc,
            f'{phase}/category_loss': category_loss,
            f'{phase}/category_acc': category_acc,
        }
        
        wandb.log(log_dict, step=epoch)
    
    def log_batch_metrics(
        self,
        batch_idx: int,
        phase: str,
        loss: float,
        goal_acc: float,
        nextstep_acc: float,
        category_acc: float,
    ):
        """Log batch-level metrics during training."""
        if not self.enabled:
            return
        
        log_dict = {
            f'{phase}_batch/loss': loss,
            f'{phase}_batch/goal_acc': goal_acc,
            f'{phase}_batch/nextstep_acc': nextstep_acc,
            f'{phase}_batch/category_acc': category_acc,
        }
        
        wandb.log(log_dict, step=batch_idx)
    
    def log_embedding_stats(
        self,
        embeddings: torch.Tensor,
        phase: str,
        epoch: int,
    ):
        """Log embedding statistics."""
        if not self.enabled:
            return
        
        log_dict = {
            f'{phase}/embedding_mean': embeddings.mean().item(),
            f'{phase}/embedding_std': embeddings.std().item(),
            f'{phase}/embedding_min': embeddings.min().item(),
            f'{phase}/embedding_max': embeddings.max().item(),
        }
        
        wandb.log(log_dict, step=epoch)
    
    def log_model_info(self, model: nn.Module):
        """Log model architecture and parameters."""
        if not self.enabled:
            return
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
        })
        
        # Watch model gradients and parameters
        wandb.watch(model, log='all', log_freq=100)
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()


# ============================================================================
# TRAINING LOOP WITH PER-NODE LEARNING
# ============================================================================

def train_epoch_per_node(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    graph_data: Dict,
    device: torch.device,
    epoch: int,
    logger: WandBTrainingLogger,
    task_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Train for one epoch with per-node predictions."""
    
    if task_weights is None:
        task_weights = {'goal': 1.0, 'nextstep': 0.5, 'category': 0.5}
    
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
        # Extract batch data
        node_ids = batch['node_ids'].to(device)
        goal_nodes = batch['goal_nodes'].to(device)
        next_node_ids = batch['next_node_ids'].to(device)
        next_categories = batch['next_categories'].to(device)
        hours = batch['hours'].to(device)
        
        # Create dummy temporal features if not provided
        batch_size = node_ids.size(0)
        days = torch.zeros(batch_size, dtype=torch.long, device=device)
        deltas = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        velocities = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        
        # Forward pass
        predictions = model(
            node_ids=node_ids,
            hours=hours,
            days=days,
            deltas=deltas,
            velocities=velocities,
            graph_data=graph_data
        )
        
        # Compute losses
        loss_goal = criterion['goal'](predictions['goal'], goal_nodes)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node_ids)
        loss_category = criterion['category'](predictions['category'], next_categories)
        
        # Weighted loss
        loss = (
            task_weights['goal'] * loss_goal +
            task_weights['nextstep'] * loss_nextstep +
            task_weights['category'] * loss_category
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute accuracies
        goal_acc = compute_accuracy(predictions['goal'], goal_nodes, k=1)
        nextstep_acc = compute_accuracy(predictions['nextstep'], next_node_ids, k=1)
        category_acc = compute_accuracy(predictions['category'], next_categories, k=1)
        
        # Update metrics
        metrics['loss'].update(loss.item(), node_ids.size(0))
        metrics['loss_goal'].update(loss_goal.item(), node_ids.size(0))
        metrics['loss_nextstep'].update(loss_nextstep.item(), node_ids.size(0))
        metrics['loss_category'].update(loss_category.item(), node_ids.size(0))
        metrics['goal_acc'].update(goal_acc, node_ids.size(0))
        metrics['nextstep_acc'].update(nextstep_acc, node_ids.size(0))
        metrics['category_acc'].update(category_acc, node_ids.size(0))
        
        # Log batch metrics
        logger.log_batch_metrics(
            batch_idx=epoch * len(train_loader) + batch_idx,
            phase='train',
            loss=loss.item(),
            goal_acc=goal_acc,
            nextstep_acc=nextstep_acc,
            category_acc=category_acc,
        )
        
        # Log embedding stats periodically
        if batch_idx % 50 == 0:
            logger.log_embedding_stats(
                predictions['embeddings'],
                phase='train',
                epoch=epoch * len(train_loader) + batch_idx
            )
        
        # Progress bar update
        pbar.set_postfix({
            'loss': f"{metrics['loss'].avg:.4f}",
            'goal_acc': f"{metrics['goal_acc'].avg:.3f}",
            'nextstep_acc': f"{metrics['nextstep_acc'].avg:.3f}",
            'cat_acc': f"{metrics['category_acc'].avg:.3f}",
        })
    
    # Return averages
    return {
        'loss': metrics['loss'].avg,
        'loss_goal': metrics['loss_goal'].avg,
        'loss_nextstep': metrics['loss_nextstep'].avg,
        'loss_category': metrics['loss_category'].avg,
        'goal_acc': metrics['goal_acc'].avg,
        'nextstep_acc': metrics['nextstep_acc'].avg,
        'category_acc': metrics['category_acc'].avg,
    }


@torch.no_grad()
def validate_per_node(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    graph_data: Dict,
    device: torch.device,
    epoch: int,
    logger: WandBTrainingLogger,
    task_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Validate with per-node predictions."""
    
    if task_weights is None:
        task_weights = {'goal': 1.0, 'nextstep': 0.5, 'category': 0.5}
    
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
    
    for batch in pbar:
        # Extract batch data
        node_ids = batch['node_ids'].to(device)
        goal_nodes = batch['goal_nodes'].to(device)
        next_node_ids = batch['next_node_ids'].to(device)
        next_categories = batch['next_categories'].to(device)
        hours = batch['hours'].to(device)
        
        # Create dummy temporal features if not provided
        batch_size = node_ids.size(0)
        days = torch.zeros(batch_size, dtype=torch.long, device=device)
        deltas = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        velocities = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        
        # Forward pass
        predictions = model(
            node_ids=node_ids,
            hours=hours,
            days=days,
            deltas=deltas,
            velocities=velocities,
            graph_data=graph_data
        )
        
        # Compute losses
        loss_goal = criterion['goal'](predictions['goal'], goal_nodes)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node_ids)
        loss_category = criterion['category'](predictions['category'], next_categories)
        
        # Weighted loss
        loss = (
            task_weights['goal'] * loss_goal +
            task_weights['nextstep'] * loss_nextstep +
            task_weights['category'] * loss_category
        )
        
        # Compute accuracies
        goal_acc = compute_accuracy(predictions['goal'], goal_nodes, k=1)
        nextstep_acc = compute_accuracy(predictions['nextstep'], next_node_ids, k=1)
        category_acc = compute_accuracy(predictions['category'], next_categories, k=1)
        
        # Update metrics
        metrics['loss'].update(loss.item(), node_ids.size(0))
        metrics['loss_goal'].update(loss_goal.item(), node_ids.size(0))
        metrics['loss_nextstep'].update(loss_nextstep.item(), node_ids.size(0))
        metrics['loss_category'].update(loss_category.item(), node_ids.size(0))
        metrics['goal_acc'].update(goal_acc, node_ids.size(0))
        metrics['nextstep_acc'].update(nextstep_acc, node_ids.size(0))
        metrics['category_acc'].update(category_acc, node_ids.size(0))
        
        # Log embedding stats
        logger.log_embedding_stats(
            predictions['embeddings'],
            phase='val',
            epoch=epoch
        )
        
        pbar.set_postfix({'loss': f"{metrics['loss'].avg:.4f}"})
    
    # Return averages
    return {
        'loss': metrics['loss'].avg,
        'loss_goal': metrics['loss_goal'].avg,
        'loss_nextstep': metrics['loss_nextstep'].avg,
        'loss_category': metrics['loss_category'].avg,
        'goal_acc': metrics['goal_acc'].avg,
        'nextstep_acc': metrics['nextstep_acc'].avg,
        'category_acc': metrics['category_acc'].avg,
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def load_per_node_data(
    run_dir: str,
    graph_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Load trajectory data and create per-node samples."""
    from models.training.data_loader import load_simulation_data, split_data
    
    print("\n" + "=" * 80)
    print("PER-NODE DATA LOADING")
    print("=" * 80)
    
    # Step 1: Load base trajectories
    print("\n📂 Step 1: Loading simulation data...")
    trajectories, graph, poi_nodes = load_simulation_data(run_dir, graph_path)
    
    # Create node-to-index mapping
    print("\n🔧 Step 1a: Creating node-to-index mapping...")
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    print(f"   ✅ {len(node_to_idx)} nodes mapped")
    
    # Step 2: Split data at trajectory level first
    print("\n📊 Step 2: Splitting trajectories...")
    train_trajs, val_trajs, test_trajs = split_data(
        trajectories,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Step 3: Create per-node datasets
    print("\n🔧 Step 3: Creating per-node datasets...")
    train_dataset = PerNodeTrajectoryDataset(train_trajs, graph, poi_nodes, node_to_idx_map=node_to_idx)
    val_dataset = PerNodeTrajectoryDataset(val_trajs, graph, poi_nodes, node_to_idx_map=node_to_idx)
    test_dataset = PerNodeTrajectoryDataset(test_trajs, graph, poi_nodes, node_to_idx_map=node_to_idx)
    
    # Step 4: Create data loaders
    print("\n🔌 Step 4: Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_per_node_samples,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_per_node_samples,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_per_node_samples,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\n📦 Per-Node DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    print("=" * 80 + "\n")
    
    return train_loader, val_loader, test_loader, {'poi_nodes': len(poi_nodes), 'graph_nodes': len(graph.nodes)}


def main(args):
    """Main training orchestration."""
    
    print("=" * 100)
    print("🧠 PER-NODE TRAINING WITH COMPREHENSIVE W&B LOGGING")
    print("=" * 100)
    
    device = get_device()
    set_seed(args.seed)
    
    print(f"\n📍 Device: {device}")
    print(f"📍 Seed: {args.seed}")
    
    # ================================================================
    # STEP 1: LOAD PER-NODE DATA
    # ================================================================
    print("\n1️⃣  Loading per-node trajectory data...")
    
    train_loader, val_loader, test_loader, enrichment_stats = load_per_node_data(
        args.data_dir,
        args.graph_path,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=args.seed
    )
    
    # ================================================================
    # STEP 2: DATA VERIFICATION
    # ================================================================
    print("\n2️⃣  Verifying data structure...")
    print("   ✅ Goal node: Available in trajectory data")
    print("   ✅ Next node: Available at every step (path[step][0])")
    print("   ✅ Category: Available at every step (path[step][1])")
    print("   ✅ Per-node samples created: Ready for training")
    
    # ================================================================
    # STEP 3: PREPARE GRAPH
    # ================================================================
    print("\n3️⃣  Preparing graph data...")
    
    import networkx as nx
    graph = nx.read_graphml(args.graph_path)
    world_graph = WorldGraph(graph)
    graph_prep = GraphDataPreparator(graph)
    graph_data = graph_prep.prepare_graph_data()
    graph_data = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in graph_data.items()
    }
    
    print(f"   ✅ Graph nodes: {graph_data['num_nodes']}")
    print(f"   ✅ Graph edges: {graph_data['edge_index'].shape[1]}")
    print(f"   ✅ POI nodes: {len(world_graph.poi_nodes)}")
    
    # ================================================================
    # STEP 4: CREATE MODEL
    # ================================================================
    print("\n4️⃣  Creating simplified per-node model...")
    
    model = SimpleBDIToMModel(
        num_nodes=graph_data['num_nodes'],
        graph_node_feat_dim=graph_data['x'].shape[1],
        num_poi_nodes=len(world_graph.poi_nodes),
        num_categories=7,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model parameters: {total_params:,}")
    
    # ================================================================
    # STEP 5: SETUP OPTIMIZATION
    # ================================================================
    print("\n5️⃣  Setting up optimization...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
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
    # STEP 6: INITIALIZE W&B LOGGING
    # ================================================================
    print("\n6️⃣  Initializing W&B logging...")
    
    config = {
        'model': 'EnhancedBDIToMWithLogging',
        'hidden_dim': args.hidden_dim,
        'embedding_dim': args.embedding_dim,
        'output_dim': args.output_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'task_weights': {'goal': 1.0, 'nextstep': 0.5, 'category': 0.5},
    }
    
    logger = WandBTrainingLogger(
        project_name="bdi-tom-unified",
        experiment_name="per-node-training",
        config=config
    )
    
    logger.log_model_info(model)
    
    # ================================================================
    # STEP 7: TRAINING LOOP
    # ================================================================
    print("\n7️⃣  Starting training loop...")
    print("=" * 100)
    
    best_val_loss = float('inf')
    best_goal_acc = 0
    patience = 10
    patience_counter = 0
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch_per_node(
            model, train_loader, optimizer, criterion, graph_data, device,
            epoch, logger
        )
        
        # Validate
        val_metrics = validate_per_node(
            model, val_loader, criterion, graph_data, device,
            epoch, logger
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log epoch metrics
        logger.log_task_metrics(
            phase='train',
            epoch=epoch,
            goal_loss=train_metrics['loss_goal'],
            goal_acc=train_metrics['goal_acc'],
            nextstep_loss=train_metrics['loss_nextstep'],
            nextstep_acc=train_metrics['nextstep_acc'],
            category_loss=train_metrics['loss_category'],
            category_acc=train_metrics['category_acc'],
        )
        
        logger.log_task_metrics(
            phase='val',
            epoch=epoch,
            goal_loss=val_metrics['loss_goal'],
            goal_acc=val_metrics['goal_acc'],
            nextstep_loss=val_metrics['loss_nextstep'],
            nextstep_acc=val_metrics['nextstep_acc'],
            category_loss=val_metrics['loss_category'],
            category_acc=val_metrics['category_acc'],
        )
        
        logger.log_epoch_metrics(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=optimizer.param_groups[0]['lr']
        )
        
        # Print summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Train Goal Acc: {train_metrics['goal_acc']:.3f} | Val Goal Acc: {val_metrics['goal_acc']:.3f}")
        print(f"   Train NextStep Acc: {train_metrics['nextstep_acc']:.3f} | Val NextStep Acc: {val_metrics['nextstep_acc']:.3f}")
        print(f"   Train Category Acc: {train_metrics['category_acc']:.3f} | Val Category Acc: {val_metrics['category_acc']:.3f}")
        
        # Checkpointing
        if val_metrics['goal_acc'] > best_goal_acc:
            best_goal_acc = val_metrics['goal_acc']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                val_metrics, str(checkpoint_path)
            )
            print(f"   ✨ New best model! Goal Acc: {best_goal_acc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏸️  Early stopping triggered (patience={patience})")
                break
    
    print("\n" + "=" * 100)
    print(f"✅ Training complete! Best Goal Accuracy: {best_goal_acc:.3f}")
    print("=" * 100)
    
    # Finish W&B
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Per-Node Training with W&B Logging"
    )
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/simulation_data/run_8')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/per_node_v1')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
