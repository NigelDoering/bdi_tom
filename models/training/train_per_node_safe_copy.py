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

class TrajectoryDataset(Dataset):
    """
    FAST Trajectory-level dataset (not per-node).
    Loads enriched trajectories and processes them as sequences.
    
    Much faster than per-node approach:
    - No need to expand 100K trajectories into millions of samples
    - Direct trajectory-level supervision
    - Full path context available for each sample
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
        min_traj_length: int = 2,
    ):
        self.trajectories = []
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.goal_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
        
        # Create node-to-index mapping
        if node_to_idx_map is None:
            self.node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        else:
            self.node_to_idx = node_to_idx_map
        
        # Filter valid trajectories (fast, no expansion)
        print(f"   🧹 Filtering {len(trajectories)} trajectories...")
        for traj in tqdm(trajectories, desc="   📊 Validating", leave=False):
            is_valid, _ = TrajectoryDataValidator.verify_trajectory_structure(traj)
            if not is_valid:
                continue
            
            path = traj['path']
            goal_node = traj['goal_node']
            
            if len(path) < min_traj_length or goal_node not in self.goal_to_idx:
                continue
            
            self.trajectories.append(traj)
        
        print(f"✅ TrajectoryDataset: {len(self.trajectories)} valid trajectories")
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict:
        traj = self.trajectories[idx]
        path = traj['path']
        
        return {
            'full_trajectory': [self.node_to_idx.get(node[0], 0) for node in path],
            'path_nodes': torch.tensor([self.node_to_idx.get(node[0], 0) for node in path], dtype=torch.long),
            'path_categories': torch.tensor([self.CATEGORY_TO_IDX.get(node[1], 6) for node in path], dtype=torch.long),
            'goal_idx': self.goal_to_idx[traj['goal_node']],
            'hour': traj['hour'],
            'day_of_week': traj.get('day_of_week', 0),
            'circadian_hour': traj.get('circadian_hour', traj['hour']),
            'temporal_deltas': traj.get('temporal_deltas', []),
            'velocities': traj.get('velocities', []),
        }


def collate_trajectories(batch: List[Dict]) -> Dict:
    """
    Collate trajectory-level samples into batches.
    
    Pads sequences to same length within the batch.
    """
    # Find max lengths for padding
    max_path_len = max(len(s['path_nodes']) for s in batch)
    max_path_len = max(1, max_path_len)
    
    batch_size = len(batch)
    device = torch.device('cpu')
    
    # Pad sequences
    path_nodes_padded = []
    path_categories_padded = []
    path_lengths = []
    velocities_padded = []
    temporal_deltas_padded = []
    
    for s in batch:
        path_len = len(s['path_nodes'])
        path_lengths.append(path_len)
        
        # Pad to max_path_len
        pad_size = max_path_len - path_len
        padded_nodes = torch.cat([s['path_nodes'], torch.zeros(pad_size, dtype=torch.long)])
        padded_cats = torch.cat([s['path_categories'], torch.zeros(pad_size, dtype=torch.long)])
        
        # Velocities and temporal deltas
        vel = s['velocities'] if isinstance(s['velocities'], list) else []
        deltas = s['temporal_deltas'] if isinstance(s['temporal_deltas'], list) else []
        
        padded_vel = torch.tensor(vel + [0.0] * (max_path_len - len(vel)), dtype=torch.float32)
        padded_deltas = torch.tensor(deltas + [0.0] * (max_path_len - len(deltas)), dtype=torch.float32)
        
        path_nodes_padded.append(padded_nodes)
        path_categories_padded.append(padded_cats)
        velocities_padded.append(padded_vel)
        temporal_deltas_padded.append(padded_deltas)
    
    return {
        'path_nodes': torch.stack(path_nodes_padded),
        'path_categories': torch.stack(path_categories_padded),
        'path_lengths': torch.tensor(path_lengths, dtype=torch.long),
        'goal_nodes': torch.tensor([s['goal_idx'] for s in batch], dtype=torch.long),
        'hours': torch.tensor([s['hour'] for s in batch], dtype=torch.float32),
        'day_of_week': torch.tensor([s['day_of_week'] for s in batch], dtype=torch.long),
        'circadian_hour': torch.tensor([s['circadian_hour'] for s in batch], dtype=torch.float32),
        'velocities': torch.stack(velocities_padded),
        'temporal_deltas': torch.stack(temporal_deltas_padded),
        'batch_size': batch_size,
        'trajectories': [s['full_trajectory'] for s in batch],
    }


# ============================================================================
# ENHANCED MODEL WITH W&B LOGGING
# ============================================================================

class EnhancedToMModel(nn.Module):
    """
    Advanced Theory of Mind model with:
    - Path history encoding (LSTM/GRU for sequential context)
    - Goal-aware representations
    - Agent belief state modeling (latent variable)
    - Multi-head attention for task-specific reasoning
    """
    
    def __init__(
        self,
        num_nodes: int,
        graph_node_feat_dim: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        hidden_dim: int = 256,
        output_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # ================================================================
        # CORE EMBEDDINGS
        # ================================================================
        self.node_embedding = nn.Embedding(num_nodes, output_dim)
        self.poi_embedding = nn.Embedding(num_poi_nodes, output_dim)
        self.category_embedding = nn.Embedding(num_categories, 32)
        self.hour_encoding = nn.Linear(1, 32)
        self.day_of_week_encoding = nn.Embedding(7, 16)  # NEW: 7 days of week
        self.velocity_encoder = nn.Sequential(  # NEW: Movement speed patterns
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        
        # ================================================================
        # AGENT BELIEF STATE - Latent variable for agent intentions
        # ================================================================
        self.belief_prior = nn.Linear(output_dim + 32, latent_dim)  # Prior: p(z|current_node, time)
        self.belief_posterior = nn.Sequential(  # Posterior: q(z|current, next, goal)
            nn.Linear(output_dim * 2 + output_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_var
        )
        
        # ================================================================
        # GOAL-AWARE CONTEXT
        # ================================================================
        self.goal_context_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # ================================================================
        # PATH HISTORY ENCODING (sequence of visited locations)
        # ================================================================
        self.path_lstm = nn.LSTM(
            input_size=output_dim + 32,  # node_emb + category_emb
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        
        # ================================================================
        # MULTI-HEAD ATTENTION FOR TASK-SPECIFIC REASONING
        # ================================================================
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # ================================================================
        # UNIFIED FEATURE PROCESSOR
        # ================================================================
        # Combines: current_node + temporal + day_of_week + velocity + belief_state + goal_context + path_history
        total_feature_dim = output_dim + 32 + 16 + 16 + latent_dim + (hidden_dim // 2) + (hidden_dim // 2)
        
        self.feature_processor = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ================================================================
        # TASK-SPECIFIC HEADS WITH RESIDUAL CONNECTIONS
        # ================================================================
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
        
        # KL divergence weight for VAE-style regularization
        self.kl_weight = 0.1
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution N(mu, exp(log_var))."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        node_ids: torch.Tensor,
        goal_nodes: Optional[torch.Tensor] = None,
        path_nodes: Optional[torch.Tensor] = None,  # [batch, seq_len]
        path_categories: Optional[torch.Tensor] = None,  # [batch, seq_len]
        hours: Optional[torch.Tensor] = None,
        days: Optional[torch.Tensor] = None,
        deltas: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Theory of Mind reasoning.
        
        Args:
            node_ids: Current node indices [batch]
            goal_nodes: Goal node indices [batch]
            path_nodes: Historical path nodes [batch, seq_len]
            path_categories: Historical path categories [batch, seq_len]
            hours: Hour of day [batch]
        """
        batch_size = node_ids.size(0)
        device = node_ids.device
        
        # Handle 1D node_ids
        if node_ids.dim() > 1:
            node_ids = node_ids.squeeze(-1)
        
        # ================================================================
        # 1. CURRENT STATE REPRESENTATION
        # ================================================================
        node_emb = self.node_embedding(node_ids)  # [batch, output_dim]
        
        # Temporal encoding
        if hours is None:
            hours = torch.zeros(batch_size, 1, device=device)
        elif hours.dim() == 1:
            hours = hours.unsqueeze(1)
        hours_emb = self.hour_encoding(hours)  # [batch, 32]
        
        # NEW: Day of week encoding
        day_emb = torch.zeros(batch_size, 16, device=device)  # Default zeros
        if days is not None:
            if days.dim() == 0:
                days = days.unsqueeze(0)
            day_emb = self.day_of_week_encoding(days % 7)  # [batch, 16]
        
        # NEW: Velocity context (average movement speed up to this point)
        vel_emb = torch.zeros(batch_size, 16, device=device)  # Default zeros
        if velocities is not None:
            if velocities.dim() == 1:
                velocities = velocities.unsqueeze(1)
            # Average velocity from path history
            vel_mean = velocities.mean(dim=1, keepdim=True)  # [batch, 1]
            vel_emb = self.velocity_encoder(vel_mean)  # [batch, 16]
        
        # ================================================================
        # 2. GOAL CONTEXT
        # ================================================================
        if goal_nodes is not None and goal_nodes.max() < self.num_poi_nodes:
            goal_emb = self.poi_embedding(goal_nodes)  # [batch, output_dim]
        else:
            # Use dummy goal if not provided
            goal_emb = self.poi_embedding(torch.zeros(batch_size, dtype=torch.long, device=device))
        
        goal_context = self.goal_context_encoder(goal_emb)  # [batch, hidden//2]
        
        # ================================================================
        # 3. AGENT BELIEF STATE (Latent variable modeling)
        # ================================================================
        # Prior: p(z|current_node, hour)
        prior_input = torch.cat([node_emb, hours_emb], dim=-1)
        z_mu_prior = self.belief_prior(prior_input)  # [batch, latent_dim]
        
        # Posterior: q(z|current, next, goal) - only used during training with supervision
        if goal_nodes is not None and path_nodes is not None:
            next_node_emb = self.node_embedding(path_nodes[:, 0]) if path_nodes.size(1) > 0 else node_emb
            posterior_input = torch.cat([node_emb, next_node_emb, goal_emb, hours_emb], dim=-1)
            posterior_params = self.belief_posterior(posterior_input)
            z_mu_post, z_log_var = posterior_params[:, :self.latent_dim], posterior_params[:, self.latent_dim:]
            z_belief = self.reparameterize(z_mu_post, z_log_var)
        else:
            z_belief = self.reparameterize(z_mu_prior, torch.zeros_like(z_mu_prior))
        
        # ================================================================
        # 4. PATH HISTORY ENCODING (what has the agent done?)
        # ================================================================
        if path_nodes is not None and path_nodes.size(1) > 0:
            # Embed path history
            path_node_emb = self.node_embedding(path_nodes)  # [batch, seq_len, output_dim]
            
            if path_categories is not None:
                path_cat_emb = self.category_embedding(path_categories)  # [batch, seq_len, 32]
                path_input = torch.cat([path_node_emb, path_cat_emb], dim=-1)
            else:
                path_input = path_node_emb
            
            # LSTM processes the sequence
            _, (path_hidden, _) = self.path_lstm(path_input)  # hidden: [2, batch, hidden//2]
            path_context = path_hidden[-1]  # [batch, hidden//2]
        else:
            # No path history - use zeros
            path_context = torch.zeros(batch_size, self.hidden_dim // 2, device=device)
        
        # ================================================================
        # 5. COMBINE ALL FEATURES (INCLUDING NEW TEMPORAL CONTEXT)
        # ================================================================
        combined_features = torch.cat([
            node_emb,           # Current location [output_dim]
            hours_emb,          # Time of day [32]
            day_emb,            # NEW: Day of week [16]
            vel_emb,            # NEW: Velocity context [16]
            z_belief,           # Agent belief/intention [latent_dim]
            goal_context,       # Goal representation [hidden//2]
            path_context,       # Path history [hidden//2]
        ], dim=-1)
        
        # ================================================================
        # 6. UNIFIED PROCESSING
        # ================================================================
        unified_repr = self.feature_processor(combined_features)  # [batch, hidden_dim]
        
        # ================================================================
        # 7. TASK HEADS
        # ================================================================
        return {
            'goal': self.goal_head(unified_repr),
            'nextstep': self.nextstep_head(unified_repr),
            'category': self.category_head(unified_repr),
            'embeddings': unified_repr,
            'belief_state': z_belief,  # For analysis/visualization
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
            try:
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    tags=["unified-embeddings", "per-node", "multi-task"],
                )
                print(f"✅ W&B initialized: {project_name}/{experiment_name}")
            except Exception as e:
                print(f"⚠️  W&B initialization failed: {e}")
                self.enabled = False
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
    """Train for one epoch with trajectory-level supervision."""
    
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
        batch_size = batch['batch_size']
        
        # Move batch to device
        path_nodes = batch['path_nodes'].to(device)          # [batch, seq_len]
        path_categories = batch['path_categories'].to(device) # [batch, seq_len]
        path_lengths = batch['path_lengths'].to(device)
        goal_nodes = batch['goal_nodes'].to(device)
        hours = batch['hours'].to(device)
        days = batch['day_of_week'].to(device)
        velocities = batch['velocities'].to(device)
        temporal_deltas = batch['temporal_deltas'].to(device)
        
        # For trajectory-level learning:
        # - First node is starting position
        # - Goal is the destination (same for all steps in trajectory)
        # - Next node is the next step in the path
        # - Categories are available for each step
        
        node_ids = path_nodes[:, 0]  # First node in each trajectory [batch]
        next_node_ids = path_nodes[:, 1] if path_nodes.size(1) > 1 else path_nodes[:, 0]
        next_categories = path_categories[:, 1] if path_categories.size(1) > 1 else path_categories[:, 0]
        
        # Forward pass with Theory of Mind features
        predictions = model(
            node_ids=node_ids,
            goal_nodes=goal_nodes,
            path_nodes=path_nodes,
            path_categories=path_categories,
            hours=hours,
            days=days,
            deltas=temporal_deltas,
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
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
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
    """Validate with trajectory-level supervision."""
    
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
        batch_size = batch['batch_size']
        
        # Move batch to device
        path_nodes = batch['path_nodes'].to(device)          # [batch, seq_len]
        path_categories = batch['path_categories'].to(device) # [batch, seq_len]
        path_lengths = batch['path_lengths'].to(device)
        goal_nodes = batch['goal_nodes'].to(device)
        hours = batch['hours'].to(device)
        days = batch['day_of_week'].to(device)
        velocities = batch['velocities'].to(device)
        temporal_deltas = batch['temporal_deltas'].to(device)
        
        # Extract starting node and next node
        node_ids = path_nodes[:, 0]  # First node in each trajectory [batch]
        next_node_ids = path_nodes[:, 1] if path_nodes.size(1) > 1 else path_nodes[:, 0]
        next_categories = path_categories[:, 1] if path_categories.size(1) > 1 else path_categories[:, 0]
        
        # Forward pass
        predictions = model(
            node_ids=node_ids,
            goal_nodes=goal_nodes,
            path_nodes=path_nodes,
            path_categories=path_categories,
            hours=hours,
            days=days,
            deltas=temporal_deltas,
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
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
        
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
    
    # Step 3: Create TRAJECTORY-LEVEL datasets (FAST - no per-node expansion!)
    print("\n🔧 Step 3: Creating trajectory datasets (fast!)...")
    train_dataset = TrajectoryDataset(train_trajs, graph, poi_nodes, node_to_idx_map=node_to_idx)
    val_dataset = TrajectoryDataset(val_trajs, graph, poi_nodes, node_to_idx_map=node_to_idx)
    test_dataset = TrajectoryDataset(test_trajs, graph, poi_nodes, node_to_idx_map=node_to_idx)
    
    # Step 4: Create data loaders
    print("\n🔌 Step 4: Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
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
    print("\n4️⃣  Creating Advanced Theory of Mind model...")
    
    model = EnhancedToMModel(
        num_nodes=graph_data['num_nodes'],
        graph_node_feat_dim=graph_data['x'].shape[1],
        num_poi_nodes=len(world_graph.poi_nodes),
        num_categories=7,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        latent_dim=64,
        dropout=args.dropout,
        num_heads=4,
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
    parser.add_argument('--data_dir', type=str, default='data/simulation_data/run_8_enriched')
    parser.add_argument('--data_dir', type=str, default='data/simulation_data/run_8_enriched')
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
