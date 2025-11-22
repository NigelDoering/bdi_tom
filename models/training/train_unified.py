import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Set MPS fallback for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ============================================================================
# IMPORTS: Utilities
# ============================================================================
from models.training.utils import (
    get_device, set_seed, save_checkpoint, load_checkpoint,
    compute_accuracy, AverageMeter, MetricsTracker
)

# ============================================================================
# IMPORTS: Data
# ============================================================================
from models.training.data_loader import enrich_and_load_data
from models.training.temporal_feature_enricher import TemporalFeatureEnricher

# ============================================================================
# IMPORTS: NEW ENHANCED ENCODERS
# ============================================================================
from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline
from models.en_encoders.enhanced_trajectory_encoder import EnhancedTrajectoryEncoder
from models.en_encoders.enhanced_map_encoder import (
    EnhancedWorldGraphEncoder, GraphDataPreparator
)
from models.en_encoders.enhanced_tom_graph_encoder import EnhancedToMGraphEncoder

# ============================================================================
# IMPORTS: Graph
# ============================================================================
from graph_controller.world_graph import WorldGraph

# ============================================================================
# IMPORTS: W&B (Optional)
# ============================================================================
try:
    from models.training.wandb_config import (
        init_wandb, log_metrics, save_model_artifact, watch_model
    )
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# MULTI-TASK MODEL WITH UNIFIED EMBEDDINGS
# ============================================================================

class BDIToMMultiTaskModelUnified(nn.Module):
    """
    Ultimate multi-task model leveraging unified embeddings.
    
    This model combines all advanced encoders:
    - Unified embeddings from node2vec + temporal + agent + fusion
    - Enhanced trajectory processing
    - Advanced graph encoding
    - Master fusion orchestration
    
    Three prediction heads:
    1. Goal Prediction (230 POIs)
    2. Next-Step Prediction (all nodes)
    3. Category Prediction (7 categories)
    """
    
    def __init__(
        self,
        num_nodes: int,
        graph_node_feat_dim: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        output_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize unified multi-task model."""
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        
        # ================================================================
        # MASTER ENCODER: Enhanced ToM Graph Encoder
        # ================================================================
        self.encoder = EnhancedToMGraphEncoder(
            num_nodes=num_nodes,
            num_agents=100,
            num_categories=num_categories,
            graph_node_feat_dim=graph_node_feat_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        # ================================================================
        # TASK 1: GOAL PREDICTION HEAD
        # ================================================================
        self.goal_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes),
        )
        
        # ================================================================
        # TASK 2: NEXT-STEP PREDICTION HEAD
        # ================================================================
        self.nextstep_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_nodes),
        )
        
        # ================================================================
        # TASK 3: CATEGORY PREDICTION HEAD
        # ================================================================
        self.category_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )
    
    def forward(
        self,
        node_ids: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        hours: Optional[torch.Tensor] = None,
        days: Optional[torch.Tensor] = None,
        deltas: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through unified multi-task model.
        
        Returns:
            Dict with 'goal', 'nextstep', 'category' logits
        """
        
        # Get unified representation
        unified_repr = self.encoder(
            node_ids, agent_ids, hours, days, deltas, velocities,
            mask=mask, graph_data=graph_data
        )
        
        # Task-specific predictions
        return {
            'goal': self.goal_head(unified_repr),
            'nextstep': self.nextstep_head(unified_repr),
            'category': self.category_head(unified_repr),
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    graph_data: Dict,
    device: torch.device,
    epoch: int,
    task_weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    
    if task_weights is None:
        task_weights = {'goal': 1.0, 'nextstep': 0.5, 'category': 0.5}
    
    model.train()
    metrics = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        node_ids = batch['node_ids'].to(device)
        agent_ids = batch.get('agent_ids').to(device) if 'agent_ids' in batch else None
        hours = batch.get('hours').to(device) if 'hours' in batch else None
        days = batch.get('days').to(device) if 'days' in batch else None
        deltas = batch.get('deltas').to(device) if 'deltas' in batch else None
        velocities = batch.get('velocities').to(device) if 'velocities' in batch else None
        mask = batch.get('mask').to(device) if 'mask' in batch else None
        goal = batch['goal'].to(device)
        next_node = batch['next_node'].to(device)
        category = batch['category'].to(device)
        
        # Forward pass
        predictions = model(
            node_ids, agent_ids, hours, days, deltas, velocities,
            mask=mask, graph_data=graph_data
        )
        
        # Compute losses
        loss_goal = criterion['goal'](predictions['goal'], goal)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node)
        loss_category = criterion['category'](predictions['category'], category)
        
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
        goal_acc = compute_accuracy(predictions['goal'], goal)
        nextstep_acc = compute_accuracy(predictions['nextstep'], next_node)
        category_acc = compute_accuracy(predictions['category'], category)
        
        # Update metrics
        metrics.update({
            'loss': loss.item(),
            'loss_goal': loss_goal.item(),
            'loss_nextstep': loss_nextstep.item(),
            'loss_category': loss_category.item(),
            'goal_acc': goal_acc,
            'nextstep_acc': nextstep_acc,
            'category_acc': category_acc,
        })
        
        # Progress bar
        pbar.set_postfix({
            'loss': f"{metrics.avg('loss'):.4f}",
            'goal_acc': f"{metrics.avg('goal_acc'):.3f}",
            'cat_acc': f"{metrics.avg('category_acc'):.3f}",
        })
    
    return metrics.get_averages()


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    graph_data: Dict,
    device: torch.device,
    task_weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """Validate model."""
    
    if task_weights is None:
        task_weights = {'goal': 1.0, 'nextstep': 0.5, 'category': 0.5}
    
    model.eval()
    metrics = AverageMeter()
    
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch in pbar:
        node_ids = batch['node_ids'].to(device)
        agent_ids = batch.get('agent_ids').to(device) if 'agent_ids' in batch else None
        hours = batch.get('hours').to(device) if 'hours' in batch else None
        days = batch.get('days').to(device) if 'days' in batch else None
        deltas = batch.get('deltas').to(device) if 'deltas' in batch else None
        velocities = batch.get('velocities').to(device) if 'velocities' in batch else None
        mask = batch.get('mask').to(device) if 'mask' in batch else None
        goal = batch['goal'].to(device)
        next_node = batch['next_node'].to(device)
        category = batch['category'].to(device)
        
        # Forward pass
        predictions = model(
            node_ids, agent_ids, hours, days, deltas, velocities,
            mask=mask, graph_data=graph_data
        )
        
        # Compute losses and accuracies
        loss_goal = criterion['goal'](predictions['goal'], goal)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node)
        loss_category = criterion['category'](predictions['category'], category)
        
        loss = (
            task_weights['goal'] * loss_goal +
            task_weights['nextstep'] * loss_nextstep +
            task_weights['category'] * loss_category
        )
        
        goal_acc = compute_accuracy(predictions['goal'], goal)
        nextstep_acc = compute_accuracy(predictions['nextstep'], next_node)
        category_acc = compute_accuracy(predictions['category'], category)
        
        metrics.update({
            'loss': loss.item(),
            'goal_acc': goal_acc,
            'nextstep_acc': nextstep_acc,
            'category_acc': category_acc,
        })
        
        pbar.set_postfix({'loss': f"{metrics.avg('loss'):.4f}"})
    
    return metrics.get_averages()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    """Main training orchestration."""
    
    # Setup
    print("=" * 100)
    print("🧠 ULTIMATE BDI-THEORY OF MIND TRAINING WITH UNIFIED EMBEDDINGS")
    print("=" * 100)
    
    device = get_device()
    set_seed(args.seed)
    
    print(f"\n📍 Device: {device}")
    print(f"📍 Seed: {args.seed}")
    
    # ================================================================
    # STEP 1: LOAD AND ENRICH DATA
    # ================================================================
    print("\n1️⃣  Loading and enriching data...")
    
    train_loader, val_loader, test_loader, enrichment_stats = enrich_and_load_data(
        args.data_dir,
        args.graph_path,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=args.seed
    )
    
    print(f"   ✅ Train batches: {len(train_loader)}")
    print(f"   ✅ Val batches: {len(val_loader)}")
    print(f"   ✅ Test batches: {len(test_loader)}")
    
    # ================================================================
    # STEP 2: PREPARE GRAPH DATA
    # ================================================================
    print("\n2️⃣  Preparing graph data...")
    
    import networkx as nx
    graph = nx.read_graphml(args.graph_path)
    world_graph = WorldGraph(graph)
    
    graph_prep = GraphDataPreparator(graph)
    graph_data = graph_prep.prepare_graph_data()
    graph_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in graph_data.items()}
    
    print(f"   ✅ Graph nodes: {graph_data['num_nodes']}")
    print(f"   ✅ Graph edges: {graph_data['edge_index'].shape[1]}")
    
    # ================================================================
    # STEP 3: CREATE MODEL
    # ================================================================
    print("\n3️⃣  Creating unified multi-task model...")
    
    model = BDIToMMultiTaskModelUnified(
        num_nodes=graph_data['num_nodes'],
        graph_node_feat_dim=graph_data['x'].shape[1],
        num_poi_nodes=len(world_graph.poi_nodes),
        num_categories=7,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        output_dim=args.output_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model parameters: {total_params:,}")
    
    # ================================================================
    # STEP 4: SETUP OPTIMIZATION
    # ================================================================
    print("\n4️⃣  Setting up optimization...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    criterion = {
        'goal': nn.CrossEntropyLoss(),
        'nextstep': nn.CrossEntropyLoss(),
        'category': nn.CrossEntropyLoss(),
    }
    
    print(f"   ✅ Optimizer: AdamW (lr={args.learning_rate})")
    print(f"   ✅ Scheduler: CosineAnnealingLR")
    
    # ================================================================
    # STEP 5: TRAINING LOOP
    # ================================================================
    print("\n5️⃣  Starting training loop...")
    print(f"   📊 Epochs: {args.num_epochs}")
    print(f"   📊 Task weights: goal=1.0, nextstep=0.5, category=0.5")
    
    best_goal_acc = 0.0
    checkpoint_dir = Path(args.checkpoint_dir) / "bdi_tom_unified"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            graph_data, device, epoch + 1,
            task_weights={'goal': 1.0, 'nextstep': 0.5, 'category': 0.5}
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, graph_data, device,
            task_weights={'goal': 1.0, 'nextstep': 0.5, 'category': 0.5}
        )
        
        # Save checkpoint
        goal_acc = val_metrics['goal_acc']
        if goal_acc > best_goal_acc:
            best_goal_acc = goal_acc
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model, optimizer, epoch, val_metrics
            )
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}")
        print(f"   Train Goal Acc: {train_metrics['goal_acc']:.3f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val Goal Acc: {val_metrics['goal_acc']:.3f} {'✨ NEW BEST' if goal_acc == best_goal_acc else ''}")
    
    # ================================================================
    # STEP 6: EVALUATE ON TEST SET
    # ================================================================
    print("\n6️⃣  Evaluating on test set...")
    
    # Load best model
    best_checkpoint = load_checkpoint(checkpoint_dir / "best_model.pt", model, device)
    
    test_metrics = validate(
        model, test_loader, criterion, graph_data, device,
        task_weights={'goal': 1.0, 'nextstep': 0.5, 'category': 0.5}
    )
    
    print(f"\n🎯 Test Results:")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   Goal Accuracy: {test_metrics['goal_acc']:.3f} ⭐")
    print(f"   Next-Step Accuracy: {test_metrics['nextstep_acc']:.3f}")
    print(f"   Category Accuracy: {test_metrics['category_acc']:.3f}")
    
    print("\n" + "=" * 100)
    print("✨ TRAINING COMPLETE!")
    print("=" * 100)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Ultimate BDI-ToM Training with Unified Embeddings"
    )

    # Data
    parser.add_argument('--data_dir', type=str, default='data/simulation_data/run_8')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
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
