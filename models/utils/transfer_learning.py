"""
TRANSFER LEARNING WITH FROZEN UNIFIED EMBEDDING PIPELINE

ARCHITECTURE:
1. Load pre-trained UnifiedEmbeddingPipeline (frozen)
2. Modular prediction heads (easily swappable)
3. Support for different architectures (Transformer, MLP, etc.)

USAGE:
    python transfer_learning.py \
        --pretrained_checkpoint checkpoints/per_node_v2/best_model.pt \
        --architecture transformer \
        --freeze_pretrained True
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ============================================================================
# IMPORTS
# ============================================================================
from models.utils.utils import (
    get_device, set_seed, save_checkpoint, load_checkpoint,
    compute_accuracy, AverageMeter
)
from models.utils.data_loader import load_simulation_data, split_data
from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline
from models.utils.train_per_node import PerNodeTrajectoryDataset, collate_per_node_samples

# W&B imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


# ============================================================================
# MODULAR PREDICTION HEADS
# ============================================================================

class TransformerPredictionHead(nn.Module):
    """
    Transformer-based prediction head for sequential modeling.
    
    Uses multi-head self-attention to capture dependencies in trajectory history.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_poi_nodes: int,
        num_nodes: int,
        num_categories: int = 7,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Positional encoding for sequence
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, embedding_dim))  # Max 512 seq len
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection and prediction heads
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Goal head
        self.goal_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes),
        )
        
        # Next step head
        self.nextstep_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_nodes),
        )
        
        # Category head
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [batch, seq_len, embedding_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [batch, seq_len, embedding_dim]
            mask: [batch, seq_len] padding mask
        
        Returns:
            Dict with 'goal', 'nextstep', 'category' predictions
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device
        
        # Add positional encoding
        pos_enc = self.pos_encoder[:, :seq_len, :].to(device)
        x = embeddings + pos_enc
        
        # Create attention mask (True for positions to mask)
        if mask is not None:
            # Convert length mask to attention mask
            attn_mask = ~mask.bool()  # Invert: True where we should attend
        else:
            attn_mask = None
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        
        # Use last valid token (before padding)
        if mask is not None:
            last_embeddings = []
            for b in range(batch_size):
                last_idx = mask[b].sum().long() - 1
                last_idx = torch.clamp(last_idx, 0, seq_len - 1)
                last_embeddings.append(x[b, last_idx])
            x = torch.stack(last_embeddings)
        else:
            x = x[:, -1, :]  # Use last token
        
        # Normalize
        x = self.norm(x)
        
        # Prediction heads
        return {
            'goal': self.goal_head(x),
            'nextstep': self.nextstep_head(x),
            'category': self.category_head(x),
        }


class MLPPredictionHead(nn.Module):
    """
    Simple MLP-based prediction head.
    
    Baseline architecture for comparison with Transformer.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_poi_nodes: int,
        num_nodes: int,
        num_categories: int = 7,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Sequence aggregation (mean pooling)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Main MLP
        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        
        # Goal head
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes),
        )
        
        # Next step head
        self.nextstep_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_nodes),
        )
        
        # Category head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [batch, seq_len, embedding_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [batch, seq_len, embedding_dim]
            mask: [batch, seq_len] padding mask (optional)
        
        Returns:
            Dict with 'goal', 'nextstep', 'category' predictions
        """
        # Mean pooling over sequence
        # [batch, seq_len, embedding_dim] -> [batch, embedding_dim]
        x = embeddings.mean(dim=1)
        
        # MLP processing
        x = self.main(x)
        
        # Prediction heads
        return {
            'goal': self.goal_head(x),
            'nextstep': self.nextstep_head(x),
            'category': self.category_head(x),
        }


class LSTMPredictionHead(nn.Module):
    """
    LSTM-based prediction head.
    
    Captures sequential dependencies with recurrent processing.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_poi_nodes: int,
        num_nodes: int,
        num_categories: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output processing
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Goal head
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_poi_nodes),
        )
        
        # Next step head
        self.nextstep_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_nodes),
        )
        
        # Category head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_categories),
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [batch, seq_len, embedding_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [batch, seq_len, embedding_dim]
            mask: [batch, seq_len] sequence lengths
        
        Returns:
            Dict with 'goal', 'nextstep', 'category' predictions
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Pack padded sequences if mask provided
        if mask is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings,
                mask.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
            x = hidden[-1]  # Use last layer hidden state
        else:
            _, (hidden, _) = self.lstm(embeddings)
            x = hidden[-1]
        
        x = self.norm(x)
        
        # Prediction heads
        return {
            'goal': self.goal_head(x),
            'nextstep': self.nextstep_head(x),
            'category': self.category_head(x),
        }


# ============================================================================
# TRANSFER LEARNING MODEL
# ============================================================================

class TransferLearningModel(nn.Module):
    """
    Transfer learning model with frozen pretrained embeddings.
    
    Loads UnifiedEmbeddingPipeline and trains new prediction head.
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_agents: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        # Pretrained pipeline params
        node_embedding_dim: int = 64,
        temporal_dim: int = 64,
        agent_dim: int = 64,
        fusion_dim: int = 128,
        # Head architecture
        architecture: str = 'transformer',
        head_hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
        freeze_embedding: bool = True,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.fusion_dim = fusion_dim
        self.architecture = architecture
        
        # ================================================================
        # LOAD PRETRAINED EMBEDDING PIPELINE
        # ================================================================
        self.embedding_pipeline = UnifiedEmbeddingPipeline(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_categories=num_categories,
            node_embedding_dim=node_embedding_dim,
            temporal_dim=temporal_dim,
            agent_dim=agent_dim,
            fusion_dim=fusion_dim,
            hidden_dim=256,  # Default hidden dim
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
            print("❄️  Embedding pipeline frozen!")
        
        # ================================================================
        # MODULAR PREDICTION HEAD
        # ================================================================
        self.prediction_head = self._create_head(
            architecture=architecture,
            embedding_dim=fusion_dim,
            num_poi_nodes=num_poi_nodes,
            num_nodes=num_nodes,
            num_categories=num_categories,
            hidden_dim=head_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        print(f"🏗️  Prediction head: {architecture}")
    
    def _create_head(
        self,
        architecture: str,
        embedding_dim: int,
        num_poi_nodes: int,
        num_nodes: int,
        num_categories: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> nn.Module:
        """Create modular prediction head based on architecture."""
        if architecture.lower() == 'transformer':
            return TransformerPredictionHead(
                embedding_dim=embedding_dim,
                num_poi_nodes=num_poi_nodes,
                num_nodes=num_nodes,
                num_categories=num_categories,
                num_heads=num_heads,
                num_layers=2,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif architecture.lower() == 'mlp':
            return MLPPredictionHead(
                embedding_dim=embedding_dim,
                num_poi_nodes=num_poi_nodes,
                num_nodes=num_nodes,
                num_categories=num_categories,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif architecture.lower() == 'lstm':
            return LSTMPredictionHead(
                embedding_dim=embedding_dim,
                num_poi_nodes=num_poi_nodes,
                num_nodes=num_nodes,
                num_categories=num_categories,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _freeze_embeddings(self):
        """Freeze embedding pipeline parameters."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = False
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding pipeline for fine-tuning."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = True
        print("🔥 Embedding pipeline unfrozen!")
    
    def forward(
        self,
        history_node_indices: torch.Tensor,  # [batch, seq_len]
        history_lengths: torch.Tensor,        # [batch]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: frozen embeddings → modular head → predictions.
        
        Args:
            history_node_indices: [batch, seq_len] node indices
            history_lengths: [batch] actual sequence length before padding
        
        Returns:
            Dict with 'goal', 'nextstep', 'category' predictions
        """
        batch_size, seq_len = history_node_indices.shape
        device = history_node_indices.device
        
        # ================================================================
        # STEP 1: GET EMBEDDINGS FROM FROZEN PIPELINE
        # ================================================================
        node_emb = self.embedding_pipeline.encode_nodes(
            history_node_indices,
            spatial_coords=None,
            categories=None,
        )  # [batch, seq_len, node_embedding_dim]
        
        # Expand to fusion_dim by padding
        node_dim = node_emb.shape[-1]
        if node_dim < self.fusion_dim:
            padding = torch.zeros(batch_size, seq_len, self.fusion_dim - node_dim, device=device)
            history_embeddings = torch.cat([node_emb, padding], dim=-1)
        else:
            history_embeddings = node_emb[:, :, :self.fusion_dim]
        
        # ================================================================
        # STEP 2: PASS THROUGH MODULAR PREDICTION HEAD
        # ================================================================
        predictions = self.prediction_head(history_embeddings, mask=history_lengths)
        
        return predictions
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained model weights."""
        print(f"📦 Loading pretrained model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load embedding pipeline weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter only embedding pipeline weights
        embedding_state = {
            k.replace('embedding_pipeline.', ''): v
            for k, v in state_dict.items()
            if k.startswith('embedding_pipeline.')
        }
        
        if embedding_state:
            self.embedding_pipeline.load_state_dict(embedding_state, strict=False)
            print("✅ Pretrained embedding pipeline loaded!")
        else:
            print("⚠️  No embedding pipeline weights found in checkpoint")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    epoch: int,
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
    
    for batch in pbar:
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        next_cat_idx = batch['next_cat_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        
        # Forward pass
        predictions = model(history_node_indices, history_lengths)
        
        # Compute losses
        loss_goal = criterion['goal'](predictions['goal'], goal_idx)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node_idx)
        loss_category = criterion['category'](predictions['category'], next_cat_idx)
        
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
        category_acc = compute_accuracy(predictions['category'], next_cat_idx)
        
        # Update metrics
        metrics['loss'].update(loss.item(), batch_size)
        metrics['loss_goal'].update(loss_goal.item(), batch_size)
        metrics['loss_nextstep'].update(loss_nextstep.item(), batch_size)
        metrics['loss_category'].update(loss_category.item(), batch_size)
        metrics['goal_acc'].update(goal_acc, batch_size)
        metrics['nextstep_acc'].update(nextstep_acc, batch_size)
        metrics['category_acc'].update(category_acc, batch_size)
        
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
    
    for batch in pbar:
        batch_size = batch['history_node_indices'].size(0)
        
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        next_node_idx = batch['next_node_idx'].to(device)
        next_cat_idx = batch['next_cat_idx'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        
        # Forward pass
        predictions = model(history_node_indices, history_lengths)
        
        # Compute losses
        loss_goal = criterion['goal'](predictions['goal'], goal_idx)
        loss_nextstep = criterion['nextstep'](predictions['nextstep'], next_node_idx)
        loss_category = criterion['category'](predictions['category'], next_cat_idx)
        
        # Weighted loss
        loss = 1.0 * loss_goal + 0.5 * loss_nextstep + 0.5 * loss_category
        
        # Compute accuracies
        goal_acc = compute_accuracy(predictions['goal'], goal_idx)
        nextstep_acc = compute_accuracy(predictions['nextstep'], next_node_idx)
        category_acc = compute_accuracy(predictions['category'], next_cat_idx)
        
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
    """Main transfer learning training."""
    
    print("=" * 100)
    print("🚀 TRANSFER LEARNING WITH FROZEN PRETRAINED EMBEDDINGS")
    print(f"   Architecture: {args.architecture.upper()}")
    print("=" * 100)
    
    device = get_device()
    set_seed(args.seed)
    
    print(f"\n📍 Device: {device}")
    print(f"📍 Seed: {args.seed}")
    print(f"📍 Architecture: {args.architecture}")
    print(f"📍 Freeze Pretrained: {args.freeze_pretrained}")
    
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
    # STEP 2: CREATE MODEL WITH PRETRAINED EMBEDDINGS
    # ================================================================
    print(f"\n2️⃣  Creating transfer learning model ({args.architecture})...")
    
    model = TransferLearningModel(
        num_nodes=len(graph.nodes()),
        num_agents=1,
        num_poi_nodes=len(poi_nodes),
        num_categories=7,
        node_embedding_dim=args.node_embedding_dim,
        temporal_dim=args.temporal_dim,
        agent_dim=args.agent_dim,
        fusion_dim=args.fusion_dim,
        architecture=args.architecture,
        head_hidden_dim=args.head_hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_embedding=args.freeze_pretrained,
    ).to(device)
    
    # Load pretrained weights
    if args.pretrained_checkpoint:
        model.load_pretrained(args.pretrained_checkpoint)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"   ✅ Total params: {total_params:,}")
    print(f"   ✅ Trainable params: {trainable_params:,}")
    print(f"   ✅ Frozen params: {frozen_params:,}")
    
    # ================================================================
    # STEP 3: SETUP OPTIMIZATION
    # ================================================================
    print("\n3️⃣  Setting up optimization...")
    
    # Only optimize trainable parameters
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
    print(f"\n4️⃣  Starting transfer learning training ({args.architecture})...")
    print("=" * 100)
    
    best_val_acc = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Epoch {epoch+1}/{args.num_epochs}")
        print(f"   LR: {lr:.2e}")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Train Goal Acc: {train_metrics['goal_acc']:.3f} | Val Goal Acc: {val_metrics['goal_acc']:.3f}")
        
        # Save checkpoint
        if val_metrics['goal_acc'] > best_val_acc:
            best_val_acc = val_metrics['goal_acc']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / f"best_{args.architecture}_model.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], val_metrics, str(checkpoint_path))
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
    print(f"✅ Transfer learning complete!")
    print(f"   Architecture: {args.architecture}")
    print(f"   Best Goal Accuracy: {best_val_acc:.3f}")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Total Time: {elapsed_hours:.2f} hours")
    print("=" * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer Learning with Frozen Pretrained Embeddings")
    
    # Pretrained model
    parser.add_argument(
        '--pretrained_checkpoint',
        type=str,
        default='checkpoints/per_node_v2/best_model.pt',
        help='Path to pretrained model checkpoint'
    )
    parser.add_argument(
        '--freeze_pretrained',
        type=bool,
        default=True,
        help='Freeze pretrained embedding pipeline'
    )
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/simulation_data/run_8_enriched')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/transfer_learning')
    
    # Architecture
    parser.add_argument(
        '--architecture',
        type=str,
        default='transformer',
        choices=['transformer', 'mlp', 'lstm'],
        help='Prediction head architecture'
    )
    
    # Model - Embedding (pretrained)
    parser.add_argument('--node_embedding_dim', type=int, default=64)
    parser.add_argument('--temporal_dim', type=int, default=64)
    parser.add_argument('--agent_dim', type=int, default=64)
    parser.add_argument('--fusion_dim', type=int, default=128)
    
    # Model - Head
    parser.add_argument('--head_hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
