"""
BASELINE TRANSFORMER MODEL FOR PER-NODE BDI-ToM PREDICTION

This module contains the PerNodeTransformerPredictor model, a transformer-based
architecture for Theory of Mind prediction in spatial trajectory tasks.

ARCHITECTURE:
- Module 1: UnifiedEmbeddingPipeline (node embeddings with Node2Vec)
- Module 2: Transformer encoder with causal masking
- Module 3: Multi-task prediction heads (goal, next-step, category)

KEY ADVANTAGE OVER LSTM:
- Processes all trajectory positions in parallel (10-30x faster training)
- Uses causal masking to ensure each position only sees past context
- No need to expand trajectories into per-node samples

The model takes full trajectories as input and predicts at all positions simultaneously:
1. Goal node (which POI the agent is heading to)
2. Next step node (immediate next location)
3. Goal category (semantic category of the destination)
"""

import os
from typing import Dict, Tuple

# Set MPS fallback for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn

from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline


class PerNodeTransformerPredictor(nn.Module):
    """
    Per-Node Theory of Mind Predictor with Transformer Architecture.
    
    DESIGN:
    - Module 1: UnifiedEmbeddingPipeline (can be frozen for transfer learning)
    - Module 2: Transformer encoder with causal masking
    - Module 3: Prediction heads (goal, nextstep, category)
    
    INPUT: node_indices [batch, seq_len]
    OUTPUT: predictions at ALL sequence positions simultaneously
    
    CAUSAL MASKING:
    - Position i can only attend to positions 0...i (not future positions)
    - This mimics per-node training where prediction at step i only uses history up to i
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
        # Transformer params
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        self.d_model = d_model
        self.fusion_dim = fusion_dim
        
        # ================================================================
        # MODULE 1: UNIFIED EMBEDDING PIPELINE
        # ================================================================
        # Match train_per_node configuration with all modalities enabled
        self.embedding_pipeline = UnifiedEmbeddingPipeline(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_categories=num_categories,
            node_embedding_dim=node_embedding_dim,
            temporal_dim=temporal_dim,
            agent_dim=agent_dim,
            fusion_dim=fusion_dim,
            hidden_dim=d_model,
            n_fusion_layers=2,
            n_heads=nhead,
            dropout=dropout,
            use_node2vec=True,
            use_temporal=False,
            use_agent=True,
            use_modality_gating=True,
            use_cross_attention=True,
        )
        
        if freeze_embedding:
            self._freeze_embeddings()
        
        # ================================================================
        # MODULE 2: PROJECTION TO d_model (if needed)
        # ================================================================
        # Project from fusion_dim to d_model if they differ
        if fusion_dim != d_model:
            self.input_projection = nn.Linear(fusion_dim, d_model)
        else:
            self.input_projection = nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)
        
        # ================================================================
        # MODULE 3: TRANSFORMER ENCODER
        # ================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # ================================================================
        # MODULE 4: PREDICTION HEADS
        # ================================================================
        
        # Goal prediction head
        self.goal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_poi_nodes),
        )
        
        # Next step prediction head
        self.nextstep_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_nodes),
        )
        
        # Category prediction head
        self.category_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_categories),
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
    
    def forward(
        self,
        node_indices: torch.Tensor,  # [batch, seq_len]
        agent_ids: torch.Tensor = None,  # [batch] - Agent IDs
        hours: torch.Tensor = None,  # [batch] - Hour of day for each trajectory
        padding_mask: torch.Tensor = None,  # [batch, seq_len] - True for padding
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: full trajectory → embeddings → transformer → predictions at all positions.
        
        Args:
            node_indices: [batch, seq_len] node indices for full trajectories
            agent_ids: [batch] agent IDs for each trajectory
            hours: [batch] hour of day (0-23) when trajectory occurred
            padding_mask: [batch, seq_len] boolean mask (True for padding positions)
        
        Returns:
            Dict with keys: 'goal', 'nextstep', 'category', 'embeddings'
            Each prediction tensor has shape [batch, seq_len, num_classes]
        """
        batch_size, seq_len = node_indices.shape
        device = node_indices.device
        
        # ================================================================
        # STEP 1: EMBED NODES WITH FULL MODALITIES
        # ================================================================
        # Expand hours to sequence length (same hour for all positions in trajectory)
        if hours is None:
            hours = torch.ones(batch_size, device=device) * 12.0  # Default to noon
        hours_seq = hours.unsqueeze(1).expand(batch_size, seq_len)  # [batch, seq_len]
        
        # Create dummy velocities (could be computed from graph distances in future)
        velocities = torch.ones(batch_size, seq_len, device=device) * 1.0  # Walking pace
        
        # Use provided agent_ids or default to zeros
        if agent_ids is None:
            agent_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Use full forward pass with all modalities (agent encoding happens internally)
        # This gives us per-node embeddings with agent, temporal, and node2vec fused
        node_embeddings = self.embedding_pipeline.forward(
            node_ids=node_indices,
            agent_ids=agent_ids,  # [batch]
            hours=hours_seq,  # [batch, seq_len]
            velocities=velocities,  # [batch, seq_len]
            spatial_coords=None,
            categories=None,
            return_per_node=True,  # Return [batch, seq_len, fusion_dim]
        )  # [batch, seq_len, fusion_dim]
        
        # ================================================================
        # STEP 2: PROJECT TO d_model
        # ================================================================
        embeddings = self.input_projection(node_embeddings)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        embeddings = self.pos_encoder(embeddings)  # [batch, seq_len, d_model]
        
        # ================================================================
        # STEP 3: CREATE CAUSAL MASK
        # ================================================================
        # Causal mask ensures position i can only attend to positions 0...i
        # This allows the model to predict from partial trajectories
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )  # [seq_len, seq_len]
        
        # Create padding mask if not provided
        if padding_mask is None:
            padding_mask = (node_indices == 0)  # [batch, seq_len]
        
        # ================================================================
        # STEP 4: TRANSFORMER ENCODING (CAUSAL)
        # ================================================================
        encoded = self.transformer_encoder(
            embeddings,
            mask=causal_mask,  # Causal mask for autoregressive prediction
            src_key_padding_mask=padding_mask,
        )  # [batch, seq_len, d_model]
        
        # ================================================================
        # STEP 5: PREDICTION HEADS (applied to all positions)
        # ================================================================
        return {
            'goal': self.goal_head(encoded),           # [batch, seq_len, num_poi_nodes]
            'nextstep': self.nextstep_head(encoded),   # [batch, seq_len, num_nodes]
            'category': self.category_head(encoded),   # [batch, seq_len, num_categories]
            'embeddings': encoded,                     # [batch, seq_len, d_model]
        }


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    Adds position information to embeddings so the model knows the order of nodes.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(1, max_len, d_model)  # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x with positional encoding added: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
