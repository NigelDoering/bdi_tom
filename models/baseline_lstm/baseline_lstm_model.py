"""
BASELINE LSTM MODEL FOR PER-NODE BDI-ToM PREDICTION

This module contains the PerNodeToMPredictor model, a baseline LSTM-based
architecture for Theory of Mind prediction in spatial trajectory tasks.

ARCHITECTURE:
- Module 1: UnifiedEmbeddingPipeline (node embeddings with Node2Vec)
- Module 2: LSTM history aggregation
- Module 3: Multi-task prediction heads (goal, next-step, category)

The model takes trajectory history as input and predicts:
1. Goal node (which POI the agent is heading to)
2. Next step node (immediate next location)
3. Goal category (semantic category of the destination)
"""

import os
from typing import Dict, Optional, Tuple

# Set MPS fallback for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn

from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline


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
            use_temporal=False,
            use_agent=True,              # Enable agent encoder (100 agents)
            use_modality_gating=True,    # Enable modality gating
            use_cross_attention=True,    # Enable cross-attention
        )
        
        if freeze_embedding:
            self._freeze_embeddings()
        
        # ================================================================
        # MODULE 2: HISTORY AGGREGATION WITH LSTM
        # ================================================================
        self.history_lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        
        # ================================================================
        # MODULE 3: PREDICTION HEADS
        # ================================================================
        
        # Simpler fusion
        self.feature_fusion = nn.Linear(fusion_dim + hidden_dim // 2, hidden_dim)
        
        # Goal prediction head - simplified
        self.goal_head = nn.Linear(hidden_dim, num_poi_nodes)

        # Next step prediction head - simplified  
        self.nextstep_head = nn.Linear(hidden_dim, num_nodes)

        # Category prediction head - simplified
        self.category_head = nn.Linear(hidden_dim, num_categories)
    
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
        agent_ids: Optional[torch.Tensor] = None,       # [batch]
        hours: Optional[torch.Tensor] = None,           # [batch]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: history → embeddings → predictions.
        
        Args:
            history_node_indices: [batch, seq_len] node indices
            history_lengths: [batch] actual sequence length before padding
            agent_ids: [batch] agent IDs for agent embedding
            hours: [batch] hour of day for temporal context
        
        Returns:
            Dict with keys: 'goal', 'nextstep', 'category', 'embeddings'
        """
        batch_size, seq_len = history_node_indices.shape
        device = history_node_indices.device
        
        # ================================================================
        # STEP 2: ENCODE TRAJECTORY HISTORY WITH UNIFIED PIPELINE
        # ================================================================
        # Build a padding mask: 1 for valid positions, 0 for padding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
        mask = (positions < history_lengths.unsqueeze(1)).float()       # (B, S)
        
        # Expand scalar hour to sequence-level for the temporal encoder
        if hours is not None:
            hours_seq = hours.unsqueeze(1).expand(-1, seq_len)  # (B, S)
        else:
            hours_seq = torch.full((batch_size, seq_len), 12.0, device=device)
        
        # Default agent_ids to zeros if not provided
        if agent_ids is None:
            agent_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        velocities = torch.ones(batch_size, seq_len, device=device)
        
        # Full unified embedding pipeline (node + temporal + agent fusion)
        history_embeddings = self.embedding_pipeline(
            node_ids=history_node_indices,
            agent_ids=agent_ids,
            hours=hours_seq,
            velocities=velocities,
            mask=mask,
            return_per_node=True,
        )  # [batch, seq_len, fusion_dim]
        # If the pipeline returns a tuple (emb, components), take the emb
        if isinstance(history_embeddings, tuple):
            history_embeddings = history_embeddings[0]
        
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
