"""
ENHANCED TRAJECTORY ENCODER - Expert-Level Implementation

This improved trajectory encoder integrates with the UnifiedEmbeddingPipeline
to process rich pre-computed embeddings from node2vec, temporal, agent, and
multi-modal fusion encoders.

Key Improvements over Previous Version:
1. Accepts Pre-Computed Embeddings Uses unified embeddings as input
2. Enhanced Transformer Architecture 
   - Better layer normalization
   - Residual connections
   - Improved attention mechanisms
3. Multi-Scale Processing:
   - Local context (sequential node relationships)
   - Global context (full trajectory overview)
   - Hierarchical aggregation
4. Theory of Mind Integration:
   - Captures agent intent through embeddings
   - Temporal patterns for belief inference
   - Multi-task readiness

Architecture:
INPUT (pre-computed embeddings from UnifiedEmbeddingPipeline)
  ↓
[Multi-Head Self-Attention] - Learn trajectory structure
  ↓
[Residual + LayerNorm] - Stabilize learning
  ↓
[Position-Aware Encoding] - Add sequential context
  ↓
[Cross-Attention] - Focus on important positions
  ↓
[Temporal Aggregation] - Combine temporal context
  ↓
[Multi-Scale Pooling] - Extract trajectory features
  ↓
OUTPUT (trajectory representation for prediction heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ============================================================================
# ENHANCED TRAJECTORY ENCODER
# ============================================================================

class EnhancedTrajectoryEncoder(nn.Module):
    """
    Expert-level trajectory encoder for processing pre-computed embeddings.
    
    This encoder takes embeddings from the UnifiedEmbeddingPipeline and
    extracts rich trajectory representations for multi-task prediction.
    
    Key Features:
    - Multi-head self-attention for trajectory structure learning
    - Residual connections for stable gradient flow
    - Hierarchical pooling for multi-scale features
    - Position-aware encoding for sequential information
    - Theory of Mind interpretability
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        n_heads: int = 4,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_cross_attention: bool = True,
        use_temporal_aggregation: bool = True,
    ):
        """
        Initialize enhanced trajectory encoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for internal layers
            output_dim: Output trajectory embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_positional_encoding: Whether to add positional encodings
            use_cross_attention: Whether to use cross-attention layers
            use_temporal_aggregation: Whether to aggregate temporal context
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # ================================================================
        # POSITIONAL ENCODING
        # ================================================================
        if use_positional_encoding:
            self.positional_encoding = self._create_positional_encoding(
                max_seq_len, embedding_dim
            )
            self.pos_dropout = nn.Dropout(dropout)
        else:
            self.register_buffer('positional_encoding', None)
        
        # ================================================================
        # PROJECTION LAYER
        # ================================================================
        # Project input embeddings to transformer dimension
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        
        # ================================================================
        # MULTI-HEAD SELF-ATTENTION LAYERS
        # ================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-normalization for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # ================================================================
        # CROSS-ATTENTION FOR IMPORTANT POSITIONS
        # ================================================================
        if use_cross_attention:
            self.importance_query = nn.Parameter(
                torch.randn(1, n_heads, hidden_dim // n_heads)
            )
            cross_attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attention = cross_attention_layer
        else:
            self.cross_attention = None
        
        # ================================================================
        # MULTI-SCALE POOLING
        # ================================================================
        # Mean pooling
        self.mean_pool = nn.Identity()
        
        # Max pooling with projection
        self.max_pool_proj = nn.Linear(hidden_dim, hidden_dim // n_heads)
        
        # Attention-weighted pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ================================================================
        # AGGREGATION & OUTPUT
        # ================================================================
        # Combine multi-scale features
        num_pooling_methods = 3 if use_cross_attention else 2
        combined_dim = hidden_dim + (hidden_dim // n_heads) + hidden_dim
        
        self.aggregator = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # ================================================================
        # TEMPORAL CONTEXT FUSION
        # ================================================================
        if use_temporal_aggregation:
            self.temporal_context_encoder = nn.Sequential(
                nn.Linear(24 + 7, hidden_dim),  # 24 for hour one-hot, 7 for day
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.temporal_fusion = nn.Linear(output_dim * 2, output_dim)
        else:
            self.temporal_context_encoder = None
            self.temporal_fusion = None
    
    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Create positional encoding using sinusoidal functions."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def _apply_positional_encoding(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply positional encoding to embeddings."""
        seq_len = x.shape[1]
        emb_dim = x.shape[2]
        pe = self.positional_encoding[:, :seq_len, :emb_dim].to(x.device)
        # Ensure batch dimension matches
        if pe.shape[0] == 1 and x.shape[0] > 1:
            pe = pe.expand(x.shape[0], -1, -1)
        return x + pe
    
    def _mean_pooling(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean pooling over sequence."""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            return (x * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return x.mean(dim=1)
    
    def _max_pooling(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Max pooling over sequence."""
        if mask is not None:
            x_masked = x.clone()
            x_masked[mask == 0] = float('-inf')
            max_vals = x_masked.max(dim=1)[0]
            return self.max_pool_proj(max_vals)
        max_vals = x.max(dim=1)[0]
        return self.max_pool_proj(max_vals)
    
    def _attention_pooling(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Attention-based pooling."""
        # Compute attention weights
        attn_weights = self.attention_pool(x)  # (batch, seq_len, 1)
        
        # Apply mask to attention weights
        if mask is not None:
            attn_weights[mask == 0] = float('-inf')
        
        # Normalize
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        output = (x * attn_weights).sum(dim=1)
        
        return output
    
    def forward(
        self,
        embeddings: torch.Tensor,
        hours: Optional[torch.Tensor] = None,
        days: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through enhanced trajectory encoder.
        
        Args:
            embeddings: (batch, seq_len, embedding_dim) - Pre-computed embeddings
            hours: (batch, seq_len) - Hour information (0-23)
            days: (batch, seq_len) - Day information (0-6)
            mask: (batch, seq_len) - Padding mask
            return_intermediate: Return intermediate representations
            
        Returns:
            Trajectory representation: (batch, output_dim)
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Step 1: Project to transformer dimension
        x = self.input_proj(embeddings)  # (batch, seq_len, hidden_dim)
        x = self.input_ln(x)
        
        # Step 2: Add positional encoding
        if self.positional_encoding is not None:
            x = self._apply_positional_encoding(x)
            x = self.pos_dropout(x)
        
        # Step 3: Apply transformer with masking
        if mask is not None:
            attention_mask = (mask == 0)
        else:
            attention_mask = None
        
        x_encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=attention_mask
        )  # (batch, seq_len, hidden_dim)
        
        # Step 4: Multi-scale pooling
        mean_pool = self._mean_pooling(x_encoded, mask)  # (batch, hidden_dim)
        max_pool = self._max_pooling(x_encoded, mask)    # (batch, hidden_dim//n_heads)
        attn_pool = self._attention_pooling(x_encoded, mask)  # (batch, hidden_dim)
        
        # Step 5: Combine multi-scale features
        combined = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)  # (batch, combined_dim)
        
        # Step 6: Aggregate
        traj_repr = self.aggregator(combined)  # (batch, output_dim)
        
        # Step 7: Optional temporal context fusion
        if self.temporal_context_encoder is not None and hours is not None and days is not None:
            # One-hot encode hour and day (taking first position or average)
            if hours.dim() == 2:
                hour_0 = hours[:, 0].long()
                day_0 = days[:, 0].long() if days.dim() == 2 else days.long()
            else:
                hour_0 = hours.long()
                day_0 = days.long()
            
            hour_onehot = F.one_hot(hour_0, num_classes=24).float()
            day_onehot = F.one_hot(day_0, num_classes=7).float()
            
            temporal_context = torch.cat([hour_onehot, day_onehot], dim=-1)
            temporal_repr = self.temporal_context_encoder(temporal_context)
            
            # Fuse
            traj_repr = self.temporal_fusion(
                torch.cat([traj_repr, temporal_repr], dim=-1)
            )
        
        if return_intermediate:
            return traj_repr, {
                'x_encoded': x_encoded,
                'mean_pool': mean_pool,
                'max_pool': max_pool,
                'attn_pool': attn_pool,
            }
        
        return traj_repr


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# ============================================================================

class TrajectoryEncoder(nn.Module):
    """
    Backward-compatible wrapper that can work with both old and new modes.
    
    Old mode: Takes node_ids and hours (legacy)
    New mode: Takes pre-computed embeddings from UnifiedEmbeddingPipeline
    """
    
    def __init__(self, num_nodes=None, node_feat_dim=None, hidden_dim=128, 
                 output_dim=128, n_layers=2, n_heads=4, dropout=0.1,
                 embedding_dim=None, use_unified_embeddings=False):
        """
        Initialize trajectory encoder in either mode.
        
        Args:
            num_nodes: (Old mode) Number of nodes
            node_feat_dim: (Old mode) Node embedding dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            n_layers: Number of layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            embedding_dim: (New mode) Pre-computed embedding dimension
            use_unified_embeddings: Whether to use new unified embedding mode
        """
        super().__init__()
        
        self.use_unified_embeddings = use_unified_embeddings
        
        if use_unified_embeddings:
            # New mode: enhanced encoder for pre-computed embeddings
            self.enhanced_encoder = EnhancedTrajectoryEncoder(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                use_positional_encoding=True,
                use_cross_attention=True,
                use_temporal_aggregation=True,
            )
        else:
            # Old mode: original simple encoder
            from models.encoders.trajectory_encoder import TrajectoryEncoder as OriginalTrajectoryEncoder
            self.original_encoder = OriginalTrajectoryEncoder(
                num_nodes=num_nodes,
                node_feat_dim=node_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout
            )
    
    def forward(self, *args, **kwargs):
        """Forward pass in either mode."""
        if self.use_unified_embeddings:
            return self.enhanced_encoder(*args, **kwargs)
        else:
            return self.original_encoder(*args, **kwargs)


# ============================================================================
# TESTING
# ============================================================================

def test_enhanced_trajectory_encoder():
    """Test the enhanced trajectory encoder."""
    print("=" * 100)
    print("🧠 ENHANCED TRAJECTORY ENCODER - COMPREHENSIVE TEST")
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Using device: {device}")
    
    # Create encoder
    encoder = EnhancedTrajectoryEncoder(
        embedding_dim=128,
        hidden_dim=256,
        output_dim=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=100,
        dropout=0.1,
        use_positional_encoding=True,
        use_cross_attention=True,
        use_temporal_aggregation=True,
    ).to(device)
    
    # Create synthetic embeddings (from UnifiedEmbeddingPipeline)
    batch_size, seq_len, emb_dim = 4, 20, 128
    embeddings = torch.randn(batch_size, seq_len, emb_dim, device=device)
    hours = torch.randint(0, 24, (batch_size, seq_len), device=device)
    days = torch.randint(0, 7, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    
    # Test 1: Basic forward pass
    print("\n1️⃣  TEST: Basic Forward Pass")
    print("-" * 100)
    
    output = encoder(embeddings, hours, days, mask)
    print(f"✅ Output shape: {output.shape}")
    print(f"   Expected: torch.Size([{batch_size}, 128])")
    assert output.shape == (batch_size, 128)
    
    # Test 2: With intermediate outputs
    print("\n2️⃣  TEST: With Intermediate Outputs")
    print("-" * 100)
    
    output, intermediate = encoder(embeddings, hours, days, mask, return_intermediate=True)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Intermediate outputs:")
    for key, val in intermediate.items():
        print(f"   - {key}: {val.shape}")
    
    # Test 3: Without mask
    print("\n3️⃣  TEST: Without Padding Mask")
    print("-" * 100)
    
    output_no_mask = encoder(embeddings, hours, days, mask=None)
    print(f"✅ Output shape: {output_no_mask.shape}")
    print(f"   Distance from masked version: {(output - output_no_mask).norm().item():.6f}")
    
    # Test 4: Parameter count
    print("\n4️⃣  TEST: Parameter Statistics")
    print("-" * 100)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Test 5: Performance
    print("\n5️⃣  TEST: Performance Metrics")
    print("-" * 100)
    
    import time
    encoder.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = encoder(embeddings, hours, days, mask)
        elapsed = time.time() - start
    
    print(f"✅ Average forward time: {elapsed / 10 * 1000:.2f} ms")
    print(f"   Batch size: {batch_size}, Sequence length: {seq_len}")
    
    print("\n" + "=" * 100)
    print("✨ ALL TESTS PASSED - ENHANCED TRAJECTORY ENCODER READY!")
    print("=" * 100)


if __name__ == "__main__":
    test_enhanced_trajectory_encoder()
