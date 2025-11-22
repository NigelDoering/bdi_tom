"""
Multi-Modal Fusion Layer - Expert Level Implementation

This is the heart of the advanced trajectory encoding system.

It integrates three critical dimensions:
1. Graph Structure Node2Vec embeddings capturing spatial/structural relationships
2. Temporal Context Rich temporal features (circadian, weekly, deltas)
3. Agent Identity Individual agent preferences and behavioral patterns

The fusion uses an attention-based mechanism that learns:
- Which dimensions matter most for prediction
- How to weight information from different modalities
- Cross-modal interactions and dependencies

This is inspired by successful multi-modal systems in:
- Vision-language models (learning cross-modal alignments)
- Multimodal transformers (selective fusion)
- Theory of Mind systems (integrating agent state + environment + time)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class ModalityGating(nn.Module):
    """
    Learnable gating mechanism for multi-modal fusion.
    
    Uses a context-dependent gating mechanism to adaptively weight
    contributions from different modalities. This prevents one modality
    from dominating and allows specialization.
    
    Inspired by mixture-of-experts and attention mechanisms.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_modalities: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize modality gating.
        
        Args:
            input_dim: Dimension of input features
            num_modalities: Number of modalities to gate (default: 3)
            hidden_dim: Hidden dimension for gating network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Context encoder (learns what aspects matter for gating)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Per-modality gating networks
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Gate modality features.
        
        Args:
            modality_features: Dict with keys like 'spatial', 'temporal', 'agent'
                              Each value is a tensor
            
        Returns:
            Dict with gated features
        """
        modality_list = list(modality_features.values())
        
        # Concatenate all modalities as context
        context = torch.cat(modality_list, dim=-1)
        context_emb = self.context_encoder(context)
        
        # Compute gates for each modality
        gates = []
        for gate_net in self.gate_networks:
            gate = gate_net(context_emb)
            gates.append(gate)
        
        # Normalize gates (softmax for interpretability)
        gate_tensor = torch.cat(gates, dim=-1)  # (batch, num_modalities)
        gate_weights = F.softmax(gate_tensor, dim=-1)
        
        # Apply gates to each modality
        gated_features = {}
        modality_keys = list(modality_features.keys())
        
        for i, key in enumerate(modality_keys):
            gated_features[key] = modality_features[key] * gate_weights[:, i:i+1]
        
        return gated_features


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for learning inter-modality dependencies.
    
    Allows each modality to attend to features from other modalities,
    enabling rich information exchange and context-dependent fusion.
    
    This is more sophisticated than simple concatenation or element-wise ops.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_modalities: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            feature_dim: Dimension of feature vectors
            num_modalities: Number of modalities
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Per-modality attention mechanisms
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_modalities)
        ])
        
        # Normalization and feedforward per modality
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_modalities)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 2, feature_dim)
            )
            for _ in range(num_modalities)
        ])
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            modality_features: Dict with modality features
            
        Returns:
            Dict with attention-refined features
        """
        modality_keys = list(modality_features.keys())
        modality_list = list(modality_features.values())
        
        refined_features = {}
        
        for i, target_key in enumerate(modality_keys):
            target_feature = modality_features[target_key]  # (batch, seq?, dim)
            
            # Add sequence dimension if missing
            if target_feature.dim() == 2:
                target_feature = target_feature.unsqueeze(1)
                need_squeeze = True
            else:
                need_squeeze = False
            
            # Concatenate other modalities as attention context
            context_list = [modality_features[k] for k in modality_keys if k != target_key]
            
            # Make all 3D if needed
            context_list = [
                f.unsqueeze(1) if f.dim() == 2 else f
                for f in context_list
            ]
            
            context = torch.cat(context_list, dim=1)  # (batch, other_seq_total, dim)
            
            # Cross-modal attention
            attn_output, _ = self.cross_attentions[i](
                target_feature, context, context
            )
            
            # Residual + norm + FFN
            attn_output = self.norm_layers[i](attn_output + target_feature)
            ffn_output = self.ffn_layers[i](attn_output)
            output = self.norm_layers[i](ffn_output + attn_output)
            
            if need_squeeze:
                output = output.squeeze(1)
            
            refined_features[target_key] = output
        
        return refined_features


class MultiModalFusion(nn.Module):
    """
    Expert-level multi-modal fusion combining three key dimensions:
    
    1. **Spatial/Structural** (via Node2Vec embeddings):
       - Captures graph topology and spatial proximity
       - Ensures spatially coherent predictions
       
    2. **Temporal** (via AdvancedTemporalEncoder):
       - Captures time-of-day, day-of-week, temporal deltas
       - Models circadian rhythms and weekly patterns
       
    3. **Agent** (via AgentContext):
       - Captures individual preferences and behavioral patterns
       - Enables personalized predictions
    
    The fusion uses:
    - Modality gating (learn importance weights)
    - Cross-modal attention (inter-modality information flow)
    - Multi-scale fusion (element-wise + concatenation + attention)
    - Hierarchical processing (local + global fusion)
    """
    
    def __init__(
        self,
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        agent_dim: int = 64,
        fusion_dim: int = 128,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        use_modality_gating: bool = True,
        use_cross_attention: bool = True
    ):
        """
        Initialize multi-modal fusion.
        
        Args:
            spatial_dim: Dimension of spatial features (Node2Vec)
            temporal_dim: Dimension of temporal features
            agent_dim: Dimension of agent features
            fusion_dim: Output dimension of fusion
            num_heads: Number of attention heads
            num_fusion_layers: Number of fusion layers
            dropout: Dropout rate
            use_modality_gating: Whether to use modality gating
            use_cross_attention: Whether to use cross-modal attention
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.agent_dim = agent_dim
        self.fusion_dim = fusion_dim
        self.use_modality_gating = use_modality_gating
        self.use_cross_attention = use_cross_attention
        
        # Input projections (normalize dimensions)
        # Project each modality to fusion_dim/3, but ensure divisibility by num_heads
        proj_dim = fusion_dim // 3
        # Adjust to be divisible by num_heads
        if proj_dim % num_heads != 0:
            proj_dim = (proj_dim // num_heads) * num_heads
        
        self.spatial_proj = nn.Linear(spatial_dim, proj_dim)
        self.temporal_proj = nn.Linear(temporal_dim, proj_dim)
        self.agent_proj = nn.Linear(agent_dim, proj_dim)
        
        # Modality gating
        if use_modality_gating:
            total_proj_dim = proj_dim * 3
            self.gating = ModalityGating(
                input_dim=total_proj_dim,
                num_modalities=3,
                hidden_dim=total_proj_dim,
                dropout=dropout
            )
        
        # Cross-modal attention
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                feature_dim=proj_dim,
                num_modalities=3,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Hierarchical fusion layers
        fusion_input_dim = proj_dim * 3
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(fusion_input_dim)
            )
            for _ in range(num_fusion_layers)
        ])
        
        # Final projection - project fused features to output dimension
        # fused is currently fusion_input_dim, project to fusion_dim
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor,
        agent_features: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Fuse multi-modal features.
        
        Args:
            spatial_features: (batch_size, spatial_dim) - Node2Vec embeddings
            temporal_features: (batch_size, temporal_dim) - Temporal embeddings
            agent_features: (batch_size, agent_dim) - Agent embeddings
            return_components: If True, return intermediate components
            
        Returns:
            Fused embeddings of shape (batch_size, fusion_dim)
            If return_components=True: tuple of (fused, components_dict)
        """
        components = {}
        
        # Project all modalities to fusion dimension
        spatial_proj = self.spatial_proj(spatial_features)
        temporal_proj = self.temporal_proj(temporal_features)
        agent_proj = self.agent_proj(agent_features)
        
        components['spatial_proj'] = spatial_proj.clone()
        components['temporal_proj'] = temporal_proj.clone()
        components['agent_proj'] = agent_proj.clone()
        
        # Store as dictionary for gating and attention
        modalities = {
            'spatial': spatial_proj,
            'temporal': temporal_proj,
            'agent': agent_proj
        }
        
        # Apply cross-modal attention (if enabled)
        if self.use_cross_attention:
            modalities = self.cross_attention(modalities)
            components['cross_attention'] = {k: v.clone() for k, v in modalities.items()}
        
        # Concatenate all modalities
        fused = torch.cat([modalities['spatial'], modalities['temporal'], modalities['agent']], dim=-1)
        
        # Apply modality gating (if enabled)
        if self.use_modality_gating:
            gated_modalities = self.gating(modalities)
            fused_gated = torch.cat([gated_modalities['spatial'], gated_modalities['temporal'], gated_modalities['agent']], dim=-1)
            components['gating'] = fused_gated.clone()
            fused = fused + fused_gated  # Residual connection
        
        # Hierarchical fusion layers
        for layer in self.fusion_layers:
            residual = fused
            fused = layer(fused)
            fused = fused + residual  # Residual connection
        
        # Final projection
        output = self.output_proj(fused)
        
        if return_components:
            return output, components
        return output


class MultiModalTrajectoryEncoder(nn.Module):
    """
    Complete trajectory encoder with multi-modal fusion.
    
    This is the final expert-level encoder that brings everything together:
    
    Input Processing:
    ├─ Node Features (sequence of node IDs)
    │  └─ Node2VecEncoder → Spatial embeddings
    ├─ Temporal Features
    │  └─ AdvancedTemporalEncoder → Temporal embeddings
    └─ Agent Features
       └─ MultiModalAgentEncoder → Agent embeddings
    
    Fusion:
    ├─ Cross-modal attention (learn modality interactions)
    ├─ Modality gating (learn importance weights)
    └─ Hierarchical fusion (multi-scale integration)
    
    Output: Rich trajectory representation ready for downstream tasks
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_agents: int,
        num_categories: int = 7,
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        agent_dim: int = 64,
        fusion_dim: int = 128,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_temporal_deltas: bool = True
    ):
        """
        Initialize multi-modal trajectory encoder.
        
        Args:
            num_nodes: Number of nodes in graph
            num_agents: Number of agents
            num_categories: Number of POI categories
            spatial_dim: Dimension of spatial embeddings
            temporal_dim: Dimension of temporal embeddings
            agent_dim: Dimension of agent embeddings
            fusion_dim: Dimension of fused output
            num_heads: Number of attention heads
            num_fusion_layers: Number of fusion layers
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding in spatial encoder
            use_temporal_deltas: Whether to use temporal deltas
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from models.en_encoders.node2vec_encoder import (
            Node2VecWithPositionalEncoding, Node2VecEncoder
        )
        from models.en_encoders.temporal_encoder import AdvancedTemporalEncoder
        from models.en_encoders.agent_encoder import MultiModalAgentEncoder
        
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.fusion_dim = fusion_dim
        self.use_temporal_deltas = use_temporal_deltas
        
        # Spatial encoder (Node2Vec with optional positional encoding)
        if use_positional_encoding:
            self.spatial_encoder = Node2VecWithPositionalEncoding(
                num_nodes=num_nodes,
                num_categories=num_categories,
                embedding_dim=spatial_dim,
                hidden_dim=128,
                n_layers=2,
                dropout=dropout
            )
        else:
            self.spatial_encoder = Node2VecEncoder(
                num_nodes=num_nodes,
                num_categories=num_categories,
                embedding_dim=spatial_dim,
                hidden_dim=128,
                n_layers=2,
                dropout=dropout
            )
        
        # Temporal encoder
        self.temporal_encoder = AdvancedTemporalEncoder(
            temporal_dim=temporal_dim,
            hidden_dim=128,
            include_day_of_week=True,
            include_temporal_deltas=use_temporal_deltas,
            include_velocity=True,
            dropout=dropout
        )
        
        # Agent encoder
        self.agent_encoder = MultiModalAgentEncoder(
            num_agents=num_agents,
            agent_emb_dim=agent_dim,
            num_categories=num_categories,
            enable_behavioral_profile=True,
            dropout=dropout
        )
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            agent_dim=agent_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            num_fusion_layers=num_fusion_layers,
            dropout=dropout,
            use_modality_gating=True,
            use_cross_attention=True
        )
        
        # Optional sequence processing (aggregate sequence to single vector)
        self.sequence_processor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(
        self,
        node_ids: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None,
        node_categories: Optional[torch.Tensor] = None,
        hours: torch.Tensor = None,
        days: Optional[torch.Tensor] = None,
        temporal_deltas: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        agent_ids: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        return_all_components: bool = False,
        aggregate_sequence: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through multi-modal encoder.
        
        Args:
            node_ids: (batch_size, seq_len) - Node IDs in trajectory
            spatial_coords: (batch_size, seq_len, 2) - Optional lat/lon
            node_categories: (batch_size, seq_len) - Optional node categories
            hours: (batch_size, seq_len) or (batch_size,) - Hour of day
            days: (batch_size,) - Day of week
            temporal_deltas: (batch_size, seq_len) - Time deltas
            velocities: (batch_size, seq_len) - Velocities
            agent_ids: (batch_size,) - Agent IDs
            mask: (batch_size, seq_len) - Padding mask
            return_all_components: If True, return all intermediate tensors
            aggregate_sequence: If True, aggregate sequence to single vector
            
        Returns:
            Fused trajectory embedding of shape (batch_size, fusion_dim)
            If aggregate_sequence=False: (batch_size, seq_len, fusion_dim)
        """
        batch_size = node_ids.shape[0]
        seq_len = node_ids.shape[1] if node_ids.dim() > 1 else 1
        device = node_ids.device
        
        # Ensure hours is 2D
        if hours is None:
            hours = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        elif hours.dim() == 1:
            hours = hours.unsqueeze(1).expand(batch_size, seq_len)
        
        # Encode spatial features
        spatial_features = self.spatial_encoder(
            node_ids, spatial_coords, node_categories, mask
        )  # (batch_size, seq_len, spatial_dim) or (batch_size, spatial_dim)
        
        # Handle sequence dimension
        if spatial_features.dim() == 3:
            # Mean pool over sequence
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                spatial_agg = (spatial_features * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                spatial_agg = spatial_features.mean(dim=1)
        else:
            spatial_agg = spatial_features
        
        # Encode temporal features
        temporal_features = self.temporal_encoder(
            hours[:, 0] if hours.dim() > 1 else hours,
            days,
            temporal_deltas,
            velocities[:, 0].unsqueeze(-1) if velocities is not None else None
        )  # (batch_size, temporal_dim)
        
        # Encode agent features
        if agent_ids is not None:
            agent_features = self.agent_encoder(
                agent_ids,
                hours[:, 0] if hours.dim() > 1 else hours,
                velocities[:, 0].unsqueeze(-1) if velocities is not None else None
            )  # (batch_size, agent_dim)
        else:
            agent_features = torch.zeros(batch_size, self.agent_encoder.agent_emb_dim, device=device)
        
        # Fuse all modalities
        fused = self.fusion(
            spatial_agg, temporal_features, agent_features, 
            return_components=False
        )
        
        # Process sequence
        output = self.sequence_processor(fused)
        
        if return_all_components:
            return output, {
                'spatial': spatial_agg,
                'temporal': temporal_features,
                'agent': agent_features,
                'fused': fused
            }
        
        return output


# ============================================================================
# TESTING AND DEMO
# ============================================================================

def test_fusion():
    """Test multi-modal fusion."""
    print("=" * 80)
    print("Testing Multi-Modal Fusion System")
    print("=" * 80)
    
    batch_size = 4
    seq_len = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test 1: Modality Gating
    print("\n1. Testing ModalityGating...")
    gating = ModalityGating(
        input_dim=192,
        num_modalities=3,
        hidden_dim=128,
        dropout=0.1
    ).to(device)
    
    modalities = {
        'spatial': torch.randn(batch_size, 64, device=device),
        'temporal': torch.randn(batch_size, 64, device=device),
        'agent': torch.randn(batch_size, 64, device=device)
    }
    
    gated = gating(modalities)
    print(f"   ✓ Gated modalities: {list(gated.keys())}")
    
    # Test 2: Cross-Modal Attention
    print("\n2. Testing CrossModalAttention...")
    cross_attn = CrossModalAttention(
        feature_dim=64,
        num_modalities=3,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    refined = cross_attn(modalities)
    print(f"   ✓ Refined modalities: {list(refined.keys())}")
    
    # Test 3: Multi-Modal Fusion
    print("\n3. Testing MultiModalFusion...")
    fusion = MultiModalFusion(
        spatial_dim=64,
        temporal_dim=64,
        agent_dim=64,
        fusion_dim=128,
        num_heads=4,
        num_fusion_layers=2,
        dropout=0.1
    ).to(device)
    
    fused, components = fusion(
        modalities['spatial'],
        modalities['temporal'],
        modalities['agent'],
        return_components=True
    )
    print(f"   ✓ Fused shape: {fused.shape}")
    print(f"   ✓ Components: {list(components.keys())}")
    
    # Test parameter count
    print("\n4. Model parameters...")
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    print("\n" + "=" * 80)
    print("✓ All fusion tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_fusion()
