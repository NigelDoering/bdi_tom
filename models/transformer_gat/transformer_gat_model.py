"""
TRANSFORMER + GAT MODEL FOR PER-NODE BDI-ToM PREDICTION

This module contains the TransformerGATPredictor model, which combines:
1. World Graph Encoder (GAT) - Captures global graph structure via attention
2. Unified Embedding Pipeline - Captures agent, temporal, and local node features
3. Fusion Layer - Combines GAT and unified embeddings
4. Transformer Encoder - Processes fused embeddings with causal masking
5. Multi-task Prediction Heads - Predicts goals, next steps, and categories

ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────────┐
│  Graph Structure → GAT Encoder → Global Node Embeddings         │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Trajectory Nodes → Unified Pipeline → Per-Node Embeddings      │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Fusion: [GAT_emb || Unified_emb] → Combined Embeddings         │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Transformer (Causal) → Predictions (Goal, Next, Category)      │
└─────────────────────────────────────────────────────────────────┘

KEY ADVANTAGES:
- GAT provides global graph context (which nodes are similar/connected)
- Unified embedding provides trajectory-specific context (agent, time, local)
- Fusion combines both perspectives for richer representations
- Transformer learns temporal dependencies with causal masking

COMPARISON TO BASELINE:
- Baseline: Uses only Node2Vec (static, pre-trained embeddings)
- GAT Model: Uses learned GAT embeddings that capture graph structure via attention
- Both use unified pipeline, but GAT adds global graph awareness
"""

import os
from typing import Dict, Tuple, Optional

# Set MPS fallback for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import networkx as nx

from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline
from models.en_encoders.world_encoder import WorldGraphEncoder, build_edge_index_from_networkx


class TransformerGATPredictor(nn.Module):
    """
    Transformer + GAT Predictor for Theory of Mind trajectory prediction.
    
    Combines:
    1. GAT-based world graph encoder (global structure)
    2. Unified embedding pipeline (trajectory-specific features)
    3. Fusion of both embedding types
    4. Transformer with causal masking
    5. Multi-task prediction heads
    
    INPUT: node_indices [batch, seq_len]
    OUTPUT: predictions at ALL sequence positions simultaneously
    
    CAUSAL MASKING:
    - Position i can only attend to positions 0...i (not future positions)
    - Enables prediction from partial trajectories
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_agents: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        # GAT encoder params
        gat_hidden_dim: int = 128,
        gat_output_dim: int = 128,
        gat_num_layers: int = 2,
        gat_num_heads: int = 4,
        use_gat_residual: bool = True,
        # Unified embedding params
        node_embedding_dim: int = 128,
        temporal_dim: int = 64,
        agent_dim: int = 128,
        fusion_dim: int = 256,
        # Combined fusion params
        combined_fusion_dim: int = 384,
        # Transformer params
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        # Control flags
        freeze_gat: bool = False,
        freeze_unified: bool = False,
    ):
        """
        Initialize Transformer + GAT predictor.
        
        Args:
            num_nodes: Total number of nodes in graph
            num_agents: Number of agents
            num_poi_nodes: Number of POI nodes (for goal prediction)
            num_categories: Number of POI categories
            
            GAT encoder params:
                gat_hidden_dim: Hidden dimension for GAT layers
                gat_output_dim: Output dimension of GAT embeddings
                gat_num_layers: Number of GAT layers
                gat_num_heads: Number of attention heads in GAT
                use_gat_residual: Use residual connections in GAT
            
            Unified embedding params:
                node_embedding_dim: Node embedding dimension
                temporal_dim: Temporal embedding dimension
                agent_dim: Agent embedding dimension
                fusion_dim: Unified pipeline output dimension
            
            Combined fusion params:
                combined_fusion_dim: Dimension after fusing GAT + unified
            
            Transformer params:
                d_model: Transformer model dimension
                nhead: Number of transformer attention heads
                num_layers: Number of transformer layers
                dim_feedforward: Transformer feedforward dimension
                dropout: Dropout rate
            
            Control flags:
                freeze_gat: Freeze GAT parameters
                freeze_unified: Freeze unified embedding parameters
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        self.gat_output_dim = gat_output_dim
        self.fusion_dim = fusion_dim
        self.combined_fusion_dim = combined_fusion_dim
        self.d_model = d_model
        
        # ================================================================
        # MODULE 1: WORLD GRAPH ENCODER (GAT)
        # ================================================================
        self.world_encoder = WorldGraphEncoder(
            num_nodes=num_nodes,
            num_categories=num_categories,
            node_feature_dim=64,
            hidden_dim=gat_hidden_dim,
            output_dim=gat_output_dim,
            num_layers=gat_num_layers,
            num_heads=gat_num_heads,
            dropout=dropout,
            use_node_features=True,
            use_residual=use_gat_residual,
        )
        
        if freeze_gat:
            self._freeze_gat()
        
        # ================================================================
        # MODULE 2: UNIFIED EMBEDDING PIPELINE
        # ================================================================
        self.embedding_pipeline = UnifiedEmbeddingPipeline(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_categories=num_categories,
            node_embedding_dim=node_embedding_dim,
            temporal_dim=temporal_dim,  # Use parameter instead of hardcoded value
            agent_dim=agent_dim,
            fusion_dim=fusion_dim,
            hidden_dim=d_model,
            n_fusion_layers=2,
            n_heads=nhead,
            dropout=dropout,
            use_node2vec=True,
            use_temporal=False,  # Disabled: needs proper temporal data
            use_agent=True,
            use_modality_gating=True,  # Match baseline transformer
            use_cross_attention=True,  # Match baseline transformer
        )
        
        if freeze_unified:
            self._freeze_unified()
        
        # ================================================================
        # MODULE 3: FUSION LAYER (GAT + Unified)
        # ================================================================
        # Concatenate GAT and unified embeddings, then project
        self.fusion_layer = nn.Sequential(
            nn.Linear(gat_output_dim + fusion_dim, combined_fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(combined_fusion_dim),
        )
        
        # ================================================================
        # MODULE 4: PROJECTION TO d_model
        # ================================================================
        if combined_fusion_dim != d_model:
            self.input_projection = nn.Linear(combined_fusion_dim, d_model)
        else:
            self.input_projection = nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)
        
        # ================================================================
        # MODULE 5: TRANSFORMER ENCODER
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
        # MODULE 6: PREDICTION HEADS (DEEPER for more capacity)
        # ================================================================
        
        # Goal prediction head (3 layers)
        self.goal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_poi_nodes),
        )
        
        # Next step prediction head (3 layers)
        self.nextstep_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_nodes),
        )
        
        # Category prediction head (3 layers)
        self.category_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_categories),
        )
    
    def set_graph_structure(
        self,
        graph: nx.Graph,
        node_to_idx: Dict[str, int],
        category_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Set the graph structure for the GAT encoder.
        Must be called before training/inference.
        
        Args:
            graph: NetworkX graph with edge structure
            node_to_idx: Mapping from node IDs to indices
            category_to_idx: Optional mapping from category names to indices
        """
        # Build edge index from graph
        edge_index, edge_weights = build_edge_index_from_networkx(graph)
        
        # Move to same device as model
        device = next(self.parameters()).device
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device)
        
        # Set graph structure in world encoder
        self.world_encoder.set_graph_structure(edge_index, edge_weights)
        
        print(f"✅ Graph structure set: {edge_index.shape[1]} edges")
    
    def _freeze_gat(self):
        """Freeze GAT encoder parameters."""
        for param in self.world_encoder.parameters():
            param.requires_grad = False
        print("❄️  GAT encoder frozen!")
    
    def _freeze_unified(self):
        """Freeze unified embedding pipeline parameters."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = False
        print("❄️  Unified embedding pipeline frozen!")
    
    def unfreeze_gat(self):
        """Unfreeze GAT encoder parameters."""
        for param in self.world_encoder.parameters():
            param.requires_grad = True
        print("🔥 GAT encoder unfrozen!")
    
    def unfreeze_unified(self):
        """Unfreeze unified embedding parameters."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = True
        print("🔥 Unified embedding pipeline unfrozen!")
    
    def forward(
        self,
        node_indices: torch.Tensor,  # [batch, seq_len]
        agent_ids: torch.Tensor = None,  # [batch] - Agent IDs
        hours: torch.Tensor = None,  # [batch] - Hour of day (optional, not used effectively)
        padding_mask: torch.Tensor = None,  # [batch, seq_len] - True for padding
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: trajectory → GAT+Unified embeddings → fusion → transformer → predictions.
        
        Args:
            node_indices: [batch, seq_len] node indices for full trajectories
            agent_ids: [batch] agent IDs for each trajectory
            hours: [batch] hour of day (optional, currently not used)
            padding_mask: [batch, seq_len] boolean mask (True for padding positions)
        
        Returns:
            Dict with keys: 'goal', 'nextstep', 'category', 'embeddings'
            Each prediction tensor has shape [batch, seq_len, num_classes]
        """
        batch_size, seq_len = node_indices.shape
        device = node_indices.device
        
        # ================================================================
        # STEP 1: GET GAT EMBEDDINGS FOR TRAJECTORY NODES
        # ================================================================
        # GAT encoder processes entire graph once, then we index by trajectory nodes
        gat_embeddings = self.world_encoder.get_embeddings(node_indices)  # [batch, seq_len, gat_output_dim]
        
        # ================================================================
        # STEP 2: GET UNIFIED EMBEDDINGS FOR TRAJECTORY NODES
        # ================================================================
        # Expand hours to sequence length (same hour for all positions in trajectory)
        if hours is None:
            hours = torch.ones(batch_size, device=device) * 12.0  # Default to noon
        hours_seq = hours.unsqueeze(1).expand(batch_size, seq_len)  # [batch, seq_len]
        
        # Create dummy velocities (same as baseline transformer)
        velocities = torch.ones(batch_size, seq_len, device=device) * 1.0  # Walking pace
        
        # Use provided agent_ids or default to zeros
        if agent_ids is None:
            agent_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Get per-node embeddings from unified pipeline (match baseline transformer)
        unified_embeddings = self.embedding_pipeline.forward(
            node_ids=node_indices,
            agent_ids=agent_ids,  # [batch]
            hours=hours_seq,  # [batch, seq_len] - expanded from [batch]
            velocities=velocities,  # [batch, seq_len]
            spatial_coords=None,
            categories=None,
            return_per_node=True,  # Return [batch, seq_len, fusion_dim]
        )  # [batch, seq_len, fusion_dim]
        
        # ================================================================
        # STEP 3: FUSE GAT AND UNIFIED EMBEDDINGS
        # ================================================================
        # Concatenate GAT and unified embeddings
        combined_embeddings = torch.cat([gat_embeddings, unified_embeddings], dim=-1)
        # [batch, seq_len, gat_output_dim + fusion_dim]
        
        # Apply fusion layer (project and normalize)
        fused_embeddings = self.fusion_layer(combined_embeddings)
        # [batch, seq_len, combined_fusion_dim]
        
        # ================================================================
        # STEP 4: PROJECT TO d_model
        # ================================================================
        embeddings = self.input_projection(fused_embeddings)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        embeddings = self.pos_encoder(embeddings)  # [batch, seq_len, d_model]
        
        # ================================================================
        # STEP 5: CREATE CAUSAL MASK
        # ================================================================
        # Causal mask ensures position i can only attend to positions 0...i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )  # [seq_len, seq_len]
        
        # Create padding mask if not provided
        if padding_mask is None:
            padding_mask = (node_indices == 0)  # [batch, seq_len]
        
        # ================================================================
        # STEP 6: TRANSFORMER ENCODING (CAUSAL)
        # ================================================================
        encoded = self.transformer_encoder(
            embeddings,
            mask=causal_mask,  # Causal mask for autoregressive prediction
            src_key_padding_mask=padding_mask,
        )  # [batch, seq_len, d_model]
        
        # ================================================================
        # STEP 7: PREDICTION HEADS (applied to all positions)
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


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test Transformer + GAT model."""
    import networkx as nx
    
    print("=" * 80)
    print("Testing Transformer + GAT Predictor")
    print("=" * 80)
    
    # Create synthetic graph
    num_nodes = 100
    num_poi = 20
    G = nx.erdos_renyi_graph(num_nodes, 0.05)
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Total nodes: {num_nodes}")
    print(f"   POI nodes: {num_poi}")
    print(f"   Edges: {G.number_of_edges()}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    model = TransformerGATPredictor(
        num_nodes=num_nodes,
        num_agents=10,
        num_poi_nodes=num_poi,
        num_categories=7,
        # GAT params
        gat_hidden_dim=128,
        gat_output_dim=128,
        gat_num_layers=2,
        gat_num_heads=4,
        # Unified params
        node_embedding_dim=128,
        agent_dim=128,
        fusion_dim=256,
        # Combined fusion
        combined_fusion_dim=384,
        # Transformer params
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
    ).to(device)
    
    # Set graph structure
    node_to_idx = {str(i): i for i in range(num_nodes)}
    model.set_graph_structure(G, node_to_idx)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🔢 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Test forward pass
    print(f"\n🔄 Testing forward pass...")
    batch_size = 4
    seq_len = 10
    
    node_indices = torch.randint(0, num_nodes, (batch_size, seq_len), device=device)
    agent_ids = torch.randint(0, 10, (batch_size,), device=device)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        predictions = model(node_indices, agent_ids, None, padding_mask)
    
    print(f"\n✅ Predictions:")
    print(f"   Goal: {predictions['goal'].shape}")
    print(f"   Next step: {predictions['nextstep'].shape}")
    print(f"   Category: {predictions['category'].shape}")
    print(f"   Embeddings: {predictions['embeddings'].shape}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
