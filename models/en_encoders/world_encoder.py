"""
World Graph Encoder using Graph Attention Networks (GAT)

This module provides a GAT-based encoder that captures the global structure of the
campus navigation graph. Unlike node2vec which provides static embeddings, the GAT
learns to aggregate information from neighboring nodes using attention mechanisms.

KEY FEATURES:
- Multi-head attention over graph structure
- Captures both spatial and semantic relationships
- Can incorporate node features (POI categories, spatial coords)
- Provides both node-level and graph-level representations

ARCHITECTURE:
1. Initial node feature encoding (category, spatial, etc.)
2. Multi-layer GAT with attention over graph edges
3. Optional graph-level pooling for global context
4. Output: rich node embeddings that encode graph structure

This is especially powerful for campus navigation where:
- Building proximity matters (nearby buildings are related)
- Category relationships matter (all dining halls form a cluster)
- Path structure matters (frequently co-traversed nodes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import numpy as np
import networkx as nx


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer (GAT).
    
    Computes attention-weighted aggregation of neighbor features:
    h_i' = σ(Σ_j α_ij W h_j)
    
    where α_ij = attention(h_i, h_j) computed via learned attention mechanism.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize GAT layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout rate
            alpha: LeakyReLU negative slope for attention
            concat: If True, concatenate multi-head outputs; else average
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation for features
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges] (source, target pairs)
            edge_weights: Optional edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Linear transformation
        h = self.W(x)  # [num_nodes, out_features]
        
        num_nodes = h.size(0)
        
        # Prepare for attention computation
        # Get source and target features
        edge_src, edge_tgt = edge_index[0], edge_index[1]
        
        # Concatenate source and target features for each edge
        h_src = h[edge_src]  # [num_edges, out_features]
        h_tgt = h[edge_tgt]  # [num_edges, out_features]
        
        # Attention mechanism: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        edge_features = torch.cat([h_src, h_tgt], dim=1)  # [num_edges, 2*out_features]
        e = self.leakyrelu(torch.matmul(edge_features, self.a).squeeze(-1))  # [num_edges]
        
        # Incorporate edge weights if provided
        if edge_weights is not None:
            e = e * edge_weights
        
        # Softmax attention per target node
        # For each target node, normalize attention over its incoming edges
        attention = torch.zeros(num_nodes, dtype=e.dtype, device=e.device)
        attention = attention.scatter_add_(0, edge_tgt, torch.exp(e))  # Sum exp(e) per target
        attention = attention[edge_tgt]  # [num_edges] - normalization factor for each edge
        alpha = torch.exp(e) / (attention + 1e-8)  # [num_edges] - normalized attention weights
        
        # Apply dropout to attention weights
        alpha = self.dropout_layer(alpha)
        
        # Aggregate neighbors with attention weights
        h_prime = torch.zeros_like(h)  # [num_nodes, out_features]
        h_weighted = h_src * alpha.unsqueeze(-1)  # [num_edges, out_features]
        h_prime = h_prime.scatter_add_(0, edge_tgt.unsqueeze(-1).expand_as(h_weighted), h_weighted)
        
        # Apply activation if concatenating (for multi-head)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer.
    
    Runs multiple attention heads in parallel and concatenates/averages outputs.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize multi-head GAT layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            concat: If True, concatenate heads; else average
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.concat = concat
        
        # Create attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features,
                out_features,
                dropout=dropout,
                alpha=alpha,
                concat=True
            ) for _ in range(num_heads)
        ])
        
        # Output dimension
        if concat:
            self.out_dim = out_features * num_heads
        else:
            self.out_dim = out_features
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multi-head attention.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weights: Optional edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Run all attention heads
        head_outputs = [att(x, edge_index, edge_weights) for att in self.attentions]
        
        # Concatenate or average
        if self.concat:
            return torch.cat(head_outputs, dim=-1)
        else:
            return torch.stack(head_outputs, dim=0).mean(dim=0)


class WorldGraphEncoder(nn.Module):
    """
    Complete GAT-based world graph encoder.
    
    Encodes the entire campus graph using multi-layer GAT with attention.
    Provides node embeddings that capture:
    - Local neighborhood structure
    - Global graph topology
    - Semantic relationships (POI categories)
    - Spatial relationships (geographic proximity)
    
    USAGE:
    1. Initialize with graph structure once
    2. Call forward() during training to get node embeddings
    3. Embeddings can be indexed by node IDs for trajectory encoding
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_categories: int = 7,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_node_features: bool = True,
        use_residual: bool = True,
    ):
        """
        Initialize world graph encoder.
        
        Args:
            num_nodes: Total number of nodes in graph
            num_categories: Number of POI categories
            node_feature_dim: Initial node feature dimension
            hidden_dim: Hidden dimension for GAT layers
            output_dim: Final output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads per layer
            dropout: Dropout rate
            use_node_features: Whether to use node features (category, spatial)
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_categories = num_categories
        self.node_feature_dim = node_feature_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_node_features = use_node_features
        self.use_residual = use_residual
        
        # ================================================================
        # INITIAL NODE FEATURE ENCODING
        # ================================================================
        if use_node_features:
            # Category embedding
            self.category_embedding = nn.Embedding(num_categories + 1, node_feature_dim // 2)
            
            # Spatial feature encoder (x, y coordinates)
            self.spatial_encoder = nn.Sequential(
                nn.Linear(2, node_feature_dim // 4),
                nn.ReLU(),
                nn.Linear(node_feature_dim // 4, node_feature_dim // 4)
            )
            
            # Combine category + spatial + learnable base embedding
            self.base_embedding = nn.Embedding(num_nodes, node_feature_dim // 4)
            initial_dim = node_feature_dim
        else:
            # Just use learnable base embeddings
            self.base_embedding = nn.Embedding(num_nodes, node_feature_dim)
            initial_dim = node_feature_dim
        
        # ================================================================
        # GAT LAYERS
        # ================================================================
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First GAT layer
        first_layer = MultiHeadGATLayer(
            in_features=initial_dim,
            out_features=hidden_dim // num_heads,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.gat_layers.append(first_layer)
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Middle GAT layers
        for _ in range(num_layers - 2):
            layer = MultiHeadGATLayer(
                in_features=hidden_dim,
                out_features=hidden_dim // num_heads,
                num_heads=num_heads,
                dropout=dropout,
                concat=True
            )
            self.gat_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Final GAT layer (average heads instead of concat)
        if num_layers > 1:
            final_layer = MultiHeadGATLayer(
                in_features=hidden_dim,
                out_features=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=False  # Average for final layer
            )
            self.gat_layers.append(final_layer)
            self.layer_norms.append(nn.LayerNorm(output_dim))
        
        # Projection to output dimension if needed
        if num_layers == 1:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
         # Graph structure (set via set_graph_structure)
        self.register_buffer('edge_index', torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer('edge_weights', torch.zeros(0, dtype=torch.float))
        
        # Cache for full graph embeddings (computed once, reused)
        self._cached_embeddings = None
        self._cache_valid = False
    
    def set_graph_structure(
        self,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ):
        """
        Set the graph structure (must be called before forward pass).
        
        Args:
            edge_index: [2, num_edges] tensor of edge connections
            edge_weights: [num_edges] optional edge weights
        """
        self.edge_index = edge_index
        if edge_weights is not None:
            self.edge_weights = edge_weights
        else:
            self.edge_weights = torch.ones(edge_index.size(1), dtype=torch.float)
        
        # Invalidate cache when graph structure changes
        self._cache_valid = False
        self._cached_embeddings = None
    
    def invalidate_cache(self):
        """Invalidate cached embeddings (call after updating model parameters)."""
        self._cache_valid = False
        self._cached_embeddings = None
    
    def precompute_embeddings(
        self,
        categories: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None
    ):
        """
        Precompute and cache embeddings for ALL nodes in the graph.
        
        This should be called:
        1. After setting graph structure
        2. After loading pretrained weights
        3. Optionally during training if you want to freeze GAT
        
        Args:
            categories: [num_nodes] category indices for all nodes
            spatial_coords: [num_nodes, 2] spatial coordinates for all nodes
        """
        with torch.no_grad():
            self._cached_embeddings = self.forward(None, categories, spatial_coords)
            self._cache_valid = True
        
        print(f"✅ Precomputed embeddings for {self.num_nodes} nodes: {self._cached_embeddings.shape}")
    
    def train(self, mode: bool = True):
        """
        Override train() to invalidate cache when entering training mode.
        
        During training, we want fresh gradients, so cache should be disabled.
        """
        super().train(mode)
        if mode:
            # Entering training mode - invalidate cache
            self._cache_valid = False
        return self
    
    def encode_node_features(
        self,
        node_indices: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode initial node features.
        
        Args:
            node_indices: [num_nodes] node indices (default: all nodes)
            categories: [num_nodes] category indices
            spatial_coords: [num_nodes, 2] spatial coordinates
            
        Returns:
            Initial node features [num_nodes, node_feature_dim]
        """
        if node_indices is None:
            node_indices = torch.arange(self.num_nodes, device=self.base_embedding.weight.device)
        
        # Base embedding
        base_emb = self.base_embedding(node_indices)
        
        if not self.use_node_features:
            return base_emb
        
        # Category embedding
        if categories is None:
            categories = torch.zeros_like(node_indices)
        cat_emb = self.category_embedding(categories)
        
        # Spatial encoding
        if spatial_coords is None:
            spatial_coords = torch.zeros(len(node_indices), 2, device=base_emb.device)
        spatial_emb = self.spatial_encoder(spatial_coords)
        
        # Concatenate all features
        node_features = torch.cat([base_emb, cat_emb, spatial_emb], dim=-1)
        
        return node_features
    
    def forward(
        self,
        node_indices: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Encode graph using GAT.
        
        Args:
            node_indices: [num_nodes] specific nodes to encode (default: all)
            categories: [num_nodes] category indices
            spatial_coords: [num_nodes, 2] spatial coordinates
            return_all_layers: If True, return embeddings from all layers
            
        Returns:
            Node embeddings [num_nodes, output_dim]
            If return_all_layers: List of embeddings from each layer
        """
        # Encode initial node features
        x = self.encode_node_features(node_indices, categories, spatial_coords)
        
        if node_indices is None:
            num_nodes = self.num_nodes
        else:
            num_nodes = len(node_indices)
        
        # Process through GAT layers
        layer_outputs = []
        
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # Store input for residual
            x_input = x
            
            # GAT layer
            x = gat_layer(x, self.edge_index, self.edge_weights)
            
            # Layer norm
            x = layer_norm(x)
            
            # Residual connection (if dimensions match and enabled)
            if self.use_residual and x.shape == x_input.shape:
                x = x + x_input
            
            # Dropout (except last layer)
            if i < len(self.gat_layers) - 1:
                x = self.dropout(x)
            
            layer_outputs.append(x)
        
        # Final projection
        x = self.output_proj(x)
        
        if return_all_layers:
            return layer_outputs
        
        return x
    
    def get_embeddings(
        self,
        node_indices: torch.Tensor,
        categories: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Get embeddings for specific nodes (convenience method).
        
        Args:
            node_indices: [batch, seq_len] or [num_nodes] node indices
            categories: Optional category indices
            spatial_coords: Optional spatial coordinates
            use_cache: If True and cache is valid, use cached embeddings
            
        Returns:
            Node embeddings with same shape as node_indices + [output_dim]
        """
        original_shape = node_indices.shape
        node_indices_flat = node_indices.reshape(-1)
        
        # Get full graph embeddings (use cache if available and training mode allows)
        if use_cache and self._cache_valid and self._cached_embeddings is not None:
            # Use cached embeddings (efficient!)
            all_embeddings = self._cached_embeddings
        else:
            # Compute fresh embeddings
            all_embeddings = self.forward(None, categories, spatial_coords)
            
            # Update cache if in eval mode
            if not self.training and use_cache:
                self._cached_embeddings = all_embeddings.detach()
                self._cache_valid = True
        
        # Index by node_indices
        embeddings = all_embeddings[node_indices_flat]
        
        # Reshape back
        embeddings = embeddings.reshape(*original_shape, -1)
        
        return embeddings


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_edge_index_from_networkx(graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert NetworkX graph to PyTorch Geometric edge_index format.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Tuple of (edge_index, edge_weights)
        - edge_index: [2, num_edges] source and target node pairs
        - edge_weights: [num_edges] edge weights (1.0 if no weight attribute)
    """
    # Create node to index mapping
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Build edge list
    edges = []
    weights = []
    
    for src, tgt, data in graph.edges(data=True):
        src_idx = node_to_idx[src]
        tgt_idx = node_to_idx[tgt]
        
        # Add both directions (undirected graph)
        edges.append([src_idx, tgt_idx])
        edges.append([tgt_idx, src_idx])
        
        # Get weight
        weight = data.get('weight', 1.0)
        weights.append(weight)
        weights.append(weight)
    
    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, num_edges]
    edge_weights = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_weights


def extract_node_features_from_networkx(
    graph: nx.Graph,
    category_to_idx: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract node features (categories and spatial coords) from NetworkX graph.
    
    Args:
        graph: NetworkX graph with 'Category', 'x', 'y' node attributes
        category_to_idx: Mapping from category names to indices
        
    Returns:
        Tuple of (categories, spatial_coords)
        - categories: [num_nodes] category indices
        - spatial_coords: [num_nodes, 2] (x, y) coordinates
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    
    categories = torch.zeros(num_nodes, dtype=torch.long)
    spatial_coords = torch.zeros(num_nodes, 2, dtype=torch.float)
    
    for idx, node in enumerate(nodes):
        data = graph.nodes[node]
        
        # Category
        cat_name = data.get('Category', 'other')
        categories[idx] = category_to_idx.get(cat_name, 0)
        
        # Spatial coordinates
        x = float(data.get('x', 0.0))
        y = float(data.get('y', 0.0))
        spatial_coords[idx] = torch.tensor([x, y])
    
    return categories, spatial_coords


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test world graph encoder."""
    import networkx as nx
    
    print("=" * 80)
    print("Testing World Graph Encoder (GAT)")
    print("=" * 80)
    
    # Create synthetic graph
    num_nodes = 100
    G = nx.erdos_renyi_graph(num_nodes, 0.05)
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Avg degree: {sum(dict(G.degree()).values()) / num_nodes:.2f}")
    
    # Convert to edge_index
    edge_index, edge_weights = build_edge_index_from_networkx(G)
    print(f"\n✅ Converted to edge_index: {edge_index.shape}")
    
    # Create encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    encoder = WorldGraphEncoder(
        num_nodes=num_nodes,
        num_categories=7,
        node_feature_dim=64,
        hidden_dim=128,
        output_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_node_features=True,
        use_residual=True
    ).to(device)
    
    # Set graph structure
    encoder.set_graph_structure(edge_index.to(device), edge_weights.to(device))
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n🔢 Model parameters: {total_params:,}")
    
    # Forward pass
    print(f"\n🔄 Running forward pass...")
    embeddings = encoder()
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Output mean: {embeddings.mean().item():.4f}")
    print(f"   Output std: {embeddings.std().item():.4f}")
    
    # Test getting specific node embeddings
    node_indices = torch.tensor([0, 5, 10, 15, 20], device=device)
    specific_embeddings = encoder.get_embeddings(node_indices)
    print(f"\n✅ Specific node embeddings: {specific_embeddings.shape}")
    
    # Test batch indexing
    batch_indices = torch.randint(0, num_nodes, (4, 10), device=device)
    batch_embeddings = encoder.get_embeddings(batch_indices)
    print(f"✅ Batch embeddings: {batch_embeddings.shape}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
