import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_controller.world_graph import WorldGraph


# -----------------------------
# World Graph Encoder (GAT)
# -----------------------------
class WorldGraphEncoder(nn.Module):
    """
    Encodes the static world graph using Graph Attention Networks (GAT).
    Processes spatial features of the campus map to create node embeddings.
    """
    def __init__(self, node_feat_dim, hidden_dim, output_dim, n_layers=2, n_heads=4, dropout=0.1):
        """
        Args:
            node_feat_dim (int): Dimension of input node features (e.g., lat, lon, category)
            hidden_dim (int): Hidden dimension for intermediate layers
            output_dim (int): Final output embedding dimension
            n_layers (int): Number of GAT layers
            n_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Node feature projection
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        
        # GAT convolution layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                dropout=dropout,
                concat=True
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for i in range(n_layers - 2):
            self.conv_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final layer (average heads instead of concatenating)
        if n_layers > 1:
            self.conv_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=output_dim,
                    heads=n_heads,
                    dropout=dropout,
                    concat=False  # Average attention heads
                )
            )
        else:
            # Single layer case - directly output to output_dim
            self.conv_layers[0] = GATConv(
                in_channels=hidden_dim,
                out_channels=output_dim,
                heads=n_heads,
                dropout=dropout,
                concat=False
            )
            self.batch_norms = nn.ModuleList()
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the graph encoder.
        
        Args:
            x: (num_nodes, node_feat_dim) - Node features
            edge_index: (2, num_edges) - Graph connectivity in COO format
            batch: (num_nodes,) - Batch assignment for nodes (optional)
            
        Returns:
            Encoded graph representation of shape (batch_size, output_dim)
        """
        # Project node features
        x = self.node_proj(x)
        x = F.relu(x)
        
        # Apply GAT convolutions with batch normalization
        for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Final convolution layer
        x = self.conv_layers[-1](x, edge_index)
        
        # Global pooling (mean over all nodes)
        if batch is not None:
            # If we have batch information, pool per graph
            x = global_mean_pool(x, batch)
        else:
            # Otherwise, take mean over all nodes
            x = x.mean(dim=0, keepdim=True)
        
        # Final projection
        x = self.output_proj(x)
        x = F.relu(x)
        
        return x
    
    def get_attention_weights(self, x, edge_index):
        """
        Get attention weights from GAT layers for visualization.
        
        Args:
            x: (num_nodes, node_feat_dim) - Node features
            edge_index: (2, num_edges) - Graph connectivity
            
        Returns:
            List of tuples (edge_index, attention_weights) for each layer
        """
        attention_weights = []
        
        # Project node features
        x = self.node_proj(x)
        x = F.relu(x)
        
        # Get attention weights from each layer
        for i, conv in enumerate(self.conv_layers):
            x, (edge_idx, alpha) = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_idx, alpha))
            
            if i < len(self.conv_layers) - 1:
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        return attention_weights


# -----------------------------
# Graph Data Preparation Utilities
# -----------------------------
class GraphDataPreparator:
    """
    Utility class to convert NetworkX graph to PyTorch Geometric format.
    """
    def __init__(self, world_graph):
        """
        Args:
            world_graph: WorldGraph object containing the campus map
        """
        self.world_graph = world_graph
        self.G = world_graph.G
        
        # Create node index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(self.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Category mapping
        self.category_to_idx = {
            'home': 0,
            'study': 1,
            'food': 2,
            'leisure': 3,
            'errands': 4,
            'health': 5,
            'None': 6
        }
    
    def prepare_node_features(self):
        """
        Extract node features from the graph.
        Returns tensor of shape (num_nodes, feature_dim)
        """
        node_features = []
        
        for node in self.G.nodes():
            data = self.G.nodes[node]
            
            # Extract features: [lat, lon, category_one_hot (7 dims), has_poi (1), opening_hours (2)]
            lat = float(data.get('y', 0.0))
            lon = float(data.get('x', 0.0))
            
            # Normalize coordinates (simple normalization)
            lat_norm = (lat - 32.8) / 0.1  # Approximate UCSD range
            lon_norm = (lon + 117.2) / 0.1
            
            # Category one-hot encoding
            category = data.get('Category', 'None')
            category_idx = self.category_to_idx.get(category, 6)
            category_onehot = [0] * 7
            category_onehot[category_idx] = 1
            
            # Has POI
            has_poi = 1.0 if 'poi_names' in data else 0.0
            
            # Opening hours (normalized to [0, 1])
            opening_hours = data.get('opening_hours', None)
            if opening_hours and isinstance(opening_hours, dict):
                open_hour = opening_hours.get('open', 0) / 24.0
                close_hour = opening_hours.get('close', 24) / 24.0
            else:
                open_hour = 0.0
                close_hour = 1.0
            
            # Combine all features
            features = [lat_norm, lon_norm] + category_onehot + [has_poi, open_hour, close_hour]
            node_features.append(features)
        
        return torch.tensor(node_features, dtype=torch.float32)
    
    def prepare_edge_index(self):
        """
        Convert edge list to PyTorch Geometric format.
        Returns tensor of shape (2, num_edges)
        """
        edge_list = []
        
        for u, v in self.G.edges():
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            
            # Add both directions for undirected graph
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def prepare_graph_data(self):
        """
        Prepare complete graph data for model input.
        
        Returns:
            Dict with 'x' (node features) and 'edge_index' (connectivity)
        """
        node_features = self.prepare_node_features()
        edge_index = self.prepare_edge_index()
        
        return {
            'x': node_features,
            'edge_index': edge_index
        }


# -----------------------------
# Main Function for Testing
# -----------------------------
def main():
    """Test the World Graph Encoder with UCSD campus data."""
    print("Testing World Graph Encoder with UCSD campus data...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load UCSD campus graph
    print("\nLoading UCSD campus graph...")
    graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Prepare graph data
    print("\nPreparing graph data...")
    data_preparator = GraphDataPreparator(world_graph)
    graph_data = data_preparator.prepare_graph_data()
    
    print(f"Node features shape: {graph_data['x'].shape}")
    print(f"Edge index shape: {graph_data['edge_index'].shape}")
    
    # Move data to device
    for key in graph_data:
        graph_data[key] = graph_data[key].to(device)
    
    # Initialize encoder
    node_feat_dim = graph_data['x'].shape[1]  # Should be 12 (2 coords + 7 categories + 1 poi + 2 hours)
    
    encoder = WorldGraphEncoder(
        node_feat_dim=node_feat_dim,
        hidden_dim=128,
        output_dim=64,
        n_layers=3,
        n_heads=4,
        dropout=0.1
    ).to(device)
    
    print(f"\nEncoder initialized with:")
    print(f"  - Input dimension: {node_feat_dim}")
    print(f"  - Hidden dimension: 128")
    print(f"  - Output dimension: 64")
    print(f"  - Number of layers: 3")
    print(f"  - Attention heads: 4")
    
    # Test forward pass
    print("\nTesting forward pass...")
    encoder.eval()
    with torch.no_grad():
        output = encoder(graph_data['x'], graph_data['edge_index'])
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0][:10]}")
    
    # Test attention weights extraction
    print("\nTesting attention weights extraction...")
    with torch.no_grad():
        attention_weights = encoder.get_attention_weights(graph_data['x'], graph_data['edge_index'])
        print(f"Number of GAT layers: {len(attention_weights)}")
        for i, (edge_idx, alpha) in enumerate(attention_weights):
            print(f"  Layer {i+1}: {alpha.shape[0]} edges, {alpha.shape[1]} attention heads")
    
    print("\n✅ World Graph Encoder test completed successfully!")
    
    # Print model summary
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()
