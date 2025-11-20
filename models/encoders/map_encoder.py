"""
World Graph Encoder - GAT-based graph encoding with Node2Vec embeddings.

This module provides a Graph Attention Network (GAT) encoder for the world graph.
Uses pre-computed Node2Vec embeddings as input to ensure a shared embedding space
with the trajectory encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class WorldGraphEncoder(nn.Module):
    """
    Encodes the static world graph using Graph Attention Networks (GAT).
    
    Uses pre-computed Node2Vec embeddings as input, ensuring the map encoder
    and trajectory encoder operate in the same embedding space before fusion.
    """
    def __init__(self, node_emb_dim, hidden_dim, output_dim, n_layers=2, n_heads=4, dropout=0.1):
        """
        Args:
            node_emb_dim (int): Dimension of pre-computed node embeddings (from Node2Vec)
            hidden_dim (int): Hidden dimension for intermediate layers
            output_dim (int): Final output embedding dimension
            n_layers (int): Number of GAT layers
            n_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Node embedding projection
        self.node_proj = nn.Linear(node_emb_dim, hidden_dim)
        
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
        
    def forward(self, node_embeddings, edge_index, batch=None):
        """
        Forward pass through the graph encoder.
        
        Args:
            node_embeddings: (num_nodes, node_emb_dim) - Pre-computed node embeddings from Node2Vec
            edge_index: (2, num_edges) - Graph connectivity in COO format
            batch: (num_nodes,) - Batch assignment for nodes (optional)
            
        Returns:
            Encoded graph representation of shape (batch_size, output_dim)
        """
        # Project node embeddings
        x = self.node_proj(node_embeddings)
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
    
    def get_attention_weights(self, node_embeddings, edge_index):
        """
        Get attention weights from GAT layers for visualization.
        
        Args:
            node_embeddings: (num_nodes, node_emb_dim) - Pre-computed node embeddings
            edge_index: (2, num_edges) - Graph connectivity
            
        Returns:
            List of tuples (edge_index, attention_weights) for each layer
        """
        attention_weights = []
        
        # Project node embeddings
        x = self.node_proj(node_embeddings)
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
