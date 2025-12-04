"""
ToM Graph Encoder - Fusion Model with Node2Vec Embeddings

This module combines the Trajectory Encoder (Transformer-based) and the World Graph Encoder (GAT-based)
to create a unified representation for Theory of Mind (ToM) trajectory prediction.

Both encoders use pre-computed Node2Vec embeddings to ensure they operate in a shared embedding space.

The fusion model processes:
1. Agent trajectories: temporal sequences of visited locations (via Transformer on Node2Vec embeddings)
2. World graph structure: spatial layout of the environment (via GAT on Node2Vec embeddings)

The two encodings are concatenated and fused through a linear layer to produce
a combined representation that captures both temporal behavioral patterns and
spatial environmental structure.
"""

import torch
import torch.nn as nn

from models.fusion_encoders_preprocessing.trajectory_encoder import TrajectoryEncoder
from models.fusion_encoders_preprocessing.map_encoder import WorldGraphEncoder


# -----------------------------
# ToM Graph Encoder (Combined)
# -----------------------------
class ToMGraphEncoder(nn.Module):
    """
    Combined encoder that processes both trajectory and world graph data using Node2Vec embeddings.
    
    This model fuses temporal trajectory patterns with spatial graph structure
    to create comprehensive representations for Theory of Mind tasks.
    
    Both encoders use pre-computed Node2Vec embeddings to ensure a shared embedding space.
    
    Architecture:
        - Trajectory branch: Transformer encoder on Node2Vec embeddings
        - World graph branch: GAT encoder on Node2Vec embeddings
        - Fusion layer: Combines both representations
    
    The model can be used for various ToM tasks such as:
        - Goal prediction
        - Next location prediction
        - Preference inference
        - Belief state estimation
    """
    
    def __init__(
        self, 
        node_emb_dim,         # Dimension of Node2Vec embeddings (shared by both encoders)
        hidden_dim,
        num_agents,           # Number of unique agents in the dataset
        output_dim=64,
        n_layers=2, 
        n_heads=4, 
        dropout=0.1
    ):
        """
        Initialize the ToM Graph Encoder.
        
        Args:
            node_emb_dim (int): Dimension of Node2Vec embeddings (e.g., 128)
            hidden_dim (int): Hidden dimension for both encoders
            num_agents (int): Total number of unique agents in the dataset
            output_dim (int): Final output dimension after fusion (default: 64)
            n_layers (int): Number of layers in both encoders (default: 2)
            n_heads (int): Number of attention heads (default: 4)
            dropout (float): Dropout rate (default: 0.1)
        """
        super().__init__()
        
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.output_dim = output_dim
        
        # Trajectory encoder (Transformer-based with agent embeddings)
        self.trajectory_encoder = TrajectoryEncoder(
            node_emb_dim=node_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_agents=num_agents,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # World graph encoder (GAT-based)
        self.world_encoder = WorldGraphEncoder(
            node_emb_dim=node_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Fusion layer - combines trajectory and world graph encodings
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, trajectory_data, graph_data):
        """
        Forward pass through the combined encoder.
        
        Args:
            trajectory_data (dict): Dictionary containing:
                - 'node_embeddings': (batch_size, seq_len, node_emb_dim) - Node2Vec embeddings for trajectory
                - 'hour': (batch_size,) - Hour of day (0-23) when trajectory occurred
                - 'agent_id': (batch_size,) - Agent ID for each trajectory
                - 'mask': (batch_size, seq_len) - Padding mask (optional)
                
            graph_data (dict): Dictionary containing:
                - 'node_embeddings': (num_nodes, node_emb_dim) - Node2Vec embeddings for all graph nodes
                - 'edge_index': (2, num_edges) - Edge connectivity
                - 'batch': (num_nodes,) - Batch assignment (optional)
        
        Returns:
            torch.Tensor: Fused encoding of shape (batch_size, output_dim)
        """
        # Encode trajectory using Transformer (with agent embeddings)
        traj_encoding = self.trajectory_encoder(
            trajectory_data['node_embeddings'],
            trajectory_data['hour'],
            trajectory_data['agent_id'],
            trajectory_data.get('mask', None)
        )  # (batch_size, output_dim)
        
        # Encode world graph using GAT
        graph_encoding = self.world_encoder(
            graph_data['node_embeddings'],
            graph_data['edge_index'],
            graph_data.get('batch', None)
        )  # (1, output_dim) or (batch_size, output_dim)
        
        # Ensure same batch size by repeating graph encoding if necessary
        # (World graph is typically the same for all trajectories in a batch)
        if traj_encoding.shape[0] != graph_encoding.shape[0]:
            graph_encoding = graph_encoding.expand(traj_encoding.shape[0], -1)
        
        # Concatenate encodings
        combined = torch.cat([traj_encoding, graph_encoding], dim=-1)  # (batch_size, output_dim * 2)
        
        # Fuse through linear layers
        fused_encoding = self.fusion(combined)  # (batch_size, output_dim)
        
        return fused_encoding
    
    def get_trajectory_encoding(self, trajectory_data):
        """
        Get only the trajectory encoding (useful for analysis).
        
        Args:
            trajectory_data (dict): Trajectory data dictionary with node_embeddings
            
        Returns:
            torch.Tensor: Trajectory encoding of shape (batch_size, output_dim)
        """
        return self.trajectory_encoder(
            trajectory_data['node_embeddings'],
            trajectory_data['hour'],
            trajectory_data.get('mask', None)
        )
    
    def get_graph_encoding(self, graph_data):
        """
        Get only the world graph encoding (useful for analysis).
        
        Args:
            graph_data (dict): Graph data dictionary with node_embeddings
            
        Returns:
            torch.Tensor: Graph encoding of shape (1, output_dim)
        """
        return self.world_encoder(
            graph_data['node_embeddings'],
            graph_data['edge_index'],
            graph_data.get('batch', None)
        )
 