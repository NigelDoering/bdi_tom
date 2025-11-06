"""
ToM Graph Encoder - Fusion Model

This module combines the Trajectory Encoder (Transformer-based) and the World Graph Encoder (GAT-based)
to create a unified representation for Theory of Mind (ToM) trajectory prediction.

The fusion model processes:
1. Agent trajectories: temporal sequences of visited locations (via Transformer)
2. World graph structure: spatial layout of the environment (via GAT)

The two encodings are concatenated and fused through a linear layer to produce
a combined representation that captures both temporal behavioral patterns and
spatial environmental structure.
"""

import torch
import torch.nn as nn
import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoders.trajectory_encoder import TrajectoryEncoder
from models.encoders.map_encoder import WorldGraphEncoder
from models.encoders.map_encoder import GraphDataPreparator


# -----------------------------
# ToM Graph Encoder (Combined)
# -----------------------------
class ToMGraphEncoder(nn.Module):
    """
    Combined encoder that processes both trajectory and world graph data.
    
    This model fuses temporal trajectory patterns with spatial graph structure
    to create comprehensive representations for Theory of Mind tasks.
    
    Architecture:
        - Trajectory branch: Transformer encoder for temporal sequences
        - World graph branch: GAT encoder for spatial structure
        - Fusion layer: Combines both representations
    
    The model can be used for various ToM tasks such as:
        - Goal prediction
        - Next location prediction
        - Preference inference
        - Belief state estimation
    """
    
    def __init__(
        self, 
        num_nodes,
        graph_node_feat_dim,  # Input dimension for world graph features
        traj_node_emb_dim,    # Embedding dimension for trajectory nodes
        hidden_dim, 
        output_dim=64,
        n_layers=2, 
        n_heads=4, 
        dropout=0.1
    ):
        """
        Initialize the ToM Graph Encoder.
        
        Args:
            num_nodes (int): Total number of nodes in the world graph
            graph_node_feat_dim (int): Dimension of input node features for world graph (e.g., 12)
            traj_node_emb_dim (int): Dimension of learned node embeddings for trajectories (e.g., 32)
            hidden_dim (int): Hidden dimension for both encoders
            output_dim (int): Final output dimension after fusion (default: 64)
            n_layers (int): Number of layers in both encoders (default: 2)
            n_heads (int): Number of attention heads (default: 4)
            dropout (float): Dropout rate (default: 0.1)
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.graph_node_feat_dim = graph_node_feat_dim
        self.traj_node_emb_dim = traj_node_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Trajectory encoder (Transformer-based)
        self.trajectory_encoder = TrajectoryEncoder(
            num_nodes=num_nodes,
            node_feat_dim=traj_node_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # World graph encoder (GAT-based)
        self.world_encoder = WorldGraphEncoder(
            node_feat_dim=graph_node_feat_dim,
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
                - 'node_ids': (batch_size, seq_len) - Node IDs in trajectory
                - 'hour': (batch_size,) - Hour of day (0-23) when trajectory occurred
                - 'mask': (batch_size, seq_len) - Padding mask (optional)
                
            graph_data (dict): Dictionary containing:
                - 'x': (num_nodes, node_feat_dim) - Node features
                - 'edge_index': (2, num_edges) - Edge connectivity
                - 'batch': (num_nodes,) - Batch assignment (optional)
        
        Returns:
            torch.Tensor: Fused encoding of shape (batch_size, output_dim)
        """
        # Encode trajectory using Transformer
        traj_encoding = self.trajectory_encoder(
            trajectory_data['node_ids'],
            trajectory_data['hour'],
            trajectory_data.get('mask', None)
        )  # (batch_size, output_dim)
        
        # Encode world graph using GAT
        graph_encoding = self.world_encoder(
            graph_data['x'],
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
            trajectory_data (dict): Trajectory data dictionary
            
        Returns:
            torch.Tensor: Trajectory encoding of shape (batch_size, output_dim)
        """
        return self.trajectory_encoder(
            trajectory_data['node_ids'],
            trajectory_data['hour'],
            trajectory_data.get('mask', None)
        )
    
    def get_graph_encoding(self, graph_data):
        """
        Get only the world graph encoding (useful for analysis).
        
        Args:
            graph_data (dict): Graph data dictionary
            
        Returns:
            torch.Tensor: Graph encoding of shape (1, output_dim)
        """
        return self.world_encoder(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data.get('batch', None)
        )


# -----------------------------
# Prediction Head for ToM Tasks
# -----------------------------
class ToMPredictionHead(nn.Module):
    """
    Prediction head that sits on top of the ToM Graph Encoder.
    Can be adapted for various ToM prediction tasks.
    """
    
    def __init__(self, input_dim, num_nodes, hidden_dim=128, task='goal_prediction'):
        """
        Initialize the prediction head.
        
        Args:
            input_dim (int): Input dimension (output_dim from ToM encoder)
            num_nodes (int): Number of nodes in the graph (for goal prediction)
            hidden_dim (int): Hidden dimension for MLP (default: 128)
            task (str): Type of task ('goal_prediction', 'next_step', 'preference')
        """
        super().__init__()
        
        self.task = task
        self.num_nodes = num_nodes
        
        if task == 'goal_prediction':
            # Predict probability distribution over goal nodes
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_nodes)
            )
        elif task == 'next_step':
            # Predict next node in trajectory
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_nodes)
            )
        elif task == 'preference':
            # Predict preference distribution over categories (e.g., 7 categories)
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 7)  # Assuming 7 POI categories
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, encoding):
        """
        Forward pass through prediction head.
        
        Args:
            encoding (torch.Tensor): Fused encoding from ToM encoder (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Task-specific predictions
                - goal_prediction: (batch_size, num_nodes) logits
                - next_step: (batch_size, num_nodes) logits
                - preference: (batch_size, 7) logits
        """
        return self.head(encoding)


# -----------------------------
# Complete ToM Model
# -----------------------------
class ToMModel(nn.Module):
    """
    Complete Theory of Mind model combining encoder and prediction head.
    """
    
    def __init__(
        self,
        num_nodes,
        graph_node_feat_dim=12,  # Default for UCSD graph
        traj_node_emb_dim=32,
        hidden_dim=128,
        output_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        task='goal_prediction'
    ):
        """
        Initialize complete ToM model.
        
        Args:
            num_nodes (int): Total number of nodes in graph
            graph_node_feat_dim (int): World graph node feature dimension (default: 12)
            traj_node_emb_dim (int): Trajectory node embedding dimension (default: 32)
            hidden_dim (int): Hidden dimension (default: 128)
            output_dim (int): Encoder output dimension (default: 64)
            n_layers (int): Number of encoder layers (default: 2)
            n_heads (int): Number of attention heads (default: 4)
            dropout (float): Dropout rate (default: 0.1)
            task (str): Prediction task type (default: 'goal_prediction')
        """
        super().__init__()
        
        # Encoder
        self.encoder = ToMGraphEncoder(
            num_nodes=num_nodes,
            graph_node_feat_dim=graph_node_feat_dim,
            traj_node_emb_dim=traj_node_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Prediction head
        self.prediction_head = ToMPredictionHead(
            input_dim=output_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            task=task
        )
        
        self.task = task
    
    def forward(self, trajectory_data, graph_data):
        """
        Forward pass through complete model.
        
        Args:
            trajectory_data (dict): Trajectory data
            graph_data (dict): World graph data
            
        Returns:
            torch.Tensor: Task-specific predictions
        """
        # Encode
        encoding = self.encoder(trajectory_data, graph_data)
        
        # Predict
        predictions = self.prediction_head(encoding)
        
        return predictions
    
    def get_encoding(self, trajectory_data, graph_data):
        """
        Get the fused encoding without prediction.
        
        Args:
            trajectory_data (dict): Trajectory data
            graph_data (dict): World graph data
            
        Returns:
            torch.Tensor: Fused encoding (batch_size, output_dim)
        """
        return self.encoder(trajectory_data, graph_data)


# -----------------------------
# Testing / Demo
# -----------------------------
def main():
    """
    Test the ToM Graph Encoder with sample data.
    """
    print("=" * 80)
    print("Testing ToM Graph Encoder (Fusion Model)")
    print("=" * 80)
    
    # Load the world graph
    from graph_controller.world_graph import WorldGraph
    graph_path = "data/processed/ucsd_walk_full.graphml"
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return
    
    print(f"\n1. Loading world graph from {graph_path}...")
    import networkx as nx
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G)
    num_nodes = len(world_graph.G.nodes())
    print(f"   ✓ Loaded graph with {num_nodes} nodes")
    
    # Prepare graph data for GAT encoder

    graph_prep = GraphDataPreparator(world_graph)
    graph_data = graph_prep.prepare_graph_data()
    print(f"   ✓ Prepared graph data: {graph_data['x'].shape[0]} nodes, {graph_data['edge_index'].shape[1]} edges")
    
    # Load trajectory data
    from models.encoders.trajectory_encoder import TrajectoryDataPreparator
    traj_path = "data/simulation_data/run_8/trajectories/all_trajectories.json"
    
    if not os.path.exists(traj_path):
        print(f"\nWarning: Trajectory file not found at {traj_path}")
        print("Creating synthetic trajectory data for testing...")
        
        # Create synthetic trajectory
        node_ids = torch.randint(1, num_nodes, (2, 20))  # 2 trajectories, 20 steps each
        timestamps = torch.linspace(0, 23, 20).unsqueeze(0).repeat(2, 1)
        mask = torch.ones(2, 20)
        
        trajectory_data = {
            'node_ids': node_ids,
            'timestamps': timestamps,
            'mask': mask
        }
        print(f"   ✓ Created synthetic trajectories: {node_ids.shape}")
    else:
        print(f"\n2. Loading trajectories from {traj_path}...")
        with open(traj_path, 'r') as f:
            trajectories = json.load(f)
        
        # Create node mapping
        node_list = list(world_graph.G.nodes())
        node_to_idx = {node: idx + 1 for idx, node in enumerate(node_list)}  # 0 reserved for padding
        node_to_idx[0] = 0  # Add explicit padding
        
        traj_prep = TrajectoryDataPreparator(node_to_idx)
        
        # Prepare first 2 trajectories as batch
        # Each agent has a list of trajectories, we'll take the first trajectory from first 2 agents
        agent_ids = list(trajectories.keys())[:2]
        sample_trajs = [trajectories[agent_id][0] for agent_id in agent_ids]  # Full trajectory dict with 'path' and 'hour'
        trajectory_data = traj_prep.prepare_batch(sample_trajs)
        print(f"   ✓ Prepared {len(sample_trajs)} trajectories from agents: {agent_ids}")
    
    # Initialize model
    print("\n3. Initializing ToM Graph Encoder...")
    model = ToMGraphEncoder(
        num_nodes=num_nodes + 1,  # +1 for padding
        graph_node_feat_dim=12,  # UCSD graph has 12-dim node features
        traj_node_emb_dim=32,
        hidden_dim=128,
        output_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    traj_params = sum(p.numel() for p in model.trajectory_encoder.parameters())
    graph_params = sum(p.numel() for p in model.world_encoder.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    
    print(f"   ✓ Model initialized")
    print(f"   - Trajectory encoder parameters: {traj_params:,}")
    print(f"   - World graph encoder parameters: {graph_params:,}")
    print(f"   - Fusion layer parameters: {fusion_params:,}")
    print(f"   - Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        # Get individual encodings
        traj_encoding = model.get_trajectory_encoding(trajectory_data)
        graph_encoding = model.get_graph_encoding(graph_data)
        
        print(f"   ✓ Trajectory encoding shape: {traj_encoding.shape}")
        print(f"   ✓ Graph encoding shape: {graph_encoding.shape}")
        
        # Get fused encoding
        fused_encoding = model(trajectory_data, graph_data)
        print(f"   ✓ Fused encoding shape: {fused_encoding.shape}")
    
    # Test complete ToM model with prediction head
    print("\n5. Testing complete ToM model with goal prediction...")
    tom_model = ToMModel(
        num_nodes=num_nodes + 1,
        graph_node_feat_dim=12,
        traj_node_emb_dim=32,
        hidden_dim=128,
        output_dim=64,
        n_layers=2,
        n_heads=4,
        task='goal_prediction'
    )
    
    tom_total_params = sum(p.numel() for p in tom_model.parameters())
    print(f"   ✓ Complete model initialized")
    print(f"   - Total parameters: {tom_total_params:,}")
    
    tom_model.eval()
    with torch.no_grad():
        predictions = tom_model(trajectory_data, graph_data)
        print(f"   ✓ Goal predictions shape: {predictions.shape}")
        print(f"   ✓ Top-3 predicted goals: {predictions[0].topk(3).indices.tolist()}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
