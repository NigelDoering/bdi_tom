"""
Goal Prediction Model

This module implements a simple goal prediction model that takes the fused
trajectory + map embeddings and outputs a probability distribution over POI nodes.

Architecture:
    Fusion Embedding (from ToMGraphEncoder) 
    → Transformer Layer (optional refinement)
    → MLP Classifier
    → Softmax → Probability Distribution over POIs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GoalPredictionModel(nn.Module):
    """
    Simple goal prediction model that outputs probability distribution over POI nodes.
    
    Takes the fused embedding from ToMGraphEncoder and predicts which POI is the goal.
    """
    
    def __init__(
        self,
        fusion_encoder,
        num_poi_nodes,
        fusion_dim=64,
        hidden_dim=128,
        n_transformer_layers=1,
        n_heads=4,
        dropout=0.2
    ):
        """
        Initialize goal prediction model.
        
        Args:
            fusion_encoder: ToMGraphEncoder instance (trajectory + map fusion)
            num_poi_nodes (int): Number of POI nodes to predict over
            fusion_dim (int): Dimension of fusion embedding (default: 64)
            hidden_dim (int): Hidden dimension for transformer and MLP (default: 128)
            n_transformer_layers (int): Number of transformer layers (default: 1)
            n_heads (int): Number of attention heads (default: 4)
            dropout (float): Dropout rate (default: 0.2)
        """
        super().__init__()
        
        self.fusion_encoder = fusion_encoder
        self.num_poi_nodes = num_poi_nodes
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # Project fusion embedding to hidden dimension
        self.input_proj = nn.Linear(fusion_dim, hidden_dim)
        
        # Transformer layer for further refinement (optional)
        if n_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        else:
            self.transformer = None
        
        # Classification head: MLP that outputs logits for each POI
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes)
        )
    
    def forward(self, trajectory_data, graph_data, return_logits=False):
        """
        Forward pass through the complete model.
        
        Args:
            trajectory_data (dict): Dictionary containing trajectory information
                - 'node_ids': (batch_size, seq_len) - Node IDs in trajectory
                - 'hour': (batch_size,) - Hour of day
                - 'mask': (batch_size, seq_len) - Padding mask
            graph_data (dict): Dictionary containing world graph information
                - 'x': (num_nodes, node_feat_dim) - Node features
                - 'edge_index': (2, num_edges) - Edge connectivity
            return_logits (bool): If True, return logits instead of probabilities
        
        Returns:
            torch.Tensor: 
                - If return_logits=False: (batch_size, num_poi_nodes) - Probability distribution
                - If return_logits=True: (batch_size, num_poi_nodes) - Logits
        """
        # Get fused embedding from trajectory + map
        fused_emb = self.fusion_encoder(trajectory_data, graph_data)  # (batch_size, fusion_dim)
        
        # Project to hidden dimension
        x = self.input_proj(fused_emb)  # (batch_size, hidden_dim)
        x = F.relu(x)
        
        # Apply transformer layer (optional refinement)
        if self.transformer is not None:
            # Transformer expects (batch, seq_len, hidden_dim), so unsqueeze
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            x = self.transformer(x)  # (batch_size, 1, hidden_dim)
            x = x.squeeze(1)  # (batch_size, hidden_dim)
        
        # Classify: output logits for each POI
        logits = self.classifier(x)  # (batch_size, num_poi_nodes)
        
        if return_logits:
            return logits
        else:
            # Return probability distribution
            return F.softmax(logits, dim=-1)
    
    def predict_top_k(self, trajectory_data, graph_data, k=5):
        """
        Predict top-k most likely goal POIs.
        
        Args:
            trajectory_data (dict): Trajectory data
            graph_data (dict): Graph data
            k (int): Number of top predictions to return (default: 5)
        
        Returns:
            tuple: (top_k_probs, top_k_indices)
                - top_k_probs: (batch_size, k) - Probabilities of top-k POIs
                - top_k_indices: (batch_size, k) - Indices of top-k POIs
        """
        probs = self.forward(trajectory_data, graph_data, return_logits=False)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        return top_k_probs, top_k_indices
    
    def compute_loss(self, trajectory_data, graph_data, true_goal_indices):
        """
        Compute cross-entropy loss for goal prediction.
        
        Args:
            trajectory_data (dict): Trajectory data
            graph_data (dict): Graph data
            true_goal_indices (torch.Tensor): (batch_size,) - True goal node indices
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        logits = self.forward(trajectory_data, graph_data, return_logits=True)
        loss = F.cross_entropy(logits, true_goal_indices)
        return loss


# -----------------------------
# Testing
# -----------------------------
def main():
    """Test the Goal Prediction Model with sample data."""
    print("=" * 80)
    print("Testing Goal Prediction Model")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load world graph
    print("\n1. Loading world graph...")
    from graph_controller.world_graph import WorldGraph
    from models.encoders.map_encoder import GraphDataPreparator
    import networkx as nx
    
    graph_path = "data/processed/ucsd_walk_full.graphml"
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return
    
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G)
    num_nodes = len(world_graph.G.nodes())
    num_poi_nodes = len(world_graph.poi_nodes)
    print(f"   ✓ Loaded graph: {num_nodes} total nodes, {num_poi_nodes} POI nodes")
    
    # Prepare graph data
    graph_prep = GraphDataPreparator(world_graph)
    graph_data = graph_prep.prepare_graph_data()
    print(f"   ✓ Graph features: {graph_data['x'].shape}")
    
    # Load trajectory data
    print("\n2. Loading trajectory data...")
    from models.encoders.trajectory_encoder import TrajectoryDataPreparator
    
    traj_path = "data/simulation_data/run_8/trajectories/all_trajectories.json"
    if not os.path.exists(traj_path):
        print(f"Error: Trajectory file not found at {traj_path}")
        return
    
    with open(traj_path, 'r') as f:
        trajectories = json.load(f)
    
    # Create node mapping
    node_list = list(world_graph.G.nodes())
    node_to_idx = {node: idx + 1 for idx, node in enumerate(node_list)}  # 0 reserved for padding
    node_to_idx['<PAD>'] = 0
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Create POI mapping (only POI nodes)
    poi_node_to_idx = {node: idx for idx, node in enumerate(world_graph.poi_nodes)}
    idx_to_poi = {idx: node for node, idx in poi_node_to_idx.items()}
    
    # Prepare sample trajectories
    traj_prep = TrajectoryDataPreparator(node_to_idx)
    agent_ids = list(trajectories.keys())[:3]
    sample_trajs = [trajectories[agent_id][0] for agent_id in agent_ids]
    trajectory_data = traj_prep.prepare_batch(sample_trajs, max_seq_len=50)
    print(f"   ✓ Loaded {len(sample_trajs)} trajectories")
    
    # Get true goal indices in POI space
    true_goals = []
    for traj in sample_trajs:
        goal_node = traj['goal_node']
        if goal_node in poi_node_to_idx:
            true_goals.append(poi_node_to_idx[goal_node])
        else:
            print(f"   Warning: Goal node {goal_node} not in POI list, using 0")
            true_goals.append(0)
    true_goal_indices = torch.tensor(true_goals, dtype=torch.long)
    
    print(f"   ✓ True goals: {[idx_to_poi[g] for g in true_goals]}")
    
    # Initialize fusion encoder
    print("\n3. Initializing ToM Fusion Encoder...")
    from models.encoders.fusion_encoder import ToMGraphEncoder
    
    fusion_encoder = ToMGraphEncoder(
        num_nodes=num_nodes + 1,  # +1 for padding
        graph_node_feat_dim=12,
        traj_node_emb_dim=32,
        hidden_dim=128,
        output_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1
    )
    print(f"   ✓ Fusion encoder initialized (output_dim=64)")
    
    # Initialize goal prediction model
    print("\n4. Initializing Goal Prediction Model...")
    goal_predictor = GoalPredictionModel(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=num_poi_nodes,
        fusion_dim=64,
        hidden_dim=128,
        n_transformer_layers=1,
        n_heads=4,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in goal_predictor.parameters())
    print(f"   ✓ Goal predictor initialized")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Predicting over {num_poi_nodes} POI nodes")
    
    # Move to device
    goal_predictor = goal_predictor.to(device)
    for key in trajectory_data:
        trajectory_data[key] = trajectory_data[key].to(device)
    for key in graph_data:
        graph_data[key] = graph_data[key].to(device)
    true_goal_indices = true_goal_indices.to(device)
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    goal_predictor.eval()
    with torch.no_grad():
        # Get probability distribution
        probs = goal_predictor(trajectory_data, graph_data)
        print(f"   ✓ Output shape: {probs.shape} (batch_size={probs.shape[0]}, num_poi_nodes={probs.shape[1]})")
        
        # Verify it's a valid probability distribution
        prob_sums = probs.sum(dim=-1)
        print(f"   ✓ Probability sums: {prob_sums.tolist()} (should be ~1.0)")
        
        # Check all probabilities are in [0, 1]
        assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities not in [0, 1]"
        print(f"   ✓ All probabilities in [0, 1]")
        
        # Get top-5 predictions for each trajectory
        print(f"\n6. Top-5 Goal Predictions:")
        top_k_probs, top_k_indices = goal_predictor.predict_top_k(trajectory_data, graph_data, k=5)
        
        for i, agent_id in enumerate(agent_ids):
            print(f"\n   Agent {agent_id}:")
            print(f"   True goal: {idx_to_poi[true_goals[i]]}")
            print(f"   Predictions:")
            for j in range(5):
                pred_idx = int(top_k_indices[i, j].item())
                pred_prob = top_k_probs[i, j].item()
                pred_node = idx_to_poi[pred_idx]
                marker = " ✓" if pred_idx == true_goals[i] else ""
                print(f"     {j+1}. {pred_node[:30]:30s} - {pred_prob:.4f}{marker}")
    
    # Test loss computation
    print(f"\n7. Testing loss computation...")
    with torch.no_grad():
        loss = goal_predictor.compute_loss(trajectory_data, graph_data, true_goal_indices)
        print(f"   ✓ Loss value: {loss.item():.4f}")
        print(f"   ✓ Loss is finite: {torch.isfinite(loss).item()}")
    
    # Test with logits
    print(f"\n8. Testing logit output...")
    with torch.no_grad():
        logits = goal_predictor(trajectory_data, graph_data, return_logits=True)
        print(f"   ✓ Logits shape: {logits.shape}")
        print(f"   ✓ Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
    print("\nModel Summary:")
    print(f"  - Input: Trajectory (variable length) + World Graph")
    print(f"  - Fusion Encoder: Combines trajectory + map → 64-dim embedding")
    print(f"  - Transformer: 1 layer with 4 attention heads")
    print(f"  - Classifier: 3-layer MLP")
    print(f"  - Output: Probability distribution over {num_poi_nodes} POI nodes")
    print(f"  - Total parameters: {total_params:,}")


if __name__ == "__main__":
    main()
