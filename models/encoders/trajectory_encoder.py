import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn

# -----------------------------
# Trajectory Encoder (Transformer)
# -----------------------------
class TrajectoryEncoder(nn.Module):
    """
    Encodes an agent's trajectory (sequence of node indices) using a Transformer.
    Also incorporates the hour of day as trajectory-level context.
    The Transformer's positional encoding handles the sequential ordering of nodes.
    """
    def __init__(self, num_nodes, node_feat_dim, hidden_dim, output_dim, n_layers=2, n_heads=4, dropout=0.1):
        """
        Args:
            num_nodes (int): Total number of unique nodes in the graph
            node_feat_dim (int): Dimension of node embeddings
            hidden_dim (int): Hidden dimension for transformer feedforward network
            output_dim (int): Final output embedding dimension
            n_layers (int): Number of transformer encoder layers
            n_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node embedding layer (learnable embeddings for each node)
        self.node_embedding = nn.Embedding(num_nodes, node_feat_dim, padding_idx=0)
        
        # Hour embedding (24 hours in a day)
        self.hour_embedding = nn.Embedding(24, node_feat_dim)
        
        # Transformer encoder (processes node sequences)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_feat_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection (combines trajectory embedding + hour embedding)
        self.output_proj = nn.Sequential(
            nn.Linear(node_feat_dim * 2, hidden_dim),  # *2 because we concatenate traj + hour
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, node_ids, hour, mask=None):
        """
        Forward pass through the trajectory encoder.
        
        Args:
            node_ids: (batch_size, seq_len) - Node IDs for each trajectory
            hour: (batch_size,) - Hour of day (0-23) when trajectory occurred
            mask: (batch_size, seq_len) - Mask for padding (1 for valid, 0 for padding)
            
        Returns:
            Encoded trajectory representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len = node_ids.shape
        
        # Embed nodes
        node_emb = self.node_embedding(node_ids)  # (batch_size, seq_len, node_feat_dim)
        
        # Create attention mask for padding
        if mask is None:
            mask = (node_ids != 0).float()  # Assume 0 is padding
        
        # Apply transformer (Transformer's built-in positional encoding handles sequence position)
        transformer_out = self.transformer(
            node_emb, 
            src_key_padding_mask=(mask == 0)
        )  # (batch_size, seq_len, node_feat_dim)
        
        # Global mean pooling (average over valid tokens only)
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        traj_emb = (transformer_out * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        # (batch_size, node_feat_dim)
        
        # Embed hour (trajectory-level temporal context)
        hour_emb = self.hour_embedding(hour)  # (batch_size, node_feat_dim)
        
        # Concatenate trajectory and hour embeddings
        combined = torch.cat([traj_emb, hour_emb], dim=-1)  # (batch_size, node_feat_dim * 2)
        
        # Project to final output dimension
        output = self.output_proj(combined)  # (batch_size, output_dim)
        
        return output


import torch
import torch.nn as nn
import json
import os

# -----------------------------
# Trajectory Data Preparation Utilities
# -----------------------------
class TrajectoryDataPreparator:
    """
    Utility class to prepare trajectory data for the encoder.
    """
    def __init__(self, node_to_idx_mapping):
        """
        Args:
            node_to_idx_mapping (dict): Mapping from node IDs to indices
        """
        self.node_to_idx = node_to_idx_mapping
        self.idx_to_node = {idx: node for node, idx in node_to_idx_mapping.items()}
    
    def prepare_single_trajectory(self, trajectory, max_seq_len=50):
        """
        Prepare a single trajectory for model input.
        
        Args:
            trajectory (dict): Trajectory data with 'path', 'hour', and 'goal_node' keys
            max_seq_len (int): Maximum sequence length (for padding/truncation)
            
        Returns:
            Tuple of (node_ids, hour, mask, goal_idx) tensors
        """
        path = trajectory['path']
        hour = trajectory['hour']
        goal_node = trajectory['goal_node']
        
        node_indices = []
        for entry in path:
            if isinstance(entry, (list, tuple)) and entry:
                node_id = entry[0]
            else:
                node_id = entry
            node_indices.append(self.node_to_idx.get(node_id, 0))
        
        # Truncate if too long
        if len(node_indices) > max_seq_len:
            node_indices = node_indices[:max_seq_len]
        
        # Pad if too short
        seq_len = len(node_indices)
        if seq_len < max_seq_len:
            node_indices += [0] * (max_seq_len - seq_len)
        
        # Create mask (1 for valid, 0 for padding)
        mask = [1.0] * seq_len + [0.0] * (max_seq_len - seq_len)
        
        # Convert goal node to index
        goal_idx = self.node_to_idx.get(goal_node, 0)
        
        return (
            torch.tensor(node_indices, dtype=torch.long),
            torch.tensor(hour, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(goal_idx, dtype=torch.long)
        )
    
    def prepare_batch(self, trajectories, max_seq_len=50):
        """
        Prepare a batch of trajectories.
        
        Args:
            trajectories (list): List of trajectory dicts
            max_seq_len (int): Maximum sequence length
            
        Returns:
            Dict with batched tensors
        """
        batch_node_ids = []
        batch_hours = []
        batch_masks = []
        batch_goals = []
        
        for traj in trajectories:
            node_ids, hour, mask, goal_idx = self.prepare_single_trajectory(traj, max_seq_len)
            batch_node_ids.append(node_ids)
            batch_hours.append(hour)
            batch_masks.append(mask)
            batch_goals.append(goal_idx)
        
        return {
            'node_ids': torch.stack(batch_node_ids),
            'hour': torch.stack(batch_hours),
            'mask': torch.stack(batch_masks),
            'goal': torch.stack(batch_goals)
        }


# -----------------------------
# Main Function for Testing
# -----------------------------
def main():
    """Test the Trajectory Encoder with sample data."""
    print("Testing Trajectory Encoder with sample trajectories...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load sample trajectory data
    print("\nLoading trajectory data...")
    data_path = os.path.join("data", "simulation_data", "run_8", "trajectories", "all_trajectories.json")
    
    if not os.path.exists(data_path):
        print(f"❌ Trajectory data not found at {data_path}")
        print("Please run the simulation first to generate trajectory data.")
        return
    
    with open(data_path, 'r') as f:
        all_trajectories = json.load(f)
    
    # Get sample trajectories from first agent
    first_agent_id = list(all_trajectories.keys())[0]
    sample_trajectories = all_trajectories[first_agent_id][:5]  # Take first 5 trajectories
    print(f"Loaded {len(sample_trajectories)} sample trajectories from {first_agent_id}")
    
    # Create node mapping from all trajectories
    print("\nCreating node mapping...")
    all_nodes = set()
    for agent_trajs in all_trajectories.values():
        for traj in agent_trajs:
            for entry in traj['path']:
                node_id = entry[0] if isinstance(entry, (list, tuple)) else entry
                all_nodes.add(node_id)
            all_nodes.add(traj['goal_node'])  # Also include goal nodes
    
    node_to_idx = {node: idx + 1 for idx, node in enumerate(sorted(all_nodes))}  # Start from 1 (0 is padding)
    node_to_idx['<PAD>'] = 0
    num_nodes = len(node_to_idx)
    print(f"Created mapping for {num_nodes} unique nodes")
    
    # Prepare trajectory data
    print("\nPreparing trajectory data...")
    data_preparator = TrajectoryDataPreparator(node_to_idx)
    trajectory_data = data_preparator.prepare_batch(sample_trajectories, max_seq_len=50)
    
    print(f"Node IDs shape: {trajectory_data['node_ids'].shape}")
    print(f"Hour shape: {trajectory_data['hour'].shape}")
    print(f"Mask shape: {trajectory_data['mask'].shape}")
    print(f"Goal shape: {trajectory_data['goal'].shape}")
    
    # Move data to device
    for key in trajectory_data:
        trajectory_data[key] = trajectory_data[key].to(device)
    
    # Initialize encoder with NEW parameters
    encoder = TrajectoryEncoder(
        num_nodes=num_nodes,
        node_feat_dim=64,      # Node embedding dimension
        hidden_dim=128,        # Transformer feedforward dimension
        output_dim=128,        # Final output dimension
        n_layers=2,
        n_heads=4,
        dropout=0.1
    ).to(device)
    
    print(f"\nEncoder initialized with:")
    print(f"  - Number of nodes: {num_nodes}")
    print(f"  - Node embedding dimension: 64")
    print(f"  - Hidden dimension: 128")
    print(f"  - Output dimension: 128")
    print(f"  - Number of layers: 2")
    print(f"  - Attention heads: 4")
    
    # Test forward pass
    print("\nTesting forward pass...")
    encoder.eval()
    with torch.no_grad():
        output = encoder(
            trajectory_data['node_ids'],
            trajectory_data['hour'],
            trajectory_data['mask']
        )
        print(f"Output shape: {output.shape}")
        print(f"Output sample (trajectory 1): {output[0][:10]}")
        print(f"Output sample (trajectory 2): {output[1][:10]}")
    
    print("\n✅ Trajectory Encoder test completed successfully!")
    
    # Print model summary
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print trajectory statistics
    print(f"\nTrajectory Statistics:")
    for i, traj in enumerate(sample_trajectories):
        path_len = len(traj['path'])
        hour = traj['hour']
        goal = traj['goal_node']
        print(f"  Trajectory {i+1}: {path_len} nodes, hour={hour}, goal={goal}")


if __name__ == "__main__":
    main()