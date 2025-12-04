"""
Data loading utilities for BDI-ToM goal prediction training.

This module provides:
- TrajectoryDataset: PyTorch Dataset for trajectory data
- Data splitting utilities (train/val/test)
- Collate functions for batching variable-length trajectories
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import networkx as nx
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_controller.world_graph import WorldGraph


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory-goal prediction task.
    
    Each sample contains:
    - trajectory: List of (node_id, goal_id) tuples
    - hour: Hour of day the trajectory occurred
    - goal_node: The true destination/goal (label)
    """
    
    def __init__(
        self,
        trajectories: List[Dict],
        graph: nx.Graph,
        poi_nodes: List[str],
        agent_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            trajectories: List of trajectory dictionaries from simulation
            graph: NetworkX graph of the world
            poi_nodes: List of POI node IDs (for goal indexing)
            agent_to_idx: Optional mapping from agent_id (string) to integer index
        """
        self.trajectories = trajectories
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.agent_to_idx = agent_to_idx or {}
        
        # Create mapping from node_id to goal index
        self.goal_to_idx = {node_id: idx for idx, node_id in enumerate(poi_nodes)}
        
        # Filter trajectories that have valid goals
        self.valid_indices = []
        for idx, traj in enumerate(trajectories):
            if traj.get('goal_node') in self.goal_to_idx:
                self.valid_indices.append(idx)
        
        print(f"📊 Dataset: {len(self.valid_indices)} / {len(trajectories)} trajectories with valid POI goals")
        if self.agent_to_idx:
            print(f"👥 Using {len(self.agent_to_idx)} unique agents")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single trajectory sample.
        
        Returns:
            Dict with keys:
            - 'trajectory': List of (node_id, goal_id) tuples
            - 'hour': int (0-23)
            - 'agent_id': int (agent index, or 0 if not available)
            - 'goal_idx': int (index in poi_nodes list)
            - 'goal_node': str (node ID of the goal)
        """
        real_idx = self.valid_indices[idx]
        traj = self.trajectories[real_idx]
        
        goal_node = traj['goal_node']
        goal_idx = self.goal_to_idx[goal_node]
        
        # Get agent_id from trajectory data, default to 0 if not present
        agent_id_str = traj.get('agent_id', 'agent_000')
        agent_id = self.agent_to_idx.get(agent_id_str, 0)
        
        return {
            'trajectory': traj['path'],
            'hour': traj['hour'],
            'agent_id': agent_id,
            'goal_idx': goal_idx,
            'goal_node': goal_node
        }


def load_simulation_data(
    run_dir: str,
    graph_path: str
) -> Tuple[List[Dict], nx.Graph, List[str]]:
    """
    Load simulation data from a run directory.
    
    Args:
        run_dir: Path to simulation run directory (e.g., 'data/simulation_data/run_8')
        graph_path: Path to the graph file (e.g., 'data/processed/ucsd_walk_semantic.graphml')
    
    Returns:
        Tuple of (trajectories, graph, poi_nodes)
    """
    # Load graph
    print(f"📂 Loading graph from {graph_path}")
    graph = nx.read_graphml(graph_path)
    
    # Use WorldGraph to get POI nodes (uses correct Category logic)
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    
    print(f"📍 Found {len(poi_nodes)} POI nodes")
    print(f"   Categories: {world_graph.relevant_categories}")
    
    # Load trajectories - check for both possible locations
    possible_paths = [
        os.path.join(run_dir, 'trajectories', 'all_trajectories.json'),
        os.path.join(run_dir, 'all_trajectories.json'),
        os.path.join(run_dir, 'trajectories.json')
    ]
    
    traj_path = None
    for path in possible_paths:
        if os.path.exists(path):
            traj_path = path
            break
    
    if traj_path is None:
        raise FileNotFoundError(
            f"Could not find trajectories file in {run_dir}. "
            f"Tried: {possible_paths}"
        )
    
    print(f"📂 Loading trajectories from {traj_path}")
    
    with open(traj_path, 'r') as f:
        data = json.load(f)
    
    # Handle different trajectory file formats
    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        # Format: {"agent_000": [traj1, traj2, ...], "agent_001": [...], ...}
        # Flatten all agent trajectories into a single list and add agent_id to each trajectory
        trajectories = []
        for agent_id, agent_trajs in data.items():
            for traj in agent_trajs:
                # Add agent_id to trajectory if not already present
                if 'agent_id' not in traj:
                    traj['agent_id'] = agent_id
                trajectories.append(traj)
        print(f"🚶 Loaded {len(trajectories)} trajectories from {len(data)} agents")
    elif isinstance(data, list):
        # Format: [traj1, traj2, traj3, ...]
        trajectories = data
        print(f"🚶 Loaded {len(trajectories)} trajectories")
    else:
        raise ValueError(f"Unexpected trajectory file format. Expected dict or list, got {type(data)}")
    
    return trajectories, graph, poi_nodes


def split_data(
    trajectories: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split trajectories into train/val/test sets.
    
    Args:
        trajectories: List of trajectory dictionaries
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_trajectories, val_trajectories, test_trajectories)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle trajectories
    np.random.seed(seed)
    indices = np.random.permutation(len(trajectories))
    
    # Calculate split points
    n_train = int(len(trajectories) * train_ratio)
    n_val = int(len(trajectories) * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create splits
    train_trajs = [trajectories[i] for i in train_indices]
    val_trajs = [trajectories[i] for i in val_indices]
    test_trajs = [trajectories[i] for i in test_indices]
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(train_trajs)} ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(val_trajs)} ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(test_trajs)} ({test_ratio*100:.0f}%)")
    
    return train_trajs, val_trajs, test_trajs


def collate_trajectories(batch: List[Dict]) -> Dict:
    """
    Collate function for batching trajectories (raw version - no embeddings).
    
    This is needed because trajectories have variable lengths.
    The collation will be handled by TrajectoryDataPreparator in the encoder.
    
    Args:
        batch: List of samples from TrajectoryDataset
    
    Returns:
        Dict with batched data:
        - 'trajectories': List of trajectory paths
        - 'hours': List of hours
        - 'agent_ids': List of agent IDs
        - 'goal_indices': Tensor of goal indices
        - 'goal_nodes': List of goal node IDs
    """
    trajectories = [sample['trajectory'] for sample in batch]
    hours = [sample['hour'] for sample in batch]
    agent_ids = [sample['agent_id'] for sample in batch]
    goal_indices = torch.tensor([sample['goal_idx'] for sample in batch], dtype=torch.long)
    goal_nodes = [sample['goal_node'] for sample in batch]
    
    return {
        'trajectories': trajectories,
        'hours': hours,
        'agent_ids': agent_ids,
        'goal_indices': goal_indices,
        'goal_nodes': goal_nodes
    }


def preprocess_trajectory_length(node_ids: List[str], max_seq_len: int = 60) -> List[str]:
    """
    Preprocess trajectory to handle variable lengths intelligently.
    
    Strategy:
    - If len <= max_seq_len: Return as-is (will be padded later)
    - If len > max_seq_len: Downsample to max_seq_len nodes while preserving:
        1. Start node (trajectory beginning)
        2. End nodes (trajectory conclusion - important for goal prediction)
        3. Evenly spaced intermediate nodes (maintain trajectory coverage)
    
    Args:
        node_ids: List of node IDs in trajectory
        max_seq_len: Maximum sequence length
    
    Returns:
        List of node IDs with length <= max_seq_len
    """
    if len(node_ids) <= max_seq_len:
        return node_ids
    
    # For long trajectories, intelligently downsample
    # Strategy: Keep more nodes at the end (important for goal prediction)
    # Allocation: 40% beginning, 60% end
    
    n_start = int(max_seq_len * 0.4)  # Keep 40% from start
    n_end = max_seq_len - n_start      # Keep 60% from end
    
    # Sample indices from start portion
    start_indices = np.linspace(0, len(node_ids) // 2, n_start, dtype=int)
    
    # Sample indices from end portion (keep final nodes for goal context)
    end_indices = np.linspace(len(node_ids) // 2, len(node_ids) - 1, n_end, dtype=int)
    
    # Combine and ensure unique, sorted indices
    all_indices = np.unique(np.concatenate([start_indices, end_indices]))
    
    # Take exactly max_seq_len nodes
    if len(all_indices) > max_seq_len:
        print("PROBLEM: More indices than max_seq_len after sampling!")
        all_indices = all_indices[:max_seq_len]
    
    return [node_ids[i] for i in all_indices]


def create_collate_fn_with_embeddings(node_embeddings, max_seq_len: int = 60, incremental_training: bool = False):
    """
    Create a collate function that converts trajectories to Node2Vec embeddings.
    
    Args:
        node_embeddings: Node2VecEmbeddings instance for converting node IDs to embeddings
        max_seq_len: Maximum sequence length for padding/truncation
        incremental_training: If True, creates multiple samples per trajectory by incrementally
                            revealing nodes (node1 -> goal, node1-2 -> goal, etc.)
    
    Returns:
        Collate function for DataLoader
    """
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function that converts trajectories to Node2Vec embeddings.
        
        Args:
            batch: List of samples from TrajectoryDataset
        
        Returns:
            Dict with:
            - 'node_embeddings': (batch_size, max_seq_len, emb_dim) tensor
            - 'hour': (batch_size,) tensor
            - 'agent_id': (batch_size,) tensor
            - 'mask': (batch_size, max_seq_len) tensor (1 = valid, 0 = padding)
            - 'goal_indices': (batch_size,) tensor (POI indices for targets)
        """
        batch_node_embs = []
        batch_hours = []
        batch_agent_ids = []
        batch_masks = []
        batch_goal_indices = []
        
        for sample in batch:
            traj_path = sample['trajectory']
            hour = sample['hour']
            agent_id = sample['agent_id']
            goal_idx = sample['goal_idx']
            goal_node = sample['goal_node']
            
            # Extract node IDs from path (handle both list and tuple formats)
            node_ids = []
            for entry in traj_path:
                if isinstance(entry, (list, tuple)) and entry:
                    node_id = entry[0]
                else:
                    node_id = entry
                node_ids.append(node_id)
            
            # **CRITICAL FIX**: Remove goal node if it's at the end
            # The model should PREDICT the goal, not just read it from the input!
            if len(node_ids) > 0 and node_ids[-1] == goal_node:
                node_ids = node_ids[:-1]
            
            # Handle empty trajectory edge case
            if len(node_ids) == 0:
                # If trajectory is empty after removing goal, use a dummy node
                # This shouldn't happen in practice but provides safety
                node_ids = [list(traj_path[0])[0] if isinstance(traj_path[0], (list, tuple)) else traj_path[0]]
            
            # Preprocess trajectory length: downsample if too long, will pad if too short
            node_ids = preprocess_trajectory_length(node_ids, max_seq_len)
            
            if incremental_training:
                # INCREMENTAL TRAINING: Create multiple samples from this trajectory
                # For trajectory [n1, n2, n3, n4] → create 4 samples:
                #   [n1] -> goal, [n1, n2] -> goal, [n1, n2, n3] -> goal, [n1, n2, n3, n4] -> goal
                
                for i in range(1, len(node_ids) + 1):
                    # Take first i nodes
                    partial_node_ids = node_ids[:i]
                    seq_len = len(partial_node_ids)
                    
                    # Get embeddings
                    node_embs = node_embeddings.trajectory_to_embeddings(partial_node_ids)
                    
                    # Pad to max_seq_len
                    if seq_len < max_seq_len:
                        padding = torch.zeros(
                            max_seq_len - seq_len,
                            node_embeddings.embedding_dim,
                            dtype=torch.float32
                        )
                        node_embs = torch.cat([node_embs, padding], dim=0)
                    
                    # Create mask
                    mask = torch.cat([
                        torch.ones(seq_len, dtype=torch.float32),
                        torch.zeros(max_seq_len - seq_len, dtype=torch.float32)
                    ])
                    
                    # Add to batch (same goal for all incremental steps)
                    batch_node_embs.append(node_embs)
                    batch_hours.append(hour)
                    batch_agent_ids.append(agent_id)
                    batch_masks.append(mask)
                    batch_goal_indices.append(goal_idx)
            else:
                # STANDARD TRAINING: Use full trajectory
                seq_len = len(node_ids)
                node_embs = node_embeddings.trajectory_to_embeddings(node_ids)
                
                # Pad if too short
                if seq_len < max_seq_len:
                    padding = torch.zeros(
                        max_seq_len - seq_len,
                        node_embeddings.embedding_dim,
                        dtype=torch.float32
                    )
                    node_embs = torch.cat([node_embs, padding], dim=0)
                
                # Create mask (1 = valid token, 0 = padding)
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.float32),
                    torch.zeros(max_seq_len - seq_len, dtype=torch.float32)
                ])
                
                batch_node_embs.append(node_embs)
                batch_hours.append(hour)
                batch_agent_ids.append(agent_id)
                batch_masks.append(mask)
                batch_goal_indices.append(goal_idx)
        
        return {
            'node_embeddings': torch.stack(batch_node_embs),
            'hour': torch.tensor(batch_hours, dtype=torch.long),
            'agent_id': torch.tensor(batch_agent_ids, dtype=torch.long),
            'mask': torch.stack(batch_masks),
            'goal_indices': torch.tensor(batch_goal_indices, dtype=torch.long)
        }
    
    return collate_fn


def create_dataloaders(
    train_trajectories: List[Dict],
    val_trajectories: List[Dict],
    test_trajectories: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    node_embeddings=None,
    batch_size: int = 32,
    num_workers: int = 0,
    max_seq_len: int = 60,
    incremental_training: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        train_trajectories: Training trajectories
        val_trajectories: Validation trajectories
        test_trajectories: Test trajectories
        graph: NetworkX graph
        poi_nodes: List of POI node IDs
        node_embeddings: Node2VecEmbeddings instance (if None, returns raw trajectories)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading (0 for Mac MPS)
        max_seq_len: Maximum sequence length for trajectory padding
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_agents)
    """
    # Create agent_to_idx mapping from all trajectories
    all_trajectories = train_trajectories + val_trajectories + test_trajectories
    unique_agents = sorted(set(traj.get('agent_id', 'agent_000') for traj in all_trajectories))
    agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(unique_agents)}
    
    print(f"👥 Found {len(unique_agents)} unique agents")
    
    train_dataset = TrajectoryDataset(train_trajectories, graph, poi_nodes, agent_to_idx)
    val_dataset = TrajectoryDataset(val_trajectories, graph, poi_nodes, agent_to_idx)
    test_dataset = TrajectoryDataset(test_trajectories, graph, poi_nodes, agent_to_idx)
    
    # Choose collate function based on whether embeddings are provided
    if node_embeddings is not None:
        collate_fn = create_collate_fn_with_embeddings(
            node_embeddings, 
            max_seq_len, 
            incremental_training=incremental_training
        )
    else:
        collate_fn = collate_trajectories
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False  # Set to False for MPS compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"\n📦 DataLoaders created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # Return dataloaders and number of agents
    return train_loader, val_loader, test_loader, len(unique_agents)


if __name__ == '__main__':
    """Test data loading."""
    
    # Test with run_8 data
    run_dir = 'data/simulation_data/run_8'
    graph_path = 'data/processed/ucsd_walk_full.graphml'
    
    if not os.path.exists(run_dir):
        print(f"❌ Run directory not found: {run_dir}")
        print(f"   Please run a simulation first")
        exit(1)
    
    # Load data
    trajectories, graph, poi_nodes = load_simulation_data(run_dir, graph_path)
    
    # Split data
    train_trajs, val_trajs, test_trajs = split_data(trajectories)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_agents = create_dataloaders(
        train_trajs, val_trajs, test_trajs, graph, poi_nodes, batch_size=16
    )
    print(f"   Number of agents: {num_agents}")
    
    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    for batch in train_loader:
        print(f"   Trajectories: {len(batch['trajectories'])}")
        print(f"   Hours: {batch['hours'][:5]}")
        print(f"   Goal indices shape: {batch['goal_indices'].shape}")
        print(f"   Sample trajectory length: {len(batch['trajectories'][0])}")
        break
    
    print("\n✅ Data loading test passed!")
