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
        poi_nodes: List[str]
    ):
        """
        Args:
            trajectories: List of trajectory dictionaries from simulation
            graph: NetworkX graph of the world
            poi_nodes: List of POI node IDs (for goal indexing)
        """
        self.trajectories = trajectories
        self.graph = graph
        self.poi_nodes = poi_nodes
        
        # Create mapping from node_id to goal index
        self.goal_to_idx = {node_id: idx for idx, node_id in enumerate(poi_nodes)}
        
        # Filter trajectories that have valid goals
        self.valid_indices = []
        for idx, traj in enumerate(trajectories):
            if traj.get('goal_node') in self.goal_to_idx:
                self.valid_indices.append(idx)
        
        print(f"📊 Dataset: {len(self.valid_indices)} / {len(trajectories)} trajectories with valid POI goals")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single trajectory sample.
        
        Returns:
            Dict with keys:
            - 'trajectory': List of (node_id, goal_id) tuples
            - 'hour': int (0-23)
            - 'goal_idx': int (index in poi_nodes list)
            - 'goal_node': str (node ID of the goal)
        """
        real_idx = self.valid_indices[idx]
        traj = self.trajectories[real_idx]
        
        goal_node = traj['goal_node']
        goal_idx = self.goal_to_idx[goal_node]
        
        return {
            'trajectory': traj['path'],
            'hour': traj['hour'],
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
        # Flatten all agent trajectories into a single list
        trajectories = []
        for agent_id, agent_trajs in data.items():
            trajectories.extend(agent_trajs)
        print(f"🚶 Loaded {len(trajectories)} trajectories from {len(data)} agents")
        # print an example trajectory
        if len(trajectories) > 0:
            print(f"   Example trajectory: {trajectories[0]}")
            print(f"Len of Trajectory variable: {len(trajectories)}")
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
    Collate function for batching trajectories.
    
    This is needed because trajectories have variable lengths.
    The collation will be handled by TrajectoryDataPreparator in the encoder.
    
    Args:
        batch: List of samples from TrajectoryDataset
    
    Returns:
        Dict with batched data:
        - 'trajectories': List of trajectory paths
        - 'hours': List of hours
        - 'goal_indices': Tensor of goal indices
        - 'goal_nodes': List of goal node IDs
    """
    trajectories = [sample['trajectory'] for sample in batch]
    hours = [sample['hour'] for sample in batch]
    goal_indices = torch.tensor([sample['goal_idx'] for sample in batch], dtype=torch.long)
    goal_nodes = [sample['goal_node'] for sample in batch]
    
    return {
        'trajectories': trajectories,
        'hours': hours,
        'goal_indices': goal_indices,
        'goal_nodes': goal_nodes
    }


def create_dataloaders(
    train_trajectories: List[Dict],
    val_trajectories: List[Dict],
    test_trajectories: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        train_trajectories: Training trajectories
        val_trajectories: Validation trajectories
        test_trajectories: Test trajectories
        graph: NetworkX graph
        poi_nodes: List of POI node IDs
        batch_size: Batch size for training
        num_workers: Number of workers for data loading (0 for Mac MPS)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TrajectoryDataset(train_trajectories, graph, poi_nodes)
    val_dataset = TrajectoryDataset(val_trajectories, graph, poi_nodes)
    test_dataset = TrajectoryDataset(test_trajectories, graph, poi_nodes)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=num_workers,
        pin_memory=False  # Set to False for MPS compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"\n📦 DataLoaders created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


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
    train_loader, val_loader, test_loader = create_dataloaders(
        train_trajs, val_trajs, test_trajs, graph, poi_nodes, batch_size=16
    )
    
    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    for batch in train_loader:
        print(f"   Trajectories: {len(batch['trajectories'])}")
        print(f"   Hours: {batch['hours'][:5]}")
        print(f"   Goal indices shape: {batch['goal_indices'].shape}")
        print(f"   Sample trajectory length: {len(batch['trajectories'][0])}")
        break
    
    print("\n✅ Data loading test passed!")
