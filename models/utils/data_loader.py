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
from archive.temporal_feature_enricher import TemporalFeatureEnricher, EnrichedTrajectoryDataset as ToMEnrichedDataset


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
    graph_path: str,
    trajectory_filename: str = 'all_trajectories.json'
) -> Tuple[List[Dict], nx.Graph, List[str]]:
    """
    Load simulation data from a run directory.
    
    Args:
        run_dir: Path to simulation run directory (e.g., 'data/simulation_data/run_8')
        graph_path: Path to the graph file (e.g., 'data/processed/ucsd_walk_semantic.graphml')
        trajectory_file: Name of trajectory file to load (default: 'trajectories.json')
                        Can also be a relative path like 'trajectories/enriched_trajectories.json'
    
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
    
    # Construct trajectory path
    # Check if trajectory_filename is already a full path or just filename
    if os.path.isabs(trajectory_filename) or trajectory_filename.startswith('trajectories/'):
        traj_path = os.path.join(run_dir, trajectory_filename)
    else:
        # Default: trajectories are in trajectories/ subdirectory
        traj_path = os.path.join(run_dir, 'trajectories', trajectory_filename)
    
    if not os.path.exists(traj_path):
        raise FileNotFoundError(
            f"Could not find trajectory file: {traj_path}\n"
            f"Specified: run_dir='{run_dir}', trajectory_file='{trajectory_filename}'"
        )
    
    print(f"📂 Loading trajectories from {traj_path}")
    
    with open(traj_path, 'r') as f:
        data = json.load(f)
    
    # Handle different trajectory file formats
    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        # Format: {"agent_000": [traj1, traj2, ...], "agent_001": [...], ...}
        # Flatten all agent trajectories into a single list while preserving agent IDs
        trajectories = []
        agent_keys = sorted(data.keys())  # Sort for consistent agent ID assignment
        for agent_idx, agent_key in enumerate(agent_keys):
            agent_trajs = data[agent_key]
            for traj in agent_trajs:
                traj['agent_id'] = agent_idx  # Assign enumerated index as agent ID (0-99)
                trajectories.append(traj)
        print(f"🚶 Loaded {len(trajectories)} trajectories from {len(agent_keys)} agents")
        print(f"   Agent IDs assigned: 0-{len(agent_keys)-1}")
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


def collate_enriched_trajectories(batch: List[Dict]) -> Dict:
    """
    Collate function for enriched trajectories with temporal features.
    
    Handles variable-length trajectories and temporal features (days, deltas, velocities).
    
    Theory of Mind Benefits:
    - Day of week patterns reveal habitual behavior
    - Temporal deltas show deliberation and planning
    - Velocities indicate confidence in navigation
    
    Args:
        batch: List of enriched samples from EnrichedTrajectoryDataset
    
    Returns:
        Dict with batched data including temporal features:
        - 'trajectories': List of trajectory paths
        - 'hours': List of hours (circadian)
        - 'days': List of day-of-week values
        - 'temporal_deltas': List of numpy arrays (variable length)
        - 'velocities': List of numpy arrays (variable length)
        - 'goal_indices': Tensor of goal indices
        - 'goal_nodes': List of goal node IDs
    """
    trajectories = [sample['trajectory'] for sample in batch]
    hours = [sample['circadian_hour'] for sample in batch]
    days = [sample['day_of_week'] for sample in batch]
    goal_indices = torch.tensor([sample['goal_idx'] for sample in batch], dtype=torch.long)
    goal_nodes = [sample['goal_node'] for sample in batch]
    
    temporal_deltas = [sample['temporal_deltas'] for sample in batch]
    velocities = [sample['velocities'] for sample in batch]
    
    return {
        'trajectories': trajectories,
        'hours': hours,
        'days': days,
        'temporal_deltas': temporal_deltas,  # Variable length per trajectory
        'velocities': velocities,  # Variable length per trajectory
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
    num_workers: int = 0,
    use_enriched: bool = False
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
        use_enriched: If True, use enriched trajectories with temporal features
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if use_enriched:
        # Check if trajectories have temporal features
        has_temporal_features = (
            train_trajectories and 
            'temporal_deltas' in train_trajectories[0] and 
            'day_of_week' in train_trajectories[0]
        )
        
        if not has_temporal_features:
            raise ValueError(
                "use_enriched=True but trajectories lack temporal features. "
                "Use enrich_and_load_data() or enrich_trajectories() first."
            )
        
        train_dataset = ToMEnrichedDataset(train_trajectories, graph, poi_nodes)
        val_dataset = ToMEnrichedDataset(val_trajectories, graph, poi_nodes)
        test_dataset = ToMEnrichedDataset(test_trajectories, graph, poi_nodes)
        
        collate_fn = collate_enriched_trajectories
    else:
        train_dataset = TrajectoryDataset(train_trajectories, graph, poi_nodes)
        val_dataset = TrajectoryDataset(val_trajectories, graph, poi_nodes)
        test_dataset = TrajectoryDataset(test_trajectories, graph, poi_nodes)
        
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
    if use_enriched:
        print(f"   ⭐ Using enriched trajectories with temporal features")
    
    return train_loader, val_loader, test_loader


def enrich_and_load_data(
    run_dir: str,
    graph_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    save_enriched: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Convenience function: Load, enrich with temporal features, and create dataloaders.
    
    This is the EXPERT-LEVEL recommended approach for BDI-Theory of Mind training.
    
    Theory of Mind Features Automatically Added:
    1. Day of week: Captures weekly behavioral patterns
    2. Temporal deltas: Time between steps shows deliberation
    3. Velocities: Movement speed indicates confidence/uncertainty
    4. Circadian patterns: Hour-based activity clustering
    
    Args:
        run_dir: Path to simulation run directory
        graph_path: Path to graph file
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        batch_size: Batch size for dataloaders
        num_workers: Number of workers (0 for MPS)
        seed: Random seed for reproducibility
        save_enriched: Whether to save enriched trajectories to disk
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, enrichment_stats)
    """
    print("\n" + "=" * 80)
    print("EXPERT-LEVEL BDI-ToM DATA LOADING WITH TEMPORAL ENRICHMENT")
    print("=" * 80)
    
    # Load base data
    print("\n📂 Step 1: Loading simulation data...")
    trajectories, graph, poi_nodes = load_simulation_data(run_dir, graph_path)
    
    # Create enricher
    print("\n🔧 Step 2: Creating temporal feature enricher...")
    enricher = TemporalFeatureEnricher(graph, simulation_duration_days=14, seed=seed)
    print(f"   ✅ Enricher ready (base walking speed: 1.4 m/s)")
    
    # Build agent map for enrichment
    print("\n💡 Step 3: Computing Theory-of-Mind temporal features...")
    enriched_trajectories = enricher.enrich_trajectories(
        trajectories, 
        agent_map=None,
        base_speed_mps=1.4,
        show_progress=True
    )
    
    # Get enrichment statistics
    enrichment_stats = enricher._compute_statistics(enriched_trajectories)
    print(f"\n📊 Enrichment Statistics:")
    print(f"   Total trajectories: {enrichment_stats['total_trajectories']}")
    print(f"   Avg trajectory length: {enrichment_stats['avg_trajectory_length']:.1f} steps")
    print(f"   Avg temporal delta: {enrichment_stats.get('overall_avg_delta', 0):.4f} hours")
    print(f"   Avg velocity: {enrichment_stats.get('overall_avg_velocity', 0):.2f} units/hour")
    print(f"   Day distribution: {enrichment_stats['day_distribution']}")
    
    # Save enriched trajectories if requested
    if save_enriched:
        enriched_save_path = os.path.join(run_dir, 'enriched_trajectories.json')
        os.makedirs(os.path.dirname(enriched_save_path), exist_ok=True)
        with open(enriched_save_path, 'w') as f:
            json.dump(enriched_trajectories, f, indent=2)
        print(f"\n💾 Saved enriched trajectories to {enriched_save_path}")
    
    # Split data
    print("\n📊 Step 4: Splitting data into train/val/test...")
    train_trajs, val_trajs, test_trajs = split_data(
        enriched_trajectories,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Create enriched dataloaders
    print("\n🔌 Step 5: Creating enriched dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_trajs, val_trajs, test_trajs,
        graph, poi_nodes,
        batch_size=batch_size,
        num_workers=num_workers,
        use_enriched=True
    )
    
    print("\n✅ Data loading complete! Ready for Theory-of-Mind training.")
    print("=" * 80 + "\n")
    
    return train_loader, val_loader, test_loader, enrichment_stats


if __name__ == '__main__':
    """Test data loading."""
    
    # Test with run_8 data
    run_dir = 'data/simulation_data/run_8'
    graph_path = 'data/processed/ucsd_walk_full.graphml'
    
    if not os.path.exists(run_dir):
        print(f"❌ Run directory not found: {run_dir}")
        print(f"   Please run a simulation first")
        exit(1)
    
    print("\n" + "=" * 80)
    print("TESTING ENRICHED DATA LOADING")
    print("=" * 80)
    
    # Use convenience function for enriched loading
    train_loader, val_loader, test_loader, stats = enrich_and_load_data(
        run_dir, graph_path, batch_size=16, save_enriched=True
    )
    
    # Test loading a batch
    print("\n🧪 Testing enriched batch loading...")
    for batch in train_loader:
        print(f"   Trajectories: {len(batch['trajectories'])}")
        print(f"   Hours: {batch['hours'][:5]}")
        print(f"   Days: {batch['days'][:5]}")
        print(f"   Goal indices shape: {batch['goal_indices'].shape}")
        print(f"   Sample trajectory length: {len(batch['trajectories'][0])}")
        print(f"   Sample temporal deltas: {batch['temporal_deltas'][0][:5]}")
        print(f"   Sample velocities: {batch['velocities'][0][:5]}")
        break
    
    print("\n✅ Enriched data loading test passed!")
