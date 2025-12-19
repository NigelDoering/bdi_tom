"""
BASELINE TRANSFORMER DATASET AND DATA UTILITIES

This module contains the dataset class and collate function specifically
designed for the transformer baseline model.

KEY DIFFERENCE FROM LSTM DATASET:
- NO per-node expansion! Keeps full trajectories intact
- Transformer processes all positions in parallel with causal masking
- 10x fewer samples (70k vs 700k)
- Much faster data loading and training

For trajectory [n1→n2→n3→n4→goal], creates ONE sample:
- nodes: [n1, n2, n3, n4]
- next_targets: [n2, n3, n4, goal]
- Transformer predicts at all 4 positions simultaneously
"""

from typing import Dict, List
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import networkx as nx


class TransformerTrajectoryDataset(Dataset):
    """
    Transformer dataset that keeps full trajectories intact.
    
    Unlike LSTM per-node expansion, this dataset stores complete trajectories.
    The transformer will process all positions in parallel using causal masking,
    effectively doing per-node prediction without dataset expansion.
    
    Each sample contains:
    - node_indices: Full sequence of nodes in trajectory (excluding final goal)
    - next_indices: Target for next-step prediction (shifted by 1, includes goal at end)
    - goal_idx: Goal POI index (same for all positions)
    - goal_cat_idx: Goal category index (same for all positions)
    """
    
    CATEGORY_TO_IDX = {
        'home': 0,
        'study': 1,
        'food': 2,
        'leisure': 3,
        'errands': 4,
        'health': 5,
        'other': 6,
    }
    
    def __init__(
        self,
        trajectories: List[Dict],
        graph: nx.Graph,
        poi_nodes: List[str],
        node_to_idx_map: Dict[str, int] = None,
        min_traj_length: int = 2,
        max_traj_length: int = 60,
    ):
        """
        Initialize transformer trajectory dataset.
        
        Args:
            trajectories: List of trajectory dictionaries with 'path' and 'goal_node'
            graph: NetworkX graph containing node attributes (including 'Category')
            poi_nodes: List of POI node IDs for goal indexing
            node_to_idx_map: Optional pre-computed node-to-index mapping
            min_traj_length: Minimum trajectory length to include (default: 2)
            max_traj_length: Maximum trajectory length (default: 50)
        """
        self.trajectories = []
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.goal_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
        self.max_traj_length = max_traj_length
        
        # Create node-to-index mapping
        if node_to_idx_map is None:
            self.node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        else:
            self.node_to_idx = node_to_idx_map
        
        # Filter and store valid trajectories (NO expansion!)
        print(f"   🧹 Processing {len(trajectories)} trajectories...")
        valid_count = 0
        skipped_length = 0
        skipped_goal = 0
        
        for traj_idx, traj in enumerate(tqdm(trajectories, desc="   📊 Loading trajectories", leave=False)):
            if 'path' not in traj or 'goal_node' not in traj:
                continue
            
            path = traj['path']
            goal_node = traj['goal_node']
            
            # Skip invalid trajectories
            if not isinstance(path, list) or len(path) < min_traj_length:
                skipped_length += 1
                continue
            
            if goal_node not in self.goal_to_idx:
                skipped_goal += 1
                continue
            
            valid_count += 1
            
            # Store the FULL trajectory (no expansion!)
            self.trajectories.append({
                'path': path,
                'goal_node': goal_node,
                'traj_idx': traj_idx,
            })
        
        print(f"✅ Loaded {valid_count} valid trajectories")
        print(f"   Skipped: {skipped_length} too short, {skipped_goal} unknown goal")
        print(f"   NO expansion - keeping trajectories intact for transformer")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single trajectory sample.
        
        Returns:
            Dict with:
            - node_indices: List of node indices (excluding goal)
            - next_indices: List of next node indices (includes goal at end)
            - goal_idx: Goal POI index
            - goal_cat_idx: Goal category index
            - seq_length: Actual length of trajectory
        """
        traj = self.trajectories[idx]
        path = traj['path']
        goal_node = traj['goal_node']
        
        # Convert path nodes to indices
        # Handle both simple node IDs and [node_id, category] tuples
        node_indices = []
        for node in path:
            node_id = node[0] if isinstance(node, (list, tuple)) else node
            if node_id in self.node_to_idx:
                node_indices.append(self.node_to_idx[node_id])
        
        # If path already includes goal, remove it
        # (we want to predict it, not include it in input)
        goal_node_idx = self.node_to_idx.get(goal_node, None)
        if goal_node_idx is not None and len(node_indices) > 0 and node_indices[-1] == goal_node_idx:
            node_indices = node_indices[:-1]
        
        # Create next-step targets (shifted by 1, with goal at the end)
        if len(node_indices) > 0:
            next_indices = node_indices[1:] + [goal_node_idx]
        else:
            # Edge case: empty trajectory
            next_indices = [goal_node_idx]
            node_indices = [goal_node_idx]
        
        # Truncate if too long
        if len(node_indices) > self.max_traj_length:
            node_indices = node_indices[:self.max_traj_length]
            next_indices = next_indices[:self.max_traj_length]
        
        # Get goal information
        goal_idx = self.goal_to_idx[goal_node]
        
        # Extract goal category from graph
        goal_cat_idx = 0
        if goal_node in self.graph.nodes:
            cat_name = self.graph.nodes[goal_node].get('Category', 'other')
            goal_cat_idx = self.CATEGORY_TO_IDX.get(cat_name, 0)
        
        # Get agent ID (default to 0 if not present for backward compatibility)
        agent_id = traj.get('agent_id', 0)
        
        # Get hour of trajectory (default to 12 if not present)
        hour = traj.get('hour', 12)
        
        return {
            'node_indices': node_indices,
            'next_indices': next_indices,
            'goal_idx': goal_idx,
            'goal_cat_idx': goal_cat_idx,
            'agent_id': agent_id,
            'hour': hour,
            'seq_length': len(node_indices),
        }


def collate_transformer_trajectories(batch: List[Dict]) -> Dict:
    """
    Collate transformer trajectory samples into a batch.
    
    Pads variable-length trajectories to the maximum length in the batch.
    Uses 0 (dummy node) for padding.
    
    Unlike per-node collate, this handles full trajectories and creates
    appropriate masks for the transformer.
    
    Args:
        batch: List of samples from TransformerTrajectoryDataset
    
    Returns:
        Dict with batched tensors:
        - 'node_indices': [batch, max_len] - Padded node sequences
        - 'next_indices': [batch, max_len] - Padded next-step targets
        - 'goal_idx': [batch] - Goal POI indices
        - 'goal_cat_idx': [batch] - Goal category indices
        - 'padding_mask': [batch, max_len] - True for padding positions
        - 'seq_lengths': [batch] - Actual sequence lengths
    """
    if not batch:
        raise ValueError("Empty batch!")
    
    # Find max sequence length in batch
    max_len = max(s['seq_length'] for s in batch)
    max_len = max(1, max_len)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    padded_nodes = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_next = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padding
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    goal_indices = []
    goal_cat_indices = []
    agent_ids = []
    hours = []
    
    # Fill tensors
    for i, sample in enumerate(batch):
        seq_len = sample['seq_length']
        
        # Copy node indices
        padded_nodes[i, :seq_len] = torch.tensor(sample['node_indices'], dtype=torch.long)
        
        # Copy next indices
        padded_next[i, :seq_len] = torch.tensor(sample['next_indices'], dtype=torch.long)
        
        # Mark non-padding positions as False
        padding_mask[i, :seq_len] = False
        
        # Store lengths
        seq_lengths[i] = seq_len
        
        # Store goal info and metadata
        goal_indices.append(sample['goal_idx'])
        goal_cat_indices.append(sample['goal_cat_idx'])
        agent_ids.append(sample['agent_id'])
        hours.append(sample['hour'])
    
    return {
        'node_indices': padded_nodes,           # [batch, max_len]
        'next_indices': padded_next,            # [batch, max_len]
        'goal_idx': torch.tensor(goal_indices, dtype=torch.long),      # [batch]
        'goal_cat_idx': torch.tensor(goal_cat_indices, dtype=torch.long),  # [batch]
        'agent_ids': torch.tensor(agent_ids, dtype=torch.long),        # [batch]
        'hours': torch.tensor(hours, dtype=torch.float),               # [batch]
        'padding_mask': padding_mask,           # [batch, max_len]
        'seq_lengths': seq_lengths,             # [batch]
    }
