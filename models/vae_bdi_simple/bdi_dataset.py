"""
BDI VAE DATASET AND DATA UTILITIES

This module contains the dataset class and collate function for the
hierarchical BDI VAE model with per-node training.

DESIGN:
- BDIVAEDataset: Expands trajectories into per-node training samples
- collate_bdi_samples: Batches variable-length histories with padding
- Includes agent_id for agent encoder in unified embedding pipeline

For each trajectory [n1→n2→n3→n4→goal] from agent_id:
- Sample 1: history=[n1],        next=n2,  goal=goal_node, agent=agent_id
- Sample 2: history=[n1→n2],     next=n3,  goal=goal_node, agent=agent_id
- Sample 3: history=[n1→n2→n3],  next=n4,  goal=goal_node, agent=agent_id

MENTAL STATE LEARNING:
- Each sample provides context for learning mental states:
  * Belief: Where has the agent been? (history trajectory)
  * Desire: What does this agent typically prefer? (agent_id + history)
  * Intention: Where are they going? (supervised by goal + next step)
"""

from typing import Dict, List
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import networkx as nx


class BDIVAEDataset(Dataset):
    """
    Per-node dataset for BDI VAE model that includes agent information.
    
    Each trajectory is expanded into multiple samples representing the agent's
    mental state at different points along their path to the goal.
    
    For trajectory [n1→n2→n3→n4→goal] from agent with id=5:
    - Sample 1: history=[n1],        next=n2,  agent_id=5
    - Sample 2: history=[n1→n2],     next=n3,  agent_id=5
    - Sample 3: history=[n1→n2→n3],  next=n4,  agent_id=5
    
    Each sample includes:
    - history_node_indices: List of node indices visited so far
    - next_node_idx: Index of the next node in the path
    - goal_cat_idx: Category index of the goal node
    - goal_idx: Index of the goal POI node
    - agent_id: Index of the agent (for agent encoder)
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
    ):
        """
        Initialize BDI VAE dataset.
        
        Args:
            trajectories: List of trajectory dictionaries with:
                - 'path': List of nodes visited
                - 'goal_node': Final destination node
                - 'agent_id': Agent identifier (required for BDI VAE)
            graph: NetworkX graph containing node attributes (including 'Category')
            poi_nodes: List of POI node IDs for goal indexing
            node_to_idx_map: Optional pre-computed node-to-index mapping
            min_traj_length: Minimum trajectory length to include (default: 2)
        """
        self.samples = []
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.goal_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
        
        # Create node-to-index mapping
        if node_to_idx_map is None:
            self.node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        else:
            self.node_to_idx = node_to_idx_map
        
        # Expand trajectories into per-node samples
        print(f"   🧹 Creating BDI per-node samples from {len(trajectories)} trajectories...")
        valid_count = 0
        sample_count = 0
        missing_agent_count = 0
        
        for traj_idx, traj in enumerate(tqdm(trajectories, desc="   📊 Expanding BDI samples", leave=False)):
            if 'path' not in traj or 'goal_node' not in traj:
                continue
            
            path = traj['path']
            goal_node = traj['goal_node']
            
            # Agent ID is required for BDI VAE
            if 'agent_id' not in traj:
                missing_agent_count += 1
                continue
            
            agent_id = traj['agent_id']
            
            # Skip invalid trajectories
            if not isinstance(path, list) or len(path) < min_traj_length:
                continue
            
            if goal_node not in self.goal_to_idx:
                continue
            
            valid_count += 1
            goal_idx = self.goal_to_idx[goal_node]
            
            # Create per-node samples from path history
            for step_idx in range(1, len(path)):
                history_path = path[:step_idx]
                next_node = path[step_idx]
                
                # Convert node IDs to indices
                # Handle both simple node IDs and [node_id, category] tuples
                try:
                    history_indices = []
                    for node in history_path:
                        # Extract node ID if it's a list/tuple
                        node_id = node[0] if isinstance(node, (list, tuple)) else node
                        history_indices.append(self.node_to_idx[node_id])
                    
                    # Extract node ID for next step
                    next_node_id = next_node[0] if isinstance(next_node, (list, tuple)) else next_node
                    next_node_idx = self.node_to_idx[next_node_id]
                except (KeyError, TypeError, IndexError):
                    continue
                
                # Extract goal category from graph
                goal_cat_idx = 0
                if goal_node in self.graph.nodes:
                    cat_name = self.graph.nodes[goal_node].get('Category', 'other')
                    goal_cat_idx = self.CATEGORY_TO_IDX.get(cat_name, 0)
                
                self.samples.append({
                    'history_node_indices': history_indices,
                    'next_node_idx': next_node_idx,
                    'goal_cat_idx': goal_cat_idx,
                    'goal_idx': goal_idx,
                    'agent_id': agent_id,  # Include agent ID
                })
                
                sample_count += 1
        
        if missing_agent_count > 0:
            print(f"   ⚠️  Warning: {missing_agent_count} trajectories missing agent_id (skipped)")
        
        print(f"✅ Created {sample_count} BDI per-node samples from {valid_count} trajectories")
        print(f"   Average samples per trajectory: {sample_count / max(valid_count, 1):.1f}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_bdi_samples(batch: List[Dict]) -> Dict:
    """
    Collate BDI per-node samples into a batch.
    
    Pads variable-length history sequences to the maximum length in the batch.
    Uses 0 (dummy node) for padding.
    
    Args:
        batch: List of samples from BDIVAEDataset
    
    Returns:
        Dict with batched tensors:
        - 'history_node_indices': [batch, max_len] - Padded node indices
        - 'history_lengths': [batch] - Actual length of each history
        - 'next_node_idx': [batch] - Next node indices
        - 'goal_cat_idx': [batch] - Goal category indices
        - 'goal_idx': [batch] - Goal POI indices
        - 'agent_id': [batch] - Agent indices
    """
    if not batch:
        raise ValueError("Empty batch!")
    
    # Find max history length in batch
    max_history_len = max(len(s['history_node_indices']) for s in batch)
    max_history_len = max(1, max_history_len)
    
    batch_size = len(batch)
    
    # Pad all histories to max length
    padded_histories = []
    history_lengths = []
    
    for sample in batch:
        history = sample['history_node_indices']
        pad_len = max_history_len - len(history)
        
        # Pad with 0 (dummy node)
        padded = history + [0] * pad_len
        padded_histories.append(torch.tensor(padded, dtype=torch.long))
        history_lengths.append(len(history))
    
    return {
        'history_node_indices': torch.stack(padded_histories),  # [batch, max_len]
        'history_lengths': torch.tensor(history_lengths, dtype=torch.long),  # [batch]
        'next_node_idx': torch.tensor([s['next_node_idx'] for s in batch], dtype=torch.long),
        'goal_cat_idx': torch.tensor([s['goal_cat_idx'] for s in batch], dtype=torch.long),
        'goal_idx': torch.tensor([s['goal_idx'] for s in batch], dtype=torch.long),
        'agent_id': torch.tensor([s['agent_id'] for s in batch], dtype=torch.long),  # [batch]
    }
