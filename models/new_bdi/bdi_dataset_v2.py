"""
ENHANCED BDI VAE DATASET V2 FOR SC-BDI-VAE V3

This dataset produces per-step samples from enriched trajectories for the
Sequential Conditional BDI-VAE.  It supports the **full unified embedding
pipeline** by extracting temporal and agent features alongside node indices.

Per-sample features produced:
    history_node_indices  – node-index sequence up to current step
    next_node_idx         – target next-step node index
    goal_idx              – POI-index of the trajectory's goal
    goal_cat_idx          – category index of the goal
    agent_id              – integer agent identifier
    path_progress         – float ∈ [0, 1]
    traj_id               – trajectory identifier
    step_idx              – position in the original trajectory

    -- Temporal (from enriched trajectories) --
    hour                  – hour of day (int, 0–23)
    day_of_week           – day of week (int, 0–6)
    history_temporal_deltas – per-step time deltas  (float list, len = step_idx)
    history_velocities    – per-step velocities     (float list, len = step_idx+1)
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import networkx as nx


class BDIVAEDatasetV2(Dataset):
    """
    Enhanced per-node dataset for SC-BDI-VAE V3.

    Supports the **enriched trajectory format** from run_8_enriched which
    contains temporal_deltas, velocities, hour, day_of_week, etc.

    If a trajectory does not have an explicit ``agent_id`` field the caller
    must assign one before passing trajectories in (see
    ``load_data_with_splits``).
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
        node_to_idx_map: Optional[Dict[str, int]] = None,
        min_traj_length: int = 2,
        include_progress: bool = True,
        include_temporal: bool = True,
    ):
        """
        Args:
            trajectories: List of trajectory dicts.  Each **must** contain
                ``path``, ``goal_node``, and ``agent_id`` (int).  May also
                contain ``hour``, ``day_of_week``, ``temporal_deltas``,
                ``velocities``.
            graph: NetworkX graph
            poi_nodes: Ordered list of POI node-ID strings (defines goal vocab)
            node_to_idx_map: Optional pre-computed node→index mapping
            min_traj_length: Minimum trajectory length to keep
            include_progress: Compute path-progress feature
            include_temporal: Extract temporal / velocity features when present
        """
        self.samples: List[Dict] = []
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.goal_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
        self.include_progress = include_progress
        self.include_temporal = include_temporal

        # Create node-to-index mapping
        if node_to_idx_map is None:
            self.node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        else:
            self.node_to_idx = node_to_idx_map

        # Track trajectory information
        self.trajectory_samples: Dict[int, List[int]] = {}

        print(f"   🧹 Creating enhanced BDI per-node samples from {len(trajectories)} trajectories...")
        valid_count = 0
        sample_count = 0
        missing_agent_count = 0

        for traj_idx, traj in enumerate(tqdm(trajectories, desc="   📊 Expanding BDI samples", leave=False)):
            if 'path' not in traj or 'goal_node' not in traj:
                continue

            path = traj['path']
            goal_node = traj['goal_node']

            # Agent ID is required
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
            total_path_length = len(path)

            # ---- Temporal features (enriched format) ----
            traj_hour = int(traj.get('hour', 0))
            traj_day  = int(traj.get('day_of_week', 0))
            traj_deltas = traj.get('temporal_deltas', None)  # len = path_len - 1
            traj_velocities = traj.get('velocities', None)   # len = path_len

            # Track samples for this trajectory
            self.trajectory_samples[traj_idx] = []

            # Create per-node samples from path history
            for step_idx in range(1, len(path)):
                history_path = path[:step_idx]
                next_node = path[step_idx]

                if include_progress:
                    path_progress = step_idx / (total_path_length - 1) if total_path_length > 1 else 0.0
                else:
                    path_progress = 0.0

                # Convert node IDs to indices
                try:
                    history_indices = []
                    for node in history_path:
                        node_id = node[0] if isinstance(node, (list, tuple)) else node
                        history_indices.append(self.node_to_idx[node_id])

                    next_node_id = next_node[0] if isinstance(next_node, (list, tuple)) else next_node
                    next_node_idx = self.node_to_idx[next_node_id]
                except (KeyError, TypeError, IndexError):
                    continue

                # Extract goal category
                goal_cat_idx = 0
                if goal_node in self.graph.nodes:
                    cat_name = self.graph.nodes[goal_node].get('Category', 'other')
                    goal_cat_idx = self.CATEGORY_TO_IDX.get(cat_name, 0)

                # ---- Slice temporal features for this prefix ----
                sample: Dict = {
                    'history_node_indices': history_indices,
                    'next_node_idx': next_node_idx,
                    'goal_cat_idx': goal_cat_idx,
                    'goal_idx': goal_idx,
                    'agent_id': agent_id,
                    'path_progress': path_progress,
                    'traj_id': traj_idx,
                    'step_idx': step_idx,
                    'total_steps': total_path_length - 1,
                    'hour': traj_hour,
                    'day_of_week': traj_day,
                }

                if include_temporal and traj_deltas is not None:
                    # deltas[i] = time from step i to step i+1, so for
                    # history_path = path[:step_idx] we need deltas[:step_idx-1]
                    # but we pad to step_idx to match history length later.
                    sample['history_temporal_deltas'] = traj_deltas[:step_idx]
                if include_temporal and traj_velocities is not None:
                    # velocities[i] = speed at step i, history has step_idx nodes
                    sample['history_velocities'] = traj_velocities[:step_idx]

                sample_idx = len(self.samples)
                self.trajectory_samples[traj_idx].append(sample_idx)
                self.samples.append(sample)
                sample_count += 1

        if missing_agent_count > 0:
            print(f"   ⚠️  Warning: {missing_agent_count} trajectories missing agent_id (skipped)")

        print(f"✅ Created {sample_count} enhanced BDI samples from {valid_count} trajectories")
        print(f"   Average samples per trajectory: {sample_count / max(valid_count, 1):.1f}")
        print(f"   Path progress: {'enabled' if include_progress else 'disabled'}")
        print(f"   Temporal features: {'enabled' if include_temporal else 'disabled'}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
    
    def get_trajectory_samples(self, traj_id: int) -> List[int]:
        """Get all sample indices from a specific trajectory."""
        return self.trajectory_samples.get(traj_id, [])
    
    def get_progress_filtered_indices(
        self, 
        min_progress: float = 0.0, 
        max_progress: float = 1.0
    ) -> List[int]:
        """Get sample indices within a progress range (for curriculum learning)."""
        return [
            i for i, s in enumerate(self.samples)
            if min_progress <= s['path_progress'] <= max_progress
        ]


def collate_bdi_samples_v2(batch: List[Dict]) -> Dict:
    """
    Collate enhanced BDI samples into a padded batch.

    Handles the temporal features (hours, days, deltas, velocities) produced
    by BDIVAEDatasetV2 when ``include_temporal=True``.
    """
    if not batch:
        raise ValueError("Empty batch!")

    # Find max history length
    max_history_len = max(len(s['history_node_indices']) for s in batch)
    max_history_len = max(1, max_history_len)

    padded_histories = []
    history_lengths = []
    padded_deltas = []
    padded_velocities = []
    has_temporal = 'history_temporal_deltas' in batch[0]

    for sample in batch:
        history = sample['history_node_indices']
        h_len = len(history)
        pad_len = max_history_len - h_len
        padded_histories.append(torch.tensor(history + [0] * pad_len, dtype=torch.long))
        history_lengths.append(h_len)

        if has_temporal:
            # deltas – might be shorter than history by 1; pad to max_history_len
            deltas = sample.get('history_temporal_deltas', [])
            padded_deltas.append(
                torch.tensor(deltas + [0.0] * (max_history_len - len(deltas)), dtype=torch.float)
            )
            velocities = sample.get('history_velocities', [])
            padded_velocities.append(
                torch.tensor(velocities + [0.0] * (max_history_len - len(velocities)), dtype=torch.float)
            )

    result = {
        'history_node_indices': torch.stack(padded_histories),
        'history_lengths': torch.tensor(history_lengths, dtype=torch.long),
        'next_node_idx': torch.tensor([s['next_node_idx'] for s in batch], dtype=torch.long),
        'goal_cat_idx': torch.tensor([s['goal_cat_idx'] for s in batch], dtype=torch.long),
        'goal_idx': torch.tensor([s['goal_idx'] for s in batch], dtype=torch.long),
        'agent_id': torch.tensor([s['agent_id'] for s in batch], dtype=torch.long),
        'path_progress': torch.tensor([s['path_progress'] for s in batch], dtype=torch.float),
        'traj_id': torch.tensor([s['traj_id'] for s in batch], dtype=torch.long),
        'step_idx': torch.tensor([s['step_idx'] for s in batch], dtype=torch.long),
        'hour': torch.tensor([s.get('hour', 0) for s in batch], dtype=torch.long),
        'day_of_week': torch.tensor([s.get('day_of_week', 0) for s in batch], dtype=torch.long),
    }

    if has_temporal:
        result['history_temporal_deltas'] = torch.stack(padded_deltas)
        result['history_velocities'] = torch.stack(padded_velocities)

    return result


class TemporalConsistencyBatchSampler:
    """
    Batch sampler that ensures samples from the same trajectory appear together.
    
    This enables computing temporal consistency losses within a batch.
    Each batch contains:
    - N trajectories
    - K consecutive samples from each trajectory
    """
    
    def __init__(
        self,
        dataset: BDIVAEDatasetV2,
        batch_size: int,
        samples_per_trajectory: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: BDIVAEDatasetV2 instance
            batch_size: Total batch size
            samples_per_trajectory: How many samples per trajectory in a batch
            shuffle: Whether to shuffle trajectories
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_trajectory = samples_per_trajectory
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get trajectories with enough samples
        self.valid_trajectories = [
            traj_id for traj_id, samples in dataset.trajectory_samples.items()
            if len(samples) >= samples_per_trajectory
        ]
        
        self.trajectories_per_batch = batch_size // samples_per_trajectory
    
    def __iter__(self):
        if self.shuffle:
            import random
            trajectory_order = random.sample(
                self.valid_trajectories, len(self.valid_trajectories)
            )
        else:
            trajectory_order = self.valid_trajectories.copy()
        
        batch = []
        for traj_id in trajectory_order:
            traj_samples = self.dataset.trajectory_samples[traj_id]
            
            # Select consecutive samples from this trajectory
            if len(traj_samples) >= self.samples_per_trajectory:
                import random
                start_idx = random.randint(
                    0, len(traj_samples) - self.samples_per_trajectory
                )
                selected = traj_samples[start_idx:start_idx + self.samples_per_trajectory]
                batch.extend(selected)
            
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        total = len(self.valid_trajectories) * self.samples_per_trajectory
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size
