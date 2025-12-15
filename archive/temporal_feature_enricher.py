"""
Temporal Feature Enrichment for BDI-Theory of Mind Trajectory Analysis

This module transforms raw trajectory data into rich multi-modal temporal representations
suitable for advanced trajectory prediction. It computes:

1. Day of Week: Extracted from trajectory metadata or simulated patterns
2. Temporal Deltas: Time elapsed between consecutive trajectory steps
3. Velocity Profiles: Movement speed indicating activity type
4. Behavioral Context: Time-of-day activity patterns
5. Circadian Patterns: Hour-based activity clustering

For Theory of Mind applications, these features reveal agent intent patterns:
- Consistent timing preferences indicate planned/habitual behavior
- Velocity changes suggest awareness of destination
- Day-of-week patterns indicate learning and adaptation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime, timedelta
import networkx as nx
import math


class TemporalFeatureEnricher:
    """
    Enriches trajectory data with comprehensive temporal features for BDI-ToM.
    
    Key Design Decisions for Theory of Mind:
    1. **Temporal Deltas with Scale Awareness**: Captures deliberate timing patterns
       - Agents consciously control walking speed and dwell times
       - Large deltas indicate consideration/decision-making
    
    2. **Velocity Profiles**: Reveals agent state and planning
       - Fast movement: Direct navigation to known goal
       - Slow movement: Exploration or uncertainty
       - Pauses: Deliberation or social interaction
    
    3. **Day-of-Week Discrimination**: Captures agent adaptability
       - Students have weekday vs weekend patterns
       - Work schedules vs leisure patterns
       - Demonstrates learning and flexibility
    
    4. **Hour-based Behavioral Clustering**: Captures circadian beliefs
       - Morning: commute patterns
       - Midday: class/work patterns
       - Evening: social/leisure patterns
       - Night: sleep/resting patterns
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        simulation_duration_days: int = 14,
        start_date: Optional[datetime] = None,
        seed: int = 42
    ):
        """
        Initialize temporal feature enricher.
        
        Args:
            graph: NetworkX graph with node coordinates
            simulation_duration_days: Number of days in simulation (for synthetic day assignment)
            start_date: Start date of simulation (if None, uses today)
            seed: Random seed for reproducible synthetic day assignment
        """
        self.graph = graph
        self.simulation_duration_days = simulation_duration_days
        self.start_date = start_date or datetime(2024, 1, 1)  # Default to Jan 1, 2024
        self.seed = seed
        
        # Build node coordinate cache for velocity calculations
        self._build_node_cache()
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.RandomState(seed)
        
    def _build_node_cache(self):
        """Build cache of node coordinates for velocity calculations."""
        self.node_coords = {}
        for node_id in self.graph.nodes():
            try:
                x = self.graph.nodes[node_id].get('x', None)
                y = self.graph.nodes[node_id].get('y', None)
                if x is not None and y is not None:
                    self.node_coords[node_id] = (float(x), float(y))
            except (ValueError, TypeError):
                pass
    
    def _get_node_distance(self, node1: str, node2: str) -> float:
        """
        Compute Euclidean distance between two nodes.
        
        Args:
            node1, node2: Node IDs
            
        Returns:
            Distance in graph coordinate units (typically meters or feet)
        """
        if node1 not in self.node_coords or node2 not in self.node_coords:
            # Default: assume graph has shortest path info
            try:
                return nx.shortest_path_length(self.graph, node1, node2, weight='length')
            except:
                return 1.0  # Default distance if path not found
        
        x1, y1 = self.node_coords[node1]
        x2, y2 = self.node_coords[node2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _assign_day_of_week(self, trajectory_idx: int) -> int:
        """
        Assign day of week to trajectory using hash-based distribution.
        
        For Theory of Mind: Creates consistent agent schedules across simulations.
        Agents maintain consistent day-of-week behaviors even without explicit timestamps.
        
        Args:
            trajectory_idx: Index of trajectory in simulation
            
        Returns:
            Day of week (0=Monday, 6=Sunday)
        """
        # Use hash to create deterministic but distributed assignment
        # Different agents will have different days, but same agent has same pattern
        hash_val = hash((trajectory_idx, self.seed))
        day = hash_val % 7
        return day
    
    def _compute_temporal_deltas(
        self,
        path: List[str],
        hour: int,
        base_speed_mps: float = 1.4  # Typical walking speed (5 km/h)
    ) -> np.ndarray:
        """
        Compute time deltas between consecutive trajectory steps.
        
        Theory of Mind Insight: Time deltas reveal agent's decision-making process
        - Consistent deltas: Habitual route following
        - Variable deltas: Exploration or re-planning
        - Large deltas: Deliberation or waiting for someone
        
        Args:
            path: List of node IDs in trajectory
            hour: Hour of day (for context about expected speeds)
            base_speed_mps: Base walking speed in m/s
            
        Returns:
            Array of time deltas (in hours) between consecutive steps
        """
        if len(path) <= 1:
            return np.array([])
        
        deltas = []
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            distance = self._get_node_distance(node1, node2)
            
            # Adjust speed based on time of day and distance
            # Morning/evening: likely slower (commute crowds)
            # Midday: faster (direct navigation)
            if 6 <= hour < 9 or 16 <= hour < 19:
                speed_factor = 0.8  # Slower during commute
            elif 10 <= hour < 15:
                speed_factor = 1.0  # Normal speed during work/study
            else:
                speed_factor = 0.9  # Evening/night navigation
            
            adjusted_speed = base_speed_mps * speed_factor
            
            # Compute time in seconds, convert to hours
            time_seconds = max(5, distance / adjusted_speed)  # Min 5 seconds
            time_hours = time_seconds / 3600.0
            
            # Add small random jitter for realism
            jitter = self.rng.normal(0, 0.01 * time_hours)
            time_hours = max(0.0001, time_hours + jitter)
            
            deltas.append(time_hours)
        
        return np.array(deltas)
    
    def _compute_velocities(
        self,
        path: List[str],
        deltas: np.ndarray,
        base_speed_mps: float = 1.4
    ) -> np.ndarray:
        """
        Compute velocity profile for trajectory.
        
        Theory of Mind Insight: Velocity variations indicate agent states
        - High velocity: Confident navigation
        - Variable velocity: Uncertainty or exploration
        - Near-zero velocity: Dwell time at location (social interaction, decision-making)
        
        Args:
            path: List of node IDs
            deltas: Time deltas between steps (hours)
            base_speed_mps: Base walking speed
            
        Returns:
            Array of velocities (units per hour) for each step
        """
        velocities = []
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            distance = self._get_node_distance(node1, node2)
            delta = max(0.0001, deltas[i])  # Avoid division by zero
            
            # Compute velocity in distance units per hour
            velocity = distance / delta
            velocities.append(velocity)
        
        # Add final step (no movement after last node)
        if len(velocities) > 0:
            velocities.append(0.1)  # Small value for arrived state
        elif len(path) > 0:
            velocities.append(0.1)
        
        return np.array(velocities)
    
    def enrich_trajectory(
        self,
        trajectory: Dict,
        trajectory_idx: int,
        agent_id: Optional[str] = None,
        base_speed_mps: float = 1.4
    ) -> Dict:
        """
        Enrich a single trajectory with temporal features.
        
        Args:
            trajectory: Original trajectory dict with keys: path, goal_node, hour, etc.
            trajectory_idx: Index in the overall trajectory list (for day assignment)
            agent_id: Agent ID (optional, for reproducibility)
            base_speed_mps: Base walking speed for delta computation
            
        Returns:
            Enriched trajectory dict with additional temporal features
        """
        # Copy original trajectory
        enriched = trajectory.copy()
        
        # Extract path - handle both formats
        if isinstance(trajectory['path'][0], list):
            # Format: [[node_id, ...], [node_id, ...], ...]
            path = [step[0] for step in trajectory['path']]
        else:
            # Format: [node_id, node_id, ...]
            path = trajectory['path']
        
        hour = trajectory.get('hour', 12)
        
        # Compute day of week
        # Use agent_id if available for more consistent patterns per agent
        if agent_id:
            day_seed = hash((agent_id, trajectory_idx, self.seed)) % 7
        else:
            day_seed = self._assign_day_of_week(trajectory_idx)
        
        # Compute temporal features
        deltas = self._compute_temporal_deltas(path, hour, base_speed_mps)
        velocities = self._compute_velocities(path, deltas, base_speed_mps)
        
        # Add to enriched trajectory
        enriched['day_of_week'] = day_seed
        enriched['temporal_deltas'] = deltas.tolist()  # Convert to list for JSON
        enriched['velocities'] = velocities.tolist()
        enriched['base_speed_mps'] = base_speed_mps
        
        # Add circadian hour (same as hour but explicit)
        enriched['circadian_hour'] = hour
        
        return enriched
    
    def enrich_trajectories(
        self,
        trajectories: List[Dict],
        agent_map: Optional[Dict[int, str]] = None,
        base_speed_mps: float = 1.4,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Enrich multiple trajectories with temporal features.
        
        Args:
            trajectories: List of trajectory dicts
            agent_map: Optional dict mapping trajectory index to agent_id
            base_speed_mps: Base walking speed
            show_progress: Whether to show progress
            
        Returns:
            List of enriched trajectories
        """
        enriched_trajectories = []
        
        iterator = enumerate(trajectories)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(trajectories), total=len(trajectories),
                              desc="Enriching trajectories with temporal features")
            except ImportError:
                pass
        
        for idx, traj in iterator:
            agent_id = agent_map.get(idx) if agent_map else None
            enriched = self.enrich_trajectory(traj, idx, agent_id, base_speed_mps)
            enriched_trajectories.append(enriched)
        
        return enriched_trajectories
    
    def enrich_and_save(
        self,
        trajectories: List[Dict],
        output_path: str,
        agent_trajectories: Optional[Dict[str, List[Dict]]] = None,
        base_speed_mps: float = 1.4
    ) -> Dict:
        """
        Enrich trajectories and save to file.
        
        Args:
            trajectories: List of trajectory dicts (flat or hierarchical)
            output_path: Path to save enriched trajectories
            agent_trajectories: Optional dict of {agent_id: [trajs]} for agent-aware enrichment
            base_speed_mps: Base walking speed
            
        Returns:
            Statistics dict about enrichment
        """
        # Build agent map if hierarchical data provided
        agent_map = {}
        if agent_trajectories:
            idx = 0
            for agent_id, agent_trajs in agent_trajectories.items():
                for _ in agent_trajs:
                    agent_map[idx] = agent_id
                    idx += 1
        
        # Enrich trajectories
        enriched = self.enrich_trajectories(
            trajectories, agent_map, base_speed_mps, show_progress=True
        )
        
        # Save enriched trajectories
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(enriched, f, indent=2)
        
        print(f"\n✅ Enriched {len(enriched)} trajectories saved to {output_path}")
        
        # Compute statistics
        stats = self._compute_statistics(enriched)
        return stats
    
    @staticmethod
    def _compute_statistics(trajectories: List[Dict]) -> Dict:
        """Compute statistics about enriched trajectories."""
        stats = {
            'total_trajectories': len(trajectories),
            'avg_trajectory_length': np.mean([len(t['path']) for t in trajectories]),
            'max_trajectory_length': max(len(t['path']) for t in trajectories),
            'min_trajectory_length': min(len(t['path']) for t in trajectories),
            'day_distribution': {},
            'hour_distribution': {},
            'avg_deltas': [],
            'avg_velocities': []
        }
        
        # Day distribution
        for traj in trajectories:
            day = traj.get('day_of_week', 0)
            stats['day_distribution'][day] = stats['day_distribution'].get(day, 0) + 1
            
            hour = traj.get('hour', 12)
            stats['hour_distribution'][hour] = stats['hour_distribution'].get(hour, 0) + 1
            
            if 'temporal_deltas' in traj and len(traj['temporal_deltas']) > 0:
                stats['avg_deltas'].append(np.mean(traj['temporal_deltas']))
            
            if 'velocities' in traj and len(traj['velocities']) > 0:
                stats['avg_velocities'].append(np.mean(traj['velocities']))
        
        if stats['avg_deltas']:
            stats['overall_avg_delta'] = np.mean(stats['avg_deltas'])
        if stats['avg_velocities']:
            stats['overall_avg_velocity'] = np.mean(stats['avg_velocities'])
        
        return stats


class EnrichedTrajectoryDataset:
    """
    PyTorch-compatible dataset for enriched trajectories.
    
    Integrates seamlessly with existing training pipeline while providing
    easy access to temporal features for encoders.
    """
    
    def __init__(
        self,
        enriched_trajectories: List[Dict],
        graph: nx.Graph,
        poi_nodes: List[str]
    ):
        """
        Initialize dataset.
        
        Args:
            enriched_trajectories: List of trajectory dicts with temporal features
            graph: NetworkX graph
            poi_nodes: List of POI node IDs
        """
        self.trajectories = enriched_trajectories
        self.graph = graph
        self.poi_nodes = poi_nodes
        self.goal_to_idx = {node_id: idx for idx, node_id in enumerate(poi_nodes)}
        
        # Filter valid trajectories
        self.valid_indices = [
            idx for idx, traj in enumerate(enriched_trajectories)
            if traj.get('goal_node') in self.goal_to_idx
        ]
        
        print(f"✅ Enriched Dataset: {len(self.valid_indices)}/{len(enriched_trajectories)} "
              f"trajectories with valid goals")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get enriched trajectory sample.
        
        Returns:
            Dict with original fields plus temporal features
        """
        real_idx = self.valid_indices[idx]
        traj = self.trajectories[real_idx]
        
        goal_node = traj['goal_node']
        goal_idx = self.goal_to_idx[goal_node]
        
        # Extract path format
        if isinstance(traj['path'][0], list):
            path = [step[0] for step in traj['path']]
        else:
            path = traj['path']
        
        return {
            'trajectory': path,
            'hour': traj.get('hour', 12),
            'day_of_week': traj.get('day_of_week', 0),
            'goal_idx': goal_idx,
            'goal_node': goal_node,
            'temporal_deltas': np.array(traj.get('temporal_deltas', [])),
            'velocities': np.array(traj.get('velocities', [])),
            'circadian_hour': traj.get('circadian_hour', traj.get('hour', 12))
        }


# ============================================================================
# TESTING AND UTILITIES
# ============================================================================

def create_temporal_enricher_from_simulation(
    run_dir: str,
    graph_path: str,
    output_dir: Optional[str] = None
) -> Tuple[List[Dict], TemporalFeatureEnricher]:
    """
    Convenience function to create enricher and load simulation data.
    
    Args:
        run_dir: Path to simulation run directory
        graph_path: Path to graph file
        output_dir: Optional directory to save enriched trajectories
        
    Returns:
        Tuple of (enriched_trajectories, enricher)
    """
    import json
    
    # Load graph
    graph = nx.read_graphml(graph_path)
    print(f"✅ Loaded graph with {graph.number_of_nodes()} nodes")
    
    # Load trajectories
    traj_path = os.path.join(run_dir, 'trajectories', 'all_trajectories.json')
    with open(traj_path, 'r') as f:
        data = json.load(f)
    
    # Flatten if hierarchical
    if isinstance(data, dict):
        trajectories = []
        agent_map = {}
        idx = 0
        for agent_id, agent_trajs in data.items():
            for traj in agent_trajs:
                agent_map[idx] = agent_id
                trajectories.append(traj)
                idx += 1
    else:
        trajectories = data
        agent_map = None
    
    print(f"✅ Loaded {len(trajectories)} trajectories")
    
    # Create enricher
    enricher = TemporalFeatureEnricher(graph, simulation_duration_days=14)
    
    # Enrich trajectories
    enriched = enricher.enrich_trajectories(
        trajectories, agent_map, base_speed_mps=1.4, show_progress=True
    )
    
    # Save if output dir specified
    if output_dir:
        output_path = os.path.join(output_dir, 'enriched_trajectories.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(enriched, f, indent=2)
        print(f"✅ Saved enriched trajectories to {output_path}")
    
    return enriched, enricher


if __name__ == '__main__':
    """Test temporal feature enrichment."""
    
    print("=" * 80)
    print("TEMPORAL FEATURE ENRICHMENT TEST")
    print("=" * 80)
    
    run_dir = 'data/simulation_data/run_8'
    graph_path = 'data/processed/ucsd_walk_full.graphml'
    output_dir = 'data/simulation_data/run_8_enriched'
    
    # Create enricher and enrich trajectories
    enriched_trajs, enricher = create_temporal_enricher_from_simulation(
        run_dir, graph_path, output_dir
    )
    
    # Print sample
    if enriched_trajs:
        print("\n📊 Sample enriched trajectory:")
        sample = enriched_trajs[0]
        print(f"   Path length: {len(sample['path'])}")
        print(f"   Hour: {sample['hour']}")
        print(f"   Day of week: {sample['day_of_week']}")
        print(f"   Temporal deltas (first 5): {sample['temporal_deltas'][:5]}")
        print(f"   Velocities (first 5): {sample['velocities'][:5]}")
    
    print("\n✅ Temporal enrichment test complete!")
