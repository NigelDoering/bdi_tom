"""simulation_controller.belief_store
--------------------------------------
Efficient storage for agent belief snapshots captured during simulation.

Each trajectory produces four arrays stored in a compressed numpy archive
(all_beliefs.npz):

    {agent_id}_{traj_idx:04d}_beliefs  float16  [n_snaps, n_pois, 24]
        temporal_belief probabilities at each snapshot step

    {agent_id}_{traj_idx:04d}_steps    int16    [n_snaps]
        path step index at which each snapshot was taken

    {agent_id}_{traj_idx:04d}_alpha    float32  [n_pois, 24]
        Beta alpha parameters at trajectory START (before any traversal)

    {agent_id}_{traj_idx:04d}_beta     float32  [n_pois, 24]
        Beta beta parameters at trajectory START (before any traversal)

A companion belief_metadata.json provides a human-readable index with the
ordered poi_nodes list and per-trajectory snapshot counts so callers can
query beliefs without loading the full NPZ.

Design choices:
- float16 for snapshots: probabilities are in [0.01, 0.99]; float16 gives
  ~0.001 precision, more than enough for analysis and ML training, and is
  3.5× smaller than float32.
- float32 for alpha/beta: these accumulate over observations and can reach
  values like 104.0; float32 preserves exact Bayesian state.
- Initial state (alpha/beta) captured before traversal begins, giving a
  complete record of the agent's accumulated cross-trajectory knowledge.
- Separate from all_trajectories.json so path data and belief data can be
  loaded independently.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


class BeliefStore:
    """Records and persists agent belief snapshots for an entire simulation run.

    Usage
    -----
    Before simulation::

        store = BeliefStore(poi_nodes=world_graph.poi_nodes)

    During simulation (called by Simulation.step)::

        # once per trajectory, BEFORE traversal starts:
        store.record_initial_state(agent_id, traj_idx, agent.belief_state)

        # once every `snapshot_interval` steps during traversal:
        store.record_snapshot(agent_id, traj_idx, step_idx, agent.belief_state)

    After simulation::

        store.save("data/simulation_data/run_X/beliefs")

    Loading later::

        store = BeliefStore.load("data/simulation_data/run_X")
        data  = store.get_trajectory("agent_000", 3)
        # data["beliefs"]  -> float16 array [n_snaps, n_pois, 24]
        # data["steps"]    -> int16   array [n_snaps]
        # data["alpha"]    -> float32 array [n_pois, 24]  (initial state)
        # data["beta"]     -> float32 array [n_pois, 24]  (initial state)
    """

    _NPZ_FILENAME = "all_beliefs.npz"
    _META_FILENAME = "belief_metadata.json"

    def __init__(
        self,
        poi_nodes: List[str],
        snapshot_interval: int = 5,
    ):
        """
        Args:
            poi_nodes: Ordered list of POI node ID strings. The order defines
                the row dimension of all stored arrays.
            snapshot_interval: Path step interval between snapshots (default 5,
                matching the simulation's update_beliefs cadence).
        """
        self.poi_nodes: List[str] = list(poi_nodes)
        self.n_pois: int = len(poi_nodes)
        self.poi_to_idx: Dict[str, int] = {n: i for i, n in enumerate(poi_nodes)}
        self.snapshot_interval: int = snapshot_interval

        # Internal buffer: key -> {"beliefs": list of [n_pois,24] float16,
        #                           "steps":   list of int,
        #                           "alpha":   [n_pois, 24] float32  (initial),
        #                           "beta":    [n_pois, 24] float32  (initial)}
        self._buffer: Dict[str, dict] = {}

        # Set after load(); None when building from scratch
        self._loaded_npz: Optional[np.lib.npyio.NpzFile] = None

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    @staticmethod
    def _traj_key(agent_id: str, traj_idx: int) -> str:
        return f"{agent_id}_{traj_idx:04d}"

    def record_initial_state(
        self,
        agent_id: str,
        traj_idx: int,
        belief_state: dict,
    ) -> None:
        """Capture the agent's alpha/beta arrays at the START of a trajectory.

        This must be called once per trajectory *before* any traversal steps,
        so the stored values reflect accumulated cross-trajectory knowledge.

        Args:
            agent_id: Agent identifier string.
            traj_idx: Zero-based trajectory index for this agent.
            belief_state: agent.belief_state dict mapping node_id ->
                {"alpha": np.ndarray[24], "beta": np.ndarray[24], ...}
        """
        alpha = np.zeros((self.n_pois, 24), dtype=np.float32)
        beta  = np.zeros((self.n_pois, 24), dtype=np.float32)

        for node_id, data in belief_state.items():
            idx = self.poi_to_idx.get(node_id)
            if idx is None:
                continue
            alpha[idx] = data["alpha"].astype(np.float32)
            beta[idx]  = data["beta"].astype(np.float32)

        key = self._traj_key(agent_id, traj_idx)
        self._buffer[key] = {
            "beliefs": [],   # list of [n_pois, 24] float16 arrays
            "steps":   [],   # list of int step indices
            "alpha":   alpha,
            "beta":    beta,
        }

    def record_snapshot(
        self,
        agent_id: str,
        traj_idx: int,
        step_idx: int,
        belief_state: dict,
    ) -> None:
        """Append a single belief snapshot at the given path step.

        Args:
            agent_id: Agent identifier string.
            traj_idx: Zero-based trajectory index for this agent.
            step_idx: Current path position (0-indexed).
            belief_state: agent.belief_state dict mapping node_id ->
                {"temporal_belief": np.ndarray[24], ...}
        """
        key = self._traj_key(agent_id, traj_idx)
        if key not in self._buffer:
            # Gracefully handle missing initial state (e.g. called standalone)
            self._buffer[key] = {
                "beliefs": [],
                "steps":   [],
                "alpha":   np.zeros((self.n_pois, 24), dtype=np.float32),
                "beta":    np.zeros((self.n_pois, 24), dtype=np.float32),
            }

        snapshot = np.zeros((self.n_pois, 24), dtype=np.float16)
        for node_id, data in belief_state.items():
            idx = self.poi_to_idx.get(node_id)
            if idx is None:
                continue
            snapshot[idx] = data["temporal_belief"].astype(np.float16)

        self._buffer[key]["beliefs"].append(snapshot)
        self._buffer[key]["steps"].append(step_idx)

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------

    def save(self, beliefs_dir: str) -> None:
        """Write all_beliefs.npz and belief_metadata.json to beliefs_dir.

        Args:
            beliefs_dir: Directory path (will be created if absent).
        """
        os.makedirs(beliefs_dir, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {}
        meta_trajs: Dict[str, dict] = {}

        # Store POI node list as object array for easy recovery
        arrays["poi_nodes"] = np.array(self.poi_nodes, dtype=object)

        for key, buf in self._buffer.items():
            n_snaps = len(buf["beliefs"])
            if n_snaps == 0:
                continue

            beliefs_arr = np.stack(buf["beliefs"], axis=0)   # [n_snaps, n_pois, 24] float16
            steps_arr   = np.array(buf["steps"], dtype=np.int16)  # [n_snaps]

            arrays[f"{key}_beliefs"] = beliefs_arr
            arrays[f"{key}_steps"]   = steps_arr
            arrays[f"{key}_alpha"]   = buf["alpha"]   # [n_pois, 24] float32
            arrays[f"{key}_beta"]    = buf["beta"]    # [n_pois, 24] float32

            meta_trajs[key] = {
                "n_snapshots":  n_snaps,
                "step_indices": steps_arr.tolist(),
            }

        npz_path = os.path.join(beliefs_dir, self._NPZ_FILENAME)
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "poi_nodes":        self.poi_nodes,
            "n_pois":           self.n_pois,
            "snapshot_interval": self.snapshot_interval,
            "n_trajectories":   len(meta_trajs),
            "trajectories":     meta_trajs,
        }
        meta_path = os.path.join(beliefs_dir, self._META_FILENAME)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"✅ BeliefStore saved: {len(meta_trajs)} trajectories → {beliefs_dir}\n"
            f"   NPZ: {os.path.getsize(npz_path) / 1e6:.1f} MB"
        )

    @classmethod
    def load(cls, run_dir: str) -> "BeliefStore":
        """Load a BeliefStore from a run directory.

        Args:
            run_dir: Root run directory (e.g. data/simulation_data/run_X/).
                     Expects a beliefs/ subdirectory containing the two files.

        Returns:
            BeliefStore instance with _loaded_npz set for lazy array access.
        """
        beliefs_dir = os.path.join(run_dir, "beliefs")
        meta_path = os.path.join(beliefs_dir, cls._META_FILENAME)
        npz_path  = os.path.join(beliefs_dir, cls._NPZ_FILENAME)

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"belief_metadata.json not found in {beliefs_dir}")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"all_beliefs.npz not found in {beliefs_dir}")

        with open(meta_path) as f:
            meta = json.load(f)

        store = cls(
            poi_nodes=meta["poi_nodes"],
            snapshot_interval=meta.get("snapshot_interval", 5),
        )
        store._loaded_npz = np.load(npz_path, allow_pickle=True)
        return store

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_trajectory(self, agent_id: str, traj_idx: int) -> dict:
        """Return all belief data for one trajectory.

        Works both for freshly-built (in-memory buffer) and loaded stores.

        Returns:
            dict with keys:
                "beliefs"  float16 [n_snaps, n_pois, 24]
                "steps"    int16   [n_snaps]
                "alpha"    float32 [n_pois, 24]   (initial state)
                "beta"     float32 [n_pois, 24]   (initial state)
        """
        key = self._traj_key(agent_id, traj_idx)

        # In-memory buffer path
        if key in self._buffer:
            buf = self._buffer[key]
            n_snaps = len(buf["beliefs"])
            if n_snaps == 0:
                beliefs_arr = np.zeros((0, self.n_pois, 24), dtype=np.float16)
                steps_arr   = np.zeros(0, dtype=np.int16)
            else:
                beliefs_arr = np.stack(buf["beliefs"], axis=0)
                steps_arr   = np.array(buf["steps"], dtype=np.int16)
            return {
                "beliefs": beliefs_arr,
                "steps":   steps_arr,
                "alpha":   buf["alpha"],
                "beta":    buf["beta"],
            }

        # Loaded NPZ path
        if self._loaded_npz is not None:
            npz = self._loaded_npz
            bkey = f"{key}_beliefs"
            if bkey not in npz:
                raise KeyError(f"Trajectory '{key}' not found in loaded NPZ.")
            return {
                "beliefs": npz[f"{key}_beliefs"],
                "steps":   npz[f"{key}_steps"],
                "alpha":   npz[f"{key}_alpha"],
                "beta":    npz[f"{key}_beta"],
            }

        raise RuntimeError(
            "BeliefStore has no data: neither in-memory buffer nor loaded NPZ."
        )

    def get_belief_at_snapshot(
        self,
        agent_id: str,
        traj_idx: int,
        snapshot_idx: int,
    ) -> np.ndarray:
        """Return [n_pois, 24] float16 belief array for a single snapshot.

        Args:
            agent_id: Agent identifier.
            traj_idx: Zero-based trajectory index.
            snapshot_idx: Index into the snapshot sequence (0 = first snapshot).
        """
        data = self.get_trajectory(agent_id, traj_idx)
        return data["beliefs"][snapshot_idx]

    def list_trajectories(self) -> List[str]:
        """Return all trajectory keys currently in the store."""
        if self._loaded_npz is not None:
            return [
                k[: -len("_beliefs")]
                for k in self._loaded_npz.files
                if k.endswith("_beliefs")
            ]
        return list(self._buffer.keys())
