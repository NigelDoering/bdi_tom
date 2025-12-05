"""Utilities for Experiment 3 belief-update analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from graph_controller.world_graph import WorldGraph


@dataclass
class PivotEpisode:
    """Trajectory containing a belief-update pivot event."""

    agent_id: str
    episode_index: int
    hour: int
    first_goal: str
    final_goal: str
    category: str
    pivot_index: int
    path: List[List[str]]

    @property
    def pre_length(self) -> int:
        return self.pivot_index

    @property
    def post_length(self) -> int:
        return max(0, len(self.path) - self.pivot_index)

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "episode_index": self.episode_index,
            "hour": self.hour,
            "first_goal": self.first_goal,
            "final_goal": self.final_goal,
            "category": self.category,
            "pivot_index": self.pivot_index,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PivotEpisode":
        return cls(
            agent_id=str(payload["agent_id"]),
            episode_index=int(payload["episode_index"]),
            hour=int(payload["hour"]),
            first_goal=str(payload["first_goal"]),
            final_goal=str(payload["final_goal"]),
            category=str(payload["category"]),
            pivot_index=int(payload["pivot_index"]),
            path=[list(step) for step in payload["path"]],
        )


@dataclass
class PhaseSample:
    """Model evaluation sample for a specific observation fraction."""

    episode_idx: int
    phase: str
    fraction: float
    path: List[List[str]]
    hour: int
    target_goal: str


PRE_PHASE = "pre"
POST_PHASE = "post"


def load_trajectories(path: Path) -> Dict[str, List[MutableMapping[str, object]]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw


def _node_category(world_graph: WorldGraph, node_id: str) -> Optional[str]:
    if node_id not in world_graph.G:
        return None
    return world_graph.G.nodes[node_id].get("Category")


def _find_pivot_index(path: Sequence[Sequence[str]], final_goal: str) -> Optional[int]:
    for idx, step in enumerate(path):
        if len(step) < 2:
            continue
        if str(step[1]) == final_goal:
            return idx
    return None


def extract_pivot_episodes(
    trajectories: Mapping[str, Sequence[Mapping[str, object]]],
    world_graph: WorldGraph,
) -> List[PivotEpisode]:
    episodes: List[PivotEpisode] = []
    for agent_id, runs in trajectories.items():
        for episode_index, payload in enumerate(runs):
            attempted = payload.get("attempted_goals", [])
            if not isinstance(attempted, list) or len(attempted) != 2:
                continue
            returned_home = bool(payload.get("returned_home", False))
            if returned_home:
                continue
            first_goal, second_goal = map(str, attempted)
            final_goal = str(payload.get("goal_node", ""))
            if final_goal != second_goal:
                continue
            if first_goal == final_goal:
                continue
            path = payload.get("path")
            if not isinstance(path, list) or not path:
                continue
            pivot_index = _find_pivot_index(path, final_goal)
            if pivot_index is None or pivot_index <= 0:
                continue
            if pivot_index >= len(path):
                continue
            first_category = _node_category(world_graph, first_goal)
            final_category = _node_category(world_graph, final_goal)
            if first_category is None or final_category is None:
                continue
            if first_category != final_category:
                continue
            episode = PivotEpisode(
                agent_id=str(agent_id),
                episode_index=episode_index,
                hour=int(payload.get("hour", 0)),
                first_goal=first_goal,
                final_goal=final_goal,
                category=first_category,
                pivot_index=pivot_index,
                path=[list(step) for step in path],
            )
            if episode.pre_length < 2 or episode.post_length < 2:
                continue
            episodes.append(episode)
    return episodes


def generate_phase_samples(
    episodes: Sequence[PivotEpisode],
    pre_fractions: Sequence[float],
    post_fractions: Sequence[float],
) -> Tuple[List[PhaseSample], int]:
    samples: List[PhaseSample] = []
    max_len = 0
    for idx, episode in enumerate(episodes):
        for fraction in pre_fractions:
            steps = max(1, math.floor(episode.pre_length * fraction))
            if steps >= episode.pre_length:
                steps = episode.pre_length - 1
            if steps <= 0:
                continue
            path_prefix = episode.path[:steps]
            samples.append(
                PhaseSample(
                    episode_idx=idx,
                    phase=PRE_PHASE,
                    fraction=fraction,
                    path=path_prefix,
                    hour=episode.hour,
                    target_goal=episode.first_goal,
                )
            )
            max_len = max(max_len, len(path_prefix))
        for fraction in post_fractions:
            steps = max(1, math.floor(episode.post_length * fraction))
            steps = min(steps, episode.post_length)
            prefix_len = episode.pivot_index + steps
            if prefix_len > len(episode.path):
                prefix_len = len(episode.path)
            path_prefix = episode.path[:prefix_len]
            samples.append(
                PhaseSample(
                    episode_idx=idx,
                    phase=POST_PHASE,
                    fraction=fraction,
                    path=path_prefix,
                    hour=episode.hour,
                    target_goal=episode.final_goal,
                )
            )
            max_len = max(max_len, len(path_prefix))
    return samples, max_len


def write_pivot_dataset(output_path: Path, episodes: Sequence[PivotEpisode], run_dir: Path, graph_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "run_dir": str(run_dir),
            "graph_path": str(graph_path),
            "total_agents": len({ep.agent_id for ep in episodes}),
        },
        "summary": {
            "total_episodes": len(episodes),
        },
        "episodes": [episode.to_dict() for episode in episodes],
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_pivot_dataset(path: Path) -> List[PivotEpisode]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    episodes_raw = payload.get("episodes", [])
    return [PivotEpisode.from_dict(obj) for obj in episodes_raw]


__all__ = [
    "PivotEpisode",
    "PhaseSample",
    "PRE_PHASE",
    "POST_PHASE",
    "extract_pivot_episodes",
    "generate_phase_samples",
    "load_pivot_dataset",
    "load_trajectories",
    "write_pivot_dataset",
]
