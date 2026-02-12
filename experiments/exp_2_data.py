"""Utilities for Experiment 2 distractor evaluation data preparation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class DistractorEpisode:
    """Single distractor episode as stored by the generator."""

    agent_id: str
    preferred_category: str
    preferred_goal: str
    distractor_category: str
    distractor_goal: str
    start_node: str
    observation_hour: int
    path: List[List[str]]
    path_length: int
    path_meters: float
    min_distance_to_distractor: float
    closest_index: int

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DistractorEpisode":
        return cls(
            agent_id=str(payload["agent_id"]),
            preferred_category=str(payload["preferred_category"]),
            preferred_goal=str(payload["preferred_goal"]),
            distractor_category=str(payload["distractor_category"]),
            distractor_goal=str(payload["distractor_goal"]),
            start_node=str(payload["start_node"]),
            observation_hour=int(payload["observation_hour"]),
            path=[list(step) for step in payload["path"]],
            path_length=int(payload["path_length"]),
            path_meters=float(payload["path_meters"]),
            min_distance_to_distractor=float(payload["min_distance_to_distractor"]),
            closest_index=int(payload["closest_index"]),
        )


@dataclass
class PrefixSample:
    """Truncated trajectory sample for a particular observation fraction."""

    episode_idx: int
    fraction: float
    path: List[List[str]]
    observation_hour: int
    preferred_goal: str
    distractor_goal: str
    closest_index: int


def load_distractor_episodes(path: Path) -> List[DistractorEpisode]:
    """Read distractor episodes produced by the generator."""

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    episodes_raw = payload.get("episodes")
    if episodes_raw is None:
        raise ValueError("Missing 'episodes' field in distractor payload")

    episodes = [DistractorEpisode.from_dict(obj) for obj in episodes_raw]
    return episodes


def prefix_length(fraction: float, path_length: int) -> int:
    """Compute number of steps to keep based on fraction and total path length."""

    max_steps = max(1, math.floor(fraction * path_length))
    return min(max_steps, path_length)


def generate_prefix_samples(
    episodes: Sequence[DistractorEpisode],
    fractions: Sequence[float],
) -> List[PrefixSample]:
    """Generate truncated trajectory samples for all observation fractions.
    
    Per the paper methodology, fractions are computed relative to the prefix
    trajectory v_{1:t*} where t* is closest_index (point of closest approach
    to distractor), NOT relative to the full path length.
    """

    samples: List[PrefixSample] = []
    for idx, episode in enumerate(episodes):
        # Use closest_index as the reference length per paper methodology
        # closest_index is the point of minimum distance to distractor (t*)
        reference_length = episode.closest_index
        
        # Skip episodes where closest_index is too small for meaningful prefixes
        if reference_length < 2:
            continue
            
        for fraction in fractions:
            steps = prefix_length(fraction, reference_length)
            #steps = prefix_length(fraction, episode.path_length)
            truncated_path = episode.path[:steps]
            samples.append(
                PrefixSample(
                    episode_idx=idx,
                    fraction=fraction,
                    path=truncated_path,
                    observation_hour=episode.observation_hour,
                    preferred_goal=episode.preferred_goal,
                    distractor_goal=episode.distractor_goal,
                    closest_index=episode.closest_index,
                )
            )
    return samples


def generate_prefix_samples_full_trajectory(
    episodes: Sequence[DistractorEpisode],
    fractions: Sequence[float],
) -> List[PrefixSample]:
    """Generate truncated trajectory samples relative to FULL trajectory length.
    
    Unlike generate_prefix_samples which uses t* (closest approach to distractor),
    this function computes fractions relative to the entire path from start to goal.
    
    f=0.5 means: see floor(0.5 × path_length) steps of the full trajectory.
    """

    samples: List[PrefixSample] = []
    for idx, episode in enumerate(episodes):
        # Use full path length as reference
        reference_length = episode.path_length
        
        # Skip episodes that are too short
        if reference_length < 2:
            continue
            
        for fraction in fractions:
            steps = prefix_length(fraction, reference_length)
            truncated_path = episode.path[:steps]
            samples.append(
                PrefixSample(
                    episode_idx=idx,
                    fraction=fraction,
                    path=truncated_path,
                    observation_hour=episode.observation_hour,
                    preferred_goal=episode.preferred_goal,
                    distractor_goal=episode.distractor_goal,
                    closest_index=episode.closest_index,
                )
            )
    return samples


__all__ = [
    "DistractorEpisode",
    "PrefixSample",
    "load_distractor_episodes",
    "generate_prefix_samples",
]
