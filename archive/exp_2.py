"""Experiment 2 distractor episode generator.

This module constructs the specialized evaluation set described in the
preference-proximity dissociation test. It reads agent preference
metadata produced by prior simulations, selects preference-aligned true
goals alongside low-preference distractors, and samples start positions
whose shortest routes pass near the distractor. The resulting episodes
mirror the structured format used across the project so downstream
analysis scripts can reuse existing data loaders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agent_controller.agent import Agent
from agent_controller.planning_utils import sample_start_node
from graph_controller.world_graph import WorldGraph

CATEGORIES: Tuple[str, ...] = ("home", "study", "food", "leisure", "errands", "health")


@dataclass(frozen=True)
class EpisodeConfig:
    """Runtime configuration for distractor episode generation."""

    run_dir: Path
    graph_path: Path
    output_path: Path
    max_per_agent: int = 100
    distance_threshold_m: float = 100.0
    temperature: float = 30.0
    max_sampling_attempts: int = 2000
    seed: int = 42

    @property
    def agents_file(self) -> Path:
        return self.run_dir / "agents" / "all_agents.json"


@dataclass
class DistractorEpisode:
    agent_id: str
    preferred_category: str
    preferred_goal: str
    distractor_category: str
    distractor_goal: str
    start_node: str
    observation_hour: int
    path: List[Tuple[str, str]]
    path_length: int
    path_meters: float
    min_distance_to_distractor: float
    closest_index: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "preferred_category": self.preferred_category,
            "preferred_goal": self.preferred_goal,
            "distractor_category": self.distractor_category,
            "distractor_goal": self.distractor_goal,
            "start_node": self.start_node,
            "observation_hour": self.observation_hour,
            "path": self.path,
            "path_length": self.path_length,
            "path_meters": round(self.path_meters, 3),
            "min_distance_to_distractor": round(self.min_distance_to_distractor, 3),
            "closest_index": self.closest_index,
        }


def load_agents(config: EpisodeConfig, world_graph: WorldGraph) -> Dict[str, Agent]:
    with config.agents_file.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    agents: Dict[str, Agent] = {}
    for agent_id, data in metadata.items():
        agent = Agent(agent_id=agent_id, world_graph=world_graph, verbose=False)
        agent.category_preferences = data["category_preferences"].copy()
        for category in CATEGORIES:
            pref_key = f"{category}_preferences"
            if pref_key in data:
                setattr(agent, pref_key, data[pref_key].copy())
        agents[agent_id] = agent
    return agents


def _ranked_keys(mapping: Dict[str, float], reverse: bool) -> List[str]:
    items = sorted(mapping.items(), key=lambda kv: kv[1], reverse=reverse)
    return [key for key, _ in items]


def _sample_from_slice(rng: np.random.Generator, keys: Sequence[str], weights: Sequence[float]) -> Optional[str]:
    filtered = [(k, w) for k, w in zip(keys, weights) if math.isfinite(w)]
    if not filtered:
        return None
    labels, raw_weights = zip(*filtered)
    total = sum(raw_weights)
    if total <= 0:
        probs = np.repeat(1.0 / len(labels), len(labels))
    else:
        probs = np.array(raw_weights, dtype=float) / total
    choice = rng.choice(len(labels), p=probs)
    return labels[int(choice)]


def choose_true_and_distractor(
    rng: np.random.Generator,
    agent: Agent,
    top_k: int = 3,
) -> Optional[Tuple[str, str, str, str]]:
    category_order = _ranked_keys(agent.category_preferences, reverse=True)
    if not category_order:
        return None
    top_slice = category_order[: min(top_k, len(category_order))]
    bottom_slice = category_order[-min(top_k, len(category_order)) :]

    true_category = rng.choice(top_slice)
    distractor_category = rng.choice(bottom_slice)

    true_pref = getattr(agent, f"{true_category}_preferences", {})
    distractor_pref = getattr(agent, f"{distractor_category}_preferences", {})

    if not true_pref or not distractor_pref:
        return None

    true_nodes = _ranked_keys(true_pref, reverse=True)[: min(top_k, len(true_pref))]
    distractor_nodes = _ranked_keys(distractor_pref, reverse=False)[: min(top_k, len(distractor_pref))]

    true_weights = [true_pref[node] for node in true_nodes]
    distractor_weights = [distractor_pref[node] for node in distractor_nodes]

    true_goal = _sample_from_slice(rng, true_nodes, true_weights)
    distractor_goal = _sample_from_slice(rng, distractor_nodes, distractor_weights)

    if true_goal is None or distractor_goal is None or true_goal == distractor_goal:
        return None
    return true_category, true_goal, distractor_category, distractor_goal


def build_simple_graph(graph: nx.Graph) -> nx.Graph:
    if not graph.is_multigraph():
        return graph
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        length = data.get("length")
        if length is None:
            continue
        if simple.has_edge(u, v):
            if length < simple[u][v].get("length", float("inf")):
                simple[u][v]["length"] = length
        else:
            simple.add_edge(u, v, length=length)
    return simple


def path_total_length(graph: nx.Graph, path: Sequence[str]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        if edge_data is None:
            raise nx.NetworkXNoPath(f"No edge between {u} and {v}")
        if isinstance(edge_data, dict) and all(isinstance(k, (int, str)) for k in edge_data.keys()) and any(
            isinstance(val, dict) for val in edge_data.values()
        ):
            first_edge = next(iter(edge_data.values()))
            total += float(first_edge.get("length", 0.0))
        else:
            total += float(edge_data.get("length", 0.0))
    return total


def sample_path_with_distractor(
    rng: np.random.Generator,
    graph: nx.Graph,
    simple_graph: nx.Graph,
    start_node: str,
    goal_node: str,
    distractor_node: str,
    distance_threshold: float,
    temperature: float,
    top_k: int = 5,
) -> Optional[Tuple[Sequence[str], float, float, int]]:
    try:
        paths_iter = nx.shortest_simple_paths(simple_graph, start_node, goal_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    candidates: List[Tuple[Sequence[str], float, List[float]]] = []
    for path in islice(paths_iter, top_k * 3):
        try:
            distances = [nx.shortest_path_length(simple_graph, node, distractor_node, weight="length") for node in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        min_distance = min(distances)
        if min_distance > distance_threshold:
            continue
        length_m = path_total_length(graph, path)
        candidates.append((path, length_m, distances))
        if len(candidates) >= top_k:
            break

    if not candidates:
        return None

    lengths = np.array([length for _, length, _ in candidates], dtype=float)
    logits = -lengths / max(temperature, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = int(rng.choice(len(candidates), p=probs))
    path, length_m, distances = candidates[idx]
    closest_idx = int(np.argmin(distances))
    return path, length_m, float(min(distances)), closest_idx


def generate_distractor_episodes(config: EpisodeConfig) -> Tuple[List[DistractorEpisode], Dict[str, int]]:
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    world_graph = WorldGraph(nx.read_graphml(config.graph_path))
    simple_graph = build_simple_graph(world_graph.G)
    agents = load_agents(config, world_graph)

    episodes: List[DistractorEpisode] = []
    per_agent_counts: Dict[str, int] = defaultdict(int)

    for agent_id, agent in tqdm(agents.items(), desc="Agents", ncols=100):
        success = 0
        attempts = 0
        while success < config.max_per_agent and attempts < config.max_sampling_attempts:
            attempts += 1
            choice = choose_true_and_distractor(rng, agent)
            if choice is None:
                continue
            true_category, true_goal, distractor_category, distractor_goal = choice

            hour = int(rng.integers(0, 24))
            try:
                start_node = sample_start_node(agent, true_goal, current_hour=hour)
            except RuntimeError:
                continue

            sampled = sample_path_with_distractor(
                rng,
                world_graph.G,
                simple_graph,
                start_node,
                true_goal,
                distractor_goal,
                config.distance_threshold_m,
                config.temperature,
            )
            if sampled is None:
                continue

            path_nodes, path_meters, min_distance, closest_idx = sampled
            annotated_path = [(node, true_goal) for node in path_nodes]

            episodes.append(
                DistractorEpisode(
                    agent_id=agent_id,
                    preferred_category=true_category,
                    preferred_goal=true_goal,
                    distractor_category=distractor_category,
                    distractor_goal=distractor_goal,
                    start_node=start_node,
                    observation_hour=hour,
                    path=annotated_path,
                    path_length=len(annotated_path),
                    path_meters=path_meters,
                    min_distance_to_distractor=min_distance,
                    closest_index=closest_idx,
                )
            )
            success += 1
            per_agent_counts[agent_id] = success
        if success < config.max_per_agent:
            per_agent_counts[agent_id] = success
    return episodes, per_agent_counts


def write_output(config: EpisodeConfig, episodes: List[DistractorEpisode], counts: Dict[str, int]) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "run_dir": str(config.run_dir),
            "graph_path": str(config.graph_path),
            "max_per_agent": config.max_per_agent,
            "distance_threshold_m": config.distance_threshold_m,
            "temperature": config.temperature,
            "seed": config.seed,
        },
        "summary": {
            "total_agents": len(counts),
            "total_episodes": len(episodes),
            "episodes_per_agent": counts,
        },
        "episodes": [episode.to_dict() for episode in episodes],
    }
    with config.output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def parse_args() -> EpisodeConfig:
    parser = argparse.ArgumentParser(description="Generate distractor episodes for Experiment 2.")
    parser.add_argument("--run-id", type=int, default=1, help="Simulation run identifier (data/simulation_data/run_<id>).")
    parser.add_argument("--graph", type=Path, default=Path("data/processed/ucsd_walk_full.graphml"), help="Path to the processed campus graph.")
    parser.add_argument("--max-per-agent", type=int, default=100, help="Maximum distractor episodes to keep per agent.")
    parser.add_argument("--distance-threshold", type=float, default=100.0, help="Maximum meters between path and distractor.")
    parser.add_argument("--temperature", type=float, default=30.0, help="Softmax temperature for path sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--attempts", type=int, default=2000, help="Sampling attempts per agent before giving up.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults to data/simulation_data/run_<id>/exp_2_distractors.json.",
    )

    args = parser.parse_args()
    run_dir = Path("data/simulation_data") / f"run_{args.run_id}"
    output_path = args.output or (run_dir / "exp_2_distractors.json")
    return EpisodeConfig(
        run_dir=run_dir,
        graph_path=args.graph,
        output_path=output_path,
        max_per_agent=args.max_per_agent,
        distance_threshold_m=args.distance_threshold,
        temperature=args.temperature,
        max_sampling_attempts=args.attempts,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    episodes, counts = generate_distractor_episodes(config)
    write_output(config, episodes, counts)
    print(f"Generated {len(episodes)} distractor episodes across {len(counts)} agents.")
    print(f"Saved to {config.output_path}")


if __name__ == "__main__":
    main()
