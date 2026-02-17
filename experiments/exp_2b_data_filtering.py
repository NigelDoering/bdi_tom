"""Experiment 2b data filtering: cross-category distractor episodes.

Scans the **test split** of run_8 trajectories and identifies episodes where
an agent's path passes within a configurable distance of a low-preference
POI from a **different (bottom-ranked) category** than the agent's
high-preference goal category.

This tests whether models assign low probability to distractors from
entirely different semantic categories — the easier, category-level version
of the distractor test.

Selection criteria (matching the original generator in ``archive/exp_2.py``):

1. The trajectory's goal category c⁺ must be among the agent's **top-k**
   preferred categories (``category_preferences`` ranked descending).
2. The goal node g* must be among the **top-k** highest-preference POIs
   within c⁺.
3. The distractor category c⁻ is drawn from the agent's **bottom-k**
   preferred categories (c⁻ ≠ c⁺).
4. The distractor node g̃ is among the **bottom-k** lowest-preference POIs
   within c⁻.
5. The path must pass within ``distance_threshold_m`` metres of g̃.

Usage:
    python experiments/exp_2b_data_filtering.py [OPTIONS]

    # With defaults (run_8, 100m threshold, test split):
    python experiments/exp_2b_data_filtering.py

    # Custom threshold:
    python experiments/exp_2b_data_filtering.py --distance_threshold 150
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from graph_controller.world_graph import WorldGraph


# ── helpers ──────────────────────────────────────────────────────────────────


def build_simple_graph(graph: nx.Graph) -> nx.Graph:
    """Collapse a MultiGraph into a simple Graph keeping shortest edges."""
    if not graph.is_multigraph():
        return graph
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        length = data.get("length")
        if length is None:
            continue
        length = float(length)
        if simple.has_edge(u, v):
            if length < simple[u][v].get("length", float("inf")):
                simple[u][v]["length"] = length
        else:
            simple.add_edge(u, v, length=length)
    return simple


def path_total_meters(graph: nx.Graph, path_nodes: Sequence[str]) -> float:
    """Sum edge lengths (meters) along a path of node IDs."""
    total = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = graph.get_edge_data(u, v)
        if edge_data is None:
            continue
        if isinstance(edge_data, dict) and any(isinstance(val, dict) for val in edge_data.values()):
            first_edge = next(iter(edge_data.values()))
            total += float(first_edge.get("length", 0.0))
        else:
            total += float(edge_data.get("length", 0.0))
    return total


def load_agent_preferences(agents_path: Path) -> Dict[str, Dict]:
    """Load all_agents.json → {agent_id: agent_data}."""
    with agents_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_category_pois(world_graph: WorldGraph, category: str) -> List[str]:
    """Return all POI node IDs belonging to *category*."""
    return [
        node
        for node in world_graph.poi_nodes
        if world_graph.G.nodes[node].get("Category") == category
    ]


# ── core filtering logic ────────────────────────────────────────────────────


def find_distractor_episodes(
    test_trajectories: List[Dict],
    agent_preferences: Dict[str, Dict],
    world_graph: WorldGraph,
    simple_graph: nx.Graph,
    distance_threshold_m: float = 100.0,
    preference_top_k: int = 3,
) -> List[Dict]:
    """Scan real test trajectories for cross-category distractor proximity.

    1. Goal category c⁺ ∈ top-k(category_preferences).
    2. Goal node g* ∈ top-k(POI preferences within c⁺).
    3. Distractor category c⁻ ∈ bottom-k(category_preferences), c⁻ ≠ c⁺.
    4. Distractor node g̃ ∈ bottom-k(POI preferences within c⁻).
    5. Path passes within ``distance_threshold_m`` metres of g̃.
    """
    episodes: List[Dict] = []
    category_poi_cache: Dict[str, List[str]] = {}

    skipped = defaultdict(int)

    for traj in tqdm(test_trajectories, desc="🔍 Scanning test trajectories"):
        # ── basic validation ──
        path = traj.get("path")
        goal_node = traj.get("goal_node")
        agent_id_int = traj.get("agent_id")

        if path is None or goal_node is None or agent_id_int is None:
            skipped["missing_fields"] += 1
            continue
        if len(path) < 3:
            skipped["too_short"] += 1
            continue

        agent_key = f"agent_{agent_id_int:03d}"
        agent_data = agent_preferences.get(agent_key)
        if agent_data is None:
            skipped["no_agent_prefs"] += 1
            continue

        # ── determine goal category ──
        if goal_node not in world_graph.G:
            skipped["goal_not_in_graph"] += 1
            continue
        goal_category = world_graph.G.nodes[goal_node].get("Category")
        if goal_category is None or goal_category == "None":
            skipped["goal_no_category"] += 1
            continue

        # ── criterion 1: goal category must be in agent's top-k categories ──
        cat_prefs = agent_data.get("category_preferences", {})
        if not cat_prefs:
            skipped["no_cat_prefs"] += 1
            continue

        sorted_cats = sorted(cat_prefs.items(), key=lambda kv: kv[1], reverse=True)
        k = min(preference_top_k, len(sorted_cats))
        top_categories = [c for c, _ in sorted_cats[:k]]
        bottom_categories = [c for c, _ in sorted_cats[-k:]]

        if goal_category not in top_categories:
            skipped["goal_not_top_category"] += 1
            continue

        # ── criterion 2: goal node must be in top-k POIs within goal category ──
        goal_cat_pref_key = f"{goal_category}_preferences"
        goal_cat_poi_prefs = agent_data.get(goal_cat_pref_key, {})
        if not goal_cat_poi_prefs:
            skipped["no_goal_cat_poi_prefs"] += 1
            continue

        sorted_goal_pois = sorted(
            goal_cat_poi_prefs.items(), key=lambda kv: kv[1], reverse=True
        )
        top_k_goal_pois = {
            node_id
            for node_id, _ in sorted_goal_pois[: min(preference_top_k, len(sorted_goal_pois))]
        }

        if goal_node not in top_k_goal_pois:
            skipped["goal_not_top_poi"] += 1
            continue

        goal_pref_value = goal_cat_poi_prefs.get(goal_node, 0.0)

        # ── criterion 3 & 4: find distractor from bottom-k category, bottom-k POI ──
        distractor_candidates_by_cat: List[Tuple[str, str, float]] = []

        for dist_cat in bottom_categories:
            if dist_cat == goal_category:
                continue

            dist_pref_key = f"{dist_cat}_preferences"
            dist_poi_prefs = agent_data.get(dist_pref_key, {})
            if not dist_poi_prefs:
                continue

            if dist_cat not in category_poi_cache:
                category_poi_cache[dist_cat] = get_category_pois(world_graph, dist_cat)

            sorted_dist_pois = sorted(dist_poi_prefs.items(), key=lambda kv: kv[1])
            bottom_k_pois = sorted_dist_pois[: min(preference_top_k, len(sorted_dist_pois))]

            for node_id, pref_val in bottom_k_pois:
                if node_id == goal_node:
                    continue
                if node_id in simple_graph:
                    distractor_candidates_by_cat.append((dist_cat, node_id, pref_val))

        if not distractor_candidates_by_cat:
            skipped["no_distractor_candidates"] += 1
            continue

        # ── extract path node IDs ──
        path_nodes = []
        for step in path:
            if isinstance(step, (list, tuple)) and len(step) >= 1:
                path_nodes.append(str(step[0]))
            else:
                path_nodes.append(str(step))

        # ── criterion 5: check proximity to each distractor candidate ──
        best_episode = None
        best_distance = float("inf")

        for dist_cat, distractor_node, distractor_pref_val in distractor_candidates_by_cat:
            distances = []
            for node in path_nodes:
                if node not in simple_graph:
                    distances.append(float("inf"))
                    continue
                try:
                    d = nx.shortest_path_length(
                        simple_graph, node, distractor_node, weight="length"
                    )
                    distances.append(float(d))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    distances.append(float("inf"))

            min_distance = min(distances)
            if min_distance > distance_threshold_m:
                continue

            if min_distance < best_distance:
                best_distance = min_distance
                closest_index = int(np.argmin(distances))
                path_meters = path_total_meters(simple_graph, path_nodes)

                best_episode = {
                    "agent_id": agent_key,
                    "preferred_category": goal_category,
                    "preferred_goal": goal_node,
                    "distractor_category": dist_cat,
                    "distractor_goal": distractor_node,
                    "start_node": path_nodes[0],
                    "observation_hour": int(traj.get("hour", 0)),
                    "path": [list(step) for step in path],
                    "path_length": len(path),
                    "path_meters": round(path_meters, 3),
                    "min_distance_to_distractor": round(min_distance, 3),
                    "closest_index": closest_index,
                    "goal_preference_value": round(goal_pref_value, 6),
                    "distractor_preference_value": round(distractor_pref_val, 6),
                }

        if best_episode is not None:
            episodes.append(best_episode)

    # Print skip reasons
    if skipped:
        print("\n📊 Skip reasons:")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"   {reason}: {count:,}")

    return episodes


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 2b: Filter test trajectories for cross-category distractor proximity."
    )
    parser.add_argument(
        "--run_dir", type=str, default="data/simulation_data/run_8",
        help="Path to simulation run directory",
    )
    parser.add_argument(
        "--graph_path", type=str, default="data/processed/ucsd_walk_full.graphml",
        help="Path to graph file",
    )
    parser.add_argument(
        "--split_file", type=str,
        default="data/simulation_data/run_8/split_data/split_indices_seed42.json",
        help="Path to split indices JSON",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/data/exp_2b_test_set.json",
        help="Output path for filtered episodes",
    )
    parser.add_argument(
        "--distance_threshold", type=float, default=100.0,
        help="Maximum graph distance (metres) from path to distractor POI",
    )
    parser.add_argument(
        "--preference_top_k", type=int, default=3,
        help="Top-k / bottom-k slice for category and POI preferences",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    graph_path = Path(args.graph_path)
    split_file = Path(args.split_file)
    output_path = Path(args.output)

    print("=" * 80)
    print("EXPERIMENT 2b — CROSS-CATEGORY DISTRACTOR FILTERING (REAL TRAJECTORIES)")
    print("=" * 80)

    # ── load graph ──
    print(f"\n📂 Loading graph from {graph_path} ...")
    graph = nx.read_graphml(str(graph_path))
    world_graph = WorldGraph(graph)
    simple_graph = build_simple_graph(world_graph.G)
    print(f"   {len(world_graph.G.nodes()):,} nodes, {len(world_graph.G.edges()):,} edges")
    print(f"   {len(world_graph.poi_nodes)} POI nodes")

    # ── load agent preferences ──
    agents_path = run_dir / "agents" / "all_agents.json"
    print(f"\n📂 Loading agent preferences from {agents_path} ...")
    agent_preferences = load_agent_preferences(agents_path)
    print(f"   {len(agent_preferences)} agents")

    # ── load trajectories and extract test split ──
    traj_path = run_dir / "trajectories" / "all_trajectories.json"
    print(f"\n📂 Loading trajectories from {traj_path} ...")
    with traj_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    all_trajectories: List[Dict] = []
    agent_keys = sorted(raw_data.keys())
    for agent_idx, agent_key in enumerate(agent_keys):
        for traj in raw_data[agent_key]:
            traj["agent_id"] = agent_idx
            all_trajectories.append(traj)
    print(f"   {len(all_trajectories):,} total trajectories")

    print(f"\n📂 Loading split indices from {split_file} ...")
    with split_file.open("r", encoding="utf-8") as fh:
        split_indices = json.load(fh)

    test_indices = split_indices["test_indices"]
    test_trajectories = [all_trajectories[i] for i in test_indices]
    print(f"   {len(test_trajectories):,} test trajectories")

    # ── filter ──
    print(f"\n🔍 Filtering for cross-category distractor proximity "
          f"(threshold={args.distance_threshold}m, top_k={args.preference_top_k}) ...")
    episodes = find_distractor_episodes(
        test_trajectories=test_trajectories,
        agent_preferences=agent_preferences,
        world_graph=world_graph,
        simple_graph=simple_graph,
        distance_threshold_m=args.distance_threshold,
        preference_top_k=args.preference_top_k,
    )

    # ── summary stats ──
    print(f"\n{'=' * 80}")
    print(f"✅ Found {len(episodes):,} qualifying episodes from "
          f"{len(test_trajectories):,} test trajectories")
    print(f"   Hit rate: {len(episodes) / max(len(test_trajectories), 1) * 100:.1f}%")

    if episodes:
        distances = [ep["min_distance_to_distractor"] for ep in episodes]
        path_lengths = [ep["path_length"] for ep in episodes]
        closest_fracs = [ep["closest_index"] / max(ep["path_length"], 1) for ep in episodes]

        print(f"\n   Distance to distractor (m):")
        print(f"     min={min(distances):.1f}  median={np.median(distances):.1f}  "
              f"max={max(distances):.1f}")
        print(f"   Path length (steps):")
        print(f"     min={min(path_lengths)}  median={np.median(path_lengths):.0f}  "
              f"max={max(path_lengths)}")
        print(f"   Closest approach (fraction of path):")
        print(f"     min={min(closest_fracs):.2f}  median={np.median(closest_fracs):.2f}  "
              f"max={max(closest_fracs):.2f}")

        # Goal category breakdown
        goal_cat_counts = defaultdict(int)
        for ep in episodes:
            goal_cat_counts[ep["preferred_category"]] += 1
        print(f"\n   Goal category breakdown (top-{args.preference_top_k} preferred):")
        for cat, count in sorted(goal_cat_counts.items(), key=lambda kv: -kv[1]):
            print(f"     {cat}: {count}")

        # Distractor category breakdown
        dist_cat_counts = defaultdict(int)
        for ep in episodes:
            dist_cat_counts[ep["distractor_category"]] += 1
        print(f"\n   Distractor category breakdown (bottom-{args.preference_top_k} preferred):")
        for cat, count in sorted(dist_cat_counts.items(), key=lambda kv: -kv[1]):
            print(f"     {cat}: {count}")

        # Verify cross-category invariant
        same_cat = sum(
            1 for ep in episodes
            if ep["preferred_category"] == ep["distractor_category"]
        )
        if same_cat > 0:
            print(f"\n   ⚠️  {same_cat} episodes share goal/distractor category (unexpected)")
        else:
            print(f"\n   ✓ All episodes have different goal and distractor categories")

        # Agent coverage
        agent_set = {ep["agent_id"] for ep in episodes}
        print(f"\n   Agents represented: {len(agent_set)} / {len(agent_preferences)}")

    # ── write output ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "run_dir": str(run_dir),
            "graph_path": str(graph_path),
            "split_file": str(split_file),
            "distance_threshold_m": args.distance_threshold,
            "preference_top_k": args.preference_top_k,
            "seed": args.seed,
            "source": "filtered_test_trajectories",
            "variant": "2b_cross_category",
        },
        "summary": {
            "total_test_trajectories": len(test_trajectories),
            "total_episodes": len(episodes),
            "hit_rate_pct": round(
                len(episodes) / max(len(test_trajectories), 1) * 100, 2
            ),
        },
        "episodes": episodes,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n💾 Saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
