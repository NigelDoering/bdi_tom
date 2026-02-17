"""Experiment 3 pivot episode extractor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.exp_3_data import extract_pivot_episodes, load_trajectories, write_pivot_dataset
from graph_controller.world_graph import WorldGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify trajectories with unexpected closures for Experiment 3.")
    parser.add_argument("--run-id", type=int, default=8, help="Simulation run identifier (data/simulation_data/run_<id>).")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/processed/ucsd_walk_full.graphml"),
        help="Path to processed campus graph.",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="Optional explicit path to all_trajectories.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON for pivot episodes. Defaults to data/simulation_data/run_<id>/exp_3_pivots.json.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to split_indices JSON. Defaults to data/simulation_data/run_<id>/split_data/split_indices_seed42.json.",
    )
    parser.add_argument(
        "--no-split-filter",
        action="store_true",
        help="Disable test-split filtering and use all trajectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path("data/simulation_data") / f"run_{args.run_id}"
    trajectories_path = args.trajectories or (run_dir / "trajectories" / "all_trajectories.json")
    output_path = args.output or (run_dir / "exp_3_pivots.json")

    if not trajectories_path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {trajectories_path}")

    graph = nx.read_graphml(args.graph)
    world_graph = WorldGraph(graph)

    trajectories = load_trajectories(trajectories_path)

    # ── Filter to test split only (prevents data leakage) ──
    if not args.no_split_filter:
        split_file = args.split_file or (run_dir / "split_data" / "split_indices_seed42.json")
        if split_file.exists():
            with split_file.open("r", encoding="utf-8") as fh:
                split_data = json.load(fh)
            test_indices = set(split_data["test_indices"])

            # Flatten trajectories to a global index, keep only test indices
            agent_keys = sorted(trajectories.keys())
            global_idx = 0
            filtered_trajectories: dict = {}
            for agent_key in agent_keys:
                agent_runs = trajectories[agent_key]
                kept = []
                for ep in agent_runs:
                    if global_idx in test_indices:
                        kept.append(ep)
                    global_idx += 1
                if kept:
                    filtered_trajectories[agent_key] = kept

            total_before = sum(len(v) for v in trajectories.values())
            total_after = sum(len(v) for v in filtered_trajectories.values())
            print(f"Test-split filter: {total_before:,} → {total_after:,} trajectories")
            trajectories = filtered_trajectories
        else:
            print(f"⚠️  Split file not found ({split_file}), using all trajectories.")

    episodes = extract_pivot_episodes(trajectories, world_graph)

    if not episodes:
        print("No qualifying pivot episodes found.")
        return

    write_pivot_dataset(output_path, episodes, run_dir=run_dir, graph_path=args.graph)

    per_agent = Counter(ep.agent_id for ep in episodes)
    print(f"Identified {len(episodes)} pivot episodes across {len(per_agent)} agents.")
    print(f"Saved dataset to {output_path}")
    print("Episodes per agent (top 10):")
    for agent_id, count in per_agent.most_common(10):
        print(f"  {agent_id}: {count}")


if __name__ == "__main__":
    main()
