"""Experiment 3 belief-update evaluation harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.exp_1 import load_model_from_checkpoint  # type: ignore
from experiments.exp_3_data import (
    POST_PHASE,
    PRE_PHASE,
    PhaseSample,
    generate_phase_samples,
    load_pivot_dataset,
)
from graph_controller.world_graph import WorldGraph
from models.encoders.map_encoder import GraphDataPreparator
from models.encoders.trajectory_encoder import TrajectoryDataPreparator
from models.utils.utils import get_device


def parse_fraction_list(spec: str) -> List[float]:
    values = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid fraction '{token}'") from exc
        if value <= 0 or value > 1:
            raise argparse.ArgumentTypeError("Fractions must be inside (0, 1]")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one fraction required")
    return values


def bootstrap_ci(values: Sequence[float], seed: int, n_samples: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resamples = rng.choice(arr, size=(n_samples, len(arr)), replace=True)
    means = resamples.mean(axis=1)
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def prepare_trajectory_preparator(graph: nx.Graph) -> TrajectoryDataPreparator:
    all_nodes = list(graph.nodes())
    node_to_idx = {node: idx + 1 for idx, node in enumerate(all_nodes)}
    node_to_idx["<PAD>"] = 0
    return TrajectoryDataPreparator(node_to_idx)


def create_graph_data(world_graph: WorldGraph, device: torch.device) -> Dict[str, torch.Tensor]:
    graph_prep = GraphDataPreparator(world_graph)
    graph_data = graph_prep.prepare_graph_data()
    return {
        "x": graph_data["x"].to(device),
        "edge_index": graph_data["edge_index"].to(device),
    }


def batch_iterator(items: Sequence[PhaseSample], batch_size: int) -> Iterable[Sequence[PhaseSample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_model(
    model_path: Path,
    device: torch.device,
    world_graph: WorldGraph,
    graph_data: Dict[str, torch.Tensor],
    preparator: TrajectoryDataPreparator,
    samples: Sequence[PhaseSample],
    pre_fractions: Sequence[float],
    post_fractions: Sequence[float],
    poi_nodes: Sequence[str],
    seed: int,
    batch_size: int,
    max_seq_len: int,
) -> Dict[str, object]:
    poi_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
    num_nodes = len(world_graph.G.nodes())
    graph_node_feat_dim = graph_data["x"].shape[1]

    model = load_model_from_checkpoint(
        str(model_path),
        num_poi_nodes=len(poi_nodes),
        num_nodes=num_nodes,
        graph_node_feat_dim=graph_node_feat_dim,
        device=device,
    )
    model.eval()

    pre_probs = {fraction: [] for fraction in pre_fractions}
    post_probs = {fraction: [] for fraction in post_fractions}
    post_entropy = {fraction: [] for fraction in post_fractions}
    belief_alignment_values: List[float] = []
    post_per_episode: Dict[int, Dict[float, float]] = defaultdict(dict)

    with torch.no_grad():
        for batch in tqdm(
            batch_iterator(samples, batch_size),
            total=math.ceil(len(samples) / batch_size),
            desc=model_path.stem,
            ncols=100,
        ):
            traj_dicts = []
            valid_samples: List[PhaseSample] = []
            for sample in batch:
                if sample.target_goal not in poi_to_idx:
                    continue
                traj_dicts.append(
                    {
                        "path": sample.path,
                        "hour": sample.hour,
                        "goal_node": sample.target_goal,
                    }
                )
                valid_samples.append(sample)

            if not traj_dicts:
                continue

            traj_batch = preparator.prepare_batch(traj_dicts, max_seq_len=max_seq_len)
            traj_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in traj_batch.items()
            }

            logits = model(traj_batch, graph_data, return_logits=True)
            probs = F.softmax(logits, dim=-1).cpu()

            for row, sample in zip(probs, valid_samples):
                target_idx = poi_to_idx[sample.target_goal]
                goal_prob = float(row[target_idx].item())
                if sample.phase == PRE_PHASE:
                    pre_probs[sample.fraction].append(goal_prob)
                    belief_alignment_values.append(goal_prob)
                else:
                    post_probs[sample.fraction].append(goal_prob)
                    post_per_episode[sample.episode_idx][sample.fraction] = goal_prob
                    entropy = -float(torch.sum(row * torch.log(row + 1e-12)).item())
                    post_entropy[sample.fraction].append(entropy)

    adaptation_deltas: List[float] = []
    if post_fractions:
        early_fraction = min(post_fractions)
        late_fraction = max(post_fractions)
        for episode_idx, fraction_map in post_per_episode.items():
            if early_fraction in fraction_map and late_fraction in fraction_map:
                adaptation_deltas.append(fraction_map[late_fraction] - fraction_map[early_fraction])

    summary = {
        "model": model_path.stem,
        "pre_phase": {
            fraction: {
                "mean": float(np.mean(pre_probs[fraction])) if pre_probs[fraction] else math.nan,
                "ci_low": None,
                "ci_high": None,
                "values": pre_probs[fraction],
            }
            for fraction in pre_fractions
        },
        "post_phase": {
            fraction: {
                "mean": float(np.mean(post_probs[fraction])) if post_probs[fraction] else math.nan,
                "ci_low": None,
                "ci_high": None,
                "values": post_probs[fraction],
            }
            for fraction in post_fractions
        },
        "post_entropy": {
            fraction: {
                "mean": float(np.mean(post_entropy[fraction])) if post_entropy[fraction] else math.nan,
                "ci_low": None,
                "ci_high": None,
                "values": post_entropy[fraction],
            }
            for fraction in post_fractions
        },
        "belief_alignment": {
            "mean": float(np.mean(belief_alignment_values)) if belief_alignment_values else math.nan,
            "ci_low": None,
            "ci_high": None,
            "values": belief_alignment_values,
        },
        "adaptation_speed": {
            "mean": float(np.mean(adaptation_deltas)) if adaptation_deltas else math.nan,
            "ci_low": None,
            "ci_high": None,
            "values": adaptation_deltas,
        },
    }

    for fraction in pre_fractions:
        summary["pre_phase"][fraction]["ci_low"], summary["pre_phase"][fraction]["ci_high"] = bootstrap_ci(
            pre_probs[fraction], seed=seed
        )
    for fraction in post_fractions:
        summary["post_phase"][fraction]["ci_low"], summary["post_phase"][fraction]["ci_high"] = bootstrap_ci(
            post_probs[fraction], seed=seed
        )
        summary["post_entropy"][fraction]["ci_low"], summary["post_entropy"][fraction]["ci_high"] = bootstrap_ci(
            post_entropy[fraction], seed=seed
        )

    belief_low, belief_high = bootstrap_ci(belief_alignment_values, seed=seed)
    summary["belief_alignment"]["ci_low"] = belief_low
    summary["belief_alignment"]["ci_high"] = belief_high

    adapt_low, adapt_high = bootstrap_ci(adaptation_deltas, seed=seed)
    summary["adaptation_speed"]["ci_low"] = adapt_low
    summary["adaptation_speed"]["ci_high"] = adapt_high

    return summary


def write_csv_summaries(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    pre_fractions: Sequence[float],
    post_fractions: Sequence[float],
) -> None:
    phase_path = output_dir / "exp_3_phase_metrics.csv"
    entropy_path = output_dir / "exp_3_post_entropy.csv"
    scalar_path = output_dir / "exp_3_scalar_metrics.csv"

    with phase_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "phase", "fraction", "metric", "mean", "ci_low", "ci_high"])
        for summary in summaries:
            model_name = summary["model"]
            for fraction in pre_fractions:
                metric = summary["pre_phase"][fraction]
                writer.writerow([
                    model_name,
                    PRE_PHASE,
                    fraction,
                    "goal_prob",
                    metric["mean"],
                    metric["ci_low"],
                    metric["ci_high"],
                ])
            for fraction in post_fractions:
                metric = summary["post_phase"][fraction]
                writer.writerow([
                    model_name,
                    POST_PHASE,
                    fraction,
                    "goal_prob",
                    metric["mean"],
                    metric["ci_low"],
                    metric["ci_high"],
                ])

    with entropy_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "fraction", "mean", "ci_low", "ci_high"])
        for summary in summaries:
            model_name = summary["model"]
            for fraction in post_fractions:
                metric = summary["post_entropy"][fraction]
                writer.writerow([
                    model_name,
                    fraction,
                    metric["mean"],
                    metric["ci_low"],
                    metric["ci_high"],
                ])

    with scalar_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "metric", "mean", "ci_low", "ci_high"])
        for summary in summaries:
            belief = summary["belief_alignment"]
            writer.writerow([
                summary["model"],
                "belief_alignment",
                belief["mean"],
                belief["ci_low"],
                belief["ci_high"],
            ])
            adaptation = summary["adaptation_speed"]
            writer.writerow([
                summary["model"],
                "adaptation_speed",
                adaptation["mean"],
                adaptation["ci_low"],
                adaptation["ci_high"],
            ])


def write_json_summary(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> None:
    payload = {
        "config": config,
        "results": list(summaries),
    }
    with (output_dir / "exp_3_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 3 belief-update episodes.")
    parser.add_argument("--run-id", type=int, default=1, help="Simulation run identifier")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/processed/ucsd_walk_full.graphml"),
        help="Path to processed campus graph",
    )
    parser.add_argument("--keepers", type=Path, default=Path("checkpoints/keepers"), help="Directory of model checkpoints")
    parser.add_argument(
        "--pivots",
        type=Path,
        default=None,
        help="Path to Experiment 3 pivot episodes JSON",
    )
    parser.add_argument(
        "--pre-fractions",
        type=parse_fraction_list,
        default=parse_fraction_list("0.25,0.5,0.75"),
        help="Comma-separated observation fractions for pre-pivot phase",
    )
    parser.add_argument(
        "--post-fractions",
        type=parse_fraction_list,
        default=parse_fraction_list("0.25,0.5,0.75"),
        help="Comma-separated observation fractions for post-pivot phase",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Directory to store metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path("data/simulation_data") / f"run_{args.run_id}"
    pivot_path = args.pivots or (run_dir / "exp_3_pivots.json")
    output_dir = args.output or (run_dir / "visualizations" / "exp_3")

    if not pivot_path.exists():
        raise FileNotFoundError(f"Pivot episodes not found: {pivot_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    graph = nx.read_graphml(args.graph)
    world_graph = WorldGraph(graph)
    graph_data = create_graph_data(world_graph, device)
    preparator = prepare_trajectory_preparator(world_graph.G)

    episodes = load_pivot_dataset(pivot_path)
    if not episodes:
        print("No Experiment 3 episodes available. Abort.")
        return

    samples, max_prefix_len = generate_phase_samples(episodes, args.pre_fractions, args.post_fractions)
    seq_len = max(args.max_seq_len, max_prefix_len)

    model_paths = sorted(path for path in args.keepers.glob("*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No checkpoints found in {args.keepers}")

    summaries: List[Dict[str, object]] = []
    for model_path in model_paths:
        summary = evaluate_model(
            model_path=model_path,
            device=device,
            world_graph=world_graph,
            graph_data=graph_data,
            preparator=preparator,
            samples=samples,
            pre_fractions=args.pre_fractions,
            post_fractions=args.post_fractions,
            poi_nodes=world_graph.poi_nodes,
            seed=args.seed,
            batch_size=args.batch_size,
            max_seq_len=seq_len,
        )
        summaries.append(summary)

    config_payload = {
        "run_id": args.run_id,
        "graph_path": str(args.graph),
        "pivot_path": str(pivot_path),
        "pre_fractions": args.pre_fractions,
        "post_fractions": args.post_fractions,
        "batch_size": args.batch_size,
        "max_seq_len": seq_len,
        "seed": args.seed,
        "models": [path.name for path in model_paths],
        "num_episodes": len(episodes),
        "num_samples": len(samples),
    }

    write_csv_summaries(output_dir, summaries, args.pre_fractions, args.post_fractions)
    write_json_summary(output_dir, summaries, config_payload)

    print(f"Processed {len(samples)} samples across {len(model_paths)} model(s).")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
