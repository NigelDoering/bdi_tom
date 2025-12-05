"""Experiment 2 preference-proximity evaluation harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import networkx as nx
import numpy as np
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.exp_1 import load_model_from_checkpoint  # type: ignore
from experiments.exp_2_data import (
    DistractorEpisode,
    PrefixSample,
    generate_prefix_samples,
    load_distractor_episodes,
)
from graph_controller.world_graph import WorldGraph
from models.encoders.map_encoder import GraphDataPreparator
from models.encoders.trajectory_encoder import TrajectoryDataPreparator
from models.training.utils import get_device


def parse_fractions(spec: str) -> List[float]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid fraction '{part}'") from exc
        if value <= 0 or value > 1:
            raise argparse.ArgumentTypeError("Fractions must be in (0, 1]")
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
    preparator = TrajectoryDataPreparator(node_to_idx)
    return preparator


def create_graph_data(world_graph: WorldGraph, device: torch.device) -> Dict[str, torch.Tensor]:
    graph_prep = GraphDataPreparator(world_graph)
    graph_data = graph_prep.prepare_graph_data()
    return {
        "x": graph_data["x"].to(device),
        "edge_index": graph_data["edge_index"].to(device),
    }


def batch_iterator(items: Sequence[PrefixSample], batch_size: int) -> Iterable[Sequence[PrefixSample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_samples_dict(episodes: Sequence[DistractorEpisode], fractions: Sequence[float]) -> Tuple[List[PrefixSample], Dict[float, List[int]]]:
    samples = generate_prefix_samples(episodes, fractions)
    indices_by_fraction: Dict[float, List[int]] = {fraction: [] for fraction in fractions}
    for idx, sample in enumerate(samples):
        indices_by_fraction[sample.fraction].append(idx)
    return samples, indices_by_fraction


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    model_path: Path,
    device: torch.device,
    world_graph: WorldGraph,
    graph_data: Dict[str, torch.Tensor],
    preparator: TrajectoryDataPreparator,
    samples: Sequence[PrefixSample],
    fractions: Sequence[float],
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

    distractor_probs = {fraction: [] for fraction in fractions}
    goal_probs = {fraction: [] for fraction in fractions}
    top1_scores = {fraction: [] for fraction in fractions}
    top5_scores = {fraction: [] for fraction in fractions}
    brier_scores = {fraction: [] for fraction in fractions}
    per_episode_probs: Dict[int, Dict[float, float]] = defaultdict(dict)

    with torch.no_grad():
        for batch in tqdm(batch_iterator(samples, batch_size), total=math.ceil(len(samples) / batch_size), desc=model_path.stem, ncols=100):
            traj_dicts = []
            valid_indices = []
            for sample in batch:
                if sample.preferred_goal not in poi_to_idx or sample.distractor_goal not in poi_to_idx:
                    continue
                traj_dicts.append(
                    {
                        "path": sample.path,
                        "hour": sample.observation_hour,
                        "goal_node": sample.preferred_goal,
                    }
                )
                valid_indices.append(sample)

            if not traj_dicts:
                continue

            traj_batch = preparator.prepare_batch(traj_dicts, max_seq_len=max_seq_len)
            traj_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in traj_batch.items()}

            logits = model(traj_batch, graph_data, return_logits=True)
            probs = F.softmax(logits, dim=-1).cpu()

            for row, sample in zip(probs, valid_indices):
                frac = sample.fraction
                pref_idx = poi_to_idx[sample.preferred_goal]
                dist_idx = poi_to_idx[sample.distractor_goal]
                goal_prob = float(row[pref_idx].item())
                dist_prob = float(row[dist_idx].item())
                goal_probs[frac].append(goal_prob)
                distractor_probs[frac].append(dist_prob)
                per_episode_probs[sample.episode_idx][frac] = dist_prob

                # Top-1 accuracy
                top1_idx = int(torch.argmax(row).item())
                top1_scores[frac].append(1.0 if top1_idx == pref_idx else 0.0)

                # Top-5 accuracy
                k = min(5, row.shape[0])
                topk_indices = torch.topk(row, k=k).indices.tolist()
                top5_scores[frac].append(1.0 if pref_idx in topk_indices else 0.0)

                # Brier score for the true goal
                target = torch.zeros_like(row)
                target[pref_idx] = 1.0
                brier = float(torch.sum((row - target) ** 2).item())
                brier_scores[frac].append(brier)

    peak_probs = [max(fraction_map.values()) for fraction_map in per_episode_probs.values() if fraction_map]

    summary = {
        "model": model_path.stem,
        "fractions": {
            fraction: {
                "distractor": {
                    "mean": float(np.mean(distractor_probs[fraction])) if distractor_probs[fraction] else math.nan,
                    "ci_low": None,
                    "ci_high": None,
                    "values": distractor_probs[fraction],
                },
                "goal": {
                    "mean": float(np.mean(goal_probs[fraction])) if goal_probs[fraction] else math.nan,
                    "ci_low": None,
                    "ci_high": None,
                    "values": goal_probs[fraction],
                },
                "top1": {
                    "mean": float(np.mean(top1_scores[fraction])) if top1_scores[fraction] else math.nan,
                    "ci_low": None,
                    "ci_high": None,
                    "values": top1_scores[fraction],
                },
                "top5": {
                    "mean": float(np.mean(top5_scores[fraction])) if top5_scores[fraction] else math.nan,
                    "ci_low": None,
                    "ci_high": None,
                    "values": top5_scores[fraction],
                },
                "brier": {
                    "mean": float(np.mean(brier_scores[fraction])) if brier_scores[fraction] else math.nan,
                    "ci_low": None,
                    "ci_high": None,
                    "values": brier_scores[fraction],
                },
            }
            for fraction in fractions
        },
        "peak_distractor": {
            "values": peak_probs,
            "mean": float(np.mean(peak_probs)) if peak_probs else math.nan,
            "ci_low": None,
            "ci_high": None,
        },
    }

    for fraction in fractions:
        summary["fractions"][fraction]["distractor"]["ci_low"], summary["fractions"][fraction]["distractor"]["ci_high"] = bootstrap_ci(
            distractor_probs[fraction], seed=seed
        )
        summary["fractions"][fraction]["goal"]["ci_low"], summary["fractions"][fraction]["goal"]["ci_high"] = bootstrap_ci(
            goal_probs[fraction], seed=seed
        )
        summary["fractions"][fraction]["top1"]["ci_low"], summary["fractions"][fraction]["top1"]["ci_high"] = bootstrap_ci(
            top1_scores[fraction], seed=seed
        )
        summary["fractions"][fraction]["top5"]["ci_low"], summary["fractions"][fraction]["top5"]["ci_high"] = bootstrap_ci(
            top5_scores[fraction], seed=seed
        )
        summary["fractions"][fraction]["brier"]["ci_low"], summary["fractions"][fraction]["brier"]["ci_high"] = bootstrap_ci(
            brier_scores[fraction], seed=seed
        )

    peak_low, peak_high = bootstrap_ci(peak_probs, seed=seed)
    summary["peak_distractor"]["ci_low"] = peak_low
    summary["peak_distractor"]["ci_high"] = peak_high

    return summary


def write_csv_summaries(output_dir: Path, summaries: List[Dict[str, object]], fractions: Sequence[float]) -> None:
    fraction_path = output_dir / "exp_2_fraction_metrics.csv"
    peak_path = output_dir / "exp_2_peak_metrics.csv"

    with fraction_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "fraction", "metric", "mean", "ci_low", "ci_high"])
        for summary in summaries:
            model_name = summary["model"]
            for fraction in fractions:
                frac_data = summary["fractions"][fraction]
                for metric_name in ("distractor", "goal", "top1", "top5", "brier"):
                    metric = frac_data[metric_name]
                    writer.writerow(
                        [
                            model_name,
                            fraction,
                            metric_name,
                            metric["mean"],
                            metric["ci_low"],
                            metric["ci_high"],
                        ]
                    )

    with peak_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "mean", "ci_low", "ci_high"])
        for summary in summaries:
            peak = summary["peak_distractor"]
            writer.writerow([
                summary["model"],
                peak["mean"],
                peak["ci_low"],
                peak["ci_high"],
            ])


def write_json_summary(output_dir: Path, summaries: List[Dict[str, object]], config: Dict[str, object]) -> None:
    payload = {
        "config": config,
        "results": summaries,
    }
    with (output_dir / "exp_2_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 2 distractor episodes.")
    parser.add_argument("--run-id", type=int, default=1, help="Simulation run identifier")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/processed/ucsd_walk_full.graphml"),
        help="Path to processed campus graph",
    )
    parser.add_argument("--keepers", type=Path, default=Path("checkpoints/keepers"), help="Directory of model checkpoints")
    parser.add_argument("--distractors", type=Path, default=None, help="Path to distractor episodes JSON")
    parser.add_argument(
        "--fractions",
        type=parse_fractions,
        default=parse_fractions("0.1,0.2,0.5,0.75,0.9"),
        help="Comma-separated list of observation fractions",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Directory to store metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path("data/simulation_data") / f"run_{args.run_id}"
    distractor_path = args.distractors or (run_dir / "exp_2_distractors.json")
    output_dir = args.output or (run_dir / "visualizations" / "exp_2")

    if not distractor_path.exists():
        raise FileNotFoundError(f"Distractor episodes not found: {distractor_path}")

    ensure_output_dir(output_dir)

    device = get_device()
    graph = nx.read_graphml(args.graph)
    world_graph = WorldGraph(graph)
    graph_data = create_graph_data(world_graph, device)

    preparator = prepare_trajectory_preparator(world_graph.G)

    episodes = load_distractor_episodes(distractor_path)
    samples, _ = build_samples_dict(episodes, args.fractions)

    # Determine sequence length dynamically to avoid truncation if requested length is too short
    max_path_len = max(len(sample.path) for sample in samples) if samples else args.max_seq_len
    seq_len = max(args.max_seq_len, max_path_len)

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
            fractions=args.fractions,
            poi_nodes=world_graph.poi_nodes,
            seed=args.seed,
            batch_size=args.batch_size,
            max_seq_len=seq_len,
        )
        summaries.append(summary)

    config_payload = {
        "run_id": args.run_id,
        "graph_path": str(args.graph),
        "distractor_path": str(distractor_path),
        "fractions": args.fractions,
        "batch_size": args.batch_size,
        "max_seq_len": seq_len,
        "seed": args.seed,
        "models": [path.name for path in model_paths],
    }

    write_csv_summaries(output_dir, summaries, args.fractions)
    write_json_summary(output_dir, summaries, config_payload)

    print(f"Processed {len(samples)} samples across {len(model_paths)} model(s).")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
