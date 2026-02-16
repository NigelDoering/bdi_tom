"""Experiment 2: Preference-Proximity Dissociation Test

Evaluates whether models maintain low distractor probability despite spatial proximity.
Models with genuine ToM should resist proximity-induced belief shifts.

Metrics per observation fraction f in {0.1, 0.2, 0.5, 0.75, 0.9}:
- Distractor probability Q_phi(g_tilde | v_{1:floor(f*t*)})
- Goal probability Q_phi(g* | v_{1:floor(f*t*)})
- Peak distractor probability (max across fractions)
- 95% CI via bootstrap resampling

Supported models:
- transformer: PerNodeTransformerPredictor
- lstm: PerNodeToMPredictor  
- bdi_vae: BDIVAEPredictor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.exp_2_data import (
    PrefixSample,
    generate_prefix_samples,
    generate_prefix_samples_full_trajectory,
    load_distractor_episodes,
)
from graph_controller.world_graph import WorldGraph
from models.utils.utils import get_device


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap mean and 95% CI."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    mean = float(arr.mean())
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def load_model(model_path: str, model_type: str, num_nodes: int, num_poi_nodes: int, device: torch.device):
    """Load model based on type."""
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    config = checkpoint.get('config', {})
    
    if model_type == "transformer":
        from models.baseline_transformer.baseline_transformer_model import PerNodeTransformerPredictor
        model = PerNodeTransformerPredictor(
            num_nodes=num_nodes,
            num_agents=config.get('num_agents', 100),
            num_poi_nodes=num_poi_nodes,
            num_categories=config.get('num_categories', 7),
            node_embedding_dim=config.get('node_embedding_dim', 128),
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('dropout', 0.1),
        )
    elif model_type == "lstm":
        from models.baseline_lstm.baseline_lstm_model import PerNodeToMPredictor
        # LSTM training uses 100 agents but saves config with 1 agent
        model = PerNodeToMPredictor(
            num_nodes=num_nodes,
            num_agents=100,  # Always 100 agents (matches transformer pretrained embeddings)
            num_poi_nodes=num_poi_nodes,
            num_categories=config.get('num_categories', 7),
            node_embedding_dim=config.get('node_embedding_dim', 64),
            temporal_dim=config.get('temporal_dim', 64),
            agent_dim=config.get('agent_dim', 64),
            fusion_dim=config.get('fusion_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.1),
            num_heads=config.get('num_heads', 4),
            freeze_embedding=False,
        )
    elif model_type == "bdi_vae":
        from models.vae_bdi_simple.bdi_vae_v3_model import SequentialConditionalBDIVAE
        # SC-BDI VAE V3 model (OURS)
        model = SequentialConditionalBDIVAE(
            num_nodes=num_nodes,
            num_agents=config.get('num_agents', 100),
            num_poi_nodes=num_poi_nodes,
            num_categories=config.get('num_categories', 7),
            node_embedding_dim=config.get('node_embedding_dim', 64),
            fusion_dim=config.get('fusion_dim', 128),
            belief_latent_dim=config.get('belief_latent_dim', 32),
            desire_latent_dim=config.get('desire_latent_dim', 16),
            intention_latent_dim=config.get('intention_latent_dim', 32),
            vae_hidden_dim=config.get('vae_hidden_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.1),
            use_progress=config.get('use_progress', True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def infer_model_type(model_path: str) -> str:
    """Auto-detect model type from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    keys_str = str(list(state_dict.keys()))
    
    if "belief_vae" in keys_str or "desire_vae" in keys_str:
        return "bdi_vae"
    elif "transformer_encoder" in keys_str:
        return "transformer"
    elif "lstm" in keys_str.lower() or "rnn" in keys_str.lower():
        return "lstm"
    elif "embedding_pipeline" in keys_str and "transformer" not in keys_str:
        return "lstm"
    else:
        return "transformer"


@torch.no_grad()
def run_inference(model, model_type: str, node_tensor: torch.Tensor, 
                  padding_mask: torch.Tensor, seq_lengths: List[int], device: torch.device) -> torch.Tensor:
    """Run model inference and return goal probabilities."""
    batch_size = node_tensor.shape[0]
    history_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=device)
    
    if model_type == "transformer":
        outputs = model(node_tensor, padding_mask=padding_mask)
        goal_logits = outputs["goal"]  # [batch, seq_len, num_poi]
        probs = []
        for j in range(batch_size):
            last_pos = seq_lengths[j] - 1
            logits = goal_logits[j, last_pos]
            probs.append(F.softmax(logits, dim=-1))
        return torch.stack(probs)
    
    elif model_type == "lstm":
        outputs = model(node_tensor, history_lengths)
        goal_logits = outputs["goal"]  # [batch, num_poi]
        return F.softmax(goal_logits, dim=-1)
    
    elif model_type == "bdi_vae":
        # SequentialConditionalBDIVAE uses keyword arguments
        # path_progress defaults to 0.5 (mid-trajectory) when not provided
        path_progress = torch.full((batch_size,), 0.5, device=device)
        outputs = model(
            history_node_indices=node_tensor,
            history_lengths=history_lengths,
            path_progress=path_progress,
            compute_loss=False,
        )
        goal_logits = outputs["goal"]  # [batch, num_poi]
        return F.softmax(goal_logits, dim=-1)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def evaluate_distractor_episodes(
    model,
    model_type: str,
    samples: List[PrefixSample],
    poi_nodes: List[str],
    node_to_idx: Dict[str, int],
    device: torch.device,
    fractions: List[float],
    batch_size: int = 32,
) -> Dict[float, Dict[str, List[float]]]:
    """Evaluate model on distractor episodes with comprehensive metrics."""
    model.eval()
    poi_to_idx = {node: idx for idx, node in enumerate(poi_nodes)}
    num_pois = len(poi_nodes)
    
    samples_by_frac: Dict[float, List[PrefixSample]] = defaultdict(list)
    for s in samples:
        samples_by_frac[s.fraction].append(s)
    
    results_by_frac: Dict[float, Dict[str, List[float]]] = {}
    
    for frac in fractions:
        frac_samples = samples_by_frac.get(frac, [])
        if not frac_samples:
            continue
        
        distractor_probs = []
        goal_probs = []
        top1_correct = []
        top5_correct = []
        brier_scores = []
        
        for i in range(0, len(frac_samples), batch_size):
            batch = frac_samples[i : i + batch_size]
            
            paths = []
            goal_indices = []
            distractor_indices = []
            
            for s in batch:
                node_ids = []
                for step in s.path:
                    node_id = step[0] if isinstance(step, (list, tuple)) else step
                    if node_id in node_to_idx:
                        node_ids.append(node_to_idx[node_id])
                
                if len(node_ids) < 1:
                    continue
                if s.preferred_goal not in poi_to_idx or s.distractor_goal not in poi_to_idx:
                    continue
                
                paths.append(node_ids)
                goal_indices.append(poi_to_idx[s.preferred_goal])
                distractor_indices.append(poi_to_idx[s.distractor_goal])
            
            if not paths:
                continue
            
            max_len = max(len(p) for p in paths)
            batch_size_actual = len(paths)
            
            node_tensor = torch.zeros(batch_size_actual, max_len, dtype=torch.long, device=device)
            padding_mask = torch.ones(batch_size_actual, max_len, dtype=torch.bool, device=device)
            seq_lengths = []
            
            for j, p in enumerate(paths):
                node_tensor[j, : len(p)] = torch.tensor(p, dtype=torch.long)
                padding_mask[j, : len(p)] = False
                seq_lengths.append(len(p))
            
            probs_batch = run_inference(model, model_type, node_tensor, padding_mask, seq_lengths, device)
            
            for j in range(batch_size_actual):
                probs = probs_batch[j]
                goal_idx = goal_indices[j]
                distractor_idx = distractor_indices[j]
                
                # Basic probabilities
                distractor_probs.append(probs[distractor_idx].item())
                goal_probs.append(probs[goal_idx].item())
                
                # Top-1 accuracy: is the highest probability the true goal?
                top1_pred = probs.argmax().item()
                top1_correct.append(1.0 if top1_pred == goal_idx else 0.0)
                
                # Top-5 accuracy: is the true goal in the top 5 predictions?
                top5_preds = probs.topk(min(5, num_pois)).indices.tolist()
                top5_correct.append(1.0 if goal_idx in top5_preds else 0.0)
                
                # Brier score: (1 - p(true_goal))^2 + sum of p(other)^2
                # Simplified: sum over all classes of (p_i - y_i)^2 where y is one-hot
                probs_np = probs.cpu().numpy()
                one_hot = np.zeros(num_pois)
                one_hot[goal_idx] = 1.0
                brier = np.sum((probs_np - one_hot) ** 2)
                brier_scores.append(brier)
        
        results_by_frac[frac] = {
            "distractor_probs": distractor_probs,
            "goal_probs": goal_probs,
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "brier_scores": brier_scores,
        }
    
    return results_by_frac


def compute_metrics(
    results_by_frac: Dict[float, Dict[str, List[float]]], fractions: List[float], seed: int
) -> Dict[str, object]:
    """Compute summary metrics with bootstrap CIs."""
    metrics = {"fractions": {}}
    
    all_distractor_probs = []
    all_goal_probs = []
    all_top1 = []
    all_top5 = []
    all_brier = []
    peak_per_episode: Dict[int, float] = defaultdict(lambda: 0.0)
    
    for frac in fractions:
        if frac not in results_by_frac:
            continue
        
        d_probs = results_by_frac[frac]["distractor_probs"]
        g_probs = results_by_frac[frac]["goal_probs"]
        top1 = results_by_frac[frac]["top1_correct"]
        top5 = results_by_frac[frac]["top5_correct"]
        brier = results_by_frac[frac]["brier_scores"]
        
        d_mean, d_lo, d_hi = bootstrap_ci(d_probs, seed=seed)
        g_mean, g_lo, g_hi = bootstrap_ci(g_probs, seed=seed)
        top1_mean, top1_lo, top1_hi = bootstrap_ci(top1, seed=seed)
        top5_mean, top5_lo, top5_hi = bootstrap_ci(top5, seed=seed)
        brier_mean, brier_lo, brier_hi = bootstrap_ci(brier, seed=seed)
        
        metrics["fractions"][frac] = {
            "distractor_prob": {"mean": d_mean, "ci_lo": d_lo, "ci_hi": d_hi, "n": len(d_probs)},
            "goal_prob": {"mean": g_mean, "ci_lo": g_lo, "ci_hi": g_hi, "n": len(g_probs)},
            "top1_accuracy": {"mean": top1_mean * 100, "ci_lo": top1_lo * 100, "ci_hi": top1_hi * 100},
            "top5_accuracy": {"mean": top5_mean * 100, "ci_lo": top5_lo * 100, "ci_hi": top5_hi * 100},
            "brier_score": {"mean": brier_mean, "ci_lo": brier_lo, "ci_hi": brier_hi},
        }
        
        all_distractor_probs.extend(d_probs)
        all_goal_probs.extend(g_probs)
        all_top1.extend(top1)
        all_top5.extend(top5)
        all_brier.extend(brier)
        for i, p in enumerate(d_probs):
            peak_per_episode[i] = max(peak_per_episode[i], p)
    
    peak_values = list(peak_per_episode.values())
    peak_mean, peak_lo, peak_hi = bootstrap_ci(peak_values, seed=seed)
    metrics["peak_distractor_prob"] = {"mean": peak_mean, "ci_lo": peak_lo, "ci_hi": peak_hi}
    
    overall_mean, overall_lo, overall_hi = bootstrap_ci(all_distractor_probs, seed=seed)
    metrics["overall_distractor_prob"] = {"mean": overall_mean, "ci_lo": overall_lo, "ci_hi": overall_hi}
    
    # Overall metrics
    top1_overall = bootstrap_ci(all_top1, seed=seed)
    top5_overall = bootstrap_ci(all_top5, seed=seed)
    brier_overall = bootstrap_ci(all_brier, seed=seed)
    goal_overall = bootstrap_ci(all_goal_probs, seed=seed)
    
    metrics["overall_top1_accuracy"] = {"mean": top1_overall[0] * 100, "ci_lo": top1_overall[1] * 100, "ci_hi": top1_overall[2] * 100}
    metrics["overall_top5_accuracy"] = {"mean": top5_overall[0] * 100, "ci_lo": top5_overall[1] * 100, "ci_hi": top5_overall[2] * 100}
    metrics["overall_brier_score"] = {"mean": brier_overall[0], "ci_lo": brier_overall[1], "ci_hi": brier_overall[2]}
    metrics["overall_goal_prob"] = {"mean": goal_overall[0], "ci_lo": goal_overall[1], "ci_hi": goal_overall[2]}
    
    return metrics


def print_results(metrics: Dict, model_name: str, n_episodes: int, n_samples: int, fractions: List[float], reference_type: str = "t*"):
    """Print formatted results table."""
    ref_label = "t* (closest approach)" if reference_type == "t*" else "full trajectory"
    
    print("\n" + "=" * 115)
    print(f"EXPERIMENT 2: PREFERENCE-PROXIMITY DISSOCIATION TEST (fractions relative to {ref_label})")
    print("=" * 115)
    print(f"Model: {model_name}")
    print(f"Episodes: {n_episodes}, Samples: {n_samples}")
    print("-" * 115)
    print(f"{'Frac':<6} {'Distractor P':<22} {'Goal P':<22} {'Top-1 Acc':<18} {'Top-5 Acc':<18} {'Brier':<15}")
    print("-" * 115)
    
    for frac in fractions:
        if frac not in metrics["fractions"]:
            continue
        m = metrics["fractions"][frac]
        d = m["distractor_prob"]
        g = m["goal_prob"]
        t1 = m["top1_accuracy"]
        t5 = m["top5_accuracy"]
        b = m["brier_score"]
        print(
            f"{frac:<6.2f} "
            f"{d['mean']:.4f} [{d['ci_lo']:.4f},{d['ci_hi']:.4f}]  "
            f"{g['mean']:.4f} [{g['ci_lo']:.4f},{g['ci_hi']:.4f}]  "
            f"{t1['mean']:5.2f}% [{t1['ci_lo']:4.1f},{t1['ci_hi']:4.1f}]  "
            f"{t5['mean']:5.2f}% [{t5['ci_lo']:4.1f},{t5['ci_hi']:4.1f}]  "
            f"{b['mean']:.4f}"
        )
    
    print("-" * 115)
    peak = metrics["peak_distractor_prob"]
    overall_d = metrics["overall_distractor_prob"]
    overall_t1 = metrics["overall_top1_accuracy"]
    overall_t5 = metrics["overall_top5_accuracy"]
    overall_b = metrics["overall_brier_score"]
    
    print(f"Peak Distractor Prob:    {peak['mean']:.4f} [{peak['ci_lo']:.4f}, {peak['ci_hi']:.4f}]")
    print(f"Overall Distractor Prob: {overall_d['mean']:.4f} [{overall_d['ci_lo']:.4f}, {overall_d['ci_hi']:.4f}]")
    print(f"Overall Top-1 Accuracy:  {overall_t1['mean']:.2f}% [{overall_t1['ci_lo']:.2f}, {overall_t1['ci_hi']:.2f}]")
    print(f"Overall Top-5 Accuracy:  {overall_t5['mean']:.2f}% [{overall_t5['ci_lo']:.2f}, {overall_t5['ci_hi']:.2f}]")
    print(f"Overall Brier Score:     {overall_b['mean']:.4f} [{overall_b['ci_lo']:.4f}, {overall_b['ci_hi']:.4f}]")
    print("=" * 115)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: Preference-Proximity Dissociation Test")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="auto", 
                        choices=["auto", "transformer", "lstm", "bdi_vae"],
                        help="Model type (auto-detect if not specified)")
    parser.add_argument("--run-id", type=int, default=8, help="Simulation run ID")
    parser.add_argument("--graph", type=Path, default=Path("data/processed/ucsd_walk_full.graphml"))
    parser.add_argument("--distractors", type=Path, default=None)
    parser.add_argument("--fractions", type=str, default="0.1,0.2,0.5,0.75,0.9")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split-indices", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    fractions = [float(f.strip()) for f in args.fractions.split(",")]
    run_dir = Path("data/simulation_data") / f"run_{args.run_id}"
    distractor_path = args.distractors or (run_dir / "exp_2_distractors.json")
    output_dir = args.output or (run_dir / "visualizations" / "exp_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not distractor_path.exists():
        raise FileNotFoundError(f"Distractor episodes not found: {distractor_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    # Auto-detect model type
    if args.model_type == "auto":
        model_type = infer_model_type(str(args.model_path))
        print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load graph
    print(f"Loading graph from {args.graph}")
    graph = nx.read_graphml(args.graph)
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    
    print(f"Graph: {len(graph.nodes())} nodes, {len(poi_nodes)} POIs")
    
    # Load episodes
    print(f"Loading distractor episodes from {distractor_path}")
    episodes = load_distractor_episodes(distractor_path)
    
    if args.split_indices and args.split_indices.exists():
        with open(args.split_indices) as f:
            splits = json.load(f)
        test_idx = set(splits.get("test_indices", []))
        episodes = [ep for i, ep in enumerate(episodes) if i in test_idx]
        print(f"Filtered to {len(episodes)} test episodes")
    
    # Load model (once, use for both evaluations)
    print(f"Loading {model_type} model from {args.model_path}")
    model = load_model(str(args.model_path), model_type, len(graph.nodes()), len(poi_nodes), device)
    
    # ========================================================================
    # EVALUATION 1: Fractions relative to t* (closest approach to distractor)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Evaluating with fractions relative to t* (closest approach)")
    print("=" * 80)
    
    samples_tstar = generate_prefix_samples(episodes, fractions)
    print(f"Generated {len(samples_tstar)} prefix samples (relative to t*)")
    
    results_tstar = evaluate_distractor_episodes(
        model=model,
        model_type=model_type,
        samples=samples_tstar,
        poi_nodes=poi_nodes,
        node_to_idx=node_to_idx,
        device=device,
        fractions=fractions,
        batch_size=args.batch_size,
    )
    
    metrics_tstar = compute_metrics(results_tstar, fractions, args.seed)
    metrics_tstar["model"] = str(args.model_path.name)
    metrics_tstar["model_type"] = model_type
    metrics_tstar["run_id"] = args.run_id
    metrics_tstar["n_episodes"] = len(episodes)
    metrics_tstar["n_samples"] = len(samples_tstar)
    metrics_tstar["reference"] = "t_star"
    
    print_results(metrics_tstar, args.model_path.name, len(episodes), len(samples_tstar), fractions, "t*")
    
    # ========================================================================
    # EVALUATION 2: Fractions relative to full trajectory length
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Evaluating with fractions relative to full trajectory")
    print("=" * 80)
    
    samples_full = generate_prefix_samples_full_trajectory(episodes, fractions)
    print(f"Generated {len(samples_full)} prefix samples (relative to full trajectory)")
    
    results_full = evaluate_distractor_episodes(
        model=model,
        model_type=model_type,
        samples=samples_full,
        poi_nodes=poi_nodes,
        node_to_idx=node_to_idx,
        device=device,
        fractions=fractions,
        batch_size=args.batch_size,
    )
    
    metrics_full = compute_metrics(results_full, fractions, args.seed)
    metrics_full["model"] = str(args.model_path.name)
    metrics_full["model_type"] = model_type
    metrics_full["run_id"] = args.run_id
    metrics_full["n_episodes"] = len(episodes)
    metrics_full["n_samples"] = len(samples_full)
    metrics_full["reference"] = "full_trajectory"
    
    print_results(metrics_full, args.model_path.name, len(episodes), len(samples_full), fractions, "full")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    # Save t* results
    output_file_tstar = output_dir / f"exp2_{args.model_path.stem}_results.json"
    with open(output_file_tstar, "w") as f:
        json.dump(metrics_tstar, f, indent=2)
    print(f"\nResults (t* reference) saved to {output_file_tstar}")
    
    # Save full trajectory results
    output_file_full = output_dir / f"exp2_{args.model_path.stem}_full_traj_results.json"
    with open(output_file_full, "w") as f:
        json.dump(metrics_full, f, indent=2)
    print(f"Results (full trajectory reference) saved to {output_file_full}")
    
    # Save combined results
    combined = {
        "model": str(args.model_path.name),
        "model_type": model_type,
        "n_episodes": len(episodes),
        "metrics_relative_to_tstar": metrics_tstar,
        "metrics_relative_to_full_trajectory": metrics_full,
    }
    output_file_combined = output_dir / f"exp2_{args.model_path.stem}_combined_results.json"
    with open(output_file_combined, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined results saved to {output_file_combined}")


if __name__ == "__main__":
    main()
