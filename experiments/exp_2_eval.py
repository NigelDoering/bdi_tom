"""Experiment 2: Preference-Proximity Dissociation Test

Evaluates whether models maintain low distractor probability despite spatial
proximity.  Supports two variants:

  • **2a (same-category):** Distractor is a low-preference POI in the *same*
    category as the goal.  Tests within-category preference discrimination.
    Episodes from ``experiments/data/exp_2a_test_set.json``.

  • **2b (cross-category):** Distractor is from a *different* (bottom-ranked)
    category.  Tests category-level reasoning.
    Episodes from ``experiments/data/exp_2b_test_set.json``.

Metrics per observation fraction f ∈ {0.1, 0.2, 0.5, 0.75, 0.9}:
  • Distractor probability  P(distractor | v_{1:t})
  • True-goal probability   P(goal | v_{1:t})
  • Top-1 / Top-5 accuracy, Brier score
  • Goal-to-distractor probability ratio
  • Peak distractor probability (max across fractions)
  • 95 % bootstrap CIs

Supported models (same loading logic as exp_1):
  • transformer — PerNodeTransformerPredictor
  • lstm        — PerNodeToMPredictor
  • sc_bdi_vae  — SequentialConditionalBDIVAE  (new_bdi)

Usage:
    # Run all three models on both variants:
    python experiments/exp_2_eval.py --run_all --variant both

    # Run only same-category variant:
    python experiments/exp_2_eval.py --run_all --variant a

    # Run only cross-category variant:
    python experiments/exp_2_eval.py --run_all --variant b

    # Single model, specific variant:
    python experiments/exp_2_eval.py \\
        --model_path checkpoints/keepers/best_model-OURS.pt \\
        --model_type sc_bdi_vae --variant a
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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ── model imports (matching exp_1.py exactly) ────────────────────────────────
from models.baseline_transformer.baseline_transformer_model import PerNodeTransformerPredictor
from models.baseline_lstm.baseline_lstm_model import PerNodeToMPredictor
from models.new_bdi.bdi_vae_v3_model import (
    SequentialConditionalBDIVAE,
    create_sc_bdi_vae_v3,
)
from models.new_bdi.bdi_dataset_v2 import (
    BDIVAEDatasetV2,
    collate_bdi_samples_v2,
)
from graph_controller.world_graph import WorldGraph
from models.utils.utils import get_device
from torch.utils.data import DataLoader, Subset


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════


def load_filtered_episodes(path: Path) -> List[Dict]:
    """Load distractor episodes from ``exp_2_data_filtering.py`` output."""
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes found in {path}")
    print(f"  📂 Loaded {len(episodes):,} distractor episodes from {path}")
    return episodes


def truncate_path(path: List, fraction: float, reference_length: int) -> List:
    """Truncate a path to ``fraction`` of ``reference_length`` steps."""
    steps = max(1, math.floor(fraction * reference_length))
    steps = min(steps, len(path))
    return path[:steps]


# ═════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP / METRICS
# ═════════════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap mean and 95% CI."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    return float(arr.mean()), float(np.percentile(boot_means, 100 * alpha / 2)), float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (copied from exp_1.py for consistency)
# ═════════════════════════════════════════════════════════════════════════════


def load_transformer_model(
    checkpoint_path: str, num_nodes: int, num_poi_nodes: int, device: torch.device
) -> PerNodeTransformerPredictor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = PerNodeTransformerPredictor(
        num_nodes=num_nodes,
        num_agents=100,
        num_poi_nodes=num_poi_nodes,
        num_categories=7,
        node_embedding_dim=128,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_lstm_model(
    checkpoint_path: str, num_nodes: int, num_poi_nodes: int, device: torch.device
) -> PerNodeToMPredictor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        config = checkpoint["config"]
        agent_key = "embedding_pipeline.agent_encoder.agent_context.agent_emb.agent_embedding.weight"
        if agent_key in checkpoint["model_state_dict"]:
            actual_num_agents = checkpoint["model_state_dict"][agent_key].shape[0]
        else:
            actual_num_agents = config.get("num_agents", 100)
        model = PerNodeToMPredictor(
            num_nodes=config.get("num_nodes", num_nodes),
            num_agents=actual_num_agents,
            num_poi_nodes=config.get("num_poi_nodes", num_poi_nodes),
            num_categories=config.get("num_categories", 7),
            node_embedding_dim=config.get("node_embedding_dim", 64),
            temporal_dim=config.get("temporal_dim", 64),
            agent_dim=config.get("agent_dim", 64),
            fusion_dim=config.get("fusion_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.1),
            num_heads=config.get("num_heads", 4),
            freeze_embedding=config.get("freeze_embedding", False),
        )
    else:
        model = PerNodeToMPredictor(
            num_nodes=num_nodes, num_agents=100, num_poi_nodes=num_poi_nodes,
            num_categories=7, node_embedding_dim=64, temporal_dim=64,
            agent_dim=64, fusion_dim=128, hidden_dim=256, dropout=0.1,
            num_heads=4, freeze_embedding=False,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_sc_bdi_vae_model(
    checkpoint_path: str, num_nodes: int, num_poi_nodes: int, num_agents: int, device: torch.device
) -> SequentialConditionalBDIVAE:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_sc_bdi_vae_v3(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=7,
        node_embedding_dim=64,
        fusion_dim=128,
        belief_latent_dim=32,
        desire_latent_dim=16,
        intention_latent_dim=32,
        vae_hidden_dim=128,
        hidden_dim=256,
        dropout=0.1,
        use_progress=False,
        use_temporal=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print("  ✅ Loaded SC-BDI-VAE checkpoint (all keys matched)")
    model = model.to(device)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _extract_node_ids(path: List) -> List[str]:
    """Extract plain node-ID strings from a path (handles [node, goal] pairs)."""
    out = []
    for step in path:
        if isinstance(step, (list, tuple)) and len(step) >= 1:
            out.append(str(step[0]))
        else:
            out.append(str(step))
    return out


def _record(
    probs: torch.Tensor,
    goal_idx: int,
    dist_idx: int,
    num_pois: int,
    d_probs: List[float],
    g_probs: List[float],
    top1: List[float],
    top5: List[float],
    brier: List[float],
) -> None:
    """Append per-sample metrics to the accumulation lists."""
    d_probs.append(probs[dist_idx].item())
    g_probs.append(probs[goal_idx].item())
    top1.append(1.0 if probs.argmax().item() == goal_idx else 0.0)
    top5_preds = probs.topk(min(5, num_pois)).indices.tolist()
    top5.append(1.0 if goal_idx in top5_preds else 0.0)
    one_hot = np.zeros(num_pois)
    one_hot[goal_idx] = 1.0
    brier.append(float(np.sum((probs.cpu().numpy() - one_hot) ** 2)))


# ═════════════════════════════════════════════════════════════════════════════
# PER-MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _evaluate_transformer(
    model: PerNodeTransformerPredictor,
    episodes: List[Dict],
    poi_nodes: List[str],
    node_to_idx: Dict[str, int],
    device: torch.device,
    fractions: List[float],
    reference: str,
    batch_size: int = 32,
) -> Dict[float, Dict[str, List[float]]]:
    """Evaluate transformer on distractor episodes."""
    model.eval()
    poi_to_idx = {n: i for i, n in enumerate(poi_nodes)}
    results: Dict[float, Dict[str, List[float]]] = {}
    n_fracs = len(fractions)

    for frac_i, frac in enumerate(fractions):
        d_probs, g_probs, top1, top5, brier = [], [], [], [], []

        # Build batch items
        items = []
        for ep in episodes:
            ref_len = ep["closest_index"] if reference == "tstar" else ep["path_length"]
            if ref_len < 2:
                continue
            trunc = truncate_path(ep["path"], frac, ref_len)
            node_ids = _extract_node_ids(trunc)
            indices = [node_to_idx[n] for n in node_ids if n in node_to_idx]
            if not indices:
                continue
            goal = ep["preferred_goal"]
            dist = ep["distractor_goal"]
            if goal not in poi_to_idx or dist not in poi_to_idx:
                continue
            items.append((indices, poi_to_idx[goal], poi_to_idx[dist]))

        # Batch inference
        n_batches = (len(items) + batch_size - 1) // batch_size
        batch_iter = range(0, len(items), batch_size)
        for i in tqdm(batch_iter, desc=f"  Transformer f={frac:.2f} ({frac_i+1}/{n_fracs})", total=n_batches, leave=False):
            batch = items[i: i + batch_size]
            max_len = max(len(x[0]) for x in batch)
            node_tensor = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            pad_mask = torch.ones(len(batch), max_len, dtype=torch.bool, device=device)
            seq_lens = []
            goal_idxs, dist_idxs = [], []

            for j, (ids, gi, di) in enumerate(batch):
                node_tensor[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                pad_mask[j, : len(ids)] = False
                seq_lens.append(len(ids))
                goal_idxs.append(gi)
                dist_idxs.append(di)

            outputs = model(node_tensor, padding_mask=pad_mask)
            goal_logits = outputs["goal"]  # [B, seq, n_poi]

            for j in range(len(batch)):
                last = seq_lens[j] - 1
                probs = F.softmax(goal_logits[j, last], dim=-1)
                _record(probs, goal_idxs[j], dist_idxs[j], len(poi_nodes),
                        d_probs, g_probs, top1, top5, brier)

        results[frac] = dict(distractor_probs=d_probs, goal_probs=g_probs,
                             top1_correct=top1, top5_correct=top5, brier_scores=brier)
    return results


@torch.no_grad()
def _evaluate_lstm(
    model: PerNodeToMPredictor,
    episodes: List[Dict],
    poi_nodes: List[str],
    node_to_idx: Dict[str, int],
    device: torch.device,
    fractions: List[float],
    reference: str,
    batch_size: int = 32,
) -> Dict[float, Dict[str, List[float]]]:
    """Evaluate LSTM on distractor episodes."""
    model.eval()
    poi_to_idx = {n: i for i, n in enumerate(poi_nodes)}
    results: Dict[float, Dict[str, List[float]]] = {}
    n_fracs = len(fractions)

    for frac_i, frac in enumerate(fractions):
        d_probs, g_probs, top1, top5, brier = [], [], [], [], []

        items = []
        for ep in episodes:
            ref_len = ep["closest_index"] if reference == "tstar" else ep["path_length"]
            if ref_len < 2:
                continue
            trunc = truncate_path(ep["path"], frac, ref_len)
            node_ids = _extract_node_ids(trunc)
            indices = [node_to_idx[n] for n in node_ids if n in node_to_idx]
            if not indices:
                continue
            goal = ep["preferred_goal"]
            dist = ep["distractor_goal"]
            if goal not in poi_to_idx or dist not in poi_to_idx:
                continue
            items.append((indices, poi_to_idx[goal], poi_to_idx[dist]))

        n_batches = (len(items) + batch_size - 1) // batch_size
        batch_iter = range(0, len(items), batch_size)
        for i in tqdm(batch_iter, desc=f"  LSTM f={frac:.2f} ({frac_i+1}/{n_fracs})", total=n_batches, leave=False):
            batch = items[i: i + batch_size]
            max_len = max(len(x[0]) for x in batch)
            node_tensor = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            lengths = []
            goal_idxs, dist_idxs = [], []

            for j, (ids, gi, di) in enumerate(batch):
                node_tensor[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                lengths.append(len(ids))
                goal_idxs.append(gi)
                dist_idxs.append(di)

            history_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            outputs = model(node_tensor, history_lengths)
            goal_logits = outputs["goal"]  # [B, n_poi]

            for j in range(len(batch)):
                probs = F.softmax(goal_logits[j], dim=-1)
                _record(probs, goal_idxs[j], dist_idxs[j], len(poi_nodes),
                        d_probs, g_probs, top1, top5, brier)

        results[frac] = dict(distractor_probs=d_probs, goal_probs=g_probs,
                             top1_correct=top1, top5_correct=top5, brier_scores=brier)
    return results


@torch.no_grad()
def _evaluate_sc_bdi_vae(
    model: SequentialConditionalBDIVAE,
    episodes: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    device: torch.device,
    fractions: List[float],
    reference: str,
    batch_size: int = 32,
) -> Dict[float, Dict[str, List[float]]]:
    """Evaluate SC-BDI-VAE on distractor episodes.

    Uses BDIVAEDatasetV3 + collate_bdi_samples_v3 for correct
    feature handling (matching exp_1.py).
    """
    model.eval()
    poi_to_idx = {n: i for i, n in enumerate(poi_nodes)}
    results: Dict[float, Dict[str, List[float]]] = {}
    n_fracs = len(fractions)

    for frac_i, frac in enumerate(fractions):
        d_probs, g_probs, top1, top5, brier = [], [], [], [], []
        print(f"    fraction {frac:.2f} ({frac_i+1}/{n_fracs}) ...")

        # Build pseudo-trajectories for BDIVAEDatasetV3
        trajs_for_dataset: List[Dict] = []
        episode_meta: List[Tuple[int, int, int]] = []  # (traj_idx, goal_poi_idx, dist_poi_idx)

        for ep in episodes:
            ref_len = ep["closest_index"] if reference == "tstar" else ep["path_length"]
            if ref_len < 2:
                continue
            trunc = truncate_path(ep["path"], frac, ref_len)
            if len(trunc) < 2:
                continue
            goal = ep["preferred_goal"]
            dist = ep["distractor_goal"]
            if goal not in poi_to_idx or dist not in poi_to_idx:
                continue

            # Derive agent_id (integer) from string like "agent_042"
            agent_key = ep.get("agent_id", "agent_000")
            try:
                agent_int = int(agent_key.split("_")[1])
            except (IndexError, ValueError):
                agent_int = 0

            traj_dict = {
                "path": trunc,
                "goal_node": goal,
                "agent_id": agent_int,
                "hour": ep.get("observation_hour", 0),
                "day_of_week": ep.get("day_of_week", 0),
            }
            traj_idx = len(trajs_for_dataset)
            trajs_for_dataset.append(traj_dict)
            episode_meta.append((traj_idx, poi_to_idx[goal], poi_to_idx[dist]))

        if not trajs_for_dataset:
            results[frac] = dict(distractor_probs=[], goal_probs=[],
                                 top1_correct=[], top5_correct=[], brier_scores=[])
            continue

        # Create dataset — takes the last sample per trajectory for full-prefix prediction
        dataset = BDIVAEDatasetV3(
            trajectories=trajs_for_dataset,
            graph=graph,
            poi_nodes=poi_nodes,
            min_traj_length=1,
            include_progress=True,
        )

        # Map traj_idx → last sample index (= prediction from the full truncated prefix)
        traj_to_last: Dict[int, int] = {}
        for traj_id, sample_indices in dataset.trajectory_samples.items():
            if sample_indices:
                traj_to_last[traj_id] = sample_indices[-1]

        # Build ordered eval: (sample_idx, goal_poi_idx, dist_poi_idx)
        eval_items: List[Tuple[int, int, int]] = []
        for traj_idx, gi, di in episode_meta:
            if traj_idx in traj_to_last:
                eval_items.append((traj_to_last[traj_idx], gi, di))

        if not eval_items:
            results[frac] = dict(distractor_probs=[], goal_probs=[],
                                 top1_correct=[], top5_correct=[], brier_scores=[])
            continue

        # Batch through DataLoader
        sample_indices = [x[0] for x in eval_items]
        eval_dataset = Subset(dataset, sample_indices)
        loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_bdi_samples_v3, num_workers=0,
        )

        all_probs: List[torch.Tensor] = []
        for batch in tqdm(loader, desc=f"  SC-BDI-VAE f={frac:.2f}", leave=False):
            history = batch["history_node_indices"].to(device)
            lengths = batch["history_lengths"].to(device)
            agents = batch["agent_id"].to(device)
            progress = batch["path_progress"].to(device)

            outputs = model(
                history_node_indices=history,
                history_lengths=lengths,
                agent_ids=agents,
                path_progress=progress,
                compute_loss=False,
            )
            all_probs.append(F.softmax(outputs["goal"], dim=-1).cpu())

        all_probs_cat = torch.cat(all_probs, dim=0)

        for j, (_, gi, di) in enumerate(eval_items):
            probs = all_probs_cat[j]
            _record(probs, gi, di, len(poi_nodes),
                    d_probs, g_probs, top1, top5, brier)

        results[frac] = dict(distractor_probs=d_probs, goal_probs=g_probs,
                             top1_correct=top1, top5_correct=top5, brier_scores=brier)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATE METRICS
# ═════════════════════════════════════════════════════════════════════════════


def compute_metrics(
    results_by_frac: Dict[float, Dict[str, List[float]]],
    fractions: List[float],
    seed: int,
    num_pois: int = 230,
) -> Dict:
    metrics: Dict = {"fractions": {}}
    all_d, all_g, all_t1, all_t5, all_b = [], [], [], [], []
    all_ratios: List[float] = []
    peak_per_ep: Dict[int, float] = defaultdict(lambda: 0.0)

    for frac in fractions:
        if frac not in results_by_frac:
            continue
        r = results_by_frac[frac]
        d_m, d_lo, d_hi = bootstrap_ci(r["distractor_probs"], seed=seed)
        g_m, g_lo, g_hi = bootstrap_ci(r["goal_probs"], seed=seed)
        t1_m, t1_lo, t1_hi = bootstrap_ci(r["top1_correct"], seed=seed)
        t5_m, t5_lo, t5_hi = bootstrap_ci(r["top5_correct"], seed=seed)
        b_m, b_lo, b_hi = bootstrap_ci(r["brier_scores"], seed=seed)

        # Goal-to-distractor probability ratio (per episode, then bootstrap)
        # Floor distractor prob at uniform baseline (1/num_pois) to avoid
        # division-by-near-zero inflating ratios to millions.
        dp_floor = 1.0 / num_pois
        ratios = [
            gp / max(dp, dp_floor)
            for gp, dp in zip(r["goal_probs"], r["distractor_probs"])
        ]
        ratio_m, ratio_lo, ratio_hi = bootstrap_ci(ratios, seed=seed)

        metrics["fractions"][frac] = {
            "distractor_prob": {"mean": d_m, "ci_lo": d_lo, "ci_hi": d_hi, "n": len(r["distractor_probs"])},
            "goal_prob": {"mean": g_m, "ci_lo": g_lo, "ci_hi": g_hi, "n": len(r["goal_probs"])},
            "goal_dist_ratio": {"mean": ratio_m, "ci_lo": ratio_lo, "ci_hi": ratio_hi},
            "top1_accuracy": {"mean": t1_m * 100, "ci_lo": t1_lo * 100, "ci_hi": t1_hi * 100},
            "top5_accuracy": {"mean": t5_m * 100, "ci_lo": t5_lo * 100, "ci_hi": t5_hi * 100},
            "brier_score": {"mean": b_m, "ci_lo": b_lo, "ci_hi": b_hi},
        }
        all_d.extend(r["distractor_probs"])
        all_g.extend(r["goal_probs"])
        all_t1.extend(r["top1_correct"])
        all_t5.extend(r["top5_correct"])
        all_b.extend(r["brier_scores"])
        all_ratios.extend(ratios)
        for i, p in enumerate(r["distractor_probs"]):
            peak_per_ep[i] = max(peak_per_ep[i], p)

    pk = list(peak_per_ep.values())
    pk_m, pk_lo, pk_hi = bootstrap_ci(pk, seed=seed)
    metrics["peak_distractor_prob"] = {"mean": pk_m, "ci_lo": pk_lo, "ci_hi": pk_hi}

    od_m, od_lo, od_hi = bootstrap_ci(all_d, seed=seed)
    metrics["overall_distractor_prob"] = {"mean": od_m, "ci_lo": od_lo, "ci_hi": od_hi}

    t1o = bootstrap_ci(all_t1, seed=seed)
    t5o = bootstrap_ci(all_t5, seed=seed)
    bo = bootstrap_ci(all_b, seed=seed)
    go = bootstrap_ci(all_g, seed=seed)
    ro = bootstrap_ci(all_ratios, seed=seed)

    metrics["overall_top1_accuracy"] = {"mean": t1o[0] * 100, "ci_lo": t1o[1] * 100, "ci_hi": t1o[2] * 100}
    metrics["overall_top5_accuracy"] = {"mean": t5o[0] * 100, "ci_lo": t5o[1] * 100, "ci_hi": t5o[2] * 100}
    metrics["overall_brier_score"] = {"mean": bo[0], "ci_lo": bo[1], "ci_hi": bo[2]}
    metrics["overall_goal_prob"] = {"mean": go[0], "ci_lo": go[1], "ci_hi": go[2]}
    metrics["overall_goal_dist_ratio"] = {"mean": ro[0], "ci_lo": ro[1], "ci_hi": ro[2]}

    # Uniform baseline reference values (1/N predictor)
    metrics["uniform_baseline"] = {
        "prob_per_poi": 1.0 / num_pois,
        "top1_accuracy": 100.0 / num_pois,
        "top5_accuracy": min(500.0 / num_pois, 100.0),
        "brier_score": (num_pois - 1) * (1.0 / num_pois) ** 2 + (1.0 - 1.0 / num_pois) ** 2,
        "goal_dist_ratio": 1.0,  # uniform assigns equal prob to goal and distractor
        "num_pois": num_pois,
    }
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ═════════════════════════════════════════════════════════════════════════════


def print_results(
    metrics: Dict, model_name: str, n_episodes: int, fractions: List[float], ref: str
):
    ref_label = "t* (closest approach)" if ref == "tstar" else "full trajectory"
    print(f"\n{'=' * 115}")
    print(f"EXPERIMENT 2 — {model_name}  (fractions relative to {ref_label})")
    print(f"{'=' * 115}")
    print(f"Episodes: {n_episodes}")
    print(f"{'-' * 115}")
    print(f"{'Frac':<6} {'Distractor P':<22} {'Goal P':<22} {'Top-1 Acc':<18} {'Top-5 Acc':<18} {'Brier':<15}")
    print(f"{'-' * 115}")

    for frac in fractions:
        if frac not in metrics["fractions"]:
            continue
        m = metrics["fractions"][frac]
        d, g = m["distractor_prob"], m["goal_prob"]
        t1, t5, b = m["top1_accuracy"], m["top5_accuracy"], m["brier_score"]
        print(
            f"{frac:<6.2f} "
            f"{d['mean']:.4f} [{d['ci_lo']:.4f},{d['ci_hi']:.4f}]  "
            f"{g['mean']:.4f} [{g['ci_lo']:.4f},{g['ci_hi']:.4f}]  "
            f"{t1['mean']:5.2f}% [{t1['ci_lo']:4.1f},{t1['ci_hi']:4.1f}]  "
            f"{t5['mean']:5.2f}% [{t5['ci_lo']:4.1f},{t5['ci_hi']:4.1f}]  "
            f"{b['mean']:.4f}"
        )

    print(f"{'-' * 115}")
    pk = metrics["peak_distractor_prob"]
    od = metrics["overall_distractor_prob"]
    ot1 = metrics["overall_top1_accuracy"]
    ot5 = metrics["overall_top5_accuracy"]
    ob = metrics["overall_brier_score"]
    ogr = metrics.get("overall_goal_dist_ratio", {})
    ub = metrics.get("uniform_baseline", {})
    print(f"Peak Distractor Prob:    {pk['mean']:.4f}  [{pk['ci_lo']:.4f}, {pk['ci_hi']:.4f}]")
    print(f"Overall Distractor Prob: {od['mean']:.4f}  [{od['ci_lo']:.4f}, {od['ci_hi']:.4f}]")
    print(f"Overall Top-1 Accuracy:  {ot1['mean']:.2f}%  [{ot1['ci_lo']:.2f}, {ot1['ci_hi']:.2f}]")
    print(f"Overall Top-5 Accuracy:  {ot5['mean']:.2f}%  [{ot5['ci_lo']:.2f}, {ot5['ci_hi']:.2f}]")
    print(f"Overall Brier Score:     {ob['mean']:.4f}  [{ob['ci_lo']:.4f}, {ob['ci_hi']:.4f}]")
    if ogr:
        print(f"Overall Goal/Dist Ratio: {ogr['mean']:.1f}x  [{ogr['ci_lo']:.1f}, {ogr['ci_hi']:.1f}]")
    if ub:
        print(f"{'-' * 115}")
        print(f"Uniform baseline (1/{ub['num_pois']}): Top-1={ub['top1_accuracy']:.2f}%  Top-5={ub['top5_accuracy']:.2f}%  Brier={ub['brier_score']:.4f}  Ratio=1.0")
    print(f"{'=' * 115}")


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE-MODEL RUNNER
# ═════════════════════════════════════════════════════════════════════════════


def run_single_model(
    model_path: str,
    model_type: str,
    model_name: str,
    episodes: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    node_to_idx: Dict[str, int],
    device: torch.device,
    fractions: List[float],
    output_dir: Path,
    batch_size: int = 32,
    seed: int = 42,
    num_agents: int = 100,
) -> Optional[Dict]:
    """Load model, evaluate on both reference modes, save results."""
    print(f"\n📥 Loading {model_type} model from {model_path} ...")
    num_nodes = len(graph.nodes())
    num_poi_nodes = len(poi_nodes)

    if model_type == "transformer":
        model = load_transformer_model(model_path, num_nodes, num_poi_nodes, device)
    elif model_type == "lstm":
        model = load_lstm_model(model_path, num_nodes, num_poi_nodes, device)
    elif model_type == "sc_bdi_vae":
        model = load_sc_bdi_vae_model(model_path, num_nodes, num_poi_nodes, num_agents, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    print(f"  ✅ {model_name} loaded")

    combined: Dict = {"model": model_name, "model_type": model_type}

    for ref, ref_label in [("tstar", "t*"), ("full", "full trajectory")]:
        print(f"\n  📊 Evaluating {model_name} — fractions relative to {ref_label} ...")

        if model_type == "transformer":
            raw = _evaluate_transformer(model, episodes, poi_nodes, node_to_idx, device, fractions, ref, batch_size)
        elif model_type == "lstm":
            raw = _evaluate_lstm(model, episodes, poi_nodes, node_to_idx, device, fractions, ref, batch_size)
        else:
            raw = _evaluate_sc_bdi_vae(model, episodes, graph, poi_nodes, device, fractions, ref, batch_size)

        metrics = compute_metrics(raw, fractions, seed, num_pois=len(poi_nodes))
        metrics["model"] = model_name
        metrics["model_type"] = model_type
        metrics["n_episodes"] = len(episodes)
        metrics["reference"] = ref

        print_results(metrics, model_name, len(episodes), fractions, ref)

        # File naming that exp_2_plot.py expects:
        #   exp2_<checkpoint_stem>_results.json        (t* reference)
        #   exp2_<checkpoint_stem>_full_traj_results.json  (full reference)
        ckpt_stem = Path(model_path).stem
        if ref == "tstar":
            out_file = output_dir / f"exp2_{ckpt_stem}_results.json"
        else:
            out_file = output_dir / f"exp2_{ckpt_stem}_full_traj_results.json"

        with out_file.open("w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"  💾 Saved → {out_file}")

        combined[f"metrics_relative_to_{ref}"] = metrics

    # Combined file
    comb_file = output_dir / f"exp2_{Path(model_path).stem}_combined_results.json"
    with comb_file.open("w") as fh:
        json.dump(combined, fh, indent=2)

    return combined


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: Preference-Proximity Dissociation Test")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["transformer", "lstm", "sc_bdi_vae"])
    parser.add_argument("--model_name", type=str, default=None, help="Display name")
    parser.add_argument("--run_all", action="store_true", help="Run all three models")
    parser.add_argument("--variant", type=str, default="both",
                        choices=["a", "b", "both"],
                        help="Which variant to run: a (same-category), b (cross-category), or both")
    parser.add_argument("--episodes_a", type=str,
                        default="experiments/data/exp_2a_test_set.json",
                        help="Path to same-category distractor episodes (variant a)")
    parser.add_argument("--episodes_b", type=str,
                        default="experiments/data/exp_2b_test_set.json",
                        help="Path to cross-category distractor episodes (variant b)")
    parser.add_argument("--graph_path", type=str,
                        default="data/processed/ucsd_walk_full.graphml")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/results",
                        help="Base directory for results (will create exp_2a/ and exp_2b/ subdirs)")
    parser.add_argument("--fractions", type=str, default="0.1,0.2,0.5,0.75,0.9")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.run_all and (args.model_path is None or args.model_type is None):
        parser.error("Either --run_all or both --model_path and --model_type are required")

    fractions = [float(f.strip()) for f in args.fractions.split(",")]
    base_output_dir = Path(args.output_dir)

    # Build list of (variant_label, episodes_path, output_dir) to run
    variant_configs: List[Tuple[str, str, Path]] = []
    if args.variant in ("a", "both"):
        variant_configs.append(("2a (same-category)", args.episodes_a, base_output_dir / "exp_2a"))
    if args.variant in ("b", "both"):
        variant_configs.append(("2b (cross-category)", args.episodes_b, base_output_dir / "exp_2b"))

    print("\n" + "=" * 100)
    print("EXPERIMENT 2: PREFERENCE-PROXIMITY DISSOCIATION TEST")
    print("=" * 100)

    device = get_device()
    print(f"🖥️  Device: {device}")

    # Load graph (shared across variants)
    print(f"\n📂 Loading graph from {args.graph_path} ...")
    graph = nx.read_graphml(args.graph_path)
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    print(f"   {len(graph.nodes()):,} nodes, {len(poi_nodes)} POIs")

    # Determine models
    if args.run_all:
        models_to_run = [
            {"path": "checkpoints/keepers/baseline_transformer_best_model.pt",
             "type": "transformer", "name": "Transformer"},
            {"path": "checkpoints/keepers/lstm_best_model.pt",
             "type": "lstm", "name": "LSTM"},
            {"path": "checkpoints/keepers/scbdi_no_progress.pt",
             "type": "sc_bdi_vae", "name": "SC-BDI-VAE"},
        ]
        print(f"\n🚀 Running all {len(models_to_run)} models ...")
    else:
        name = args.model_name or Path(args.model_path).stem
        models_to_run = [{"path": args.model_path, "type": args.model_type, "name": name}]

    num_agents = 100

    for variant_label, episodes_path, output_dir in variant_configs:
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"## VARIANT {variant_label}")
        print("#" * 100)

        # Load distractor episodes for this variant
        print(f"\n📂 Loading distractor episodes from {episodes_path} ...")
        try:
            episodes = load_filtered_episodes(Path(episodes_path))
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Cannot load episodes for {variant_label}: {e}")
            print(f"   Run the corresponding filtering script first:")
            if "2a" in variant_label:
                print(f"   python experiments/exp_2a_data_filtering.py")
            else:
                print(f"   python experiments/exp_2b_data_filtering.py")
            continue

        for cfg in models_to_run:
            print("\n" + "=" * 80)
            print(f"📊 Evaluating: {cfg['name']} [{variant_label}]")
            print("=" * 80)
            try:
                run_single_model(
                    model_path=cfg["path"],
                    model_type=cfg["type"],
                    model_name=cfg["name"],
                    episodes=episodes,
                    graph=graph,
                    poi_nodes=poi_nodes,
                    node_to_idx=node_to_idx,
                    device=device,
                    fractions=fractions,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    num_agents=num_agents,
                )
            except Exception as e:
                print(f"❌ Error running {cfg['name']}: {e}")
                import traceback
                traceback.print_exc()

        # ── generate plots for this variant ──
        print(f"\n📊 Generating visualizations for {variant_label} ...")
        try:
            from experiments.exp_2_plot import load_results, generate_plots_for_reference

            for ref_type, suffix in [("tstar", "_tstar"), ("full", "_full_traj")]:
                results = load_results(output_dir, ref_type)
                if results:
                    print(f"  Plotting {ref_type} reference ({len(results)} models) ...")
                    generate_plots_for_reference(results, output_dir, ref_type, suffix)
                else:
                    print(f"  ⚠️  No {ref_type} results found to plot")
        except Exception as e:
            print(f"  ⚠️  Plotting failed: {e}")
            print(f"  You can run plots manually: python experiments/exp_2_plot.py --results-dir {output_dir}")

    print(f"\n✅ Experiment 2 complete!")
    for _, _, od in variant_configs:
        print(f"   Results saved to: {od}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
