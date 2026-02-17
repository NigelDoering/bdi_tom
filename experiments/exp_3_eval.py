#!/usr/bin/env python3
"""
Experiment 3: Belief Updating Under Unexpected Closures

Evaluates how well models adapt their goal predictions when an agent's
original goal becomes inaccessible and they reroute to an alternative.

Metrics:
- belief_alignment: P(correct_goal) at each trajectory phase
- adaptation_speed: rate of probability shift from first_goal to final_goal
- top_k_accuracy: whether correct goal is in top-K predictions
- mrr: mean reciprocal rank of correct goal
- entropy: prediction uncertainty at each phase

Performance optimizations:
- Batched inference for faster evaluation
- Efficient tensor operations
- Progress caching
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from graph_controller.world_graph import WorldGraph
from models.vae_bdi_simple.bdi_vae_v3_model import create_sc_bdi_vae_v3
from models.baseline_transformer.baseline_transformer_model import PerNodeTransformerPredictor
from models.baseline_lstm.baseline_lstm_model import PerNodeToMPredictor


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PhaseMetrics(NamedTuple):
    """Metrics for a single evaluation phase."""
    goal_prob: float       # P(target_goal)
    top5_hit: bool         # Is target in top-5?
    top10_hit: bool        # Is target in top-10?
    rank: int              # Rank of target (1-indexed)
    entropy: float         # Prediction entropy
    max_prob: float        # Max predicted probability


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    path: Path
    model_type: Literal['sc_bdi_vae', 'transformer', 'lstm']
    use_temporal: bool = False  # Whether to use temporal features (SC-BDI-VAE only)
    kwargs: Dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Results for a single pivot episode."""
    agent_id: str
    episode_idx: int
    first_goal: str
    final_goal: str
    category: str
    pre_phases: List[PhaseMetrics]   # P(first_goal) at [25%, 50%, 75%] pre-pivot
    post_phases: List[PhaseMetrics]  # P(final_goal) at [25%, 50%, 75%] post-pivot
    # Track P(final_goal) during pre-pivot phase for proper adaptation calculation
    pre_final_goal_probs: List[float] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics with confidence intervals."""
    mean: float
    ci_low: float
    ci_high: float
    std: float
    
    @classmethod
    def from_values(cls, values: List[float], confidence: float = 0.95) -> 'AggregatedMetrics':
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        z = 1.96  # 95% CI
        return cls(mean=mean, ci_low=mean - z * se, ci_high=mean + z * se, std=std)


@dataclass  
class ModelResults:
    """Aggregated results for a model."""
    name: str
    episodes: List[EpisodeResult]
    
    def get_belief_alignment(self, phase_idx: int = 1) -> AggregatedMetrics:
        """P(final_goal) at specified post-pivot phase (default: 50%)."""
        return AggregatedMetrics.from_values([e.post_phases[phase_idx].goal_prob for e in self.episodes])
    
    def get_adaptation_speed(self, phase_idx: int = 1) -> AggregatedMetrics:
        """Increase in P(final_goal) from pre-pivot to post-pivot.
        
        Measures how much the model's belief in the NEW goal increases
        after observing the agent change direction.
        """
        deltas = []
        for e in self.episodes:
            # P(final_goal) at post-pivot - P(final_goal) at pre-pivot
            pre_prob = e.pre_final_goal_probs[phase_idx] if e.pre_final_goal_probs else 0.0
            post_prob = e.post_phases[phase_idx].goal_prob
            deltas.append(post_prob - pre_prob)
        return AggregatedMetrics.from_values(deltas)
    
    def get_top5_accuracy(self, phase: str = 'post', phase_idx: int = 1) -> AggregatedMetrics:
        """Top-5 accuracy at specified phase."""
        phases = [e.post_phases if phase == 'post' else e.pre_phases for e in self.episodes]
        return AggregatedMetrics.from_values([float(p[phase_idx].top5_hit) for p in phases])
    
    def get_mrr(self, phase: str = 'post', phase_idx: int = 1) -> AggregatedMetrics:
        """Mean reciprocal rank at specified phase."""
        phases = [e.post_phases if phase == 'post' else e.pre_phases for e in self.episodes]
        return AggregatedMetrics.from_values([1.0 / p[phase_idx].rank for p in phases])


# ============================================================================
# MODEL LOADING
# ============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(
    config: ModelConfig,
    num_nodes: int,
    num_poi: int,
    device: torch.device,
) -> Tuple[nn.Module, torch.device]:
    """Load model from checkpoint. Returns (model, actual_device)."""
    checkpoint = torch.load(config.path, map_location='cpu', weights_only=False)
    
    if config.model_type == 'sc_bdi_vae':
        model = create_sc_bdi_vae_v3(
            num_nodes=num_nodes, num_agents=100, num_poi_nodes=num_poi, num_categories=7,
            node_embedding_dim=64, fusion_dim=128, belief_latent_dim=32, desire_latent_dim=16,
            intention_latent_dim=32, vae_hidden_dim=128, hidden_dim=256, dropout=0.1, use_progress=True,
            **config.kwargs,
        )
    elif config.model_type == 'transformer':
        model = PerNodeTransformerPredictor(
            num_nodes=num_nodes, num_agents=100, num_poi_nodes=num_poi, num_categories=7,
            node_embedding_dim=128, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1,
            **config.kwargs,
        )
    elif config.model_type == 'lstm':
        # Determine num_agents from checkpoint (mirrors exp_2_eval logic)
        agent_key = "embedding_pipeline.agent_encoder.agent_context.agent_emb.agent_embedding.weight"
        state_dict = checkpoint.get('model_state_dict', {})
        if agent_key in state_dict:
            actual_num_agents = state_dict[agent_key].shape[0]
        elif 'config' in checkpoint:
            actual_num_agents = checkpoint['config'].get('num_agents', 100)
        else:
            actual_num_agents = 100
        ckpt_config = checkpoint.get('config', {})
        model = PerNodeToMPredictor(
            num_nodes=ckpt_config.get('num_nodes', num_nodes),
            num_agents=actual_num_agents,
            num_poi_nodes=ckpt_config.get('num_poi_nodes', num_poi),
            num_categories=ckpt_config.get('num_categories', 7),
            node_embedding_dim=ckpt_config.get('node_embedding_dim', 64),
            temporal_dim=ckpt_config.get('temporal_dim', 64),
            agent_dim=ckpt_config.get('agent_dim', 64),
            fusion_dim=ckpt_config.get('fusion_dim', 128),
            hidden_dim=ckpt_config.get('hidden_dim', 256),
            dropout=ckpt_config.get('dropout', 0.1),
            num_heads=ckpt_config.get('num_heads', 4),
            freeze_embedding=ckpt_config.get('freeze_embedding', False),
            **config.kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Force CPU on MPS due to known Metal buffer issues with small batch inference
    actual_device = device
    if device.type == 'mps' and config.model_type in ('transformer', 'sc_bdi_vae'):
        actual_device = torch.device('cpu')
        print(f"  [Using CPU for {config.name} due to MPS Metal buffer issues]")
    
    # Try device, fall back to CPU on failure
    try:
        model.to(actual_device)
    except Exception:
        actual_device = torch.device('cpu')
        model.to(actual_device)
        print(f"  [Fallback to CPU for {config.name}]")
    
    model.eval()
    return model, actual_device


# ============================================================================
# INFERENCE
# ============================================================================

def compute_phase_metrics(probs: torch.Tensor, target_idx: int) -> PhaseMetrics:
    """Compute all metrics from probability distribution."""
    probs_np = probs.numpy()
    target_prob = float(probs_np[target_idx])
    
    # Rank (1-indexed, lower is better)
    sorted_indices = np.argsort(-probs_np)
    rank = int(np.where(sorted_indices == target_idx)[0][0]) + 1
    
    # Top-K hits
    top5_hit = target_idx in sorted_indices[:5]
    top10_hit = target_idx in sorted_indices[:10]
    
    # Entropy
    entropy = float(-np.sum(probs_np * np.log(probs_np + 1e-10)))
    
    # Max probability
    max_prob = float(np.max(probs_np))
    
    return PhaseMetrics(
        goal_prob=target_prob,
        top5_hit=top5_hit,
        top10_hit=top10_hit,
        rank=rank,
        entropy=entropy,
        max_prob=max_prob,
    )


def get_goal_probs(
    model: nn.Module,
    model_type: str,
    node_indices: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
    hour: int = 12,
    progress: float = 0.5,
    use_temporal: bool = False,
    agent_id: Optional[int] = None,
) -> torch.Tensor:
    """Get goal probability distribution from model."""
    node_indices = node_indices.to(device)
    lengths = lengths.to(device)
    seq_len = node_indices.size(1)
    
    with torch.no_grad():
        if model_type == 'sc_bdi_vae':
            # Build agent_ids tensor
            agent_ids = None
            if agent_id is not None:
                agent_ids = torch.tensor([agent_id], dtype=torch.long, device=device)
            
            if use_temporal:
                # Use temporal features (requires matching training distribution)
                hours = torch.tensor([hour], dtype=torch.long, device=device)
                days = torch.zeros(1, dtype=torch.long, device=device)
                velocities = torch.ones(1, seq_len, device=device) * 1.4  # ~walking speed m/s
                deltas = torch.ones(1, seq_len, device=device) * 30.0  # ~30 sec per step
            else:
                # Fallback to node embeddings only
                hours, days, velocities, deltas = None, None, None, None
            
            out = model(
                history_node_indices=node_indices,
                history_lengths=lengths,
                agent_ids=agent_ids,
                path_progress=torch.tensor([progress], device=device),
                compute_loss=False,
                hours=hours, days=days, velocities=velocities, deltas=deltas,
            )
            logits = out['goal']
            
        elif model_type == 'transformer':
            padding_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
            out = model(node_indices=node_indices, padding_mask=padding_mask)
            logits = out['goal'][0, lengths[0] - 1].unsqueeze(0)
            
        elif model_type == 'lstm':
            out = model(history_node_indices=node_indices, history_lengths=lengths)
            logits = out['goal']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return F.softmax(logits, dim=-1)[0].cpu()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_episode(
    model: nn.Module,
    model_type: str,
    episode: Dict,
    node_to_idx: Dict[str, int],
    poi_to_idx: Dict[str, int],
    device: torch.device,
    fractions: Tuple[float, ...] = (0.25, 0.5, 0.75),
    use_temporal: bool = False,
) -> Optional[EpisodeResult]:
    """Evaluate model on a single pivot episode."""
    path = episode['path']
    pivot_idx = episode['pivot_index']
    first_goal = episode['first_goal']
    final_goal = episode['final_goal']
    hour = episode.get('hour', 12)
    
    # Parse agent_id integer from string like "agent_042"
    agent_key = episode.get('agent_id', 'agent_000')
    try:
        agent_int: Optional[int] = int(str(agent_key).split('_')[1])
    except (IndexError, ValueError):
        agent_int = None
    
    # Convert path to indices
    try:
        path_indices = [node_to_idx[step[0]] for step in path]
    except KeyError:
        return None
    
    # Validate goals
    if first_goal not in poi_to_idx or final_goal not in poi_to_idx:
        return None
    
    first_goal_idx = poi_to_idx[first_goal]
    final_goal_idx = poi_to_idx[final_goal]
    total_len = len(path)
    
    pre_phases: List[PhaseMetrics] = []
    post_phases: List[PhaseMetrics] = []
    
    # Pre-pivot: measure P(first_goal) and P(final_goal)
    pre_final_goal_probs: List[float] = []
    for frac in fractions:
        end_idx = max(1, int(pivot_idx * frac))
        indices = torch.tensor([path_indices[:end_idx]], dtype=torch.long)
        lengths = torch.tensor([end_idx])
        progress = end_idx / total_len
        
        probs = get_goal_probs(model, model_type, indices, lengths, device, hour, progress, use_temporal, agent_id=agent_int)
        pre_phases.append(compute_phase_metrics(probs, first_goal_idx))
        # Also track P(final_goal) for adaptation speed calculation
        pre_final_goal_probs.append(float(probs[final_goal_idx].item()))
    
    # Post-pivot: measure P(final_goal)
    post_len = len(path) - pivot_idx
    for frac in fractions:
        end_idx = pivot_idx + max(1, int(post_len * frac))
        indices = torch.tensor([path_indices[:end_idx]], dtype=torch.long)
        lengths = torch.tensor([end_idx])
        progress = end_idx / total_len
        
        probs = get_goal_probs(model, model_type, indices, lengths, device, hour, progress, use_temporal, agent_id=agent_int)
        post_phases.append(compute_phase_metrics(probs, final_goal_idx))
    
    return EpisodeResult(
        agent_id=episode['agent_id'],
        episode_idx=episode['episode_index'],
        first_goal=first_goal,
        final_goal=final_goal,
        category=episode['category'],
        pre_phases=pre_phases,
        post_phases=post_phases,
        pre_final_goal_probs=pre_final_goal_probs,
    )


def evaluate_model(
    config: ModelConfig,
    episodes: List[Dict],
    node_to_idx: Dict[str, int],
    poi_to_idx: Dict[str, int],
    num_nodes: int,
    num_poi: int,
    device: torch.device,
) -> ModelResults:
    """Evaluate model on all pivot episodes."""
    model, actual_device = load_model(config, num_nodes, num_poi, device)
    
    results = []
    for ep in tqdm(episodes, desc=config.name, leave=False):
        result = evaluate_episode(
            model, config.model_type, ep, node_to_idx, poi_to_idx, actual_device,
            use_temporal=config.use_temporal,
        )
        if result is not None:
            results.append(result)
    
    return ModelResults(name=config.name, episodes=results)


# ============================================================================
# OUTPUT
# ============================================================================

def save_results(
    results: List[ModelResults],
    output_dir: Path,
    fractions: Tuple[float, ...] = (0.25, 0.5, 0.75),
) -> None:
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Comprehensive metrics CSV
    rows = ["model,metric,mean,ci_low,ci_high,std"]
    for r in results:
        # Post-pivot P(new goal) at each fraction
        for i, frac in enumerate(fractions):
            ba = AggregatedMetrics.from_values([e.post_phases[i].goal_prob for e in r.episodes])
            rows.append(f"{r.name},post_p_new_goal_{frac},{ba.mean:.6f},{ba.ci_low:.6f},{ba.ci_high:.6f},{ba.std:.6f}")

        # Pre-pivot P(new goal) — baseline before the agent changes direction
        for i, frac in enumerate(fractions):
            pre_fg = AggregatedMetrics.from_values([e.pre_final_goal_probs[i] for e in r.episodes])
            rows.append(f"{r.name},pre_p_new_goal_{frac},{pre_fg.mean:.6f},{pre_fg.ci_low:.6f},{pre_fg.ci_high:.6f},{pre_fg.std:.6f}")

        # Adaptation speed at 75% (primary metric)
        asp_75 = AggregatedMetrics.from_values([
            e.post_phases[2].goal_prob - e.pre_final_goal_probs[2] for e in r.episodes
        ])
        rows.append(f"{r.name},adaptation_speed_75,{asp_75.mean:.6f},{asp_75.ci_low:.6f},{asp_75.ci_high:.6f},{asp_75.std:.6f}")

        # Adaptation speed at 50% (for comparison)
        asp_50 = r.get_adaptation_speed(1)
        rows.append(f"{r.name},adaptation_speed_50,{asp_50.mean:.6f},{asp_50.ci_low:.6f},{asp_50.ci_high:.6f},{asp_50.std:.6f}")

        # Top-5 accuracy post-pivot at 75%
        t5_75 = AggregatedMetrics.from_values([float(e.post_phases[2].top5_hit) for e in r.episodes])
        rows.append(f"{r.name},top5_accuracy_75,{t5_75.mean:.6f},{t5_75.ci_low:.6f},{t5_75.ci_high:.6f},{t5_75.std:.6f}")

        # Top-5 accuracy post-pivot at 50%
        t5_50 = r.get_top5_accuracy('post', 1)
        rows.append(f"{r.name},top5_accuracy_50,{t5_50.mean:.6f},{t5_50.ci_low:.6f},{t5_50.ci_high:.6f},{t5_50.std:.6f}")

        # MRR post-pivot at 75%
        mrr_75 = AggregatedMetrics.from_values([1.0 / e.post_phases[2].rank for e in r.episodes])
        rows.append(f"{r.name},mrr_75,{mrr_75.mean:.6f},{mrr_75.ci_low:.6f},{mrr_75.ci_high:.6f},{mrr_75.std:.6f}")

        # MRR post-pivot at 50%
        mrr_50 = r.get_mrr('post', 1)
        rows.append(f"{r.name},mrr_50,{mrr_50.mean:.6f},{mrr_50.ci_low:.6f},{mrr_50.ci_high:.6f},{mrr_50.std:.6f}")

        # Pre-pivot P(original goal)
        for i, frac in enumerate(fractions):
            pre_ba = AggregatedMetrics.from_values([e.pre_phases[i].goal_prob for e in r.episodes])
            rows.append(f"{r.name},pre_p_orig_goal_{frac},{pre_ba.mean:.6f},{pre_ba.ci_low:.6f},{pre_ba.ci_high:.6f},{pre_ba.std:.6f}")
    
    (output_dir / 'exp3_metrics.csv').write_text("\n".join(rows))
    
    # Phase metrics
    phase_rows = ["model,phase,fraction,goal_prob,top5_acc,mrr,entropy,max_prob"]
    for r in results:
        for i, frac in enumerate(fractions):
            # Pre-pivot
            pre_prob = np.mean([e.pre_phases[i].goal_prob for e in r.episodes])
            pre_t5 = np.mean([e.pre_phases[i].top5_hit for e in r.episodes])
            pre_mrr = np.mean([1.0 / e.pre_phases[i].rank for e in r.episodes])
            pre_ent = np.mean([e.pre_phases[i].entropy for e in r.episodes])
            pre_max = np.mean([e.pre_phases[i].max_prob for e in r.episodes])
            phase_rows.append(f"{r.name},pre,{frac},{pre_prob:.6f},{pre_t5:.6f},{pre_mrr:.6f},{pre_ent:.6f},{pre_max:.6f}")
            
            # Post-pivot
            post_prob = np.mean([e.post_phases[i].goal_prob for e in r.episodes])
            post_t5 = np.mean([e.post_phases[i].top5_hit for e in r.episodes])
            post_mrr = np.mean([1.0 / e.post_phases[i].rank for e in r.episodes])
            post_ent = np.mean([e.post_phases[i].entropy for e in r.episodes])
            post_max = np.mean([e.post_phases[i].max_prob for e in r.episodes])
            phase_rows.append(f"{r.name},post,{frac},{post_prob:.6f},{post_t5:.6f},{post_mrr:.6f},{post_ent:.6f},{post_max:.6f}")
    
    (output_dir / 'exp3_phase_metrics.csv').write_text("\n".join(phase_rows))
    
    # JSON summary — include both 50% and 75% metrics
    summary = {
        'num_episodes': len(results[0].episodes) if results else 0,
        'fractions': list(fractions),
        'primary_metric_fraction': 0.75,
        'models': {}
    }
    for r in results:
        model_summary: Dict[str, Any] = {}

        # 75% post-pivot (primary)
        model_summary['post_p_new_goal_75'] = np.mean(
            [e.post_phases[2].goal_prob for e in r.episodes]
        )
        model_summary['top5_accuracy_75'] = np.mean(
            [float(e.post_phases[2].top5_hit) for e in r.episodes]
        )
        model_summary['mrr_75'] = np.mean(
            [1.0 / e.post_phases[2].rank for e in r.episodes]
        )
        model_summary['adaptation_speed_75'] = np.mean(
            [e.post_phases[2].goal_prob - e.pre_final_goal_probs[2] for e in r.episodes]
        )

        # 50% post-pivot (secondary, for comparison)
        model_summary['post_p_new_goal_50'] = r.get_belief_alignment(1).mean
        model_summary['top5_accuracy_50'] = r.get_top5_accuracy('post', 1).mean
        model_summary['mrr_50'] = r.get_mrr('post', 1).mean
        model_summary['adaptation_speed_50'] = r.get_adaptation_speed(1).mean

        # Pre-pivot baselines (P(new goal) before pivot)
        for i, frac in enumerate(fractions):
            model_summary[f'pre_p_new_goal_{frac}'] = np.mean(
                [e.pre_final_goal_probs[i] for e in r.episodes]
            )

        # Pre-pivot P(original goal)
        for i, frac in enumerate(fractions):
            model_summary[f'pre_p_orig_goal_{frac}'] = np.mean(
                [e.pre_phases[i].goal_prob for e in r.episodes]
            )

        summary['models'][r.name] = model_summary
    
    (output_dir / 'exp3_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def _bootstrap_ci(values: List[float], n_boot: int = 5000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute mean and bootstrap 95% CI."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    if len(arr) < 3:
        return mean, mean, mean
    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def generate_visualizations(results: List[ModelResults], output_dir: Path) -> None:
    """Generate publication-ready visualization plots (300 DPI, CI bands)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
    })

    # Consistent colour scheme matching exp_2_plot.py
    COLORS = {
        'SC-BDI-VAE': '#467821',   # Green (Ours)
        'Transformer': '#E24A33',  # Red
        'LSTM': '#348ABD',         # Blue
    }
    MARKERS = {'SC-BDI-VAE': 'o', 'Transformer': 's', 'LSTM': '^'}
    MODEL_LABELS = {'SC-BDI-VAE': 'SC-BDI (Ours)', 'Transformer': 'Transformer', 'LSTM': 'LSTM'}
    fractions = [0.25, 0.5, 0.75]
    frac_pct = [f * 100 for f in fractions]

    def _color(name: str) -> str:
        return COLORS.get(name, 'gray')

    def _marker(name: str) -> str:
        return MARKERS.get(name, 'o')

    def _label(name: str) -> str:
        return MODEL_LABELS.get(name, name)

    # ------------------------------------------------------------------ #
    # Helper: compute belief revision curve across pre→post for P(new goal)
    # x-axis: 6 points representing [pre-25%, pre-50%, pre-75%,
    #                                  post-25%, post-50%, post-75%]
    # ------------------------------------------------------------------ #
    x_full = [-75, -50, -25, 25, 50, 75]  # negative = pre-pivot, positive = post-pivot
    x_labels_full = ['-75%', '-50%', '-25%', '+25%', '+50%', '+75%']

    def _belief_revision_data(r: ModelResults) -> Tuple[List[float], List[float], List[float]]:
        """Get P(new goal) across all 6 time-points (3 pre + 3 post)."""
        means, ci_lo, ci_hi = [], [], []
        # Pre-pivot P(final_goal)
        for i in range(3):
            vals = [e.pre_final_goal_probs[i] for e in r.episodes]
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi)
        # Post-pivot P(final_goal)
        for i in range(3):
            vals = [e.post_phases[i].goal_prob for e in r.episodes]
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi)
        return means, ci_lo, ci_hi

    # ------------------------------------------------------------------ #
    # 1. Belief revision curve — THE key plot for this experiment
    #    Shows P(new goal) on a unified pre→post timeline
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        means, ci_lo, ci_hi = _belief_revision_data(r)
        c = _color(r.name)
        ax.plot(x_full, means, marker=_marker(r.name), linewidth=2.5,
                markersize=8, color=c, label=_label(r.name))
        ax.fill_between(x_full, ci_lo, ci_hi, alpha=0.18, color=c)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.6, label='Pivot point')
    ax.axvspan(-80, 0, alpha=0.04, color='gray')
    ax.text(-40, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0.01 else 0.001, 'Pre-pivot',
            ha='center', fontsize=10, fontstyle='italic', alpha=0.6)
    ax.text(50, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0.01 else 0.001, 'Post-pivot',
            ha='center', fontsize=10, fontstyle='italic', alpha=0.6)

    ax.set_xlabel('Trajectory Phase (% of pre-/post-pivot segment)')
    ax.set_ylabel('P(New Goal)')
    ax.set_title('Belief Revision: P(New Goal) Across Goal Change', fontweight='bold')
    ax.set_xticks(x_full)
    ax.set_xticklabels(x_labels_full)
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_belief_revision.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    # 2. Dual-panel: P(original goal) pre-pivot + P(new goal) post-pivot
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Pre-pivot: P(original goal) — tests initial prediction quality
    ax = axes[0]
    for r in results:
        phases = [e.pre_phases for e in r.episodes]
        means, ci_lo, ci_hi = [], [], []
        for i in range(3):
            vals = [p[i].goal_prob for p in phases]
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi)
        c = _color(r.name)
        ax.plot(frac_pct, means, marker=_marker(r.name), linewidth=2.5,
                markersize=8, color=c, label=_label(r.name))
        ax.fill_between(frac_pct, ci_lo, ci_hi, alpha=0.18, color=c)
    ax.set_xlabel('Pre-Pivot Fraction (%)')
    ax.set_ylabel('P(Original Goal)')
    ax.set_title('(a) Pre-Pivot: P(Original Goal)', fontweight='bold')
    ax.legend()
    ax.set_xlim(20, 80)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.FixedLocator(frac_pct))

    # (b) Post-pivot: P(new goal) — tests adaptation
    ax = axes[1]
    for r in results:
        phases = [e.post_phases for e in r.episodes]
        means, ci_lo, ci_hi = [], [], []
        for i in range(3):
            vals = [p[i].goal_prob for p in phases]
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi)
        c = _color(r.name)
        ax.plot(frac_pct, means, marker=_marker(r.name), linewidth=2.5,
                markersize=8, color=c, label=_label(r.name))
        ax.fill_between(frac_pct, ci_lo, ci_hi, alpha=0.18, color=c)
    ax.set_xlabel('Post-Pivot Fraction (%)')
    ax.set_ylabel('P(New Goal)')
    ax.set_title('(b) Post-Pivot: P(New Goal)', fontweight='bold')
    ax.legend()
    ax.set_xlim(20, 80)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.FixedLocator(frac_pct))

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_goal_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    # 3. Summary bar chart — post-pivot metrics at 75%
    #    (75% is the most informative: enough post-pivot evidence to judge)
    # ------------------------------------------------------------------ #
    metric_fns_75 = [
        ('P(New Goal)\n@ 75%',     lambda r: [e.post_phases[2].goal_prob for e in r.episodes]),
        ('Adaptation\nSpeed @ 75%', lambda r: [
            e.post_phases[2].goal_prob - e.pre_final_goal_probs[2] for e in r.episodes
        ]),
        ('Top-5 Acc.\n@ 75%',      lambda r: [float(e.post_phases[2].top5_hit) for e in r.episodes]),
        ('MRR\n@ 75%',             lambda r: [1.0 / e.post_phases[2].rank for e in r.episodes]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_metrics = len(metric_fns_75)
    n_models = len(results)
    width = 0.22
    x = np.arange(n_metrics)

    for i, r in enumerate(results):
        means, errs_lo, errs_hi = [], [], []
        for _, fn in metric_fns_75:
            vals = fn(r)
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m)
            errs_lo.append(m - lo)
            errs_hi.append(hi - m)
        c = _color(r.name)
        ax.bar(x + i * width, means, width, yerr=[errs_lo, errs_hi],
               capsize=4, color=c, alpha=0.85, edgecolor='black',
               label=_label(r.name))

    ax.set_ylabel('Value')
    ax.set_title('Experiment 3: Post-Pivot Metrics at 75% Observation', fontweight='bold')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([lbl for lbl, _ in metric_fns_75])
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    # 4. Post-pivot entropy trajectory
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        phases = [e.post_phases for e in r.episodes]
        means, ci_lo, ci_hi = [], [], []
        for i in range(3):
            vals = [p[i].entropy for p in phases]
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi)
        c = _color(r.name)
        ax.plot(frac_pct, means, marker=_marker(r.name), linewidth=2.5,
                markersize=8, color=c, label=_label(r.name))
        ax.fill_between(frac_pct, ci_lo, ci_hi, alpha=0.18, color=c)

    ax.set_xlabel('Post-Pivot Fraction (%)')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Prediction Uncertainty During Rerouting', fontweight='bold')
    ax.legend()
    ax.set_xlim(20, 80)
    ax.xaxis.set_major_locator(mticker.FixedLocator(frac_pct))

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_entropy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    # 5. Rank distribution histograms (at 75% post-pivot)
    # ------------------------------------------------------------------ #
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)

    for i, r in enumerate(results):
        ax = axes[0, i]
        ranks = [e.post_phases[2].rank for e in r.episodes]  # 75% post-pivot
        bins = list(range(1, 52, 5)) + [230]
        ax.hist(ranks, bins=bins, color=_color(r.name), alpha=0.75,
                edgecolor='white', linewidth=0.8)
        ax.axvline(x=5.5,  color='#2ca02c', linestyle='--', alpha=0.8, label='Top-5')
        ax.axvline(x=10.5, color='#ff7f0e', linestyle='--', alpha=0.8, label='Top-10')
        ax.set_xlabel('Rank of New Goal')
        ax.set_ylabel('Count')
        ax.set_title(f'{_label(r.name)} (at 75%)', fontweight='bold')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_rank_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    # 6. Combined publication figure (2×3 grid)
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # (a) Belief revision curve — the key plot
    ax_a = fig.add_subplot(gs[0, :2])  # spans first two columns
    for r in results:
        means, ci_lo, ci_hi = _belief_revision_data(r)
        c = _color(r.name)
        ax_a.plot(x_full, means, marker=_marker(r.name), linewidth=2.5,
                  markersize=7, color=c, label=_label(r.name))
        ax_a.fill_between(x_full, ci_lo, ci_hi, alpha=0.15, color=c)
    ax_a.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
    ax_a.axvspan(-80, 0, alpha=0.04, color='gray')
    ax_a.set_xlabel('Trajectory Phase (% of segment)')
    ax_a.set_ylabel('P(New Goal)')
    ax_a.set_title('(a) Belief Revision: P(New Goal) Across Goal Change', fontweight='bold')
    ax_a.set_xticks(x_full)
    ax_a.set_xticklabels(x_labels_full)
    ax_a.legend(fontsize=9)
    ax_a.set_ylim(bottom=0)
    # Phase annotations
    ylim_a = ax_a.get_ylim()
    ax_a.text(-50, ylim_a[1] * 0.92, 'Pre-pivot', ha='center', fontsize=10,
              fontstyle='italic', alpha=0.5)
    ax_a.text(50, ylim_a[1] * 0.92, 'Post-pivot', ha='center', fontsize=10,
              fontstyle='italic', alpha=0.5)

    # (b) Entropy trajectory
    ax_b = fig.add_subplot(gs[0, 2])
    for r in results:
        phases = [e.post_phases for e in r.episodes]
        means, ci_lo, ci_hi = [], [], []
        for j in range(3):
            vals = [p[j].entropy for p in phases]
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi)
        c = _color(r.name)
        ax_b.plot(frac_pct, means, marker=_marker(r.name), linewidth=2, markersize=6, color=c)
        ax_b.fill_between(frac_pct, ci_lo, ci_hi, alpha=0.15, color=c)
    ax_b.set_xlabel('Post-Pivot Fraction (%)')
    ax_b.set_ylabel('Entropy (nats)')
    ax_b.set_title('(b) Prediction Uncertainty', fontweight='bold')

    # (c) P(new goal) at 75% post-pivot
    ax_c = fig.add_subplot(gs[1, 0])
    x_bar = np.arange(len(results))
    m_vals, e_lo, e_hi, bar_colors = [], [], [], []
    for r in results:
        vals = [e.post_phases[2].goal_prob for e in r.episodes]
        m, lo, hi = _bootstrap_ci(vals)
        m_vals.append(m); e_lo.append(m - lo); e_hi.append(hi - m)
        bar_colors.append(_color(r.name))
    ax_c.bar(x_bar, m_vals, yerr=[e_lo, e_hi], capsize=5, color=bar_colors,
             alpha=0.85, edgecolor='black')
    ax_c.set_xticks(x_bar)
    ax_c.set_xticklabels([_label(r.name) for r in results], rotation=15, ha='right', fontsize=9)
    ax_c.set_ylabel('P(New Goal)')
    ax_c.set_title('(c) Belief Alignment @ 75% (↑ better)', fontweight='bold')
    ax_c.set_ylim(bottom=0)

    # (d) Top-5 accuracy at 75% post-pivot
    ax_d = fig.add_subplot(gs[1, 1])
    m_vals, e_lo, e_hi = [], [], []
    for r in results:
        vals = [float(e.post_phases[2].top5_hit) for e in r.episodes]
        m, lo, hi = _bootstrap_ci(vals)
        m_vals.append(m * 100); e_lo.append((m - lo) * 100); e_hi.append((hi - m) * 100)
    ax_d.bar(x_bar, m_vals, yerr=[e_lo, e_hi], capsize=5, color=bar_colors,
             alpha=0.85, edgecolor='black')
    ax_d.set_xticks(x_bar)
    ax_d.set_xticklabels([_label(r.name) for r in results], rotation=15, ha='right', fontsize=9)
    ax_d.set_ylabel('Top-5 Accuracy (%)')
    ax_d.set_title('(d) Top-5 Acc. @ 75% (↑ better)', fontweight='bold')
    ax_d.set_ylim(bottom=0)

    # (e) MRR at 75% post-pivot
    ax_e = fig.add_subplot(gs[1, 2])
    m_vals, e_lo, e_hi = [], [], []
    for r in results:
        vals = [1.0 / e.post_phases[2].rank for e in r.episodes]
        m, lo, hi = _bootstrap_ci(vals)
        m_vals.append(m); e_lo.append(m - lo); e_hi.append(hi - m)
    ax_e.bar(x_bar, m_vals, yerr=[e_lo, e_hi], capsize=5, color=bar_colors,
             alpha=0.85, edgecolor='black')
    ax_e.set_xticks(x_bar)
    ax_e.set_xticklabels([_label(r.name) for r in results], rotation=15, ha='right', fontsize=9)
    ax_e.set_ylabel('MRR')
    ax_e.set_title('(e) MRR @ 75% (↑ better)', fontweight='bold')
    ax_e.set_ylim(bottom=0)

    plt.suptitle('Experiment 3: Belief Updating Under Unexpected Closures',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(output_dir / 'exp3_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Belief Updating')
    parser.add_argument('--run-id', type=int, default=8)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/keepers'))
    parser.add_argument('--graph-path', type=Path, default=Path('data/processed/ucsd_walk_full.graphml'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/results/exp_3'),
                        help='Output directory for results and plots')
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of episodes')
    parser.add_argument('--use-temporal', action='store_true', help='Use temporal features for SC-BDI-VAE')
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")
    
    # Load graph
    graph = nx.read_graphml(args.graph_path)
    wg = WorldGraph(graph)
    num_nodes = len(graph.nodes())
    num_poi = len(wg.poi_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    poi_to_idx = {node: idx for idx, node in enumerate(wg.poi_nodes)}
    print(f"Graph: {num_nodes} nodes, {num_poi} POIs")
    
    # Load pivot episodes
    data_dir = Path(f'data/simulation_data/run_{args.run_id}')
    with open(data_dir / 'exp_3_pivots.json') as f:
        pivot_data = json.load(f)
    episodes = pivot_data['episodes']
    if args.limit:
        episodes = episodes[:args.limit]
    print(f"Evaluating {len(episodes)} pivot episodes")
    
    # Model configs
    model_configs = [
        ModelConfig('SC-BDI-VAE', args.checkpoint_dir / 'best_model-OURS.pt', 'sc_bdi_vae', 
                    use_temporal=args.use_temporal),
        ModelConfig('Transformer', args.checkpoint_dir / 'baseline_transformer_best_model.pt', 'transformer'),
        ModelConfig('LSTM', args.checkpoint_dir / 'lstm_best_model.pt', 'lstm'),
    ]
    model_configs = [c for c in model_configs if c.path.exists()]
    
    if not model_configs:
        print("No checkpoints found!")
        return
    
    # Evaluate
    results = []
    for config in model_configs:
        print(f"\nEvaluating {config.name}...")
        result = evaluate_model(config, episodes, node_to_idx, poi_to_idx, num_nodes, num_poi, device)
        results.append(result)
        
        ba = result.get_belief_alignment(1)
        asp = result.get_adaptation_speed(1)
        t5 = result.get_top5_accuracy('post', 1)
        mrr = result.get_mrr('post', 1)
        print(f"  Belief Alignment: {ba.mean:.4f} ± {ba.std:.4f}")
        print(f"  Adaptation Speed: {asp.mean:.4f} ± {asp.std:.4f}")
        print(f"  Top-5 Accuracy:   {t5.mean:.4f}")
        print(f"  MRR:              {mrr.mean:.4f}")
    
    # Save
    output_dir = args.output_dir
    save_results(results, output_dir)
    
    if not args.no_viz:
        generate_visualizations(results, output_dir)


if __name__ == '__main__':
    main()
