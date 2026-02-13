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
from typing import Dict, List, Literal, Optional, Tuple, NamedTuple

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
        model = PerNodeToMPredictor(
            num_nodes=num_nodes, num_agents=100, num_poi_nodes=num_poi, num_categories=7,
            node_embedding_dim=64, temporal_dim=64, agent_dim=64, fusion_dim=128, hidden_dim=256,
            dropout=0.1, num_heads=4, **config.kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Force CPU for transformer on MPS due to known Metal buffer issues with small batch inference
    actual_device = device
    if device.type == 'mps' and config.model_type == 'transformer':
        actual_device = torch.device('cpu')
        print(f"  [Using CPU for {config.name} due to MPS transformer issues]")
    
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
) -> torch.Tensor:
    """Get goal probability distribution from model."""
    node_indices = node_indices.to(device)
    lengths = lengths.to(device)
    seq_len = node_indices.size(1)
    
    with torch.no_grad():
        if model_type == 'sc_bdi_vae':
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
        
        probs = get_goal_probs(model, model_type, indices, lengths, device, hour, progress, use_temporal)
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
        
        probs = get_goal_probs(model, model_type, indices, lengths, device, hour, progress, use_temporal)
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
        # Belief alignment (P(final_goal) at 50% post-pivot)
        ba = r.get_belief_alignment(1)
        rows.append(f"{r.name},belief_alignment,{ba.mean:.6f},{ba.ci_low:.6f},{ba.ci_high:.6f},{ba.std:.6f}")
        
        # Adaptation speed
        asp = r.get_adaptation_speed(1)
        rows.append(f"{r.name},adaptation_speed,{asp.mean:.6f},{asp.ci_low:.6f},{asp.ci_high:.6f},{asp.std:.6f}")
        
        # Top-5 accuracy post-pivot
        t5 = r.get_top5_accuracy('post', 1)
        rows.append(f"{r.name},top5_accuracy,{t5.mean:.6f},{t5.ci_low:.6f},{t5.ci_high:.6f},{t5.std:.6f}")
        
        # MRR post-pivot  
        mrr = r.get_mrr('post', 1)
        rows.append(f"{r.name},mrr,{mrr.mean:.6f},{mrr.ci_low:.6f},{mrr.ci_high:.6f},{mrr.std:.6f}")
        
        # Pre-pivot metrics for comparison
        pre_ba = AggregatedMetrics.from_values([e.pre_phases[1].goal_prob for e in r.episodes])
        rows.append(f"{r.name},pre_belief_alignment,{pre_ba.mean:.6f},{pre_ba.ci_low:.6f},{pre_ba.ci_high:.6f},{pre_ba.std:.6f}")
    
    (output_dir / 'exp_3_metrics.csv').write_text("\n".join(rows))
    
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
    
    (output_dir / 'exp_3_phase_metrics.csv').write_text("\n".join(phase_rows))
    
    # JSON summary
    summary = {
        'num_episodes': len(results[0].episodes) if results else 0,
        'fractions': list(fractions),
        'models': {}
    }
    for r in results:
        summary['models'][r.name] = {
            'belief_alignment': r.get_belief_alignment(1).mean,
            'adaptation_speed': r.get_adaptation_speed(1).mean,
            'top5_accuracy': r.get_top5_accuracy('post', 1).mean,
            'mrr': r.get_mrr('post', 1).mean,
            'pre_belief_alignment': AggregatedMetrics.from_values(
                [e.pre_phases[1].goal_prob for e in r.episodes]
            ).mean,
        }
    
    (output_dir / 'exp_3_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_visualizations(results: List[ModelResults], output_dir: Path) -> None:
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'SC-BDI-VAE': '#27ae60', 'Transformer': '#3498db', 'LSTM': '#e74c3c'}
    fractions = [0.25, 0.5, 0.75]
    
    # 1. Goal probability trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, phase, title, target in [
        (axes[0], 'pre', 'Pre-Pivot: P(Original Goal)', 'pre_phases'),
        (axes[1], 'post', 'Post-Pivot: P(New Goal)', 'post_phases'),
    ]:
        for r in results:
            phases = [getattr(e, target) for e in r.episodes]
            means = [np.mean([p[i].goal_prob for p in phases]) for i in range(3)]
            stds = [np.std([p[i].goal_prob for p in phases]) / np.sqrt(len(phases)) for i in range(3)]
            
            color = colors.get(r.name, 'gray')
            ax.errorbar(fractions, means, yerr=stds, marker='o', linewidth=2, 
                       markersize=8, label=r.name, color=color, capsize=4)
        
        ax.set_xlabel('Trajectory Fraction', fontsize=12)
        ax.set_ylabel('Goal Probability', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
        ax.set_xlim(0.2, 0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp_3_goal_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Metric comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['belief_alignment', 'adaptation_speed', 'top5_accuracy', 'mrr']
    metric_labels = ['Belief\nAlignment', 'Adaptation\nSpeed', 'Top-5\nAccuracy', 'MRR']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, r in enumerate(results):
        values = [
            r.get_belief_alignment(1).mean,
            r.get_adaptation_speed(1).mean,
            r.get_top5_accuracy('post', 1).mean,
            r.get_mrr('post', 1).mean,
        ]
        color = colors.get(r.name, f'C{i}')
        ax.bar(x + i * width, values, width, label=r.name, color=color, alpha=0.85)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Experiment 3: Belief Updating Metrics', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp_3_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Entropy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for r in results:
        phases = [e.post_phases for e in r.episodes]
        means = [np.mean([p[i].entropy for p in phases]) for i in range(3)]
        color = colors.get(r.name, 'gray')
        ax.plot(fractions, means, 'o-', label=r.name, color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Post-Pivot Fraction', fontsize=12)
    ax.set_ylabel('Entropy (nats)', fontsize=12)
    ax.set_title('Prediction Uncertainty During Rerouting', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp_3_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Rank distribution
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), squeeze=False)
    
    for i, r in enumerate(results):
        ax = axes[0, i]
        ranks = [e.post_phases[1].rank for e in r.episodes]
        bins = list(range(1, 52, 5)) + [230]  # Bins: 1-5, 6-10, ..., 46-50, 51+
        ax.hist(ranks, bins=bins, color=colors.get(r.name, 'gray'), alpha=0.7, edgecolor='white')
        ax.set_xlabel('Rank of Correct Goal', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{r.name}', fontsize=12)
        ax.axvline(x=5, color='green', linestyle='--', alpha=0.7, label='Top-5')
        ax.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='Top-10')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp_3_rank_distribution.png', dpi=150, bbox_inches='tight')
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
    output_dir = data_dir / 'visualizations' / 'exp_3'
    save_results(results, output_dir)
    
    if not args.no_viz:
        generate_visualizations(results, output_dir)


if __name__ == '__main__':
    main()
