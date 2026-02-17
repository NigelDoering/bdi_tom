"""
Experiment 2: Visualization of Preference-Proximity Dissociation Results

Generates publication-ready plots comparing model performance on distractor
episodes.  Works for both variant 2a (same-category) and 2b (cross-category);
just point ``--results-dir`` at the appropriate directory.

Usage:
    # Same-category results:
    python experiments/exp_2_plot.py --results-dir experiments/results/exp_2a

    # Cross-category results:
    python experiments/exp_2_plot.py --results-dir experiments/results/exp_2b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Style & constants ────────────────────────────────────────────────────────

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

COLORS = {
    'transformer': '#E24A33',  # Red
    'lstm': '#348ABD',         # Blue
    'bdi_vae': '#467821',      # Green (OURS)
}
MARKERS = {
    'transformer': 's',
    'lstm': '^',
    'bdi_vae': 'o',
}
MODEL_LABELS = {
    'baseline_transformer_best_model': 'Transformer',
    'lstm_best_model': 'LSTM',
    'best_model-OURS': 'SC-BDI (Ours)',
}
UNIFORM_COLOR = '#888888'
UNIFORM_STYLE = {'color': UNIFORM_COLOR, 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.7}


# ── Helper functions ─────────────────────────────────────────────────────────


def load_results(results_dir: Path, reference_type: str = "tstar") -> Dict[str, Dict]:
    """Load all exp2_*.json result files from directory."""
    results = {}

    if reference_type == "full":
        pattern = "exp2_*_full_traj_results.json"
        for f in results_dir.glob(pattern):
            with open(f) as fp:
                data = json.load(fp)
            model_stem = f.stem.replace("exp2_", "").replace("_full_traj_results", "")
            results[model_stem] = data
    else:
        for f in results_dir.glob("exp2_*_results.json"):
            if "_full_traj_results" in f.name or "_combined_results" in f.name:
                continue
            with open(f) as fp:
                data = json.load(fp)
            if "fractions" not in data:
                continue
            model_stem = f.stem.replace("exp2_", "").replace("_results", "")
            results[model_stem] = data

    return results


def get_model_color(model_name: str) -> str:
    if 'transformer' in model_name.lower():
        return COLORS['transformer']
    elif 'lstm' in model_name.lower():
        return COLORS['lstm']
    return COLORS['bdi_vae']


def get_model_marker(model_name: str) -> str:
    if 'transformer' in model_name.lower():
        return MARKERS['transformer']
    elif 'lstm' in model_name.lower():
        return MARKERS['lstm']
    return MARKERS['bdi_vae']


def get_model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def get_reference_label(reference_type: str) -> str:
    if reference_type == "full":
        return "Fraction of Full Trajectory"
    return "Fraction to t* (Closest Approach)"


def _uniform_baseline(data: Dict) -> Dict:
    """Extract or compute uniform baseline from result data."""
    ub = data.get("uniform_baseline", None)
    if ub is not None:
        return ub
    # Fallback: compute from n_pois or assume 230
    n = 230
    return {
        "prob_per_poi": 1.0 / n,
        "top1_accuracy": 100.0 / n,
        "top5_accuracy": min(500.0 / n, 100.0),
        "brier_score": (n - 1) * (1.0 / n) ** 2 + (1.0 - 1.0 / n) ** 2,
        "goal_dist_ratio": 1.0,
        "num_pois": n,
    }


def _fracs_and_x(data: Dict) -> Tuple[List[float], List[float]]:
    """Return sorted fractions and their %-scaled x values."""
    fractions = sorted(float(f) for f in data["fractions"].keys())
    return fractions, [f * 100 for f in fractions]


def _curve(data: Dict, fractions: List[float], metric: str):
    """Extract means, ci_lo, ci_hi for a metric across fractions."""
    means = [data["fractions"][str(f)][metric]["mean"] for f in fractions]
    lo = [data["fractions"][str(f)][metric]["ci_lo"] for f in fractions]
    hi = [data["fractions"][str(f)][metric]["ci_hi"] for f in fractions]
    return means, lo, hi


# ── Individual plots ─────────────────────────────────────────────────────────


def plot_distractor_probability(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ref_label = get_reference_label(reference_type)

    for model_name, data in results.items():
        fracs, x = _fracs_and_x(data)
        means, ci_lo, ci_hi = _curve(data, fracs, "distractor_prob")
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        ax.plot(x, means, marker='o', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)

    ub = _uniform_baseline(next(iter(results.values())))
    ax.axhline(y=ub["prob_per_poi"], label=f'Uniform (1/{ub["num_pois"]})', **UNIFORM_STYLE)

    ax.set_xlabel("Trajectory Observed (%)", fontsize=12)
    ax.set_ylabel("Distractor Probability", fontsize=12)
    ax.set_title(f"Distractor Probability vs Observation\n({ref_label})", fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xticks([10, 20, 50, 75, 90])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_goal_probability(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ref_label = get_reference_label(reference_type)

    for model_name, data in results.items():
        fracs, x = _fracs_and_x(data)
        means, ci_lo, ci_hi = _curve(data, fracs, "goal_prob")
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        ax.plot(x, means, marker='s', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)

    ub = _uniform_baseline(next(iter(results.values())))
    ax.axhline(y=ub["prob_per_poi"], label=f'Uniform (1/{ub["num_pois"]})', **UNIFORM_STYLE)

    ax.set_xlabel("Trajectory Observed (%)", fontsize=12)
    ax.set_ylabel("True Goal Probability", fontsize=12)
    ax.set_title(f"True Goal Probability vs Observation\n({ref_label})", fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xticks([10, 20, 50, 75, 90])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_comparison(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ref_label = get_reference_label(reference_type)
    ub = _uniform_baseline(next(iter(results.values())))

    for ax_i, (metric, ylabel, title_str, ub_val) in enumerate([
        ("top1_accuracy", "Top-1 Accuracy (%)", "Top-1 Goal Prediction Accuracy", ub["top1_accuracy"]),
        ("top5_accuracy", "Top-5 Accuracy (%)", "Top-5 Goal Prediction Accuracy", ub["top5_accuracy"]),
    ]):
        ax = axes[ax_i]
        for model_name, data in results.items():
            fracs, x = _fracs_and_x(data)
            if metric not in data["fractions"][str(fracs[0])]:
                continue
            means, ci_lo, ci_hi = _curve(data, fracs, metric)
            color = get_model_color(model_name)
            label = get_model_label(model_name)
            ax.plot(x, means, marker='o', linewidth=2.5, markersize=8, color=color, label=label)
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)

        ax.axhline(y=ub_val, label='Uniform', **UNIFORM_STYLE)
        ax.set_xlabel("Trajectory Observed (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title_str}\n({ref_label})", fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xticks([10, 20, 50, 75, 90])
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_brier_score(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ref_label = get_reference_label(reference_type)

    for model_name, data in results.items():
        fracs, x = _fracs_and_x(data)
        if "brier_score" not in data["fractions"][str(fracs[0])]:
            continue
        means, ci_lo, ci_hi = _curve(data, fracs, "brier_score")
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        ax.plot(x, means, marker='^', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)

    ub = _uniform_baseline(next(iter(results.values())))
    ax.axhline(y=ub["brier_score"], label='Uniform', **UNIFORM_STYLE)

    ax.set_xlabel("Trajectory Observed (%)", fontsize=12)
    ax.set_ylabel("Brier Score (lower is better)", fontsize=12)
    ax.set_title(f"Prediction Calibration: Brier Score\n({ref_label})", fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xticks([10, 20, 50, 75, 90])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_goal_dist_ratio(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    """Plot goal-to-distractor probability ratio across observation fractions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ref_label = get_reference_label(reference_type)

    for model_name, data in results.items():
        fracs, x = _fracs_and_x(data)
        if "goal_dist_ratio" not in data["fractions"][str(fracs[0])]:
            continue
        means, ci_lo, ci_hi = _curve(data, fracs, "goal_dist_ratio")
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        ax.plot(x, means, marker='D', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)

    ax.axhline(y=1.0, label='Uniform (ratio = 1)', **UNIFORM_STYLE)

    ax.set_xlabel("Trajectory Observed (%)", fontsize=12)
    ax.set_ylabel("P(Goal) / P(Distractor)", fontsize=12)
    ax.set_title(f"Goal-to-Distractor Probability Ratio\n({ref_label})", fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xticks([10, 20, 50, 75, 90])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_bars(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    """Plot summary bar chart comparing overall metrics."""
    models = list(results.keys())
    n_models = len(models)
    ref_short = "t*" if reference_type == "tstar" else "Full Traj"
    ub = _uniform_baseline(next(iter(results.values())))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    bar_configs = [
        ("overall_goal_dist_ratio", "P(Goal) / P(Distractor)", f"Overall Goal/Dist Ratio (↑ better) [{ref_short}]", ub["goal_dist_ratio"]),
        ("overall_top5_accuracy", "Top-5 Accuracy (%)", f"Overall Top-5 Accuracy (↑ better)", ub["top5_accuracy"]),
        ("overall_top1_accuracy", "Top-1 Accuracy (%)", f"Overall Top-1 Accuracy (↑ better)", ub["top1_accuracy"]),
        ("overall_brier_score", "Brier Score", f"Overall Brier Score (↓ better)", ub["brier_score"]),
    ]

    for ax, (metric_key, ylabel, title, ub_val) in zip(axes.flat, bar_configs):
        if metric_key in results[models[0]]:
            x = np.arange(n_models)
            means = [results[m][metric_key]["mean"] for m in models]
            ci_lo = [results[m][metric_key]["ci_lo"] for m in models]
            ci_hi = [results[m][metric_key]["ci_hi"] for m in models]
            errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
            colors = [get_model_color(m) for m in models]
            labels = [get_model_label(m) for m in models]

            ax.bar(x, means, yerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black')
            ax.axhline(y=ub_val, **UNIFORM_STYLE, label='Uniform')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8)
            if metric_key != "overall_brier_score":
                ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "Re-run eval for\nthis metric", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ── Combined publication figure ──────────────────────────────────────────────


def plot_combined_figure(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    """Create a single publication-ready 2×3 figure.

    Layout:
      (a) P(True Goal)            (b) Goal/Distractor Ratio     (c) Top-1 Accuracy
      (d) Top-5 Accuracy          (e) Brier Score               (f) Summary Bars
    """
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    ref_label = get_reference_label(reference_type)
    models = list(results.keys())
    n_models = len(models)
    ub = _uniform_baseline(next(iter(results.values())))

    # ── Helper to draw a curve panel ──
    def _draw_curve(ax, metric: str, ylabel: str, title: str,
                    ub_val: float, ub_label: str = 'Uniform',
                    marker: str = 'o', log_y: bool = False,
                    show_legend: bool = False):
        for model_name, data in results.items():
            fracs, x = _fracs_and_x(data)
            if metric not in data["fractions"][str(fracs[0])]:
                continue
            means, ci_lo, ci_hi = _curve(data, fracs, metric)
            color = get_model_color(model_name)
            mk = get_model_marker(model_name)
            label = get_model_label(model_name)
            ax.plot(x, means, marker=mk, linewidth=2, markersize=6,
                    color=color, label=label)
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.12, color=color)

        ax.axhline(y=ub_val, label=ub_label, **UNIFORM_STYLE)
        ax.set_xlabel("Trajectory Observed (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([10, 20, 50, 75, 90])
        if log_y:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_ylim(bottom=0.5)
        else:
            ax.set_ylim(bottom=0)
        if show_legend:
            ax.legend(loc='upper left', fontsize=9)

    # (a) Goal probability — the primary metric
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_curve(ax_a, "goal_prob", "P(True Goal)",
                "(a) True Goal Probability",
                ub["prob_per_poi"], f'Uniform (1/{ub["num_pois"]})',
                show_legend=True)

    # (b) Goal-to-distractor ratio — reframes distractor story
    ax_b = fig.add_subplot(gs[0, 1])
    has_ratio = "goal_dist_ratio" in next(iter(results.values()))["fractions"].get(
        list(next(iter(results.values()))["fractions"].keys())[0], {}
    )
    if has_ratio:
        _draw_curve(ax_b, "goal_dist_ratio", "P(Goal) / P(Distractor)",
                    "(b) Goal / Distractor Ratio",
                    1.0, 'Uniform (ratio = 1)',
                    marker='D', log_y=False)
    else:
        ax_b.text(0.5, 0.5, "Re-run eval for\nratio metric",
                  ha='center', va='center', transform=ax_b.transAxes)
        ax_b.set_title("(b) Goal / Distractor Ratio", fontweight='bold')

    # (c) Top-1 accuracy
    ax_c = fig.add_subplot(gs[0, 2])
    _draw_curve(ax_c, "top1_accuracy", "Top-1 Accuracy (%)",
                "(c) Top-1 Accuracy",
                ub["top1_accuracy"], 'Uniform')

    # (d) Top-5 accuracy
    ax_d = fig.add_subplot(gs[1, 0])
    _draw_curve(ax_d, "top5_accuracy", "Top-5 Accuracy (%)",
                "(d) Top-5 Accuracy",
                ub["top5_accuracy"], 'Uniform')

    # (e) Brier score
    ax_e = fig.add_subplot(gs[1, 1])
    _draw_curve(ax_e, "brier_score", "Brier Score (↓ better)",
                "(e) Brier Score",
                ub["brier_score"], 'Uniform')

    # (f) Summary bars — overall metrics, 4 grouped bars per model
    ax_f = fig.add_subplot(gs[1, 2])
    bar_metrics = [
        ("overall_goal_prob", "P(Goal)"),
        ("overall_top1_accuracy", "Top-1 (%)"),
        ("overall_top5_accuracy", "Top-5 (%)"),
    ]
    n_metrics = len(bar_metrics)
    bar_w = 0.22
    x_base = np.arange(n_metrics)

    for mi, model_name in enumerate(models):
        data = results[model_name]
        vals = []
        errs_lo = []
        errs_hi = []
        for metric_key, _ in bar_metrics:
            if metric_key in data:
                m = data[metric_key]
                # Scale P(Goal) to % for visual comparability
                if metric_key == "overall_goal_prob":
                    vals.append(m["mean"] * 100)
                    errs_lo.append((m["mean"] - m["ci_lo"]) * 100)
                    errs_hi.append((m["ci_hi"] - m["mean"]) * 100)
                else:
                    vals.append(m["mean"])
                    errs_lo.append(m["mean"] - m["ci_lo"])
                    errs_hi.append(m["ci_hi"] - m["mean"])
            else:
                vals.append(0)
                errs_lo.append(0)
                errs_hi.append(0)

        color = get_model_color(model_name)
        label = get_model_label(model_name)
        offset = (mi - (n_models - 1) / 2) * bar_w
        ax_f.bar(x_base + offset, vals, bar_w, yerr=[errs_lo, errs_hi],
                 capsize=3, color=color, alpha=0.85, edgecolor='black',
                 linewidth=0.5, label=label)

    # Uniform reference lines for each metric group
    ub_vals = [ub["prob_per_poi"] * 100, ub["top1_accuracy"], ub["top5_accuracy"]]
    for xi, uv in enumerate(ub_vals):
        ax_f.plot([xi - 0.4, xi + 0.4], [uv, uv], **UNIFORM_STYLE)

    tick_labels = [lbl for _, lbl in bar_metrics]
    # Relabel P(Goal) to show it's in %
    tick_labels[0] = "P(Goal) ×100"
    ax_f.set_xticks(x_base)
    ax_f.set_xticklabels(tick_labels, fontsize=9)
    ax_f.set_ylabel("Value")
    ax_f.set_title("(f) Overall Metrics (↑ better)", fontweight='bold')
    ax_f.legend(fontsize=8, loc='upper left')
    ax_f.set_ylim(bottom=0)

    plt.suptitle(f"Experiment 2: Preference-Proximity Dissociation Test\n({ref_label})",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ── Entry points ─────────────────────────────────────────────────────────────


def generate_plots_for_reference(results: Dict[str, Dict], output_dir: Path,
                                  reference_type: str, suffix: str):
    """Generate all plots for a specific reference type."""
    print(f"\n--- Generating plots for {reference_type} reference ---")

    plot_goal_probability(results, output_dir / f"exp2_goal_prob{suffix}.png", reference_type)
    plot_goal_dist_ratio(results, output_dir / f"exp2_goal_dist_ratio{suffix}.png", reference_type)
    plot_distractor_probability(results, output_dir / f"exp2_distractor_prob{suffix}.png", reference_type)
    plot_accuracy_comparison(results, output_dir / f"exp2_accuracy{suffix}.png", reference_type)
    plot_brier_score(results, output_dir / f"exp2_brier{suffix}.png", reference_type)
    plot_summary_bars(results, output_dir / f"exp2_summary_bars{suffix}.png", reference_type)
    plot_combined_figure(results, output_dir / f"exp2_combined{suffix}.png", reference_type)


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 2 results")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("experiments/results/exp_2"),
                        help="Directory containing exp2_*_results.json files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (defaults to results-dir)")
    parser.add_argument("--reference", type=str, default="both",
                        choices=["tstar", "full", "both"],
                        help="Which reference type to plot: tstar, full, or both")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reference in ["tstar", "both"]:
        print(f"\nLoading t*-relative results from {args.results_dir}")
        results_tstar = load_results(args.results_dir, "tstar")
        if results_tstar:
            print(f"Found {len(results_tstar)} models: {list(results_tstar.keys())}")
            generate_plots_for_reference(results_tstar, output_dir, "tstar", "_tstar")
        else:
            print("No t*-relative results found!")

    if args.reference in ["full", "both"]:
        print(f"\nLoading full-trajectory-relative results from {args.results_dir}")
        results_full = load_results(args.results_dir, "full")
        if results_full:
            print(f"Found {len(results_full)} models: {list(results_full.keys())}")
            generate_plots_for_reference(results_full, output_dir, "full", "_full_traj")
        else:
            print("No full-trajectory-relative results found!")

    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
