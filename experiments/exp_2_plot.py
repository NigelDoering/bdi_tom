"""
Experiment 2: Visualization of Preference-Proximity Dissociation Results

Generates comprehensive plots comparing model performance on distractor episodes.

Usage:
    python experiments/exp_2_plot.py --results-dir data/simulation_data/run_8/visualizations/exp_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'transformer': '#E24A33',  # Red
    'lstm': '#348ABD',         # Blue  
    'bdi_vae': '#467821',      # Green (OURS)
}
MODEL_LABELS = {
    'baseline_transformer_best_model': 'Transformer',
    'lstm_best_model': 'LSTM',
    'best_model-OURS': 'SC-BDI (Ours)',
}


def load_results(results_dir: Path, reference_type: str = "tstar") -> Dict[str, Dict]:
    """Load all exp2_*.json result files from directory.
    
    Args:
        results_dir: Directory containing result files
        reference_type: "tstar" for t*-relative fractions, "full" for full-trajectory fractions
    
    Returns:
        Dictionary mapping model names to their results
    """
    results = {}
    
    if reference_type == "full":
        # Load full trajectory results
        pattern = "exp2_*_full_traj_results.json"
        for f in results_dir.glob(pattern):
            with open(f) as fp:
                data = json.load(fp)
            model_stem = f.stem.replace("exp2_", "").replace("_full_traj_results", "")
            results[model_stem] = data
    else:
        # Load t* results (original format)
        for f in results_dir.glob("exp2_*_results.json"):
            # Skip full trajectory files and combined files
            if "_full_traj_results" in f.name or "_combined_results" in f.name:
                continue
            with open(f) as fp:
                data = json.load(fp)
            # Check that the file has the expected structure
            if "fractions" not in data:
                continue
            model_stem = f.stem.replace("exp2_", "").replace("_results", "")
            results[model_stem] = data
    
    return results


def get_model_color(model_name: str) -> str:
    """Get color for model based on type."""
    if 'transformer' in model_name.lower():
        return COLORS['transformer']
    elif 'lstm' in model_name.lower():
        return COLORS['lstm']
    else:
        return COLORS['bdi_vae']


def get_model_label(model_name: str) -> str:
    """Get display label for model."""
    return MODEL_LABELS.get(model_name, model_name)


def get_reference_label(reference_type: str) -> str:
    """Get display label for reference type."""
    if reference_type == "full":
        return "Fraction of Full Trajectory"
    else:
        return "Fraction to t* (Closest Approach)"


def plot_distractor_probability(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    """Plot distractor probability across observation fractions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ref_label = get_reference_label(reference_type)
    
    for model_name, data in results.items():
        fractions = sorted([float(f) for f in data["fractions"].keys()])
        means = [data["fractions"][str(f)]["distractor_prob"]["mean"] for f in fractions]
        ci_lo = [data["fractions"][str(f)]["distractor_prob"]["ci_lo"] for f in fractions]
        ci_hi = [data["fractions"][str(f)]["distractor_prob"]["ci_hi"] for f in fractions]
        
        x = [f * 100 for f in fractions]
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        
        ax.plot(x, means, marker='o', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)
    
    ax.set_xlabel(f"Trajectory Observed (% of {ref_label.split(' ')[-1]})", fontsize=12)
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
    """Plot goal probability across observation fractions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ref_label = get_reference_label(reference_type)
    
    for model_name, data in results.items():
        fractions = sorted([float(f) for f in data["fractions"].keys()])
        means = [data["fractions"][str(f)]["goal_prob"]["mean"] for f in fractions]
        ci_lo = [data["fractions"][str(f)]["goal_prob"]["ci_lo"] for f in fractions]
        ci_hi = [data["fractions"][str(f)]["goal_prob"]["ci_hi"] for f in fractions]
        
        x = [f * 100 for f in fractions]
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        
        ax.plot(x, means, marker='s', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)
    
    ax.set_xlabel(f"Trajectory Observed (% of {ref_label.split(' ')[-1]})", fontsize=12)
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
    """Plot Top-1 and Top-5 accuracy across observation fractions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ref_label = get_reference_label(reference_type)
    
    # Top-1 Accuracy
    ax = axes[0]
    for model_name, data in results.items():
        fractions = sorted([float(f) for f in data["fractions"].keys()])
        if "top1_accuracy" not in data["fractions"][str(fractions[0])]:
            continue
        means = [data["fractions"][str(f)]["top1_accuracy"]["mean"] for f in fractions]
        ci_lo = [data["fractions"][str(f)]["top1_accuracy"]["ci_lo"] for f in fractions]
        ci_hi = [data["fractions"][str(f)]["top1_accuracy"]["ci_hi"] for f in fractions]
        
        x = [f * 100 for f in fractions]
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        
        ax.plot(x, means, marker='o', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)
    
    ax.set_xlabel(f"Trajectory Observed (%)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(f"Top-1 Goal Prediction Accuracy\n({ref_label})", fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xticks([10, 20, 50, 75, 90])
    ax.set_ylim(bottom=0)
    
    # Top-5 Accuracy
    ax = axes[1]
    for model_name, data in results.items():
        fractions = sorted([float(f) for f in data["fractions"].keys()])
        if "top5_accuracy" not in data["fractions"][str(fractions[0])]:
            continue
        means = [data["fractions"][str(f)]["top5_accuracy"]["mean"] for f in fractions]
        ci_lo = [data["fractions"][str(f)]["top5_accuracy"]["ci_lo"] for f in fractions]
        ci_hi = [data["fractions"][str(f)]["top5_accuracy"]["ci_hi"] for f in fractions]
        
        x = [f * 100 for f in fractions]
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        
        ax.plot(x, means, marker='s', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)
    
    ax.set_xlabel(f"Trajectory Observed (%)", fontsize=12)
    ax.set_ylabel("Top-5 Accuracy (%)", fontsize=12)
    ax.set_title(f"Top-5 Goal Prediction Accuracy\n({ref_label})", fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xticks([10, 20, 50, 75, 90])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_brier_score(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    """Plot Brier score across observation fractions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ref_label = get_reference_label(reference_type)
    
    for model_name, data in results.items():
        fractions = sorted([float(f) for f in data["fractions"].keys()])
        if "brier_score" not in data["fractions"][str(fractions[0])]:
            continue
        means = [data["fractions"][str(f)]["brier_score"]["mean"] for f in fractions]
        ci_lo = [data["fractions"][str(f)]["brier_score"]["ci_lo"] for f in fractions]
        ci_hi = [data["fractions"][str(f)]["brier_score"]["ci_hi"] for f in fractions]
        
        x = [f * 100 for f in fractions]
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        
        ax.plot(x, means, marker='^', linewidth=2.5, markersize=8, color=color, label=label)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=color)
    
    ax.set_xlabel(f"Trajectory Observed (%)", fontsize=12)
    ax.set_ylabel("Brier Score (lower is better)", fontsize=12)
    ax.set_title(f"Prediction Calibration: Brier Score\n({ref_label})", fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xticks([10, 20, 50, 75, 90])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_bars(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    """Plot summary bar chart comparing overall metrics."""
    models = list(results.keys())
    n_models = len(models)
    
    ref_label = get_reference_label(reference_type)
    ref_short = "t*" if reference_type == "tstar" else "Full Traj"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Peak Distractor Probability (lower is better)
    ax = axes[0, 0]
    x = np.arange(n_models)
    means = [results[m]["peak_distractor_prob"]["mean"] for m in models]
    ci_lo = [results[m]["peak_distractor_prob"]["ci_lo"] for m in models]
    ci_hi = [results[m]["peak_distractor_prob"]["ci_hi"] for m in models]
    errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
    colors = [get_model_color(m) for m in models]
    labels = [get_model_label(m) for m in models]
    
    bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel("Peak Distractor Probability")
    ax.set_title(f"Peak Distractor Probability (↓ better) [{ref_short}]")
    ax.set_ylim(bottom=0)
    
    # 2. Overall Distractor Probability (lower is better)
    ax = axes[0, 1]
    means = [results[m]["overall_distractor_prob"]["mean"] for m in models]
    ci_lo = [results[m]["overall_distractor_prob"]["ci_lo"] for m in models]
    ci_hi = [results[m]["overall_distractor_prob"]["ci_hi"] for m in models]
    errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
    
    bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel("Overall Distractor Probability")
    ax.set_title("Overall Distractor Probability (↓ better)")
    ax.set_ylim(bottom=0)
    
    # 3. Top-1 Accuracy (higher is better)
    ax = axes[1, 0]
    if "overall_top1_accuracy" in results[models[0]]:
        means = [results[m]["overall_top1_accuracy"]["mean"] for m in models]
        ci_lo = [results[m]["overall_top1_accuracy"]["ci_lo"] for m in models]
        ci_hi = [results[m]["overall_top1_accuracy"]["ci_hi"] for m in models]
        errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
        
        bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("Overall Top-1 Accuracy (↑ better)")
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, "Run updated eval\nfor Top-1 metrics", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Top-1 Accuracy (not available)")
    
    # 4. Brier Score (lower is better)
    ax = axes[1, 1]
    if "overall_brier_score" in results[models[0]]:
        means = [results[m]["overall_brier_score"]["mean"] for m in models]
        ci_lo = [results[m]["overall_brier_score"]["ci_lo"] for m in models]
        ci_hi = [results[m]["overall_brier_score"]["ci_hi"] for m in models]
        errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
        
        bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel("Brier Score")
        ax.set_title("Overall Brier Score (↓ better)")
    else:
        ax.text(0.5, 0.5, "Run updated eval\nfor Brier metrics", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Brier Score (not available)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_figure(results: Dict[str, Dict], output_path: Path, reference_type: str = "tstar"):
    """Create a single publication-ready figure with all key metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    ref_label = get_reference_label(reference_type)
    ref_short = "t*" if reference_type == "tstar" else "Full Trajectory"
    
    # Create 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    models = list(results.keys())
    n_models = len(models)
    
    # Plot 1: Distractor Probability vs Fraction
    ax1 = fig.add_subplot(gs[0, 0])
    for model_name, data in results.items():
        fractions = sorted([float(f) for f in data["fractions"].keys()])
        means = [data["fractions"][str(f)]["distractor_prob"]["mean"] for f in fractions]
        ci_lo = [data["fractions"][str(f)]["distractor_prob"]["ci_lo"] for f in fractions]
        ci_hi = [data["fractions"][str(f)]["distractor_prob"]["ci_hi"] for f in fractions]
        x = [f * 100 for f in fractions]
        color = get_model_color(model_name)
        label = get_model_label(model_name)
        ax1.plot(x, means, marker='o', linewidth=2, markersize=6, color=color, label=label)
        ax1.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)
    ax1.set_xlabel("Trajectory Observed (%)")
    ax1.set_ylabel("P(Distractor)")
    ax1.set_title("(a) Distractor Probability", fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Goal Probability vs Fraction
    ax2 = fig.add_subplot(gs[0, 1])
    for model_name, data in results.items():
        fractions = sorted([float(f) for f in data["fractions"].keys()])
        means = [data["fractions"][str(f)]["goal_prob"]["mean"] for f in fractions]
        ci_lo = [data["fractions"][str(f)]["goal_prob"]["ci_lo"] for f in fractions]
        ci_hi = [data["fractions"][str(f)]["goal_prob"]["ci_hi"] for f in fractions]
        x = [f * 100 for f in fractions]
        color = get_model_color(model_name)
        ax2.plot(x, means, marker='s', linewidth=2, markersize=6, color=color)
        ax2.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)
    ax2.set_xlabel("Trajectory Observed (%)")
    ax2.set_ylabel("P(True Goal)")
    ax2.set_title("(b) True Goal Probability", fontweight='bold')
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Top-1 Accuracy vs Fraction
    ax3 = fig.add_subplot(gs[0, 2])
    has_top1 = "top1_accuracy" in results[models[0]]["fractions"][list(results[models[0]]["fractions"].keys())[0]]
    if has_top1:
        for model_name, data in results.items():
            fractions = sorted([float(f) for f in data["fractions"].keys()])
            means = [data["fractions"][str(f)]["top1_accuracy"]["mean"] for f in fractions]
            ci_lo = [data["fractions"][str(f)]["top1_accuracy"]["ci_lo"] for f in fractions]
            ci_hi = [data["fractions"][str(f)]["top1_accuracy"]["ci_hi"] for f in fractions]
            x = [f * 100 for f in fractions]
            color = get_model_color(model_name)
            ax3.plot(x, means, marker='^', linewidth=2, markersize=6, color=color)
            ax3.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)
        ax3.set_xlabel("Trajectory Observed (%)")
        ax3.set_ylabel("Top-1 Accuracy (%)")
        ax3.set_title("(c) Top-1 Accuracy", fontweight='bold')
        ax3.set_ylim(bottom=0)
    else:
        ax3.text(0.5, 0.5, "Re-run eval for\nTop-1 metrics", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("(c) Top-1 Accuracy", fontweight='bold')
    
    # Plot 4: Summary Bars - Peak Distractor
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(n_models)
    means = [results[m]["peak_distractor_prob"]["mean"] for m in models]
    ci_lo = [results[m]["peak_distractor_prob"]["ci_lo"] for m in models]
    ci_hi = [results[m]["peak_distractor_prob"]["ci_hi"] for m in models]
    errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
    colors = [get_model_color(m) for m in models]
    labels = [get_model_label(m) for m in models]
    ax4.bar(x, means, yerr=errors, capsize=4, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax4.set_ylabel("Peak P(Distractor)")
    ax4.set_title("(d) Peak Distractor Prob. (↓ better)", fontweight='bold')
    ax4.set_ylim(bottom=0)
    
    # Plot 5: Summary Bars - Top-5 Accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    if "overall_top5_accuracy" in results[models[0]]:
        means = [results[m]["overall_top5_accuracy"]["mean"] for m in models]
        ci_lo = [results[m]["overall_top5_accuracy"]["ci_lo"] for m in models]
        ci_hi = [results[m]["overall_top5_accuracy"]["ci_hi"] for m in models]
        errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
        ax5.bar(x, means, yerr=errors, capsize=4, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax5.set_ylabel("Top-5 Accuracy (%)")
        ax5.set_title("(e) Overall Top-5 Accuracy (↑ better)", fontweight='bold')
        ax5.set_ylim(bottom=0)
    else:
        ax5.text(0.5, 0.5, "Re-run eval", ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title("(e) Top-5 Accuracy", fontweight='bold')
    
    # Plot 6: Summary Bars - Brier Score
    ax6 = fig.add_subplot(gs[1, 2])
    if "overall_brier_score" in results[models[0]]:
        means = [results[m]["overall_brier_score"]["mean"] for m in models]
        ci_lo = [results[m]["overall_brier_score"]["ci_lo"] for m in models]
        ci_hi = [results[m]["overall_brier_score"]["ci_hi"] for m in models]
        errors = [[m - l for m, l in zip(means, ci_lo)], [h - m for m, h in zip(means, ci_hi)]]
        ax6.bar(x, means, yerr=errors, capsize=4, color=colors, alpha=0.8, edgecolor='black')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax6.set_ylabel("Brier Score")
        ax6.set_title("(f) Overall Brier Score (↓ better)", fontweight='bold')
    else:
        ax6.text(0.5, 0.5, "Re-run eval", ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title("(f) Brier Score", fontweight='bold')
    
    plt.suptitle(f"Experiment 2: Preference-Proximity Dissociation Test\n({ref_label})", fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_plots_for_reference(results: Dict[str, Dict], output_dir: Path, reference_type: str, suffix: str):
    """Generate all plots for a specific reference type."""
    print(f"\n--- Generating plots for {reference_type} reference ---")
    
    plot_distractor_probability(results, output_dir / f"exp2_distractor_prob{suffix}.png", reference_type)
    plot_goal_probability(results, output_dir / f"exp2_goal_prob{suffix}.png", reference_type)
    plot_accuracy_comparison(results, output_dir / f"exp2_accuracy{suffix}.png", reference_type)
    plot_brier_score(results, output_dir / f"exp2_brier{suffix}.png", reference_type)
    plot_summary_bars(results, output_dir / f"exp2_summary_bars{suffix}.png", reference_type)
    plot_combined_figure(results, output_dir / f"exp2_combined{suffix}.png", reference_type)


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 2 results")
    parser.add_argument("--results-dir", type=Path, 
                        default=Path("data/simulation_data/run_8/visualizations/exp_2"),
                        help="Directory containing exp2_*_results.json files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (defaults to results-dir)")
    parser.add_argument("--reference", type=str, default="both", choices=["tstar", "full", "both"],
                        help="Which reference type to plot: tstar, full, or both")
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for t* reference (original)
    if args.reference in ["tstar", "both"]:
        print(f"\nLoading t*-relative results from {args.results_dir}")
        results_tstar = load_results(args.results_dir, "tstar")
        
        if results_tstar:
            print(f"Found {len(results_tstar)} models: {list(results_tstar.keys())}")
            generate_plots_for_reference(results_tstar, output_dir, "tstar", "_tstar")
        else:
            print("No t*-relative results found!")
    
    # Generate plots for full trajectory reference (new)
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
