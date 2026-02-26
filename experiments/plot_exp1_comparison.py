"""
Plot Top-1 Goal Prediction Accuracy comparison for Experiment 1.

Usage
-----
# Default: all four trained models + naive baseline
python experiments/plot_exp1_comparison.py

# Custom subset
python experiments/plot_exp1_comparison.py \
    --models lstm transformer sc_bdi_progress \
    --output experiments/results/exp_1/my_comparison.png

Available model keys
--------------------
  lstm              LSTM Baseline
  transformer       Transformer Baseline
  sc_bdi_progress   SC-BDI w/ Progress  (v2 checkpoint)
  sc_bdi_noprogress SC-BDI w/o Progress
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Registry: model_key → (csv_path, display_label, line_color, marker)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results" / "exp_1"

MODEL_REGISTRY = {
    "lstm": (
        RESULTS_DIR / "exp1_lstm_best_model_results.csv",
        "LSTM Baseline",
        "#2196F3",  # blue
        "D",
    ),
    "transformer": (
        RESULTS_DIR / "exp1_baseline_transformer_best_model_results.csv",
        "Transformer Baseline",
        "#FF9800",  # orange
        "s",
    ),
    "sc_bdi_progress": (
        RESULTS_DIR / "exp1_sc_bdi_progress_v2_results.csv",
        "SC-BDI w/ Progress",
        "#9C27B0",  # purple
        "o",
    ),
    "sc_bdi_noprogress": (
        RESULTS_DIR / "exp1_sc_bdi_no_progress_results.csv",
        "SC-BDI w/o Progress",
        "#E91E63",  # pink
        "^",
    ),
    "sc_bdi_gru": (
        RESULTS_DIR / "exp1_sc_bdi_gru_results.csv",
        "SC-BDI + GRU",
        "#4CAF50",  # green
        "p",
    ),
}

NAIVE_ACCURACY = 43.4
PROPORTIONS = [15, 30, 45, 60, 75, 90]

DEFAULT_OUTPUT = RESULTS_DIR / "exp1_comparison_plot.png"


# ---------------------------------------------------------------------------

def build_plot(model_keys: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for key in model_keys:
        if key not in MODEL_REGISTRY:
            print(f"[warn] Unknown model key '{key}' — skipping. "
                  f"Valid keys: {list(MODEL_REGISTRY)}")
            continue

        csv_path, label, color, marker = MODEL_REGISTRY[key]

        if not csv_path.exists():
            print(f"[warn] CSV not found for '{key}': {csv_path} — skipping.")
            continue

        df = pd.read_csv(csv_path)
        ax.plot(
            PROPORTIONS,
            df["top1_accuracy"].values,
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=8,
            label=label,
            zorder=3,
        )

    # Naive baseline
    ax.axhline(NAIVE_ACCURACY, color="#607D8B", linewidth=1.8,
               linestyle="--", zorder=2, label=f"Naïve Baseline ({NAIVE_ACCURACY}%)")

    # Axes
    ax.set_xlabel("Observed Trajectory (%)", fontsize=13, labelpad=8)
    ax.set_ylabel("Top-1 Goal Prediction Accuracy (%)", fontsize=13, labelpad=8)
    ax.set_title(
        "Top-1 Goal Prediction Accuracy vs. Observed Trajectory",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xticks(PROPORTIONS)
    ax.set_xticklabels([f"{p}%" for p in PROPORTIONS])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.set_ylim(35, 80)
    ax.grid(True, alpha=0.35, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Exp-1 Top-1 accuracy comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        metavar="KEY",
        help=(
            "Which models to include. "
            f"Choices: {list(MODEL_REGISTRY.keys())}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path. Default: experiments/results/exp_1/exp1_comparison_plot.png",
    )
    args = parser.parse_args()
    build_plot(args.models, args.output)


if __name__ == "__main__":
    main()
