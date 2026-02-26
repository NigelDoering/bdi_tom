"""
Analyzes the agent destination distribution in the trajectory dataset.

This script tests the hypothesis that SC-BDI's ~43% flat accuracy in exp_1
is explained purely by agents having heavily skewed destination preferences —
i.e., a simple "always predict the most frequent destination" baseline achieves
the same accuracy without any sequence modeling.

Usage:
    python experiments/analyze_destination_distribution.py
    python experiments/analyze_destination_distribution.py --data_path data/simulation_data/run_8/trajectories/all_trajectories.json
    python experiments/analyze_destination_distribution.py --output_dir experiments/results/exp_1
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_agent_goals(data_path: str) -> tuple[dict, list]:
    """Load trajectories and extract goal node for each agent's trips."""
    with open(data_path) as f:
        data = json.load(f)

    agent_goals = defaultdict(list)
    for agent_key in sorted(data.keys()):
        for traj in data[agent_key]:
            path = traj["path"]
            goal = traj.get("goal_node", path[-1])
            if isinstance(goal, (list, tuple)):
                goal = goal[0]
            agent_goals[agent_key].append(goal)

    all_goals = sorted({g for goals in agent_goals.values() for g in goals})
    return dict(agent_goals), all_goals


def compute_agent_stats(agent_goals: dict) -> dict:
    """Compute per-agent accuracy statistics for the most-frequent-destination baseline."""
    stats = {}
    for agent_key, goals in agent_goals.items():
        n = len(goals)
        counts = Counter(goals)
        sorted_counts = sorted(counts.values(), reverse=True)
        probs = np.array(sorted_counts) / n

        stats[agent_key] = {
            "top1":    sorted_counts[0] / n * 100,
            "top5":    sum(sorted_counts[:5]) / n * 100,
            "top10":   sum(sorted_counts[:10]) / n * 100,
            "entropy": -np.sum(probs * np.log2(probs + 1e-10)),
            "n_unique": len(counts),
            "sorted_counts": sorted_counts,
        }
    return stats


def print_summary(agent_stats: dict, all_goals: list):
    top1   = [s["top1"]    for s in agent_stats.values()]
    top5   = [s["top5"]    for s in agent_stats.values()]
    top10  = [s["top10"]   for s in agent_stats.values()]
    ent    = [s["entropy"] for s in agent_stats.values()]

    print("=" * 60)
    print("AGENT DESTINATION DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"  Agents:            {len(agent_stats)}")
    print(f"  Unique goal nodes: {len(all_goals)}")
    print(f"  Uniform entropy:   {np.log2(len(all_goals)):.2f} bits")
    print()
    print("  'Always predict most-frequent destination' baseline:")
    print(f"    Top-1  accuracy: {np.mean(top1):.1f}%  "
          f"(std={np.std(top1):.1f}%, min={np.min(top1):.1f}%, max={np.max(top1):.1f}%)")
    print(f"    Top-5  accuracy: {np.mean(top5):.1f}%  (std={np.std(top5):.1f}%)")
    print(f"    Top-10 accuracy: {np.mean(top10):.1f}%  (std={np.std(top10):.1f}%)")
    print(f"    Mean entropy:    {np.mean(ent):.2f} bits  "
          f"(effective choices ≈ {2**np.mean(ent):.0f})")
    print("=" * 60)


def make_plot(agent_goals: dict, agent_stats: dict, all_goals: list, output_dir: str):
    num_agents = len(agent_goals)
    num_goals  = len(all_goals)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Sorted goal distribution for 4 representative agents
    ax = axes[0, 0]
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]["top1"])
    colors_a = ["#D65F5F", "#EE854A", "#4878D0", "#6ACC65"]
    for i, pct in enumerate([0, 25, 50, 75]):
        idx = int(pct / 100 * (num_agents - 1))
        agent_key, s = sorted_agents[idx]
        vals = np.array(s["sorted_counts"]) / len(agent_goals[agent_key]) * 100
        ax.bar(np.arange(len(vals)) + i * 0.2, vals, width=0.2,
               alpha=0.85, color=colors_a[i],
               label=f"{agent_key}  (top1={s['top1']:.0f}%)")
    ax.set_xlabel("Destination rank", fontsize=11)
    ax.set_ylabel("Visit frequency (%)", fontsize=11)
    ax.set_title("(A) Goal distribution — 4 example agents", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, 30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (B) Histogram of top-1 accuracy across all agents
    ax = axes[0, 1]
    top1_vals = [s["top1"] for s in agent_stats.values()]
    ax.hist(top1_vals, bins=20, color="#4878D0", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(top1_vals), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(top1_vals):.1f}%")
    ax.axvline(43.0, color="green", linestyle=":", linewidth=2,
               label="SC-BDI exp_1 ≈ 43%")
    ax.set_xlabel("Top-1 'most frequent destination' accuracy (%)", fontsize=11)
    ax.set_ylabel("Number of agents", fontsize=11)
    ax.set_title("(B) Top-1 accuracy — always predict most\ncommon destination per agent", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (C) Average goal distribution over all POI nodes
    ax = axes[1, 0]
    goal_to_idx = {g: i for i, g in enumerate(all_goals)}
    avg_probs = np.zeros(num_goals)
    for agent_key, goals in agent_goals.items():
        n = len(goals)
        for goal, count in Counter(goals).items():
            avg_probs[goal_to_idx[goal]] += count / n / num_agents
    sorted_avg = np.sort(avg_probs)[::-1]
    ax.bar(range(num_goals), sorted_avg * 100, color="#956CB4", alpha=0.85, width=1.0)
    ax.set_xlabel("Goal node (sorted by avg frequency)", fontsize=11)
    ax.set_ylabel("Average visit probability (%)", fontsize=11)
    ax.set_title(f"(C) Average goal distribution across all agents\n({num_goals} POI nodes)", fontsize=11, fontweight="bold")
    ax.set_xlim(-1, num_goals)
    ax.grid(True, alpha=0.3)

    # (D) Cumulative top-K coverage
    ax = axes[1, 1]
    all_coverage = []
    for agent_key, goals in agent_goals.items():
        n = len(goals)
        sorted_vals = sorted(Counter(goals).values(), reverse=True)
        cumul = np.cumsum(sorted_vals) / n * 100
        if len(cumul) < num_goals:
            cumul = np.concatenate([cumul, np.full(num_goals - len(cumul), 100.0)])
        all_coverage.append(cumul)
    all_coverage = np.array(all_coverage)
    mean_cov = np.mean(all_coverage, axis=0)
    p25 = np.percentile(all_coverage, 25, axis=0)
    p75 = np.percentile(all_coverage, 75, axis=0)
    k = np.arange(1, num_goals + 1)
    ax.plot(k, mean_cov, color="#D65F5F", linewidth=2, label="Mean across agents")
    ax.fill_between(k, p25, p75, alpha=0.2, color="#D65F5F", label="25th–75th percentile")
    ax.axhline(43, color="green", linestyle=":", linewidth=1.5, label="SC-BDI exp_1 ≈ 43%")
    ax.annotate(f"Top-1: {mean_cov[0]:.1f}%",  xy=(1, mean_cov[0]),  fontsize=9, xytext=(8,  mean_cov[0]  - 5))
    ax.annotate(f"Top-5: {mean_cov[4]:.1f}%",  xy=(5, mean_cov[4]),  fontsize=9, xytext=(10, mean_cov[4]  - 5))
    ax.annotate(f"Top-10: {mean_cov[9]:.1f}%", xy=(10, mean_cov[9]), fontsize=9, xytext=(15, mean_cov[9]  - 5))
    ax.set_xlabel("Number of top destinations (K)", fontsize=11)
    ax.set_ylabel("Cumulative coverage (%)", fontsize=11)
    ax.set_title("(D) Cumulative traffic covered by top-K destinations", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 50)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Agent Destination Distribution Analysis\n"
        "The SC-BDI model's ~43% flat accuracy matches a trivial 'predict most-frequent destination' baseline",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = Path(output_dir) / "agent_destination_analysis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze agent destination distributions.")
    parser.add_argument(
        "--data_path",
        default="data/simulation_data/run_8/trajectories/all_trajectories.json",
        help="Path to all_trajectories.json",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/results/exp_1",
        help="Directory to save the output plot",
    )
    args = parser.parse_args()

    print(f"Loading trajectories from: {args.data_path}")
    agent_goals, all_goals = load_agent_goals(args.data_path)
    agent_stats = compute_agent_stats(agent_goals)
    print_summary(agent_stats, all_goals)
    make_plot(agent_goals, agent_stats, all_goals, args.output_dir)


if __name__ == "__main__":
    main()
