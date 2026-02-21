"""
Analyzes whether SC-BDI is simply always predicting the same node (i.e. the
agent's most-frequent destination) rather than making trajectory-informed
predictions.

For each agent, across all their test trajectories at every proportion, we
record what node the model predicted and check:
  1. How often the model predicts the exact same node every time (per agent)
  2. Whether that repeated prediction is the agent's actual top-1 destination
  3. The model's effective prediction entropy vs. the data entropy

Usage:
    python experiments/analyze_model_prediction_diversity.py
    python experiments/analyze_model_prediction_diversity.py \
        --checkpoint checkpoints/keepers/sc_bdi_progress_v2.pt \
        --data_path  data/simulation_data/run_8/trajectories/all_trajectories.json \
        --output_dir experiments/results/exp_1
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Make sure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_controller.world_graph import WorldGraph
from models.baseline_transformer.baseline_transformer_model import PerNodeTransformerPredictor
from models.baseline_transformer.baseline_transformer_dataset import TransformerTrajectoryDataset, collate_transformer_trajectories
from models.new_bdi.bdi_dataset_v3 import BDIVAEDatasetV3, collate_bdi_samples_v3
from models.new_bdi.bdi_vae_v3_model import create_sc_bdi_vae_v3


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_graph_and_pois(graph_path: str):
    G = nx.read_graphml(graph_path)
    wg = WorldGraph(G)
    poi_nodes = wg.poi_nodes
    return G, poi_nodes


def load_trajectories(data_path: str):
    with open(data_path) as f:
        data = json.load(f)
    # Build agent key → integer index mapping (sorted, same as training)
    agent_keys = sorted(data.keys())
    agent_key_to_id = {k: i for i, k in enumerate(agent_keys)}
    trajs = []
    for agent_key in agent_keys:
        for traj in data[agent_key]:
            traj = dict(traj)
            traj["agent_id"] = agent_key_to_id[agent_key]  # int, not string
            traj["agent_key"] = agent_key                   # keep string for reference
            trajs.append(traj)
    return trajs


def load_split(split_path: str, trajectories: list, split: str = "test"):
    with open(split_path) as f:
        splits = json.load(f)
    # Keys are stored as e.g. 'test_indices', 'train_indices', 'val_indices'
    key = f"{split}_indices"
    indices = splits[key]
    return [trajectories[i] for i in indices]


def get_agent_top1_destinations(trajectories: list) -> dict:
    """Return the most-visited goal node for each agent across all trajectories."""
    agent_goals = defaultdict(list)
    for traj in trajectories:
        path = traj["path"]
        goal = traj.get("goal_node", path[-1])
        if isinstance(goal, (list, tuple)):
            goal = goal[0]
        agent_goals[traj["agent_id"]].append(goal)
    return {
        agent: Counter(goals).most_common(1)[0][0]
        for agent, goals in agent_goals.items()
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, num_nodes: int, num_poi_nodes: int,
               num_agents: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    has_progress = any(
        "progress" in k for k in ckpt["model_state_dict"].keys()
    )
    print(f"  use_progress={has_progress}")
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
        use_progress=has_progress,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, has_progress


def load_transformer(checkpoint_path: str, num_nodes: int, num_poi_nodes: int,
                     device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, trajectories: list, graph: nx.Graph,
                        poi_nodes: list, device: torch.device,
                        proportion: float, batch_size: int = 64,
                        use_progress: bool = True,
                        model_type: str = "sc_bdi_vae"):
    """
    Run the model on every trajectory at `proportion` and return, per sample:
        agent_id, predicted_poi_idx, true_goal_idx
    """
    # Truncate trajectories
    truncated = []
    for traj in trajectories:
        path = traj["path"]
        if len(path) < 2:
            continue
        cut = max(1, int(len(path) * proportion))
        t = dict(traj)
        t["path"] = path[:cut]
        truncated.append(t)

    if model_type == "transformer":
        return _collect_transformer_predictions(
            model, truncated, graph, poi_nodes, device, proportion, batch_size
        )

    # --- SC-BDI path ---
    dataset = BDIVAEDatasetV3(
        trajectories=truncated,
        graph=graph,
        poi_nodes=poi_nodes,
        min_traj_length=1,
        include_progress=True,
    )
    if len(dataset) == 0:
        return []

    # Last sample per trajectory only
    eval_indices = [
        sample_indices[-1]
        for sample_indices in dataset.trajectory_samples.values()
        if sample_indices
    ]
    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_bdi_samples_v3,
        num_workers=0,
    )

    records = []

    for batch in loader:
        history_node_indices = batch["history_node_indices"].to(device)
        history_lengths      = batch["history_lengths"].to(device)
        agent_ids            = batch["agent_id"].to(device)
        goal_idx             = batch["goal_idx"]          # keep on CPU

        # Correct path progress
        path_progress = torch.full(
            (history_node_indices.shape[0],), proportion,
            dtype=torch.float, device=device,
        )

        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_ids,
            path_progress=path_progress if use_progress else None,
            compute_loss=False,
        )
        preds = outputs["goal"].argmax(dim=1).cpu()  # (B,)

        for i in range(len(preds)):
            records.append({
                "agent_id":  agent_ids[i].item(),
                "pred_idx":  preds[i].item(),
                "goal_idx":  goal_idx[i].item(),
            })

    return records


@torch.no_grad()
def _collect_transformer_predictions(model, truncated: list, graph: nx.Graph,
                                     poi_nodes: list, device: torch.device,
                                     proportion: float, batch_size: int):
    dataset = TransformerTrajectoryDataset(truncated, graph, poi_nodes)
    if len(dataset) == 0:
        return []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transformer_trajectories,
        num_workers=0,
    )
    records = []
    for batch in loader:
        node_indices  = batch["node_indices"].to(device)
        agent_ids     = batch["agent_ids"].to(device)
        hours         = batch["hours"].to(device)
        padding_mask  = batch["padding_mask"].to(device)
        goal_idx      = batch["goal_idx"]  # CPU

        preds_dict = model(node_indices, agent_ids, hours, padding_mask)
        # Transformer returns per-node logits (B, seq, num_poi); take last valid step
        goal_logits = preds_dict["goal"]  # (B, seq, num_poi)
        # padding_mask: True=padding, False=valid → (~mask).sum() = valid length
        lengths = (~padding_mask).sum(dim=1).long()  # (B,) number of valid tokens
        last_logits = goal_logits[
            torch.arange(goal_logits.size(0)), (lengths - 1).clamp(min=0)
        ]  # (B, num_poi)
        preds = last_logits.argmax(dim=1).cpu()

        for i in range(len(preds)):
            records.append({
                "agent_id": agent_ids[i].item(),
                "pred_idx": preds[i].item(),
                "goal_idx": goal_idx[i].item(),
            })
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(records: list, agent_id_to_key: dict,
            agent_top1_idx: dict) -> dict:
    """
    Per-agent: how often does the model predict the same node?
    Is that node the agent's true top-1?
    """
    agent_preds  = defaultdict(list)
    agent_goals  = defaultdict(list)

    for r in records:
        aid = r["agent_id"]
        agent_preds[aid].append(r["pred_idx"])
        agent_goals[aid].append(r["goal_idx"])

    results = {}
    for aid, preds in agent_preds.items():
        n = len(preds)
        counter = Counter(preds)
        top_pred, top_count = counter.most_common(1)[0]

        # Fraction of times model outputs its own most-common prediction
        repetition_rate = top_count / n * 100

        # Is that prediction equal to the agent's true top-1 destination?
        true_top1 = agent_top1_idx.get(aid, -1)
        is_top1   = (top_pred == true_top1)

        # Model prediction entropy
        probs = np.array(list(counter.values())) / n
        pred_entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Accuracy on this agent
        goals = agent_goals[aid]
        correct = sum(p == g for p, g in zip(preds, goals))
        accuracy = correct / n * 100

        results[aid] = {
            "repetition_rate": repetition_rate,
            "is_predicting_true_top1": is_top1,
            "n_unique_preds": len(counter),
            "pred_entropy": pred_entropy,
            "accuracy": accuracy,
        }

    return results


def print_analysis(per_agent: dict, proportion: float):
    rep_rates  = [v["repetition_rate"]           for v in per_agent.values()]
    is_top1    = [v["is_predicting_true_top1"]   for v in per_agent.values()]
    n_unique   = [v["n_unique_preds"]             for v in per_agent.values()]
    entropies  = [v["pred_entropy"]               for v in per_agent.values()]

    print(f"\n  Proportion = {int(proportion*100)}%")
    print(f"    Avg repetition rate (model's top pred):  {np.mean(rep_rates):.1f}%")
    print(f"    Agents where top pred == true top-1:     {sum(is_top1)}/{len(is_top1)} ({sum(is_top1)/len(is_top1)*100:.0f}%)")
    print(f"    Avg unique predictions per agent:        {np.mean(n_unique):.1f}")
    print(f"    Avg model prediction entropy:            {np.mean(entropies):.2f} bits")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(all_per_agent: dict, proportions: list, output_dir: str,
              model_name: str = "sc_bdi_progress_v2"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    prop_labels = [f"{int(p*100)}%" for p in proportions]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(proportions)))

    # (A) Repetition rate per proportion (violin)
    ax = axes[0, 0]
    data_rep = [[v["repetition_rate"] for v in all_per_agent[p].values()]
                for p in proportions]
    vp = ax.violinplot(data_rep, positions=range(len(proportions)),
                       showmedians=True, showextrema=True)
    for body in vp["bodies"]:
        body.set_alpha(0.7)
    ax.set_xticks(range(len(proportions)))
    ax.set_xticklabels(prop_labels)
    ax.set_xlabel("Trajectory Observed (%)", fontsize=11)
    ax.set_ylabel("Repetition rate (%)", fontsize=11)
    ax.set_title("(A) How often does the model predict\nthe same node for each agent?", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # (B) % agents whose top prediction == their true top-1 destination
    ax = axes[0, 1]
    pct_top1 = [
        sum(v["is_predicting_true_top1"] for v in all_per_agent[p].values())
        / len(all_per_agent[p]) * 100
        for p in proportions
    ]
    ax.bar(prop_labels, pct_top1, color="#4878D0", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Trajectory Observed (%)", fontsize=11)
    ax.set_ylabel("% of agents", fontsize=11)
    ax.set_title("(B) Agents where model's most-predicted node\n== agent's true top-1 destination", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    for i, v in enumerate(pct_top1):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (C) Number of unique predictions per agent per proportion
    ax = axes[1, 0]
    data_uniq = [[v["n_unique_preds"] for v in all_per_agent[p].values()]
                 for p in proportions]
    bp = ax.boxplot(data_uniq, labels=prop_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Trajectory Observed (%)", fontsize=11)
    ax.set_ylabel("# unique predicted nodes per agent", fontsize=11)
    ax.set_title("(C) Diversity of model predictions per agent\n(1 = always same node)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (D) Model prediction entropy vs proportion
    ax = axes[1, 1]
    mean_ent = [np.mean([v["pred_entropy"] for v in all_per_agent[p].values()])
                for p in proportions]
    std_ent  = [np.std([v["pred_entropy"]  for v in all_per_agent[p].values()])
                for p in proportions]
    ax.errorbar(prop_labels, mean_ent, yerr=std_ent,
                fmt="o-", color="#D65F5F", linewidth=2, markersize=7,
                capsize=5, label="Model prediction entropy")
    ax.set_xlabel("Trajectory Observed (%)", fontsize=11)
    ax.set_ylabel("Prediction entropy (bits)", fontsize=11)
    ax.set_title("(D) Model prediction entropy per proportion\n(0 = always same node, higher = more diverse)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Prediction Diversity Analysis: {model_name}\n"
        "Is the model always predicting the same destination for each agent?",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = Path(output_dir) / f"{model_name}_prediction_diversity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/keepers/sc_bdi_progress_v2.pt")
    parser.add_argument("--model_type",   default="sc_bdi_vae", choices=["sc_bdi_vae", "transformer"])
    parser.add_argument("--model_name",   default="sc_bdi_progress_v2",
                        help="Name used in plot title and output filename")
    parser.add_argument("--data_path",    default="data/simulation_data/run_8/trajectories/all_trajectories.json")
    parser.add_argument("--split_path",   default="data/simulation_data/run_8/split_data/split_indices_seed42.json")
    parser.add_argument("--graph_path",   default="data/processed/ucsd_walk_full.graphml")
    parser.add_argument("--output_dir",   default="experiments/results/model_analysis")
    parser.add_argument("--split",        default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size",   type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading graph and POI nodes...")
    graph, poi_nodes = load_graph_and_pois(args.graph_path)
    print(f"  Nodes: {len(graph.nodes())}  POI nodes: {len(poi_nodes)}")

    print("Loading trajectories...")
    all_trajs = load_trajectories(args.data_path)
    trajs = load_split(args.split_path, all_trajs, split=args.split)
    print(f"  {args.split} trajectories: {len(trajs)}")

    # Agent index mapping — agent_id is already an int (0–99) from load_trajectories
    num_agents = len({t["agent_id"] for t in all_trajs})
    agent_id_to_key = {t["agent_id"]: t.get("agent_key", f"agent_{t['agent_id']:03d}")
                       for t in all_trajs}

    # True top-1 destination per agent (by integer poi index)
    poi_to_idx = {n: i for i, n in enumerate(poi_nodes)}
    agent_top1_raw = get_agent_top1_destinations(all_trajs)  # {int_id: goal_node_str}
    agent_top1_idx = {
        k: poi_to_idx.get(v, -1)
        for k, v in agent_top1_raw.items()
    }

    print(f"\nLoading model from {args.checkpoint} (type={args.model_type})...")
    use_progress = False
    if args.model_type == "sc_bdi_vae":
        model, use_progress = load_model(
            args.checkpoint, len(graph.nodes()), len(poi_nodes), num_agents, device
        )
    else:
        model = load_transformer(
            args.checkpoint, len(graph.nodes()), len(poi_nodes), device
        )

    proportions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    all_per_agent = {}

    print("\n" + "=" * 60)
    print(f"PREDICTION DIVERSITY ANALYSIS — {args.model_name}")
    print("=" * 60)

    for proportion in proportions:
        records = collect_predictions(
            model, trajs, graph, poi_nodes, device,
            proportion=proportion, batch_size=args.batch_size,
            use_progress=use_progress,
            model_type=args.model_type,
        )
        per_agent = analyze(records, agent_id_to_key, agent_top1_idx)
        all_per_agent[proportion] = per_agent
        print_analysis(per_agent, proportion)

    make_plot(all_per_agent, proportions, args.output_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
