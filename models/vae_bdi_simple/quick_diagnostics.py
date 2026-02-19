#!/usr/bin/env python3
"""
QUICK DIAGNOSTICS FOR SC-BDI V3 MODEL

Run this script to:
1. Understand why accuracy appears "low"
2. Visualize latent spaces
3. Check if the model is learning

Usage:
    python models/vae_bdi_simple/quick_diagnostics.py [--checkpoint PATH]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.new_bdi.bdi_vae_v3_model import create_sc_bdi_vae_v3
from models.new_bdi.bdi_dataset_v2 import BDIVAEDatasetV2, collate_bdi_samples_v2
from models.utils.utils import get_device
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    VISUALIZATION_OK = True
except ImportError:
    VISUALIZATION_OK = False
    print("⚠️ matplotlib/sklearn not available. Install for visualizations.")


def explain_accuracy(num_goals: int, accuracy: float):
    """Explain what the accuracy means in context."""
    random_acc = 100.0 / num_goals
    improvement = accuracy / random_acc if random_acc > 0 else 0
    
    print("\n" + "=" * 70)
    print("📊 ACCURACY EXPLAINED")
    print("=" * 70)
    print(f"\n🎯 Your Setup:")
    print(f"   • Number of possible goals (POIs): {num_goals}")
    print(f"   • Random chance accuracy: {random_acc:.2f}%")
    print(f"   • Current model accuracy: {accuracy:.2f}%")
    print(f"   • Improvement over random: {improvement:.1f}x")
    
    print(f"\n📈 What This Means:")
    if improvement < 1.5:
        print("   ⚠️  Model is near random. This is NORMAL for:")
        print("      - Early training (epochs 1-5)")
        print("      - When KL weight is still annealing")
    elif improvement < 5:
        print("   🔵 Model is learning! It's identifying patterns.")
        print("      - Keep training for more epochs")
        print("      - InfoNCE loss should be decreasing")
    elif improvement < 20:
        print("   ✅ Good performance! Model is significantly better than random.")
    else:
        print("   🎉 Excellent! Model has strong predictive power.")
    
    print(f"\n💡 Key Insight:")
    print(f"   With {num_goals} goals, even 5% accuracy means the model is")
    print(f"   correctly ranking the true goal in the top ~{int(5 * num_goals / 100)} most likely!")
    
    # Top-K analysis
    for k in [5, 10, 20]:
        topk_random = min(100.0, k * 100.0 / num_goals)
        print(f"   • Top-{k} random chance: {topk_random:.1f}%")
    print("=" * 70)


@torch.no_grad()
def compute_topk_accuracy(
    model,
    dataloader,
    device,
    ks=[1, 5, 10, 20],
    max_batches=50,
):
    """Compute top-k accuracy to show the model is learning."""
    model.eval()
    
    correct = {k: 0 for k in ks}
    total = 0
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        agent_id = batch['agent_id'].to(device)
        path_progress = batch['path_progress'].to(device)
        
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            path_progress=path_progress,
            compute_loss=False,
        )
        
        goal_logits = outputs['goal']
        batch_size = goal_logits.shape[0]
        total += batch_size
        
        for k in ks:
            topk_preds = goal_logits.topk(k, dim=1).indices
            correct[k] += (topk_preds == goal_idx.unsqueeze(1)).any(dim=1).sum().item()
    
    accuracies = {k: 100.0 * correct[k] / total for k in ks}
    return accuracies


@torch.no_grad()
def analyze_predictions(
    model,
    dataloader,
    device,
    max_batches=20,
):
    """Analyze what the model is predicting."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_progress = []
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        agent_id = batch['agent_id'].to(device)
        path_progress = batch['path_progress'].to(device)
        
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_id,
            path_progress=path_progress,
            compute_loss=False,
        )
        
        probs = F.softmax(outputs['goal'], dim=1)
        preds = probs.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(goal_idx.cpu())
        all_probs.append(probs.max(dim=1).values.cpu())
        all_progress.append(path_progress.cpu())
    
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = torch.cat(all_probs).numpy()
    progress = torch.cat(all_progress).numpy()
    
    return preds, targets, probs, progress


def visualize_latent_spaces(model, dataloader, device, save_path):
    """Quick latent space visualization."""
    if not VISUALIZATION_OK:
        print("⚠️ Skipping visualization (matplotlib not available)")
        return
    
    model.eval()
    
    # Collect embeddings
    belief_z_all = []
    desire_z_all = []
    intention_z_all = []
    goals_all = []
    progress_all = []
    
    max_samples = 2000
    collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if collected >= max_samples:
                break
            
            batch_size = batch['history_node_indices'].size(0)
            
            history_node_indices = batch['history_node_indices'].to(device)
            history_lengths = batch['history_lengths'].to(device)
            agent_id = batch['agent_id'].to(device)
            path_progress = batch['path_progress'].to(device)
            goal_idx = batch['goal_idx']
            
            outputs = model(
                history_node_indices=history_node_indices,
                history_lengths=history_lengths,
                agent_ids=agent_id,
                path_progress=path_progress,
                compute_loss=False,
            )
            
            belief_z_all.append(outputs['belief_z'].cpu().numpy())
            desire_z_all.append(outputs['desire_z'].cpu().numpy())
            intention_z_all.append(outputs['intention_z'].cpu().numpy())
            goals_all.append(goal_idx.numpy())
            progress_all.append(path_progress.cpu().numpy())
            
            collected += batch_size
    
    belief_z = np.concatenate(belief_z_all, axis=0)[:max_samples]
    desire_z = np.concatenate(desire_z_all, axis=0)[:max_samples]
    intention_z = np.concatenate(intention_z_all, axis=0)[:max_samples]
    goals = np.concatenate(goals_all, axis=0)[:max_samples]
    progress = np.concatenate(progress_all, axis=0)[:max_samples]
    
    # Get top-10 most frequent goals for coloring
    goal_counts = np.bincount(goals.astype(int))
    top10_goals = np.argsort(goal_counts)[-10:]
    goal_colors = np.full(len(goals), -1)
    for i, g in enumerate(top10_goals):
        goal_colors[goals == g] = i
    
    # Dimensionality reduction with PCA (faster than t-SNE)
    print("   Computing PCA projections...")
    pca = PCA(n_components=2)
    desire_2d = pca.fit_transform(desire_z)
    intention_2d = pca.fit_transform(intention_z)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Desire latent colored by top-10 goals
    ax = axes[0, 0]
    mask_other = goal_colors == -1
    ax.scatter(desire_2d[mask_other, 0], desire_2d[mask_other, 1],
               c='lightgray', alpha=0.2, s=3, label='Other goals')
    for i in range(10):
        mask = goal_colors == i
        ax.scatter(desire_2d[mask, 0], desire_2d[mask, 1],
                   alpha=0.6, s=8, label=f'Goal {top10_goals[i]}')
    ax.set_title('Desire Latent (z_d) by Top-10 Goals')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    
    # 2. Desire latent colored by progress
    ax = axes[0, 1]
    scatter = ax.scatter(desire_2d[:, 0], desire_2d[:, 1],
                         c=progress, cmap='viridis', alpha=0.5, s=5)
    plt.colorbar(scatter, ax=ax, label='Path Progress')
    ax.set_title('Desire Latent (z_d) by Progress')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    
    # 3. Intention latent colored by goals
    ax = axes[1, 0]
    ax.scatter(intention_2d[mask_other, 0], intention_2d[mask_other, 1],
               c='lightgray', alpha=0.2, s=3)
    for i in range(10):
        mask = goal_colors == i
        ax.scatter(intention_2d[mask, 0], intention_2d[mask, 1],
                   alpha=0.6, s=8)
    ax.set_title('Intention Latent (z_i) by Top-10 Goals')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    
    # 4. Intention latent colored by progress
    ax = axes[1, 1]
    scatter = ax.scatter(intention_2d[:, 0], intention_2d[:, 1],
                         c=progress, cmap='viridis', alpha=0.5, s=5)
    plt.colorbar(scatter, ax=ax, label='Path Progress')
    ax.set_title('Intention Latent (z_i) by Progress')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Quick diagnostics for SC-BDI V3")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sc_bdi_v3/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                        default='data/simulation_data/run_8/trajectories')
    parser.add_argument('--graph_path', type=str,
                        default='data/processed/ucsd_walk_full.graphml')
    parser.add_argument('--split_path', type=str,
                        default='data/simulation_data/run_8/split_data/split_indices_seed42.json')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--output_dir', type=str, default='artifacts/diagnostics')
    args = parser.parse_args()
    
    device = get_device()
    print(f"\n🖥️  Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"\n⚠️  Checkpoint not found: {args.checkpoint}")
        print("   Run some training first, or specify --checkpoint PATH")
        
        # Still explain accuracy
        explain_accuracy(num_goals=735, accuracy=2.4)  # Example values
        return
    
    # Load data
    print("\n📂 Loading data...")
    import networkx as nx
    
    graph = nx.read_graphml(args.graph_path)
    
    with open(Path(args.data_dir) / 'all_trajectories.json') as f:
        traj_data = json.load(f)
    
    # Flatten trajectories
    trajectories = []
    if isinstance(traj_data, dict):
        sorted_agents = sorted(traj_data.keys())
        for agent_idx, agent_key in enumerate(sorted_agents):
            for traj in traj_data[agent_key]:
                traj['agent_id'] = agent_idx
                trajectories.append(traj)
    else:
        trajectories = traj_data
    
    # Get POI nodes using WorldGraph — same as simple BDI-VAE (230 nodes)
    from graph_controller.world_graph import WorldGraph
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    
    # Load split indices
    with open(args.split_path) as f:
        splits = json.load(f)
    val_indices = splits['val_indices'][:5000]  # Subset for speed
    val_trajs = [trajectories[i] for i in val_indices if i < len(trajectories)]
    
    # Create dataset
    val_dataset = BDIVAEDatasetV2(
        trajectories=val_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=node_to_idx,
        include_progress=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_bdi_samples_v2,
    )
    
    # Load model
    print("\n🏗️  Loading model...")
    num_nodes = graph.number_of_nodes()
    num_agents = len(set(t.get('agent_id', 0) for t in trajectories))
    num_poi_nodes = len(poi_nodes)
    num_categories = len(val_dataset.CATEGORY_TO_IDX)
    
    model = create_sc_bdi_vae_v3(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=num_categories,
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"   ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    
    # Top-K accuracy
    topk_acc = compute_topk_accuracy(model, val_loader, device, ks=[1, 5, 10, 20, 50])
    
    print("\n" + "=" * 70)
    print("📈 TOP-K ACCURACY")
    print("=" * 70)
    for k, acc in topk_acc.items():
        random_topk = min(100.0, k * 100.0 / num_poi_nodes)
        improvement = acc / random_topk if random_topk > 0 else 0
        print(f"   Top-{k:2d}: {acc:5.1f}% (random: {random_topk:.1f}%, {improvement:.1f}x better)")
    print("=" * 70)
    
    # Explain the accuracy
    explain_accuracy(num_goals=num_poi_nodes, accuracy=topk_acc[1])
    
    # Analyze predictions by progress
    preds, targets, probs, progress = analyze_predictions(model, val_loader, device)
    
    print("\n" + "=" * 70)
    print("📊 ACCURACY BY PATH PROGRESS")
    print("=" * 70)
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    for low, high in bins:
        mask = (progress >= low) & (progress < high)
        if mask.sum() > 0:
            acc = 100.0 * (preds[mask] == targets[mask]).mean()
            print(f"   {int(low*100):3d}-{int(high*100):3d}%: {acc:.1f}% accuracy ({mask.sum()} samples)")
    print("=" * 70)
    
    # Visualization
    if VISUALIZATION_OK:
        os.makedirs(args.output_dir, exist_ok=True)
        viz_path = os.path.join(args.output_dir, 'quick_latent_viz.png')
        print(f"\n🎨 Creating visualization...")
        visualize_latent_spaces(model, val_loader, device, viz_path)
    
    print("\n✅ Diagnostics complete!")


if __name__ == '__main__':
    main()
