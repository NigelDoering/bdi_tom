#!/usr/bin/env python3
"""
Publication-Quality BDI Latent Space Visualizations
====================================================

This script generates clear, interpretable, publication-ready visualizations
of the Belief-Desire-Intention (BDI) latent spaces learned by the SC-BDI VAE.

WHAT EACH VISUALIZATION SHOWS:
------------------------------

1. **Latent Space Overview (PCA/t-SNE)**
   - Shows how samples cluster in the learned latent space
   - Good separation = model learned meaningful representations
   - Colors indicate different goals, categories, or agents
   
2. **BDI Component Comparison**
   - Side-by-side view of Belief, Desire, and Intention spaces
   - Each component should capture different aspects:
     * Belief (z_b): Where the agent thinks they are, spatial understanding
     * Desire (z_d): What the agent wants (goal preferences)
     * Intention (z_i): Current action plan (combines B + D)

3. **Progress-Stratified Accuracy**
   - Shows how prediction accuracy changes along trajectory
   - Key insight: Accuracy should INCREASE as agent gets closer to goal
   - This proves the model learns goal-directed behavior

4. **Goal Clustering Analysis**  
   - Do similar goals cluster together in latent space?
   - Good clustering = model learned goal semantics

5. **Agent Behavior Patterns**
   - Do different agents show distinct patterns?
   - Reveals individual preference learning

6. **Category Organization**
   - Are POI categories (food, study, leisure, etc.) separated?
   - Shows semantic understanding of destination types

Author: BDI-ToM Project
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vae_bdi_simple.bdi_dataset_v2 import BDIVAEDatasetV2, collate_bdi_samples_v2
from models.vae_bdi_simple.bdi_vae_v3_model import create_sc_bdi_vae_v3


# =============================================================================
# STYLE CONFIGURATION - Publication Quality
# =============================================================================

# Set up publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Custom color palettes
CATEGORY_COLORS = {
    'food': '#E74C3C',       # Red
    'study': '#3498DB',      # Blue  
    'leisure': '#2ECC71',    # Green
    'health': '#9B59B6',     # Purple
    'errands': '#F39C12',    # Orange
    'home': '#1ABC9C',       # Teal
    'other': '#95A5A6',      # Gray
}

# Colorblind-friendly palette for goals (using ColorBrewer)
GOAL_CMAP = plt.cm.tab20
AGENT_CMAP = plt.cm.Set3
PROGRESS_CMAP = 'viridis'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_model_and_data(
    checkpoint_path: str,
    graph_path: str = "data/processed/ucsd_walk_full.graphml",
    data_dir: str = "data/simulation_data/run_8/trajectories",
    split: str = "val",
    max_samples: int = 50000,  # Limit for visualization (memory)
    device: str = "cuda",
) -> Tuple[torch.nn.Module, Dict[str, np.ndarray], Dict]:
    """Load model and collect latent representations."""
    
    print("=" * 60)
    print("LOADING MODEL AND DATA")
    print("=" * 60)
    
    # Load graph
    print(f"\n📂 Loading graph from {graph_path}...")
    graph = nx.read_graphml(graph_path)
    print(f"   ✓ {graph.number_of_nodes()} nodes")
    
    # Build node mapping
    node_to_idx = {n: i for i, n in enumerate(sorted(graph.nodes()))}
    
    # Load trajectories
    traj_path = Path(data_dir) / "all_trajectories.json"
    print(f"📂 Loading trajectories from {traj_path}...")
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    # Parse trajectories
    trajectories = []
    sorted_agents = sorted(traj_data.keys())
    for agent_idx, agent_key in enumerate(sorted_agents):
        for traj in traj_data[agent_key]:
            traj['agent_id'] = agent_idx
            trajectories.append(traj)
    print(f"   ✓ {len(trajectories)} trajectories from {len(sorted_agents)} agents")
    
    # Get POI nodes using WorldGraph — same as simple BDI-VAE (230 nodes)
    from graph_controller.world_graph import WorldGraph
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    print(f"   ✓ {len(poi_nodes)} POI nodes (categories: {world_graph.relevant_categories})")
    
    # Load split indices
    split_path = "data/processed/split_indices.json"
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            splits = json.load(f)
        if split == 'train':
            indices = splits['train']
        elif split == 'val':
            indices = splits['val']
        else:
            indices = list(range(len(trajectories)))
    else:
        n = len(trajectories)
        indices = list(range(int(0.8*n), n)) if split == 'val' else list(range(int(0.8*n)))
    
    split_trajs = [trajectories[i] for i in indices]
    print(f"   ✓ Using {len(split_trajs)} {split} trajectories")
    
    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = BDIVAEDatasetV2(
        trajectories=split_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=node_to_idx,
        include_progress=True,
    )
    print(f"   ✓ {len(dataset)} samples")
    
    # Get metadata
    num_nodes = len(node_to_idx)
    num_agents = len(sorted_agents)
    num_poi_nodes = len(dataset.goal_to_idx)
    categories = set()
    for n, d in graph.nodes(data=True):
        cat = d.get('clean_categories')
        if cat and cat not in [None, '', 'None']:
            categories.add(cat)
    num_categories = max(7, len(categories))
    
    # Build reverse mappings for interpretability
    idx_to_goal = {v: k for k, v in dataset.goal_to_idx.items()}
    
    # Get category for each goal
    goal_to_category = {}
    for n, d in graph.nodes(data=True):
        if n in dataset.goal_to_idx:
            cat = d.get('clean_categories', 'other')
            if cat in [None, '', 'None']:
                cat = 'other'
            goal_to_category[n] = cat
    
    metadata = {
        'num_nodes': num_nodes,
        'num_agents': num_agents,
        'num_poi_nodes': num_poi_nodes,
        'num_categories': num_categories,
        'idx_to_goal': idx_to_goal,
        'goal_to_category': goal_to_category,
        'categories': list(categories),
    }
    
    # Load model
    print("\n🏗️  Loading model...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = create_sc_bdi_vae_v3(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=num_categories,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✓ Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"   ✓ Val accuracy: {checkpoint.get('val_metric', '?'):.2f}%")
    
    # Collect latents (subsample if too large)
    print("\n🔍 Collecting latent representations...")
    
    # Subsample for visualization
    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"   ⚠ Subsampled to {max_samples} for visualization")
    
    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0,
        collate_fn=collate_bdi_samples_v2,
    )
    
    all_data = {
        'z_b': [], 'z_d': [], 'z_i': [],
        'goal_ids': [], 'category_ids': [], 'agent_ids': [],
        'progress': [], 'goal_logits': [],
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="   Collecting"):
            outputs = model(
                history_node_indices=batch['history_node_indices'].to(device),
                history_lengths=batch['history_lengths'].to(device),
                agent_ids=batch['agent_id'].to(device),
                path_progress=batch['path_progress'].to(device),
                goal_idx=batch['goal_idx'].to(device),
                goal_cat_idx=batch['goal_cat_idx'].to(device),
                compute_loss=False,
            )
            
            all_data['z_b'].append(outputs['belief_z'].cpu().numpy())
            all_data['z_d'].append(outputs['desire_z'].cpu().numpy())
            all_data['z_i'].append(outputs['intention_z'].cpu().numpy())
            all_data['goal_logits'].append(outputs['goal'].cpu().numpy())
            all_data['goal_ids'].append(batch['goal_idx'].numpy())
            all_data['category_ids'].append(batch['goal_cat_idx'].numpy())
            all_data['agent_ids'].append(batch['agent_id'].numpy())
            all_data['progress'].append(batch['path_progress'].numpy())
    
    # Concatenate
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)
    
    print(f"\n   ✓ Collected {len(all_data['z_b'])} samples")
    print(f"   ✓ Latent dim: {all_data['z_b'].shape[1]}")
    print(f"   ✓ Unique goals: {len(np.unique(all_data['goal_ids']))}")
    
    return model, all_data, metadata


# =============================================================================
# VISUALIZATION 1: BDI Component Comparison (Main Figure)
# =============================================================================

def plot_bdi_comparison(
    data: Dict[str, np.ndarray],
    metadata: Dict,
    output_path: str,
    method: str = 'pca',
    n_samples: int = 10000,
) -> None:
    """
    Create the main BDI comparison figure showing all three latent spaces.
    
    INTERPRETATION:
    - Each column shows one BDI component (Belief, Desire, Intention)
    - Top row: colored by goal category (food, study, leisure, etc.)
    - Bottom row: colored by trajectory progress (0% = start, 100% = near goal)
    
    WHAT TO LOOK FOR:
    - Desire (z_d) should show the clearest category separation (it encodes goals)
    - Belief (z_b) might show spatial patterns (where agent is)
    - Intention (z_i) combines both, should show goal-directed structure
    - Progress coloring should show gradients in Intention space
    """
    
    print(f"\n{'='*60}")
    print("GENERATING BDI COMPONENT COMPARISON")
    print(f"{'='*60}")
    
    # Subsample for cleaner visualization
    if len(data['z_b']) > n_samples:
        idx = np.random.choice(len(data['z_b']), n_samples, replace=False)
    else:
        idx = np.arange(len(data['z_b']))
    
    z_b = data['z_b'][idx]
    z_d = data['z_d'][idx]
    z_i = data['z_i'][idx]
    categories = data['category_ids'][idx]
    progress = data['progress'][idx]
    
    # Compute embeddings
    print(f"   Computing {method.upper()} embeddings...")
    
    if method == 'pca':
        reducer_b = PCA(n_components=2, random_state=42)
        reducer_d = PCA(n_components=2, random_state=42)
        reducer_i = PCA(n_components=2, random_state=42)
        
        emb_b = reducer_b.fit_transform(z_b)
        emb_d = reducer_d.fit_transform(z_d)
        emb_i = reducer_i.fit_transform(z_i)
        
        var_b = reducer_b.explained_variance_ratio_.sum() * 100
        var_d = reducer_d.explained_variance_ratio_.sum() * 100
        var_i = reducer_i.explained_variance_ratio_.sum() * 100
    else:  # tsne
        perplexity = min(30, len(z_b) // 4)
        
        reducer_b = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, init='pca')
        reducer_d = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, init='pca')
        reducer_i = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, init='pca')
        
        print("      Computing t-SNE for Belief...")
        emb_b = reducer_b.fit_transform(z_b)
        print("      Computing t-SNE for Desire...")
        emb_d = reducer_d.fit_transform(z_d)
        print("      Computing t-SNE for Intention...")
        emb_i = reducer_i.fit_transform(z_i)
        
        var_b = var_d = var_i = None
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2)
    
    # Category names for legend
    cat_names = ['food', 'study', 'leisure', 'health', 'errands', 'home', 'other']
    cat_colors = [CATEGORY_COLORS.get(c, '#95A5A6') for c in cat_names]
    
    # Top row: colored by category
    embeddings = [emb_b, emb_d, emb_i]
    titles = ['Belief (z_b)', 'Desire (z_d)', 'Intention (z_i)']
    variances = [var_b, var_d, var_i]
    
    for col, (emb, title, var) in enumerate(zip(embeddings, titles, variances)):
        ax = fig.add_subplot(gs[0, col])
        
        # Plot each category
        for cat_idx, cat_name in enumerate(cat_names):
            mask = categories == cat_idx
            if mask.sum() > 0:
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=CATEGORY_COLORS.get(cat_name, '#95A5A6'),
                    alpha=0.5, s=8, label=cat_name.capitalize(),
                    edgecolors='none',
                )
        
        subtitle = f" ({var:.1f}% var)" if var else ""
        ax.set_title(f"{title}{subtitle}", fontweight='bold')
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        
        if col == 2:  # Add legend to last plot
            ax.legend(loc='upper right', markerscale=2, framealpha=0.9)
    
    # Bottom row: colored by progress
    for col, (emb, title) in enumerate(zip(embeddings, titles)):
        ax = fig.add_subplot(gs[1, col])
        
        scatter = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=progress, cmap=PROGRESS_CMAP,
            alpha=0.6, s=8, edgecolors='none',
            vmin=0, vmax=1,
        )
        
        ax.set_title(f"{title} by Progress", fontweight='bold')
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        
        if col == 2:  # Add colorbar to last plot
            cbar = plt.colorbar(scatter, ax=ax, label='Path Progress')
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['0%\n(Start)', '25%', '50%', '75%', '100%\n(Goal)'])
    
    # Main title
    fig.suptitle(
        'BDI Latent Space Analysis: Category and Progress Structure',
        fontsize=16, fontweight='bold', y=0.98,
    )
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Saved: {output_path}")
    
    return emb_b, emb_d, emb_i


# =============================================================================
# VISUALIZATION 2: Progress-Stratified Accuracy (Key Result!)
# =============================================================================

def plot_progress_accuracy(
    data: Dict[str, np.ndarray],
    metadata: Dict,
    output_path: str,
) -> None:
    """
    Show how prediction accuracy changes with trajectory progress.
    
    INTERPRETATION:
    This is THE KEY PLOT for demonstrating goal-directed behavior!
    
    - X-axis: Path progress (0% = just started, 100% = about to arrive)
    - Y-axis: Goal prediction accuracy
    
    WHAT TO EXPECT:
    - Accuracy should INCREASE as progress increases
    - This proves: "The closer an agent gets to their goal, the more
      confident the model becomes about what that goal is"
    - This is exactly what we'd expect from a good Theory of Mind model!
    
    WHAT'S "GOOD":
    - Clear upward trend
    - Final accuracy (90-100% progress) should be highest
    - Even early predictions (10-20%) being above random (0.14%) is good
    """
    
    print(f"\n{'='*60}")
    print("GENERATING PROGRESS-ACCURACY ANALYSIS")
    print(f"{'='*60}")
    
    # Compute predictions
    goal_logits = data['goal_logits']
    goal_ids = data['goal_ids']
    progress = data['progress']
    
    predictions = np.argmax(goal_logits, axis=1)
    correct = (predictions == goal_ids).astype(float)
    
    # Also compute top-5 accuracy
    top5_preds = np.argsort(goal_logits, axis=1)[:, -5:]
    top5_correct = np.array([goal_ids[i] in top5_preds[i] for i in range(len(goal_ids))]).astype(float)
    
    # Bin by progress
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    top1_accs = []
    top5_accs = []
    counts = []
    
    for i in range(n_bins):
        mask = (progress >= bins[i]) & (progress < bins[i+1])
        if mask.sum() > 0:
            top1_accs.append(correct[mask].mean() * 100)
            top5_accs.append(top5_correct[mask].mean() * 100)
            counts.append(mask.sum())
        else:
            top1_accs.append(0)
            top5_accs.append(0)
            counts.append(0)
    
    # Calculate random baseline
    n_goals = goal_logits.shape[1]
    random_top1 = 100 / n_goals
    random_top5 = min(100, 500 / n_goals)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Accuracy vs Progress
    ax1 = axes[0]
    
    # Plot lines
    ax1.plot(bin_centers * 100, top1_accs, 'o-', color='#3498DB', 
             linewidth=2.5, markersize=10, label='Top-1 Accuracy', zorder=3)
    ax1.plot(bin_centers * 100, top5_accs, 's-', color='#2ECC71',
             linewidth=2.5, markersize=10, label='Top-5 Accuracy', zorder=3)
    
    # Random baselines
    ax1.axhline(y=random_top1, color='#E74C3C', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Random Top-1 ({random_top1:.2f}%)')
    ax1.axhline(y=random_top5, color='#F39C12', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Random Top-5 ({random_top5:.2f}%)')
    
    # Fill area showing improvement over random
    ax1.fill_between(bin_centers * 100, random_top1, top1_accs, 
                     alpha=0.2, color='#3498DB')
    
    ax1.set_xlabel('Path Progress (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Goal Prediction Accuracy vs. Trajectory Progress', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, max(max(top5_accs) * 1.1, 50))
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate(
        f'Final accuracy:\n{top1_accs[-1]:.1f}%',
        xy=(95, top1_accs[-1]), xytext=(75, top1_accs[-1] + 10),
        fontsize=10, ha='center',
        arrowprops=dict(arrowstyle='->', color='gray'),
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8),
    )
    
    # Right plot: Sample distribution
    ax2 = axes[1]
    
    bars = ax2.bar(bin_centers * 100, counts, width=8, color='#9B59B6', 
                   alpha=0.7, edgecolor='white')
    
    ax2.set_xlabel('Path Progress (%)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution Across Progress', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Saved: {output_path}")
    
    # Print summary
    print(f"\n   📊 KEY RESULTS:")
    print(f"      • Early progress (0-20%):  Top-1 = {np.mean(top1_accs[:2]):.1f}%")
    print(f"      • Mid progress (40-60%):   Top-1 = {np.mean(top1_accs[4:6]):.1f}%")
    print(f"      • Late progress (80-100%): Top-1 = {np.mean(top1_accs[8:]):.1f}%")
    print(f"      • Improvement factor: {top1_accs[-1]/max(top1_accs[0], 0.01):.1f}x from start to end")
    print(f"      • vs Random ({random_top1:.2f}%): {top1_accs[-1]/random_top1:.0f}x better at trajectory end")


# =============================================================================
# VISUALIZATION 3: Category Separation Analysis
# =============================================================================

def plot_category_analysis(
    data: Dict[str, np.ndarray],
    metadata: Dict,
    output_path: str,
    n_samples: int = 10000,
) -> None:
    """
    Analyze how well categories are separated in Desire latent space.
    
    INTERPRETATION:
    - Shows clustering quality for each POI category
    - Better separation = model learned category semantics
    - Desire space should show clearest category structure
    """
    
    print(f"\n{'='*60}")
    print("GENERATING CATEGORY ANALYSIS")
    print(f"{'='*60}")
    
    # Subsample
    if len(data['z_d']) > n_samples:
        idx = np.random.choice(len(data['z_d']), n_samples, replace=False)
    else:
        idx = np.arange(len(data['z_d']))
    
    z_d = data['z_d'][idx]
    categories = data['category_ids'][idx]
    
    # Compute PCA
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(z_d)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter plot
    ax1 = axes[0]
    cat_names = ['food', 'study', 'leisure', 'health', 'errands', 'home', 'other']
    
    handles = []
    for cat_idx, cat_name in enumerate(cat_names):
        mask = categories == cat_idx
        if mask.sum() > 0:
            scatter = ax1.scatter(
                emb[mask, 0], emb[mask, 1],
                c=CATEGORY_COLORS.get(cat_name, '#95A5A6'),
                alpha=0.6, s=20, label=f'{cat_name.capitalize()} (n={mask.sum():,})',
                edgecolors='none',
            )
            handles.append(scatter)
            
            # Add centroid
            centroid = emb[mask].mean(axis=0)
            ax1.scatter(centroid[0], centroid[1], c='black', s=100, 
                       marker='x', linewidths=3, zorder=5)
    
    ax1.set_xlabel('PCA 1', fontsize=12)
    ax1.set_ylabel('PCA 2', fontsize=12)
    ax1.set_title('Desire Space (z_d) by Category\n(× = category centroid)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Right: Per-category cluster quality
    ax2 = axes[1]
    
    # Compute silhouette score per category
    unique_cats = np.unique(categories)
    cat_silhouettes = []
    cat_labels = []
    cat_counts = []
    
    for cat_idx in unique_cats:
        if cat_idx < len(cat_names):
            mask = categories == cat_idx
            if mask.sum() > 50:  # Need enough samples
                # Within-category variance
                cat_emb = emb[mask]
                centroid = cat_emb.mean(axis=0)
                within_var = np.mean(np.sum((cat_emb - centroid)**2, axis=1))
                
                # Between-category distance
                other_mask = categories != cat_idx
                other_centroid = emb[other_mask].mean(axis=0)
                between_dist = np.sqrt(np.sum((centroid - other_centroid)**2))
                
                # Ratio (higher = better separation)
                separation = between_dist / (np.sqrt(within_var) + 1e-6)
                
                cat_silhouettes.append(separation)
                cat_labels.append(cat_names[cat_idx].capitalize())
                cat_counts.append(mask.sum())
    
    # Plot bars
    colors = [CATEGORY_COLORS.get(l.lower(), '#95A5A6') for l in cat_labels]
    bars = ax2.barh(cat_labels, cat_silhouettes, color=colors, alpha=0.8, edgecolor='white')
    
    ax2.set_xlabel('Separation Score (higher = better)', fontsize=12)
    ax2.set_title('Category Separation Quality', fontsize=14, fontweight='bold')
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    # Add count labels
    for bar, count in zip(bars, cat_counts):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'n={count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# VISUALIZATION 4: Agent-Specific Patterns
# =============================================================================

def plot_agent_analysis(
    data: Dict[str, np.ndarray],
    metadata: Dict,
    output_path: str,
    n_agents: int = 10,
    n_samples: int = 10000,
) -> None:
    """
    Show how different agents cluster in latent space.
    
    INTERPRETATION:
    - Do agents show distinct behavioral patterns?
    - Different colored clusters = agents have learned different preferences
    """
    
    print(f"\n{'='*60}")
    print("GENERATING AGENT PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    # Get top agents by sample count
    unique_agents, counts = np.unique(data['agent_ids'], return_counts=True)
    top_agents = unique_agents[np.argsort(counts)[-n_agents:]]
    
    # Filter to top agents and subsample
    agent_mask = np.isin(data['agent_ids'], top_agents)
    indices = np.where(agent_mask)[0]
    
    if len(indices) > n_samples:
        indices = np.random.choice(indices, n_samples, replace=False)
    
    z_i = data['z_i'][indices]  # Use Intention space
    agents = data['agent_ids'][indices]
    progress = data['progress'][indices]
    
    # Compute PCA
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(z_i)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: By agent
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    for i, agent_id in enumerate(top_agents):
        mask = agents == agent_id
        ax1.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[colors[i]], alpha=0.5, s=15,
            label=f'Agent {agent_id}',
            edgecolors='none',
        )
    
    ax1.set_xlabel('PCA 1', fontsize=12)
    ax1.set_ylabel('PCA 2', fontsize=12)
    ax1.set_title(f'Intention Space (z_i) by Agent\n(Top {n_agents} agents by sample count)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9, ncol=2, fontsize=8)
    
    # Right: Single agent trajectory
    ax2 = axes[1]
    
    # Pick one agent with many samples
    best_agent = top_agents[0]
    agent_mask = agents == best_agent
    agent_emb = emb[agent_mask]
    agent_progress = progress[agent_mask]
    
    # Sort by progress for trajectory view
    sort_idx = np.argsort(agent_progress)
    agent_emb = agent_emb[sort_idx]
    agent_progress = agent_progress[sort_idx]
    
    scatter = ax2.scatter(
        agent_emb[:, 0], agent_emb[:, 1],
        c=agent_progress, cmap=PROGRESS_CMAP,
        alpha=0.7, s=30, edgecolors='none',
    )
    
    # Draw trajectory arrows (sample every N points)
    step = max(1, len(agent_emb) // 20)
    for i in range(0, len(agent_emb) - step, step):
        ax2.annotate(
            '', xy=agent_emb[i+step], xytext=agent_emb[i],
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3),
        )
    
    ax2.set_xlabel('PCA 1', fontsize=12)
    ax2.set_ylabel('PCA 2', fontsize=12)
    ax2.set_title(f'Agent {best_agent} Trajectory Evolution\n(colored by progress)',
                  fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax2, label='Path Progress')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['Start', 'Mid', 'Goal'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# VISUALIZATION 5: Latent Correlation Analysis
# =============================================================================

def plot_latent_correlations(
    data: Dict[str, np.ndarray],
    metadata: Dict,
    output_path: str,
) -> None:
    """
    Analyze correlations between BDI components.
    
    INTERPRETATION:
    - Shows how Belief, Desire, and Intention relate to each other
    - Some correlation expected (I = f(B, D))
    - But each should capture unique information
    """
    
    print(f"\n{'='*60}")
    print("GENERATING CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    # Compute mean latent values per sample
    z_b_mean = data['z_b'].mean(axis=1)
    z_d_mean = data['z_d'].mean(axis=1)
    z_i_mean = data['z_i'].mean(axis=1)
    progress = data['progress']
    
    # Create correlation matrix from first few dimensions
    n_dims = min(8, data['z_b'].shape[1])
    combined = np.hstack([
        data['z_b'][:, :n_dims],
        data['z_d'][:, :n_dims],
        data['z_i'][:, :n_dims],
    ])
    
    corr_matrix = np.corrcoef(combined.T)
    
    # Create figure
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2])
    
    # Left: Correlation heatmap
    ax1 = fig.add_subplot(gs[0])
    
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add grid lines
    for i in [n_dims, 2*n_dims]:
        ax1.axhline(i - 0.5, color='black', linewidth=2)
        ax1.axvline(i - 0.5, color='black', linewidth=2)
    
    # Labels
    ax1.set_xticks([n_dims//2, n_dims + n_dims//2, 2*n_dims + n_dims//2])
    ax1.set_xticklabels(['Belief', 'Desire', 'Intention'])
    ax1.set_yticks([n_dims//2, n_dims + n_dims//2, 2*n_dims + n_dims//2])
    ax1.set_yticklabels(['Belief', 'Desire', 'Intention'])
    ax1.set_title('Latent Dimension Correlations', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Middle: Component correlations with progress
    ax2 = fig.add_subplot(gs[1])
    
    # Subsample for scatter
    n_plot = min(5000, len(progress))
    idx = np.random.choice(len(progress), n_plot, replace=False)
    
    ax2.scatter(progress[idx], z_b_mean[idx], alpha=0.3, s=5, c='#3498DB', label='Belief')
    ax2.scatter(progress[idx], z_d_mean[idx], alpha=0.3, s=5, c='#E74C3C', label='Desire')
    ax2.scatter(progress[idx], z_i_mean[idx], alpha=0.3, s=5, c='#2ECC71', label='Intention')
    
    ax2.set_xlabel('Path Progress', fontsize=12)
    ax2.set_ylabel('Mean Latent Value', fontsize=12)
    ax2.set_title('Latent Means vs Progress', fontsize=14, fontweight='bold')
    ax2.legend(markerscale=3)
    
    # Right: Variance analysis
    ax3 = fig.add_subplot(gs[2])
    
    # Compute variance per dimension for each component
    b_var = data['z_b'].var(axis=0)
    d_var = data['z_d'].var(axis=0)
    i_var = data['z_i'].var(axis=0)
    
    # Handle different latent dimensions by plotting separately
    max_dim = max(len(b_var), len(d_var), len(i_var))
    
    # Plot each as a line
    ax3.plot(range(len(b_var)), b_var, 'o-', label=f'Belief (dim={len(b_var)})', 
             color='#3498DB', alpha=0.8, markersize=4)
    ax3.plot(range(len(d_var)), d_var, 's-', label=f'Desire (dim={len(d_var)})', 
             color='#E74C3C', alpha=0.8, markersize=4)
    ax3.plot(range(len(i_var)), i_var, '^-', label=f'Intention (dim={len(i_var)})', 
             color='#2ECC71', alpha=0.8, markersize=4)
    
    ax3.set_xlabel('Latent Dimension', fontsize=12)
    ax3.set_ylabel('Variance', fontsize=12)
    ax3.set_title('Latent Dimension Variances', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.set_xlim(-0.5, max_dim - 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# VISUALIZATION 6: Summary Dashboard
# =============================================================================

def plot_summary_dashboard(
    data: Dict[str, np.ndarray],
    metadata: Dict,
    output_path: str,
) -> None:
    """
    Create a single summary figure with key insights.
    
    This is the ONE FIGURE you'd put in a paper abstract.
    """
    
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY DASHBOARD")
    print(f"{'='*60}")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Subsample
    n_samples = min(10000, len(data['z_b']))
    idx = np.random.choice(len(data['z_b']), n_samples, replace=False)
    
    # Compute embeddings
    pca_b = PCA(n_components=2, random_state=42).fit_transform(data['z_b'][idx])
    pca_d = PCA(n_components=2, random_state=42).fit_transform(data['z_d'][idx])
    pca_i = PCA(n_components=2, random_state=42).fit_transform(data['z_i'][idx])
    
    categories = data['category_ids'][idx]
    progress = data['progress'][idx]
    
    cat_names = ['food', 'study', 'leisure', 'health', 'errands', 'home', 'other']
    
    # 1. Belief space by category
    ax1 = fig.add_subplot(gs[0, 0])
    for cat_idx in range(len(cat_names)):
        mask = categories == cat_idx
        if mask.sum() > 0:
            ax1.scatter(pca_b[mask, 0], pca_b[mask, 1], 
                       c=CATEGORY_COLORS.get(cat_names[cat_idx], '#95A5A6'),
                       alpha=0.4, s=8, edgecolors='none')
    ax1.set_title('Belief (z_b)\nSpatial Understanding', fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # 2. Desire space by category
    ax2 = fig.add_subplot(gs[0, 1])
    for cat_idx in range(len(cat_names)):
        mask = categories == cat_idx
        if mask.sum() > 0:
            ax2.scatter(pca_d[mask, 0], pca_d[mask, 1],
                       c=CATEGORY_COLORS.get(cat_names[cat_idx], '#95A5A6'),
                       alpha=0.4, s=8, edgecolors='none')
    ax2.set_title('Desire (z_d)\nGoal Preferences', fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    
    # 3. Intention space by progress
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(pca_i[:, 0], pca_i[:, 1], c=progress, 
                         cmap=PROGRESS_CMAP, alpha=0.4, s=8, edgecolors='none')
    ax3.set_title('Intention (z_i)\nAction Plans', fontweight='bold')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Progress')
    
    # 4. Progress-accuracy curve (KEY RESULT)
    ax4 = fig.add_subplot(gs[1, :2])
    
    goal_logits = data['goal_logits']
    goal_ids = data['goal_ids']
    all_progress = data['progress']
    
    predictions = np.argmax(goal_logits, axis=1)
    correct = (predictions == goal_ids).astype(float)
    
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    accs = []
    for i in range(n_bins):
        mask = (all_progress >= bins[i]) & (all_progress < bins[i+1])
        accs.append(correct[mask].mean() * 100 if mask.sum() > 0 else 0)
    
    random_baseline = 100 / goal_logits.shape[1]
    
    ax4.fill_between(bin_centers * 100, random_baseline, accs, alpha=0.3, color='#3498DB')
    ax4.plot(bin_centers * 100, accs, 'o-', color='#3498DB', linewidth=3, markersize=12)
    ax4.axhline(y=random_baseline, color='#E74C3C', linestyle='--', linewidth=2, 
                label=f'Random baseline ({random_baseline:.2f}%)')
    
    ax4.set_xlabel('Path Progress (%)', fontsize=12)
    ax4.set_ylabel('Goal Prediction Accuracy (%)', fontsize=12)
    ax4.set_title('KEY RESULT: Accuracy Increases with Progress\n'
                  '(Model becomes more confident as agent approaches goal)',
                  fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.set_xlim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = accs[-1] / max(random_baseline, 0.01)
    ax4.annotate(
        f'{improvement:.0f}× better\nthan random',
        xy=(90, accs[-1]), xytext=(70, accs[-1] * 0.7),
        fontsize=11, ha='center', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2),
        bbox=dict(boxstyle='round', facecolor='#2ECC71', alpha=0.2),
    )
    
    # 5. Legend / info box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Create legend
    legend_elements = [mpatches.Patch(facecolor=CATEGORY_COLORS[cat], label=cat.capitalize())
                       for cat in cat_names if cat in CATEGORY_COLORS]
    ax5.legend(handles=legend_elements, loc='center', title='POI Categories',
               fontsize=11, title_fontsize=12, ncol=2)
    
    # Add stats box
    stats_text = f"""
    Dataset Statistics:
    • Samples: {len(data['z_b']):,}
    • Unique goals: {len(np.unique(data['goal_ids']))}
    • Unique agents: {len(np.unique(data['agent_ids']))}
    • Latent dim: {data['z_b'].shape[1]}
    
    Model Performance:
    • Final accuracy: {accs[-1]:.1f}%
    • vs Random: {improvement:.0f}× better
    • Improvement: {accs[-1] - accs[0]:.1f}pp
    """
    ax5.text(0.5, 0.3, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Main title
    fig.suptitle('BDI-VAE Latent Space Analysis Summary', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality BDI latent space visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    python visualize_bdi_latents.py --checkpoint checkpoints/sc_bdi_v3/best_model.pt
    python visualize_bdi_latents.py --checkpoint checkpoints/sc_bdi_v3/best_model.pt --method tsne
    python visualize_bdi_latents.py --checkpoint checkpoints/sc_bdi_v3/best_model.pt --output_dir ./paper_figures

OUTPUT FILES:
    1. bdi_comparison_{method}.png     - Main BDI component comparison
    2. progress_accuracy.png           - KEY: Accuracy vs progress
    3. category_analysis.png           - Category separation
    4. agent_patterns.png              - Agent-specific patterns
    5. latent_correlations.png         - Correlation analysis
    6. summary_dashboard.png           - One-figure summary

INTERPRETATION GUIDE:
    See docstrings in each plot function for detailed interpretation.
        """,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./artifacts/bdi_visualizations',
                        help='Output directory for figures')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Maximum samples for visualization (memory limit)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'all'],
                        help='Data split to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  BDI LATENT SPACE VISUALIZATION TOOL")
    print("  Publication-Quality Figures for Theory of Mind Analysis")
    print("=" * 70)
    
    # Load model and data
    model, data, metadata = load_model_and_data(
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        device=args.device,
    )
    
    # Generate all visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. BDI Component Comparison
    plot_bdi_comparison(
        data, metadata, 
        output_path=str(output_dir / f'bdi_comparison_{args.method}.png'),
        method=args.method,
    )
    
    # 2. Progress-Accuracy Analysis (KEY RESULT!)
    plot_progress_accuracy(
        data, metadata,
        output_path=str(output_dir / 'progress_accuracy.png'),
    )
    
    # 3. Category Analysis
    plot_category_analysis(
        data, metadata,
        output_path=str(output_dir / 'category_analysis.png'),
    )
    
    # 4. Agent Patterns
    plot_agent_analysis(
        data, metadata,
        output_path=str(output_dir / 'agent_patterns.png'),
    )
    
    # 5. Latent Correlations
    plot_latent_correlations(
        data, metadata,
        output_path=str(output_dir / 'latent_correlations.png'),
    )
    
    # 6. Summary Dashboard
    plot_summary_dashboard(
        data, metadata,
        output_path=str(output_dir / 'summary_dashboard.png'),
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 All figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"   • {f.name}")
    
    print("\n📖 INTERPRETATION GUIDE:")
    print("   1. bdi_comparison: Shows BDI spaces - Desire should show category structure")
    print("   2. progress_accuracy: KEY - accuracy should increase with progress")
    print("   3. category_analysis: Shows how well categories are separated")
    print("   4. agent_patterns: Shows individual agent behaviors")
    print("   5. latent_correlations: Shows relationships between BDI components")
    print("   6. summary_dashboard: One-figure summary for papers")


if __name__ == '__main__':
    main()
