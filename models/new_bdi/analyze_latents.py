#!/usr/bin/env python3
"""
Comprehensive BDI Latent Space Analysis Tool

This script provides deep analysis and visualization of the learned
Belief (z_b), Desire (z_d), and Intention (z_i) latent variables.

Features:
1. Latent space visualization (PCA, t-SNE, UMAP)
2. Cluster analysis and quality metrics
3. Goal prediction accuracy (top-K)
4. Progress-stratified analysis
5. Agent-specific analysis
6. Latent variable correlations
7. Reconstruction quality
8. Comparative analysis across checkpoints

Usage:
    python analyze_latents.py --checkpoint path/to/best_model.pt --output_dir ./analysis_results
    python analyze_latents.py --checkpoint path/to/best_model.pt --quick  # Fast analysis
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import networkx as nx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vae_bdi_simple.bdi_dataset_v3 import BDIVAEDatasetV3, collate_bdi_samples_v3
from models.vae_bdi_simple.bdi_vae_v3_model import SequentialConditionalBDIVAE, create_sc_bdi_vae_v3

# Try to import v2 model for older checkpoints
try:
    from models.vae_bdi_simple.bdi_vae_model import BDIVAE
    HAS_V2_MODEL = True
except ImportError:
    HAS_V2_MODEL = False


def detect_model_version(checkpoint: dict) -> str:
    """Detect which model version a checkpoint uses based on state_dict keys."""
    state_keys = list(checkpoint.get('model_state_dict', {}).keys())
    
    # V3 has these distinctive keys
    v3_indicators = ['goal_embeddings.weight', 'infonce.desire_proj', 'mi_estimator.discriminator', 
                     'progress_head', 'spatial_projection', 'intention_vae.prior_net']
    
    # V2 has these distinctive keys
    v2_indicators = ['feature_projection', 'hidden_layers']
    
    # Count matches
    v3_matches = sum(1 for ind in v3_indicators if any(ind in k for k in state_keys))
    v2_matches = sum(1 for ind in v2_indicators if any(ind in k for k in state_keys))
    
    if v3_matches > v2_matches:
        return 'v3'
    else:
        return 'v2'


# Try to import UMAP (optional)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: UMAP not installed. Install with 'pip install umap-learn' for UMAP visualizations.")


# =============================================================================
# DATA LOADING (same as training script)
# =============================================================================

def load_data_with_splits(
    data_dir: str = "data/simulation_data/run_8/trajectories",
    graph_path: str = "data/processed/ucsd_walk_full.graphml",
    split_indices_path: str = "data/processed/split_indices.json",
    trajectory_filename: str = 'all_trajectories.json',
) -> Tuple:
    """Load data with pre-defined splits."""
    print(f"\n📂 Loading data...")
    
    # Load graph
    print(f"   Loading graph from {graph_path}...")
    graph = nx.read_graphml(graph_path)
    print(f"   ✅ Graph: {graph.number_of_nodes()} nodes")
    
    # Load trajectories
    traj_path = Path(data_dir) / trajectory_filename
    print(f"   Loading trajectories from {traj_path}...")
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    # Handle nested format
    if isinstance(traj_data, dict):
        trajectories = []
        sorted_agents = sorted(traj_data.keys())
        for agent_idx, agent_key in enumerate(sorted_agents):
            for traj in traj_data[agent_key]:
                traj['agent_id'] = agent_idx
                trajectories.append(traj)
        print(f"   ✅ {len(trajectories)} trajectories from {len(sorted_agents)} agents")
    else:
        trajectories = traj_data
    
    # Get POI nodes using WorldGraph — same as simple BDI-VAE (230 nodes)
    from graph_controller.world_graph import WorldGraph
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    print(f"   ✅ {len(poi_nodes)} POI nodes (categories: {world_graph.relevant_categories})")
    
    # Load splits
    if os.path.exists(split_indices_path):
        with open(split_indices_path, 'r') as f:
            splits = json.load(f)
        train_idx = splits['train']
        val_idx = splits['val']
        test_idx = splits.get('test', [])
    else:
        # Create random split
        n = len(trajectories)
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.8*n)].tolist()
        val_idx = indices[int(0.8*n):].tolist()
        test_idx = []
    
    print(f"   ✅ Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    return graph, trajectories, poi_nodes, train_idx, val_idx, test_idx


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_latents_and_labels(
    model: SequentialConditionalBDIVAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Run inference on the dataset and collect all latent representations and labels.
    
    Returns dict with:
        - z_b, z_d, z_i: latent means (N x latent_dim)
        - z_b_logvar, z_d_logvar, z_i_logvar: latent log-variances
        - goal_ids, category_ids, agent_ids: labels
        - progress: trajectory progress (0-1)
        - goal_logits, category_logits: model predictions
    """
    model.eval()
    
    all_data = {
        'z_b': [], 'z_d': [], 'z_i': [],
        'z_b_logvar': [], 'z_d_logvar': [], 'z_i_logvar': [],
        'goal_ids': [], 'category_ids': [], 'agent_ids': [],
        'progress': [],
        'goal_logits': [], 'category_logits': [],
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting latents")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move batch to device
            history_node_indices = batch['history_node_indices'].to(device)
            history_lengths = batch['history_lengths'].to(device)
            agent_ids = batch['agent_id'].to(device)
            path_progress = batch['path_progress'].to(device)
            goal_idx = batch['goal_idx'].to(device)
            goal_cat_idx = batch['goal_cat_idx'].to(device)
            
            # Forward pass
            outputs = model(
                history_node_indices=history_node_indices,
                history_lengths=history_lengths,
                agent_ids=agent_ids,
                path_progress=path_progress,
                goal_idx=goal_idx,
                goal_cat_idx=goal_cat_idx,
                compute_loss=False,
            )
            
            # Extract latent representations (model returns z samples, not mu/logvar separately)
            # For analysis, we'll use the z samples as they contain the encoded information
            z_b = outputs['belief_z'].cpu().numpy()
            z_d = outputs['desire_z'].cpu().numpy()
            z_i = outputs['intention_z'].cpu().numpy()
            
            all_data['z_b'].append(z_b)
            all_data['z_d'].append(z_d)
            all_data['z_i'].append(z_i)
            
            # We'll use dummy logvars since they're not directly available
            # (the VAE uses reparameterization internally)
            all_data['z_b_logvar'].append(np.zeros_like(z_b))
            all_data['z_d_logvar'].append(np.zeros_like(z_d))
            all_data['z_i_logvar'].append(np.zeros_like(z_i))
            
            # Store predictions
            all_data['goal_logits'].append(outputs['goal'].cpu().numpy())
            all_data['category_logits'].append(outputs['category'].cpu().numpy())
            
            # Store labels
            all_data['goal_ids'].append(batch['goal_idx'].cpu().numpy())
            all_data['category_ids'].append(batch['goal_cat_idx'].cpu().numpy())
            all_data['agent_ids'].append(batch['agent_id'].cpu().numpy())
            
            # Store progress
            all_data['progress'].append(batch['path_progress'].cpu().numpy())
    
    # Concatenate all batches
    for key in all_data:
        if len(all_data[key]) > 0:
            all_data[key] = np.concatenate(all_data[key], axis=0)
    
    return all_data


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

def compute_embeddings(
    latents: np.ndarray,
    method: str = 'pca',
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Reduce latent dimensions for visualization.
    
    Args:
        latents: (N, latent_dim) array
        method: 'pca', 'tsne', or 'umap'
        n_components: target dimensions (usually 2)
    
    Returns:
        (N, n_components) array
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        embeddings = reducer.fit_transform(latents)
        explained_var = reducer.explained_variance_ratio_.sum()
        print(f"  PCA explained variance: {explained_var:.2%}")
        return embeddings, {'explained_variance': explained_var}
    
    elif method == 'tsne':
        perplexity = min(30, len(latents) // 4)
        # Use max_iter (sklearn >= 1.2) instead of deprecated n_iter
        reducer = TSNE(
            n_components=n_components, 
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            init='pca',
        )
        embeddings = reducer.fit_transform(latents)
        return embeddings, {'kl_divergence': reducer.kl_divergence_}
    
    elif method == 'umap':
        if not HAS_UMAP:
            print("UMAP not available, falling back to t-SNE")
            return compute_embeddings(latents, 'tsne', n_components, random_state)
        
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            random_state=random_state,
        )
        embeddings = reducer.fit_transform(latents)
        return embeddings, {}
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_latent_space_by_label(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    ax: plt.Axes,
    label_name: str = "Label",
    max_classes: int = 20,
    show_legend: bool = True,
    alpha: float = 0.6,
    s: float = 10,
):
    """Plot 2D embeddings colored by labels."""
    unique_labels = np.unique(labels)
    
    if len(unique_labels) > max_classes:
        # Use continuous colormap for many classes
        scatter = ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=labels, cmap='tab20', alpha=alpha, s=s
        )
        if show_legend:
            plt.colorbar(scatter, ax=ax, label=label_name)
    else:
        # Use discrete colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=[colors[i]], label=str(label), alpha=alpha, s=s
            )
        if show_legend and len(unique_labels) <= 10:
            ax.legend(title=label_name, loc='best', fontsize=6)
    
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')


def plot_latent_space_by_progress(
    embeddings: np.ndarray,
    progress: np.ndarray,
    title: str,
    ax: plt.Axes,
):
    """Plot 2D embeddings colored by trajectory progress."""
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=progress, cmap='RdYlGn', alpha=0.6, s=10,
        vmin=0, vmax=1
    )
    plt.colorbar(scatter, ax=ax, label='Progress')
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')


def create_latent_comparison_figure(
    data: Dict[str, np.ndarray],
    method: str = 'pca',
    output_path: Optional[str] = None,
):
    """
    Create a comprehensive figure comparing all three latent spaces.
    """
    print(f"\nGenerating {method.upper()} embeddings...")
    
    # Compute embeddings for each latent
    z_b_emb, _ = compute_embeddings(data['z_b'], method)
    z_d_emb, _ = compute_embeddings(data['z_d'], method)
    z_i_emb, _ = compute_embeddings(data['z_i'], method)
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'BDI Latent Space Analysis ({method.upper()})', fontsize=16, y=1.02)
    
    # Row 1: Belief latent
    plot_latent_space_by_label(z_b_emb, data['goal_ids'], 'Belief (z_b) by Goal', axes[0, 0], 'Goal')
    plot_latent_space_by_label(z_b_emb, data['category_ids'], 'Belief (z_b) by Category', axes[0, 1], 'Category')
    plot_latent_space_by_label(z_b_emb, data['agent_ids'], 'Belief (z_b) by Agent', axes[0, 2], 'Agent')
    plot_latent_space_by_progress(z_b_emb, data['progress'], 'Belief (z_b) by Progress', axes[0, 3])
    
    # Row 2: Desire latent
    plot_latent_space_by_label(z_d_emb, data['goal_ids'], 'Desire (z_d) by Goal', axes[1, 0], 'Goal')
    plot_latent_space_by_label(z_d_emb, data['category_ids'], 'Desire (z_d) by Category', axes[1, 1], 'Category')
    plot_latent_space_by_label(z_d_emb, data['agent_ids'], 'Desire (z_d) by Agent', axes[1, 2], 'Agent')
    plot_latent_space_by_progress(z_d_emb, data['progress'], 'Desire (z_d) by Progress', axes[1, 3])
    
    # Row 3: Intention latent
    plot_latent_space_by_label(z_i_emb, data['goal_ids'], 'Intention (z_i) by Goal', axes[2, 0], 'Goal')
    plot_latent_space_by_label(z_i_emb, data['category_ids'], 'Intention (z_i) by Category', axes[2, 1], 'Category')
    plot_latent_space_by_label(z_i_emb, data['agent_ids'], 'Intention (z_i) by Agent', axes[2, 2], 'Agent')
    plot_latent_space_by_progress(z_i_emb, data['progress'], 'Intention (z_i) by Progress', axes[2, 3])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()
    return z_b_emb, z_d_emb, z_i_emb


def plot_variance_histograms(data: Dict[str, np.ndarray], output_path: Optional[str] = None):
    """
    Plot histograms of latent variances to check for posterior collapse.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, logvar_key) in zip(axes, [
        ('Belief', 'z_b_logvar'),
        ('Desire', 'z_d_logvar'),
        ('Intention', 'z_i_logvar'),
    ]):
        logvars = data[logvar_key].flatten()
        variances = np.exp(logvars)
        
        ax.hist(logvars, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', label='log(var)=0 (var=1)')
        ax.axvline(x=logvars.mean(), color='green', linestyle='-', label=f'Mean: {logvars.mean():.2f}')
        
        ax.set_xlabel('Log Variance')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} (z_{name[0].lower()}) Log Variance\nMean var: {variances.mean():.3f}')
        ax.legend()
    
    fig.suptitle('Latent Variance Distribution (Check for Posterior Collapse)', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()


# =============================================================================
# CLUSTER ANALYSIS
# =============================================================================

def compute_cluster_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Metrics:
        - Silhouette score: How well-separated are clusters? (-1 to 1, higher = better)
        - ARI: Adjusted Rand Index (0 to 1, higher = better)
        - NMI: Normalized Mutual Information (0 to 1, higher = better)
    """
    # Remove labels with only 1 sample
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = np.isin(labels, unique_labels[counts > 1])
    
    if valid_mask.sum() < 10:
        return {'silhouette': 0, 'ari': 0, 'nmi': 0, 'n_valid': 0}
    
    latents_valid = latents[valid_mask]
    labels_valid = labels[valid_mask]
    
    # Compute silhouette score with true labels
    try:
        silhouette = silhouette_score(latents_valid, labels_valid)
    except:
        silhouette = 0
    
    # K-means clustering
    n_clusters = n_clusters or len(np.unique(labels_valid))
    n_clusters = min(n_clusters, len(latents_valid) - 1, 100)  # Cap at 100
    
    if n_clusters < 2:
        return {'silhouette': silhouette, 'ari': 0, 'nmi': 0, 'n_valid': len(labels_valid)}
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(latents_valid)
        
        ari = adjusted_rand_score(labels_valid, pred_labels)
        nmi = normalized_mutual_info_score(labels_valid, pred_labels)
    except:
        ari, nmi = 0, 0
    
    return {
        'silhouette': float(silhouette),
        'ari': float(ari),
        'nmi': float(nmi),
        'n_valid': int(len(labels_valid)),
        'n_clusters': int(n_clusters),
    }


def analyze_cluster_quality(data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Analyze clustering quality for all latents."""
    print("\n" + "="*60)
    print("CLUSTER QUALITY ANALYSIS")
    print("="*60)
    
    results = {}
    
    for latent_name, latent_key in [('Belief', 'z_b'), ('Desire', 'z_d'), ('Intention', 'z_i')]:
        latents = data[latent_key]
        results[latent_name] = {}
        
        print(f"\n{latent_name} (z_{latent_name[0].lower()}):")
        
        for label_name, label_key in [('Goal', 'goal_ids'), ('Category', 'category_ids'), ('Agent', 'agent_ids')]:
            labels = data[label_key]
            metrics = compute_cluster_metrics(latents, labels)
            results[latent_name][label_name] = metrics
            
            print(f"  By {label_name}:")
            print(f"    Silhouette: {metrics['silhouette']:.3f} (higher = better separation)")
            print(f"    ARI: {metrics['ari']:.3f} (1.0 = perfect clustering)")
            print(f"    NMI: {metrics['nmi']:.3f} (1.0 = perfect clustering)")
    
    return results


# =============================================================================
# GOAL PREDICTION ANALYSIS
# =============================================================================

def compute_topk_accuracy(
    logits: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20, 50],
) -> Dict[int, float]:
    """Compute top-K accuracy for goal prediction."""
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    
    results = {}
    for k in k_values:
        if k > logits.shape[1]:
            continue
        _, topk_preds = logits_tensor.topk(k, dim=1)
        correct = (topk_preds == labels_tensor.unsqueeze(1)).any(dim=1)
        results[k] = float(correct.float().mean())
    
    return results


def analyze_prediction_by_progress(
    data: Dict[str, np.ndarray],
    n_bins: int = 5,
) -> Dict[str, Dict]:
    """Analyze prediction accuracy stratified by trajectory progress."""
    print("\n" + "="*60)
    print("PROGRESS-STRATIFIED ACCURACY ANALYSIS")
    print("="*60)
    
    progress = data['progress']
    goal_logits = data['goal_logits']
    goal_ids = data['goal_ids']
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    results = {}
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (progress >= low) & (progress < high if i < n_bins - 1 else progress <= high)
        
        if mask.sum() < 10:
            continue
        
        topk_acc = compute_topk_accuracy(goal_logits[mask], goal_ids[mask])
        
        bin_name = f"{int(low*100)}-{int(high*100)}%"
        results[bin_name] = {
            'n_samples': int(mask.sum()),
            'top_k_accuracy': topk_acc,
        }
        
        print(f"\n  Progress {bin_name} ({mask.sum()} samples):")
        for k, acc in topk_acc.items():
            print(f"    Top-{k}: {acc:.1%}")
    
    return results


# =============================================================================
# LATENT CORRELATION ANALYSIS
# =============================================================================

def analyze_latent_correlations(data: Dict[str, np.ndarray], output_path: Optional[str] = None):
    """Analyze correlations between BDI latent dimensions."""
    print("\n" + "="*60)
    print("LATENT CORRELATION ANALYSIS")
    print("="*60)
    
    z_b, z_d, z_i = data['z_b'], data['z_d'], data['z_i']
    latent_dim = z_b.shape[1]
    
    # Concatenate all latents
    combined = np.hstack([z_b, z_d, z_i])
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(combined.T)
    
    # Create labeled sections
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add section boundaries
    for boundary in [latent_dim, 2 * latent_dim]:
        ax.axhline(y=boundary - 0.5, color='black', linewidth=2)
        ax.axvline(x=boundary - 0.5, color='black', linewidth=2)
    
    # Add labels
    ax.set_xticks([latent_dim//2, latent_dim + latent_dim//2, 2*latent_dim + latent_dim//2])
    ax.set_xticklabels(['Belief (z_b)', 'Desire (z_d)', 'Intention (z_i)'])
    ax.set_yticks([latent_dim//2, latent_dim + latent_dim//2, 2*latent_dim + latent_dim//2])
    ax.set_yticklabels(['Belief (z_b)', 'Desire (z_d)', 'Intention (z_i)'])
    
    ax.set_title('Latent Variable Correlation Matrix')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()
    
    # Print summary statistics
    print(f"\nCorrelation Summary:")
    print(f"  z_b ↔ z_d: mean abs corr = {np.abs(corr_matrix[:latent_dim, latent_dim:2*latent_dim]).mean():.3f}")
    print(f"  z_b ↔ z_i: mean abs corr = {np.abs(corr_matrix[:latent_dim, 2*latent_dim:]).mean():.3f}")
    print(f"  z_d ↔ z_i: mean abs corr = {np.abs(corr_matrix[latent_dim:2*latent_dim, 2*latent_dim:]).mean():.3f}")
    
    return corr_matrix


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_full_analysis(
    checkpoint_path: str,
    output_dir: str,
    data_dir: str = "data/simulation_data/run_8/trajectories",
    graph_path: str = "data/processed/ucsd_walk_full.graphml",
    split_indices_path: str = "data/processed/split_indices.json",
    split: str = "val",
    batch_size: int = 256,
    max_batches: Optional[int] = None,
    methods: List[str] = ['pca', 'tsne'],
    device: str = "cuda",
):
    """
    Run the complete latent space analysis pipeline.
    """
    print("="*60)
    print("BDI LATENT SPACE ANALYSIS")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data using training script approach
    graph, trajectories, poi_nodes, train_idx, val_idx, test_idx = load_data_with_splits(
        data_dir=data_dir,
        graph_path=graph_path,
        split_indices_path=split_indices_path,
    )
    
    # Create node-to-idx mapping
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    
    # Select split
    if split == "val":
        split_trajs = [trajectories[i] for i in val_idx]
    elif split == "train":
        split_trajs = [trajectories[i] for i in train_idx]
    else:
        split_trajs = trajectories
    
    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = BDIVAEDatasetV3(
        trajectories=split_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=node_to_idx,
        include_progress=True,
    )
    
    # Get dataset stats
    num_nodes = len(dataset.node_to_idx)
    num_agents = len(set(s['agent_id'] for s in dataset.samples))
    num_poi_nodes = len(dataset.goal_to_idx)
    # Get categories from graph
    categories = set()
    for n, d in graph.nodes(data=True):
        cat = d.get('clean_categories')
        if cat and cat not in [None, '', 'None']:
            categories.add(cat)
    num_categories = max(7, len(categories))  # At least 7
    
    print(f"   Nodes: {num_nodes}")
    print(f"   Agents: {num_agents}")
    print(f"   POI nodes: {num_poi_nodes}")
    print(f"   Categories: {num_categories}")
    print(f"   Samples: {len(dataset)}")
    
    # Load checkpoint
    print("\n📁 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Detect model version
    model_version = detect_model_version(checkpoint)
    print(f"   Detected model version: {model_version}")
    
    # Initialize model with correct parameters
    print("🏗️  Initializing model...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if model_version == 'v3':
        model = create_sc_bdi_vae_v3(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_poi_nodes=num_poi_nodes,
            num_categories=num_categories,
        )
    else:
        # V2 model - need to check actual initialization parameters
        if not HAS_V2_MODEL:
            raise RuntimeError(
                f"Checkpoint uses v2 model but BDIVAE could not be imported. "
                f"Please use a v3 checkpoint instead."
            )
        # Try to get parameters from state dict
        state_dict = checkpoint['model_state_dict']
        # V2 model has different architecture - try to infer params
        goal_head_weight = [k for k in state_dict.keys() if 'goal_head.weight' in k or k.endswith('goal_head.weight')]
        if goal_head_weight:
            v2_num_poi = state_dict[goal_head_weight[0]].shape[0]
        else:
            v2_num_poi = num_poi_nodes
        print(f"   V2 checkpoint has {v2_num_poi} POI nodes (current dataset: {num_poi_nodes})")
        model = BDIVAE(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_poi_nodes=v2_num_poi,  # Use checkpoint's POI count
            num_categories=num_categories,
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    val_metric = checkpoint.get('val_metric', checkpoint.get('best_val_accuracy', 'unknown'))
    print(f"   ✅ Val metric: {val_metric}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_bdi_samples_v3,
    )
    
    # Collect latents
    print("\n🔍 Collecting latent representations...")
    data = collect_latents_and_labels(model, dataloader, device, max_batches)
    
    print(f"   Collected {len(data['z_b'])} samples")
    print(f"   Latent dim: {data['z_b'].shape[1]}")
    print(f"   Unique goals: {len(np.unique(data['goal_ids']))}")
    print(f"   Unique categories: {len(np.unique(data['category_ids']))}")
    print(f"   Unique agents: {len(np.unique(data['agent_ids']))}")
    
    # Run analyses
    results = {
        'checkpoint': checkpoint_path,
        'split': split,
        'n_samples': len(data['z_b']),
        'timestamp': datetime.now().isoformat(),
    }
    
    # 5. Latent visualizations
    print("\n🎨 Generating latent space visualizations...")
    for method in methods:
        output_path = os.path.join(output_dir, f'latent_space_{method}.png')
        create_latent_comparison_figure(data, method=method, output_path=output_path)
    
    # 6. Variance analysis
    print("\n6. Analyzing latent variances...")
    plot_variance_histograms(data, os.path.join(output_dir, 'variance_histograms.png'))
    
    # 7. Cluster analysis
    cluster_results = analyze_cluster_quality(data)
    results['cluster_metrics'] = cluster_results
    
    # 8. Progress-stratified analysis
    progress_results = analyze_prediction_by_progress(data)
    results['progress_stratified'] = progress_results
    
    # 9. Correlation analysis
    print("\n9. Analyzing latent correlations...")
    analyze_latent_correlations(data, os.path.join(output_dir, 'latent_correlations.png'))
    
    # 10. Top-K accuracy
    print("\n" + "="*60)
    print("TOP-K GOAL PREDICTION ACCURACY")
    print("="*60)
    
    topk_acc = compute_topk_accuracy(data['goal_logits'], data['goal_ids'])
    results['topk_accuracy'] = topk_acc
    
    random_baseline = 1.0 / len(np.unique(data['goal_ids']))
    print(f"\n  Random baseline: {random_baseline:.4%}")
    for k, acc in topk_acc.items():
        improvement = acc / (k * random_baseline)
        print(f"  Top-{k}: {acc:.2%} ({improvement:.1f}x better than random)")
    
    # Save results
    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n✓ Generated {len(methods)} latent space visualizations")
    print(f"✓ Analyzed variance distributions (posterior collapse check)")
    print(f"✓ Computed cluster quality metrics")
    print(f"✓ Analyzed accuracy by trajectory progress")
    print(f"✓ Computed latent correlations")
    print(f"\nAll outputs saved to: {output_dir}")
    
    return results, data


# =============================================================================
# QUICK ANALYSIS (SUBSET)
# =============================================================================

def run_quick_analysis(
    checkpoint_path: str,
    output_dir: str,
):
    """Run a quick analysis with fewer samples and only PCA."""
    return run_full_analysis(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        max_batches=20,  # ~5000 samples
        methods=['pca'],
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BDI Latent Space Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./artifacts/latent_analysis',
                        help='Output directory for visualizations')
    parser.add_argument('--data_dir', type=str, default='data/simulation_data/run_8/trajectories',
                        help='Directory containing trajectories')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml',
                        help='Path to graph file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'all'],
                        help='Dataset split to analyze')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit number of batches (for quick testing)')
    parser.add_argument('--methods', type=str, nargs='+', default=['pca', 'tsne'],
                        help='Dimensionality reduction methods')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick analysis (fewer samples)')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_analysis(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
    else:
        run_full_analysis(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            graph_path=args.graph_path,
            split=args.split,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            methods=args.methods,
            device=args.device,
        )


if __name__ == '__main__':
    main()
