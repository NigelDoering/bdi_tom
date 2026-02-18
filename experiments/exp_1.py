"""
Experiment 1: Model Performance vs Trajectory Observation

This experiment evaluates model performance at different trajectory observation points
(15%, 30%, 45%, 60%, 75%, 90%) and reports:
- Top-1 accuracy
- Top-5 accuracy
- Brier score (calibration metric)

Results are visualized as line charts showing performance progression.

Usage:
    python experiments/exp_1.py --model_path checkpoints/keepers/baseline_transformer_best_model.pt --model_type transformer
    python experiments/exp_1.py --model_path checkpoints/keepers/lstm_best_model.pt --model_type lstm
    python experiments/exp_1.py --model_path checkpoints/keepers/best_model-OURS.pt --model_type sc_bdi_vae

    # Run all three models:
    python experiments/exp_1.py --run_all
"""

import os
import sys
import argparse
from pathlib import Path

# Set MPS fallback for Mac compatibility BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_transformer.baseline_transformer_model import PerNodeTransformerPredictor
from models.baseline_transformer.baseline_transformer_dataset import TransformerTrajectoryDataset, collate_transformer_trajectories
from models.baseline_lstm.baseline_lstm_model import PerNodeToMPredictor
from models.baseline_lstm.baseline_lstm_dataset import PerNodeTrajectoryDataset, collate_per_node_samples
from models.new_bdi.bdi_vae_v3_model import SequentialConditionalBDIVAE, create_sc_bdi_vae_v3
from models.new_bdi.bdi_dataset_v3 import BDIVAEDatasetV3, collate_bdi_samples_v3
from models.utils.data_loader import load_simulation_data
from models.utils.utils import get_device
from torch.utils.data import DataLoader, Subset


def compute_brier_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Brier score for probabilistic predictions.
    
    Brier score measures the mean squared difference between predicted
    probabilities and the actual outcomes. Lower is better.
    
    Args:
        logits: (batch_size, num_classes) - raw model outputs
        targets: (batch_size,) - true class indices
    
    Returns:
        Brier score (0 = perfect, 2 = worst possible)
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Create one-hot encoding of targets
    num_classes = logits.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    # Brier score: mean squared error between probabilities and true labels
    brier = torch.mean((probs - targets_one_hot) ** 2).item()
    
    return brier


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: (batch_size, num_classes) - raw model outputs
        targets: (batch_size,) - true class indices
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as percentage
    """
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1).float()
    accuracy = correct.mean().item() * 100
    return accuracy


@torch.no_grad()
def evaluate_transformer_at_proportion(
    model: PerNodeTransformerPredictor,
    trajectories: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    device: torch.device,
    proportion: float,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate transformer model at a specific trajectory proportion.
    
    Args:
        model: Trained transformer model
        trajectories: List of trajectory dictionaries
        graph: Graph structure
        poi_nodes: List of POI nodes
        device: Device to run on
        proportion: Proportion of trajectory to observe (e.g., 0.15 for 15%)
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with top1, top5, and brier scores
    """
    model.eval()
    
    # Truncate trajectories to proportion
    truncated_trajs = []
    for traj in trajectories:
        path = traj['path']
        if len(path) > 2:
            truncate_idx = max(1, int(len(path) * proportion))
            truncated_path = path[:truncate_idx]
        else:
            truncated_path = path[:1] if len(path) > 0 else path
        
        truncated_traj = traj.copy()
        truncated_traj['path'] = truncated_path
        truncated_trajs.append(truncated_traj)
    
    # Create dataset with truncated trajectories
    dataset = TransformerTrajectoryDataset(truncated_trajs, graph, poi_nodes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transformer_trajectories,
        num_workers=0
    )
    
    all_logits = []
    all_targets = []
    
    for batch in tqdm(loader, desc=f'{int(proportion*100)}% trajectory', leave=False):
        # Move to device
        node_indices = batch['node_indices'].to(device)
        agent_ids = batch['agent_ids'].to(device)
        hours = batch['hours'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        
        # Forward pass
        predictions = model(node_indices, agent_ids, hours, padding_mask)
        goal_logits = predictions['goal']  # [batch, seq_len, num_poi_nodes]
        
        # Get prediction at last valid position for each trajectory
        seq_lengths = batch['seq_lengths']
        batch_logits = []
        for i in range(len(seq_lengths)):
            last_pos = seq_lengths[i] - 1
            batch_logits.append(goal_logits[i, last_pos])
        
        batch_logits = torch.stack(batch_logits)  # [batch, num_poi_nodes]
        
        all_logits.append(batch_logits.cpu())
        all_targets.append(goal_idx.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    top1 = compute_top_k_accuracy(all_logits, all_targets, k=1)
    top5 = compute_top_k_accuracy(all_logits, all_targets, k=5)
    brier = compute_brier_score(all_logits, all_targets)
    
    return {
        'top1': top1,
        'top5': top5,
        'brier': brier
    }


@torch.no_grad()
def evaluate_lstm_at_proportion(
    model: PerNodeToMPredictor,
    trajectories: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    device: torch.device,
    proportion: float,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate LSTM model at a specific trajectory proportion.
    
    Args:
        model: Trained LSTM model
        trajectories: List of trajectory dictionaries
        graph: Graph structure
        poi_nodes: List of POI nodes
        device: Device to run on
        proportion: Proportion of trajectory to observe
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with top1, top5, and brier scores
    """
    model.eval()
    
    # Truncate trajectories to the specified proportion
    truncated_trajs = []
    for traj in trajectories:
        path = traj['path']
        if len(path) < 2:
            continue
        
        # Truncate to proportion
        truncate_idx = max(1, int(len(path) * proportion))
        truncated_path = path[:truncate_idx]
        
        # Create truncated trajectory
        truncated_traj = traj.copy()
        truncated_traj['path'] = truncated_path
        truncated_trajs.append(truncated_traj)
    
    if len(truncated_trajs) == 0:
        return {'top1': 0.0, 'top5': 0.0, 'brier': 2.0}
    
    # Create dataset - this will expand each trajectory into per-node samples
    # We want the last sample from each trajectory (most complete history)
    dataset = PerNodeTrajectoryDataset(
        trajectories=truncated_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        min_traj_length=1,
    )
    
    if len(dataset) == 0:
        return {'top1': 0.0, 'top5': 0.0, 'brier': 2.0}
    
    # For LSTM, we typically use all per-node samples OR just the final ones
    # For fair comparison with evaluation at a specific point, use only the last sample
    # from each trajectory (the one with the full truncated history)
    # However, PerNodeTrajectoryDataset doesn't track which samples belong to which trajectory
    # So for simplicity, we'll just use all samples and average (similar to training)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_per_node_samples,
        num_workers=0
    )
    
    all_logits = []
    all_targets = []
    
    for batch in tqdm(loader, desc=f'{int(proportion*100)}% trajectory', leave=False):
        # Move to device
        history_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        
        # Forward pass
        predictions = model(history_indices, history_lengths)
        goal_logits = predictions['goal']  # [batch, num_poi_nodes]
        
        all_logits.append(goal_logits.cpu())
        all_targets.append(goal_idx.cpu())
    
    if len(all_logits) == 0:
        return {'top1': 0.0, 'top5': 0.0, 'brier': 2.0}
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    top1 = compute_top_k_accuracy(all_logits, all_targets, k=1)
    top5 = compute_top_k_accuracy(all_logits, all_targets, k=5)
    brier = compute_brier_score(all_logits, all_targets)
    
    return {
        'top1': top1,
        'top5': top5,
        'brier': brier
    }


def load_transformer_model(checkpoint_path: str, num_nodes: int, num_poi_nodes: int, device: torch.device) -> PerNodeTransformerPredictor:
    """Load transformer model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same config as training
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def load_lstm_model(checkpoint_path: str, num_nodes: int, num_poi_nodes: int, device: torch.device) -> PerNodeToMPredictor:
    """Load LSTM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Use config from checkpoint if available, otherwise use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        
        # Detect actual num_agents from agent embedding weight shape
        # (the config may have incorrect value)
        agent_key = 'embedding_pipeline.agent_encoder.agent_context.agent_emb.agent_embedding.weight'
        if agent_key in checkpoint['model_state_dict']:
            actual_num_agents = checkpoint['model_state_dict'][agent_key].shape[0]
        else:
            actual_num_agents = config.get('num_agents', 100)
        
        model = PerNodeToMPredictor(
            num_nodes=config.get('num_nodes', num_nodes),
            num_agents=actual_num_agents,  # Use detected value, not config
            num_poi_nodes=config.get('num_poi_nodes', num_poi_nodes),
            num_categories=config.get('num_categories', 7),
            node_embedding_dim=config.get('node_embedding_dim', 64),
            temporal_dim=config.get('temporal_dim', 64),
            agent_dim=config.get('agent_dim', 64),
            fusion_dim=config.get('fusion_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.1),
            num_heads=config.get('num_heads', 4),
            freeze_embedding=config.get('freeze_embedding', False),
        )
    else:
        # Fallback to original defaults
        model = PerNodeToMPredictor(
            num_nodes=num_nodes,
            num_agents=100,
            num_poi_nodes=num_poi_nodes,
            num_categories=7,
            node_embedding_dim=64,
            temporal_dim=64,
            agent_dim=64,
            fusion_dim=128,
            hidden_dim=256,
            dropout=0.1,
            num_heads=4,
            freeze_embedding=False,
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def load_sc_bdi_vae_model(
    checkpoint_path: str, 
    num_nodes: int, 
    num_poi_nodes: int, 
    num_agents: int,
    device: torch.device
) -> SequentialConditionalBDIVAE:
    """Load SC-BDI-VAE V3 model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same config as training
    model = create_sc_bdi_vae_v3(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=7,
        # Embedding dimensions
        node_embedding_dim=64,
        fusion_dim=128,
        # VAE dimensions
        belief_latent_dim=32,
        desire_latent_dim=16,
        intention_latent_dim=32,
        vae_hidden_dim=128,
        # Prediction
        hidden_dim=256,
        dropout=0.1,
        # Options
        use_progress=False,
        use_temporal=True,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  ✅ Loaded SC-BDI-VAE checkpoint (all keys matched)")
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_sc_bdi_vae_at_proportion(
    model: SequentialConditionalBDIVAE,
    trajectories: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    device: torch.device,
    proportion: float,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate SC-BDI-VAE V3 model at a specific trajectory proportion.
    
    Uses BDIVAEDatasetV3 and collate_bdi_samples_v3 for proper data handling.
    
    Args:
        model: Trained SC-BDI-VAE V3 model
        trajectories: List of trajectory dictionaries
        graph: Graph structure
        poi_nodes: List of POI nodes
        device: Device to run on
        proportion: Proportion of trajectory to observe (e.g., 0.15 for 15%)
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with top1, top5, and brier scores
    """
    model.eval()
    
    # Truncate trajectories to the specified proportion
    truncated_trajs = []
    for traj in trajectories:
        path = traj['path']
        if len(path) < 2:
            continue
        
        # Truncate to proportion
        truncate_idx = max(1, int(len(path) * proportion))
        truncated_path = path[:truncate_idx]
        
        # Create truncated trajectory
        truncated_traj = traj.copy()
        truncated_traj['path'] = truncated_path
        truncated_trajs.append(truncated_traj)
    
    if len(truncated_trajs) == 0:
        return {'top1': 0.0, 'top5': 0.0, 'brier': 2.0}
    
    # Create dataset using BDIVAEDatasetV3
    # This will expand each trajectory into per-node samples
    # We only care about the final sample from each trajectory (last observation point)
    dataset = BDIVAEDatasetV3(
        trajectories=truncated_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        min_traj_length=1,
        include_progress=True,
    )
    
    if len(dataset) == 0:
        return {'top1': 0.0, 'top5': 0.0, 'brier': 2.0}
    
    # For evaluation, we want to use the last sample from each trajectory
    # (the sample with the most information - the full truncated history)
    # Get indices of final samples for each trajectory
    eval_indices = []
    for traj_id, sample_indices in dataset.trajectory_samples.items():
        if len(sample_indices) > 0:
            # Get the last sample (most complete history up to truncation point)
            eval_indices.append(sample_indices[-1])
    
    # Create subset dataset with only final samples
    eval_dataset = Subset(dataset, eval_indices)
    
    # Create dataloader with proper collation
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_bdi_samples_v3,
        num_workers=0
    )
    
    all_logits = []
    all_targets = []
    
    for batch in tqdm(loader, desc=f'{int(proportion*100)}% trajectory', leave=False):
        # Move to device
        history_node_indices = batch['history_node_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        agent_ids = batch['agent_id'].to(device)
        path_progress = batch['path_progress'].to(device)
        
        # Forward pass (inference only, no loss computation)
        outputs = model(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_ids,
            path_progress=path_progress,
            compute_loss=False,
        )
        
        goal_logits = outputs['goal']  # [batch, num_poi_nodes]
        
        all_logits.append(goal_logits.cpu())
        all_targets.append(goal_idx.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    top1 = compute_top_k_accuracy(all_logits, all_targets, k=1)
    top5 = compute_top_k_accuracy(all_logits, all_targets, k=5)
    brier = compute_brier_score(all_logits, all_targets)
    
    return {
        'top1': top1,
        'top5': top5,
        'brier': brier
    }


def plot_results(results: Dict, proportions: List[float], output_dir: str, model_name: str):
    """Generate separate plots for each metric."""
    # Extract metrics
    top1_scores = [results[p]['top1'] for p in proportions]
    top5_scores = [results[p]['top5'] for p in proportions]
    brier_scores = [results[p]['brier'] for p in proportions]
    
    percentages = [int(p * 100) for p in proportions]
    
    # ===== TOP-1 ACCURACY PLOT =====
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(percentages, top1_scores, marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
    ax.set_xlabel('Trajectory Observed (%)', fontsize=14)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=14)
    ax.set_title(f'Top-1 Goal Prediction Accuracy\n{model_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'exp1_{model_name}_top1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Top-1 plot saved: {output_path}")
    plt.close()
    
    # ===== TOP-5 ACCURACY PLOT =====
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(percentages, top5_scores, marker='s', linewidth=2.5, markersize=10, color='#F77F00')
    ax.set_xlabel('Trajectory Observed (%)', fontsize=14)
    ax.set_ylabel('Top-5 Accuracy (%)', fontsize=14)
    ax.set_title(f'Top-5 Goal Prediction Accuracy\n{model_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'exp1_{model_name}_top5.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Top-5 plot saved: {output_path}")
    plt.close()
    
    # ===== BRIER SCORE PLOT =====
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(percentages, brier_scores, marker='^', linewidth=2.5, markersize=10, color='#06A77D')
    ax.set_xlabel('Trajectory Observed (%)', fontsize=14)
    ax.set_ylabel('Brier Score', fontsize=14)
    ax.set_title(f'Prediction Calibration (Brier Score)\n{model_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits based on data range for better visibility
    min_brier = min(brier_scores)
    max_brier = max(brier_scores)
    brier_range = max_brier - min_brier
    y_min = max(0, min_brier - 0.1 * brier_range)
    y_max = min(2, max_brier + 0.1 * brier_range)
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'exp1_{model_name}_brier.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Brier plot saved: {output_path}")
    plt.close()


def save_results_csv(results: Dict, proportions: List[float], output_dir: str, model_name: str):
    """Save results to CSV file."""
    data = {
        'proportion': [f'{int(p*100)}%' for p in proportions],
        'top1_accuracy': [results[p]['top1'] for p in proportions],
        'top5_accuracy': [results[p]['top5'] for p in proportions],
        'brier_score': [results[p]['brier'] for p in proportions],
    }
    
    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, f'exp1_{model_name}_results.csv')
    df.to_csv(output_path, index=False)
    print(f"  💾 CSV saved: {output_path}")


def print_results_table(results: Dict, proportions: List[float], model_name: str):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1 RESULTS - {model_name}")
    print(f"{'='*80}")
    print(f"{'Observed':<12} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Brier Score':<12}")
    print(f"{'-'*80}")
    
    for proportion in proportions:
        metrics = results[proportion]
        print(f"{int(proportion*100):3d}%         "
              f"{metrics['top1']:6.2f}%      "
              f"{metrics['top5']:6.2f}%      "
              f"{metrics['brier']:.4f}")
    
    print(f"{'='*80}\n")


def plot_comparison(all_results: Dict[str, Dict], output_dir: str):
    """
    Generate comparison plots for all models.
    
    Args:
        all_results: Dict mapping model_name -> {proportion -> {top1, top5, brier}}
        output_dir: Directory to save plots
    """
    proportions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    percentages = [int(p * 100) for p in proportions]
    
    # Define colors and markers for each model
    model_styles = {
        'Transformer': {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-'},
        'LSTM': {'color': '#F77F00', 'marker': 's', 'linestyle': '--'},
        'SC-BDI-VAE': {'color': '#06A77D', 'marker': '^', 'linestyle': '-.'},
    }
    
    # Default style for unknown models
    default_colors = ['#E63946', '#9B5DE5', '#00BBF9', '#F15BB5']
    
    # ===== TOP-1 ACCURACY COMPARISON =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        top1_scores = [results[p]['top1'] for p in proportions]
        style = model_styles.get(model_name, {
            'color': default_colors[idx % len(default_colors)],
            'marker': 'o',
            'linestyle': '-'
        })
        ax.plot(percentages, top1_scores, 
                marker=style['marker'], 
                linewidth=2.5, 
                markersize=10, 
                color=style['color'],
                linestyle=style['linestyle'],
                label=model_name)
    
    ax.set_xlabel('Trajectory Observed (%)', fontsize=14)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=14)
    ax.set_title('Top-1 Goal Prediction Accuracy\nModel Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp1_comparison_top1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Comparison Top-1 plot saved: {output_path}")
    plt.close()
    
    # ===== TOP-5 ACCURACY COMPARISON =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        top5_scores = [results[p]['top5'] for p in proportions]
        style = model_styles.get(model_name, {
            'color': default_colors[idx % len(default_colors)],
            'marker': 's',
            'linestyle': '-'
        })
        ax.plot(percentages, top5_scores, 
                marker=style['marker'], 
                linewidth=2.5, 
                markersize=10, 
                color=style['color'],
                linestyle=style['linestyle'],
                label=model_name)
    
    ax.set_xlabel('Trajectory Observed (%)', fontsize=14)
    ax.set_ylabel('Top-5 Accuracy (%)', fontsize=14)
    ax.set_title('Top-5 Goal Prediction Accuracy\nModel Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp1_comparison_top5.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Comparison Top-5 plot saved: {output_path}")
    plt.close()
    
    # ===== BRIER SCORE COMPARISON =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    all_brier = []
    for idx, (model_name, results) in enumerate(all_results.items()):
        brier_scores = [results[p]['brier'] for p in proportions]
        all_brier.extend(brier_scores)
        style = model_styles.get(model_name, {
            'color': default_colors[idx % len(default_colors)],
            'marker': '^',
            'linestyle': '-'
        })
        ax.plot(percentages, brier_scores, 
                marker=style['marker'], 
                linewidth=2.5, 
                markersize=10, 
                color=style['color'],
                linestyle=style['linestyle'],
                label=model_name)
    
    ax.set_xlabel('Trajectory Observed (%)', fontsize=14)
    ax.set_ylabel('Brier Score', fontsize=14)
    ax.set_title('Prediction Calibration (Brier Score)\nModel Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits based on data range
    min_brier = min(all_brier)
    max_brier = max(all_brier)
    brier_range = max_brier - min_brier
    y_min = max(0, min_brier - 0.1 * brier_range)
    y_max = min(2, max_brier + 0.1 * brier_range)
    ax.set_ylim([y_min, y_max])
    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp1_comparison_brier.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Comparison Brier plot saved: {output_path}")
    plt.close()
    
    # ===== SAVE COMBINED CSV =====
    combined_data = {'proportion': [f'{int(p*100)}%' for p in proportions]}
    for model_name, results in all_results.items():
        combined_data[f'{model_name}_top1'] = [results[p]['top1'] for p in proportions]
        combined_data[f'{model_name}_top5'] = [results[p]['top5'] for p in proportions]
        combined_data[f'{model_name}_brier'] = [results[p]['brier'] for p in proportions]
    
    df = pd.DataFrame(combined_data)
    output_path = os.path.join(output_dir, 'exp1_comparison_results.csv')
    df.to_csv(output_path, index=False)
    print(f"  💾 Comparison CSV saved: {output_path}")


def run_single_model(
    model_path: str,
    model_type: str,
    model_name: str,
    test_trajectories: List[Dict],
    graph: nx.Graph,
    poi_nodes: List[str],
    device: torch.device,
    output_dir: str,
    batch_size: int = 32,
    num_agents: int = 100,
) -> Dict:
    """
    Run experiment 1 for a single model.
    
    Returns:
        Dict of results by proportion
    """
    print(f"\n📥 Loading {model_type} model from {model_path}...")
    
    num_nodes = len(graph.nodes())
    num_poi_nodes = len(poi_nodes)
    
    if model_type == 'transformer':
        model = load_transformer_model(model_path, num_nodes, num_poi_nodes, device)
    elif model_type == 'lstm':
        model = load_lstm_model(model_path, num_nodes, num_poi_nodes, device)
    elif model_type == 'sc_bdi_vae':
        model = load_sc_bdi_vae_model(model_path, num_nodes, num_poi_nodes, num_agents, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"✅ Model loaded successfully")
    
    # Define trajectory proportions to evaluate
    proportions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    
    # Evaluate at each proportion
    results = {}
    print(f"\n📊 Evaluating {model_name} at {len(proportions)} trajectory proportions...")
    
    for proportion in proportions:
        if model_type == 'transformer':
            metrics = evaluate_transformer_at_proportion(
                model, test_trajectories, graph, poi_nodes,
                device, proportion, batch_size
            )
        elif model_type == 'lstm':
            metrics = evaluate_lstm_at_proportion(
                model, test_trajectories, graph, poi_nodes,
                device, proportion, batch_size
            )
        elif model_type == 'sc_bdi_vae':
            metrics = evaluate_sc_bdi_vae_at_proportion(
                model, test_trajectories, graph, poi_nodes,
                device, proportion, batch_size
            )
        
        results[proportion] = metrics
        
        print(f"  {int(proportion*100):3d}%: "
              f"Top-1={metrics['top1']:6.2f}%, "
              f"Top-5={metrics['top5']:6.2f}%, "
              f"Brier={metrics['brier']:.4f}")
    
    # Print results table
    print_results_table(results, proportions, model_name)
    
    # Generate plots
    print(f"\n📊 Generating visualizations for {model_name}...")
    plot_results(results, proportions, output_dir, model_name)
    
    # Save results to CSV
    print(f"\n💾 Saving results for {model_name}...")
    save_results_csv(results, proportions, output_dir, model_name)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Performance vs Trajectory Observation")
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (e.g., checkpoints/keepers/best_model-OURS.pt)')
    parser.add_argument('--model_type', type=str, default=None, 
                        choices=['transformer', 'lstm', 'sc_bdi_vae'],
                        help='Type of model: transformer, lstm, or sc_bdi_vae')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name for output files (default: uses model filename)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all three models (transformer, lstm, sc_bdi_vae)')
    parser.add_argument('--run_dir', type=str, default='data/simulation_data/run_8',
                        help='Path to simulation run directory')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml',
                        help='Path to graph file')
    parser.add_argument('--split_file', type=str, default='data/simulation_data/run_8/split_data/split_indices_seed42.json',
                        help='Path to split indices JSON file')
    parser.add_argument('--output_dir', type=str, default='experiments/results/exp_1',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.run_all and (args.model_path is None or args.model_type is None):
        parser.error("Either --run_all or both --model_path and --model_type are required")
    
    print("\n" + "="*100)
    print("EXPERIMENT 1: MODEL PERFORMANCE VS TRAJECTORY OBSERVATION")
    print("="*100)
    
    # Get device
    device = get_device()
    print(f"🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    all_trajectories, graph, poi_nodes = load_simulation_data(args.run_dir, args.graph_path)
    
    # Load split indices
    print(f"📂 Loading split indices from: {args.split_file}")
    with open(args.split_file, 'r') as f:
        split_indices = json.load(f)
    
    # Get test trajectories using split indices
    test_indices = split_indices['test_indices']
    test_trajectories = [all_trajectories[i] for i in test_indices]
    
    # Ensure agent_id is set for all trajectories
    for i, traj in enumerate(test_trajectories):
        if 'agent_id' not in traj:
            # Derive agent_id from trajectory index (assuming 1000 trajs per agent)
            traj['agent_id'] = test_indices[i] // 1000
    
    print(f"  Test samples: {len(test_trajectories)}")
    print(f"  POI nodes: {len(poi_nodes)}")
    print(f"  Total nodes: {len(graph.nodes())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define models to run
    if args.run_all:
        models_to_run = [
            {
                'path': 'checkpoints/keepers/baseline_transformer_best_model.pt',
                'type': 'transformer',
                'name': 'Transformer'
            },
            {
                'path': 'checkpoints/keepers/lstm_best_model.pt',
                'type': 'lstm',
                'name': 'LSTM'
            },
            {
                'path': 'checkpoints/keepers/scbdi_no_progress.pt',
                'type': 'sc_bdi_vae',
                'name': 'SC-BDI-VAE'
            },
        ]
        print(f"\n🚀 Running all {len(models_to_run)} models...")
    else:
        model_name = args.model_name if args.model_name else Path(args.model_path).stem
        models_to_run = [
            {
                'path': args.model_path,
                'type': args.model_type,
                'name': model_name
            }
        ]
        print(f"\n🚀 Running single model: {model_name} ({args.model_type})")
    
    # Infer number of agents from data
    num_agents = len(set(traj.get('agent_id', 0) for traj in test_trajectories))
    num_agents = max(100, num_agents)  # Default to at least 100
    
    all_results = {}
    
    for model_config in models_to_run:
        print("\n" + "="*80)
        print(f"📊 Evaluating: {model_config['name']}")
        print("="*80)
        
        try:
            results = run_single_model(
                model_path=model_config['path'],
                model_type=model_config['type'],
                model_name=model_config['name'],
                test_trajectories=test_trajectories,
                graph=graph,
                poi_nodes=poi_nodes,
                device=device,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_agents=num_agents,
            )
            all_results[model_config['name']] = results
        except Exception as e:
            print(f"❌ Error running {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison plot if multiple models
    if len(all_results) > 1:
        print(f"\n📊 Generating comparison plots...")
        plot_comparison(all_results, args.output_dir)
    
    print(f"\n✅ Experiment 1 complete!")
    print(f"   Results saved to: {args.output_dir}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
