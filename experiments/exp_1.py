"""
Experiment 1: Model Performance vs Trajectory Observation

This experiment evaluates model performance at different trajectory observation points
(15%, 30%, 45%, 60%, 75%, 90%) and reports:
- Top-1 accuracy
- Top-5 accuracy
- Brier score (calibration metric)

Results are visualized as line charts showing performance progression.

Usage:
    python experiments/exp_1.py --model_path checkpoints/baseline_transformer/best_model.pt --model_type transformer
    python experiments/exp_1.py --model_path checkpoints/baseline_lstm/best_model.pt --model_type lstm
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
    
    # Truncate trajectories and create per-node samples
    per_node_samples = []
    for traj in trajectories:
        path = traj['path']
        if len(path) > 2:
            truncate_idx = max(1, int(len(path) * proportion))
            truncated_path = path[:truncate_idx]
        else:
            truncated_path = path[:1] if len(path) > 0 else path
        
        # Create per-node sample at final position
        if len(truncated_path) > 0:
            per_node_samples.append({
                'history': truncated_path,
                'goal_node': traj['goal_node'],
                'agent_id': traj.get('agent_id', 0),
                'hour': traj.get('hour', 12)
            })
    
    # Create dataset
    dataset = PerNodeTrajectoryDataset(per_node_samples, graph, poi_nodes)
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
        history_indices = batch['history_indices'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        goal_idx = batch['goal_idx'].to(device)
        
        # Forward pass
        predictions = model(history_indices, history_lengths)
        goal_logits = predictions['goal']  # [batch, num_poi_nodes]
        
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
    
    # Create model with same config as training
    model = PerNodeToMPredictor(
        num_nodes=num_nodes,
        num_agents=100,
        num_poi_nodes=num_poi_nodes,
        num_categories=7,
        node_embedding_dim=128,
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


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Performance vs Trajectory Observation")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints/baseline_transformer/best_model.pt)')
    parser.add_argument('--model_type', type=str, required=True, choices=['transformer', 'lstm'],
                        help='Type of model: transformer or lstm')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name for output files (default: uses model filename)')
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
    
    # Determine model name
    if args.model_name is None:
        args.model_name = Path(args.model_path).stem
    
    print("\n" + "="*100)
    print("EXPERIMENT 1: MODEL PERFORMANCE VS TRAJECTORY OBSERVATION")
    print("="*100)
    print(f"Model: {args.model_name} ({args.model_type})")
    print(f"Checkpoint: {args.model_path}")
    print("="*100 + "\n")
    
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
    
    print(f"  Test samples: {len(test_trajectories)}")
    print(f"  POI nodes: {len(poi_nodes)}")
    print(f"  Total nodes: {len(graph.nodes())}")
    
    # Load model
    print(f"\n📥 Loading {args.model_type} model...")
    if args.model_type == 'transformer':
        model = load_transformer_model(args.model_path, len(graph.nodes()), len(poi_nodes), device)
    else:
        model = load_lstm_model(args.model_path, len(graph.nodes()), len(poi_nodes), device)
    
    print(f"✅ Model loaded successfully")
    
    # Define trajectory proportions to evaluate
    proportions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    
    # Evaluate at each proportion
    results = {}
    print(f"\n📊 Evaluating at {len(proportions)} trajectory proportions...")
    
    for proportion in proportions:
        if args.model_type == 'transformer':
            metrics = evaluate_transformer_at_proportion(
                model, test_trajectories, graph, poi_nodes,
                device, proportion, args.batch_size
            )
        else:
            metrics = evaluate_lstm_at_proportion(
                model, test_trajectories, graph, poi_nodes,
                device, proportion, args.batch_size
            )
        
        results[proportion] = metrics
        
        print(f"  {int(proportion*100):3d}%: "
              f"Top-1={metrics['top1']:6.2f}%, "
              f"Top-5={metrics['top5']:6.2f}%, "
              f"Brier={metrics['brier']:.4f}")
    
    # Print results table
    print_results_table(results, proportions, args.model_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print(f"\n📊 Generating visualizations...")
    plot_results(results, proportions, args.output_dir, args.model_name)
    
    # Save results to CSV
    print(f"\n💾 Saving results...")
    save_results_csv(results, proportions, args.output_dir, args.model_name)
    
    print(f"\n✅ Experiment 1 complete!")
    print(f"   Results saved to: {args.output_dir}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
