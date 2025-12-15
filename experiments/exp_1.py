"""
Experiment 1: Model Performance vs Trajectory Observation

This experiment evaluates model performance at different trajectory observation points
(15%, 30%, 45%, 60%, 75%, 90%) and reports:
- Top-1 accuracy
- Top-5 accuracy
- Brier score (calibration metric)

Results are visualized as line charts showing performance progression.
"""

import os
import sys

# Set MPS fallback for Mac compatibility BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archive.utils import get_device, load_checkpoint, compute_accuracy
from archive.data_loader import load_simulation_data, split_data, create_dataloaders
from models.transformer_predictor import GoalPredictionModel
from models.encoders.trajectory_encoder import TrajectoryDataPreparator
from models.encoders.map_encoder import GraphDataPreparator
from models.encoders.fusion_encoder import ToMGraphEncoder
from graph_controller.world_graph import WorldGraph


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


@torch.no_grad()
def evaluate_at_proportion(
    model: torch.nn.Module,
    test_loader,
    trajectory_prep: TrajectoryDataPreparator,
    graph_data: dict,
    device: torch.device,
    proportion: float
) -> Dict[str, float]:
    """
    Evaluate model performance at a specific trajectory proportion.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        trajectory_prep: Trajectory preparator
        graph_data: Graph data
        device: Device to run on
        proportion: Proportion of trajectory to observe (e.g., 0.15 for 15%)
    
    Returns:
        Dict with top1, top5, and brier scores
    """
    model.eval()
    
    all_logits = []
    all_targets = []
    
    for batch in tqdm(test_loader, desc=f'{int(proportion*100)}% trajectory', leave=False):
        # Truncate trajectories to proportion
        truncated_trajs = []
        for i in range(len(batch['trajectories'])):
            path = batch['trajectories'][i]
            if len(path) > 2:
                truncate_idx = max(1, int(len(path) * proportion))
                truncated_path = path[:truncate_idx]
            else:
                truncated_path = path[:1] if len(path) > 0 else path
            
            truncated_trajs.append({
                'path': truncated_path,
                'hour': batch['hours'][i],
                'goal_node': batch['goal_nodes'][i]
            })
        
        # Prepare batch
        traj_batch = trajectory_prep.prepare_batch(truncated_trajs)
        
        # Move to device
        for key in traj_batch:
            if isinstance(traj_batch[key], torch.Tensor):
                traj_batch[key] = traj_batch[key].to(device)
        
        targets = batch['goal_indices'].to(device)
        
        # Forward pass
        logits = model(traj_batch, graph_data, return_logits=True)
        
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    top1 = compute_accuracy(all_logits, all_targets, k=1)
    top5 = compute_accuracy(all_logits, all_targets, k=5)
    brier = compute_brier_score(all_logits, all_targets)
    
    return {
        'top1': top1,
        'top5': top5,
        'brier': brier
    }


def load_model_from_checkpoint(
    checkpoint_path: str,
    num_poi_nodes: int,
    num_nodes: int,
    graph_node_feat_dim: int,
    device: torch.device
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_poi_nodes: Number of POI nodes
        num_nodes: Total number of graph nodes
        graph_node_feat_dim: Dimension of graph node features
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Load checkpoint to get model hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters from checkpoint (stored during training)
    # Default values match baseline configuration
    fusion_dim = 64
    transformer_dim = 128
    num_transformer_layers = 1
    num_heads = 4
    dropout = 0.1
    
    # Initialize fusion encoder
    fusion_encoder = ToMGraphEncoder(
        num_nodes=num_nodes,
        graph_node_feat_dim=graph_node_feat_dim,
        traj_node_emb_dim=32,
        hidden_dim=64,
        output_dim=fusion_dim,
        n_layers=2,
        n_heads=4,
        dropout=dropout
    )
    
    # Initialize goal prediction model
    model = GoalPredictionModel(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=num_poi_nodes,
        fusion_dim=fusion_dim,
        hidden_dim=transformer_dim,
        n_transformer_layers=num_transformer_layers,
        n_heads=num_heads,
        dropout=dropout
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def plot_results(
    results: Dict[str, Dict[float, Dict[str, float]]],
    proportions: List[float],
    output_dir: str
):
    """
    Generate line charts for each metric across trajectory proportions.
    
    Args:
        results: Dict mapping model_name -> {proportion -> {metric -> value}}
        proportions: List of trajectory proportions evaluated
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    metrics = [
        ('top1', 'Top-1 Accuracy (%)', 'Higher is Better'),
        ('top5', 'Top-5 Accuracy (%)', 'Higher is Better'),
        ('brier', 'Brier Score', 'Lower is Better')
    ]
    
    # Convert proportions to percentages for x-axis
    proportions_pct = [p * 100 for p in proportions]
    
    for metric_key, metric_name, direction in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot each model
        for model_name, model_results in results.items():
            values = [model_results[p][metric_key] for p in proportions]
            plt.plot(proportions_pct, values, marker='o', linewidth=2, markersize=8, label=model_name)
        
        plt.xlabel('Trajectory Observed (%)', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} vs Trajectory Observation\n({direction})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(proportions_pct)
        
        # Set y-axis limits based on metric
        if metric_key in ['top1', 'top5']:
            plt.ylim(0, 100)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'{metric_key}_vs_observation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def print_results_table(
    results: Dict[str, Dict[float, Dict[str, float]]],
    proportions: List[float]
):
    """
    Print results as formatted tables.
    
    Args:
        results: Dict mapping model_name -> {proportion -> {metric -> value}}
        proportions: List of trajectory proportions evaluated
    """
    print("\n" + "=" * 100)
    print("EXPERIMENT 1 RESULTS")
    print("=" * 100)
    
    for model_name, model_results in results.items():
        print(f"\n📊 Model: {model_name}")
        print("-" * 100)
        print(f"{'Observation':<15} {'Top-1 (%)':<12} {'Top-5 (%)':<12} {'Brier Score':<12}")
        print("-" * 100)
        
        for proportion in proportions:
            metrics = model_results[proportion]
            print(f"{int(proportion*100):>3}% observed   "
                  f"{metrics['top1']:>10.2f}   "
                  f"{metrics['top5']:>10.2f}   "
                  f"{metrics['brier']:>10.4f}")
        
        print("-" * 100)
    
    print("\n" + "=" * 100)


def save_results_csv(
    results: Dict[str, Dict[float, Dict[str, float]]],
    proportions: List[float],
    output_dir: str
):
    """
    Save results to CSV file.
    
    Args:
        results: Dict mapping model_name -> {proportion -> {metric -> value}}
        proportions: List of trajectory proportions evaluated
        output_dir: Directory to save CSV
    """
    import csv
    
    csv_path = os.path.join(output_dir, 'exp1_results.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Model', 'Observation (%)', 'Top-1 (%)', 'Top-5 (%)', 'Brier Score'])
        
        # Data rows
        for model_name, model_results in results.items():
            for proportion in proportions:
                metrics = model_results[proportion]
                writer.writerow([
                    model_name,
                    int(proportion * 100),
                    f"{metrics['top1']:.2f}",
                    f"{metrics['top5']:.2f}",
                    f"{metrics['brier']:.4f}"
                ])
    
    print(f"  ✓ Saved: {csv_path}")


def main():
    print("\n" + "=" * 100)
    print("EXPERIMENT 1: Model Performance vs Trajectory Observation")
    print("=" * 100)
    
    # Configuration
    run_dir = "data/simulation_data/run_8"
    graph_path = "data/processed/ucsd_walk_full.graphml"
    keepers_dir = "checkpoints/keepers"
    output_dir = "data/simulation_data/run_8/visualizations/exp_1"
    
    # Trajectory proportions to evaluate
    proportions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    
    print(f"\n📁 Configuration:")
    print(f"  Data: {run_dir}")
    print(f"  Graph: {graph_path}")
    print(f"  Models: {keepers_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Proportions: {[int(p*100) for p in proportions]}%")
    
    # Get device
    device = get_device()
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    trajectories, graph, poi_nodes = load_simulation_data(run_dir, graph_path)
    
    # Split data
    train_trajs, val_trajs, test_trajs = split_data(
        trajectories,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print(f"  Train: {len(train_trajs)}, Val: {len(val_trajs)}, Test: {len(test_trajs)}")
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        train_trajs, val_trajs, test_trajs,
        graph, poi_nodes,
        batch_size=32,
        num_workers=0
    )
    
    # Initialize data preparators
    world_graph = WorldGraph(graph)
    all_nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    trajectory_prep = TrajectoryDataPreparator(node_to_idx)
    graph_prep = GraphDataPreparator(world_graph)
    graph_data_dict = graph_prep.prepare_graph_data()
    
    # Prepare graph data for device
    graph_data = {
        'x': graph_data_dict['x'].to(device),
        'edge_index': graph_data_dict['edge_index'].to(device)
    }
    
    # Get model configuration details
    num_nodes = len(graph.nodes())
    num_poi_nodes = len(poi_nodes)
    graph_node_feat_dim = graph_data_dict['x'].shape[1]
    
    print(f"  POI nodes: {num_poi_nodes}")
    print(f"  Total nodes: {num_nodes}")
    
    # Find all model checkpoints in keepers directory
    keepers_path = Path(keepers_dir)
    if not keepers_path.exists():
        print(f"\n❌ Error: Keepers directory not found: {keepers_dir}")
        print(f"   Please train models and save them to {keepers_dir}")
        return
    
    model_files = list(keepers_path.glob("*.pt"))
    if not model_files:
        print(f"\n❌ Error: No model checkpoints found in {keepers_dir}")
        print(f"   Please save trained models (*.pt) to this directory")
        return
    
    print(f"\n🔍 Found {len(model_files)} model(s):")
    for model_file in model_files:
        print(f"  - {model_file.name}")
    
    # Evaluate each model
    results = {}
    
    for model_file in model_files:
        model_name = model_file.stem  # Filename without extension
        print(f"\n{'='*100}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*100}")
        
        # Load model
        print(f"📥 Loading model from {model_file}...")
        model = load_model_from_checkpoint(
            str(model_file),
            num_poi_nodes,
            num_nodes,
            graph_node_feat_dim,
            device
        )
        
        # Evaluate at each proportion
        model_results = {}
        print(f"\n📊 Evaluating at {len(proportions)} trajectory proportions...")
        
        for proportion in proportions:
            metrics = evaluate_at_proportion(
                model, test_loader, trajectory_prep,
                graph_data, device, proportion
            )
            model_results[proportion] = metrics
            
            print(f"  {int(proportion*100):3d}%: "
                  f"Top-1={metrics['top1']:6.2f}%, "
                  f"Top-5={metrics['top5']:6.2f}%, "
                  f"Brier={metrics['brier']:.4f}")
        
        results[model_name] = model_results
    
    # Print results table
    print_results_table(results, proportions)
    
    # Generate plots
    print(f"\n📊 Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    plot_results(results, proportions, output_dir)
    
    # Save results to CSV
    print(f"\n💾 Saving results...")
    save_results_csv(results, proportions, output_dir)
    
    print(f"\n✅ Experiment 1 complete!")
    print(f"   Results saved to: {output_dir}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
