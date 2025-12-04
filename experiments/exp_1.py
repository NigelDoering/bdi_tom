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

from models.utils.utils import get_device, load_checkpoint, compute_accuracy
from models.data_loader_utils.data_loader import load_simulation_data, split_data, create_dataloaders
from models.baseline_transformer.transformer_predictor import GoalPredictionModel
from models.fusion_encoders_preprocessing.fusion_encoder import ToMGraphEncoder
from models.node2vec_preprocessing.node_embeddings import Node2VecEmbeddings
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
    graph_data: dict,
    device: torch.device,
    proportion: float
) -> Dict[str, float]:
    """
    Evaluate model performance at a specific trajectory proportion.
    
    Args:
        model: Trained model
        test_loader: Test data loader (with Node2Vec embeddings)
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
        # Get node embeddings and mask
        node_embeddings = batch['node_embeddings']  # (batch, max_seq_len, emb_dim)
        mask = batch['mask']  # (batch, max_seq_len)
        
        # Get actual sequence lengths
        seq_lens = mask.sum(dim=1).long()  # (batch,)
        
        # Calculate truncated lengths based on proportion
        truncated_lens = torch.maximum(
            torch.ones_like(seq_lens),
            (seq_lens.float() * proportion).long()
        )
        
        # Create truncated mask
        mask_truncated = torch.zeros_like(mask)
        for i, trunc_len in enumerate(truncated_lens):
            mask_truncated[i, :trunc_len] = 1.0
        
        # Prepare batch with truncated mask
        traj_batch = {
            'node_embeddings': node_embeddings.to(device),
            'hour': batch['hour'].to(device),
            'agent_id': batch['agent_id'].to(device),
            'mask': mask_truncated.to(device)
        }
        
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
    node_emb_dim: int,
    num_agents: int,
    device: torch.device
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_poi_nodes: Number of POI nodes
        node_emb_dim: Node2Vec embedding dimension (typically 128)
        num_agents: Number of agents in the simulation
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters from checkpoint or use defaults
    # These match the typical training configuration
    fusion_dim = 64
    encoder_hidden_dim = 64
    predictor_hidden_dim = 32
    num_encoder_layers = 1
    num_transformer_layers = 0  # Typically 0 in current config
    num_heads = 4
    dropout = 0.3
    
    # Initialize fusion encoder with new API
    fusion_encoder = ToMGraphEncoder(
        node_emb_dim=node_emb_dim,
        hidden_dim=encoder_hidden_dim,
        num_agents=num_agents,
        output_dim=fusion_dim,
        n_layers=num_encoder_layers,
        n_heads=num_heads,
        dropout=dropout
    )
    
    # Initialize goal prediction model
    model = GoalPredictionModel(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=num_poi_nodes,
        fusion_dim=fusion_dim,
        hidden_dim=predictor_hidden_dim,
        n_transformer_layers=num_transformer_layers,
        n_heads=num_heads,
        dropout=dropout
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  ✓ Checkpoint metrics: Val Top-1={metrics.get('val_top1', '?'):.2f}%")
    
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
    keepers_dir = "checkpoints/incremental_training"  # Directory containing trained models
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
    
    # Load pre-computed Node2Vec embeddings (same ones used during training)
    print(f"\n🔢 Loading Node2Vec embeddings...")
    embeddings_path = Path("data/processed/node2vec_embeddings.pkl")
    
    if not embeddings_path.exists():
        print(f"\n❌ Error: Node2Vec embeddings not found at {embeddings_path}")
        print(f"   These embeddings should have been created during training.")
        print(f"   Please ensure the model was trained with the correct embeddings.")
        return
    
    print(f"  Loading from: {embeddings_path}")
    node_emb_manager = Node2VecEmbeddings(embedding_dim=128)
    node_emb_manager.load(str(embeddings_path))
    node_embeddings: torch.Tensor = node_emb_manager.embedding_matrix  # type: ignore
    node_emb_dim = node_embeddings.shape[1]
    
    print(f"  ✓ Node embeddings: {node_embeddings.shape}")
    
    # Create test dataloader with Node2Vec embeddings manager (not just the matrix)
    _, _, test_loader, num_agents = create_dataloaders(
        train_trajs, val_trajs, test_trajs,
        graph, poi_nodes,
        batch_size=32,
        num_workers=0,
        node_embeddings=node_emb_manager,  # Pass the manager, not the matrix
        incremental_training=True
    )
    
    # Prepare world graph for evaluation
    world_graph = WorldGraph(graph)
    
    # Create node to index mapping (should match the Node2Vec embedding order)
    all_nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Extract edge connectivity from graph and convert to indices
    edge_list = list(graph.edges())
    edge_index = torch.tensor([
        [node_to_idx[e[0]] for e in edge_list],
        [node_to_idx[e[1]] for e in edge_list]
    ], dtype=torch.long)
    
    # Prepare graph data structure (node embeddings + edge connectivity)
    graph_data = {
        'node_embeddings': node_embeddings.to(device),
        'edge_index': edge_index.to(device)
    }
    
    # Model configuration
    num_poi_nodes = len(poi_nodes)
    
    print(f"  POI nodes: {num_poi_nodes}")
    print(f"  Total agents: {num_agents}")
    print(f"  Node embedding dim: {node_emb_dim}")
    
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
            node_emb_dim,
            num_agents,
            device
        )
        
        # Evaluate at each proportion
        model_results = {}
        print(f"\n📊 Evaluating at {len(proportions)} trajectory proportions...")
        
        for proportion in proportions:
            metrics = evaluate_at_proportion(
                model, test_loader,
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
