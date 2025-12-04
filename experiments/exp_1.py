"""
Experiment 1: Model Performance vs Trajectory Observation

Evaluates how model accuracy changes as it observes more of the trajectory.
Tests at 15%, 30%, 45%, 60%, 75%, 90% observation points.
"""

import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils.utils import get_device, compute_accuracy
from models.data_loader_utils.data_loader import load_simulation_data, split_data, create_dataloaders
from models.baseline_transformer.transformer_predictor import GoalPredictionModel
from models.fusion_encoders_preprocessing.fusion_encoder import ToMGraphEncoder
from models.node2vec_preprocessing.node_embeddings import Node2VecEmbeddings
from graph_controller.world_graph import WorldGraph


def compute_brier_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Brier score (lower is better)."""
    probs = F.softmax(logits, dim=-1)
    num_classes = logits.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    brier = torch.mean((probs - targets_one_hot) ** 2).item()
    return brier


@torch.no_grad()
def evaluate_at_proportion(
    model: torch.nn.Module,
    data_loader,
    graph_data: dict,
    device: torch.device,
    proportion: float,
    debug: bool = False
) -> Dict[str, float]:
    """
    Evaluate model when observing only a proportion of trajectories.
    Simply truncates the mask to hide later parts of the trajectory.
    """
    model.eval()
    
    all_logits = []
    all_targets = []
    
    batch_idx = 0
    for batch in tqdm(data_loader, desc=f'{int(proportion*100):>3}%', leave=False):
        # Get the mask and compute actual sequence lengths
        mask = batch['mask']  # (batch, seq_len)
        seq_lens = mask.sum(dim=1).long()  # (batch,)
        
        # Calculate truncated lengths (at least 1)
        trunc_lens = torch.maximum(
            torch.ones_like(seq_lens),
            (seq_lens.float() * proportion).long()
        )
        
        # Create truncated mask: 1 for first N positions, 0 for rest
        mask_trunc = torch.zeros_like(mask)
        for i, trunc_len in enumerate(trunc_lens):
            mask_trunc[i, :trunc_len] = 1.0
        
        # Debug first batch
        if debug and batch_idx == 0:
            print(f"\n  DEBUG at {int(proportion*100)}%:")
            for i in range(min(3, len(seq_lens))):
                orig_len = seq_lens[i].item()
                trunc_len = trunc_lens[i].item()
                actual_pct = (trunc_len / orig_len * 100) if orig_len > 0 else 0
                print(f"    Sample {i}: {orig_len} nodes → {trunc_len} nodes ({actual_pct:.1f}%)")
        
        batch_idx += 1
        
        # Prepare batch with truncated mask
        traj_batch = {
            'node_embeddings': batch['node_embeddings'].to(device),
            'hour': batch['hour'].to(device),
            'agent_id': batch['agent_id'].to(device),
            'mask': mask_trunc.to(device)  # Use truncated mask
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
    
    return {'top1': top1, 'top5': top5, 'brier': brier}


def plot_results(results: Dict, proportions: List[float], output_dir: str):
    """Generate line charts showing accuracy progression."""
    os.makedirs(output_dir, exist_ok=True)
    
    proportions_pct = [p * 100 for p in proportions]
    
    metrics = [
        ('top1', 'Top-1 Accuracy (%)', 'Higher is Better'),
        ('top5', 'Top-5 Accuracy (%)', 'Higher is Better'),
        ('brier', 'Brier Score', 'Lower is Better')
    ]
    
    for metric_key, metric_name, direction in metrics:
        plt.figure(figsize=(10, 6))
        
        for model_name, model_results in results.items():
            values = [model_results[p][metric_key] for p in proportions]
            plt.plot(proportions_pct, values, marker='o', linewidth=2, markersize=8, label=model_name)
        
        plt.xlabel('Trajectory Observed (%)', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} vs Trajectory Observation\n({direction})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(proportions_pct)
        
        if metric_key in ['top1', 'top5']:
            plt.ylim(0, 100)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{metric_key}_vs_observation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def save_results_csv(results: Dict, proportions: List[float], output_dir: str):
    """Save results to CSV."""
    import csv
    
    csv_path = os.path.join(output_dir, 'exp1_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Observation (%)', 'Top-1 (%)', 'Top-5 (%)', 'Brier Score'])
        
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
    checkpoint_path = "checkpoints/incremental_training/best_model.pt"
    output_dir = "data/simulation_data/run_8/visualizations/exp_1"
    proportions = [.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    
    device = get_device()
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    trajectories, graph, poi_nodes = load_simulation_data(run_dir, graph_path)
    train_trajs, val_trajs, test_trajs = split_data(trajectories, seed=42, run_dir=run_dir)
    print(f"  Test: {len(test_trajs)} trajectories")
    
    # Load Node2Vec embeddings
    print(f"\n🔢 Loading Node2Vec embeddings...")
    embeddings_path = Path("data/processed/node2vec_embeddings.pkl")
    node_emb_manager = Node2VecEmbeddings(embedding_dim=128)
    node_emb_manager.load(str(embeddings_path))
    print(f"  ✓ Loaded: {node_emb_manager.embedding_matrix.shape}") # type: ignore
    
    # Check trajectory lengths
    traj_lengths = [len(t['path']) for t in test_trajs]
    print(f"  ✓ Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, "
          f"mean={sum(traj_lengths)/len(traj_lengths):.1f}")
    
    # Create test dataloader (without incremental training - we want full trajectories)
    print(f"\n📦 Creating dataloaders...")
    _, _, test_loader, num_agents = create_dataloaders(
        train_trajs, val_trajs, test_trajs,
        graph, poi_nodes,
        batch_size=32,
        num_workers=0,
        node_embeddings=node_emb_manager,
        incremental_training=False  # Full trajectories
    )
    print(f"  ✓ Test batches: {len(test_loader)}, Agents: {num_agents}")
    
    # Prepare graph data
    all_nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    edge_list = list(graph.edges())
    edge_index = torch.tensor([
        [node_to_idx[e[0]] for e in edge_list],
        [node_to_idx[e[1]] for e in edge_list]
    ], dtype=torch.long)
    
    graph_data = {
        'node_embeddings': node_emb_manager.embedding_matrix.to(device), # type: ignore
        'edge_index': edge_index.to(device)
    }
    
    # Load model
    print(f"\n📥 Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model (using standard hyperparameters)
    fusion_encoder = ToMGraphEncoder(
        node_emb_dim=128,
        hidden_dim=64,
        num_agents=num_agents,
        output_dim=64,
        n_layers=1,
        n_heads=4,
        dropout=0.3
    )
    
    model = GoalPredictionModel(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=len(poi_nodes),
        fusion_dim=64,
        hidden_dim=32,
        n_transformer_layers=0,
        n_heads=4,
        dropout=0.3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Loaded (Epoch {checkpoint.get('epoch', '?')})")
    
    # Evaluate at each proportion
    print(f"\n📊 Evaluating at {len(proportions)} proportions...")
    results = {'best_model': {}}
    
    for idx, proportion in enumerate(proportions):
        debug_mode = True  # Debug first proportion only
        metrics = evaluate_at_proportion(model, test_loader, graph_data, device, proportion, debug=debug_mode)
        results['best_model'][proportion] = metrics
        print(f"  {int(proportion*100):3d}%: Top-1={metrics['top1']:6.2f}%, "
              f"Top-5={metrics['top5']:6.2f}%, Brier={metrics['brier']:.4f}")
    
    # Save results
    print(f"\n💾 Saving results to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    plot_results(results, proportions, output_dir)
    save_results_csv(results, proportions, output_dir)
    
    print(f"\n✅ Experiment complete!")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
