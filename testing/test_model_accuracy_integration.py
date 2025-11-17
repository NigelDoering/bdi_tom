"""
Integration test for model accuracy computation.

This script:
1. Loads the trained model checkpoint
2. Loads some real trajectory data
3. Runs inference to get predictions
4. Verifies the accuracy computation makes sense with real data
"""
import os
import sys

# Set MPS fallback for Mac compatibility BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import networkx as nx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.training.utils import compute_accuracy, load_checkpoint, get_device
from models.training.data_loader import load_simulation_data, split_data, create_dataloaders
from models.transformer_predictor import GoalPredictionModel
from models.encoders.trajectory_encoder import TrajectoryDataPreparator
from models.encoders.map_encoder import GraphDataPreparator
from models.encoders.fusion_encoder import ToMGraphEncoder
from graph_controller.world_graph import WorldGraph


def main():
    print("\n" + "="*80)
    print("🧪 MODEL ACCURACY INTEGRATION TEST")
    print("="*80 + "\n")
    
    # Paths
    checkpoint_path = "checkpoints/baseline_transformer/best_model.pt"
    run_dir = "data/simulation_data/run_8"
    graph_path = "data/processed/ucsd_walk_full.graphml"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print(f"   Please train the model first or specify a different checkpoint path.")
        return
    
    print(f"📂 Loading data from {run_dir}")
    print(f"📂 Loading graph from {graph_path}\n")
    
    # Set device
    device = get_device()
    print(f"🖥️  Using device: {device}\n")
    
    # Load data
    trajectories, graph, poi_nodes = load_simulation_data(run_dir, graph_path)
    print(f"✓ Loaded {len(trajectories)} trajectories")
    print(f"✓ Loaded graph with {graph.number_of_nodes()} nodes\n")
    
    # Get POI nodes and world graph
    world_graph = WorldGraph(graph)
    num_poi_nodes = len(poi_nodes)
    print(f"✓ Found {num_poi_nodes} POI nodes\n")
    
    # Split data (just to get a small test set)
    train_data, val_data, test_data = split_data(
        trajectories, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15,
        seed=42
    )
    print(f"✓ Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test\n")
    
    # Create data loaders (use small batch for testing)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        graph, poi_nodes,
        batch_size=8,  # Small batch for testing
        num_workers=0
    )
    
    # Create node-to-index mapping
    all_nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Prepare data preparators
    trajectory_prep = TrajectoryDataPreparator(node_to_idx)
    graph_prep = GraphDataPreparator(world_graph)
    graph_data = graph_prep.prepare_graph_data()
    
    print(f"✓ Created data loaders (batch_size=8)")
    print(f"✓ Prepared graph data\n")
    
    # Get dimensions for model initialization
    num_nodes = len(graph.nodes())
    graph_node_feat_dim = graph_data['x'].shape[1]  # 12 features per node
    
    # Initialize model architecture (matching training config)
    fusion_encoder = ToMGraphEncoder(
        num_nodes=num_nodes,
        graph_node_feat_dim=graph_node_feat_dim,
        traj_node_emb_dim=32,
        hidden_dim=64,
        output_dim=64,  # fusion_dim
        n_layers=2,
        n_heads=4,
        dropout=0.1
    )
    
    model = GoalPredictionModel(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=num_poi_nodes,
        fusion_dim=64,
        hidden_dim=128,
        n_transformer_layers=1,
        n_heads=4,
        dropout=0.1
    )
    
    print(f"✓ Initialized model architecture\n")
    
    # Load checkpoint
    print(f"📥 Loading checkpoint from {checkpoint_path}")
    epoch, loss, metrics = load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded successfully!")
    print(f"   Checkpoint metrics: {metrics}\n")
    
    # Get a batch of test data
    print("="*80)
    print("🔍 TESTING WITH REAL DATA")
    print("="*80 + "\n")
    
    test_iter = iter(test_loader)
    batch = next(test_iter)
    
    batch_size = len(batch['trajectories'])
    print(f"📦 Test batch size: {batch_size}")
    print(f"   Sample trajectory lengths: {[len(p) for p in batch['trajectories'][:5]]}")
    print(f"   Sample goals: {batch['goal_indices'][:5].tolist()}\n")
    
    # Prepare batch data
    traj_dicts = []
    for i in range(len(batch['trajectories'])):
        traj_dicts.append({
            'path': batch['trajectories'][i],
            'hour': batch['hours'][i],
            'goal_node': batch['goal_nodes'][i]
        })
    
    traj_batch = trajectory_prep.prepare_batch(traj_dicts)
    
    # Move to device
    for key in traj_batch:
        if isinstance(traj_batch[key], torch.Tensor):
            traj_batch[key] = traj_batch[key].to(device)
    
    # Get true targets
    targets = batch['goal_indices'].to(device)
    
    print(f"✓ Prepared batch:")
    print(f"   Trajectory batch keys: {list(traj_batch.keys())}")
    print(f"   Targets: {targets.shape}\n")
    
    # Prepare graph data for device
    graph_data_device = {}
    for key, value in graph_data.items():
        if isinstance(value, torch.Tensor):
            graph_data_device[key] = value.to(device)
        else:
            graph_data_device[key] = value
    
    # Run inference
    with torch.no_grad():
        logits = model(traj_batch, graph_data_device, return_logits=True)
    
    print(f"✓ Model predictions:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Expected: (batch_size={batch_size}, num_poi_nodes={num_poi_nodes})")
    print(f"   Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]\n")
    
    # Get top predictions
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_indices = probs.topk(5, dim=-1)
    
    print("="*80)
    print("📊 DETAILED SAMPLE ANALYSIS")
    print("="*80 + "\n")
    
    # Show details for first 3 samples
    num_samples_to_show = min(3, batch_size)
    for i in range(num_samples_to_show):
        true_goal = targets[i].item()
        top5_preds = top5_indices[i].tolist()
        top5_prob_vals = top5_probs[i].tolist()
        
        print(f"Sample {i}:")
        print(f"  True goal index: {true_goal}")
        print(f"  Top-5 predictions:")
        for rank, (pred_idx, prob) in enumerate(zip(top5_preds, top5_prob_vals), 1):
            marker = "✓" if pred_idx == true_goal else " "
            print(f"    {marker} Rank {rank}: POI {pred_idx:3d} (prob: {prob:.4f})")
        
        # Check if true goal is in top-5
        in_top5 = true_goal in top5_preds
        print(f"  True goal in top-5: {'✓ Yes' if in_top5 else '✗ No'}\n")
    
    print("="*80)
    print("🎯 ACCURACY COMPUTATION")
    print("="*80 + "\n")
    
    # Compute accuracies
    top1_acc = compute_accuracy(logits, targets, k=1)
    top5_acc = compute_accuracy(logits, targets, k=5)
    
    print(f"Batch Accuracy (on {batch_size} samples):")
    print(f"  Top-1: {top1_acc:.2f}%")
    print(f"  Top-5: {top5_acc:.2f}%\n")
    
    # Verify computation manually
    print("="*80)
    print("✅ MANUAL VERIFICATION")
    print("="*80 + "\n")
    
    # Top-1: Check if argmax equals target
    top1_preds = logits.argmax(dim=-1)
    manual_top1_correct = (top1_preds == targets).sum().item()
    manual_top1_acc = (manual_top1_correct / batch_size) * 100
    
    print(f"Manual Top-1 computation:")
    print(f"  Correct predictions: {manual_top1_correct}/{batch_size}")
    print(f"  Accuracy: {manual_top1_acc:.2f}%")
    print(f"  Function result: {top1_acc:.2f}%")
    print(f"  Match: {'✓ Yes' if abs(manual_top1_acc - top1_acc) < 0.01 else '✗ NO'}\n")
    
    # Top-5: Check if target in top-5
    _, top5_all = logits.topk(5, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top5_all)
    manual_top5_correct = (top5_all == targets_expanded).any(dim=-1).sum().item()
    manual_top5_acc = (manual_top5_correct / batch_size) * 100
    
    print(f"Manual Top-5 computation:")
    print(f"  Correct predictions: {manual_top5_correct}/{batch_size}")
    print(f"  Accuracy: {manual_top5_acc:.2f}%")
    print(f"  Function result: {top5_acc:.2f}%")
    print(f"  Match: {'✓ Yes' if abs(manual_top5_acc - top5_acc) < 0.01 else '✗ NO'}\n")
    
    # Verify relationships
    print("="*80)
    print("🔍 SANITY CHECKS")
    print("="*80 + "\n")
    
    checks_passed = True
    
    # Check 1: Top-5 >= Top-1
    if top5_acc >= top1_acc:
        print("✓ Top-5 accuracy >= Top-1 accuracy (as expected)")
    else:
        print("✗ ERROR: Top-5 accuracy < Top-1 accuracy (should be impossible!)")
        checks_passed = False
    
    # Check 2: Accuracies in valid range [0, 100]
    if 0 <= top1_acc <= 100 and 0 <= top5_acc <= 100:
        print("✓ Accuracies in valid range [0%, 100%]")
    else:
        print(f"✗ ERROR: Accuracies out of range: Top-1={top1_acc}, Top-5={top5_acc}")
        checks_passed = False
    
    # Check 3: Logits shape is correct
    if logits.shape == (batch_size, num_poi_nodes):
        print(f"✓ Logits shape is correct: {logits.shape}")
    else:
        print(f"✗ ERROR: Logits shape mismatch: {logits.shape} vs expected ({batch_size}, {num_poi_nodes})")
        checks_passed = False
    
    # Check 4: Probabilities sum to ~1
    prob_sums = probs.sum(dim=-1)
    all_close_to_one = torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    if all_close_to_one:
        print(f"✓ Softmax probabilities sum to 1.0")
    else:
        print(f"✗ ERROR: Probabilities don't sum to 1: {prob_sums[:5].tolist()}")
        checks_passed = False
    
    # Check 5: Manual and function computations match
    if abs(manual_top1_acc - top1_acc) < 0.01 and abs(manual_top5_acc - top5_acc) < 0.01:
        print("✓ Manual computation matches function results")
    else:
        print("✗ ERROR: Manual and function computations don't match!")
        checks_passed = False
    
    print("\n" + "="*80)
    if checks_passed:
        print("🎉 ALL CHECKS PASSED! Accuracy metrics are working correctly.")
    else:
        print("❌ SOME CHECKS FAILED! Review errors above.")
    print("="*80 + "\n")
    
    # Test on a larger sample
    print("="*80)
    print("📈 TESTING ON FULL TEST SET")
    print("="*80 + "\n")
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Prepare data
            traj_dicts = []
            for i in range(len(batch['trajectories'])):
                traj_dicts.append({
                    'path': batch['trajectories'][i],
                    'hour': batch['hours'][i],
                    'goal_node': batch['goal_nodes'][i]
                })
            
            traj_batch = trajectory_prep.prepare_batch(traj_dicts)
            
            # Move to device
            for key in traj_batch:
                if isinstance(traj_batch[key], torch.Tensor):
                    traj_batch[key] = traj_batch[key].to(device)
            
            targets_batch = batch['goal_indices'].to(device)
            
            # Run inference
            logits = model(traj_batch, graph_data_device, return_logits=True)
            
            all_logits.append(logits.cpu())
            all_targets.append(targets_batch.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")
    
    # Concatenate all results
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"\n✓ Processed entire test set: {len(all_targets)} samples\n")
    
    # Compute overall test accuracy
    test_top1 = compute_accuracy(all_logits, all_targets, k=1)
    test_top5 = compute_accuracy(all_logits, all_targets, k=5)
    
    print(f"Full Test Set Accuracy:")
    print(f"  Top-1: {test_top1:.2f}%")
    print(f"  Top-5: {test_top5:.2f}%\n")
    
    # Compare to random baseline
    random_baseline = (1.0 / num_poi_nodes) * 100
    improvement = test_top1 / random_baseline
    
    print(f"Comparison to Random Baseline:")
    print(f"  Random guessing: {random_baseline:.2f}%")
    print(f"  Model top-1: {test_top1:.2f}%")
    print(f"  Improvement: {improvement:.1f}x better than random\n")
    
    print("="*80)
    print("✅ Integration test complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained a model (checkpoint exists)")
        print("  2. Data in the correct location")
        print("  3. Graph file available\n")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
