"""
End-to-end training script for BDI-ToM goal prediction model.

This script trains the complete pipeline:
- TrajectoryEncoder (transformer-based)
- WorldGraphEncoder (GAT-based)
- ToMGraphEncoder (fusion layer)
- GoalPredictionModel (transformer + classifier)

All components are trained end-to-end with gradient flow through the entire pipeline.
"""

import os
import sys

# Set MPS fallback for Mac compatibility BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.training.utils import (
    get_device, set_seed, save_checkpoint, load_checkpoint,
    compute_accuracy, AverageMeter, MetricsTracker
)
from models.training.data_loader import (
    load_simulation_data, split_data, create_dataloaders
)
from models.training.wandb_config import (
    init_wandb, log_metrics, log_incremental_validation,
    save_model_artifact, watch_model, WandBConfig, get_run_name_from_config
)
from models.transformer_predictor import GoalPredictionModel
from models.encoders.fusion_encoder import ToMGraphEncoder
from models.data.node_embeddings import get_or_create_embeddings, Node2VecEmbeddings
from graph_controller.world_graph import WorldGraph
import networkx as nx


# -----------------------------
# Data Preparation Utilities
# -----------------------------
def prepare_graph_data(graph, node_embeddings):
    """
    Prepare graph data using Node2Vec embeddings.
    
    Args:
        graph: NetworkX graph
        node_embeddings: Node2VecEmbeddings instance
    
    Returns:
        Dict with graph data ready for model input
    """
    # Get embeddings for all nodes (already in correct order)
    node_emb_matrix = node_embeddings.embedding_matrix
    
    # Use existing node-to-index mapping from node_embeddings
    node_to_idx = node_embeddings.node_to_idx
    
    # Manually create edge_index from graph edges
    edge_list = []
    for u, v in graph.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        # Add both directions for undirected graph
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])
    
    # Convert to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return {
        'node_embeddings': node_emb_matrix,
        'edge_index': edge_index
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    graph_data: dict,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (avg_loss, avg_top1_acc, avg_top5_acc)
    """
    model.train()
    
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Prepare trajectory batch - move embeddings to device
        traj_batch = {
            'node_embeddings': batch['node_embeddings'].to(device),
            'hour': batch['hour'].to(device),
            'mask': batch['mask'].to(device)
        }
        targets = batch['goal_indices'].to(device)
        
        # Forward pass
        logits = model(traj_batch, graph_data, return_logits=True)
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute metrics
        top1_acc = compute_accuracy(logits, targets, k=1)
        top5_acc = compute_accuracy(logits, targets, k=5)
        
        # Update meters
        batch_size = targets.size(0)  # Get batch size from targets tensor
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1_acc, batch_size)
        top5_meter.update(top5_acc, batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'top1': f'{top1_meter.avg:.2f}%',
            'top5': f'{top5_meter.avg:.2f}%'
        })
    
    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    graph_data: dict,
    device: torch.device,
    split_name: str = 'Val'
) -> tuple:
    """
    Validate the model.
    
    Returns:
        Tuple of (avg_loss, avg_top1_acc, avg_top5_acc)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    progress_bar = tqdm(val_loader, desc=split_name)
    
    for batch in progress_bar:
        # Prepare trajectory batch - move embeddings to device
        traj_batch = {
            'node_embeddings': batch['node_embeddings'].to(device),
            'hour': batch['hour'].to(device),
            'mask': batch['mask'].to(device)
        }
        targets = batch['goal_indices'].to(device)
        
        # Forward pass
        logits = model(traj_batch, graph_data, return_logits=True)
        loss = criterion(logits, targets)
        
        # Compute metrics
        top1_acc = compute_accuracy(logits, targets, k=1)
        top5_acc = compute_accuracy(logits, targets, k=5)
        
        # Update meters
        batch_size = targets.size(0)  # Get batch size from targets tensor
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1_acc, batch_size)
        top5_meter.update(top5_acc, batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'top1': f'{top1_meter.avg:.2f}%',
            'top5': f'{top5_meter.avg:.2f}%'
        })
    
    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def validate_incremental(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    graph_data: dict,
    device: torch.device,
    truncation_ratios: list = [0.25, 0.5, 0.75]
) -> dict:
    """
    Validate model at different trajectory truncation points.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        trajectory_prep: Trajectory data preparator
        graph_data: Graph data
        device: Device to run on
        truncation_ratios: List of truncation ratios to test (e.g., [0.25, 0.5, 0.75])
    
    Returns:
        Dict with results for each truncation ratio
    """
    model.eval()
    
    # Initialize meters for each truncation ratio
    results = {}
    for ratio in truncation_ratios:
        results[ratio] = {
            'loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter()
        }
    
    # Track trajectory lengths for debugging
    traj_lengths = {ratio: [] for ratio in truncation_ratios}
    
    for batch in tqdm(val_loader, desc='Incremental Val'):
        # For each truncation ratio
        for ratio in truncation_ratios:
            # Truncate embeddings based on valid sequence lengths (using mask)
            node_embeddings_full = batch['node_embeddings']  # (batch, max_seq_len, emb_dim)
            mask_full = batch['mask']  # (batch, max_seq_len)
            
            # Get actual sequence lengths from mask
            seq_lens = mask_full.sum(dim=1).long()  # (batch,)
            
            # Calculate truncated lengths
            truncated_lens = torch.maximum(
                torch.ones_like(seq_lens),
                (seq_lens.float() * ratio).long()
            )
            
            # Track lengths for debugging
            if len(traj_lengths[ratio]) < 5:
                for orig_len, trunc_len in zip(seq_lens.tolist(), truncated_lens.tolist()):
                    if len(traj_lengths[ratio]) < 5:
                        traj_lengths[ratio].append((orig_len, trunc_len))
            
            # Create truncated mask
            mask_truncated = torch.zeros_like(mask_full)
            for i, trunc_len in enumerate(truncated_lens):
                mask_truncated[i, :trunc_len] = 1.0
            
            # Prepare truncated batch
            traj_batch = {
                'node_embeddings': node_embeddings_full.to(device),  # Keep full embeddings, mask handles truncation
                'hour': batch['hour'].to(device),
                'mask': mask_truncated.to(device)
            }
            targets = batch['goal_indices'].to(device)
            
            # Forward pass
            logits = model(traj_batch, graph_data, return_logits=True)
            loss = criterion(logits, targets)

            
            
            # Compute metrics
            top1_acc = compute_accuracy(logits, targets, k=1)
            top5_acc = compute_accuracy(logits, targets, k=5)
            
            # Update meters
            batch_size = targets.size(0)  # Get batch size from targets tensor
            results[ratio]['loss'].update(loss.item(), batch_size)
            results[ratio]['top1'].update(top1_acc, batch_size)
            results[ratio]['top5'].update(top5_acc, batch_size)
    
    # Debug: Print trajectory lengths to verify truncation is working
    print(f"\n🔍 Debug - Trajectory lengths (original → truncated):")
    for ratio in truncation_ratios:
        if traj_lengths[ratio]:
            sample_lengths = traj_lengths[ratio][:3]
            print(f"   {int(ratio*100)}%: {sample_lengths}")
    
    # Convert to simple dict with averages
    summary = {}
    for ratio in truncation_ratios:
        summary[ratio] = {
            'loss': results[ratio]['loss'].avg,
            'top1': results[ratio]['top1'].avg,
            'top5': results[ratio]['top5'].avg
        }
    
    return summary


def main(args):
    """Main training function."""
    
    print("\n" + "=" * 80)
    print("BDI-ToM GOAL PREDICTION TRAINING")
    print("=" * 80 + "\n")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Load data
    print("\n📂 Loading data...")
    trajectories, graph, poi_nodes = load_simulation_data(args.run_dir, args.graph_path)
    
    # Split data
    train_trajs, val_trajs, test_trajs = split_data(
        trajectories,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Initialize Node2Vec embeddings (before creating dataloaders)
    print("\n🎯 Initializing Node2Vec embeddings...")
    
    # Create WorldGraph wrapper
    world_graph = WorldGraph(graph)
    num_nodes = len(graph.nodes())
    
    # Get or create Node2Vec embeddings (cached to avoid retraining)
    node_emb_cache_path = os.path.join('data', 'processed', 'node2vec_embeddings.pkl')
    node_embeddings = get_or_create_embeddings(
        graph,
        node_emb_cache_path,
        embedding_dim=args.node_emb_dim,
        walk_length=80,
        num_walks=10,
        p=1.0,
        q=1.0,
        window_size=10,
        num_workers=4,
        seed=args.seed,
        force_retrain=False  # Set to True to retrain embeddings
    )
    
    print(f"   ✅ Node2Vec embeddings ready: {node_embeddings.num_nodes} nodes, {node_embeddings.embedding_dim} dims")
    
    # Create dataloaders with embedding conversion
    print("\n📦 Creating dataloaders with Node2Vec preprocessing...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_trajs, val_trajs, test_trajs,
        graph, poi_nodes,
        node_embeddings=node_embeddings,  # Pass embeddings for preprocessing
        batch_size=args.batch_size,
        num_workers=0,  # Keep 0 for MPS compatibility
        max_seq_len=50
    )
    
    # Prepare graph data using Node2Vec embeddings
    print("\n🔧 Preparing graph data...")
    graph_data_dict = prepare_graph_data(graph, node_embeddings)
    
    # Initialize fusion encoder
    print(f"\n🏗️  Building model...")
    print(f"   Number of nodes: {num_nodes}")
    print(f"   Node embedding dim: {args.node_emb_dim}")
    print(f"   Number of POI nodes: {len(poi_nodes)}")
    
    fusion_encoder = ToMGraphEncoder(
        node_emb_dim=args.node_emb_dim,
        hidden_dim=64,
        output_dim=args.fusion_dim,
        n_layers=2,
        n_heads=4,
        dropout=args.dropout
    )
    
    # Initialize goal prediction model
    model = GoalPredictionModel(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=len(poi_nodes),
        fusion_dim=args.fusion_dim,
        hidden_dim=args.transformer_dim,
        n_transformer_layers=args.num_transformer_layers,
        n_heads=args.num_heads,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Prepare graph data as dict (kept on device)
    graph_data = {
        'node_embeddings': graph_data_dict['node_embeddings'].to(device),
        'edge_index': graph_data_dict['edge_index'].to(device)
    }
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Initialize W&B if enabled
    wandb_run = None
    if args.use_wandb:
        config = vars(args).copy()
        config['model_name'] = 'baseline_transformer'  # Identifier for this model type
        config['num_poi_nodes'] = len(world_graph.poi_nodes)
        config['num_graph_nodes'] = graph.number_of_nodes()
        config['num_train_samples'] = len(train_loader.dataset) # type: ignore
        config['num_val_samples'] = len(val_loader.dataset) # type: ignore
        config['num_test_samples'] = len(test_loader.dataset) # type: ignore
        
        # Generate descriptive run name if not provided
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = get_run_name_from_config(config)
            print(f"📊 Auto-generated W&B run name: {run_name}")
        
        wandb_run = init_wandb(
            project_name=args.wandb_project,
            run_name=run_name,
            config=config,
            tags=args.wandb_tags.split(',') if args.wandb_tags else None
        )
        
        # Watch model gradients and parameters
        if wandb_run is not None:
            watch_model(model)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.num_epochs} epochs...")
    print("=" * 80 + "\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_top1, train_top5 = train_epoch(
            model, train_loader, optimizer, criterion,
            graph_data, device, epoch
        )
        
        # Validate with incremental truncation
        print(f"\n📊 Incremental Validation (at different trajectory portions):")
        val_results = validate_incremental(
            model, val_loader, criterion,
            graph_data, device,
            truncation_ratios=[0.25, 0.5, 0.75]
        )
        
        # Print incremental results
        for ratio, ratio_metrics in val_results.items():
            print(f"   {int(ratio*100)}% of trajectory: "
                  f"Loss={ratio_metrics['loss']:.4f}, "
                  f"Top-1={ratio_metrics['top1']:.2f}%, "
                  f"Top-5={ratio_metrics['top5']:.2f}%")
        
        # Use 75% truncation results for main validation metrics
        val_loss = val_results[0.75]['loss']
        val_top1 = val_results[0.75]['top1']
        val_top5 = val_results[0.75]['top5']
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update metrics
        metrics.update(
            train_loss, val_loss,
            train_top1, val_top1,
            train_top5, val_top5
        )
        
        # Log to W&B
        if wandb_run is not None:
            # Log main training metrics
            log_metrics({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/top1_accuracy': train_top1,
                'train/top5_accuracy': train_top5,
                'val/loss': val_loss,
                'val/top1_accuracy': val_top1,
                'val/top5_accuracy': val_top5,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Log incremental validation results
            log_incremental_validation(val_results, epoch)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train - Loss: {train_loss:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}%")
        print(f"   Val   - Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%")
        
        # Save checkpoint if best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                {'val_top1': val_top1, 'val_top5': val_top5},
                checkpoint_path
            )
            print(f"   🌟 New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, 'latest_model.pt')
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            {'val_top1': val_top1, 'val_top5': val_top5},
            latest_path
        )
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⏹️  Early stopping after {epoch} epochs (patience: {args.early_stop_patience})")
            break
    
    # Save metrics
    metrics_path = os.path.join(args.checkpoint_dir, 'metrics.json')
    metrics.save(metrics_path)
    
    # Save best model as W&B artifact
    if wandb_run is not None:
        best_checkpoint = os.path.join(args.checkpoint_dir, 'best_model.pt')
        save_model_artifact(
            model_path=best_checkpoint,
            name=f'best_model_epoch_{epoch}',
            artifact_type='model',
            metadata={
                'best_val_top1': best_val_acc,
                'epochs_trained': epoch,
                'final_val_loss': val_loss
            }
        )
    
    # Print training summary
    metrics.print_summary()
    
    # Test evaluation with best model
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    
    best_checkpoint = os.path.join(args.checkpoint_dir, 'best_model.pt')
    load_checkpoint(best_checkpoint, model)
    model = model.to(device)
    
    test_loss, test_top1, test_top5 = validate(
        model, test_loader, criterion,
        graph_data, device, 'Test'
    )
    
    print(f"\n🎯 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Top-1 Accuracy: {test_top1:.2f}%")
    print(f"   Top-5 Accuracy: {test_top5:.2f}%")
    
    # Log test results to W&B
    if wandb_run is not None:
        log_metrics({
            'test/loss': test_loss,
            'test/top1_accuracy': test_top1,
            'test/top5_accuracy': test_top5
        })
        wandb_run.finish()
        print("📊 W&B run finished")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BDI-ToM goal prediction model')
    
    # Data arguments
    parser.add_argument('--run_dir', type=str, default='data/simulation_data/run_8',
                        help='Path to simulation run directory')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml',
                        help='Path to graph file')
    
    # Data split arguments
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--node_emb_dim', type=int, default=128,
                        help='Node2Vec embedding dimension')
    parser.add_argument('--fusion_dim', type=int, default=64,
                        help='Fusion layer dimension')
    parser.add_argument('--transformer_dim', type=int, default=128,
                        help='Transformer dimension')
    parser.add_argument('--num_transformer_layers', type=int, default=1,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/baseline_transformer',
                        help='Directory to save checkpoints')
    
    # W&B arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable Weights & Biases tracking')
    parser.add_argument('--wandb_project', type=str, default=WandBConfig.PROJECT_NAME,
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (auto-generated if not provided)')
    parser.add_argument('--wandb_tags', type=str, default=None,
                        help='Comma-separated W&B tags')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    main(args)
