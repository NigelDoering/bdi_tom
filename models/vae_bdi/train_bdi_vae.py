"""
End-to-end training script for BDI-VAE goal prediction model.

This script trains the complete BDI (Belief-Desire-Intention) hierarchical VAE pipeline:
- TrajectoryEncoder (transformer-based)
- WorldGraphEncoder (GAT-based)
- ToMGraphEncoder (fusion layer)
- BDIVAEPredictor (Belief VAE → Desire VAE → Intention VAE → classifier)

The model learns hierarchical latent representations:
- Belief level: Agent's understanding of environment
- Desire level: Agent's goals/preferences
- Intention level: Agent's action plans

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

from models.utils.utils import (
    get_device, set_seed, save_checkpoint, load_checkpoint,
    compute_accuracy, AverageMeter, MetricsTracker
)
from models.data_loader_utils.data_loader import (
    load_simulation_data, split_data, create_dataloaders
)
from models.utils.wandb_config import (
    init_wandb, log_metrics, log_incremental_validation,
    save_model_artifact, watch_model, WandBConfig
)
from models.vae_bdi.bdi_vae_predictor import BDIVAEPredictor
from models.fusion_encoders_preprocessing.trajectory_encoder import TrajectoryDataPreparator
from models.fusion_encoders_preprocessing.map_encoder import GraphDataPreparator
from models.fusion_encoders_preprocessing.fusion_encoder import ToMGraphEncoder
from graph_controller.world_graph import WorldGraph


def get_run_name_from_config(config: dict) -> str:
    """
    Generate descriptive W&B run name from config.
    
    Format: bdi_vae_lr{lr}_bs{bs}_bl{belief}_dl{desire}_il{intention}
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        str: Descriptive run name
    """
    lr = config.get('learning_rate', 0.001)
    bs = config.get('batch_size', 32)
    bl = config.get('belief_latent_dim', 48)
    dl = config.get('desire_latent_dim', 32)
    il = config.get('intention_latent_dim', 24)
    
    return f"bdi_vae_lr{lr}_bs{bs}_bl{bl}_dl{dl}_il{il}"


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    trajectory_prep: TrajectoryDataPreparator,
    graph_data: dict,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (avg_loss, avg_top1_acc, avg_top5_acc, avg_loss_dict)
    """
    model.train()
    
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    # Track individual loss components
    classification_loss_meter = AverageMeter()
    vae_loss_meter = AverageMeter()
    belief_loss_meter = AverageMeter()
    desire_loss_meter = AverageMeter()
    intention_loss_meter = AverageMeter()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Prepare trajectory data
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
        
        targets = batch['goal_indices'].to(device)
        
        # Forward pass - compute loss with VAE components
        total_loss, loss_dict = model.compute_loss(traj_batch, graph_data, targets)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute accuracy metrics (need separate forward pass without VAE params)
        with torch.no_grad():
            logits = model(traj_batch, graph_data, return_logits=True)
            top1_acc = compute_accuracy(logits, targets, k=1)
            top5_acc = compute_accuracy(logits, targets, k=5)
        
        # Update meters
        batch_size = len(batch['trajectories'])
        loss_meter.update(total_loss.item(), batch_size)
        top1_meter.update(top1_acc, batch_size)
        top5_meter.update(top5_acc, batch_size)
        
        classification_loss_meter.update(loss_dict['classification_loss'], batch_size)
        vae_loss_meter.update(loss_dict['total_vae_loss'], batch_size)
        belief_loss_meter.update(loss_dict['belief_loss'], batch_size)
        desire_loss_meter.update(loss_dict['desire_loss'], batch_size)
        intention_loss_meter.update(loss_dict['intention_loss'], batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{classification_loss_meter.avg:.3f}',
            'vae': f'{vae_loss_meter.avg:.3f}',
            'top1': f'{top1_meter.avg:.2f}%'
        })
    
    avg_loss_dict = {
        'classification_loss': classification_loss_meter.avg,
        'total_vae_loss': vae_loss_meter.avg,
        'belief_loss': belief_loss_meter.avg,
        'desire_loss': desire_loss_meter.avg,
        'intention_loss': intention_loss_meter.avg
    }
    
    return loss_meter.avg, top1_meter.avg, top5_meter.avg, avg_loss_dict


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    trajectory_prep: TrajectoryDataPreparator,
    graph_data: dict,
    device: torch.device,
    split_name: str = 'Val'
) -> tuple:
    """
    Validate the model.
    
    Returns:
        Tuple of (avg_loss, avg_top1_acc, avg_top5_acc, avg_loss_dict)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    # Track individual loss components
    classification_loss_meter = AverageMeter()
    vae_loss_meter = AverageMeter()
    belief_loss_meter = AverageMeter()
    desire_loss_meter = AverageMeter()
    intention_loss_meter = AverageMeter()
    
    progress_bar = tqdm(val_loader, desc=split_name)
    
    for batch in progress_bar:
        # Prepare trajectory data
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
        
        targets = batch['goal_indices'].to(device)
        
        # Forward pass
        total_loss, loss_dict = model.compute_loss(traj_batch, graph_data, targets)
        logits = model(traj_batch, graph_data, return_logits=True)
        
        # Compute metrics
        top1_acc = compute_accuracy(logits, targets, k=1)
        top5_acc = compute_accuracy(logits, targets, k=5)
        
        # Update meters
        batch_size = len(batch['trajectories'])
        loss_meter.update(total_loss.item(), batch_size)
        top1_meter.update(top1_acc, batch_size)
        top5_meter.update(top5_acc, batch_size)
        
        classification_loss_meter.update(loss_dict['classification_loss'], batch_size)
        vae_loss_meter.update(loss_dict['total_vae_loss'], batch_size)
        belief_loss_meter.update(loss_dict['belief_loss'], batch_size)
        desire_loss_meter.update(loss_dict['desire_loss'], batch_size)
        intention_loss_meter.update(loss_dict['intention_loss'], batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{classification_loss_meter.avg:.3f}',
            'vae': f'{vae_loss_meter.avg:.3f}',
            'top1': f'{top1_meter.avg:.2f}%'
        })
    
    avg_loss_dict = {
        'classification_loss': classification_loss_meter.avg,
        'total_vae_loss': vae_loss_meter.avg,
        'belief_loss': belief_loss_meter.avg,
        'desire_loss': desire_loss_meter.avg,
        'intention_loss': intention_loss_meter.avg
    }
    
    return loss_meter.avg, top1_meter.avg, top5_meter.avg, avg_loss_dict


@torch.no_grad()
def validate_incremental(
    model: nn.Module,
    val_loader: DataLoader,
    trajectory_prep: TrajectoryDataPreparator,
    graph_data: dict,
    device: torch.device,
    truncation_ratios: list = [0.25, 0.5, 0.75]
) -> dict:
    """
    Validate model at different trajectory truncation points.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
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
            'top5': AverageMeter(),
            'classification_loss': AverageMeter(),
            'vae_loss': AverageMeter()
        }
    
    for batch in tqdm(val_loader, desc='Incremental Val'):
        # For each truncation ratio
        for ratio in truncation_ratios:
            # Truncate trajectories
            truncated_trajs = []
            for i in range(len(batch['trajectories'])):
                path = batch['trajectories'][i]
                if len(path) > 2:
                    truncate_idx = max(1, int(len(path) * ratio))
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
            total_loss, loss_dict = model.compute_loss(traj_batch, graph_data, targets)
            logits = model(traj_batch, graph_data, return_logits=True)
            
            # Compute metrics
            top1_acc = compute_accuracy(logits, targets, k=1)
            top5_acc = compute_accuracy(logits, targets, k=5)
            
            # Update meters
            batch_size = len(batch['trajectories'])
            results[ratio]['loss'].update(total_loss.item(), batch_size)
            results[ratio]['top1'].update(top1_acc, batch_size)
            results[ratio]['top5'].update(top5_acc, batch_size)
            results[ratio]['classification_loss'].update(loss_dict['classification_loss'], batch_size)
            results[ratio]['vae_loss'].update(loss_dict['total_vae_loss'], batch_size)
    
    # Convert to simple dict with averages
    summary = {}
    for ratio in truncation_ratios:
        summary[ratio] = {
            'loss': results[ratio]['loss'].avg,
            'top1': results[ratio]['top1'].avg,
            'top5': results[ratio]['top5'].avg,
            'classification_loss': results[ratio]['classification_loss'].avg,
            'vae_loss': results[ratio]['vae_loss'].avg
        }
    
    return summary


def main(args):
    """Main training function."""
    
    print("\n" + "=" * 80)
    print("BDI-VAE GOAL PREDICTION TRAINING")
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
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_trajs, val_trajs, test_trajs,
        graph, poi_nodes,
        batch_size=args.batch_size,
        num_workers=0  # Keep 0 for MPS compatibility
    )
    
    # Initialize data preparators
    print("\n🔧 Initializing encoders...")
    
    # Create WorldGraph wrapper
    world_graph = WorldGraph(graph)
    
    # Create node-to-index mapping for trajectory encoder
    all_nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    trajectory_prep = TrajectoryDataPreparator(node_to_idx)
    graph_prep = GraphDataPreparator(world_graph)
    graph_data_dict = graph_prep.prepare_graph_data()
    
    # Get node feature dimensions
    num_nodes = len(graph.nodes())
    graph_node_feat_dim = graph_data_dict['x'].shape[1]  # 12 features per node
    
    # Initialize fusion encoder
    print(f"\n🏗️  Building BDI-VAE model...")
    print(f"   Number of nodes: {num_nodes}")
    print(f"   Graph feature dim: {graph_node_feat_dim}")
    print(f"   Number of POI nodes: {len(poi_nodes)}")
    print(f"   BDI hierarchy: Belief({args.belief_latent_dim}) → Desire({args.desire_latent_dim}) → Intention({args.intention_latent_dim})")
    
    fusion_encoder = ToMGraphEncoder(
        num_nodes=num_nodes,
        graph_node_feat_dim=graph_node_feat_dim,
        traj_node_emb_dim=32,
        hidden_dim=64,
        output_dim=args.fusion_dim,
        n_layers=2,
        n_heads=4,
        dropout=args.dropout
    )
    
    # Initialize BDI-VAE prediction model
    model = BDIVAEPredictor(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=len(poi_nodes),
        fusion_dim=args.fusion_dim,
        belief_latent_dim=args.belief_latent_dim,
        desire_latent_dim=args.desire_latent_dim,
        intention_latent_dim=args.intention_latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        kl_weight_belief=args.kl_weight_belief,
        kl_weight_desire=args.kl_weight_desire,
        kl_weight_intention=args.kl_weight_intention
    )
    model = model.to(device)
    
    # Prepare graph data as dict (kept on device)
    graph_data = {
        'x': graph_data_dict['x'].to(device),
        'edge_index': graph_data_dict['edge_index'].to(device)
    }
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer (no separate criterion needed - model.compute_loss handles it)
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
        config['model_name'] = 'bdi_vae'  # Identifier for this model type
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
        train_loss, train_top1, train_top5, train_loss_dict = train_epoch(
            model, train_loader, optimizer,
            trajectory_prep, graph_data, device, epoch
        )
        
        # Validate with incremental truncation
        print(f"\n📊 Incremental Validation (at different trajectory portions):")
        val_results = validate_incremental(
            model, val_loader,
            trajectory_prep, graph_data, device,
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
        val_cls_loss = val_results[0.75]['classification_loss']
        val_vae_loss = val_results[0.75]['vae_loss']
        
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
                'train/classification_loss': train_loss_dict['classification_loss'],
                'train/total_vae_loss': train_loss_dict['total_vae_loss'],
                'train/belief_loss': train_loss_dict['belief_loss'],
                'train/desire_loss': train_loss_dict['desire_loss'],
                'train/intention_loss': train_loss_dict['intention_loss'],
                'val/loss': val_loss,
                'val/top1_accuracy': val_top1,
                'val/top5_accuracy': val_top5,
                'val/classification_loss': val_cls_loss,
                'val/vae_loss': val_vae_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Log incremental validation results
            log_incremental_validation(val_results, epoch)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train - Loss: {train_loss:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}%")
        print(f"           Classification: {train_loss_dict['classification_loss']:.4f}, VAE: {train_loss_dict['total_vae_loss']:.4f}")
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
            name=f'best_bdi_vae_epoch_{epoch}',
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
    
    test_loss, test_top1, test_top5, test_loss_dict = validate(
        model, test_loader,
        trajectory_prep, graph_data, device, 'Test'
    )
    
    print(f"\n🎯 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Top-1 Accuracy: {test_top1:.2f}%")
    print(f"   Top-5 Accuracy: {test_top5:.2f}%")
    print(f"   Classification Loss: {test_loss_dict['classification_loss']:.4f}")
    print(f"   Total VAE Loss: {test_loss_dict['total_vae_loss']:.4f}")
    print(f"     - Belief Loss: {test_loss_dict['belief_loss']:.4f}")
    print(f"     - Desire Loss: {test_loss_dict['desire_loss']:.4f}")
    print(f"     - Intention Loss: {test_loss_dict['intention_loss']:.4f}")
    
    # Log test results to W&B
    if wandb_run is not None:
        log_metrics({
            'test/loss': test_loss,
            'test/top1_accuracy': test_top1,
            'test/top5_accuracy': test_top5,
            'test/classification_loss': test_loss_dict['classification_loss'],
            'test/total_vae_loss': test_loss_dict['total_vae_loss'],
            'test/belief_loss': test_loss_dict['belief_loss'],
            'test/desire_loss': test_loss_dict['desire_loss'],
            'test/intention_loss': test_loss_dict['intention_loss']
        })
        wandb_run.finish()
        print("📊 W&B run finished")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BDI-VAE goal prediction model')
    
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
    
    # Model arguments - Fusion encoder
    parser.add_argument('--fusion_dim', type=int, default=64,
                        help='Fusion layer dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for VAEs and MLPs')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Model arguments - BDI VAE latent dimensions
    parser.add_argument('--belief_latent_dim', type=int, default=48,
                        help='Latent dimension for Belief VAE')
    parser.add_argument('--desire_latent_dim', type=int, default=32,
                        help='Latent dimension for Desire VAE')
    parser.add_argument('--intention_latent_dim', type=int, default=24,
                        help='Latent dimension for Intention VAE')
    
    # Model arguments - VAE KL weights
    parser.add_argument('--kl_weight_belief', type=float, default=1.0,
                        help='KL divergence weight for Belief VAE')
    parser.add_argument('--kl_weight_desire', type=float, default=1.0,
                        help='KL divergence weight for Desire VAE')
    parser.add_argument('--kl_weight_intention', type=float, default=1.0,
                        help='KL divergence weight for Intention VAE')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/bdi_vae',
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
