"""
Hyperparameter search script for baseline transformer model.

This script runs multiple training configurations with different hyperparameters
and tracks all results in W&B for easy comparison.

Usage:
    python scripts/hyperparameter_search.py [--num_epochs 25] [--seed 42]
"""

import os
import sys
import subprocess
import argparse
from itertools import product

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_training_config(config, run_num, total_runs, args):
    """
    Run training with a specific hyperparameter configuration.
    
    Args:
        config: Dict with hyperparameter values
        run_num: Current run number
        total_runs: Total number of runs
        args: Command line arguments
    """
    print("\n" + "━" * 80)
    print(f"Run {run_num}/{total_runs}")
    print("━" * 80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key:25s} = {value}")
    print("━" * 80)
    
    # Build command
    cmd = [
        "python", "models/training/train_baseline_transformer.py",
        "--run_dir", args.run_dir,
        "--graph_path", args.graph_path,
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(config['learning_rate']),
        "--batch_size", str(config['batch_size']),
        "--fusion_dim", str(config['fusion_dim']),
        "--transformer_dim", str(config['transformer_dim']),
        "--num_transformer_layers", str(config['num_transformer_layers']),
        "--num_heads", str(config['num_heads']),
        "--dropout", str(config['dropout']),
        "--checkpoint_dir", f"{args.checkpoint_base}/run_{run_num}",
        "--seed", str(args.seed),
        "--use_wandb",
        "--wandb_tags", f"hp-search,run-{run_num}"
    ]
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Run {run_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Run {run_num} failed with error: {e}")
        return False


def get_hyperparameter_grid():
    """
    Define the hyperparameter search grid.
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    # ========================================
    # Strategy: Test one parameter at a time, plus some combinations
    # ========================================
    
    # Baseline configuration
    baseline = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'fusion_dim': 64,
        'transformer_dim': 128,
        'num_transformer_layers': 1,
        'num_heads': 4,
        'dropout': 0.1
    }
    configs.append(baseline.copy())
    
    # ========================================
    # Learning Rate Sweep (most important!)
    # ========================================
    for lr in [0.0001, 0.0003, 0.0005, 0.003, 0.005]:
        config = baseline.copy()
        config['learning_rate'] = lr
        configs.append(config)
    
    # ========================================
    # Batch Size Variations
    # ========================================
    for bs in [8, 16, 64]:
        config = baseline.copy()
        config['batch_size'] = bs
        configs.append(config)
    
    # ========================================
    # Model Capacity: Transformer Dimension
    # ========================================
    for td in [64, 192, 256]:
        config = baseline.copy()
        config['transformer_dim'] = td
        configs.append(config)
    
    # ========================================
    # Fusion Dimension
    # ========================================
    for fd in [32, 128]:
        config = baseline.copy()
        config['fusion_dim'] = fd
        configs.append(config)
    
    # ========================================
    # Architecture Depth
    # ========================================
    for nl in [2, 3]:
        config = baseline.copy()
        config['num_transformer_layers'] = nl
        configs.append(config)
    
    # ========================================
    # Attention Heads
    # ========================================
    for nh in [2, 8]:
        config = baseline.copy()
        config['num_heads'] = nh
        configs.append(config)
    
    # ========================================
    # Dropout (Regularization)
    # ========================================
    for dp in [0.05, 0.2, 0.3]:
        config = baseline.copy()
        config['dropout'] = dp
        configs.append(config)
    
    # Remove duplicates (keeps first occurrence)
    unique_configs = []
    seen = set()
    for config in configs:
        config_tuple = tuple(sorted(config.items()))
        if config_tuple not in seen:
            seen.add(config_tuple)
            unique_configs.append(config)
    
    return unique_configs


def get_custom_configs():
    """
    Define hand-picked configurations based on intuition.
    
    These are combinations that might work well together.
    """
    return [
        # High capacity model with conservative LR
        {
            'learning_rate': 0.0005,
            'batch_size': 64,
            'fusion_dim': 128,
            'transformer_dim': 256,
            'num_transformer_layers': 2,
            'num_heads': 8,
            'dropout': 0.15
        },
        # Fast training configuration
        {
            'learning_rate': 0.003,
            'batch_size': 64,
            'fusion_dim': 64,
            'transformer_dim': 128,
            'num_transformer_layers': 1,
            'num_heads': 4,
            'dropout': 0.2
        },
        # Small but effective
        {
            'learning_rate': 0.001,
            'batch_size': 32,
            'fusion_dim': 32,
            'transformer_dim': 64,
            'num_transformer_layers': 1,
            'num_heads': 2,
            'dropout': 0.1
        }
    ]


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for baseline transformer')
    
    parser.add_argument('--run_dir', type=str, default='data/simulation_data/run_8',
                        help='Path to simulation run directory')
    parser.add_argument('--graph_path', type=str, default='data/processed/ucsd_walk_full.graphml',
                        help='Path to graph file')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of epochs per configuration')
    parser.add_argument('--checkpoint_base', type=str, default='checkpoints/hp_search',
                        help='Base directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'custom', 'both'],
                        help='Search mode: grid (systematic), custom (hand-picked), or both')
    parser.add_argument('--max_runs', type=int, default=None,
                        help='Maximum number of runs (for testing, use subset of configs)')
    
    args = parser.parse_args()
    
    # Get configurations based on mode
    if args.mode == 'grid':
        configs = get_hyperparameter_grid()
    elif args.mode == 'custom':
        configs = get_custom_configs()
    else:  # both
        configs = get_hyperparameter_grid() + get_custom_configs()
    
    # Limit number of runs if specified
    if args.max_runs is not None:
        configs = configs[:args.max_runs]
    
    total_runs = len(configs)
    
    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH FOR BASELINE TRANSFORMER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Total runs: {total_runs}")
    print(f"  Epochs per run: {args.num_epochs}")
    print(f"  Total epochs: {total_runs * args.num_epochs}")
    print(f"  Checkpoint base: {args.checkpoint_base}")
    print(f"  Seed: {args.seed}")
    print(f"\nAll runs will be tracked in W&B with tag: hp-search")
    print("=" * 80)
    
    # Confirm before starting
    response = input("\nProceed with hyperparameter search? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run all configurations
    successful_runs = 0
    failed_runs = 0
    
    for i, config in enumerate(configs, 1):
        success = run_training_config(config, i, total_runs, args)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
    
    # Print final summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Successful runs: {successful_runs}/{total_runs}")
    print(f"  Failed runs: {failed_runs}/{total_runs}")
    print(f"\nCheckpoints saved to: {args.checkpoint_base}/")
    print(f"\nW&B Analysis:")
    print(f"  1. Go to: https://wandb.ai/your-username/bdi-tom-goal-prediction")
    print(f"  2. Filter by tag: hp-search")
    print(f"  3. Compare metrics: val/top1_accuracy, val/top5_accuracy")
    print(f"  4. Sort by best val/top1_accuracy")
    print(f"  5. Check incremental metrics: val/25%_top1, val/50%_top1, val/75%_top1")
    print(f"\nNext Steps:")
    print(f"  - Identify best configuration from W&B dashboard")
    print(f"  - Run full training (100 epochs) with best config")
    print(f"  - Consider ensemble of top 3-5 configurations")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
