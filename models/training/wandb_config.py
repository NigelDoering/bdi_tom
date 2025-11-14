"""
Weights & Biases (W&B) integration for experiment tracking.

This module provides utilities for tracking model training experiments,
including hyperparameters, metrics, and model artifacts.
"""

import wandb
import torch
import os
from typing import Dict, Any, Optional


def init_wandb(
    project_name: str = "bdi-tom-goal-prediction",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    enabled: bool = True
):
    """
    Initialize Weights & Biases tracking.
    
    Args:
        project_name: W&B project name
        run_name: Name for this specific run (auto-generated if None)
        config: Dictionary of hyperparameters and configuration
        tags: List of tags for categorizing runs (e.g., ['baseline', 'transformer'])
        notes: Optional description of the experiment
        enabled: Whether to enable W&B tracking (useful for debugging)
    
    Returns:
        wandb.Run object or None if disabled
    """
    if not enabled:
        print("📊 W&B tracking disabled")
        return None
    
    # Initialize W&B
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags,
        notes=notes,
        reinit=True  # Allow multiple runs in same script
    )
    
    if run is not None:
        print(f"📊 W&B tracking enabled: {run.url}")
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (e.g., epoch number)
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_incremental_validation(
    val_results: Dict[float, Dict[str, float]],
    epoch: int,
    prefix: str = "val"
):
    """
    Log incremental validation results (25%, 50%, 75% trajectory).
    
    Args:
        val_results: Dict mapping truncation ratios to metrics
        epoch: Current epoch number
        prefix: Metric prefix (default: 'val')
    """
    if wandb.run is None:
        return
    
    metrics = {}
    for ratio, results in val_results.items():
        ratio_pct = int(ratio * 100)
        metrics[f"{prefix}/{ratio_pct}%_loss"] = results['loss']
        metrics[f"{prefix}/{ratio_pct}%_top1"] = results['top1']
        metrics[f"{prefix}/{ratio_pct}%_top5"] = results['top5']
    
    wandb.log(metrics, step=epoch)


def save_model_artifact(
    model_path: str,
    name: str = "model",
    artifact_type: str = "model",
    metadata: Optional[Dict] = None
):
    """
    Save model as W&B artifact for versioning.
    
    Args:
        model_path: Path to saved model file
        name: Artifact name
        artifact_type: Type of artifact (default: 'model')
        metadata: Optional metadata to attach
    """
    if wandb.run is None:
        return
    
    artifact = wandb.Artifact(
        name=name,
        type=artifact_type,
        metadata=metadata or {}
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"💾 Model artifact saved: {name}")


def finish_run():
    """Finish the W&B run."""
    if wandb.run is not None:
        wandb.finish()
        print("📊 W&B run finished")


def watch_model(model: torch.nn.Module, log_freq: int = 100):
    """
    Watch model gradients and parameters during training.
    
    Args:
        model: PyTorch model to watch
        log_freq: How often to log gradients (in batches)
    """
    if wandb.run is not None:
        wandb.watch(model, log="all", log_freq=log_freq)


def get_run_name_from_config(config: Dict[str, Any]) -> str:
    """
    Generate a descriptive run name from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Generated run name
    """
    model_name = config.get('model_name', 'baseline_transformer')
    lr = config.get('learning_rate', 0.001)
    bs = config.get('batch_size', 32)
    hidden = config.get('transformer_dim', 128)
    layers = config.get('num_transformer_layers', 1)
    
    return f"{model_name}_lr{lr}_bs{bs}_h{hidden}_l{layers}"


class WandBConfig:
    """Configuration class for W&B settings."""
    
    PROJECT_NAME = "bdi-tom-goal-prediction"
    ENTITY = None  # Set to your W&B username/team if needed
    
    # Common tags for different model types
    TAGS_BASELINE = ["baseline", "transformer"]
    TAGS_GNN = ["gnn", "graph-network"]
    TAGS_ATTENTION = ["attention", "transformer"]
    
    # Metrics to track
    TRACKED_METRICS = [
        "train/loss",
        "train/top1_acc",
        "train/top5_acc",
        "val/25%_loss",
        "val/25%_top1",
        "val/25%_top5",
        "val/50%_loss",
        "val/50%_top1",
        "val/50%_top5",
        "val/75%_loss",
        "val/75%_top1",
        "val/75%_top5",
        "learning_rate",
        "epoch"
    ]


# Utility function to check if W&B is available
def is_wandb_available() -> bool:
    """Check if W&B is installed and can be imported."""
    try:
        import wandb
        return True
    except ImportError:
        return False
