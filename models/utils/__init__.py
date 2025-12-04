"""
Utility module for BDI-ToM models.

This module provides utilities for training:
- Device management (CUDA/MPS/CPU)
- Checkpointing and metrics tracking
- Accuracy computation
- WandB integration
"""

from .utils import (
    get_device,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    compute_accuracy,
    AverageMeter,
    MetricsTracker
)

from .wandb_config import (
    init_wandb,
    log_metrics,
    save_model_artifact,
    watch_model,
    WandBConfig,
    get_run_name_from_config
)

__all__ = [
    # Training utilities
    'get_device',
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'compute_accuracy',
    'AverageMeter',
    'MetricsTracker',
    # WandB utilities
    'init_wandb',
    'log_metrics',
    'save_model_artifact',
    'watch_model',
    'WandBConfig',
    'get_run_name_from_config'
]
