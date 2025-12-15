"""
Training module for BDI-ToM models.

This module provides complete infrastructure for training the goal prediction model:
- Data loading and preprocessing
- Training loop with validation
- Device management (CUDA/MPS/CPU)
- Checkpointing and metrics tracking
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

__all__ = [
    'get_device',
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'compute_accuracy',
    'AverageMeter',
    'MetricsTracker',
]
