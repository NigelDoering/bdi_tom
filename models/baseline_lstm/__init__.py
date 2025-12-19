"""
Baseline LSTM Model for BDI-ToM Prediction

This package contains the baseline LSTM architecture for Theory of Mind
prediction in spatial navigation tasks.

Components:
- baseline_lstm_model: PerNodeToMPredictor model class
- baseline_lstm_dataset: PerNodeTrajectoryDataset and collate function
- train_baseline_lstm: Training script with W&B logging

Usage:
    from models.baseline_lstm import PerNodeToMPredictor, PerNodeTrajectoryDataset
    
    # Or run training:
    python -m models.baseline_lstm.train_baseline_lstm
"""

from .baseline_lstm_model import PerNodeToMPredictor
from .baseline_lstm_dataset import PerNodeTrajectoryDataset, collate_per_node_samples

__all__ = [
    'PerNodeToMPredictor',
    'PerNodeTrajectoryDataset',
    'collate_per_node_samples',
]
