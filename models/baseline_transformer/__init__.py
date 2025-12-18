"""
Baseline Transformer Model for Per-Node ToM Prediction

This package contains a transformer-based model for Theory of Mind (ToM) prediction
on trajectory data. Unlike the LSTM baseline which uses per-node expansion, this
transformer processes full trajectories in parallel using causal masking.

KEY ADVANTAGES:
- 10-30x faster training (no dataset expansion)
- Parallel position processing with causal masking
- Better for training embedding pipeline (transfer to LSTM/VAE)

Components:
- PerNodeTransformerPredictor: Main transformer model
- TransformerTrajectoryDataset: Dataset keeping full trajectories
- collate_transformer_trajectories: Batch collation with padding
- train_baseline_transformer: Training script

Usage:
    from models.baseline_transformer import PerNodeTransformerPredictor
    
    model = PerNodeTransformerPredictor(
        num_nodes=664,
        num_pois=230,
        num_categories=7,
        node_embedding_dim=128,
        d_model=256,
        nhead=8,
        num_layers=4,
    )
"""

from .baseline_transformer_model import PerNodeTransformerPredictor
from .baseline_transformer_dataset import TransformerTrajectoryDataset, collate_transformer_trajectories

__all__ = [
    'PerNodeTransformerPredictor',
    'TransformerTrajectoryDataset',
    'collate_transformer_trajectories',
]
