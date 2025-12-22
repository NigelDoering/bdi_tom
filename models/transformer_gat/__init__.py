"""
Transformer + GAT Model for Campus Navigation Prediction

This package combines Graph Attention Networks (GAT) with Transformer architecture
for multi-task trajectory prediction on campus navigation data.
"""

from .transformer_gat_model import TransformerGATPredictor, PositionalEncoding

__all__ = [
    'TransformerGATPredictor',
    'PositionalEncoding',
]
