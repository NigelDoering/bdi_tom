"""
BDI VAE Simple Module

Hierarchical Variational Autoencoder for learning mental state representations
(Beliefs, Desires, Intentions) without labels.
"""

from models.vae_bdi_simple.bdi_vae_model import BDIVAEPredictor
from models.vae_bdi_simple.vae_components import (
    BeliefVAE,
    DesireVAE,
    IntentionVAE,
    VAEEncoder,
    VAEDecoder,
    vae_loss,
    free_bits_kl,
)
from models.vae_bdi_simple.bdi_dataset import BDIVAEDataset, collate_bdi_samples

__all__ = [
    'BDIVAEPredictor',
    'BeliefVAE',
    'DesireVAE',
    'IntentionVAE',
    'VAEEncoder',
    'VAEDecoder',
    'vae_loss',
    'free_bits_kl',
    'BDIVAEDataset',
    'collate_bdi_samples',
]
