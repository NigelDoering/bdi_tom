#!/usr/bin/env python3
"""Quick test of the diagnostic function."""
import sys
sys.path.insert(0, '/home/rahm/TOM/bdi_tom')

from models.vae_bdi_simple.train_bdi_vae_v3 import print_epoch_diagnostics

print_epoch_diagnostics(
    epoch=0,
    train_metrics={'loss': 10.0, 'goal_acc': 2.1, 'desire_goal_acc': 2.0, 'infonce_loss': 7.6},
    val_metrics={'loss': 9.7, 'goal_acc': 2.4, 'desire_goal_acc': 2.4},
    num_poi_nodes=735,
    num_categories=7,
    kl_weight=0.0,
)

print("\n✅ Test passed!")
