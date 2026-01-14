#!/usr/bin/env python3
"""Test script for C²D-BDI model."""

import sys
sys.path.insert(0, '.')

import torch
print("PyTorch imported successfully")

from models.vae_bdi_simple.bdi_vae_v2_model import CausallyConstrainedBDIVAE
print("CausallyConstrainedBDIVAE imported successfully")

# Test instantiation
model = CausallyConstrainedBDIVAE(
    num_nodes=1000,
    num_agents=100,
    num_poi_nodes=50,
    num_categories=7,
)
print(f"✅ Model instantiated successfully!")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
batch_size = 4
seq_len = 10
history = torch.randint(0, 1000, (batch_size, seq_len))
lengths = torch.randint(1, seq_len+1, (batch_size,))
progress = torch.rand(batch_size)
next_node = torch.randint(0, 1000, (batch_size,))
goal_cat = torch.randint(0, 7, (batch_size,))

outputs = model(
    history_node_indices=history,
    history_lengths=lengths,
    path_progress=progress,
    compute_loss=True,
    next_node_idx=next_node,
    goal_cat_idx=goal_cat,
)

print("✅ Forward pass successful!")
print(f"   Belief z shape: {outputs['belief_z'].shape}")
print(f"   Desire z shape: {outputs['desire_z'].shape}")
print(f"   Intention z shape: {outputs['intention_z'].shape}")
print(f"   Goal logits shape: {outputs['goal'].shape}")
print(f"   Total VAE loss: {outputs['total_vae_loss'].item():.4f}")
print(f"   Belief TC loss: {outputs['belief_tc_loss'].item():.4f}")
print(f"   Desire TC loss: {outputs['desire_tc_loss'].item():.4f}")
print(f"   B-D MI loss: {outputs['bd_mi_loss'].item():.4f}")

print("\n✅ All tests passed!")
