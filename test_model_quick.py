#!/usr/bin/env python3
"""Quick test of V3 model with full unified embedding pipeline."""
import json, sys, os
os.environ['TQDM_DISABLE'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

import torch
import networkx as nx
from graph_controller.world_graph import WorldGraph
from models.new_bdi.bdi_dataset_v2 import BDIVAEDatasetV2, collate_bdi_samples_v2
from models.new_bdi.bdi_vae_v3_model import create_sc_bdi_vae_v3

print("Loading graph...", flush=True)
graph = nx.read_graphml('data/processed/ucsd_walk_full.graphml')
num_nodes = graph.number_of_nodes()

print("Loading enriched trajectories (first 50)...", flush=True)
with open('data/simulation_data/run_8_enriched/enriched_trajectories.json') as f:
    trajectories = json.load(f)
for idx, traj in enumerate(trajectories):
    traj['agent_id'] = idx // 1000

wg = WorldGraph(graph)
poi_nodes = wg.poi_nodes
node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

subset = trajectories[:50]
ds = BDIVAEDatasetV2(
    trajectories=subset, graph=graph, poi_nodes=poi_nodes,
    node_to_idx_map=node_to_idx, include_progress=True, include_temporal=True,
)
print(f"  Dataset: {len(ds)} samples", flush=True)

from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=8, collate_fn=collate_bdi_samples_v2)
batch = next(iter(loader))

# Create model
print("\nCreating SC-BDI-VAE V3 model with full pipeline...", flush=True)
device = torch.device('cpu')
model = create_sc_bdi_vae_v3(
    num_nodes=num_nodes,
    num_agents=100,
    num_poi_nodes=len(poi_nodes),
    num_categories=7,
    node_embedding_dim=64,
    fusion_dim=128,
    belief_latent_dim=32,
    desire_latent_dim=16,
    intention_latent_dim=32,
    vae_hidden_dim=128,
    hidden_dim=256,
    use_progress=True,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}", flush=True)
print(f"  Pipeline use_temporal: {model.embedding_pipeline.use_temporal}", flush=True)
print(f"  Pipeline use_agent: {model.embedding_pipeline.use_agent}", flush=True)
print(f"  Pipeline use_node2vec: {model.embedding_pipeline.use_node2vec}", flush=True)

# Forward pass
print("\nRunning forward pass with temporal features...", flush=True)
outputs = model(
    history_node_indices=batch['history_node_indices'].to(device),
    history_lengths=batch['history_lengths'].to(device),
    agent_ids=batch['agent_id'].to(device),
    path_progress=batch['path_progress'].to(device),
    compute_loss=True,
    next_node_idx=batch['next_node_idx'].to(device),
    goal_idx=batch['goal_idx'].to(device),
    goal_cat_idx=batch['goal_cat_idx'].to(device),
    hours=batch['hour'].to(device),
    days=batch['day_of_week'].to(device),
    deltas=batch['history_temporal_deltas'].to(device),
    velocities=batch['history_velocities'].to(device),
)

print(f"\nOutput keys: {list(outputs.keys())}", flush=True)
print(f"  goal logits:     {list(outputs['goal'].shape)}", flush=True)
print(f"  nextstep logits: {list(outputs['nextstep'].shape)}", flush=True)
print(f"  category logits: {list(outputs['category'].shape)}", flush=True)
print(f"  belief_z:        {list(outputs['belief_z'].shape)}", flush=True)
print(f"  desire_z:        {list(outputs['desire_z'].shape)}", flush=True)
print(f"  intention_z:     {list(outputs['intention_z'].shape)}", flush=True)
print(f"  total_vae_loss:  {outputs['total_vae_loss'].item():.4f}", flush=True)
print(f"  infonce_loss:    {outputs['infonce_loss'].item():.4f}", flush=True)
print(f"  desire_goal_loss:{outputs['desire_goal_loss'].item():.4f}", flush=True)

# Backward pass test
total_loss = outputs['total_vae_loss'] + torch.nn.functional.cross_entropy(outputs['goal'], batch['goal_idx'].to(device))
total_loss.backward()
print(f"\n  Backward pass: OK (loss={total_loss.item():.4f})", flush=True)

print("\n✅ Full model forward+backward test PASSED!", flush=True)
