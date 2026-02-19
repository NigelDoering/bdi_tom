#!/usr/bin/env python3
"""Quick test of updated dataset with enriched trajectories."""
import json, sys, os
os.environ['TQDM_DISABLE'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx
from graph_controller.world_graph import WorldGraph
from models.new_bdi.bdi_dataset_v2 import BDIVAEDatasetV2, collate_bdi_samples_v2

print("Loading graph...", flush=True)
graph = nx.read_graphml('data/processed/ucsd_walk_full.graphml')
print(f"  Graph: {graph.number_of_nodes()} nodes", flush=True)

print("Loading enriched trajectories...", flush=True)
with open('data/simulation_data/run_8_enriched/enriched_trajectories.json') as f:
    trajectories = json.load(f)
print(f"  Loaded {len(trajectories)} trajectories", flush=True)

# Assign agent_id (block of 1000 per agent)
for idx, traj in enumerate(trajectories):
    traj['agent_id'] = idx // 1000

wg = WorldGraph(graph)
poi_nodes = wg.poi_nodes
print(f"  POI nodes: {len(poi_nodes)}", flush=True)

node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

# Test with small subset
subset = trajectories[:20]
print(f"\nCreating dataset from {len(subset)} trajectories...", flush=True)
ds = BDIVAEDatasetV2(
    trajectories=subset,
    graph=graph,
    poi_nodes=poi_nodes,
    node_to_idx_map=node_to_idx,
    include_progress=True,
    include_temporal=True,
)
print(f"  Dataset samples: {len(ds)}", flush=True)

# Test a single sample
sample = ds[0]
print(f"\nSample keys: {list(sample.keys())}", flush=True)
for k, v in sample.items():
    if isinstance(v, list):
        print(f"  {k}: list len={len(v)}", flush=True)
    else:
        print(f"  {k}: {v}", flush=True)

# Test collation
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, collate_fn=collate_bdi_samples_v2)
batch = next(iter(loader))
print(f"\nBatch contents:", flush=True)
for k, v in batch.items():
    print(f"  {k}: shape={list(v.shape)} dtype={v.dtype}", flush=True)

print("\n✅ Dataset test PASSED!", flush=True)
