"""
run_small_sim.py
----------------
Quick helper to regenerate the small analysis dataset used for
development and CI.  Canonical test runs are:
  run_test_base    — base hours config (hours_config_base.json)
  run_test_dynamic — dynamic hours config (hours_config_dynamic.json)

Usage:
    uv run python simulation_analysis/run_small_sim.py
    uv run python simulation_analysis/run_small_sim.py --n_agents 5 --n_trajs 20 --run_id my_test

Produces: data/simulation_data/run_<id>/{agents,trajectories,beliefs}/
"""

import sys
import os

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import random
import numpy as np
import networkx as nx
from tqdm import tqdm

from graph_controller.world_graph import WorldGraph
from agent_controller.agent import Agent
from simulation_controller.simulation import Simulation
from simulation_controller.belief_store import BeliefStore
from simulation_controller.simulation_runner import sample_simulation_hour


def run(n_agents: int, n_trajs: int, run_id: str, seed: int = 42,
        hours_config: str = None) -> str:
    """Run a small simulation and return the output directory path."""
    np.random.seed(seed)
    random.seed(seed)

    base_dir    = f"data/simulation_data/run_{run_id}"
    agents_dir  = os.path.join(base_dir, "agents")
    traj_dir    = os.path.join(base_dir, "trajectories")
    beliefs_dir = os.path.join(base_dir, "beliefs")
    for d in [agents_dir, traj_dir, beliefs_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Loading graph...")
    G_raw = nx.read_graphml("data/processed/ucsd_walk_full.graphml")
    world_graph = WorldGraph(G_raw, hours_config_path=hours_config)

    print(f"Creating {n_agents} agents...")
    agents = []
    agents_meta = {}
    for i in range(n_agents):
        aid = f"agent_{i:03d}"
        agent = Agent(agent_id=aid, world_graph=world_graph, verbose=False)
        agents.append(agent)
        agents_meta[aid] = agent.to_dict()

    with open(os.path.join(agents_dir, "all_agents.json"), "w") as f:
        json.dump(agents_meta, f, indent=2)

    belief_store = BeliefStore(poi_nodes=world_graph.poi_nodes)
    sim = Simulation(world_graph, belief_store=belief_store, verbose=False)
    for agent in agents:
        sim.register_agent(agent)

    total = n_agents * n_trajs
    with tqdm(total=total, desc=f"Simulating (run_{run_id})", ncols=80) as pbar:
        for agent in agents:
            for _ in range(n_trajs):
                hour = sample_simulation_hour()
                sim.step(
                    agent_id=agent.id,
                    current_hour=hour,
                    path_temp=30.0,
                    belief_update_dist=100,
                )
                pbar.update(1)

    with open(os.path.join(traj_dir, "all_trajectories.json"), "w") as f:
        json.dump(sim.trajectories, f, indent=2, ensure_ascii=False)

    belief_store.save(beliefs_dir)
    print(f"Done → {base_dir}")
    return base_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a small simulation for analysis.")
    parser.add_argument("--n_agents",     type=int, default=10)
    parser.add_argument("--n_trajs",      type=int, default=50)
    parser.add_argument("--run_id",       type=str, default="analysis_small")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--hours_config", type=str, default=None,
                        help="Path to hours config JSON (e.g. data/processed/hours_config_base.json)")
    args = parser.parse_args()

    run(args.n_agents, args.n_trajs, args.run_id, args.seed, args.hours_config)
