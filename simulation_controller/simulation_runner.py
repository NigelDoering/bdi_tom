import os
import json
import random
from tqdm import tqdm
import networkx as nx
import argparse

# Add project root to path to allow absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from graph_controller.world_graph import WorldGraph
from agent_controller.agent import Agent
from simulation_controller.simulation import Simulation


def run_simulation():
    # Parse command-line arguments
    args = parse_args()
    n_agents = args.num_agents
    m_trajectories = args.num_trajectories
    run_id = args.run_id

    print(f"Running simulation: {n_agents} agents, {m_trajectories} trajectories each, run ID = {run_id}")
    # Paths
    base_dir = f"data/simulation_data/run_{run_id}"
    agents_dir = os.path.join(base_dir, "agents")
    traj_dir = os.path.join(base_dir, "trajectories")
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    # Load and wrap the graph
    graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G)

    # Create agents and collect their save data
    agents = []
    agents_metadata = {}
    for i in range(n_agents):
        agent_id = f"agent_{i:03d}"
        agent = Agent(agent_id=agent_id, world_graph=world_graph)
        agents.append(agent)
        agents_metadata[agent_id] = agent.to_dict()

    # Save all agents to single JSON
    with open(os.path.join(agents_dir, "all_agents.json"), "w") as f:
        json.dump(agents_metadata, f, indent=2, ensure_ascii=False)

    # Initialize simulation
    sim = Simulation(world_graph)
    for agent in agents:
        sim.register_agent(agent)

    # Run simulation for each agent
    print(f"Running simulation: {n_agents} agents × {m_trajectories} trajectories")
    simulation_hours = list(range(6, 24)) + list(range(0, 3))  # 06:00 to 02:00

    for agent in tqdm(agents, desc="Simulating Agents"):
        for _ in range(m_trajectories):
            hour = random.choice(simulation_hours)
            sim.step(
                agent_id=agent.id,
                current_hour=hour,
                path_temp=30.0,
                belief_update_dist=100
            )

    # Save all trajectories to single JSON
    with open(os.path.join(traj_dir, "all_trajectories.json"), "w") as f:
        json.dump(sim.trajectories, f, indent=2, ensure_ascii=False
        )

    print(f"\n✅ Simulation complete. Data saved to: {base_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run trajectory simulation and save agent/trajectory data."
    )
    parser.add_argument(
        "-n", "--num_agents", type=int, required=True,
        help="Number of agents to simulate."
    )
    parser.add_argument(
        "-m", "--num_trajectories", type=int, required=True,
        help="Number of trajectories per agent."
    )
    parser.add_argument(
        "-x", "--run_id", type=int, required=True,
        help="Run identifier to store outputs under data/simulation_data/run_<X>/"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Example hardcoded run (replace with argparse if needed)
    run_simulation()