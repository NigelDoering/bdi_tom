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
from simulation_controller.belief_store import BeliefStore


def sample_simulation_hour():
    """
    Sample an hour for simulation using a bell-curve-like distribution with a flat peak.

    Distribution characteristics:
    - Hours 6-8am: Uncommon (low probability, ramping up)
    - Hours 11am-3pm: Peak (flat, high probability)
    - Hours after 8pm: Uncommon (low probability, tapering down)
    - Supports hours from 6am to 2am (wraps past midnight)

    Returns:
        int: Hour in 24-hour format (6-23 or 0-2)
    """
    # Define the hours we want to sample from
    hours = list(range(6, 24)) + list(range(0, 3))  # 06:00 to 02:00

    # Create a probability distribution
    # We'll use a piecewise function:
    # - 6-8am (6,7,8): Ramp up (weights: 0.3, 0.5, 0.7)
    # - 9-10am (9,10): Transition (weights: 0.9, 1.0)
    # - 11am-3pm (11,12,13,14,15): Peak plateau (weights: 1.2)
    # - 4-7pm (16,17,18,19): Transition down (weights: 1.0, 0.9, 0.7, 0.5)
    # - 8pm-2am (20,21,22,23,0,1,2): Taper off (weights: 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05)

    weights = {
        6: 0.3,   # 6am - uncommon
        7: 0.5,   # 7am - ramping up
        8: 0.7,   # 8am
        9: 0.9,   # 9am - approaching peak
        10: 1.0,  # 10am
        11: 1.2,  # 11am - peak starts
        12: 1.2,  # 12pm - peak
        13: 1.2,  # 1pm - peak
        14: 1.2,  # 2pm - peak
        15: 1.2,  # 3pm - peak ends
        16: 1.0,  # 4pm - transition down
        17: 0.9,  # 5pm
        18: 0.7,  # 6pm
        19: 0.5,  # 7pm
        20: 0.4,  # 8pm - uncommon
        21: 0.3,  # 9pm
        22: 0.2,  # 10pm
        23: 0.15, # 11pm
        0: 0.1,   # 12am
        1: 0.08,  # 1am
        2: 0.05,  # 2am - very uncommon
    }

    # Get weights in the same order as hours
    probs = [weights[h] for h in hours]

    # Normalize to create a probability distribution
    total = sum(probs)
    probs = [p / total for p in probs]

    # Sample an hour using the weighted distribution
    return random.choices(hours, weights=probs, k=1)[0]


def run_simulation():
    # Parse command-line arguments
    args = parse_args()

    # --test_beliefs overrides -n / -m / -x with fixed test parameters
    if args.test_beliefs:
        n_agents = 10
        m_trajectories = 50
        run_id = "test_beliefs"
        if not args.quiet:
            print("🧪 --test_beliefs mode: 10 agents × 50 trajectories → run_test_beliefs")
    else:
        n_agents = args.num_agents
        m_trajectories = args.num_trajectories
        run_id = args.run_id

    if not args.quiet:
        print(f"Running simulation: {n_agents} agents, {m_trajectories} trajectories each, run ID = {run_id}")

    # Paths
    base_dir = f"data/simulation_data/run_{run_id}"
    agents_dir  = os.path.join(base_dir, "agents")
    traj_dir    = os.path.join(base_dir, "trajectories")
    beliefs_dir = os.path.join(base_dir, "beliefs")
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
        agent = Agent(agent_id=agent_id, world_graph=world_graph, verbose=not args.quiet)
        agents.append(agent)
        agents_metadata[agent_id] = agent.to_dict()

    # Save all agents to single JSON
    with open(os.path.join(agents_dir, "all_agents.json"), "w") as f:
        json.dump(agents_metadata, f, indent=2, ensure_ascii=False)

    # Build BeliefStore with the ordered POI node list from the graph
    belief_store = BeliefStore(poi_nodes=world_graph.poi_nodes)

    # Initialize simulation (belief_store wired in)
    sim = Simulation(world_graph, belief_store=belief_store, verbose=not args.quiet)
    for agent in agents:
        sim.register_agent(agent)

    # Run simulation for each agent
    total_trajectories = n_agents * m_trajectories
    if args.quiet:
        with tqdm(total=total_trajectories, desc="Simulating", ncols=80,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            for agent in agents:
                for _ in range(m_trajectories):
                    hour = sample_simulation_hour()
                    sim.step(
                        agent_id=agent.id,
                        current_hour=hour,
                        path_temp=30.0,
                        belief_update_dist=100,
                    )
                    pbar.update(1)
    else:
        print(f"Running simulation: {n_agents} agents × {m_trajectories} trajectories")
        for agent in tqdm(agents, desc="Simulating Agents"):
            for _ in range(m_trajectories):
                hour = sample_simulation_hour()
                sim.step(
                    agent_id=agent.id,
                    current_hour=hour,
                    path_temp=30.0,
                    belief_update_dist=100,
                )

    # Save trajectories (no inline belief data)
    with open(os.path.join(traj_dir, "all_trajectories.json"), "w") as f:
        json.dump(sim.trajectories, f, indent=2, ensure_ascii=False)

    # Save beliefs to separate compressed archive
    belief_store.save(beliefs_dir)

    if not args.quiet:
        print(f"\n✅ Simulation complete. Data saved to: {base_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run trajectory simulation and save agent/trajectory data."
    )
    parser.add_argument(
        "-n", "--num_agents", type=int, default=None,
        help="Number of agents to simulate."
    )
    parser.add_argument(
        "-m", "--num_trajectories", type=int, default=None,
        help="Number of trajectories per agent."
    )
    parser.add_argument(
        "-x", "--run_id", type=str, default=None,
        help="Run identifier to store outputs under data/simulation_data/run_<X>/"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Quiet mode: suppress verbose output, only show progress bar."
    )
    parser.add_argument(
        "--test_beliefs", action="store_true",
        help=(
            "Run a small validation test: 10 agents × 50 trajectories saved to "
            "run_test_beliefs. Overrides -n / -m / -x. Verifies the BeliefStore "
            "output format end-to-end."
        )
    )

    args = parser.parse_args()

    # Validate: unless --test_beliefs is set, -n / -m / -x are all required
    if not args.test_beliefs:
        missing = [
            flag for flag, val in [("-n", args.num_agents),
                                   ("-m", args.num_trajectories),
                                   ("-x", args.run_id)]
            if val is None
        ]
        if missing:
            parser.error(
                f"The following arguments are required when not using --test_beliefs: "
                f"{', '.join(missing)}"
            )

    return args


if __name__ == "__main__":
    run_simulation()
