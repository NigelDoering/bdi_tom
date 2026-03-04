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
    Sample a trajectory hour from a distribution that covers all 24 hours
    with a mild preference for daytime.

    The previous distribution concentrated ~85% of trajectories between
    6am and 8pm, causing agents to almost never observe closed locations
    and therefore never accumulate the downward (beta) belief updates
    needed for meaningful belief learning.

    The new distribution spans the full day with a peak-to-trough ratio
    of ~6x (vs ~24x previously).  Roughly 54% of samples fall in core
    daytime hours (9am-5pm), 25% in the evening (6pm-midnight), and 12%
    in the early morning (midnight-6am).  This ensures agents regularly
    encounter both open and closed locations, generating the bidirectional
    Bayesian updates that make belief tracking meaningful.

    Distribution rationale (university campus):
    - Late night / early morning (0-5): students returning late, early gym
    - Morning ramp (6-8): commuters, early classes
    - Daytime (9-17): lectures, study, lunch — moderate preference
    - Evening (18-21): dinner, evening study, social
    - Late evening (22-23): winding down

    Returns:
        int: Hour in 24-hour format (0-23)
    """
    weights = {
        0:  0.50,   # midnight
        1:  0.30,   # 1am
        2:  0.10,   # 2am
        3:  0.05,   # 3am
        4:  0.05,   # 4am
        5:  0.05,   # 5am
        6:  0.20,   # 6am
        7:  0.70,   # 7am
        8:  0.90,   # 8am
        9:  1.10,   # 9am
        10: 1.20,   # 10am
        11: 1.20,   # 11am
        12: 1.30,   # 12pm - mild peak
        13: 1.20,   # 1pm
        14: 1.10,   # 2pm
        15: 1.00,   # 3pm
        16: 0.90,   # 4pm
        17: 0.90,   # 5pm
        18: 1.00,   # 6pm - dinner
        19: 0.90,   # 7pm
        20: 0.80,   # 8pm
        21: 0.70,   # 9pm
        22: 0.60,   # 10pm
        23: 0.50,   # 11pm
    }

    hours = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]

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

    # Load and wrap the graph, applying hours config if provided
    graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G, hours_config_path=args.hours_config)

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
    parser.add_argument(
        "--hours_config", type=str, default=None,
        metavar="PATH",
        help=(
            "Path to a JSON hours config file that overrides the opening hours "
            "embedded in the graph.  Use data/processed/hours_config_base.json "
            "for the standard UCSD schedule, or hours_config_dynamic.json for "
            "the test-set world state with modified popular-node hours.  "
            "If omitted, graph-embedded hours are used (backward compatible)."
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
