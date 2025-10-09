import sys
import os
import networkx as nx

# Add project root to path to allow module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_controller.agent import Agent
from simulation_controller.simulation import Simulation
from graph_controller.world_graph import WorldGraph
from visualization_controller.map_plotter import plot_graph_with_trajectory  # ✅ NEW import

def main():
    """
    Main function to run the simulation test.
    """
    print("Starting simulation test...")

    # Load the graph
    graph_path = os.path.join(project_root, 'data', 'processed', 'ucsd_walk_full.graphml')
    try:
        G = nx.read_graphml(graph_path)
        print(f"Graph loaded successfully from: {graph_path}")
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except FileNotFoundError:
        print(f"Error: Graph file not found at {graph_path}")
        return

    # Create a WorldGraph object to hold the graph and ensure data is clean
    world_graph = WorldGraph(G)

    # Create an agent
    agent = Agent(agent_id="test_agent_001", world_graph=world_graph)
    print("Agent created successfully.")
    print(agent)

    # Initialize and run the simulation
    simulation = Simulation(world_graph)
    simulation.register_agent(agent)
    print("\nRunning simulation step...")
    simulation.step(agent_id=agent.id, current_hour=10, path_temp=30, belief_update_dist=100)

    # Print out the resulting trajectory
    print("\n--- Simulation Results ---")
    trajectories = simulation.trajectories[agent.id]
    if trajectories:
        last_trajectory = trajectories[-1]
        print(f"Agent: {agent.id}")
        print(f"  Hour: {last_trajectory['hour']}")
        print(f"  Goal: {last_trajectory['goal_node']}")
        print(f"  Path Length: {len(last_trajectory['path'])}")
        print(f"  Path Start: {last_trajectory['path'][0]}")
        print(f"  Path End: {last_trajectory['path'][-1]}")

        # ✅ Save and visualize trajectory
        map_output_path = os.path.join(project_root, "outputs", f"{agent.id}_trajectory_map.html")
        os.makedirs(os.path.dirname(map_output_path), exist_ok=True)
        plot_graph_with_trajectory(G, last_trajectory, save_path=map_output_path)
        print(f"Trajectory map saved to {map_output_path}")

    else:
        print("No trajectories were generated.")

if __name__ == "__main__":
    main()