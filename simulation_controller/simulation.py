import random
from agent_controller.agent import Agent
from agent_controller.planning_utils import plan_path

class Simulation:
    def __init__(self, world_graph, verbose=True):
        """
        Args:
            world_graph (WorldGraph): The world graph wrapper.
            verbose (bool): Whether to print status messages during simulation.
        """
        self.G = world_graph.G
        self.agents = {}  # {agent_id: Agent}
        self.trajectories = {}  # {agent_id: list of paths}
        self.verbose = verbose

    def register_agent(self, agent):
        """
        Register an agent to the simulation.
        """
        self.agents[agent.id] = agent
        self.trajectories[agent.id] = []

    def step(self, agent_id, current_hour, path_temp=100.0, belief_update_dist=100):
        """
        Executes one full planning and traversal step for a single agent.

        This involves planning a path (with retry logic for closed goals),
        traversing it, and updating beliefs along the way.

        Args:
            agent_id (str): The ID of the agent to step.
            current_hour (int): The current hour of the simulation.
            path_temp (float): Temperature for stochastic path planning.
            belief_update_dist (int): Distance within which to update beliefs.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            if self.verbose:
                print(f"Agent {agent_id} not found.")
            return

        # Plan a path with retry logic for closed goals
        try:
            plan_result = plan_path(agent, current_hour, temperature=path_temp)
        except RuntimeError as e:
            if self.verbose:
                print(f"Error planning path for agent {agent_id}: {e}")
            return
        
        # Extract plan details
        annotated_path = plan_result['path']  # List of (node, goal) tuples
        goal_node = plan_result['goal_node']
        start_node = plan_result['start_node']
        attempts = plan_result['attempts']
        returned_home = plan_result['returned_home']
        attempted_goals = plan_result['attempted_goals']
        
        # Initialize belief snapshots list
        belief_snapshots = []
        
        # Traverse the path and update beliefs periodically
        for i, (node, _) in enumerate(annotated_path):
            # Update beliefs every 5 nodes along the path
            if i % 5 == 0:
                agent.update_beliefs(
                    current_node=node, 
                    hour=current_hour, 
                    distance=belief_update_dist
                )
                
                # Capture belief snapshot after update
                temporal_beliefs = {}
                for node_id, belief_data in agent.belief_state.items():
                    # Convert numpy array to list for JSON serialization
                    temporal_beliefs[node_id] = belief_data["temporal_belief"].tolist()
                
                belief_snapshots.append({
                    "step_index": i,
                    "node": node,
                    "temporal_beliefs": temporal_beliefs
                })
        
        # Store the generated trajectory with belief snapshots
        trajectory_data = {
            "path": annotated_path,  # List of (node_id, goal_node) tuples
            "goal_node": goal_node,
            "start_node": start_node,
            "hour": current_hour,
            "agent_id": agent_id,  # Add agent_id for clarity
            "attempts": attempts,
            "returned_home": returned_home,
            "attempted_goals": attempted_goals,
            "belief_snapshots": belief_snapshots  # Add belief history
        }
        self.trajectories[agent_id].append(trajectory_data)
        
        if self.verbose:
            path_len = len(annotated_path)
            attempts_str = f"{attempts} attempt(s)"
            home_str = " (returned home)" if returned_home else ""
            print(f"Agent {agent_id}: path length {path_len}, {attempts_str}{home_str}")




