import random
from agent_controller.agent import Agent
from agent_controller.planning_utils import sample_start_node, sample_goal_node, plan_stochastic_path

class Simulation:
    def __init__(self, world_graph):
        """
        Args:
            G (networkx.Graph): The annotated graph.
        """
        self.G = world_graph.G
        self.agents = {}  # {agent_id: Agent}
        self.trajectories = {}  # {agent_id: list of paths}

    def register_agent(self, agent):
        """
        Register an agent to the simulation.
        """
        self.agents[agent.id] = agent
        self.trajectories[agent.id] = []

    def step(self, agent_id, current_hour, path_temp=100.0, belief_update_dist=100):
        """
        Executes one full planning and traversal step for a single agent.

        This involves sampling a goal, planning a path, traversing it, and
        updating beliefs along the way.

        Args:
            agent_id (str): The ID of the agent to step.
            current_hour (int): The current hour of the simulation.
            path_temp (float): Temperature for stochastic path planning.
            belief_update_dist (int): Distance within which to update beliefs.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            print(f"Agent {agent_id} not found.")
            return

        # 1. Sample a goal for the agent
        goal_node = sample_goal_node(agent, current_hour)

        # 2. Sample a start node that is reasonably far from the goal
        start_node = sample_start_node(agent, goal_node, current_hour=current_hour)
  

        # 3. Plan a stochastic path from start to goal
        try:
            path = plan_stochastic_path(agent, start_node, goal_node, temperature=path_temp)
        except RuntimeError as e:
            print(f"Error planning path for agent {agent_id}: {e}")
            return
        
        # Store the generated trajectory
        trajectory_data = {
            "path": path,
            "goal_node": goal_node,
            "hour": current_hour
        }
        self.trajectories[agent_id].append(trajectory_data)

        # 4. Traverse the path and update beliefs periodically
        for i, node in enumerate(path):
            # Update beliefs every 5 nodes along the path
            if i % 5 == 0:
                agent.update_beliefs(
                    current_node=node, 
                    hour=current_hour, 
                    distance=belief_update_dist
                )
        
        print(f"Agent {agent_id} completed a path of length {len(path)} from {start_node} to {goal_node}.")




