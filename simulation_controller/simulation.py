import random
from agent_controller.agent import Agent
from agent_controller.planning_utils import plan_path


class Simulation:
    def __init__(self, world_graph, belief_store=None, verbose=True):
        """
        Args:
            world_graph (WorldGraph): The world graph wrapper.
            belief_store (BeliefStore | None): Optional belief store for
                recording per-trajectory belief snapshots separately from
                trajectory path data. When None, no belief recording occurs.
            verbose (bool): Whether to print status messages during simulation.
        """
        self.G = world_graph.G
        self.agents = {}        # {agent_id: Agent}
        self.trajectories = {}  # {agent_id: list of trajectory dicts}
        self.belief_store = belief_store
        self.verbose = verbose

        # Track how many trajectories have been run per agent so BeliefStore
        # can use a stable zero-based trajectory index.
        self._traj_counters = {}  # {agent_id: int}

    def register_agent(self, agent):
        """Register an agent to the simulation."""
        self.agents[agent.id] = agent
        self.trajectories[agent.id] = []
        self._traj_counters[agent.id] = 0

    def step(self, agent_id, current_hour, path_temp=100.0, belief_update_dist=100):
        """
        Executes one full planning and traversal step for a single agent.

        Plans a path (with retry logic for closed goals), traverses it, and
        updates beliefs along the way. If a BeliefStore is attached, the
        agent's initial belief state and per-snapshot beliefs are recorded
        there instead of being embedded inline in the trajectory dict.

        Args:
            agent_id (str): The ID of the agent to step.
            current_hour (int): The current hour of the simulation.
            path_temp (float): Temperature for stochastic path planning.
            belief_update_dist (int): Distance (metres) within which to update
                beliefs.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            if self.verbose:
                print(f"Agent {agent_id} not found.")
            return

        traj_idx = self._traj_counters[agent_id]

        # Capture the agent's belief state BEFORE traversal so the store
        # records accumulated cross-trajectory knowledge.
        if self.belief_store is not None:
            self.belief_store.record_initial_state(agent_id, traj_idx, agent.belief_state)

        # Plan a path with retry logic for closed goals
        try:
            plan_result = plan_path(agent, current_hour, temperature=path_temp)
        except RuntimeError as e:
            if self.verbose:
                print(f"Error planning path for agent {agent_id}: {e}")
            return

        # Extract plan details
        annotated_path  = plan_result['path']          # List of (node, goal) tuples
        goal_node       = plan_result['goal_node']
        start_node      = plan_result['start_node']
        attempts        = plan_result['attempts']
        returned_home   = plan_result['returned_home']
        attempted_goals = plan_result['attempted_goals']

        # Traverse the path and update beliefs periodically
        for i, (node, _) in enumerate(annotated_path):
            # Update beliefs every 5 nodes along the path
            if i % 5 == 0:
                agent.update_beliefs(
                    current_node=node,
                    hour=current_hour,
                    distance=belief_update_dist,
                )

                if self.belief_store is not None:
                    self.belief_store.record_snapshot(
                        agent_id=agent_id,
                        traj_idx=traj_idx,
                        step_idx=i,
                        belief_state=agent.belief_state,
                    )

        # Build trajectory record (belief data lives in BeliefStore, not here)
        trajectory_data = {
            "path":            annotated_path,
            "goal_node":       goal_node,
            "start_node":      start_node,
            "hour":            current_hour,
            "agent_id":        agent_id,
            "attempts":        attempts,
            "returned_home":   returned_home,
            "attempted_goals": attempted_goals,
        }
        self.trajectories[agent_id].append(trajectory_data)
        self._traj_counters[agent_id] += 1

        if self.verbose:
            path_len = len(annotated_path)
            home_str = " (returned home)" if returned_home else ""
            print(f"Agent {agent_id}: path length {path_len}, "
                  f"{attempts} attempt(s){home_str}")
