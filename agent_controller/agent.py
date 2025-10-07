import uuid
import numpy as np
import haversine as hs
from graph_controller.world_graph import WorldGraph


"""agent_controller.agent
---------------------------------
Defines the Agent class which encapsulates simple preference sampling
and (placeholder) plan-building for an agent moving on a graph of POIs.

Design notes:
- Category-level preferences are always sampled on construction.
- Node-level preferences (e.g. `food_preferences`) require a graph `G`
  with node attribute `Category`. If `G` is not provided the node-level
  preference dicts are created empty and the caller can initialize them
  later with `agent.G = G; agent.set_all_preferences()`.
"""


class Agent:
    """A lightweight agent with hierarchical preferences.

    Attributes:
        id (str): short unique identifier for the agent instance.
        G (networkx.Graph | None): map graph used for node-level preferences.
        category_preferences (dict): weights over top-level categories.
        <category>_preferences (dict): per-node weights for that category.
        belief_state (dict): placeholder for agent's internal beliefs.
        current_goal: the current target node (if any).
        current_path (list): list of nodes representing the planned path.
    """

    def __init__(self, agent_id, world_graph):
        """Create an Agent.

        Args:
            agent_id (str | None): optional external id. If falsy, a short
                UUID is generated automatically.
            G (networkx.Graph | None): optional graph used to create
                node-level preference distributions. If omitted, node-level
                preference dicts are initialized empty and can be filled
                later by calling the corresponding setters after assigning
                `agent.G`.
        """
        print("Initializing Agent...")
        # Identifier
        self.id = agent_id if agent_id else str(uuid.uuid4())[:8]

        # Random decay rate for memory (exponential forgetting)
        self.decay_rate = np.random.uniform(0.90, 0.99)

        # Graph may be None at construction time. Assign early so setters
        # can use it if provided.
        self.world_graph = world_graph
        self.G = world_graph.G
        self.goal_nodes = self.world_graph.poi_nodes

        # Set the belief probabilities for all the goal nodes
        self.init_beliefs()

        # Sample high-level category preferences (always available)
        self.set_category_preferences()

        # Only sample node-level preferences when a graph is provided.
        # If no graph is present, create empty dict placeholders so code
        # accessing e.g. `agent.food_preferences` doesn't raise AttributeError.
        if self.G is not None:
            self.set_all_preferences()
        else:
            for cat in ["home", "study", "food", "leisure", "errands", "health"]:
                setattr(self, f"{cat}_preferences", {})



    # -----------------------------------
    # Node-level preference setters below
    # -----------------------------------

    def init_beliefs(self):
        """
        Initializes the agent's belief state for each goal node as a Beta(2,2)
        distribution over each hour of the day (00 to 23). Assumes self.goal_nodes
        has been set and contains all relevant node IDs.
        """
        if not hasattr(self, "goal_nodes"):
            raise ValueError("Agent must have self.goal_nodes defined before initializing beliefs.")

        self.belief_state = {}  # Clear any existing beliefs

        for node_id in self.goal_nodes:
            alpha = np.full(24, 2.0)
            beta = np.full(24, 2.0)

            # Increase confidence for common open hours (9am to 5pm)
            for h in range(9, 17):
                alpha[h] = 4.0
                beta[h] = 1.0

            temporal_belief = alpha / (alpha + beta)

            self.belief_state[node_id] = {
                "alpha": alpha,
                "beta": beta,
                "temporal_belief": temporal_belief
            }

    def update_beliefs(self, current_node, hour, distance=100):
        """
        Update beliefs over nearby goal nodes based on whether they are observed open/closed at the current hour.
        Only nodes within `distance` meters are considered.

        Args:
            current_node (int): Node ID where the agent currently is.
            hour (int): Current hour (0–23).
            distance (float): Radius (in meters) within which to update beliefs.
        """
        if self.G is None:
            raise ValueError("Graph G must be set to update beliefs.")

        # Precompute (lat, lon) for current node
        node_lat = self.G.nodes[current_node]['y']
        node_lon = self.G.nodes[current_node]['x']

        for node_id in self.goal_nodes:
            target_data = self.G.nodes[node_id]
            target_lat = target_data['y']
            target_lon = target_data['x']

            # Compute haversine distance (in meters)
            dist = hs.haversine((node_lat, node_lon), (target_lat, target_lon))
            if dist > distance:
                continue

            # Check whether the goal node is expected to be open at the current hour
            hours = target_data.get("opening_hours")
            if hours is None:
                continue  # No info, skip

            is_open = (hours["open"] <= hour < hours["close"])

            alpha = self.belief_state[node_id]["alpha"]
            beta = self.belief_state[node_id]["beta"]

            # Apply exponential decay to all hours first
            alpha[hour] *= self.decay_rate
            beta[hour] *= self.decay_rate

            # Update Beta parameters based on observation
            if is_open:
                alpha[hour] += 1
            else:
                beta[hour] += 1

            # Update cached temporal belief at this hour
            a, b = alpha[hour], beta[hour]
            self.belief_state[node_id]["temporal_belief"][hour] = np.clip(a / (a + b), 0.01, 0.99)

    def _set_preferences(self, category_name, concentration_func):
        """
        Generic setter: samples from Dirichlet and assigns to <category>_preferences
        """
        # Ensure a graph is available before attempting to read nodes.
        if self.G is None:
            raise ValueError("Graph G must be provided to set preferences.")

        # Collect nodes matching the requested top-level category. The
        # graph is expected to have a node attribute named 'Category'.
        candidate_nodes = [
            node for node, data in self.G.nodes(data=True)
            if data.get("Category") == category_name
        ]

        if not candidate_nodes:
            raise ValueError(f"No nodes found in category '{category_name}'")

        # Concentration vector determines how concentrated the Dirichlet
        # sampling will be. Larger values -> more even distribution.
        alpha = concentration_func(len(candidate_nodes))
        weights = np.random.dirichlet(alpha)

        # Store a small rounded float per node for reproducibility and
        # readability when printing/debugging.
        pref_dict = {
            node: float(round(w, 5))
            for node, w in zip(candidate_nodes, weights)
        }

        setattr(self, f"{category_name}_preferences", pref_dict)

    def set_category_preferences(self):
        """
        Samples high-level category preferences using a fixed Dirichlet prior
        that encodes general agent behavior patterns.
        """
        categories = ["home", "study", "food", "leisure", "errands", "health"]

        # Higher concentration on typical frequent activities
        alpha = np.array([4.0, 3.0, 2.5, 1.5, 1.0, 0.5])  # home > study > food > leisure > errands > health

        weights = np.random.dirichlet(alpha)

        self.category_preferences = {
            cat: float(round(w, 4)) for cat, w in zip(categories, weights)
        }

    def set_home_preferences(self):
        # Mostly one node (e.g., your residence)
        def alpha_fn(n): 
            alpha = np.ones(n) * 0.01
            # Give one node a large mass so the agent has a strong "home" node
            alpha[np.random.choice(n)] = 5.0
            return alpha

        self._set_preferences("home", alpha_fn)

    def set_food_preferences(self):
        # Mild concentration, maybe 2-3 favorites
        def alpha_fn(n):
            alpha = np.ones(n) * 0.3
            # Boost a few nodes so the distribution has several favorites
            for _ in range(min(3, n)):
                alpha[np.random.choice(n)] += 2.0
            return alpha

        self._set_preferences("food", alpha_fn)

    def set_study_preferences(self):
        # Mild concentration, maybe 2-3 favorites
        def alpha_fn(n):
            alpha = np.ones(n) * 0.4
            # Boost a few nodes so the distribution has several favorites
            for _ in range(min(3, n)):
                alpha[np.random.choice(n)] += 2.0
            return alpha

        self._set_preferences("study", alpha_fn)

    def set_health_preferences(self):
        # One primary care node + few alternates
        def alpha_fn(n):
            alpha = np.ones(n) * 0.1
            # Make one node noticeably more likely
            alpha[np.random.choice(n)] = 3.0
            return alpha

        self._set_preferences("health", alpha_fn)

    def set_errands_preferences(self):
        # Fairly spread, low preference
        def alpha_fn(n):
            return np.ones(n) * 0.5

        self._set_preferences("errands", alpha_fn)

    def set_leisure_preferences(self):
        # Slight preference spread, a couple of favorites
        def alpha_fn(n):
            alpha = np.ones(n) * 0.2
            for _ in range(min(2, n)):
                alpha[np.random.choice(n)] += 1.5
            return alpha

        self._set_preferences("leisure", alpha_fn)

    def set_all_preferences(self):
        """
        Call all category-specific setters in one step
        """
        # Order is not important here; this convenience method simply
        # initializes the node-level preference dictionaries for all
        # categories from the currently assigned graph `self.G`.
        self.set_food_preferences()
        self.set_study_preferences()
        self.set_health_preferences()
        self.set_home_preferences()
        self.set_errands_preferences()
        self.set_leisure_preferences()

    def __str__(self):
        """Provides a human-readable summary of the agent's preferences."""
        s = f"Agent {self.id}\n"
        s += "-" * (len(s) - 1) + "\n"

        # Category preferences
        s += "Category Preferences:\n"
        if hasattr(self, 'category_preferences'):
            # Sort by weight descending for readability
            sorted_prefs = sorted(
                self.category_preferences.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
            for category, weight in sorted_prefs:
                s += f"  - {category:<10}: {weight:.4f}\n"
        else:
            s += "  - Not set.\n"
        s += "\n"

        # Node-level preference summaries
        s += "Node-Level Preferences:\n"
        categories = ["home", "study", "food", "leisure", "errands", "health"]
        for cat in categories:
            prefs_attr = f"{cat}_preferences"
            if hasattr(self, prefs_attr):
                prefs = getattr(self, prefs_attr)
                if prefs:
                    total = sum(prefs.values())
                    # Use a tolerance for floating point comparisons. The rtol is
                    # increased slightly to account for accumulated rounding errors
                    # from individual preference weights.
                    is_normalized = np.isclose(total, 1.0, rtol=.001)
                    status = "OK" if is_normalized else "FAIL"
                    
                    s += f"  - {cat:<10}: {status} (sum={total:.4f}, {len(prefs)} nodes)\n"

                    # Get top 3 nodes by preference weight
                    sorted_nodes = sorted(
                        prefs.items(), 
                        key=lambda item: item[1], 
                        reverse=True
                    )
                    top_3 = sorted_nodes[:3]

                    s += f"    - Top 3:\n"
                    for node, weight in top_3:
                        # The node ID can be long, so we'll truncate it for display
                        node_display = node if len(node) < 30 else node[:27] + "..."
                        s += f"      - {node_display:<30}: {weight:.4f}\n"

                else:
                    s += f"  - {cat:<10}: Not set (empty)\n"
            else:
                s += f"  - {cat:<10}: Not available\n"
        
        return s

    def __repr__(self):
        """Formal representation, calls __str__ for readable output."""
        return self.__str__()

    def build_plan(self):
        """
        Placeholder for plan-building logic.
        Step 1: sample category from category_preferences.
        Step 2: sample a node from the corresponding node-level preferences.
        Step 3: compute path to that node.
        """
        # TODO: implement planning/scheduling logic. A simple implementation
        # could sample a target category using `self.category_preferences`,
        # then select a target node from the corresponding
        # `<category>_preferences` dict and call a path-finding routine on
        # `self.G` (e.g., networkx.shortest_path) to populate
        # `self.current_path`.
        raise NotImplementedError("Agent.build_plan is a placeholder")
