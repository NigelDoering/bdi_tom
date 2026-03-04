import uuid
import numpy as np
import haversine as hs


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

    def __init__(self, agent_id, world_graph, verbose=False):
        """Create an Agent.

        Args:
            agent_id (str | None): optional external id. If falsy, a short
                UUID is generated automatically.
            G (networkx.Graph | None): optional graph used to create
                node-level preference distributions. If omitted, node-level
                preference dicts are initialized empty and can be filled
                later by calling the corresponding setters after assigning
                `agent.G`.
            verbose (bool): Whether to print initialization messages.
        """
        if verbose:
            print("Initializing Agent...")
        # Identifier
        self.id = agent_id if agent_id else str(uuid.uuid4())[:8]

        # Random decay rate for memory (exponential forgetting)
        self.decay_rate = np.random.uniform(0.00005, 0.00015)

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
        Initializes the agent's belief state for each goal node using an
        uninformative Beta(1, 1) prior — a uniform distribution over [0, 1] —
        for every hour of the day (00 to 23).

        Initialises the agent's belief state using a weakly informative prior
        derived from each node's known opening hours.

        Nodes with opening_hours in the graph get:
          - Beta(3, 1) ≡ belief = 0.75 for hours the node is scheduled open
          - Beta(1, 3) ≡ belief = 0.25 for hours it is scheduled closed
        Nodes without opening_hours default to Beta(1, 1) ≡ 0.5 (uninformative).

        A weakly informative prior is preferable to a flat Beta(1,1) everywhere
        because it immediately provides a meaningful signal to the goal-selection
        formula: nodes that are scheduled closed at the current hour start with
        lower belief and receive less selection weight before any observations
        have been made.  Observations then update these priors in the standard
        Bayesian way, refining rather than replacing them.

        The decay target is the same weakly informative prior so forgetting
        pulls beliefs back toward the scheduled pattern, not toward 0.5.
        """
        if not hasattr(self, "goal_nodes"):
            raise ValueError("Agent must have self.goal_nodes defined before initializing beliefs.")

        self.belief_state = {}  # Clear any existing beliefs
        self.belief_prior = {}  # Store initial priors for decay

        for node_id in self.goal_nodes:
            alpha = np.ones(24, dtype=np.float64)
            beta  = np.ones(24, dtype=np.float64)

            # Seed prior from opening hours if available
            if self.G is not None:
                hours_info = self.G.nodes[node_id].get('opening_hours')
                if hours_info and isinstance(hours_info, dict):
                    open_h  = hours_info.get('open',  0)
                    close_h = hours_info.get('close', 24)
                    for h in range(24):
                        if close_h < open_h:  # wraparound (e.g. 22:00–02:00)
                            is_open = (h >= open_h or h < close_h)
                        else:
                            is_open = (open_h <= h < close_h)
                        if is_open:
                            alpha[h] = 3.0  # Beta(3,1) → mean = 0.75
                            beta[h]  = 1.0
                        else:
                            alpha[h] = 1.0  # Beta(1,3) → mean = 0.25
                            beta[h]  = 3.0

            temporal_belief = alpha / (alpha + beta)

            self.belief_state[node_id] = {
                "alpha": alpha.copy(),
                "beta":  beta.copy(),
                "temporal_belief": temporal_belief,
            }

            # Decay target: same weakly informative prior
            self.belief_prior[node_id] = {
                "alpha": alpha.copy(),
                "beta":  beta.copy(),
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
            dist = hs.haversine((node_lat, node_lon), (target_lat, target_lon), unit=hs.Unit.METERS)
            
            alpha = self.belief_state[node_id]["alpha"]
            beta = self.belief_state[node_id]["beta"]
            
            if dist <= distance:
                # POI is SEEN - update based on observation, NO decay
                hours = target_data.get("opening_hours")
                if hours is None:
                    continue  # No opening hours info, skip
                
                is_open = self._is_node_open(node_id, hour)
                
                # Update Beta parameters based on observation (no decay)
                if is_open:
                    alpha[hour] += 1
                else:
                    beta[hour] += 1
            else:
                # POI is NOT SEEN - decay toward original prior for all 24 hours
                # This gradually moves beliefs back to the agent's initial beliefs
                prior_alpha = self.belief_prior[node_id]["alpha"]
                prior_beta = self.belief_prior[node_id]["beta"]
                
                # Exponential moving average toward prior
                # decay_rate controls how fast we forget (higher = faster forgetting)
                alpha[:] = (1 - self.decay_rate) * alpha + self.decay_rate * prior_alpha
                beta[:] = (1 - self.decay_rate) * beta + self.decay_rate * prior_beta

            # Update cached temporal belief for all hours (since decay affects all hours)
            self.belief_state[node_id]["temporal_belief"] = np.clip(
                alpha / (alpha + beta), 0.01, 0.99
            )

    def _is_node_open(self, node_id, hour):
        """Helper to determine whether a POI is open at a given hour."""
        node_data = self.G.nodes[node_id]
        hours = node_data.get("opening_hours")
        if hours is None:
            return True

        open_time = hours.get("open", 0)
        close_time = hours.get("close", 24)

        # Support hours that end at or past midnight by normalizing the range.
        if close_time < open_time:
            close_time += 24
            hour_check = hour if hour >= open_time else hour + 24
        else:
            hour_check = hour

        return open_time <= hour_check < close_time

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
        alpha = np.array([1.5, 3.0, 2.5, 1.5, 1.0, 0.5])  # study > food > home/leisure > errands > health

        weights = np.random.dirichlet(alpha)

        self.category_preferences = {
            cat: float(round(w, 4)) for cat, w in zip(categories, weights)
        }

    def get_category_preferences(self, hour):
        """
        Returns the agent's category preferences for a given hour.
        
        For now, returns the same preferences regardless of hour.
        This can be extended to support time-dependent preferences in the future.
        
        Args:
            hour (int): The current hour (0-23).
            
        Returns:
            dict: Category preferences mapping category names to probabilities.
        """
        if not hasattr(self, 'category_preferences'):
            raise ValueError("Category preferences have not been initialized for this agent.")
        
        return self.category_preferences

    def set_home_preferences(self):
        # One preferred "home" node with mild concentration; all other nodes
        # get α=1.0 so the top node captures ~8% instead of ~89%.
        def alpha_fn(n): 
            alpha = np.ones(n) * 1.0
            alpha[np.random.choice(n)] = 5.0
            return alpha

        self._set_preferences("home", alpha_fn)

    def set_food_preferences(self):
        # Flat α=2.0 across all food nodes — higher concentration parameter
        # means draws cluster tighter around uniform (top node ~8.5%).
        def alpha_fn(n):
            return np.ones(n) * 2.0

        self._set_preferences("food", alpha_fn)

    def set_study_preferences(self):
        # Flat α=3.0 over all 101 study nodes — draws concentrate tighter
        # around uniform (top node ~3% vs ~5% at α=1.0).
        def alpha_fn(n):
            return np.ones(n) * 3.0

        self._set_preferences("study", alpha_fn)

    def set_health_preferences(self):
        # Mild preference for one health node; all others get α=1.0 so the
        # top node captures ~24% instead of ~73%.
        def alpha_fn(n):
            alpha = np.ones(n) * 1.0
            alpha[np.random.choice(n)] = 2.0
            return alpha

        self._set_preferences("health", alpha_fn)

    def set_errands_preferences(self):
        # α=3.0 for all 8 errands nodes — keeps top node ~24% vs ~43% at α=0.5.
        def alpha_fn(n):
            return np.ones(n) * 3.0

        self._set_preferences("errands", alpha_fn)

    def set_leisure_preferences(self):
        # α=3.0 for all 8 leisure nodes — draws cluster tighter around
        # uniform so top node is ~24% instead of ~34% at α=1.0.
        def alpha_fn(n):
            return np.ones(n) * 3.0

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

    def to_dict(self):
        """
        Returns a dictionary representation of the agent that can be saved to JSON.
        Excludes dynamic state like current belief updates or paths.
        """
        agent_data = {
            "id": self.id,
            "decay_rate": round(self.decay_rate, 4),  # round for readability
            "category_preferences": self.category_preferences,
        }

        # Include subcategory (node-level) preferences
        for category in ["home", "study", "food", "leisure", "errands", "health"]:
            cat_attr = f"{category}_preferences"
            if hasattr(self, cat_attr):
                agent_data[cat_attr] = getattr(self, cat_attr)

        return agent_data
