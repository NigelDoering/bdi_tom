import numpy as np
import networkx as nx

def sample_goal_node(agent, current_hour, category=None, exclude_nodes=None, return_category=False):
    """
    Sample a goal node for the agent based on its preferences and current beliefs.

    Args:
        agent (Agent): The agent for whom to sample a goal node.
        current_hour (int): The current hour of the day (0-23).
        category (str, optional): If provided, only sample from this category.
        exclude_nodes (set, optional): Set of node IDs to exclude from sampling.
        return_category (bool): If True, return (node_id, category) tuple instead of just node_id.
        
    Returns:
        node_id (str) or (node_id, category) tuple: The sampled goal node ID, 
        optionally with its category. Returns None (or (None, None)) if no valid node found.
    """
    exclude_nodes = exclude_nodes or set()
    category_probs = agent.get_category_preferences(current_hour)

    # Sample or use provided category
    if category is None:
        # Extract category names and their corresponding probabilities
        categories = list(category_probs.keys())
        probabilities = list(category_probs.values())
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()  # Normalize to ensure sum=1

        # Sample a single category based on the agent's high-level preferences
        sampled_category = np.random.choice(categories, p=probabilities)
    else:
        sampled_category = category

    # Retrieve the node-level preferences for the chosen category
    node_preferences = getattr(agent, f"{sampled_category}_preferences")

    # Filter out excluded nodes
    available_nodes = {k: v for k, v in node_preferences.items() if k not in exclude_nodes}
    
    if not available_nodes:
        return (None, sampled_category) if return_category else None

    # Get the temporal belief for each candidate node at the current hour
    belief_probs = {
        node_id: agent.belief_state[node_id]['temporal_belief'][current_hour]
        for node_id in available_nodes.keys()
    }

    # Combine preferences with the odds-ratio form of the belief.
    #
    # Why odds-ratio instead of linear (pref * belief)?
    #   Linear: belief=0.75 → weight×0.75, belief=0.25 → weight×0.25  →  3× difference
    #   Odds:   belief=0.75 → weight×3.0,  belief=0.25 → weight×0.33  →  9× difference
    #
    # More importantly, when all beliefs sit near 0.5 (uninformative prior), the
    # linear form multiplies every preference by ~0.5, which cancels in
    # normalisation — beliefs have zero effect on rankings.  The odds-ratio form
    # maps 0.5 → 1.0 (neutral), so equal beliefs still cancel, but any divergence
    # from 0.5 creates a proportional boost or penalty.
    #
    # The hard BELIEF_THRESHOLD is removed: the 9× odds-ratio penalty already
    # strongly discourages closed nodes without hard-blocking entire categories,
    # which prevented off-peak trajectories from being generated at all.
    combined_weights = {}
    for node_id in available_nodes:
        pref   = available_nodes[node_id]
        belief = belief_probs[node_id]
        odds   = belief / (1.0 - belief + 1e-9)   # 0.75→3.0, 0.5→1.0, 0.25→0.33
        combined_weights[node_id] = pref * odds

    if not combined_weights:
        return (None, sampled_category) if return_category else None

    # Normalize
    total = sum(combined_weights.values())
    if total == 0:
        return (None, sampled_category) if return_category else None
    normalized_weights = {node: w / total for node, w in combined_weights.items()}

    # Sample
    node_ids = list(normalized_weights.keys())
    probs = list(normalized_weights.values())
    sampled_node = np.random.choice(node_ids, p=probs)
    
    if return_category:
        return sampled_node, sampled_category
    return sampled_node

def sample_start_node(agent, goal_node, current_hour, min_dist=500, max_tries=100):
    """
    Samples a start node based on the agent's preferences and beliefs from an earlier time.
    
    Uses the same preference-based sampling as goal selection, but with beliefs from
    3-4 hours before the current hour to represent where the agent was previously.
    Ensures the start node is sufficiently far from the goal node.

    Args:
        agent (Agent): The agent, which holds the graph `G`.
        goal_node (str): The ID of the goal node.
        current_hour (int): The current hour of the day (0-23).
        min_dist (int | float): The minimum shortest path distance required
            between the start and goal nodes. The distance is based on the
            'length' attribute of the graph edges.
        max_tries (int): The maximum number of attempts to find a suitable node.

    Returns:
        str: The ID of a suitable start node.

    Raises:
        RuntimeError: If a suitable start node cannot be found within `max_tries`.
    """
    # Use beliefs from 3-4 hours earlier (wrapping around 24-hour clock)
    hours_back = np.random.randint(3, 5)  # Random between 3-4 hours
    start_hour = (current_hour - hours_back) % 24
    
    for _ in range(max_tries):
        # Sample a candidate start node using preference-based sampling from earlier hour
        candidate_start = sample_goal_node(agent, start_hour)
        
        if candidate_start is None:
            continue
        
        # Ensure the start and goal nodes are not the same
        if candidate_start == goal_node:
            continue
        
        try:
            # Calculate the shortest path length (distance)
            distance = nx.shortest_path_length(
                agent.G, 
                source=candidate_start, 
                target=goal_node, 
                weight='length'
            )
            
            # If the distance is sufficient, we've found our node
            if distance > min_dist:
                return candidate_start
        except nx.NetworkXNoPath:
            # Nodes are disconnected, try again
            continue
    
    # If we exit the loop, we failed to find a suitable node
    raise RuntimeError(
        f"Could not find a start node at least {min_dist}m away from "
        f"{goal_node} within {max_tries} attempts."
    )

def _sample_stochastic_path(agent, start_node, goal_node, temperature=100.0):
    """
    Internal helper: Plans a path from start to goal by sampling from the top K shortest paths.

    The function finds the K shortest paths, then samples one based on a 
    probability distribution that favors shorter paths. The `temperature` parameter
    controls the randomness of the selection.

    Args:
        agent (Agent): The agent, which holds the graph `G`.
        start_node (str): The ID of the starting node.
        goal_node (str): The ID of the goal node.
        temperature (float): Controls the randomness of path selection. Higher values
            lead to more uniform selection across paths. Lower values strongly favor
            the shortest path. Must be > 0.

    Returns:
        list[str]: A list of node IDs representing the sampled path.

    Raises:
        ValueError: If temperature is not positive.
        RuntimeError: If no path can be found between start and goal.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    
    if start_node == goal_node:
        return [start_node]

    # If the graph is a MultiGraph, convert to simple Graph for pathfinding
    # Keep only the shortest edge between each pair of nodes
    G_simple = agent.G
    if agent.G.is_multigraph():
        G_simple = nx.Graph()
        G_simple.add_nodes_from(agent.G.nodes(data=True))
        for u, v, data in agent.G.edges(data=True):
            if G_simple.has_edge(u, v):
                existing_length = G_simple[u][v].get('length', float('inf'))
                new_length = data.get('length', float('inf'))
                if new_length < existing_length:
                    G_simple[u][v]['length'] = new_length
            else:
                G_simple.add_edge(u, v, **data)

    # --- Robust path existence check ---
    if not G_simple.has_node(start_node) or not G_simple.has_node(goal_node):
        raise RuntimeError(f"Start or goal node does not exist in the graph: {start_node}, {goal_node}")
    if not nx.has_path(G_simple, start_node, goal_node):
        raise RuntimeError(f"No path exists between {start_node} and {goal_node} in the graph.")
    # -----------------------------------

    # Find the top K shortest paths (fast and robust approach)
    K = 5
    candidate_paths = []

    try:
        # Primary shortest path (fast)
        main_path = nx.shortest_path(G_simple, source=start_node, target=goal_node, weight='length')
        candidate_paths.append(main_path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise RuntimeError(f"No path found from {start_node} to {goal_node}.")
    except Exception as e:
        # Any unexpected error, try to fail gracefully
        raise RuntimeError(f"Error computing shortest path: {e}")

    # If we only need one path, return it quickly
    if K == 1:
        return candidate_paths[0]

    # Attempt to find simple alternative paths by removing single edges from the main path
    # This is cheaper than full Yen's algorithm and avoids complex graph copying.
    try:
        for i in range(len(main_path) - 1):
            if len(candidate_paths) >= K:
                break
            u = main_path[i]
            v = main_path[i + 1]

            # Create a lightweight view copy of the graph and remove the edge (u,v)
            G_mod = G_simple.copy()
            if G_mod.has_edge(u, v):
                try:
                    G_mod.remove_edge(u, v)
                except Exception:
                    continue

            try:
                alt_path = nx.shortest_path(G_mod, source=start_node, target=goal_node, weight='length')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            except Exception:
                # If any error occurs, skip this alternate
                continue

            tup = tuple(alt_path)
            if tup not in (tuple(p) for p in candidate_paths):
                candidate_paths.append(alt_path)
                if len(candidate_paths) >= K:
                    break
    except Exception:
        # If anything goes wrong in alternate generation, just proceed with whatever we have
        pass

    if not candidate_paths:
        raise RuntimeError(f"No path found from {start_node} to {goal_node} after attempts.")

    # Calculate the length of each path using stored base lengths (avoid repeated get_edge_data calls)
    edge_base = {}
    for u, v, d in G_simple.edges(data=True):
        base_len = float(d.get('length', 1e6))
        edge_base[(u, v)] = base_len
        edge_base[(v, u)] = base_len

    path_lengths = []
    for path in candidate_paths:
        total_length = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # prefer base length if available
            if (u, v) in edge_base:
                total_length += edge_base[(u, v)]
            else:
                edge_data = agent.G.get_edge_data(u, v)
                if edge_data is None:
                    total_length += 1e6
                elif isinstance(edge_data, dict) and 'length' in edge_data:
                    total_length += float(edge_data.get('length', 0.0))
                else:
                    try:
                        lengths = [float(ed.get('length', 0.0)) for ed in edge_data.values()]
                        total_length += min(lengths) if lengths else 0.0
                    except Exception:
                        total_length += 0.0
        path_lengths.append(total_length)

    # Convert lengths to probabilities using softmax
    logits = [-length / max(1e-6, temperature) for length in path_lengths]
    stable_logits = np.array(logits) - np.max(logits)
    probabilities = np.exp(stable_logits)
    probabilities_sum = probabilities.sum()
    if probabilities_sum <= 0 or not np.isfinite(probabilities_sum):
        probabilities = np.ones(len(candidate_paths), dtype=float) / len(candidate_paths)
    else:
        probabilities /= probabilities_sum

    selected_idx = np.random.choice(len(candidate_paths), p=probabilities)

    return candidate_paths[selected_idx]


def yen_k_shortest_paths(G, source, target, K=5, weight='length'):
    """
    Simple implementation of Yen's algorithm for K shortest loopless paths.
    Returns a list of paths (each path is a list of nodes) up to K paths.
    This implementation makes a shallow copy of the graph for spur computations
    and uses Dijkstra (networkx.shortest_path) for spur path finding.
    """
    if source == target:
        return [[source]]
    try:
        first_path = nx.shortest_path(G, source=source, target=target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    A = [first_path]
    B = []  # list of tuples (cost, path)

    def path_cost(path):
        cost = 0.0
        for u, v in zip(path[:-1], path[1:]):
            data = G.get_edge_data(u, v)
            if data is None:
                cost += 1e6
            elif isinstance(data, dict) and 'length' in data:
                cost += float(data.get('length', 0.0))
            else:
                try:
                    lengths = [float(ed.get('length', 0.0)) for ed in data.values()]
                    cost += min(lengths) if lengths else 0.0
                except Exception:
                    cost += 0.0
        return cost

    import heapq

    for k in range(1, K):
        for i in range(len(A[-1]) - 1):
            spur_node = A[-1][i]
            root_path = A[-1][: i + 1]

            # Make a copy for modifications
            G_copy = G.copy()

            # Remove the edges that would create previously found paths with same root
            for p in A:
                if len(p) > i and p[: i + 1] == root_path:
                    u = p[i]
                    v = p[i + 1]
                    if G_copy.has_edge(u, v):
                        G_copy.remove_edge(u, v)

            # Remove nodes in root_path except spur_node
            for n in root_path[:-1]:
                if G_copy.has_node(n):
                    G_copy.remove_node(n)

            try:
                spur_path = nx.shortest_path(G_copy, source=spur_node, target=target, weight=weight)
                total_path = root_path[:-1] + spur_path
                total_cost = path_cost(total_path)
                heapq.heappush(B, (total_cost, total_path))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # No spur path found for this root, skip
                continue

        # If no candidates, break
        if not B:
            break

        # Pop smallest candidate that's not already in A
        while B:
            cost, path_candidate = heapq.heappop(B)
            if path_candidate not in A:
                A.append(path_candidate)
                break
        else:
            break

    return A

def _is_node_open(agent, node_id, current_hour):
    """
    Check if a POI node is open at the given hour.
    
    Args:
        agent (Agent): The agent (provides access to graph).
        node_id (str): The node ID to check.
        current_hour (int): The current hour (0-23).
        
    Returns:
        bool: True if the node is open (or has no opening hours), False otherwise.
    """
    node_data = agent.G.nodes[node_id]
    opening_hours = node_data.get('opening_hours')
    
    # If no opening hours specified, assume always open
    if not opening_hours or not isinstance(opening_hours, dict):
        return True
    
    open_hour = opening_hours.get('open', 0)
    close_hour = opening_hours.get('close', 24)
    
    # Handle wraparound (e.g., open 22, close 2 means 10pm-2am)
    if close_hour < open_hour:
        return current_hour >= open_hour or current_hour < close_hour
    else:
        return open_hour <= current_hour < close_hour


def plan_path(agent, current_hour, temperature=100.0):
    """
    Plans a path from a sampled start node to a sampled goal node.

    The agent always travels to its chosen goal.  When it arrives, the
    trajectory is annotated with whether the goal turned out to be open
    or closed at the current hour.  The agent does **not** reroute to
    home when the goal is closed — doing so created a severe home-bias
    in the dataset because at off-peak hours most non-home POIs are
    closed, causing a cascade of retries that almost always terminated
    at a (24 h) home node.

    Instead, "goal was closed" is recorded as a flag so downstream
    consumers can decide how to treat the trajectory.  Belief updates
    during traversal still teach the agent about opening hours for
    future goal selection via the Bayesian belief system.

    Args:
        agent (Agent): The agent for whom to plan.
        current_hour (int): The current hour (0-23).
        temperature (float): Temperature for stochastic path selection.

    Returns:
        dict: Contains:
            - 'path': List of (node_id, goal_node) tuples for the journey
            - 'goal_node': The destination node
            - 'goal_category': The category of the sampled goal
            - 'start_node': The starting node
            - 'goal_open': True if the goal was open at current_hour
            - 'attempts': Always 1 (kept for backward compat)
            - 'returned_home': Always False (kept for backward compat)
            - 'attempted_goals': Single-element list with the goal
    """
    # Sample initial goal and category
    result = sample_goal_node(agent, current_hour, return_category=True)

    if result is None or result[0] is None:
        raise RuntimeError(f"Could not sample initial goal node for agent {agent.id}")

    goal_node, sampled_category = result

    # Sample start node
    start_node = sample_start_node(agent, goal_node, current_hour=current_hour)

    # Plan stochastic path from start to goal
    path_nodes = _sample_stochastic_path(agent, start_node, goal_node, temperature)

    # Check whether the goal is actually open at this hour
    goal_open = _is_node_open(agent, goal_node, current_hour)

    # Build annotated path: list of (node_id, goal_node) tuples
    annotated_path = [(node, goal_node) for node in path_nodes]

    return {
        'path': annotated_path,
        'goal_node': goal_node,
        'goal_category': sampled_category,
        'start_node': start_node,
        'goal_open': goal_open,
        'attempts': 1,
        'returned_home': False,
        'attempted_goals': [goal_node],
    }


def _sample_home_node(agent):
    """
    Sample the agent's home node (highest probability node in home category).
    Used only as a last resort when no other open node can be found.

    Args:
        agent (Agent): The agent.

    Returns:
        str: The home node ID.
    """
    home_prefs = agent.home_preferences
    if not home_prefs:
        raise RuntimeError(f"Agent {agent.id} has no home preferences")

    # Return the node with highest preference
    return max(home_prefs.items(), key=lambda x: x[1])[0]


def _combine_path_segments(segments):
    """
    Combine multiple path segments into a single annotated path.
    
    Args:
        segments: List of (path, goal_node) tuples where path is a list of node IDs.
        
    Returns:
        list: List of (node_id, goal_node) tuples representing the full journey.
    """
    combined = []
    
    for path, goal in segments:
        for i, node in enumerate(path):
            # Skip first node if it's the same as last node from previous segment
            if combined and i == 0 and node == combined[-1][0]:
                continue
            combined.append((node, goal))
    
    return combined








