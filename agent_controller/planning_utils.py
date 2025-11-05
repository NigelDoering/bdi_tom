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

    # Combine preferences and beliefs
    combined_weights = {}
    for node_id in available_nodes:
        pref = available_nodes[node_id]
        belief = belief_probs[node_id]
        combined_weights[node_id] = pref * belief

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
        
        # For each pair of nodes with edges, keep only the shortest one
        for u, v, data in agent.G.edges(data=True):
            if G_simple.has_edge(u, v):
                # Compare with existing edge and keep shorter one
                existing_length = G_simple[u][v].get('length', float('inf'))
                new_length = data.get('length', float('inf'))
                if new_length < existing_length:
                    G_simple[u][v]['length'] = new_length
            else:
                G_simple.add_edge(u, v, **data)

    # Find the top K shortest paths
    K = 5
    try:
        # nx.shortest_simple_paths returns a generator of paths sorted by length
        paths_generator = nx.shortest_simple_paths(
            G_simple, 
            source=start_node, 
            target=goal_node, 
            weight='length'
        )
        
        # Take the first K paths
        candidate_paths = []
        for i, path in enumerate(paths_generator):
            if i >= K:
                break
            candidate_paths.append(path)
        
        if not candidate_paths:
            raise RuntimeError(f"No path found from {start_node} to {goal_node}.")
            
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise RuntimeError(f"No path found from {start_node} to {goal_node}.")
    
    # Calculate the length of each path
    path_lengths = []
    for path in candidate_paths:
        total_length = 0
        for i in range(len(path) - 1):
            edge_data = agent.G.get_edge_data(path[i], path[i+1])
            # Handle multi-edge case by taking the first edge's length
            if isinstance(edge_data, dict) and 'length' in edge_data:
                total_length += edge_data['length']
            else:
                # Multi-graph case - take first edge
                total_length += list(edge_data.values())[0]['length']
        path_lengths.append(total_length)
    
    # Convert lengths to probabilities using softmax
    # Negate lengths because shorter = better, and softmax gives higher prob to larger values
    logits = [-length / temperature for length in path_lengths]
    
    # Apply softmax with numerical stability
    stable_logits = np.array(logits) - np.max(logits)
    probabilities = np.exp(stable_logits)
    probabilities /= np.sum(probabilities)
    
    # Sample a path based on the probabilities
    selected_idx = np.random.choice(len(candidate_paths), p=probabilities)
    
    return candidate_paths[selected_idx]


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


def plan_path(agent, current_hour, temperature=100.0, max_attempts=4):
    """
    Plans a path with retry logic for closed goals. If a goal is closed, the agent
    samples another goal from the same category. After multiple failures, there's
    an increasing probability of giving up and returning home.
    
    Args:
        agent (Agent): The agent for whom to plan.
        current_hour (int): The current hour (0-23).
        temperature (float): Temperature for stochastic path selection.
        max_attempts (int): Maximum number of goal attempts before forcing home.
        
    Returns:
        dict: Contains:
            - 'path': List of (node_id, goal_node) tuples showing the full journey
            - 'goal_node': The final goal node
            - 'start_node': The starting node
            - 'attempts': Number of goal attempts made
            - 'returned_home': Boolean indicating if agent gave up and went home
            - 'attempted_goals': List of all goal nodes attempted
    """
    # Sample initial goal and category
    result = sample_goal_node(agent, current_hour, return_category=True)
    
    if result is None or result[0] is None:
        raise RuntimeError(f"Could not sample initial goal node for agent {agent.id}")
    
    goal_node, sampled_category = result
    
    # Sample start node
    start_node = sample_start_node(agent, goal_node, current_hour=current_hour)
    
    # Track state across attempts
    current_start = start_node
    attempted_goals = []
    all_segments = []  # List of (path, goal) tuples
    
    for attempt_num in range(1, max_attempts + 1):
        attempted_goals.append(goal_node)
        
        # Plan path to current goal
        try:
            path_segment = _sample_stochastic_path(agent, current_start, goal_node, temperature)
        except RuntimeError as e:
            # Can't find path to this goal, try another
            if attempt_num < max_attempts:
                goal_node = sample_goal_node(agent, current_hour, category=sampled_category, 
                                            exclude_nodes=set(attempted_goals))
                if goal_node is None:
                    break  # No more goals available in this category
                continue
            else:
                break  # Give up after max attempts
        
        # Check if goal is open
        if not _is_node_open(agent, goal_node, current_hour):
            # Goal is closed! Store this segment but try again
            all_segments.append((path_segment, goal_node))
            
            # Calculate probability of giving up and going home
            # Exponential increase: 0.2, 0.5, 0.8, 0.95 for attempts 1,2,3,4
            give_up_prob = 1 - (0.8 ** attempt_num)
            
            if attempt_num >= max_attempts or np.random.random() < give_up_prob:
                # Give up and go home
                home_node = _sample_home_node(agent)
                current_start = path_segment[-1]  # Start from where we left off
                
                try:
                    home_path = _sample_stochastic_path(agent, current_start, home_node, temperature)
                    all_segments.append((home_path, home_node))
                    
                    # Combine all segments
                    full_path = _combine_path_segments(all_segments)
                    
                    return {
                        'path': full_path,
                        'goal_node': home_node,
                        'start_node': start_node,
                        'attempts': attempt_num + 1,  # +1 for home attempt
                        'returned_home': True,
                        'attempted_goals': attempted_goals
                    }
                except RuntimeError:
                    # Can't even get home, just return what we have
                    full_path = _combine_path_segments(all_segments)
                    return {
                        'path': full_path,
                        'goal_node': goal_node,
                        'start_node': start_node,
                        'attempts': attempt_num,
                        'returned_home': False,
                        'attempted_goals': attempted_goals
                    }
            
            # Try another goal from same category
            current_start = path_segment[-1]  # Continue from current location
            new_goal = sample_goal_node(agent, current_hour, category=sampled_category,
                                       exclude_nodes=set(attempted_goals))
            
            if new_goal is None:
                # No more goals in category, go home
                home_node = _sample_home_node(agent)
                try:
                    home_path = _sample_stochastic_path(agent, current_start, home_node, temperature)
                    all_segments.append((home_path, home_node))
                except RuntimeError:
                    pass
                
                full_path = _combine_path_segments(all_segments)
                return {
                    'path': full_path,
                    'goal_node': attempted_goals[-1],
                    'start_node': start_node,
                    'attempts': attempt_num,
                    'returned_home': True,
                    'attempted_goals': attempted_goals
                }
            
            goal_node = new_goal
            
        else:
            # Goal is open! Success!
            all_segments.append((path_segment, goal_node))
            full_path = _combine_path_segments(all_segments)
            
            return {
                'path': full_path,
                'goal_node': goal_node,
                'start_node': start_node,
                'attempts': attempt_num,
                'returned_home': False,
                'attempted_goals': attempted_goals
            }
    
    # Fallback: return what we have if we exhausted attempts
    if all_segments:
        full_path = _combine_path_segments(all_segments)
    else:
        # No segments at all, create trivial path
        full_path = [(start_node, goal_node)]
    
    return {
        'path': full_path,
        'goal_node': attempted_goals[-1] if attempted_goals else goal_node,
        'start_node': start_node,
        'attempts': len(attempted_goals),
        'returned_home': False,
        'attempted_goals': attempted_goals
    }


def _sample_home_node(agent):
    """
    Sample the agent's home node (highest probability node in home category).
    
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








