import numpy as np
import networkx as nx

def sample_goal_node(agent, current_hour):
    """
    Sample a goal node for the agent based on its preferences and current beliefs.

    Args:
        agent (Agent): The agent for whom to sample a goal node.
        current_hour (int): The current hour of the day (0-23).
    Returns:
        node_id (str): The sampled goal node ID.
    """

    category_probs = agent.category_preferences

    # Extract category names and their corresponding probabilities
    categories = list(category_probs.keys())
    probabilities = list(category_probs.values())
    probabilities = probabilities / np.array(probabilities).sum()  # Normalize to ensure sum=1

    # Sample a single category based on the agent's high-level preferences
    sampled_category = np.random.choice(categories, p=probabilities)

    # Retrieve the node-level preferences for the chosen category
    node_preferences = getattr(agent, f"{sampled_category}_preferences")

    # Get the temporal belief for each candidate node at the current hour
    belief_probs = {
        node_id: agent.belief_state[node_id]['temporal_belief'][current_hour]
        for node_id in node_preferences.keys()
    }

        # Combine preferences and beliefs
    combined_weights = {}
    for node_id in node_preferences:
        pref = node_preferences[node_id]
        belief = belief_probs[node_id]
        combined_weights[node_id] = pref * belief

    # Normalize
    total = sum(combined_weights.values())
    if total == 0:
        return None  # fallback needed
    normalized_weights = {node: w / total for node, w in combined_weights.items()}

    # Sample
    node_ids = list(normalized_weights.keys())
    probs = list(normalized_weights.values())
    sampled_node = np.random.choice(node_ids, p=probs)
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
        category_probs = agent.category_preferences
        
        # Extract category names and their corresponding probabilities
        categories = list(category_probs.keys())
        probabilities = list(category_probs.values())
        probabilities = probabilities / np.array(probabilities).sum()
        
        # Sample a category based on the agent's high-level preferences
        sampled_category = np.random.choice(categories, p=probabilities)
        
        # Retrieve the node-level preferences for the chosen category
        node_preferences = getattr(agent, f"{sampled_category}_preferences")
        
        # Get the temporal belief for each candidate node at the START hour
        belief_probs = {
            node_id: agent.belief_state[node_id]['temporal_belief'][start_hour]
            for node_id in node_preferences.keys()
        }
        
        # Combine preferences and beliefs
        combined_weights = {}
        for node_id in node_preferences:
            pref = node_preferences[node_id]
            belief = belief_probs[node_id]
            combined_weights[node_id] = pref * belief
        
        # Normalize
        total = sum(combined_weights.values())
        if total == 0:
            continue  # Try again if all weights are zero
        
        normalized_weights = {node: w / total for node, w in combined_weights.items()}
        
        # Sample a candidate start node
        node_ids = list(normalized_weights.keys())
        probs = list(normalized_weights.values())
        candidate_start = np.random.choice(node_ids, p=probs)
        
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

def plan_stochastic_path(agent, start_node, goal_node, temperature=100.0):
    """
    Plans a path from start to goal by sampling from the top K shortest paths.

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
        max_path_len (int): Unused in this implementation but kept for API compatibility.

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








