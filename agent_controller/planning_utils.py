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

def sample_start_node(agent, goal_node, min_dist=500, max_tries=100):
    """
    Samples a random start node from the agent's graph that is not too close
    to the specified goal node.

    Args:
        agent (Agent): The agent, which holds the graph `G`.
        goal_node (str): The ID of the goal node.
        min_dist (int | float): The minimum shortest path distance required
            between the start and goal nodes. The distance is based on the
            'length' attribute of the graph edges.
        max_tries (int): The maximum number of attempts to find a suitable node.

    Returns:
        str: The ID of a suitable start node.

    Raises:
        RuntimeError: If a suitable start node cannot be found within `max_tries`.
    """
    nodes = list(agent.G.nodes)
    for _ in range(max_tries):
        # Randomly select a potential start node
        candidate_start = np.random.choice(nodes)

        # Ensure the start and goal nodes are not the same
        if candidate_start == goal_node:
            continue


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

    
    # If we exit the loop, we failed to find a suitable node
    raise RuntimeError(
        f"Could not find a start node at least {min_dist}m away from "
        f"{goal_node} within {max_tries} attempts."
    )

def plan_stochastic_path(agent, start_node, goal_node, temperature=100.0, max_path_len=500):
    """
    Plans a path from start to goal with some stochasticity.

    The pathfinding tends to follow the shortest path but can make non-optimal
    choices. The degree of randomness is controlled by the `temperature` param.

    Args:
        agent (Agent): The agent, which holds the graph `G`.
        start_node (str): The ID of the starting node.
        goal_node (str): The ID of the goal node.
        temperature (float): Controls the randomness of the path. Higher values
            lead to more random, less optimal paths. Lower values make the
            path closer to the true shortest path. Must be > 0.
        max_path_len (int): A safeguard to prevent infinite loops in case of
            unexpected graph structures or high temperatures.

    Returns:
        list[str]: A list of node IDs representing the path.

    Raises:
        ValueError: If temperature is not positive.
        RuntimeError: If a path cannot be found or if the path exceeds the max length.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    
    if start_node == goal_node:
        return [start_node]

    path = [start_node]
    current_node = start_node

    while current_node != goal_node:
        if len(path) >= max_path_len:
            raise RuntimeError("Path planning exceeded maximum length.")

        neighbors = list(agent.G.neighbors(current_node))
        
        # Avoid immediately going back to the previous node to prevent simple loops,
        # but only if there are other options available.
        if len(path) > 1:
            previous_node = path[-2]
            if previous_node in neighbors and len(neighbors) > 1:
                neighbors.remove(previous_node)

        if not neighbors:
            raise RuntimeError(f"Node {current_node} has no forward neighbors.")

        # Calculate the "goodness" of each neighbor based on its distance to the goal
        distances = {}
        for neighbor in neighbors:
            try:
                dist = nx.shortest_path_length(
                    agent.G, source=neighbor, target=goal_node, weight='length'
                )
                distances[neighbor] = dist
            except nx.NetworkXNoPath:
                # This neighbor doesn't connect to the goal, so ignore it
                continue
        
        if not distances:
            raise RuntimeError(f"No neighbors of {current_node} connect to the goal.")

        # Convert distances to probabilities using a softmax function
        # Lower distance = higher probability
        node_list = list(distances.keys())
        
        # The negative inverse of distance, scaled by temperature
        # We use negative because softmax gives higher probability to higher values
        logits = [-distances[node] / temperature for node in node_list]
        
        # Subtract max for numerical stability before applying exp
        stable_logits = np.array(logits) - np.max(logits)
        probabilities = np.exp(stable_logits)
        probabilities /= np.sum(probabilities)

        # Sample the next node based on the calculated probabilities
        next_node = np.random.choice(node_list, p=probabilities)
        path.append(next_node)
        current_node = next_node

    return path








