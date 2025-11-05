"""Visualize trajectories with multiple goal attempts on the UCSD campus map."""

import os
import sys
import json
import random
import argparse
import networkx as nx
import folium
from folium import PolyLine

# Add project root for import resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graph_controller.world_graph import WorldGraph


def visualize_multi_goal_trajectories(G, trajectories, save_path=None, show_pois=False):
    """
    Plot trajectories with attempted goal markers.
    
    Args:
        G: NetworkX graph with node coordinates
        trajectories: List of (agent_id, trajectory_dict) tuples or dict of {agent_id: [trajectory_dicts]}
        save_path: Optional path to save HTML output
        show_pois: Whether to show POI markers on the map
        
    Returns:
        folium.Map object
    """
    m = folium.Map(location=[32.8801, -117.2340], zoom_start=16)

    # 1. Optionally plot POI nodes
    if show_pois:
        for node_id, data in G.nodes(data=True):
            category = data.get("Category")
            if category == "None" or category is None:
                continue

            if "poi_names" not in data:
                continue

            lat = data.get("y")
            lon = data.get("x")
            if lat is None or lon is None:
                continue

            poi_names = data.get("poi_names", "—")
            opening_hours = data.get("opening_hours")
            hours_str = f"{opening_hours['open']:02d}:00–{opening_hours['close']:02d}:00" if isinstance(opening_hours, dict) else "—"

            popup = folium.Popup(
                html=f"<b>Node:</b> {node_id}<br>"
                     f"<b>Category:</b> {category}<br>"
                     f"<b>Opening Hours:</b> {hours_str}<br>"
                     f"<b>POI Names:</b> {poi_names}",
                max_width=350
            )

            folium.CircleMarker(
                location=(lat, lon),
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.6,
                popup=popup
            ).add_to(m)

    # 2. Normalize trajectories input to list of (agent_id, traj) tuples
    if isinstance(trajectories, dict):
        traj_list = []
        for agent_id, agent_trajs in trajectories.items():
            for traj in agent_trajs:
                traj_list.append((agent_id, traj))
    elif isinstance(trajectories, list):
        traj_list = trajectories
    else:
        raise ValueError("trajectories must be a dict or list of (agent_id, traj) tuples")

    # 3. Plot each trajectory
    colors = ["red", "green", "purple", "orange", "darkred", "cadetblue", "black", "darkgreen"]
    color_idx = 0

    for agent_id, traj in traj_list:
        # Extract path coordinates
        coords = []
        for entry in traj.get("path", []):
            # Handle both annotated (node, goal) tuples and plain node IDs
            node_id = entry[0] if isinstance(entry, (list, tuple)) else entry
            if node_id in G.nodes:
                node_data = G.nodes[node_id]
                lat = node_data.get("y")
                lon = node_data.get("x")
                if lat is not None and lon is not None:
                    coords.append((lat, lon))

        if not coords:
            continue

        # Draw path line
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        hour = traj.get("hour", "?")
        attempts = traj.get("attempts", 1)  # Fixed: was "planning_attempts"
        tooltip = f"{agent_id} @ hour {hour} | {attempts} attempts"
        
        PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.8,
            tooltip=tooltip
        ).add_to(m)

        # Mark start node
        start_hour = traj.get("hour", "?")
        folium.Marker(
            coords[0],
            icon=folium.Icon(color="green", icon="play"),
            popup=f"{agent_id} Start",
            tooltip=f"{agent_id} Start @ {start_hour}:00"
        ).add_to(m)

        # Mark attempted goals along the path
        # Note: The final goal will be marked here too, so no need for duplicate marking
        attempted_goals = traj.get("attempted_goals", [])
        final_goal = traj.get("goal_node")
        
        # Always show attempted goals if they exist (even if just 1)
        if attempted_goals:
            for idx, goal_node in enumerate(attempted_goals, start=1):
                if goal_node not in G.nodes:
                    continue
                    
                goal_data = G.nodes[goal_node]
                goal_lat = goal_data.get("y")
                goal_lon = goal_data.get("x")
                if goal_lat is None or goal_lon is None:
                    continue

                # Get POI information
                poi_names = goal_data.get("poi_names", "Unknown POI")
                category = goal_data.get("Category", "Unknown")
                opening_hours = goal_data.get("opening_hours")
                hours_str = f"{opening_hours['open']:02d}:00–{opening_hours['close']:02d}:00" if isinstance(opening_hours, dict) else "—"

                # Determine if this was the successful goal or a failed attempt
                is_final = (goal_node == final_goal)
                
                # Use different colors: orange for failed attempts, red for final goal
                if is_final:
                    icon_color = "red"
                    icon_name = "flag"
                    status = "Final Goal (Reached)"
                else:
                    icon_color = "orange"
                    icon_name = "times"  # X mark - better than "remove"
                    status = "Failed Attempt (Closed)"

                popup_html = (
                    f"<b>{agent_id} - Attempt {idx}</b><br>"
                    f"<b>Status:</b> {status}<br>"
                    f"<b>POI:</b> {poi_names}<br>"
                    f"<b>Category:</b> {category}<br>"
                    f"<b>Opening Hours:</b> {hours_str}<br>"
                    f"<b>Hour:</b> {traj.get('hour', '?')}:00<br>"
                    f"<b>Node ID:</b> {goal_node}"
                )
                
                # Create tooltip with hour and category
                hour_display = traj.get('hour', '?')
                tooltip_text = f"Attempt {idx} @ {hour_display}:00 | {category} | {poi_names}"

                folium.Marker(
                    (goal_lat, goal_lon),
                    icon=folium.Icon(color=icon_color, icon=icon_name),
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=tooltip_text
                ).add_to(m)

    if save_path:
        m.save(save_path)
        print(f"✅ Map saved to {save_path}")

    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize trajectories with multiple goal attempts from a simulation run."
    )
    parser.add_argument(
        "-x", "--run_id", type=int, required=True,
        help="Run ID to load data from: data/simulation_data/run_<X>/"
    )
    parser.add_argument(
        "-n", "--num_show", type=int, default=5,
        help="Number of multi-goal trajectories to visualize (default=5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id
    num_to_show = args.num_show

    base_dir = os.path.join("data", "simulation_data", f"run_{run_id}")
    traj_path = os.path.join(base_dir, "trajectories", "all_trajectories.json")
    graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")

    # Load graph
    print("📡 Loading graph...")
    G = nx.read_graphml(graph_path)
    G = WorldGraph(G).G  # Ensure clean data

    # Load trajectory data
    print("📡 Loading trajectories...")
    if not os.path.exists(traj_path):
        print(f"❌ Could not find: {traj_path}")
        return

    with open(traj_path, "r") as f:
        all_trajectories = json.load(f)

    # Filter for trajectories with multiple goal attempts
    print("🔍 Finding trajectories with multiple goal attempts...")
    multi_goal_trajectories = []
    for agent_id, agent_trajs in all_trajectories.items():
        for traj in agent_trajs:
            attempted_goals = traj.get("attempted_goals", [])
            attempts = traj.get("attempts", 1)  # Fixed: was "planning_attempts"
            
            # Include if multiple attempts or multiple attempted goals
            if len(attempted_goals) > 1 or attempts > 1:
                multi_goal_trajectories.append((agent_id, traj))

    if not multi_goal_trajectories:
        print("❌ No trajectories with multiple goal attempts found in this run.")
        return

    print(f"✅ Found {len(multi_goal_trajectories)} trajectories with multiple goal attempts")

    # Select subset to display (randomly sample)
    if len(multi_goal_trajectories) <= num_to_show:
        selected = multi_goal_trajectories
    else:
        selected = random.sample(multi_goal_trajectories, num_to_show)
    print(f"📊 Visualizing {len(selected)} randomly selected trajectories...")

    # Plot the selected trajectories
    output_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"multi_goal_attempts_{len(selected)}_trajectories.html")
    
    visualize_multi_goal_trajectories(G, selected, save_path=output_path, show_pois=False)
    print(f"✅ Visualization complete! Open: {output_path}")


if __name__ == "__main__":
    main()
