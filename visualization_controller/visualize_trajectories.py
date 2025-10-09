import os
import sys
import json
import argparse
import networkx as nx
import random
import folium
from folium import PolyLine

# Add project root for import resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graph_controller.world_graph import WorldGraph


def plot_graph_with_multiple_trajectories(G, trajectories, save_path=None, show_only_pois=True):
    m = folium.Map(location=[32.8801, -117.2340], zoom_start=16)

    # 1. Plot POI nodes
    for node_id, data in G.nodes(data=True):
        category = data.get("Category")
        if category == "None" or category is None:
            continue

        if show_only_pois and "poi_names" not in data:
            continue

        lat = data["y"]
        lon = data["x"]
        poi_names = data.get("poi_names", "—")
        poi_types = data.get("poi_types", "—")
        opening_hours = data.get("opening_hours")
        hours_str = f"{opening_hours['open']:02d}:00–{opening_hours['close']:02d}:00" if isinstance(opening_hours, dict) else "—"

        popup = folium.Popup(
            html=f"<b>Node:</b> {node_id}<br>"
                 f"<b>Category:</b> {category}<br>"
                 f"<b>Opening Hours:</b> {hours_str}<br>"
                 f"<b>POI Names:</b> {poi_names}<br>"
                 f"<b>POI Types:</b> {poi_types}",
            max_width=350
        )

        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m)

    # 2. Overlay multiple trajectories
    colors = ["red", "green", "purple", "orange", "darkred", "cadetblue", "black"]
    color_cycle = (c for c in colors * 100)  # Just reuse colors if n > len(colors)

    for agent_id, agent_trajs in trajectories.items():
        for traj in agent_trajs:
            coords = [
                (G.nodes[n]["y"], G.nodes[n]["x"])
                for n in traj["path"]
                if n in G.nodes
            ]
            if not coords:
                continue

            color = next(color_cycle)
            PolyLine(
                coords,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f"{agent_id} @ {traj['hour']:02d}:00"
            ).add_to(m)

            folium.Marker(
                coords[0],
                icon=folium.Icon(color="green", icon="play"),
                popup=f"{agent_id} Start"
            ).add_to(m)

            folium.Marker(
                coords[-1],
                icon=folium.Icon(color="red", icon="flag"),
                popup=f"{agent_id} Goal"
            ).add_to(m)

    if save_path:
        m.save(save_path)
        print(f"✅ Map saved to {save_path}")

    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize N trajectories from a simulation run."
    )
    parser.add_argument(
        "-x", "--run_id", type=int, required=True,
        help="Run ID to load data from: data/simulation_data/run_<X>/"
    )
    parser.add_argument(
        "-n", "--num_show", type=int, default=5,
        help="Number of trajectories to visualize (default=5)"
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

    # Randomly select a few trajectories across agents
    selected = {}
    all_agent_ids = list(all_trajectories.keys())
    random.shuffle(all_agent_ids)

    count = 0
    for agent_id in all_agent_ids:
        if count >= num_to_show:
            break
        agent_trajs = all_trajectories[agent_id]
        if agent_trajs:
            selected[agent_id] = [random.choice(agent_trajs)]
            count += 1

    # Plot the selected trajectories
    output_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"visualization_{num_to_show}_trajectories.html")
    plot_graph_with_multiple_trajectories(G, selected, save_path=output_path)


if __name__ == "__main__":
    main()