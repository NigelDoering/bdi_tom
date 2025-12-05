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


def extract_node_id(entry):
    if isinstance(entry, (list, tuple)):
        return entry[0]
    return entry


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
            # Extract node IDs from path (handling tuple format: [(node_id, goal_id), ...])
            path_nodes = []
            for entry in traj["path"]:
                node_id = extract_node_id(entry)
                path_nodes.append(node_id)
            
            coords = [
                (G.nodes[n]["y"], G.nodes[n]["x"])
                for n in path_nodes
                if n in G.nodes
            ]
            if not coords:
                continue

            color = next(color_cycle)
            tooltip_bits = [f"{agent_id} @ {traj['hour']:02d}:00"]
            preferred_goal = traj.get("goal_node")
            if preferred_goal:
                tooltip_bits.append(f"goal={preferred_goal}")
            distractor_goal = traj.get("distractor_goal")
            if distractor_goal:
                tooltip_bits.append(f"distractor={distractor_goal}")
            episode_idx = traj.get("episode_idx")
            if episode_idx is not None:
                tooltip_bits.append(f"episode={episode_idx}")

            PolyLine(
                coords,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=" | ".join(tooltip_bits)
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

            if distractor_goal and distractor_goal in G.nodes:
                distractor_coords = (G.nodes[distractor_goal]["y"], G.nodes[distractor_goal]["x"])
                folium.Marker(
                    distractor_coords,
                    icon=folium.Icon(color="blue", icon="info-sign"),
                    popup=f"{agent_id} Distractor"
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
    parser.add_argument(
        "-d", "--dataset", choices=["baseline", "distractor"], default="baseline",
        help="Dataset to visualize: baseline trajectories or Experiment 2 distractors"
    )
    parser.add_argument(
        "--seed", type=int, default=13,
        help="Random seed used when sampling trajectories"
    )
    parser.add_argument(
        "--agent-ids", type=str, default=None,
        help="Optional comma-separated list of agent IDs to include"
    )
    parser.add_argument(
        "--episode-indices", type=str, default=None,
        help=(
            "Episode selectors: for distractor data, provide comma-separated episode indices; "
            "for baseline data, use agent_id:index (e.g., agent_003:7)."
        )
    )
    parser.add_argument(
        "--output-name", type=str, default=None,
        help="Custom suffix for the output HTML filename"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id
    num_to_show = args.num_show

    base_dir = os.path.join("data", "simulation_data", f"run_{run_id}")
    graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")

    # Load graph
    print("📡 Loading graph...")
    G = nx.read_graphml(graph_path)
    G = WorldGraph(G).G  # Ensure clean data

    # Load trajectory data
    print("📡 Loading trajectories...")
    random.seed(args.seed)

    agent_filter = None
    if args.agent_ids:
        agent_filter = {agent.strip() for agent in args.agent_ids.split(",") if agent.strip()}

    distractor_indices = []
    baseline_selectors = []
    if args.episode_indices:
        for token in args.episode_indices.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                agent_part, idx_part = token.split(":", 1)
                agent_part = agent_part.strip()
                idx_part = idx_part.strip()
                if not agent_part or not idx_part:
                    print(f"⚠️ Ignoring invalid selector '{token}'")
                    continue
                try:
                    baseline_selectors.append((agent_part, int(idx_part)))
                except ValueError:
                    print(f"⚠️ Ignoring invalid selector '{token}'")
            else:
                try:
                    distractor_indices.append(int(token))
                except ValueError:
                    print(f"⚠️ Ignoring invalid episode index '{token}'")

    entries = []
    entries_by_agent = {}

    if args.dataset == "baseline":
        traj_path = os.path.join(base_dir, "trajectories", "all_trajectories.json")
        if not os.path.exists(traj_path):
            print(f"❌ Could not find: {traj_path}")
            return
        with open(traj_path, "r", encoding="utf-8") as f:
            all_trajectories = json.load(f)

        for agent_id, trajs in all_trajectories.items():
            if agent_filter and agent_id not in agent_filter:
                continue
            for idx, traj in enumerate(trajs):
                entry = {
                    "agent_id": agent_id,
                    "episode_idx": idx,
                    "path": traj["path"],
                    "goal_node": traj.get("goal_node"),
                    "distractor_goal": traj.get("distractor_goal"),
                    "hour": traj["hour"],
                }
                entries.append(entry)
                entries_by_agent.setdefault(agent_id, []).append(entry)

        if distractor_indices:
            print("⚠️ Episode indices without agent prefix apply only to distractor dataset; ignoring.")

    else:
        distractor_path = os.path.join(base_dir, "exp_2_distractors.json")
        if not os.path.exists(distractor_path):
            print(f"❌ Could not find: {distractor_path}")
            return
        with open(distractor_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        episodes = payload.get("episodes", [])

        for idx, episode in enumerate(episodes):
            agent_id = episode["agent_id"]
            if agent_filter and agent_id not in agent_filter:
                continue
            entry = {
                "agent_id": agent_id,
                "episode_idx": idx,
                "path": episode["path"],
                "goal_node": episode["preferred_goal"],
                "distractor_goal": episode.get("distractor_goal"),
                "hour": episode["observation_hour"],
                "min_distance_to_distractor": episode.get("min_distance_to_distractor"),
            }
            entries.append(entry)
            entries_by_agent.setdefault(agent_id, []).append(entry)

    if not entries:
        print("❌ No trajectories found with the provided filters.")
        return

    selected_entries = []
    if args.dataset == "distractor" and distractor_indices:
        index_map = {entry["episode_idx"]: entry for entry in entries}
        for idx in distractor_indices:
            entry = index_map.get(idx)
            if entry is None:
                print(f"⚠️ Episode index {idx} not found in filtered trajectories.")
                continue
            selected_entries.append(entry)
        if not selected_entries:
            print("❌ No valid episode indices remained after filtering.")
            return
    elif args.dataset == "baseline" and baseline_selectors:
        for agent_id, idx in baseline_selectors:
            agent_entries = entries_by_agent.get(agent_id)
            if agent_entries is None:
                print(f"⚠️ Agent '{agent_id}' not present in filtered trajectories.")
                continue
            if idx < 0 or idx >= len(agent_entries):
                print(f"⚠️ Episode index {idx} out of range for agent '{agent_id}'.")
                continue
            selected_entries.append(agent_entries[idx])
        if not selected_entries:
            print("❌ No valid baseline selectors remained after filtering.")
            return
    else:
        random.shuffle(entries)
        selected_entries = entries[: min(num_to_show, len(entries))]

    selected = {}
    for chosen in selected_entries:
        agent_id = chosen["agent_id"]

        # Validate that every node in the path exists in the graph
        missing_nodes = []
        for step in chosen["path"]:
            node_id = extract_node_id(step)
            if node_id not in G.nodes:
                missing_nodes.append(node_id)
        if missing_nodes:
            preview = ", ".join(missing_nodes[:5])
            print(f"⚠️ Missing nodes for {agent_id}: {preview} (showing up to 5)")

        selected.setdefault(agent_id, []).append(chosen)

    # Plot the selected trajectories
    output_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_tag = "distractor" if args.dataset == "distractor" else "baseline"

    if args.output_name:
        suffix = args.output_name
    elif args.dataset == "distractor" and distractor_indices:
        suffix = f"{output_tag}_episodes_{'-'.join(str(idx) for idx in distractor_indices)}"
    elif args.dataset == "baseline" and baseline_selectors:
        compact = [f"{agent.replace(' ', '_')}-{idx}" for agent, idx in baseline_selectors]
        suffix = f"{output_tag}_episodes_{'-'.join(compact)}"
    else:
        suffix = f"{output_tag}_{len(selected_entries)}_trajectories"

    output_path = os.path.join(output_dir, f"visualization_{suffix}.html")
    plot_graph_with_multiple_trajectories(G, selected, save_path=output_path)


if __name__ == "__main__":
    main()