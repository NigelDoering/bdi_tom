# visualization_controller/map_plotter.py

import osmnx as ox
import folium
import matplotlib.pyplot as plt

def plot_static_map(
    graph,
    save_path=None,
    show=True,
    title="UCSD Pedestrian Map",
    bgcolor="#f7f7f7",
    node_color="#333333",
    edge_color="#999999",
    node_size=10,
    edge_linewidth=0.5,
    dpi=200
):
    fig, ax = ox.plot_graph(
        graph,
        bgcolor=bgcolor,
        node_color=node_color,
        edge_color=edge_color,
        node_size=node_size,
        edge_linewidth=edge_linewidth,
        show=False,
        close=False,
        dpi=dpi
    )

    ax.set_title(title, fontsize=14)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Map saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()



import folium

def plot_interactive_graph_nodes(G, save_path=None, show_only_pois=True):
    """
    Creates an interactive map showing graph nodes with optional POI annotations and opening hours.
    Only nodes with non-None 'Category' values are plotted.

    Args:
        G: The OSMnx graph.
        save_path: Optional path to save the HTML file.
        show_only_pois: If True, only nodes with POI info will be shown.
    """
    # Keep the original map style and fixed center
    m = folium.Map(location=[32.8801, -117.2340], zoom_start=16)

    for node_id, data in G.nodes(data=True):
        category = data.get("Category")
        if category == "None" or category is None:
            continue  # Skip nodes with no category

        if show_only_pois and "poi_names" not in data:
            continue

        lat = data["y"]
        lon = data["x"]
        poi_names = data.get("poi_names", "—")
        poi_types = data.get("poi_types", "—")

        # Format opening hours if available
        opening_hours = data.get("opening_hours")
        if isinstance(opening_hours, dict):
            hours_str = f"{opening_hours['open']:02d}:00–{opening_hours['close']:02d}:00"
        else:
            hours_str = "—"

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

    if save_path:
        m.save(save_path)
        print(f"Interactive map saved to {save_path}")

    return m