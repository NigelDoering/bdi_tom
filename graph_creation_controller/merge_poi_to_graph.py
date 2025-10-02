import os
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# Input files
GRAPH_IN_PATH = "data/raw/ucsd_walk.graphml"
POI_IN_PATH = "data/raw/ucsd_named_buildings.geojson"

# Output file
GRAPH_OUT_PATH = "data/processed/ucsd_walk_semantic.graphml"

def load_data():
    print("Loading UCSD pedestrian graph...")
    G = ox.load_graphml(GRAPH_IN_PATH)
    
    print("Loading named POIs...")
    pois = gpd.read_file(POI_IN_PATH)
    pois = pois[pois['geometry'].notnull()].copy()

    # Drop duplicates if needed
    pois = pois.drop_duplicates(subset=["name", "geometry"])

    return G, pois

def map_pois_to_nodes(G, pois):
    print("Mapping POIs to nearest graph nodes...")

    # Project to UTM for accurate centroid computation
    pois = pois.to_crs(epsg=32611)  # UTM Zone 11N
    pois["geometry"] = pois["geometry"].centroid
    pois = pois.to_crs(epsg=4326)   # Back to lat/lon to match graph

    # Map POIs to nearest nodes
    pois["nearest_node"] = ox.distance.nearest_nodes(
        G,
        X=pois.geometry.x,
        Y=pois.geometry.y,
        return_dist=False
    )
    return pois

def annotate_graph_nodes(G, pois):
    print("Annotating graph nodes with POI information...")

    for _, row in pois.iterrows():
        node = row["nearest_node"]
        if "poi_names" not in G.nodes[node]:
            G.nodes[node]["poi_names"] = []
            G.nodes[node]["poi_types"] = []

        # Store both name and type
        G.nodes[node]["poi_names"].append(row.get("name", ""))
        G.nodes[node]["poi_types"].append(row.get("amenity") or row.get("building") or "unknown")

def main():
    if os.path.exists(GRAPH_OUT_PATH):
        print(f"Semantic graph already exists at {GRAPH_OUT_PATH}. Skipping.")
        return

    G, pois = load_data()
    pois = map_pois_to_nodes(G, pois)
    annotate_graph_nodes(G, pois)

    # Create output directory if needed
    os.makedirs(os.path.dirname(GRAPH_OUT_PATH), exist_ok=True)

    print(f"Saving annotated graph to {GRAPH_OUT_PATH}...")
    ox.save_graphml(G, GRAPH_OUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()