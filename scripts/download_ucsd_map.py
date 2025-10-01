import os
import osmnx as ox

ox.settings.use_cache = False

GRAPH_PATH = "data/raw/ucsd_walk.graphml"

def main():
    if os.path.exists(GRAPH_PATH):
        print(f"Graph already exists at {GRAPH_PATH}. Skipping download.")
        return

    print("Downloading UCSD pedestrian graph from OpenStreetMap...")
    G = ox.graph_from_place(
        "University of California San Diego, La Jolla, California, USA",
        network_type='walk'
    )

    print(f"Saving graph to {GRAPH_PATH}...")
    ox.save_graphml(G, GRAPH_PATH)
    print("Done.")

if __name__ == "__main__":
    main()