import os
import osmnx as ox
import geopandas as gpd

ox.settings.use_cache = False

# Output paths
GRAPH_PATH = "data/raw/ucsd_walk.graphml"
POI_PATH = "data/raw/ucsd_named_buildings.geojson"

# Place query
PLACE_NAME = "University of California San Diego, La Jolla, California, USA"

def download_graph():
    if os.path.exists(GRAPH_PATH):
        print(f"Graph already exists at {GRAPH_PATH}. Skipping graph download.")
    else:
        print("Downloading UCSD pedestrian graph from OpenStreetMap...")
        G = ox.graph_from_place(PLACE_NAME, network_type="walk")
        print(f"Saving graph to {GRAPH_PATH}...")
        ox.save_graphml(G, GRAPH_PATH)
        print("Graph download complete.")

def download_named_buildings():
    if os.path.exists(POI_PATH):
        print(f"POI data already exists at {POI_PATH}. Skipping POI download.")
    else:
        print("Downloading named buildings and amenities from OpenStreetMap...")
        tags = {
            "building": True,
            "amenity": True,
            "name": True
        }
        gdf = ox.features_from_place(PLACE_NAME, tags=tags) # type: ignore
        named = gdf[gdf["name"].notnull()][["name", "amenity", "building", "geometry"]]
        print(f"Saving {len(named)} named buildings/amenities to {POI_PATH}...")
        named.to_file(POI_PATH, driver="GeoJSON")
        print("POI download complete.")

def main():
    download_graph()
    download_named_buildings()

if __name__ == "__main__":
    main()