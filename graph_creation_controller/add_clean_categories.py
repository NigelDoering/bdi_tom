import os
import osmnx as ox

# Input and output paths
GRAPH_IN_PATH = "data/processed/ucsd_walk_semantic.graphml"
GRAPH_OUT_PATH = "data/processed/ucsd_walk_labeled.graphml"

# Mapping from raw POI types to cleaned simulation categories
CATEGORY_MAP = {
    # Food
    "restaurant": "food",
    "cafe": "food",
    "fast_food": "food",
    "pub": "food",
    "bar": "food",
    "internet_cafe": "food",

    # Study
    "university": "study",
    "library": "study",
    "college": "study",
    "research_institute": "study",

    # Home
    "apartments": "home",
    "dormitory": "home",
    "residential": "home",
    "student_accommodation": "home",

    # Health
    "hospital": "health",
    "clinic": "health",
    "pharmacy": "health",
    "doctors": "health",

    # Errands
    "atm": "errands",
    "bank": "errands",
    "post_office": "errands",
    "parcel_locker": "errands",
    "charging_station": "errands",

    # Leisure
    "theatre": "leisure",
    "cinema": "leisure",
    "arts_centre": "leisure",
}

def assign_category(poi_types):
    if not poi_types:
        return None
    for raw_type in poi_types:
        if raw_type in CATEGORY_MAP:
            return CATEGORY_MAP[raw_type]
    return None

def main():
    if os.path.exists(GRAPH_OUT_PATH):
        print(f"Labeled graph already exists at {GRAPH_OUT_PATH}. Skipping.")
        return

    print("Loading semantic graph...")
    G = ox.load_graphml(GRAPH_IN_PATH)

    print("Assigning cleaned Category labels to nodes...")
    for node, data in G.nodes(data=True):
        poi_types = data.get("poi_types")
        category = assign_category(poi_types)
        G.nodes[node]["Category"] = category

    print(f"Saving updated graph to {GRAPH_OUT_PATH}...")
    os.makedirs(os.path.dirname(GRAPH_OUT_PATH), exist_ok=True)
    ox.save_graphml(G, GRAPH_OUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()