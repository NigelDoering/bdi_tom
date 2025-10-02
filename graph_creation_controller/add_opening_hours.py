import os
import random
import osmnx as ox

GRAPH_IN = "data/processed/ucsd_walk_labeled.graphml"
GRAPH_OUT = "data/processed/ucsd_walk_hours.graphml"

def sample_opening_hours(category):
    """
    Returns a dict with {'open': hour_0_24, 'close': hour_0_24} for a given category.
    """
    if category == "food":
        open_time = random.choice([6, 7, 8, 9])
        close_time = random.choice([14, 16, 18, 20])
    elif category == "study":
        open_time = random.choice([8, 9, 10])
        close_time = random.choice([17, 18, 20, 22])
    elif category == "home":
        return {"open": 0, "close": 24}
    elif category == "health":
        open_time = random.choice([8, 9, 10])
        close_time = random.choice([15, 16, 17])
    elif category == "errands":
        open_time = random.choice([9, 10, 11])
        close_time = random.choice([16, 17, 18, 20])
    elif category == "leisure":
        open_time = random.choice([10, 12, 13, 14])
        close_time = random.choice([20, 21, 22])
    else:
        return None

    # Make sure opening < closing
    if close_time <= open_time:
        close_time = open_time + 1

    return {"open": open_time, "close": close_time}

def main():
    if os.path.exists(GRAPH_OUT):
        print(f"Output graph already exists at {GRAPH_OUT}. Skipping.")
        return

    print("Loading graph...")
    G = ox.load_graphml(GRAPH_IN)

    print("Assigning synthetic opening hours to nodes with a Category...")
    updated = 0
    for node, data in G.nodes(data=True):
        category = data.get("Category")
        if category:
            hours = sample_opening_hours(category)
            if hours:
                G.nodes[node]["opening_hours"] = hours
                updated += 1

    print(f"Updated {updated} nodes with opening hours.")
    print(f"Saving updated graph to {GRAPH_OUT}...")
    os.makedirs(os.path.dirname(GRAPH_OUT), exist_ok=True)
    ox.save_graphml(G, GRAPH_OUT)
    print("Done.")

if __name__ == "__main__":
    main()