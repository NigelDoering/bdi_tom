import networkx as nx
import ast

class WorldGraph:
    def __init__(self, G, relevant_categories=None):
        """
        Args:
            G (networkx.Graph): The full map graph.
            relevant_categories (list of str): Which POI categories to track. If None, uses default set.
        """
        self.G = G

        # Ensure edge lengths are numeric, as they can be loaded as strings from GraphML
        for u, v, data in self.G.edges(data=True):
            if 'length' in data and isinstance(data['length'], str):
                try:
                    data['length'] = float(data['length'])
                except (ValueError, TypeError):
                    # This is a fallback, but we should investigate if it happens
                    print(f"Warning: Could not convert edge length '{data['length']}' to float for edge ({u}, {v}).")

        # Also ensure node coordinates are numeric for distance calculations
        for node, data in self.G.nodes(data=True):
            for attr in ['x', 'y']:
                if attr in data and isinstance(data[attr], str):
                    try:
                        data[attr] = float(data[attr])
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert node attribute '{attr}' with value '{data[attr]}' to float for node {node}.")
            
            # Convert opening_hours from string to dict
            if 'opening_hours' in data and isinstance(data['opening_hours'], str):
                try:
                    data['opening_hours'] = ast.literal_eval(data['opening_hours'])
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse opening_hours string '{data['opening_hours']}' for node {node}.")

        self.relevant_categories = relevant_categories or [
            "home", "study", "food", "leisure", "errands", "health"
        ]

        # Precompute list of node IDs that are in any relevant category
        self.poi_nodes = self._extract_relevant_nodes()

    def _extract_relevant_nodes(self):
        relevant_nodes = [
            node for node, data in self.G.nodes(data=True)
            if data.get("Category") in self.relevant_categories
        ]
        return relevant_nodes

    def get_node_data(self, node_id):
        return self.G.nodes[node_id]