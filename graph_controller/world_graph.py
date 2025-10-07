import networkx as nx

class WorldGraph:
    def __init__(self, G, relevant_categories=None):
        """
        Args:
            G (networkx.Graph): The full map graph.
            relevant_categories (list of str): Which POI categories to track. If None, uses default set.
        """
        self.G = G
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