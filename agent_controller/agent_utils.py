import random

def sample_preferences(categories=None):
    """
    Sample Dirichlet-distributed preferences over categories.
    """
    if categories is None:
        categories = ["food", "study", "health", "home", "errands", "leisure"]

    weights = [random.expovariate(1.0) for _ in categories]
    total = sum(weights)
    norm_weights = {cat: round(w / total, 3) for cat, w in zip(categories, weights)}
    return norm_weights