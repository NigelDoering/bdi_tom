# BDI-ToM UCSD Simulation

This project simulates pedestrian trajectories on the UCSD campus to support research in Theory of Mind (ToM), with a focus on modeling latent beliefs, desires, and intentions from observable behavior in structured environments.

---

## 📍 UCSD Pedestrian Graph

We use OpenStreetMap data to construct a walkable graph of the UCSD campus using [OSMnx](https://github.com/gboeing/osmnx).

### ✅ Preprocessed Graph

The UCSD pedestrian graph has already been downloaded and saved as a GraphML file here: data/raw/ucsd_walk.graphml

This file captures the pedestrian network as it appeared on **September 30, 2025**, and is the **official ground-truth topology** for all simulations and training runs.

> ⚠️ **Important**: Do **not** re-download the graph using `osmnx.graph_from_place` or similar functions.  
> Doing so may lead to subtle inconsistencies across different versions of OSM or OSMnx, which can break reproducibility.

### 👎 Do NOT do this:

```python
# DON'T do this in your own scripts
ox.graph_from_place("University of California San Diego, ...")