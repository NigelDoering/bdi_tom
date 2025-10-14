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
```

---

## � Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

### Installing Dependencies

1. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies from the lock file**:
   ```bash
   uv sync
   ```

   This will create a virtual environment in `.venv/` and install all required packages with exact versions specified in `uv.lock`.

3. **Activate the environment** (optional, but recommended for interactive work):
   ```bash
   source .venv/bin/activate
   ```

### Running Scripts

You can run scripts either by:

- **With `uv run`** (automatically uses the correct environment):
  ```bash
  uv run python simulation_controller/simulation_runner.py -n 10 -m 20 -x 1
  ```

- **After activating the environment**:
  ```bash
  source .venv/bin/activate
  python simulation_controller/simulation_runner.py -n 10 -m 20 -x 1
  ```

---

## �🚀 Running Simulations

The main simulation script generates synthetic pedestrian trajectory data for multiple agents on the UCSD campus graph.

### Command

```bash
python simulation_controller/simulation_runner.py -n <NUM_AGENTS> -m <NUM_TRAJECTORIES> -x <RUN_ID>
```

### Required Flags

| Flag | Long Form | Type | Description |
|------|-----------|------|-------------|
| `-n` | `--num_agents` | `int` | Number of agents to simulate |
| `-m` | `--num_trajectories` | `int` | Number of trajectories per agent |
| `-x` | `--run_id` | `int` | Run identifier for organizing output data |

### Example

```bash
# Simulate 10 agents, each generating 20 trajectories, saved as run_1
python simulation_controller/simulation_runner.py -n 10 -m 20 -x 1
```

### Output Structure

The simulation saves data to `data/run_<RUN_ID>/`:

```
data/run_1/
├── agents/
│   └── all_agents.json          
├── trajectories/
│   └── all_trajectories.json   
└── visualizations/      
    └── visualization_N_trajectories.html
```

### What Gets Generated

- **Agent Metadata**: Each agent's category preferences (home, study, food, leisure, errands, health) and node-level preference distributions
- **Trajectories**: Paths through the graph with:
  - `path`: List of node IDs representing the route
  - `goal_node`: The destination node ID
  - `hour`: The time of day (0-23) when the trajectory was generated

---

## 📊 Visualizing Trajectories

After running a simulation, you can visualize the generated trajectories on an interactive map:

```bash
python visualization_controller/visualize_trajectories.py -x <RUN_ID> -n <NUM_TO_SHOW>
```

### Flags

| Flag | Long Form | Type | Description |
|------|-----------|------|-------------|
| `-x` | `--run_id` | `int` | Run ID to visualize |
| `-n` | `--num_show` | `int` | Number of trajectories to display (default: 5) |

### Example

```bash
# Visualize 5 random trajectories from run_3
python visualization_controller/visualize_trajectories.py -n 5 -x 3
```

The output HTML map will be saved to:
```
data/run_3/visualizations/visualization_5_trajectories.html
```

You can then open this file in your web browser to view an interactive map with the selected trajectories overlaid on the UCSD campus graph