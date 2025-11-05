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
data/simulation_data/run_1/
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

You can then open this file in your web browser to view an interactive map with the selected trajectories overlaid on the UCSD campus graph.

---

## 🎯 Visualizing Multi-Goal Trajectories

The simulation includes retry logic for closed POIs. When an agent arrives at a closed location, they sample another goal from the same category. After multiple failures, they may give up and return home. You can visualize these multi-attempt trajectories separately:

```bash
python visualization_controller/multi_goal_visualizer.py -x <RUN_ID> -n <NUM_TO_SHOW>
```

### Flags

| Flag | Long Form | Type | Description |
|------|-----------|------|-------------|
| `-x` | `--run_id` | `int` | Run ID to visualize |
| `-n` | `--num_show` | `int` | Number of multi-goal trajectories to display (default: 5) |

### Example

```bash
# Visualize 10 random multi-goal trajectories from run_3
python visualization_controller/multi_goal_visualizer.py -x 3 -n 10
```

### What You'll See

The visualization includes:
- **Path lines**: Colored routes showing the agent's journey
- **Green play button**: Starting location
- **Orange X markers**: Failed goal attempts (POI was closed)
- **Red flag**: Final successful goal (or home if agent gave up)

**Hover over markers** to see:
- Attempt number, hour, category, and POI name
- Example: `"Attempt 2 @ 14:00 | food | Price Center"`

**Click markers** for detailed popup with:
- Status (Failed/Reached)
- POI name and category
- Opening hours
- Node ID

### Output Location

```
data/run_3/visualizations/multi_goal_attempts_10_trajectories.html
```

### How Multi-Goal Attempts Work

1. Agent samples a goal and travels toward it
2. Upon arrival, checks if POI is open at current hour
3. If **closed**: Samples another goal from same category (excludes previous attempts)
4. **Give-up probability** increases with each failure:
   - Attempt 1: 20% chance → home
   - Attempt 2: 50% chance → home  
   - Attempt 3: 80% chance → home
   - Attempt 4: 95% chance → home (forced)
5. When giving up, agent returns to highest-preference home node

**Trajectory Data Structure**:
- `path`: List of `(node_id, goal_node)` tuples showing which goal was active at each step
- `attempted_goals`: List of all goal nodes tried
- `attempts`: Total number of goal attempts
- `returned_home`: Boolean indicating if agent gave up

---

## 📁 Project Directory Structure

```
bdi_tom/
├── agent_controller/           # Agent behavior and planning logic
│   ├── agent.py               # Agent class with preferences and beliefs
│   └── planning_utils.py      # Goal sampling and path planning functions
│
├── data/                      # All data files (graphs, simulations)
│   ├── raw/                   # Original downloaded data
│   │   └── ucsd_walk.graphml  # Ground-truth UCSD pedestrian graph
│   ├── processed/             # Cleaned/annotated graphs
│   │   ├── ucsd_walk_full.graphml      # Graph with POI annotations
│   │   ├── ucsd_walk_labeled.graphml   # Intermediate processing
│   │   └── ucsd_walk_semantic.graphml  # Semantic annotations
│   └── simulation_data/       # Simulation outputs
│       └── run_<X>/           # Data for run X
│           ├── agents/        # Agent configurations
│           ├── trajectories/  # Generated paths
│           └── visualizations/# HTML map visualizations
│
├── graph_controller/          # Graph data structures
│   └── world_graph.py         # WorldGraph wrapper class
│
├── graph_creation_controller/ # Scripts to build/annotate graphs
│   ├── download_ucsd_map.py   # Initial OSM download (DO NOT USE)
│   ├── merge_poi_to_graph.py  # Add POI nodes to graph
│   ├── add_clean_categories.py # Categorize POIs
│   └── add_opening_hours.py   # Add temporal information
│
├── simulation_controller/     # Main simulation logic
│   ├── simulation.py          # Simulation class (single agent step)
│   └── simulation_runner.py   # Main entry point (runs full simulation)
│
├── visualization_controller/  # Visualization scripts
│   ├── map_plotter.py         # Base plotting utilities
│   ├── visualize_trajectories.py      # Standard trajectory viewer
│   └── multi_goal_visualizer.py       # Multi-attempt trajectory viewer
│
├── models/                    # ML models (if applicable)
│   └── encoders/              # Neural network encoders
│
├── notebooks/                 # Jupyter notebooks for exploration
│   └── *.ipynb               # Analysis and prototyping
│
├── testing/                   # Test scripts and utilities
│
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
└── README.md                 # This file
```

### Key Files

- **`agent_controller/agent.py`**: Defines the `Agent` class with hierarchical preferences (category-level and node-level), temporal beliefs, and memory decay
- **`agent_controller/planning_utils.py`**: Functions for sampling goals, start nodes, and planning paths with retry logic for closed POIs
- **`simulation_controller/simulation_runner.py`**: Main script to generate trajectory datasets
- **`graph_controller/world_graph.py`**: Wrapper for NetworkX graphs with POI-specific utilities
- **`data/processed/ucsd_walk_full.graphml`**: The annotated graph used by all simulations (includes POI categories, opening hours, coordinates)

### Data Flow

1. **Graph Creation** (`graph_creation_controller/`) → Annotated graph in `data/processed/`
2. **Simulation** (`simulation_controller/`) → Reads graph, generates trajectories → Saves to `data/simulation_data/run_X/`
3. **Visualization** (`visualization_controller/`) → Reads trajectories, renders HTML maps → Saves to `data/simulation_data/run_X/visualizations/`
4. **Analysis** (`notebooks/` or `models/`) → Loads trajectories for research/training

---