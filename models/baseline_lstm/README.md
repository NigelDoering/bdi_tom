# Baseline LSTM Model

This folder contains the baseline LSTM architecture for BDI Theory of Mind prediction.

## Structure

```
baseline_lstm/
├── __init__.py                  # Package exports
├── baseline_lstm_model.py       # PerNodeToMPredictor model definition
├── baseline_lstm_dataset.py     # PerNodeTrajectoryDataset and collate function
├── train_baseline_lstm.py       # Training script with W&B logging
└── README.md                    # This file
```

## Components

### 1. Model (`baseline_lstm_model.py`)
- **PerNodeToMPredictor**: LSTM-based model with three prediction heads
  - Goal prediction (which POI)
  - Next-step prediction (immediate next node)
  - Category prediction (semantic goal category)

### 2. Dataset (`baseline_lstm_dataset.py`)
- **PerNodeTrajectoryDataset**: Expands trajectories into per-node training samples
- **collate_per_node_samples**: Batches variable-length histories with padding

### 3. Training (`train_baseline_lstm.py`)
- Complete training pipeline with W&B logging
- Uses pre-defined train/val/test splits
- Early stopping and checkpoint saving

## Quick Start

### Basic Training
```bash
python -m models.baseline_lstm.train_baseline_lstm
```

### Custom Configuration
```bash
python -m models.baseline_lstm.train_baseline_lstm \
    --num_epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_dim 512
```

### All Arguments
```bash
# Data paths
--data_dir                    # Default: data/simulation_data/run_8
--trajectory_filename         # Default: trajectories/trajectories.json
--split_indices_path          # Default: data/simulation_data/run_8/split_data/split_indices_seed42.json
--graph_path                  # Default: data/processed/ucsd_walk_full.graphml
--checkpoint_dir              # Default: checkpoints/baseline_lstm

# Model architecture
--node_embedding_dim 64       # Node embedding dimension
--temporal_dim 64             # Temporal encoding dimension
--agent_dim 64                # Agent encoding dimension
--fusion_dim 128              # Fusion layer dimension
--hidden_dim 256              # LSTM hidden dimension
--num_heads 4                 # Attention heads
--dropout 0.1                 # Dropout rate
--freeze_embedding            # Freeze embedding pipeline (flag)

# Training
--batch_size 32               # Training batch size
--num_epochs 50               # Number of epochs
--learning_rate 0.001         # Learning rate
--weight_decay 1e-5           # Weight decay
--patience 10                 # Early stopping patience
--seed 42                     # Random seed
```

## Data Format

### Input Data
- **Trajectories**: `data/simulation_data/run_8/trajectories/trajectories.json`
  - List of trajectory dictionaries with `path` and `goal_node` keys
- **Split Indices**: `data/simulation_data/run_8/split_data/split_indices_seed42.json`
  - Pre-defined train/val/test indices for reproducibility
- **Graph**: `data/processed/ucsd_walk_full.graphml`
  - UCSD campus graph with node attributes (including `Category`)

### Data Splits (seed=42)
- Train: 70,000 trajectories (70%)
- Val: 15,000 trajectories (15%)
- Test: 15,000 trajectories (15%)

## Model Architecture

```
INPUT: trajectory history [n1, n2, n3, ...]
  ↓
[UnifiedEmbeddingPipeline]
  • Node2Vec embeddings
  • Temporal encoding
  • Agent encoding
  ↓
[2-Layer LSTM]
  • Hidden dim: 128 (default 256 / 2)
  • Aggregates trajectory history
  ↓
[Feature Fusion Layer]
  • Combines current + history context
  • Hidden dim: 256 (default)
  ↓
[Prediction Heads]
  • Goal Head → 230 POIs
  • Next-Step Head → 664 nodes
  • Category Head → 7 categories
  ↓
OUTPUT: {goal, nextstep, category, embeddings}
```

## Training Details

### Loss Function
```
total_loss = 1.0 × goal_loss + 0.5 × nextstep_loss + 0.5 × category_loss
```

### Optimization
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR (T_max=num_epochs, eta_min=lr×0.01)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: Patience=10 epochs on validation goal accuracy

### Metrics Tracked
- Loss (total, goal, nextstep, category)
- Accuracy (goal, nextstep, category)
- Learning rate
- Gradient norms (via W&B)

## W&B Logging

The training script logs to Weights & Biases project `bdi-tom-baseline-lstm`:

- **Batch-level**: Loss and accuracy for each batch
- **Epoch-level**: Train/val metrics with comparison plots
- **Model info**: Parameter counts and architecture details
- **Checkpoints**: Best model path and metrics

## Usage in Python

```python
from models.baseline_lstm import (
    PerNodeToMPredictor,
    PerNodeTrajectoryDataset,
    collate_per_node_samples
)

# Create model
model = PerNodeToMPredictor(
    num_nodes=664,
    num_agents=1,
    num_poi_nodes=230,
    num_categories=7,
    hidden_dim=256,
)

# Create dataset
dataset = PerNodeTrajectoryDataset(
    trajectories=train_trajs,
    graph=graph,
    poi_nodes=poi_nodes,
)

# Create data loader
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_per_node_samples,
)

# Forward pass
batch = next(iter(loader))
predictions = model(
    batch['history_node_indices'],
    batch['history_lengths']
)
# predictions = {'goal': ..., 'nextstep': ..., 'category': ..., 'embeddings': ...}
```

## Checkpoints

Model checkpoints are saved to `checkpoints/baseline_lstm/best_model.pt` and include:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Training epoch
- `loss`: Validation loss
- `metrics`: All validation metrics
- `config`: Model configuration for reconstruction

## Notes

- The model uses per-node expansion: each trajectory creates multiple training samples
- Category prediction now correctly uses the goal node's category (not next-step)
- The model can be trained with frozen embeddings using `--freeze_embedding`
