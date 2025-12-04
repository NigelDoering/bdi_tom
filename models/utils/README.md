# BDI-ToM Model Training

This directory contains the training infrastructure for end-to-end training of the BDI-ToM goal prediction model.

## Overview

The training pipeline trains the complete model end-to-end:

1. **TrajectoryEncoder** (Transformer-based): Encodes agent trajectory sequences
2. **WorldGraphEncoder** (GAT-based): Encodes spatial graph structure  
3. **ToMGraphEncoder** (Fusion layer): Combines trajectory + map encodings
4. **GoalPredictionModel** (Transformer + Classifier): Predicts goal from fused representation

## Files

### `train_baseline_transformer.py`
Main training script with:
- Automatic device detection (CUDA → MPS → CPU)
- Data loading and preprocessing
- Train/validation/test split
- Training loop with progress bars
- Validation and early stopping
- Checkpointing (best model + latest model)
- Metrics tracking and logging

### `data_loader.py`
Data loading utilities:
- `TrajectoryDataset`: PyTorch Dataset for trajectory data
- `load_simulation_data()`: Load trajectories and graph from run directory
- `split_data()`: Train/val/test split with reproducible seeding
- `create_dataloaders()`: Create PyTorch DataLoaders with batching
- `collate_trajectories()`: Custom collate function for variable-length trajectories

### `utils.py`
Training utilities:
- `get_device()`: Device detection (CUDA → MPS → CPU)
- `set_seed()`: Set random seeds for reproducibility
- `save_checkpoint()` / `load_checkpoint()`: Model checkpointing
- `compute_accuracy()`: Top-k accuracy computation
- `AverageMeter`: Running average tracker
- `MetricsTracker`: Track and save training metrics

## Quick Start

### 1. Basic Training

```bash
# From project root
python models/training/train_baseline_transformer.py \
    --run_dir output/run_8 \
    --graph_path data/processed/ucsd_walk_semantic.graphml
```

### 2. Custom Hyperparameters

```bash
python models/training/train_baseline_transformer.py \
    --run_dir output/run_8 \
    --batch_size 64 \
    --num_epochs 200 \
    --learning_rate 0.0005 \
    --dropout 0.2 \
    --checkpoint_dir checkpoints/my_experiment
```

### 3. Using the Training Script

```bash
# Make executable
chmod +x scripts/train_baseline_transformer_model.sh

# Run training
./scripts/train_baseline_transformer_model.sh
```

## Command Line Arguments

### Data Arguments
- `--run_dir`: Path to simulation run directory (default: `output/run_8`)
- `--graph_path`: Path to graph file (default: `data/processed/ucsd_walk_semantic.graphml`)
- `--train_ratio`: Training set ratio (default: 0.7)
- `--val_ratio`: Validation set ratio (default: 0.15)
- `--test_ratio`: Test set ratio (default: 0.15)

### Training Arguments
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--weight_decay`: Weight decay for optimizer (default: 1e-4)
- `--early_stop_patience`: Early stopping patience (default: 15)

### Model Arguments
- `--fusion_dim`: Fusion layer dimension (default: 64)
- `--transformer_dim`: Transformer dimension (default: 128)
- `--num_transformer_layers`: Number of transformer layers (default: 1)
- `--num_heads`: Number of attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.1)

### Other Arguments
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints/baseline_transformer`)
- `--seed`: Random seed for reproducibility (default: 42)

### W&B Arguments (Experiment Tracking)
- `--use_wandb`: Enable Weights & Biases tracking (flag, default: False)
- `--wandb_project`: W&B project name (default: `bdi-tom-goal-prediction`)
- `--wandb_run_name`: W&B run name (auto-generated if not provided)
- `--wandb_tags`: Comma-separated W&B tags (e.g., `baseline,transformer,v1`)

## Experiment Tracking with W&B

The training script supports [Weights & Biases](https://wandb.ai/) for experiment tracking.

### Setup

1. Install wandb:
```bash
pip install wandb
```

2. Login to W&B:
```bash
wandb login
```

### Usage

Enable W&B tracking with the `--use_wandb` flag:

```bash
python models/training/train_baseline_transformer.py \
    --run_dir data/simulation_data/run_8 \
    --graph_path data/processed/ucsd_walk_full.graphml \
    --use_wandb \
    --wandb_tags baseline,transformer,lr0.001
```

### What Gets Tracked

W&B automatically tracks:
- **Training metrics**: loss, top-1/top-5 accuracy per epoch
- **Validation metrics**: loss, top-1/top-5 accuracy per epoch
- **Incremental validation**: Performance at 25%, 50%, 75% trajectory observation
- **Test metrics**: Final performance on test set
- **Hyperparameters**: All model and training configuration
- **System metrics**: GPU/CPU usage, memory
- **Model artifacts**: Best model checkpoint with metadata

### Custom Run Names

Runs are auto-named based on key hyperparameters:
```
baseline_transformer_lr0.001_bs32_fd64_td128
```

This includes:
- Model type: `baseline_transformer`
- Learning rate: `lr0.001`
- Batch size: `bs32`
- Fusion dim: `fd64`
- Transformer dim: `td128`
- Layers/heads (if non-default): `l2_h8`

Override with `--wandb_run_name`:
```bash
--use_wandb --wandb_run_name my_custom_experiment_v2
```

**Example auto-generated names:**
```
baseline_transformer_lr0.001_bs32_fd64_td128       # Default settings
baseline_transformer_lr0.0005_bs64_fd64_td256      # Higher capacity
baseline_transformer_lr0.001_bs32_fd64_td128_l2_h8 # More layers/heads
```

### Comparing Experiments

All runs in the same project are automatically comparable in the W&B dashboard. View:
- Metric trends across epochs
- Hyperparameter correlations
- Model performance distributions
- Incremental validation curves

### Example: Training Multiple Models

```bash
# Baseline model
python models/training/train_baseline_transformer.py \
    --use_wandb --wandb_tags baseline,v1

# Increased capacity
python models/training/train_baseline_transformer.py \
    --transformer_dim 256 --num_transformer_layers 2 \
    --use_wandb --wandb_tags high-capacity,v1

# Different learning rate
python models/training/train_baseline_transformer.py \
    --learning_rate 0.0005 \
    --use_wandb --wandb_tags low-lr,v1
```

All three experiments will appear in your W&B project dashboard for easy comparison.
- `--seed`: Random seed for reproducibility (default: 42)

## Output

### Checkpoints
Saved to `--checkpoint_dir`:
- `best_model.pt`: Model with best validation accuracy
- `latest_model.pt`: Most recent model checkpoint
- `metrics.json`: Training metrics history (losses, accuracies)

### Checkpoint Contents
Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'loss': float,
    'metrics': {'val_top1': float, 'val_top5': float}
}
```

### Metrics JSON
```json
{
    "train_losses": [1.234, 0.987, ...],
    "val_losses": [1.345, 1.012, ...],
    "train_accs": [45.67, 56.78, ...],
    "val_accs": [43.21, 54.32, ...],
    "train_top5_accs": [78.90, 85.43, ...],
    "val_top5_accs": [76.54, 83.21, ...]
}
```

## Device Support

The training script automatically detects and uses the best available device:

1. **CUDA (NVIDIA GPUs)**: Fastest, uses `torch.cuda`
2. **MPS (Apple Silicon)**: Mac M1/M2/M3 GPU acceleration
3. **CPU**: Fallback for systems without GPU

```python
# Device selection happens automatically
device = get_device()
# Output examples:
# 🚀 Using CUDA GPU: NVIDIA GeForce RTX 3090
# 🚀 Using Apple Silicon GPU (MPS)
# ⚠️  Using CPU (training will be slower)
```

## Training Process

### Epoch Flow
1. **Training**: 
   - Forward pass through model
   - Compute cross-entropy loss
   - Backward pass with gradient clipping (max_norm=1.0)
   - Optimizer step
   - Track loss and top-1/top-5 accuracy

2. **Validation**:
   - Evaluate on validation set without gradients
   - Compute validation loss and accuracies
   - Update learning rate scheduler

3. **Checkpointing**:
   - Save if validation accuracy improves
   - Save latest checkpoint every epoch
   - Early stop if no improvement for N epochs

### Learning Rate Schedule
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
  - Reduces LR by 0.5 when validation loss plateaus
  - Patience: 5 epochs

### Early Stopping
- Monitors validation top-1 accuracy
- Stops if no improvement for `--early_stop_patience` epochs (default: 15)

## Example Output

```
================================================================================
BDI-ToM GOAL PREDICTION TRAINING
================================================================================

🚀 Using Apple Silicon GPU (MPS)

📂 Loading data...
📂 Loading graph from data/processed/ucsd_walk_semantic.graphml
📍 Found 45 POI nodes
📂 Loading trajectories from output/run_8/trajectories.json
🚶 Loaded 5000 trajectories

📊 Data Split:
   Train: 3500 (70%)
   Val:   750 (15%)
   Test:  750 (15%)

📦 DataLoaders created:
   Batch size: 32
   Train batches: 110
   Val batches:   24
   Test batches:  24

🔧 Initializing encoders...

🏗️  Building model...
   Number of nodes: 1234
   Graph feature dim: 12
   Number of POI nodes: 45
   Total parameters: 156,789
   Trainable parameters: 156,789

🚀 Starting training for 100 epochs...
================================================================================

Epoch 1/100
----------------------------------------
Epoch 1: 100%|████████| 110/110 [00:45<00:00, loss=2.1234, top1=25.34%, top5=62.11%]
Val: 100%|████████████| 24/24 [00:05<00:00, loss=2.0123, top1=28.45%, top5=65.23%]

📊 Epoch 1 Summary:
   Train - Loss: 2.1234, Top-1: 25.34%, Top-5: 62.11%
   Val   - Loss: 2.0123, Top-1: 28.45%, Top-5: 65.23%
💾 Checkpoint saved: checkpoints/goal_predictor/best_model.pt
   🌟 New best validation accuracy: 28.45%

...

================================================================================
TRAINING SUMMARY
================================================================================

Best Validation Accuracy:
  Epoch 45: 78.92%

Best Validation Top-5 Accuracy:
  Epoch 43: 94.56%

Best Validation Loss:
  Epoch 47: 0.5234

Final Epoch (50):
  Train Loss: 0.4123 | Val Loss: 0.5567
  Train Acc:  81.23% | Val Acc:  77.45%
  Train Top-5: 95.67% | Val Top-5: 93.21%
================================================================================

================================================================================
FINAL TEST EVALUATION
================================================================================
Test: 100%|████████████| 24/24 [00:05<00:00, loss=0.5432, top1=76.89%, top5=93.45%]

🎯 Test Results:
   Loss: 0.5432
   Top-1 Accuracy: 76.89%
   Top-5 Accuracy: 93.45%
================================================================================
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 16 or 8)
- Reduce model size: `--fusion_dim 32 --transformer_dim 64`

### Training Too Slow
- Check device: Should see "Using CUDA GPU" or "Using Apple Silicon GPU (MPS)"
- Reduce validation frequency (modify code to validate every N epochs)
- Increase batch size if you have memory headroom

### Poor Accuracy
- Increase model capacity: `--fusion_dim 128 --transformer_dim 256`
- Add more transformer layers: `--num_transformer_layers 2`
- Reduce dropout: `--dropout 0.05`
- Train longer: `--num_epochs 200`
- Reduce learning rate: `--learning_rate 0.0005`

### Overfitting (train acc >> val acc)
- Increase dropout: `--dropout 0.3`
- Increase weight decay: `--weight_decay 0.001`
- Use smaller model: `--fusion_dim 32 --transformer_dim 64`
- Add more training data (run longer simulation)

## Loading a Trained Model

```python
import torch
from models.baseline_transformer.transformer_predictor import GoalPredictionModel
from models.fusion_encoders_preprocessing.fusion_encoder import ToMGraphEncoder
from models.utils.utils import load_checkpoint

# Initialize model (same architecture as training)
# Note: ToMGraphEncoder API has changed - see current implementation
fusion_encoder = ToMGraphEncoder(
    node_emb_dim=128,
    num_agents=100,
    traj_node_emb_dim=32,
    hidden_dim=64,
    output_dim=64
)

model = GoalPredictionModel(
    fusion_encoder=fusion_encoder,
    num_poi_nodes=45,
    fusion_dim=64,
    hidden_dim=128
)

# Load checkpoint
epoch, loss, metrics = load_checkpoint(
    'checkpoints/baseline_transformer/best_model.pt',
    model
)

# Use model for prediction
model.eval()
with torch.no_grad():
    predictions = model(traj_batch, graph_data, return_probs=True)
    top_k_goals = model.predict_top_k(traj_batch, graph_data, k=5)
```

## Next Steps

After training:
1. Evaluate on test set (done automatically at end of training)
2. Visualize predictions vs ground truth
3. Analyze failure cases
4. Tune hyperparameters based on results
5. Collect more training data if needed
6. Experiment with different architectures
