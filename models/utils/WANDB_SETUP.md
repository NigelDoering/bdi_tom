# Weights & Biases Setup Guide

This guide will help you set up and use W&B experiment tracking for the BDI-ToM project.

## Installation

1. Install the wandb package:
```bash
pip install wandb
```

2. Create a W&B account at [wandb.ai](https://wandb.ai/) (free for individuals)

3. Login from your terminal:
```bash
wandb login
```

This will prompt you to paste your API key, which you can find at [wandb.ai/authorize](https://wandb.ai/authorize).

## Quick Start

### Run training with W&B tracking:

```bash
python models/training/train_baseline_transformer.py \
    --run_dir data/simulation_data/run_8 \
    --graph_path data/processed/ucsd_walk_full.graphml \
    --use_wandb
```

After starting, you'll see a message like:
```
📊 W&B tracking enabled: https://wandb.ai/your-username/bdi-tom-goal-prediction/runs/abc123
```

Click the link to view your experiment dashboard in real-time!

## Experiment Workflow

### 1. Baseline Experiment
First, run a baseline to establish performance:

```bash
python models/training/train_baseline_transformer.py \
    --use_wandb \
    --wandb_tags baseline,first-run \
    --num_epochs 50
```

### 2. Hyperparameter Tuning
Try different learning rates:

```bash
# Lower learning rate
python models/training/train_baseline_transformer.py \
    --learning_rate 0.0005 \
    --use_wandb \
    --wandb_tags lr-tuning,lr0.0005

# Higher learning rate
python models/training/train_baseline_transformer.py \
    --learning_rate 0.003 \
    --use_wandb \
    --wandb_tags lr-tuning,lr0.003
```

Try different model capacities:

```bash
# Deeper transformer
python models/training/train_baseline_transformer.py \
    --num_transformer_layers 2 \
    --transformer_dim 256 \
    --use_wandb \
    --wandb_tags architecture,deep-transformer

# More attention heads
python models/training/train_baseline_transformer.py \
    --num_heads 8 \
    --use_wandb \
    --wandb_tags architecture,multihead-8
```

### 3. Compare All Runs
Go to your W&B project dashboard and:
- View all runs in a table
- Compare metrics across runs
- Filter by tags
- Create custom charts
- Export results

## What W&B Tracks

### Metrics Logged Per Epoch
- `train/loss`: Training loss
- `train/top1_accuracy`: Training top-1 accuracy (%)
- `train/top5_accuracy`: Training top-5 accuracy (%)
- `val/loss`: Validation loss
- `val/top1_accuracy`: Validation top-1 accuracy (%)
- `val/top5_accuracy`: Validation top-5 accuracy (%)
- `val/25%_loss`, `val/25%_top1`, `val/25%_top5`: Performance at 25% trajectory observation
- `val/50%_loss`, `val/50%_top1`, `val/50%_top5`: Performance at 50% trajectory observation
- `val/75%_loss`, `val/75%_top1`, `val/75%_top5`: Performance at 75% trajectory observation
- `learning_rate`: Current learning rate

### Final Test Metrics
- `test/loss`: Test loss
- `test/top1_accuracy`: Test top-1 accuracy (%)
- `test/top5_accuracy`: Test top-5 accuracy (%)

### Hyperparameters
All command-line arguments are saved as config, including:
- Model architecture (fusion_dim, transformer_dim, num_heads, etc.)
- Training settings (batch_size, learning_rate, dropout, etc.)
- Data information (num_poi_nodes, num_train_samples, etc.)

### Model Artifacts
The best model checkpoint is saved as a versioned artifact with metadata:
- `best_val_top1`: Best validation accuracy achieved
- `epochs_trained`: Total epochs trained
- `final_val_loss`: Final validation loss

## Advanced Usage

### Custom Run Names
```bash
python models/training/train_baseline_transformer.py \
    --use_wandb \
    --wandb_run_name "experiment_v3_with_regularization" \
    --wandb_tags v3,regularization,dropout0.3 \
    --dropout 0.3
```

### Offline Mode
If you don't have internet during training:
```bash
wandb offline

python models/training/train_baseline_transformer.py --use_wandb

# Later, sync the results
wandb sync
```

### Disable W&B Temporarily
Just remove the `--use_wandb` flag:
```bash
python models/training/train_baseline_transformer.py
# Training proceeds normally without W&B
```

## Tips for Effective Experiment Tracking

1. **Use Descriptive Tags**: Tag experiments by what you're testing
   - `baseline`, `ablation`, `architecture`, `hyperparameter-tuning`
   - Specific values: `lr0.001`, `bs64`, `dropout0.2`

2. **Name Important Runs**: Use `--wandb_run_name` for significant experiments
   - "final_baseline_model"
   - "best_architecture_v2"
   - "paper_submission_model"

3. **Group Related Experiments**: Use consistent tags for related runs
   ```bash
   # All learning rate experiments
   --wandb_tags lr-sweep,experiment-1
   ```

4. **Monitor During Training**: Click the W&B link to watch training live
   - See if loss is decreasing smoothly
   - Check for overfitting (train vs val gap)
   - Monitor incremental validation curves

5. **Compare After Training**: Use W&B dashboard to:
   - Sort runs by best validation accuracy
   - Plot learning curves side-by-side
   - Find hyperparameter correlations
   - Download model artifacts

## Example: Full Experiment Suite

```bash
# 1. Baseline
python models/training/train_baseline_transformer.py \
    --use_wandb --wandb_tags baseline,v1 \
    --checkpoint_dir checkpoints/baseline_v1

# 2. Increased capacity
python models/training/train_baseline_transformer.py \
    --transformer_dim 256 --num_transformer_layers 2 \
    --use_wandb --wandb_tags high-capacity,v1 \
    --checkpoint_dir checkpoints/high_capacity_v1

# 3. More regularization
python models/training/train_baseline_transformer.py \
    --dropout 0.3 --weight_decay 0.001 \
    --use_wandb --wandb_tags regularization,v1 \
    --checkpoint_dir checkpoints/regularized_v1

# 4. Lower learning rate with more epochs
python models/training/train_baseline_transformer.py \
    --learning_rate 0.0003 --num_epochs 150 \
    --use_wandb --wandb_tags long-training,v1 \
    --checkpoint_dir checkpoints/long_training_v1
```

After running these, go to W&B and compare:
- Which architecture performs best?
- Do incremental validation curves differ?
- Is there overfitting in any model?
- Which model should be used for final evaluation?

## Troubleshooting

### "wandb: ERROR Unable to connect"
- Check your internet connection
- Use `wandb offline` mode and sync later

### "wandb: ERROR api_key not configured"
- Run `wandb login` and paste your API key
- Get key from [wandb.ai/authorize](https://wandb.ai/authorize)

### "Run already exists"
- This is normal when restarting training
- Each new run gets a unique ID automatically

### Want to disable W&B without removing the flag?
```bash
export WANDB_MODE=disabled
python models/training/train_baseline_transformer.py --use_wandb
```

## Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B PyTorch Guide](https://docs.wandb.ai/guides/integrations/pytorch)
- [Example Projects](https://wandb.ai/gallery)
