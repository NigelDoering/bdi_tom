# Baseline Transformer Model

Transformer-based baseline model for per-node Theory of Mind (ToM) prediction on trajectory data.

## Key Differences from LSTM Baseline

| Feature | LSTM Baseline | Transformer Baseline |
|---------|---------------|---------------------|
| Dataset | Per-node expansion (~700k samples) | Full trajectories (~70k samples) |
| Processing | Sequential (LSTM) | Parallel (Transformer + causal mask) |
| Speed | Slower (1x baseline) | 10-30x faster |
| Use Case | Final model training | Embedding pipeline training + transfer |

## Architecture

```
Input: [node1, node2, ..., nodeN]
  ↓
UnifiedEmbeddingPipeline (Node2Vec + optional temporal/agent)
  ↓
Projection Layer (fusion_dim → d_model)
  ↓
Positional Encoding
  ↓
Transformer Encoder (4 layers, 8 heads) + Causal Masking
  ↓
Three Prediction Heads:
  - Goal Prediction: [batch, seq_len, num_pois]
  - Next-Step Prediction: [batch, seq_len, num_nodes]
  - Category Prediction: [batch, seq_len, num_categories]
```

**Causal Masking**: Position `i` can only attend to positions `0...i`, ensuring each position's prediction depends only on past context (mimics per-node training without dataset expansion).

## Training

### Basic Training
```bash
python -m models.baseline_transformer.train_baseline_transformer \
    --num_epochs 50 \
    --batch_size 64 \
    --lr 1e-3
```

### With Embedding Pipeline Saving (for Transfer Learning)
```bash
python -m models.baseline_transformer.train_baseline_transformer \
    --num_epochs 50 \
    --batch_size 64 \
    --save_embedding_pipeline \
    --log_percentiles
```

### Key Arguments
- `--save_embedding_pipeline`: Save embedding pipeline separately after each epoch (for transfer to LSTM/VAE)
- `--log_percentiles`: Log accuracy at 5%, 50%, 90% trajectory positions (validation only)
- `--d_model`: Transformer model dimension (default: 256)
- `--nhead`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 4)
- `--batch_size`: Batch size (default: 64, can be larger than LSTM)

## Transfer Learning Strategy

1. **Train Transformer First** (fast):
   ```bash
   python -m models.baseline_transformer.train_baseline_transformer \
       --num_epochs 50 \
       --save_embedding_pipeline
   ```

2. **Load Embeddings into LSTM**:
   ```python
   from models.utils.utils import load_embedding_pipeline
   
   # Create LSTM model
   lstm_model = PerNodeToMPredictor(...)
   
   # Load pre-trained embeddings
   load_embedding_pipeline('checkpoints/baseline_transformer/best_embedding_pipeline.pt', lstm_model)
   
   # Freeze embeddings (optional)
   lstm_model.freeze_embedding_pipeline()
   
   # Train LSTM predictor only
   train(lstm_model, ...)
   ```

## Checkpointing

When `--save_embedding_pipeline` is enabled:

- **Full Model Checkpoints**:
  - `best_model.pt`: Best model by validation accuracy
  - `checkpoint_epoch_{N}.pt`: Periodic checkpoints (every 5 epochs)

- **Embedding Pipeline Checkpoints** (for transfer learning):
  - `best_embedding_pipeline.pt`: Best embedding pipeline
  - `embedding_pipeline_epoch_{N}.pt`: Periodic embedding checkpoints

## W&B Logging (Fixed)

Now correctly logs **average** accuracy across all batches:

- `train/goal_acc`, `train/nextstep_acc`, `train/category_acc`: Training accuracies
- `val/goal_acc`, `val/nextstep_acc`, `val/category_acc`: Validation accuracies
- `val_by_position/5%_goal_acc`, `val_by_position/50%_goal_acc`, `val_by_position/90%_goal_acc`: Accuracy at trajectory percentiles (if `--log_percentiles` enabled)

## Dataset Format

Unlike LSTM baseline, this dataset keeps full trajectories intact:

```python
# LSTM: Expands to per-node samples
# [n1→n2→n3→n4→goal] becomes:
# sample1: history=[n1], next=n2
# sample2: history=[n1,n2], next=n3
# sample3: history=[n1,n2,n3], next=n4
# sample4: history=[n1,n2,n3,n4], next=goal

# Transformer: Single sample
# sample: nodes=[n1,n2,n3,n4], next=[n2,n3,n4,goal]
# Predicts at ALL positions simultaneously with causal masking
```

## Model Files

- `baseline_transformer_model.py`: PerNodeTransformerPredictor class
- `baseline_transformer_dataset.py`: TransformerTrajectoryDataset and collate function
- `train_baseline_transformer.py`: Training script with W&B logging
- `__init__.py`: Package exports
- `README.md`: This file

## Performance Expectations

- **Training Speed**: 10-30x faster than LSTM baseline
- **Accuracy**: Similar to LSTM (both use same embedding pipeline)
- **Memory**: Slightly higher (full sequences in memory, but fewer total samples)
- **Best Use**: Pre-training embeddings, then transfer to other architectures
