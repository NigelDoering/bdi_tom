## BDI VAE Training Script - Accuracy Metrics Update

### Changes Made

#### 1. Added Top-5 Goal Accuracy
- **Training function**: Compute top-5 accuracy for goal predictions
- **Validation function**: Compute top-5 accuracy for goal predictions
- Tracks whether the true goal is in the top-5 predictions

#### 2. Added All Missing Accuracy Metrics
The model now logs:
- **Goal Top-1 Accuracy**: Original goal prediction accuracy
- **Goal Top-5 Accuracy**: Whether true goal is in top-5 predictions
- **Next Node Accuracy**: Accuracy of next step prediction
- **Goal Category Accuracy**: Accuracy of goal category prediction

#### 3. Organized W&B Metrics into Subcategories
Metrics are now organized into clear hierarchies for better visualization:

**Accuracy Metrics:**
- `train/accuracy/goal_top1` - Top-1 goal accuracy (train)
- `train/accuracy/goal_top5` - Top-5 goal accuracy (train)
- `train/accuracy/next_node` - Next node prediction accuracy (train)
- `train/accuracy/goal_category` - Goal category accuracy (train)
- `val/accuracy/goal_top1` - Top-1 goal accuracy (validation)
- `val/accuracy/goal_top5` - Top-5 goal accuracy (validation)
- `val/accuracy/next_node` - Next node prediction accuracy (validation)
- `val/accuracy/goal_category` - Goal category accuracy (validation)

**Percentile Accuracy:**
- `train/accuracy/goal_top1_15%` - Accuracy at 15th percentile history length
- `train/accuracy/goal_top1_50%` - Accuracy at 50th percentile history length
- `train/accuracy/goal_top1_85%` - Accuracy at 85th percentile history length
- `val/accuracy/goal_top1_15%` - Val accuracy at 15th percentile
- `val/accuracy/goal_top1_50%` - Val accuracy at 50th percentile
- `val/accuracy/goal_top1_85%` - Val accuracy at 85th percentile

**Loss Metrics:**
- `train/loss/total` - Total combined loss
- `train/loss/prediction` - Total prediction loss
- `train/loss/vae` - Total VAE loss
- `train/loss/goal` - Goal prediction loss
- `train/loss/next_node` - Next node prediction loss
- `train/loss/category` - Category prediction loss
- (Same for `val/loss/*`)

**VAE Breakdown:**
- `train/vae/belief_loss` - Belief VAE total loss
- `train/vae/belief_recon_loss` - Belief reconstruction loss
- `train/vae/belief_kl_loss` - Belief KL divergence
- `train/vae/desire_loss` - Desire VAE total loss
- `train/vae/desire_recon_loss` - Desire reconstruction loss
- `train/vae/desire_kl_loss` - Desire KL divergence
- `train/vae/intention_loss` - Intention VAE total loss
- `train/vae/intention_recon_loss` - Intention reconstruction loss
- `train/vae/intention_kl_loss` - Intention KL divergence
- (Same for `val/vae/*`)

#### 4. Enhanced Console Output
The epoch summary now shows all accuracy metrics:
```
📊 Epoch X Summary:
   Train - Loss: X.XXXX | VAE: X.XXXX | Pred: X.XXXX
           Goal Top-1: XX.X% | Top-5: XX.X% | Next: XX.X% | Cat: XX.X%
   Val   - Loss: X.XXXX | VAE: X.XXXX | Pred: X.XXXX
           Goal Top-1: XX.X% | Top-5: XX.X% | Next: XX.X% | Cat: XX.X%
```

### Benefits

1. **Complete Accuracy Tracking**: All prediction heads now have accuracy metrics logged
2. **Better W&B Organization**: Metrics are grouped by type (accuracy, loss, vae) for easier comparison and plotting
3. **Top-K Analysis**: Top-5 accuracy helps understand if the model is "close" when it's wrong
4. **Parity with Baseline Models**: Now logs the same accuracy types as LSTM/Transformer models

### Usage

Run training as before:
```bash
python -m models.vae_bdi_simple.train_bdi_vae \
    --node_embedding_dim 128 \
    --temporal_dim 128 \
    --agent_dim 128 \
    --fusion_dim 128 \
    --pretrained_embedding checkpoints/keepers/unified_embedding.pt \
    --freeze_embedding \
    --log_percentiles \
    --use_kl_annealing \
    --kl_anneal_epochs 10 \
    --vae_loss_weight 0.05 \
    --beta_belief 0.5 \
    --beta_desire 0.5 \
    --beta_intention 0.5
```

All new metrics will be automatically logged to W&B and displayed in the console.
