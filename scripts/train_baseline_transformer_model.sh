#!/bin/bash
# Training script for BDI-ToM goal prediction model

echo "Starting BDI-ToM Goal Prediction Training"
echo "=========================================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training with default parameters
# Add --use_wandb to enable experiment tracking
python models/training/train_baseline_transformer.py \
    --run_dir data/simulation_data/run_8 \
    --graph_path data/processed/ucsd_walk_full.graphml \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --early_stop_patience 15 \
    --fusion_dim 64 \
    --transformer_dim 128 \
    --num_transformer_layers 1 \
    --num_heads 4 \
    --dropout 0.1 \
    --checkpoint_dir checkpoints/baseline_transformer \
    --seed 42

echo ""
echo "Training complete! Check checkpoints/baseline_transformer/ for results."
