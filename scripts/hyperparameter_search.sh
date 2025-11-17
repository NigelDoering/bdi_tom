#!/bin/bash
# Hyperparameter search for baseline transformer model
# This script runs multiple training configurations to find optimal hyperparameters

echo "🔍 Starting Hyperparameter Search for Baseline Transformer"
echo "============================================================"
echo ""
echo "Configuration: 20 runs × 25 epochs = 500 total epochs"
echo "All runs will be tracked in W&B with tag: hp-search"
echo ""

# Set common parameters
RUN_DIR="data/simulation_data/run_8"
GRAPH_PATH="data/processed/ucsd_walk_full.graphml"
NUM_EPOCHS=25
SEED=42
CHECKPOINT_BASE="checkpoints/hp_search"

# Counter for run number
RUN_NUM=0

# Function to run training
run_training() {
    local lr=$1
    local bs=$2
    local fd=$3
    local td=$4
    local nl=$5
    local nh=$6
    local dp=$7
    
    RUN_NUM=$((RUN_NUM + 1))
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run $RUN_NUM/20: lr=$lr, bs=$bs, fd=$fd, td=$td, nl=$nl, nh=$nh, dp=$dp"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Create checkpoint directory for this run
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/run_${RUN_NUM}"
    
    # Run training with W&B enabled
    python models/training/train_baseline_transformer.py \
        --run_dir "$RUN_DIR" \
        --graph_path "$GRAPH_PATH" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$lr" \
        --batch_size "$bs" \
        --fusion_dim "$fd" \
        --transformer_dim "$td" \
        --num_transformer_layers "$nl" \
        --num_heads "$nh" \
        --dropout "$dp" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --seed "$SEED" \
        --use_wandb \
        --wandb_tags "hp-search,run-$RUN_NUM"
    
    echo "✓ Run $RUN_NUM complete"
}

# ============================================================
# Hyperparameter Grid
# ============================================================
# We'll test different combinations focusing on:
# - Learning rate: Critical for convergence
# - Batch size: Affects gradient quality and speed
# - Model capacity: fusion_dim, transformer_dim, layers, heads
# - Regularization: dropout

echo "Starting 20 hyperparameter configurations..."
echo ""

# Run 1: Baseline (current default)
run_training 0.001 32 64 128 1 4 0.1

# Run 2-4: Learning rate variations
run_training 0.0005 32 64 128 1 4 0.1  # Lower LR
run_training 0.003 32 64 128 1 4 0.1   # Higher LR
run_training 0.0001 32 64 128 1 4 0.1  # Very low LR

# Run 5-7: Batch size variations
run_training 0.001 16 64 128 1 4 0.1   # Smaller batches
run_training 0.001 64 64 128 1 4 0.1   # Larger batches
run_training 0.001 8 64 128 1 4 0.1    # Very small batches

# Run 8-10: Model capacity (transformer_dim)
run_training 0.001 32 64 64 1 4 0.1    # Smaller transformer
run_training 0.001 32 64 256 1 4 0.1   # Larger transformer
run_training 0.001 32 64 192 1 4 0.1   # Medium-large transformer

# Run 11-12: Fusion dimension
run_training 0.001 32 32 128 1 4 0.1   # Smaller fusion
run_training 0.001 32 128 128 1 4 0.1  # Larger fusion

# Run 13-14: Number of transformer layers
run_training 0.001 32 64 128 2 4 0.1   # 2 layers
run_training 0.001 32 64 128 3 4 0.1   # 3 layers

# Run 15-16: Number of attention heads
run_training 0.001 32 64 128 1 8 0.1   # More heads
run_training 0.001 32 64 128 1 2 0.1   # Fewer heads

# Run 17-18: Dropout variations
run_training 0.001 32 64 128 1 4 0.2   # Higher dropout
run_training 0.001 32 64 128 1 4 0.05  # Lower dropout

# Run 19-20: Best combinations (intuition-based)
run_training 0.0005 64 64 192 2 8 0.15  # Large model, conservative LR
run_training 0.001 32 128 256 2 4 0.1   # High capacity balanced

echo ""
echo "============================================================"
echo "🎉 Hyperparameter Search Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $CHECKPOINT_BASE/"
echo "View all runs in W&B: Filter by tag 'hp-search'"
echo ""
echo "Next steps:"
echo "  1. Go to your W&B project: bdi-tom-goal-prediction"
echo "  2. Filter runs by tag: hp-search"
echo "  3. Compare val/top1_accuracy across all runs"
echo "  4. Select best configuration for full training (100 epochs)"
echo ""
