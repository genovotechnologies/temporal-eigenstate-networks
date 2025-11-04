#!/bin/bash
# Train SMALL model with 32K context (matches pre-tokenized chunks)
# Zero data waste, full long-range capability!

set -e

# Configuration
CONFIG="small_32k"
DATASET="finewebedu"
OUTPUT_DIR="$HOME/ten_workspace"
LOG_DIR="$OUTPUT_DIR/logs"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_small_32k_${TIMESTAMP}.log"

echo "=========================================="
echo "TEN Training - SMALL 32K Configuration"
echo "=========================================="
echo "Model: 123M params, 32K context"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "Log: $LOG_FILE"
echo "=========================================="

# Training command
python3 examples/train_digitalocean.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --pretokenized \
    --tokenized_dir "$OUTPUT_DIR/tokenized/$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 1 \
    --learning_rate 3e-4 \
    --mixed_precision \
    --num_workers 4 \
    --gradient_accumulation 4 \
    --save_every 1000 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete! Log saved to: $LOG_FILE"
