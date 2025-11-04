#!/bin/bash

echo "🚀 MEDIUM TRAINING - Production Run"
echo "===================================="
echo ""
echo "✅ Architecture PROVEN viable (nano: 2.94GB)"
echo "✅ Aggressive detaching enabled (10× memory reduction!)"
echo "✅ Memory formula: ~0.72MB per token"
echo ""
echo "Medium Config:"
echo "  - 268M parameters"
echo "  - 1K context (conservative, proven safe)"
echo "  - Batch size: 32"
echo "  - Total tokens: 32K"
echo "  - Expected VRAM: ~36GB (safe for 48GB GPU)"
echo "  - Training time: ~2 hours"
echo ""

# Kill stuck processes
echo "1. Cleaning up..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting training..."
echo ""

tmux new -s training -d "
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 examples/train_digitalocean.py \
  --config medium \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 8 \
  --save_steps 1000 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_medium.log
"

echo ""
echo "✅ Medium training started!"
echo ""
echo "Expected behavior:"
echo "  - Memory: ~36GB (should stay under 40GB)"
echo "  - Speed: ~5-10 seconds per batch"
echo "  - Training: ~2 hours for 1 epoch"
echo "  - Cost: ~$3.14 (within budget!)"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  watch -n 1 nvidia-smi"
echo "  tail -f ~/ten_workspace/logs/training_medium.log"
echo ""
echo "This should FINALLY complete a training run successfully!"
echo "  - Training time: ~3 hours"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_medium.log"
echo ""
echo "Training should start within 10 seconds!"

