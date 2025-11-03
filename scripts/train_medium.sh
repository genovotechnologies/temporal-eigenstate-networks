#!/bin/bash

echo "🎯 Starting Training with MEDIUM CONFIG (OOM-safe)"
echo "===================================================="
echo ""
echo "Why Medium instead of Large:"
echo "  - Large: 1.4B params with 32K context = 41GB VRAM (OOM!)"
echo "  - Medium: 850M params with 16K context = 25-30GB VRAM (fits!)"
echo "  - Still a very powerful model for your 32K chunks"
echo ""

# Kill stuck processes
echo "1. Cleaning up..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting training with Medium config..."
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
  --gradient_accumulation 4 \
  --save_steps 2000 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_medium.log
"

echo ""
echo "✅ Training started with Medium config!"
echo ""
echo "Model specs:"
echo "  - Parameters: 850M (vs 1.4B large)"
echo "  - Context: 16K tokens (chunks will be truncated from 32K)"
echo "  - Batch size: 32"
echo "  - VRAM usage: ~25-30GB (safe!)"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_medium.log"
echo ""
echo "Expected: Training starts within 10 seconds!"
