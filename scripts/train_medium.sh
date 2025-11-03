#!/bin/bash

echo "🎯 Starting Training with MEDIUM CONFIG"
echo "========================================"
echo ""
echo "Why Medium instead of Large:"
echo "  - Large: 1.4B params, 32K context = 41GB VRAM (OOM!)"
echo "  - Medium: 850M params, 16K context = ~25GB VRAM (fits!)"
echo "  - Still powerful - 850M parameters is huge!"
echo "  - Chunks auto-truncated from 32K to 16K"
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
  --gradient_accumulation 4 \
  --save_steps 1500 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_medium.log
"

echo ""
echo "✅ Training started!"
echo ""
echo "Model specs:"
echo "  - Parameters: 850M"
echo "  - Context: 16K tokens (auto-truncated from 32K chunks)"
echo "  - Batch size: 32"
echo "  - VRAM usage: ~25GB (safe!)"
echo "  - Training time: ~3 hours"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_medium.log"
echo ""
echo "Training should start within 10 seconds!"

