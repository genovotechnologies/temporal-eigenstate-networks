#!/bin/bash

echo "🔥 Starting Training - SMALL CONFIG (Most Memory Efficient)"
echo "============================================================="
echo ""
echo "Why Small config:"
echo "  - Medium: 570M params, 16K context = OOM even at batch_size 32!"
echo "  - Small: 216M params, 8K context = ~10-15GB VRAM (very safe!)"
echo "  - Trains faster (~1.5 hours vs 3 hours)"
echo "  - Costs less (~$2.35 vs $4.71)"
echo "  - Fits within your $2.44 remaining credit!"
echo ""

# Kill stuck processes
echo "1. Cleaning up..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting training with Small config..."
echo ""

tmux new -s training -d "
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 examples/train_digitalocean.py \
  --config small \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 2 \
  --save_steps 1000 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_small.log
"

echo ""
echo "✅ Training started!"
echo ""
echo "Model specs:"
echo "  - Parameters: 216M (manageable size)"
echo "  - Context: 8K tokens (truncated from 32K chunks)"
echo "  - Batch size: 64"
echo "  - Gradient accumulation: 2 (effective batch = 128)"
echo "  - VRAM usage: ~10-15GB (very safe!)"
echo "  - Training time: ~1.5 hours"
echo "  - Cost: ~$2.35 (within your $2.44 budget!)"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_small.log"
echo ""
echo "Training should start within 10 seconds!"
