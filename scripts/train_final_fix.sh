#!/bin/bash

echo "🚀 Starting Training with MULTIPROCESSING FIX"
echo "============================================="
echo ""
echo "✓ Critical fix applied: mp.set_start_method('spawn') before CUDA init"
echo "✓ This fixes the 'bootstrapping phase' error"
echo "✓ Workers should spawn in ~12 seconds"
echo ""

# Kill stuck processes
echo "1. Cleaning up stuck processes..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting training..."
echo ""

tmux new -s training -d "
CUDA_VISIBLE_DEVICES=0 \
python3 examples/train_digitalocean.py \
  --config large \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 4 \
  --save_steps 2500 \
  --num_workers 4 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_final.log
"

echo ""
echo "✅ Training started!"
echo ""
echo "Expected behavior:"
echo "  - Workers spawn: ~12 seconds (be patient!)"
echo "  - First batch: ~20-30 seconds"
echo "  - Then: ~8s per batch"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_final.log"
echo ""
echo "You should see progress within 30 seconds!"
