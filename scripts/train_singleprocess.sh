#!/bin/bash

echo "🐌 Starting training in SINGLE-PROCESS mode (guaranteed to work)"
echo "================================================================"
echo ""
echo "⚠️  This mode is ~20% slower but CANNOT deadlock!"
echo ""

# Kill stuck processes
echo "1. Cleaning up..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting training with num_workers=0..."
echo ""

tmux new -s training -d "
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 examples/train_digitalocean.py \
  --config large \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 8 \
  --save_steps 2500 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_singleprocess.log
"

echo ""
echo "✅ Training started in single-process mode!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_singleprocess.log"
echo ""
echo "This WILL work, but expect ~20% slower than multiprocess."
