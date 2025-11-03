#!/bin/bash

echo "🔥 Starting Training - LARGE CONFIG with TINY BATCH SIZE"
echo "========================================================="
echo ""
echo "Strategy: Reduce batch_size from 16 to 4"
echo "  - Keeps large 1.4B model with 32K context"  
echo "  - Uses gradient_accumulation=16 for effective batch_size=64"
echo "  - Should fit in ~35-40GB VRAM"
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
  --config large \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 16 \
  --save_steps 1000 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  --max_seq_len 16384 \
  2>&1 | tee /root/ten_workspace/logs/training_large_reduced.log
"

echo ""
echo "✅ Training started!"
echo ""
echo "Config:"
echo "  - Model: Large (1.4B params)"
echo "  - Sequence length: 16K (truncated from 32K chunks)"
echo "  - Batch size per step: 4 (reduced from 16)"
echo "  - Gradient accumulation: 16 steps"
echo "  - Effective batch size: 64 samples"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_large_reduced.log"
echo ""
echo "If this OOMs, we'll switch to medium config instead."
