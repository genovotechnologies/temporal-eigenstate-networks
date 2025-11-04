#!/bin/bash

echo "🚀 EFFICIENT SMALL TRAINING"
echo "============================="
echo ""
echo "REALISTIC MEMORY USAGE:"
echo "  - ~130M parameters"
echo "  - 512 context"
echo "  - Batch size: 16"
echo "  - Gradient accum: 2"
echo "  - Expected: ~8-10GB VRAM"
echo ""

# Kill stuck processes
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "Starting training..."
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
  --batch_size 16 \
  --max_seq_len 512 \
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
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_small.log"
echo ""
