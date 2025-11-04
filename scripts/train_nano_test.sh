#!/bin/bash

echo "🔬 NANO TEST - Absolute Minimum Memory"
echo "========================================"
echo ""
echo "Testing with SMALLEST possible config:"
echo "  - 25M parameters (4 layers, 512 dim)"
echo "  - 512 context (half of micro)"
echo "  - Batch size: 8 (quarter of micro)"
echo "  - Chunk size: 16 timesteps (aggressive detaching)"
echo ""
echo "Expected VRAM: <2GB forward + <2GB backward = <4GB total"
echo "If this FAILS: Architecture fundamentally broken"
echo ""

# Kill stuck processes
echo "1. Cleaning up..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting NANO test..."
echo ""

tmux new -s training -d "
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 examples/train_digitalocean.py \
  --config nano \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 32 \
  --save_steps 10000 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_nano_test.log
"

echo ""
echo "✅ NANO test started!"
echo ""
echo "Memory tracking enabled - watch for:"
echo "  Forward: Should be <2GB"
echo "  Loss: Should be <2GB" 
echo "  Backward: Should be <2GB"
echo "  TOTAL: Should be <4GB"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_nano_test.log"
echo ""
echo "This is the FINAL test. If this uses >4GB, we need to:"
echo "1. Remove output projection (use smaller vocab)"
echo "2. Use activation checkpointing library"
echo "3. Completely rewrite temporal loop"
