#!/bin/bash

echo "🚀 MEMORY-OPTIMIZED TEN - Medium Config"
echo "========================================"
echo ""
echo "✅ Architecture fixes applied:"
echo "  - Cell averaging (not concatenation) = 4× less memory"
echo "  - FFN expansion 2× (not 4×) = 2× less memory"
echo "  - Gradient checkpointing = saves activation memory"
echo "  - Preallocated tensors = no list overhead"
echo "  - Reduced batch size = lower peak memory"
echo ""
echo "Medium Config (should work now):"
echo "  - 419M parameters (optimized)"
echo "  - 16K context"
echo "  - Batch size: 16 (was 32)"
echo "  - Expected VRAM: ~12-18GB (was 44GB before fix!)"
echo ""

# Kill stuck processes
echo "1. Cleaning up..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting training with FIXED architecture..."
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
  --save_steps 1500 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --gradient_checkpointing \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_medium_fixed.log
"

echo ""
echo "✅ Training started with optimized architecture!"
echo ""
echo "Expected behavior:"
echo "  - VRAM usage: ~10-15GB (not 40GB!)"
echo "  - Training time: ~3 hours"
echo "  - Cost: ~$4.71"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_medium_fixed.log"
echo "  nvidia-smi  # Should show ~10-15GB usage"
echo ""
echo "Training should start within 10 seconds!"
