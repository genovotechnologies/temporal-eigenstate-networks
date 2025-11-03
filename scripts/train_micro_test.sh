#!/bin/bash

echo "🔬 MICRO TEST - Memory Debugging"
echo "================================="
echo ""
echo "✅ ALL fixes applied:"
echo "  1. Cell averaging (not concatenation) = 4× less memory"
echo "  2. FFN expansion 2× (not 4×) = 2× less memory"
echo "  3. NO state tracking during training = MASSIVE savings!"
echo "  4. NO gradient checkpointing (was causing double forward!)"
echo "  5. Aggressive garbage collection every 10 steps"
echo "  6. Explicit tensor deletion after each batch"
echo ""
echo "Micro Config (should definitely work):"
echo "  - ~100M parameters"
echo "  - 4K context"
echo "  - Batch size: 16"
echo "  - Expected VRAM: ~4-6GB MAX"
echo ""

# Kill stuck processes
echo "1. Cleaning up..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo "2. Starting MICRO test..."
echo ""

tmux new -s training -d "
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 examples/train_digitalocean.py \
  --config micro \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 16 \
  --save_steps 1500 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee /root/ten_workspace/logs/training_micro_test.log
"

echo ""
echo "✅ MICRO test started!"
echo ""
echo "Expected behavior:"
echo "  - VRAM usage: ~4-6GB MAX (not 43GB!)"
echo "  - Training starts immediately"
echo "  - No OOM errors"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  watch -n 1 nvidia-smi"
echo "  tail -f ~/ten_workspace/logs/training_micro_test.log"
echo ""
echo "If this WORKS: We can scale up gradually"
echo "If this FAILS: Something fundamentally broken beyond architecture"
