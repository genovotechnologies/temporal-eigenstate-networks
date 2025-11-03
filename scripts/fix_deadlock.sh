#!/bin/bash

echo "🔧 Fixing DataLoader Deadlock Issue"
echo "===================================="
echo ""

# Kill stuck training
echo "1. Killing stuck training processes..."
tmux kill-session -t training 2>/dev/null || true
pkill -f "train_digitalocean.py" 2>/dev/null || true
sleep 2

# Clear CUDA cache
echo "2. Clearing GPU memory..."
nvidia-smi --gpu-reset 2>/dev/null || true

echo "3. Restarting training with FIXED settings..."
echo ""
echo "Changes applied:"
echo "  ✓ Changed multiprocessing from 'fork' to 'spawn' (prevents deadlock)"
echo "  ✓ Reduced prefetch_factor from 16 to 4 (reduces memory pressure)"
echo "  ✓ Added --no_compile option to disable torch.compile if needed"
echo ""
echo "Starting training in tmux session..."

cd /root/temporal-eigenstate-networks

# Start training with fixed settings
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
  2>&1 | tee /root/ten_workspace/logs/training_fixed.log
"

echo ""
echo "✅ Training restarted!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t training"
echo "  tail -f ~/ten_workspace/logs/training_fixed.log"
echo ""
echo "If still hangs, try with torch.compile disabled:"
echo "  Add --no_compile flag to the command"
