#!/bin/bash
# Training launch script for DigitalOcean L40S GPU
# Optimized for 1.8B parameter model with 32K context

# Create logs directory if missing
mkdir -p ~/ten_workspace/logs

# Stop any previous training session just in case
tmux kill-session -t training 2>/dev/null || true

# Launch training in a detached tmux session with logging
tmux new -s training -d 'python3 examples/train_digitalocean.py \
  --config large \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --mixed_precision \
  --gradient_accumulation 4 \
  --save_steps 2500 \
  --num_workers 6 \
  --learning_rate 3e-4 \
  --output_dir /root/ten_workspace 2>&1 | tee /root/ten_workspace/logs/training.log'

echo "✅ Training launched in tmux session 'training'"
echo ""
echo "📊 Monitor progress:"
echo "  tmux attach -t training          # Attach to session"
echo "  tail -f ~/ten_workspace/logs/training.log  # Watch logs"
echo ""
echo "🔍 Check GPU usage:"
echo "  watch -n 1 nvidia-smi            # Real-time GPU stats"
echo ""
echo "⏹️  Stop training:"
echo "  tmux kill-session -t training    # Kill session"
