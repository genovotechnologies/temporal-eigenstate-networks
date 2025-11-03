#!/bin/bash
# Quick restart script after fixing PyTorch compatibility issues

echo "🔄 Restarting training with fixed PyTorch compatibility..."

# Stop any existing session
tmux kill-session -t training 2>/dev/null || true

# Create logs directory
mkdir -p ~/ten_workspace/logs

# Launch training with corrected code
tmux new -s training -d "cd /root/temporal-eigenstate-networks && python3 examples/train_digitalocean.py \
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
  --output_dir /root/ten_workspace 2>&1 | tee /root/ten_workspace/logs/training.log"

echo ""
echo "✅ Training restarted with fixed code!"
echo ""
echo "📊 Monitor commands:"
echo "  tmux attach -t training                          # Attach to session"
echo "  tail -f ~/ten_workspace/logs/training.log        # Watch logs"
echo "  watch -n 1 nvidia-smi                            # GPU usage"
echo ""
echo "🔥 Training 1.8B parameter model on 10B tokens!"
echo "   Expected: ~2 hours, ~$3 cost"
