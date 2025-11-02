#!/bin/bash
# Restart training with optimized batch sizes that actually use your 48GB GPU!

echo "🛑 Killing current training (underutilizing GPU)..."
pkill -f train_digitalocean.py

echo "⏳ Waiting for process to stop..."
sleep 3

echo "🧹 Clearing GPU memory..."
nvidia-smi --gpu-reset || true

echo ""
echo "📊 GPU Status Before:"
nvidia-smi

echo ""
echo "="*80
echo "🚀 RESTARTING TRAINING WITH MASSIVE MODEL"
echo "="*80
echo ""
echo "🔥 OLD CONFIG (pathetic):"
echo "  Model: 164M parameters"
echo "  Batch: 16"
echo "  Context: 8K tokens"
echo "  GPU usage: 1.2GB / 48GB (2.5%)"
echo ""
echo "💪 NEW CONFIG (BEAST MODE):"
echo "  Model: 1.8 BILLION parameters"
echo "  Batch: 16 (with 32K context!)"
echo "  Context: 32,768 tokens (4× longer!)"
echo "  GPU usage: ~35-40GB / 48GB (80-85%)"
echo ""
echo "Expected improvements:"
echo "  - 11× more parameters (164M → 1.8B)"
echo "  - 4× longer context (8K → 32K)"
echo "  - 30× more GPU memory used"
echo "  - REAL billion-parameter model!"
echo ""

cd /root/temporal-eigenstate-networks
source /root/ten_venv/bin/activate

# Get the updated script from repo
git pull origin main || echo "⚠️  Could not pull latest - using local version"

echo "Starting BEAST MODE training in 3 seconds..."
sleep 3

# Run with MASSIVE config
python3 examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --epochs 2 \
    --mixed_precision \
    --save_steps 2500 \
    --max_seq_len 32768 \
    --gradient_accumulation 2

echo ""
echo "✅ Training restarted with optimized configuration!"
