#!/bin/bash
# Quick commands to restart with MASSIVE configs

echo "="*80
echo "🔥 BEAST MODE CONFIGS - ACTUALLY USE YOUR 48GB GPU!"
echo "="*80
echo ""

cat << 'EOF'

YOUR CURRENT TRAINING IS PATHETIC:
  ❌ 164M parameters (should be 1-3 BILLION)
  ❌ 1.2GB GPU usage (should be 35-45GB)
  ❌ 8K context (should be 32K)
  ❌ Only 2.5% GPU utilization

NEW CONFIGS AVAILABLE:

1. SMALL (350M params, 8K context)
   python3 examples/train_digitalocean.py \
       --config small \
       --dataset finewebedu \
       --epochs 2 \
       --mixed_precision

2. MEDIUM (850M params, 16K context) ⭐ RECOMMENDED
   python3 examples/train_digitalocean.py \
       --config medium \
       --dataset finewebedu \
       --epochs 2 \
       --mixed_precision \
       --gradient_accumulation 2

3. LARGE (1.8B params, 32K context) 🔥 BEAST MODE
   python3 examples/train_digitalocean.py \
       --config large \
       --dataset finewebedu \
       --epochs 1 \
       --mixed_precision \
       --gradient_accumulation 4

4. XLARGE (3.2B params, 32K context) 💀 MAXIMUM
   python3 examples/train_digitalocean.py \
       --config xlarge \
       --dataset finewebedu \
       --epochs 1 \
       --mixed_precision \
       --gradient_accumulation 8

WHAT TO DO NOW:

1. Kill current training (it's wasting time):
   pkill -f train_digitalocean.py

2. Update the code:
   cd /root/temporal-eigenstate-networks
   git pull

3. Start BEAST MODE:
   tmux attach -t training
   # Then run one of the commands above

RECOMMENDED: Medium config (850M params, 16K context)
- Uses ~25-30GB GPU
- Still fast enough for 4-5 hours
- 16K context is excellent
- 850M params is respectable

Or go FULL BEAST: Large config (1.8B params, 32K context)
- Uses ~35-40GB GPU
- Takes longer but MUCH better model
- 32K context = handles books/long docs
- 1.8B params = competitive with GPT-2 XL

EOF

echo ""
echo "Run this script to see options, then kill training and restart!"
