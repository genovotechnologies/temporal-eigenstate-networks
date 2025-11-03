#!/bin/bash

echo "🔬 Running DataLoader diagnostic tests..."
echo "=========================================="
echo ""

# Kill any stuck processes first
echo "1. Cleaning up stuck processes..."
tmux kill-session -t training 2>/dev/null || true
pkill -f train_digitalocean.py 2>/dev/null || true
sleep 2

cd /root/temporal-eigenstate-networks

echo ""
echo "2. Testing chunk loading directly..."
python3 test_chunk_loading.py

echo ""
echo "3. Testing DataLoader in isolation..."
timeout 60 python3 test_dataloader.py

echo ""
echo "=========================================="
echo "✅ Diagnostic complete!"
echo ""
echo "If tests pass but training still hangs, the issue is in:"
echo "  - Model forward pass"
echo "  - torch.compile() interaction"
echo "  - Mixed precision setup"
