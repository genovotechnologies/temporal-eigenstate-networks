#!/bin/bash
#
# DigitalOcean GPU Droplet Setup Script for TEN Training
# GPU: L40S or RTX 6000 Ada (48GB VRAM)
# Cost: $1.57/hour (~$7.85 for 5 hours)
#
# This script sets up your DigitalOcean GPU droplet for training
# Temporal Eigenstate Networks on long-range tasks
#

set -e  # Exit on error

echo "=========================================="
echo "TEN Training Setup for DigitalOcean GPU"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. Check GPU availability
echo "1. Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please ensure you're on a GPU droplet."
    exit 1
fi

nvidia-smi
echo ""

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
print_status "GPU detected: $GPU_NAME"
print_status "VRAM: $GPU_MEMORY"
echo ""

# Verify we have the expected GPU
if [[ $GPU_NAME == *"L40S"* ]] || [[ $GPU_NAME == *"RTX 6000"* ]]; then
    print_status "Correct GPU detected! Ready for training."
else
    print_warning "Unexpected GPU: $GPU_NAME"
    print_warning "This script is optimized for L40S or RTX 6000 Ada"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. System updates
echo "2. Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl htop nvtop tmux vim
print_status "System packages updated"
echo ""

# 3. Check Python
echo "3. Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "$PYTHON_VERSION found"
else
    print_error "Python 3 not found. Installing..."
    sudo apt-get install -y python3 python3-pip python3-venv
fi
echo ""

# 4. Check CUDA
echo "4. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_status "CUDA $CUDA_VERSION found"
else
    print_warning "CUDA toolkit not found in PATH"
    print_warning "PyTorch will use bundled CUDA libraries"
fi
echo ""

# 5. Setup Python virtual environment
echo "5. Setting up Python virtual environment..."
VENV_DIR="$HOME/ten_venv"
if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $VENV_DIR
        python3 -m venv $VENV_DIR
    fi
else
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
print_status "Virtual environment activated: $VENV_DIR"
echo ""

# 6. Install PyTorch with CUDA support
echo "6. Installing PyTorch with CUDA support..."
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
print_status "PyTorch installed"

# Verify PyTorch can see GPU
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF
echo ""

# 7. Clone TEN repository
echo "7. Setting up TEN repository..."
REPO_DIR="$HOME/temporal-eigenstate-networks"

if [ -d "$REPO_DIR" ]; then
    print_warning "Repository already exists at $REPO_DIR"
    read -p "Pull latest changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd $REPO_DIR
        git pull
        print_status "Repository updated"
    fi
else
    print_warning "Repository will be cloned. If private, you'll need authentication."
    cd $HOME
    
    # Check if we need authentication
    read -p "Is the repository private? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Please enter your GitHub Personal Access Token:"
        read -s GITHUB_TOKEN
        echo ""
        git clone https://${GITHUB_TOKEN}@github.com/genovotechnologies/temporal-eigenstate-networks.git
    else
        git clone https://github.com/genovotechnologies/temporal-eigenstate-networks.git
    fi
    print_status "Repository cloned to $REPO_DIR"
fi
echo ""

# 8. Install TEN package
echo "8. Installing TEN package and dependencies..."
cd $REPO_DIR
pip install -e . -q
pip install datasets transformers wandb accelerate -q
print_status "TEN package installed"
echo ""

# 9. Test installation
echo "9. Testing TEN installation..."
python3 << EOF
import sys
sys.path.insert(0, 'src')
from model import TemporalEigenstateConfig, TemporalEigenstateNetwork
import torch

config = TemporalEigenstateConfig(
    d_model=256,
    n_layers=4,
    num_eigenstates=64,
    vocab_size=50000
)
model = TemporalEigenstateNetwork(config)
print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Test on GPU
if torch.cuda.is_available():
    model = model.cuda()
    x = torch.randint(0, 50000, (2, 128)).cuda()
    with torch.no_grad():
        y = model(x)
    print(f"✓ GPU forward pass successful: {x.shape} -> {y.shape}")
EOF
print_status "Installation test passed!"
echo ""

# 10. Create workspace directory
echo "10. Setting up workspace..."
WORKSPACE_DIR="$HOME/ten_workspace"
mkdir -p $WORKSPACE_DIR/{data,checkpoints,logs,results}
print_status "Workspace created at $WORKSPACE_DIR"
echo ""

# 11. Download sample dataset (optional)
echo "11. Dataset preparation..."
read -p "Download sample dataset (IMDb, ~80MB)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 << EOF
from datasets import load_dataset
print("Downloading IMDb dataset...")
dataset = load_dataset("imdb")
print(f"✓ Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
EOF
    print_status "Dataset ready"
else
    print_warning "Skipped dataset download. You can download later."
fi
echo ""

# 12. Create quick-start training script
echo "12. Creating training scripts..."
cat > $WORKSPACE_DIR/train_large_model.sh << 'SCRIPT_EOF'
#!/bin/bash
# Quick-start training script for large model

source $HOME/ten_venv/bin/activate
cd $HOME/temporal-eigenstate-networks

python3 << EOF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import sys
sys.path.insert(0, 'src')
from model import TemporalEigenstateConfig, TemporalEigenstateNetwork
from train import Trainer

print("="*80)
print("Training Large TEN Model on DigitalOcean GPU")
print("="*80)

# Configuration for 48GB GPU
config = TemporalEigenstateConfig(
    d_model=1024,           # Large model
    n_heads=16,
    n_layers=8,
    d_ff=4096,
    max_seq_len=8192,       # Long sequences
    num_eigenstates=128,
    dropout=0.1,
    vocab_size=50257,
)

print(f"\nModel Configuration:")
print(f"  Hidden dim: {config.d_model}")
print(f"  Layers: {config.n_layers}")
print(f"  Max sequence: {config.max_seq_len}")
print(f"  Eigenstates: {config.num_eigenstates}")

# Create model
device = torch.device('cuda')
model = TemporalEigenstateNetwork(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ Model created: {num_params:,} parameters ({num_params/1e6:.1f}M)")

# Load dataset (using subset for demo)
print("\nLoading dataset...")
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Setup training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print("\n✓ Ready to train!")
print("Estimated training time: 4-5 hours")
print("Cost: ~$7.85 on DigitalOcean")
EOF
SCRIPT_EOF

chmod +x $WORKSPACE_DIR/train_large_model.sh
print_status "Training script created: $WORKSPACE_DIR/train_large_model.sh"
echo ""

# 13. Create monitoring script
cat > $WORKSPACE_DIR/monitor.sh << 'MONITOR_EOF'
#!/bin/bash
# Monitor GPU usage during training

watch -n 1 "nvidia-smi; echo ''; echo 'Press Ctrl+C to exit'"
MONITOR_EOF

chmod +x $WORKSPACE_DIR/monitor.sh
print_status "Monitoring script created: $WORKSPACE_DIR/monitor.sh"
echo ""

# 14. Create cost tracker
cat > $WORKSPACE_DIR/cost_tracker.py << 'COST_EOF'
#!/usr/bin/env python3
"""Track training cost on DigitalOcean GPU"""
import time
import datetime

HOURLY_RATE = 1.57  # DigitalOcean L40S/RTX 6000 rate
FREE_CREDIT = 15.00

print("="*60)
print("DigitalOcean GPU Cost Tracker")
print("="*60)
print(f"Hourly rate: ${HOURLY_RATE}/hour")
print(f"Free credit: ${FREE_CREDIT}")
print("="*60)

start_time = time.time()
print(f"\nStarted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nPress Ctrl+C to stop and see total cost")

try:
    while True:
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        cost = hours * HOURLY_RATE
        remaining = FREE_CREDIT - cost
        
        print(f"\rElapsed: {elapsed/60:.1f} min | Cost: ${cost:.2f} | Remaining credit: ${remaining:.2f}", end='', flush=True)
        time.sleep(1)
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    cost = hours * HOURLY_RATE
    remaining = FREE_CREDIT - cost
    
    print("\n" + "="*60)
    print("Training Session Summary")
    print("="*60)
    print(f"Total time: {elapsed/60:.1f} minutes ({hours:.2f} hours)")
    print(f"Total cost: ${cost:.2f}")
    print(f"Remaining credit: ${remaining:.2f}")
    print("="*60)
COST_EOF

chmod +x $WORKSPACE_DIR/cost_tracker.py
print_status "Cost tracker created: $WORKSPACE_DIR/cost_tracker.py"
echo ""

# 15. Setup tmux session
echo "15. Setting up tmux for persistent training..."
cat > $HOME/.tmux.conf << 'TMUX_EOF'
# TEN Training tmux configuration
set -g mouse on
set -g history-limit 10000
set -g status-bg colour235
set -g status-fg colour255
TMUX_EOF
print_status "tmux configured"
echo ""

# 16. Create README with instructions
cat > $WORKSPACE_DIR/README.md << 'README_EOF'
# TEN Training on DigitalOcean GPU

## Quick Start

### 1. Start a tmux session (recommended for long training)
```bash
tmux new -s ten_training
```

### 2. Activate environment
```bash
source $HOME/ten_venv/bin/activate
cd $HOME/temporal-eigenstate-networks
```

### 3. Monitor GPU (in separate tmux pane)
```bash
# Split tmux pane: Ctrl+b then "
./monitor.sh
```

### 4. Track costs (in another pane)
```bash
# Split again: Ctrl+b then "
python3 cost_tracker.py
```

### 5. Start training
```bash
python3 examples/your_training_script.py
```

## Tmux Shortcuts

- `Ctrl+b "` - Split pane horizontally
- `Ctrl+b %` - Split pane vertically
- `Ctrl+b arrow` - Navigate between panes
- `Ctrl+b d` - Detach from session (keeps running)
- `tmux attach -t ten_training` - Reattach to session

## GPU Specs

- **GPU**: L40S or RTX 6000 Ada
- **VRAM**: 48 GB
- **Cost**: $1.57/hour
- **Your budget**: $15 free credit = ~9.5 hours

## Estimated Times

- Small model (256 dim, 4 layers): ~1 hour
- Medium model (512 dim, 6 layers): ~2-3 hours
- Large model (1024 dim, 8 layers): ~4-5 hours

## Tips

1. **Use tmux**: Your training continues even if you disconnect
2. **Monitor costs**: Run cost_tracker.py to stay within budget
3. **Save checkpoints**: Models save to `$HOME/ten_workspace/checkpoints`
4. **Watch GPU**: Run `nvidia-smi` or `nvtop` to monitor usage

## Getting Results

Results saved to: `$HOME/ten_workspace/results/`
Checkpoints saved to: `$HOME/ten_workspace/checkpoints/`

## Stopping Early

If you need to stop training:
1. In tmux: `Ctrl+C` to stop script
2. Check cost: `python3 cost_tracker.py` 
3. Download results before destroying droplet!

## Downloading Results

```bash
# From your local machine:
scp -r root@your-droplet-ip:~/ten_workspace/results ./
scp -r root@your-droplet-ip:~/ten_workspace/checkpoints ./
```
README_EOF

print_status "Instructions saved to $WORKSPACE_DIR/README.md"
echo ""

# Final summary
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "GPU Information:"
echo "  Name: $GPU_NAME"
echo "  VRAM: $GPU_MEMORY"
echo "  Cost: \$1.57/hour"
echo "  Your budget: \$15.00 = ~9.5 hours"
echo ""
echo "Environment:"
echo "  Virtual env: $VENV_DIR"
echo "  Repository: $REPO_DIR"
echo "  Workspace: $WORKSPACE_DIR"
echo ""
echo "Next Steps:"
echo "  1. Start tmux session:"
echo "     tmux new -s ten_training"
echo ""
echo "  2. Activate environment:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  3. Monitor GPU:"
echo "     ./monitor.sh"
echo ""
echo "  4. Track costs:"
echo "     python3 cost_tracker.py"
echo ""
echo "  5. Start training:"
echo "     cd $REPO_DIR"
echo "     python3 examples/your_script.py"
echo ""
echo "  6. Read instructions:"
echo "     cat $WORKSPACE_DIR/README.md"
echo ""
print_status "Happy training! 🚀"
echo ""
print_warning "Remember: Your \$15 credit gives you ~9.5 hours of GPU time"
print_warning "5 hours for training + 4.5 hours for experiments!"
echo ""
