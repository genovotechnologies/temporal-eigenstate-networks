# GPU Training Guide - Quick Start

## 🚀 Quick Commands

### Option 1: Direct Python Command (Recommended for Testing)
```bash
# Navigate to the repository
cd /path/to/temporal-eigenstate-networks

# Run with GPU (automatic detection)
python examples/train_digitalocean.py \
    --config small \
    --dataset finewebedu \
    --epochs 2 \
    --mixed_precision \
    --output_dir ./runs/test_run
```

### Option 2: Using Shell Scripts (Recommended for Production)
```bash
# Navigate to the repository
cd /path/to/temporal-eigenstate-networks

# Make scripts executable
chmod +x scripts/*.sh

# Run small config (216M params, 8K context, ~10-15GB VRAM)
./scripts/train_small.sh

# Or run medium config (850M params, 16K context, ~25-30GB VRAM)
./scripts/train_medium.sh

# Or beast mode (1.8B+ params, 32K context, requires 40GB+ VRAM)
./scripts/beast_mode.sh
```

### Option 3: Using tmux (Recommended for Long Training)
```bash
# Start a tmux session
tmux new -s training

# Inside tmux, run training
cd /path/to/temporal-eigenstate-networks
python examples/train_digitalocean.py \
    --config medium \
    --dataset finewebedu \
    --epochs 3 \
    --mixed_precision \
    --gradient_accumulation 2 \
    --output_dir ./runs/production

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
# View logs: tail -f ./runs/production/training.log
```

## 📋 Prerequisites

### 1. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Or install the package
pip install -e .
```

### 2. Verify GPU is Available
```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
GPU Available: True
GPU Name: NVIDIA L40S (or your GPU model)
```

## 🎯 Configuration Options

### Available Configs

| Config | Params | Context | VRAM | Training Time | Use Case |
|--------|--------|---------|------|---------------|----------|
| **micro** | 25M | 2K | ~2-4GB | 15 min | Quick testing |
| **nano** | 50M | 4K | ~4-6GB | 30 min | Development |
| **small** | 216M | 8K | ~10-15GB | 1.5 hours | Most efficient |
| **medium** | 850M | 16K | ~25-30GB | 3-4 hours | Production ⭐ |
| **large** | 1.8B | 32K | ~40-45GB | 6-8 hours | Beast mode 🔥 |

### Command Line Arguments

```bash
python examples/train_digitalocean.py \
    --config medium \              # Model size (see table above)
    --dataset finewebedu \         # Dataset name (finewebedu, openwebtext, wikitext)
    --pretokenized \               # Use pretokenized data (faster)
    --tokenized_dir /path/to/data \  # Path to tokenized data
    --epochs 3 \                   # Number of training epochs
    --batch_size 32 \              # Batch size (auto-adjusted per config)
    --gradient_accumulation 2 \    # Gradient accumulation steps
    --learning_rate 3e-4 \         # Learning rate
    --mixed_precision \            # Enable AMP (2× speedup)
    --save_steps 1000 \            # Save checkpoint every N steps
    --eval_steps 500 \             # Evaluate every N steps
    --output_dir ./runs/my_run \   # Output directory
    --num_workers 4 \              # DataLoader workers
    --no_compile                   # Disable torch.compile (for compatibility)
```

## 🔥 Recommended Configurations by GPU

### For 16GB GPU (RTX 4080, V100 16GB)
```bash
python examples/train_digitalocean.py \
    --config small \
    --dataset finewebedu \
    --epochs 2 \
    --mixed_precision \
    --gradient_accumulation 2 \
    --batch_size 32
```

### For 24GB GPU (RTX 3090, RTX 4090, A5000)
```bash
python examples/train_digitalocean.py \
    --config medium \
    --dataset finewebedu \
    --epochs 2 \
    --mixed_precision \
    --gradient_accumulation 2 \
    --batch_size 32
```

### For 40GB+ GPU (A100, L40S, RTX 6000 Ada)
```bash
python examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --epochs 2 \
    --mixed_precision \
    --gradient_accumulation 4 \
    --batch_size 16
```

## 📊 Monitoring Training

### View Live Training Progress
```bash
# If using tmux
tmux attach -t training

# Or tail the log file
tail -f ./runs/my_run/training.log

# Or use watch with nvidia-smi
watch -n 1 nvidia-smi
```

### Check GPU Usage
```bash
# Live GPU monitoring
nvidia-smi -l 1

# Or more detailed
watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'
```

### TensorBoard (if enabled)
```bash
tensorboard --logdir ./runs --port 6006

# Then open in browser: http://localhost:6006
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 16  # or 8, or 4

# Increase gradient accumulation
--gradient_accumulation 4  # or 8

# Use smaller config
--config small  # instead of medium

# Enable gradient checkpointing (if needed)
# Edit config in train_digitalocean.py:
# use_gradient_checkpointing=True (though currently disabled for stability)
```

### Slow Training
```bash
# Enable mixed precision (2× speedup)
--mixed_precision

# Increase number of workers
--num_workers 8  # match your CPU cores

# Use pretokenized data
--pretokenized --tokenized_dir /path/to/tokenized

# Ensure TF32 is enabled (automatic on Ampere+ GPUs)
```

### Training Hangs or Deadlocks
```bash
# Use single worker
--num_workers 0

# Or run the diagnostic script
./scripts/diagnose_hang.sh

# Or use single process mode
./scripts/train_singleprocess.sh
```

## 📈 Performance Optimization Tips

### 1. Use Mixed Precision
Enables automatic mixed precision (AMP) for 2× speedup:
```bash
--mixed_precision
```

### 2. Optimize Batch Size
Find the largest batch size that fits in your VRAM:
```bash
# Start small and increase
--batch_size 8   # Test
--batch_size 16  # Increase
--batch_size 32  # Keep increasing until OOM
--batch_size 24  # Then back off slightly
```

### 3. Use Gradient Accumulation
Simulate larger batch sizes without more VRAM:
```bash
# Effective batch size = batch_size × gradient_accumulation
--batch_size 16 --gradient_accumulation 4  # Effective: 64
```

### 4. Pretokenize Your Data
Tokenize once, train faster:
```bash
# Pretokenize (one-time)
python scripts/pretokenize_and_pack.py \
    --dataset finewebedu \
    --output_dir ./tokenized_data

# Then train
python examples/train_digitalocean.py \
    --pretokenized \
    --tokenized_dir ./tokenized_data
```

## 🎓 Complete Training Example

Here's a full end-to-end example:

```bash
# 1. Clone and setup
cd ~
git clone https://github.com/genovotechnologies/temporal-eigenstate-networks.git
cd temporal-eigenstate-networks

# 2. Install dependencies
pip install -e .

# 3. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# 4. Create output directories
mkdir -p runs logs tokenized_data

# 5. Start training in tmux
tmux new -s training
python examples/train_digitalocean.py \
    --config medium \
    --dataset finewebedu \
    --epochs 3 \
    --mixed_precision \
    --gradient_accumulation 2 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --save_steps 1000 \
    --eval_steps 500 \
    --output_dir ./runs/$(date +%Y%m%d_%H%M%S) \
    --num_workers 4 \
    2>&1 | tee ./logs/training_$(date +%Y%m%d_%H%M%S).log

# 6. Detach from tmux (Ctrl+B then D)
# 7. Monitor from another terminal
watch -n 1 nvidia-smi

# 8. Reattach to see progress
tmux attach -t training
```

## 📁 Output Structure

After training, you'll have:
```
runs/
└── 20251104_143022/
    ├── checkpoint_1000.pt       # Model checkpoint
    ├── checkpoint_2000.pt
    ├── best_model.pt            # Best model by validation loss
    ├── final_model.pt           # Final model
    ├── config.json              # Training config
    ├── training.log             # Training logs
    └── tensorboard/             # TensorBoard logs (if enabled)
```

## ⚡ Quick Start (TL;DR)

```bash
# Fastest way to start training on your GPU:
cd temporal-eigenstate-networks
pip install -e .
python examples/train_digitalocean.py --config medium --dataset finewebedu --epochs 2 --mixed_precision
```

That's it! Training will start and you'll see progress bars and GPU utilization metrics.

---

**Need Help?**
- Check logs: `tail -f ./runs/*/training.log`
- Monitor GPU: `nvidia-smi -l 1`
- Troubleshooting: See `docs/TRAINING_QUICKSTART.md`
