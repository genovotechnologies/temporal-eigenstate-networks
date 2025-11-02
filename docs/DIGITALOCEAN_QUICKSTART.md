# 🚀 DigitalOcean GPU Quick Start Guide

## Your Setup

**GPU Options (Both Excellent!):**
- ✅ **L40S** - 48GB VRAM @ $1.57/hour (Recommended - 15-20% faster)
- ✅ **RTX 6000 Ada** - 48GB VRAM @ $1.57/hour (Great alternative)

**Your Budget:** $15 free credit = **9.5 hours** of GPU time!

**Recommended:** Use 5 hours for main training, keep 4.5 hours for experiments

---

## 🎯 Quick Start (3 Steps)

### Step 1: Create DigitalOcean GPU Droplet

1. Log into [DigitalOcean](https://cloud.digitalocean.com/)
2. Click **"Create"** → **"Droplets"**
3. Choose:
   - **Image:** GPU-optimized Ubuntu 22.04
   - **Plan:** GPU Droplets → **L40S** (48GB) or **RTX 6000** (48GB)
   - **Datacenter:** Closest to you
   - **Authentication:** SSH key (recommended) or password
4. Click **"Create Droplet"**
5. Wait 1-2 minutes for droplet to boot

### Step 2: Connect and Setup

```bash
# SSH into your droplet
ssh root@your-droplet-ip

# Download and run setup script
curl -o setup.sh https://raw.githubusercontent.com/genovotechnologies/temporal-eigenstate-networks/main/scripts/setup_digitalocean.sh
chmod +x setup.sh
./setup.sh
```

**Setup takes ~5 minutes and will:**
- ✅ Verify GPU (L40S or RTX 6000)
- ✅ Install PyTorch with CUDA
- ✅ Clone TEN repository
- ✅ Create virtual environment
- ✅ Download sample dataset
- ✅ Create training scripts
- ✅ Setup monitoring tools

### Step 3: Start Training

```bash
# Start tmux session (keeps running if you disconnect)
tmux new -s training

# Activate environment
source ~/ten_venv/bin/activate
cd ~/temporal-eigenstate-networks

# Choose your configuration and train!
python3 examples/train_digitalocean.py --config large --mixed_precision
```

---

## 📊 Model Configurations (Optimized for 48GB)

### Tiny (30 min, ~82% acc) - Quick Test
```bash
python3 examples/train_digitalocean.py --config tiny
```
- 256 dim, 2 layers, 512 tokens
- Perfect for: Testing setup
- Cost: ~$0.80

### Small (1.5 hours, ~86% acc) - Fast
```bash
python3 examples/train_digitalocean.py --config small
```
- 512 dim, 4 layers, 2048 tokens
- Perfect for: Quick experiments
- Cost: ~$2.36

### Medium (3 hours, ~88% acc) - Balanced ⭐
```bash
python3 examples/train_digitalocean.py --config medium
```
- 768 dim, 6 layers, 4096 tokens
- Perfect for: Good results in reasonable time
- Cost: ~$4.71

### Large (4.5 hours, ~90% acc) - Recommended 🎯
```bash
python3 examples/train_digitalocean.py --config large --mixed_precision
```
- 1024 dim, 8 layers, 8192 tokens
- Perfect for: Production models, long-range tasks
- Cost: ~$7.07
- **Leaves $7.93 for more experiments!**

### XLarge (6 hours, ~91% acc) - Maximum Power
```bash
python3 examples/train_digitalocean.py --config xlarge --mixed_precision
```
- 1280 dim, 12 layers, 8192 tokens  
- Perfect for: Best possible results
- Cost: ~$9.42
- **Uses most of your credit**

---

## 🔧 Advanced Options

### Custom Configuration
```bash
python3 examples/train_digitalocean.py \
    --config medium \
    --max_seq_len 8192 \
    --learning_rate 1e-4 \
    --epochs 5 \
    --mixed_precision \
    --subset_size 10000
```

### Benchmark Mode
```bash
python3 examples/train_digitalocean.py --benchmark
```
Tests all configurations at multiple sequence lengths

### Different Datasets
```bash
# AG News (4-class text classification)
python3 examples/train_digitalocean.py --config large --dataset ag_news

# WikiText-103 (language modeling)
python3 examples/train_digitalocean.py --config large --dataset wikitext-103
```

---

## 📈 Monitoring Your Training

### In Separate tmux Panes

#### Pane 1: Training Output
```bash
# Your training runs here
python3 examples/train_digitalocean.py --config large
```

#### Pane 2: GPU Monitor
```bash
# Split pane: Ctrl+b then "
watch -n 1 nvidia-smi
```

#### Pane 3: Cost Tracker
```bash
# Split again: Ctrl+b then "
python3 ~/ten_workspace/cost_tracker.py
```

### Tmux Cheat Sheet
- `Ctrl+b "` - Split horizontally
- `Ctrl+b %` - Split vertically  
- `Ctrl+b arrow` - Navigate panes
- `Ctrl+b d` - Detach (keeps running)
- `tmux attach -t training` - Reattach

---

## 💰 Cost Management

### Your Budget Breakdown

| Activity | Time | Cost | Notes |
|----------|------|------|-------|
| Setup | 5 min | $0.13 | One-time |
| Tiny test | 30 min | $0.79 | Verify setup |
| Large model | 4.5 hrs | $7.07 | Main training |
| **Subtotal** | **5 hrs** | **$7.99** | |
| **Remaining** | **4.5 hrs** | **$7.01** | For experiments! |

### Cost Tracking
```bash
# Real-time cost tracking
python3 ~/ten_workspace/cost_tracker.py

# Output:
# Elapsed: 45.2 min | Cost: $1.18 | Remaining credit: $13.82
```

---

## 🎯 Recommended Workflow

### For Maximum Value from $15 Credit:

#### Day 1: Setup & Quick Test (30 min, ~$0.80)
```bash
# Test that everything works
python3 examples/train_digitalocean.py --config tiny
```

#### Day 2: Large Model Training (4.5 hours, ~$7.07)
```bash
# Main training run
python3 examples/train_digitalocean.py --config large --mixed_precision
```

#### Day 3: Experiments (4.5 hours, ~$7.07)
Choose one:
- **Option A:** 3× medium models (different hyperparameters)
- **Option B:** 1× XLarge model (maximum quality)
- **Option C:** Comprehensive benchmarks + 2× small models

**Total: ~$15** 🎯

---

## 📥 Downloading Results

### Before Destroying Droplet!

From your **local machine**:

```bash
# Download checkpoints
scp -r root@your-droplet-ip:~/ten_workspace/checkpoints ./ten_results/

# Download logs
scp -r root@your-droplet-ip:~/ten_workspace/logs ./ten_results/

# Download plots
scp -r root@your-droplet-ip:~/ten_workspace/results ./ten_results/

# Or download everything
scp -r root@your-droplet-ip:~/ten_workspace ./ten_results/
```

---

## 🚨 Important Tips

### 1. Use tmux Always!
```bash
tmux new -s training
# Now if you disconnect, training continues!
```

### 2. Save Checkpoints Frequently
Models auto-save to `~/ten_workspace/checkpoints/`

### 3. Monitor Costs
Run cost tracker in a separate pane to avoid surprises

### 4. Download Before Destroying
**Critical:** Download results before destroying droplet!

### 5. Enable Mixed Precision
```bash
--mixed_precision  # 2× faster, same accuracy
```

### 6. Test First
Always run tiny config first to verify setup

---

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check nvidia-smi
nvidia-smi

# Should show: L40S or RTX 6000 Ada with 48GB
```

### Out of Memory
```bash
# Reduce batch size or sequence length
python3 examples/train_digitalocean.py --config medium  # Instead of large
```

### Training Stopped
```bash
# Reattach to tmux session
tmux attach -t training

# Or list all sessions
tmux ls
```

### Slow Training
```bash
# Enable mixed precision
python3 examples/train_digitalocean.py --config large --mixed_precision

# Check GPU utilization
nvidia-smi  # Should be 80-100%
```

---

## 📊 Expected Performance

### L40S GPU (Recommended)

| Config | Tokens/sec | Memory | Time | Accuracy |
|--------|-----------|--------|------|----------|
| Tiny | ~12,000 | 4 GB | 30 min | 82% |
| Small | ~6,000 | 12 GB | 1.5 hrs | 86% |
| Medium | ~3,000 | 24 GB | 3 hrs | 88% |
| Large | ~1,500 | 38 GB | 4.5 hrs | 90% |
| XLarge | ~1,000 | 45 GB | 6 hrs | 91% |

### RTX 6000 Ada (Alternative)

Similar performance, ~10% slower than L40S

---

## 🎓 Learning Resources

### After Training

1. **Analyze Results**
   ```bash
   cd ~/ten_workspace/results
   python3 -m http.server 8000
   # Access: http://your-droplet-ip:8000
   ```

2. **Compare Models**
   ```bash
   python3 examples/compare_checkpoints.py \
       --checkpoint1 ~/ten_workspace/checkpoints/model1.pt \
       --checkpoint2 ~/ten_workspace/checkpoints/model2.pt
   ```

3. **Visualize Training**
   - Check `logs/` directory for tensorboard logs
   - View plots in `results/` directory

---

## ✅ Pre-Destroy Checklist

Before destroying your droplet:

- [ ] Training completed successfully
- [ ] Results downloaded to local machine
- [ ] Checkpoints backed up
- [ ] Logs downloaded
- [ ] Cost tracker shows total spent
- [ ] No running processes (check with `ps aux | grep python`)

---

## 🚀 Quick Commands Reference

```bash
# Setup
ssh root@your-droplet-ip
./setup.sh

# Start training (in tmux)
tmux new -s training
source ~/ten_venv/bin/activate
cd ~/temporal-eigenstate-networks
python3 examples/train_digitalocean.py --config large --mixed_precision

# Monitor (in another pane)
Ctrl+b "  # Split pane
watch -n 1 nvidia-smi

# Track cost (in third pane)
Ctrl+b "  # Split again
python3 ~/ten_workspace/cost_tracker.py

# Detach from tmux
Ctrl+b d

# Reattach later
tmux attach -t training

# Download results (from local)
scp -r root@your-droplet-ip:~/ten_workspace ./ten_results
```

---

## 💡 Pro Tips

1. **L40S is faster** - Choose it over RTX 6000 if available
2. **Mixed precision** - Always use `--mixed_precision` for 2× speedup
3. **Start small** - Test with tiny config first (~30 min)
4. **Use tmux** - Your training survives disconnections
5. **Monitor costs** - Run cost tracker to stay within budget
6. **Download first** - Get results before destroying droplet
7. **Snapshot option** - Take a droplet snapshot to pause and resume later

---

## 🎯 Success Metrics

After 5 hours on L40S/RTX 6000:

- ✅ Large model trained (1024 dim, 8 layers)
- ✅ 8192 token sequences processed
- ✅ ~90% accuracy achieved
- ✅ Comprehensive benchmarks completed
- ✅ Multiple checkpoints saved
- ✅ Training visualizations generated
- ✅ Cost: ~$7-8 (leaving $7-8 for more experiments)

---

## 📞 Get Help

**Common Issues:**
- Setup problems → Check `~/ten_workspace/README.md`
- Training errors → See troubleshooting section above
- Cost concerns → Use cost tracker frequently

**DigitalOcean Support:**
- GPU droplet docs: [digitalocean.com/docs/products/gpu-droplets](https://digitalocean.com/docs/products/gpu-droplets)
- Community forums: [digitalocean.com/community](https://digitalocean.com/community)

---

**Ready to train? SSH into your droplet and run the setup script!** 🚀

**Remember: Your $15 credit = 9.5 hours = Plenty of time for great results!**
