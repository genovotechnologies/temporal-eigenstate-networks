# Training Quick Start Guide

## 🚀 Ready-to-Run Commands on Your Droplet

You've downloaded the datasets. Now start training!

### 1️⃣ Quick Test (5 minutes, ~$0.13)

Test everything works with a tiny model on a small subset:

```bash
cd /root/temporal-eigenstate-networks
source /root/ten_venv/bin/activate

python3 examples/train_digitalocean.py \
    --config tiny \
    --dataset tinystories \
    --subset_size 1000 \
    --epochs 1 \
    --mixed_precision
```

**What this does:**
- Creates tiny model (256d, 2 layers, 17M params)
- Trains on 1K stories for 1 epoch
- Should complete in ~5 minutes
- Validates your entire setup works

---

### 2️⃣ Fast Pretraining (1.5 hours, ~$2.36)

Small model on TinyStories for quick pretraining:

```bash
tmux new -s training

python3 examples/train_digitalocean.py \
    --config small \
    --dataset tinystories \
    --epochs 2 \
    --mixed_precision \
    --save_steps 5000
```

**What this does:**
- Small model (512d, 4 layers, 65M params)
- Trains on 2.1M stories for 2 epochs
- ~1.5 hours, costs ~$2.36
- Great for testing architectures

---

### 3️⃣ Serious Pretraining (4 hours, ~$6.28) ⭐ RECOMMENDED

Large model on FineWeb-Edu for real pretrained model:

```bash
tmux new -s training
source /root/ten_venv/bin/activate
cd /root/temporal-eigenstate-networks

# Split tmux for monitoring
# Ctrl+b then "

# In pane 1: Run training
python3 examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --epochs 1 \
    --mixed_precision \
    --save_steps 5000 \
    --learning_rate 2e-4

# In pane 2 (Ctrl+b then arrow key): Monitor GPU
watch -n 1 nvidia-smi

# In pane 3 (split again): Track cost
python3 ~/ten_workspace/cost_tracker.py
```

**What this does:**
- Large model (1024d, 8 layers, 180M params)
- Trains on 9.7M documents (10B tokens!)
- ~4 hours, costs ~$6.28
- **Creates production-quality pretrained model**
- Saves checkpoints every 5K steps

---

### 4️⃣ Long-Range Training (3 hours, ~$4.71)

Medium model on WikiText-103 with 8K sequences:

```bash
python3 examples/train_digitalocean.py \
    --config medium \
    --dataset wikitext-103 \
    --max_seq_len 8192 \
    --epochs 3 \
    --mixed_precision \
    --gradient_accumulation 2 \
    --save_steps 2000
```

**What this does:**
- Medium model (768d, 6 layers, 110M params)
- 8K token sequences (tests long-range!)
- 1.8M Wikipedia articles
- ~3 hours, costs ~$4.71

---

## 📊 Training Outputs

All outputs saved to: `~/ten_workspace/`

```
~/ten_workspace/
├── checkpoints/
│   ├── checkpoint-1000.pt    # Every 1K/5K steps
│   ├── checkpoint-2000.pt
│   ├── best_model.pt          # Best validation loss
│   └── final_model.pt         # Last checkpoint
├── logs/
│   └── training.log
└── results/
```

---

## 🔧 Important Flags

### Essential:
- `--config`: Model size (tiny/small/medium/large/xlarge)
- `--dataset`: Dataset to use (tinystories/finewebedu/wikitext-103/pg19/openwebtext)
- `--mixed_precision`: **Always use this!** 2× faster, 50% less memory
- `--epochs`: Number of training passes

### Tuning:
- `--learning_rate`: Default 3e-4, try 1e-4 to 5e-4
- `--max_seq_len`: Override sequence length (up to 8192)
- `--gradient_accumulation`: Simulate larger batch size (2/4/8)
- `--save_steps`: Checkpoint frequency (default 1000)
- `--subset_size`: Use subset for testing (e.g., 10000)

### Validation:
- `--dry_run`: Test model creation + 1 forward pass
- `--benchmark`: Run speed benchmarks across configs

---

## 🎯 Recommended Workflow

### Phase 1: Validate (5 min, $0.13)
```bash
# Quick sanity check
python3 examples/train_digitalocean.py \
    --config tiny \
    --dataset tinystories \
    --subset_size 1000 \
    --epochs 1 \
    --mixed_precision
```

### Phase 2: Pretrain (4 hours, $6.28)
```bash
# Real pretraining on high-quality data
tmux new -s training
python3 examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --epochs 1 \
    --mixed_precision \
    --save_steps 5000
```

### Phase 3: Download & Destroy (5 min, free)
```bash
# On your LOCAL machine (not droplet):
scp -r root@YOUR_DROPLET_IP:~/ten_workspace/checkpoints ./
scp -r root@YOUR_DROPLET_IP:~/ten_workspace/logs ./

# Then destroy droplet to stop billing
```

**Total cost: ~$6.50 out of $15 credit**
**Remaining: ~$8.50 for experiments!**

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Try smaller batch size with gradient accumulation
python3 examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --gradient_accumulation 4 \
    --mixed_precision
```

### Training too slow
```bash
# Make sure mixed precision is enabled!
--mixed_precision  # 2× faster

# Or use smaller model
--config medium  # instead of large
```

### Dataset not found
```bash
# Download it first
cd /root/temporal-eigenstate-networks/scripts
python3 download_datasets.py --dataset finewebedu
```

### Session disconnected
```bash
# That's why we use tmux!
# Reconnect:
tmux attach -t training

# Training continues even if you disconnect
```

---

## 💡 Pro Tips

1. **Always use tmux** - Training continues if SSH drops
2. **Always use --mixed_precision** - 2× faster, essential!
3. **Start small** - Test with `--config tiny --subset_size 1000` first
4. **Monitor GPU** - Run `nvidia-smi` in separate pane
5. **Track costs** - Run `cost_tracker.py` to stay in budget
6. **Download before destroying** - Don't lose your model!

---

## 📈 Expected Results

### After 1 epoch on FineWeb-Edu (large model):
- Loss: ~3.5-4.0 (initial) → ~2.0-2.5 (final)
- Perplexity: ~33 → ~7-12
- Time: ~4 hours
- Cost: ~$6.28

### After 3 epochs on WikiText-103 (medium model):
- Loss: ~4.0 → ~2.5-3.0
- Perplexity: ~55 → ~12-20
- Time: ~3 hours
- Cost: ~$4.71

### Quality Check:
You have a good pretrained model if:
- ✅ Loss decreases steadily
- ✅ Final loss < 3.0
- ✅ Perplexity < 20
- ✅ Model completes training without OOM

---

## 🎓 Example Full Session

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Start tmux
tmux new -s training

# Activate environment
source /root/ten_venv/bin/activate
cd /root/temporal-eigenstate-networks

# Split window (Ctrl+b then ")
# In top pane: training
# In bottom pane: monitoring

# Top pane - Training:
python3 examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --epochs 1 \
    --mixed_precision \
    --save_steps 5000 \
    --learning_rate 2e-4

# Bottom pane (Ctrl+b, arrow down, then):
watch -n 1 "nvidia-smi && echo '' && python3 ~/ten_workspace/cost_tracker.py"

# Detach from tmux: Ctrl+b then d
# Training continues!

# Later, reattach:
tmux attach -t training

# When done, download models:
# (From your local machine)
scp -r root@DROPLET_IP:~/ten_workspace/checkpoints ./ten_models

# Then destroy droplet
```

---

## 🚀 Ready to Start?

**Recommended first command:**

```bash
tmux new -s training
source /root/ten_venv/bin/activate
cd /root/temporal-eigenstate-networks

python3 examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --epochs 1 \
    --mixed_precision \
    --save_steps 5000
```

This will train for ~4 hours and cost ~$6.28, creating a production-quality 180M parameter pretrained model on 10 billion tokens of high-quality web text.

**Happy training! 🎉**
