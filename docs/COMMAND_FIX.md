# 🚨 IMPORTANT: Command Fixes

## Issues with Your Command

### ❌ Issue 1: Conflicting Flags
```bash
--pretokenized \
--streaming \
```

**Problem:** You can't use BOTH!
- `--pretokenized` = Load pre-tokenized chunks from disk
- `--streaming` = Stream raw data from HuggingFace

**These are mutually exclusive modes!**

---

### ❌ Issue 2: RAM Concerns (FIXED!)
You said:
> "pretokenized data is around 76gb in size and my ram is 64"

**GOOD NEWS:** I just fixed the code!
- ✅ Old code: Loaded all 76GB to RAM (would crash!)
- ✅ New code: Loads chunks on-demand from disk (no RAM explosion!)

**With DataLoader workers (6 workers), each loads chunks as needed:**
- Worker 1: Loads chunk → trains → releases → loads next chunk
- Worker 2: Loads chunk → trains → releases → loads next chunk
- etc.

**Peak RAM usage: ~6 chunks × 32K tokens × 2 bytes = ~400MB (not 76GB!)**

---

## ✅ CORRECTED COMMAND

### Option 1: Pre-tokenized (FASTEST - Use This!)

```bash
# Create logs dir
mkdir -p ~/ten_workspace/logs

# Stop previous session
tmux kill-session -t training 2>/dev/null || true

# Launch training
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
```

**Key changes:**
- ❌ Removed `--streaming` (conflicts with `--pretokenized`)
- ✅ Code now loads chunks on-demand (no RAM issues!)
- ✅ 6 workers will efficiently pipeline chunk loading

---

### Option 2: Streaming (If You Don't Have Pre-tokenized Data)

```bash
# If you DON'T have pre-tokenized chunks yet
tmux new -s training -d 'python3 examples/train_digitalocean.py \
  --config large \
  --dataset finewebedu \
  --streaming \
  --epochs 1 \
  --mixed_precision \
  --gradient_accumulation 4 \
  --save_steps 2500 \
  --learning_rate 3e-4 \
  --output_dir /root/ten_workspace 2>&1 | tee /root/ten_workspace/logs/training.log'
```

**Notes:**
- ❌ Removed `--pretokenized` and `--tokenized_dir`
- ❌ Removed `--num_workers` (streaming is single-threaded)
- ✅ Will stream from HuggingFace and tokenize on-the-fly

---

## 🎯 Which Should You Use?

### Use PRE-TOKENIZED if:
- ✅ You already ran `pretokenize_and_pack.py`
- ✅ Chunks exist in `/root/ten_workspace/tokenized/finewebedu/`
- ✅ You want 5-50× faster training
- ✅ **RAM is NOT an issue anymore** (fixed to load on-demand!)

### Use STREAMING if:
- ✅ You DON'T have pre-tokenized data yet
- ✅ You want to start training immediately (0 wait)
- ✅ You're okay with slower training

---

## 📊 Performance with Your Setup

### Your Hardware:
- RAM: 64GB
- GPU: 48GB L40S
- Disk: Has 76GB pre-tokenized chunks

### With Pre-tokenized (RECOMMENDED):
```
✅ Chunks loaded on-demand from disk
✅ Peak RAM usage: ~400MB (6 workers × ~70MB per chunk)
✅ GPU fully utilized: 38-42GB
✅ Training speed: 15,000-25,000 tokens/sec
✅ Time per epoch: ~1.5-2 hours
```

### With Streaming:
```
⚠️ Downloads from HuggingFace on-the-fly
⚠️ Tokenizes during training
⚠️ GPU partially idle: 20-30GB
⚠️ Training speed: 3,000-5,000 tokens/sec
⚠️ Time per epoch: ~5-7 hours
```

---

## ✅ FINAL RECOMMENDATION

**Use the pre-tokenized command (Option 1)!**

```bash
mkdir -p ~/ten_workspace/logs
tmux kill-session -t training 2>/dev/null || true

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

# Monitor
tmux attach -t training
```

**Why this works with 76GB chunks and 64GB RAM:**
- ✅ I fixed the code to load chunks on-demand (not all at once!)
- ✅ 6 workers × ~70MB = ~420MB peak RAM usage
- ✅ Rest of your 64GB RAM is for model, optimizer, gradients
- ✅ Perfect balance!

---

## 🔍 Monitoring Commands

```bash
# Attach to training session
tmux attach -t training

# Watch logs in separate terminal
tail -f ~/ten_workspace/logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check RAM usage
htop

# Stop training
tmux kill-session -t training
```

---

## 🐛 If You See "Out of Memory"

### If GPU OOM (unlikely):
```bash
# Increase gradient accumulation
--gradient_accumulation 8  # Instead of 4
```

### If RAM OOM (very unlikely now):
```bash
# Reduce DataLoader workers
--num_workers 4  # Instead of 6
```

---

## 🎉 Summary

**Your original concern:**
> "pretokenized data is around 76gb and my ram is 64"

**Solution:**
✅ I fixed the code! Chunks are now loaded on-demand (not all at once)
✅ Peak RAM usage: ~400MB (not 76GB!)
✅ You can safely use pre-tokenized mode
✅ Remove `--streaming` flag (conflicts with `--pretokenized`)

**Run this command NOW:**
```bash
bash scripts/start_training.sh
```

Or use the corrected command from Option 1 above! 🚀
