# DataLoader Deadlock Fix

## Problem

Training hangs at `0/18968 [00:00<?, ?it/s]` - the DataLoader is frozen and never delivers the first batch.

**Root Cause:** The `'fork'` multiprocessing context combined with `torch.compile()` causes a deadlock with CUDA operations.

## Solution

### Quick Fix (Run This Now!)

```bash
cd /root/temporal-eigenstate-networks
git pull  # Get the fixes
bash scripts/fix_deadlock.sh
```

This will:
1. Kill the stuck training process
2. Clear GPU memory
3. Restart with fixed settings

### What Changed

1. **Multiprocessing context:** `'fork'` → `'spawn'`
   - Fork copies the entire process memory, which conflicts with CUDA
   - Spawn creates fresh worker processes (safer but slightly slower startup)

2. **Prefetch factor:** `16` → `4`
   - Reduces memory pressure from prefetching
   - Still keeps GPU fed (24 batches pipelined with 6 workers)

3. **Added `--no_compile` flag:**
   - Disable torch.compile() if it causes issues
   - Trades ~10% speed for stability

### Manual Restart

If the script doesn't work, manually run:

```bash
# 1. Kill stuck process
tmux kill-session -t training
pkill -f train_digitalocean.py

# 2. Wait for GPU to clear
nvidia-smi  # Should show 0 MiB used after a few seconds

# 3. Restart training
cd /root/temporal-eigenstate-networks

tmux new -s training -d "
python3 examples/train_digitalocean.py \
  --config large \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 4 \
  --save_steps 2500 \
  --num_workers 4 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --output_dir /root/ten_workspace/runs/\$(date +%F_%H-%M) \
  2>&1 | tee ~/ten_workspace/logs/training_fixed.log
"

# 4. Monitor
tmux attach -t training
```

### If Still Hangs

Try with torch.compile disabled:

```bash
# Add --no_compile flag
python3 examples/train_digitalocean.py \
  --config large \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 4 \
  --save_steps 2500 \
  --num_workers 4 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/$(date +%F_%H-%M)
```

### Nuclear Option: Single-Process Mode

If all else fails, disable multiprocessing entirely:

```bash
python3 examples/train_digitalocean.py \
  --config large \
  --dataset finewebedu \
  --pretokenized \
  --tokenized_dir /root/ten_workspace/tokenized/finewebedu \
  --epochs 1 \
  --gradient_accumulation 4 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --mixed_precision \
  --no_compile \
  --output_dir /root/ten_workspace/runs/$(date +%F_%H-%M)
```

This will be slower (~20% overhead) but guaranteed to work.

## Technical Details

### Why Fork Causes Deadlocks

1. Fork copies entire parent process memory (including CUDA context)
2. CUDA operations in forked processes can deadlock
3. torch.compile() makes this worse by caching compiled kernels

### Why Spawn Works

1. Spawn creates fresh processes with clean CUDA context
2. Each worker initializes CUDA independently
3. No memory conflicts between parent and workers

### Performance Impact

- **Fork:** Fast startup (~0.1s per worker), but deadlock-prone
- **Spawn:** Slower startup (~2-3s per worker), but stable
- **With 4 workers:** ~10s startup delay, then full speed

## Verification

After restart, you should see:

```
✓ Using 'spawn' multiprocessing (prevents deadlocks with torch.compile)
✓ Prefetching: 24 batches in pipeline!

Epoch 1:   0%|          | 1/18968 [00:05<30:15:42,  5.75s/it]
```

Key: **The progress bar should update within 10-20 seconds** (after worker spawn)

If you see iteration count increasing (`1/18968`, `2/18968`, etc.), you're good!

## Cost Impact

- **Time lost:** 8 hours @ $1.57/hr = **$12.56 wasted** 😢
- **Remaining credit:** $15.00 - $12.56 = **$2.44 left**
- **Training time needed:** ~4.5 hours
- **Required:** $7.07

**You'll need to add ~$5 credit to finish training!**

## Prevention

Always use these settings for multi-GPU/compiled training:

```python
DataLoader(
    ...,
    num_workers=4,
    multiprocessing_context='spawn',  # NOT 'fork'!
    prefetch_factor=4,
    persistent_workers=True
)
```
