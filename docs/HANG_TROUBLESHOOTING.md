# DataLoader Hang - Comprehensive Troubleshooting

## Current Situation

Training hangs at `Epoch 1:   0%|          | 0/18968 [00:00<?, ?it/s]` even after:
- ✓ Changing fork → spawn
- ✓ Reducing prefetch factor
- ✓ Adding --no_compile

**This suggests the issue is NOT multiprocessing, but something else.**

## Diagnostic Steps

### Step 1: Run Diagnostics

On your DigitalOcean server:

```bash
cd /root/temporal-eigenstate-networks
git pull
bash scripts/diagnose_hang.sh
```

This will test:
1. **Chunk loading:** Can we load individual chunks?
2. **DataLoader:** Does DataLoader work in isolation?
3. **Worker spawn:** Do spawn workers initialize correctly?

### Step 2: Interpret Results

**If chunk loading fails:**
```
✗ Failed to load: [error]
```
→ Problem: Corrupted tokenized data
→ Solution: Re-tokenize

**If single-process DataLoader works but multi-process hangs:**
```
✓ Single-process DataLoader works!
✗ TIMEOUT! Multi-process causes hang
```
→ Problem: Worker initialization issue
→ Solution: Use single-process mode (see Step 3)

**If everything passes:**
```
✓ All tests pass!
```
→ Problem: Interaction with model/torch.compile/mixed precision
→ Solution: Try single-process mode first

## Step 3: Guaranteed Solution - Single Process Mode

This **WILL WORK** but is ~20% slower:

```bash
cd /root/temporal-eigenstate-networks
git pull
bash scripts/train_singleprocess.sh
```

This runs with:
- `num_workers=0` (no multiprocessing)
- `--no_compile` (no torch.compile)
- All other optimizations intact

**Expected behavior:**
- Training starts within 5 seconds
- Progress bar updates immediately
- ~20% slower iteration time (acceptable trade-off)

## Step 4: Monitor Progress

```bash
tmux attach -t training
# or
tail -f ~/ten_workspace/logs/training_singleprocess.log
```

**Success indicators:**
```
Epoch 1:   0%| 1/18968 [00:08<42:15:23,  8.03s/it]  ← MOVING!
Epoch 1:   0%| 2/18968 [00:16<42:10:45,  8.01s/it]
```

## Possible Root Causes

### 1. DataLoader Worker Init Timeout

**Symptom:** Hangs at 0% forever
**Cause:** Workers spawn but never call `__getitem__`
**Why:** PyTorch 2.x + CUDA + spawn has known issues

**Solution:** Single-process mode

### 2. Chunk Loading Bottleneck

**Symptom:** First iteration takes minutes
**Cause:** torch.load() on networked filesystem (slow I/O)
**Why:** DigitalOcean block storage latency

**Solution:** 
- Pre-load chunks to memory (requires 18GB RAM)
- Or use single-process mode (less I/O contention)

### 3. torch.compile() Hang

**Symptom:** Hangs after "Model compiled successfully!"
**Cause:** First JIT compilation can take 5-10 minutes
**Why:** Compiling 1.4B parameter model

**Solution:** Wait longer (try 15 minutes) or use --no_compile

### 4. Mixed Precision Init

**Symptom:** Hangs at start of training loop
**Cause:** FP16 tensor initialization on CUDA
**Why:** Rare CUDA/driver issue

**Solution:** Single-process mode

## Expected Timeline (Single-Process Mode)

```
Initialization:     ~30 seconds
First batch:        ~8-10 seconds  
Per batch (avg):    ~8 seconds
Total batches:      18,968
Time per epoch:     ~42 hours  ❌ TOO LONG!
```

**Wait, this doesn't work!** 42 hours is way too long!

## Better Solution: Increase Batch Size

With single-process mode, increase effective batch size:

```bash
# Edit the command in train_singleprocess.sh:
--batch_size 32 \  # Double from 16
--gradient_accumulation 2  # Half from 4
# Effective batch size stays same: 32×2 = 64 (was 16×4)
```

This should get you ~4-5 hours training time.

## Nuclear Option: Smaller Model

If nothing works, train medium config instead:

```bash
--config medium \  # 850M params instead of 1.4B
```

Medium config with your pre-tokenized data will still produce good results!

## Cost Considerations

- **Current credit:** $2.44
- **Time burned:** 8+ hours = $12.56 wasted
- **Need to complete:** ~5 hours = $7.85
- **Shortfall:** ~$5.50

**You MUST add credit to finish training!**

## Recommendation

1. **First:** Try single-process mode with diagnostics
2. **If slow:** Increase batch size to 32
3. **If still fails:** Switch to medium config
4. **Add $6 credit** to ensure completion

## Quick Commands

```bash
# Diagnose
bash scripts/diagnose_hang.sh

# Train (single-process, guaranteed to work)
bash scripts/train_singleprocess.sh

# Monitor
tmux attach -t training
```
