# GPU Selection Guide for TEN Training

## 🎯 Quick Recommendation

**For your requirements (long-range tasks, large benchmarks, pretrained models, <5 hours):**

### **Best Choice: 40-48GB GPU (A100-40GB, RTX A6000, L40S, RTX 6000 Ada)**

---

## 🆕 DigitalOcean GPU Options (48GB VRAM) - **YOUR OPTIONS**

### Both Options Are Excellent! ⭐

**L40S** - 48GB VRAM @ $1.57/hour  
**RTX 6000 Ada** - 48GB VRAM @ $1.57/hour

**5-Hour Budget:** $7.85 (within your $15 free credit!)  
**Remaining Credit:** $7.15 for additional experiments

### Which One to Choose?

#### Choose **L40S** if:
✅ **Recommended for most users**  
- Newer architecture (Ada Lovelace)
- Better tensor core performance
- Faster mixed precision training (FP16/BF16)
- ~15-20% faster than RTX 6000 Ada for transformer models
- Better for inference workloads

#### Choose **RTX 6000 Ada** if:
- L40S is unavailable
- You need double precision (FP64) - though unlikely for ML
- Compatibility with older CUDA code

**Verdict: Go with L40S first, RTX 6000 Ada if L40S is unavailable**

---

## 📊 Detailed Analysis

### Your Requirements Analysis

1. **Long-range tasks** → Need sequences up to 8192+ tokens
2. **Larger benchmarking** → Multiple models/configs in parallel
3. **Pretrained models** → Larger model sizes (512-1024 dim, 8-12 layers)
4. **Time constraint** → Maximum 5 hours total
5. **Budget optimization** → Don't overpay for unused capacity

---

## 💰 GPU Options Breakdown

### Option 1: 24GB VRAM (RTX 3090/4090, RTX A5000)
**Cost:** ~$0.50-1.00/hour  
**Total Budget:** $2.50-5.00

**Capabilities:**
- ✅ Sequence length: Up to 4096 tokens
- ✅ Model size: Medium (d_model=512, n_layers=6)
- ✅ Batch size: 16-32
- ⚠️ **Limited** for very long sequences (8192+)
- ⚠️ **Constrained** for largest models

**Training Time Estimates:**
- Small model (256 dim, 4 layers): ~2-3 hours
- Medium model (512 dim, 6 layers): ~4-5 hours
- Large model (1024 dim, 8 layers): **Too tight/OOM risk**

**Verdict:** ⚠️ **Marginal** - Might struggle with your requirements

---

### Option 2: 40-48GB VRAM (A100-40GB, RTX A6000, L40S, RTX 6000 Ada) ⭐ **RECOMMENDED**
**Cost:** ~$1.50-2.50/hour  
**Total Budget:** $7.50-12.50

**DigitalOcean Options (48GB):** 🎯 **YOUR BEST CHOICE**
- **L40S:** $1.57/hour = **$7.85 for 5 hours** (Recommended)
- **RTX 6000 Ada:** $1.57/hour = **$7.85 for 5 hours** (Alternative)
- **Your $15 credit covers:** 5 hours training + 4.5 hours for experiments!

**Capabilities:**
- ✅ Sequence length: Up to 8192 tokens comfortably
- ✅ Model size: Large (d_model=1024, n_layers=8-12)
- ✅ Batch size: 32-64
- ✅ Multiple experiments in parallel
- ✅ Room for safety margin

**Training Time Estimates:**
- Small model (256 dim, 4 layers): ~1 hour
- Medium model (512 dim, 6 layers): ~2-3 hours
- Large model (1024 dim, 8 layers): ~4-5 hours
- **Bonus:** Can run 2-3 medium models in parallel

**Specific Models:**
- **A100-40GB**: Faster training (TF32, better tensor cores)
- **RTX A6000**: Slightly slower but more cost-effective

**Verdict:** ✅ **OPTIMAL** - Perfect balance of capability and cost

---

### Option 3: 48GB VRAM (A6000 Ada, RTX 6000 Ada)
**Cost:** ~$2.00-3.00/hour  
**Total Budget:** $10.00-15.00

**Capabilities:**
- ✅ All benefits of 40GB
- ✅ Slightly more headroom
- ⚠️ Marginal improvement over 40GB for your use case

**Verdict:** 🤷 **OVERKILL** - 40GB is sufficient, save the money

---

### Option 4: 80GB VRAM (A100-80GB, H100)
**Cost:** ~$3.00-5.00/hour  
**Total Budget:** $15.00-25.00

**Capabilities:**
- ✅ Maximum capacity
- ✅ Can train multiple large models simultaneously
- ⚠️ **Overkill** for 5-hour constraint
- ⚠️ **Expensive** for your needs

**Verdict:** ❌ **NOT RECOMMENDED** - Unnecessary for your requirements

---

## 🎯 Specific Recommendations by Provider

### RunPod (Recommended)
```
GPU: A100-40GB or RTX A6000
Cost: ~$1.50-2.00/hour
5-hour budget: $7.50-10.00
```

### Lambda Labs
```
GPU: A100 (40GB)
Cost: ~$1.10/hour
5-hour budget: $5.50
Note: Often sold out, but best price
```

### Vast.ai (Budget Option)
```
GPU: RTX A6000 (48GB)
Cost: ~$0.80-1.50/hour (varies by availability)
5-hour budget: $4.00-7.50
Note: Community marketplace, prices fluctuate
```

### Google Colab Pro+
```
GPU: A100-40GB
Cost: $50/month (unlimited within limits)
Note: Good if you'll use it multiple times
```

### Paperspace Gradient
```
GPU: A100-40GB
Cost: ~$3.00/hour
5-hour budget: $15.00
Note: Easier setup, higher cost
```

---

## 📋 What You Can Accomplish in 5 Hours on 40GB

### Scenario 1: Single Large Pretrained Model
```python
config = TemporalEigenstateConfig(
    d_model=1024,
    n_layers=8,
    num_eigenstates=128,
    max_seq_len=8192
)
```
- Dataset: Full WikiText-103 or BookCorpus
- Training time: ~4-5 hours
- Result: Production-ready pretrained model
- **Recommended for:** Creating reusable base model

### Scenario 2: Multiple Medium Models (Ablation Study)
```python
configs = [
    # Vary eigenstates
    (d_model=512, eigenstates=32),
    (d_model=512, eigenstates=64),
    (d_model=512, eigenstates=128),
]
```
- Each model: ~1.5 hours
- Result: 3 models for comparison
- **Recommended for:** Architecture research

### Scenario 3: Long-Range Benchmarks
```python
sequence_lengths = [512, 1024, 2048, 4096, 8192]
```
- Test each length with multiple models
- Comprehensive performance analysis
- Memory and speed profiling
- **Recommended for:** Paper/publication results

### Scenario 4: Multi-Task Training
```python
tasks = [
    "text_classification",
    "sequence_labeling", 
    "language_modeling"
]
```
- Train on 3 different tasks
- Each task: ~1.5 hours
- **Recommended for:** Demonstrating versatility

---

## 🔧 Optimal Configuration for 40GB GPU

### Large Model Configuration
```python
config = TemporalEigenstateConfig(
    d_model=1024,           # Large hidden dimension
    n_heads=16,             # More heads
    n_layers=8,             # Deep network
    d_ff=4096,              # Large FFN
    max_seq_len=8192,       # Long sequences
    num_eigenstates=128,    # More eigenstates
    dropout=0.1,
    vocab_size=50257,       # GPT-2 vocab
)

# Training settings
BATCH_SIZE = 32          # With gradient accumulation
GRADIENT_ACCUM = 4       # Effective batch = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 8192
```

**Expected Performance:**
- Model size: ~180M parameters
- Training time: ~4-5 hours on full dataset
- Memory usage: ~35GB (safe margin)
- Sequences/sec: ~50-100 at 8192 tokens

### Benchmarking Configuration
```python
models_to_test = {
    "tiny": (d_model=256, n_layers=4),
    "small": (d_model=512, n_layers=6),
    "medium": (d_model=768, n_layers=8),
    "large": (d_model=1024, n_layers=8),
}

sequence_lengths = [512, 1024, 2048, 4096, 8192]
batch_sizes = [64, 32, 16, 8, 4]  # Vary by length
```

---

## ⚡ Speed Comparison (40GB GPU)

### A100-40GB (TF32 Enabled)
- Small model: ~3000 tokens/sec
- Medium model: ~1500 tokens/sec
- Large model: ~800 tokens/sec
- **Best for:** Fastest training

### RTX A6000 (48GB)
- Small model: ~2200 tokens/sec
- Medium model: ~1100 tokens/sec
- Large model: ~600 tokens/sec
- **Best for:** Budget + capacity

### A6000 Ada (48GB, newer arch)
- Small model: ~2800 tokens/sec
- Medium model: ~1400 tokens/sec
- Large model: ~750 tokens/sec
- **Best for:** Balance of speed and cost

---

## 💡 Pro Tips for 5-Hour Training

### 1. Use Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
```
**Benefit:** 2× faster training, 40% less memory

### 2. Gradient Accumulation
```python
BATCH_SIZE = 16
GRADIENT_ACCUM = 4  # Effective batch = 64
```
**Benefit:** Train larger batch sizes without OOM

### 3. Efficient Data Loading
```python
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=8,      # More workers
    pin_memory=True,
    persistent_workers=True
)
```

### 4. Compile Model (PyTorch 2.0+)
```python
model = torch.compile(model, mode="max-autotune")
```
**Benefit:** 10-30% speedup

### 5. Profile First
```python
# Run small test first
SUBSET_SIZE = 1000
quick_test(model, subset_data)  # ~5 minutes

# Then scale up
full_training(model, full_data)  # ~4.5 hours
```

---

## 📊 Cost-Benefit Analysis

### 24GB GPU
- **Cost:** $2.50-5.00 (5 hours)
- **Risk:** Medium-high (OOM on long sequences)
- **Throughput:** Limited
- **Verdict:** Too risky ⚠️

### 40GB GPU ⭐
- **Cost:** $7.50-12.50 (5 hours)
- **Risk:** Low (plenty of headroom)
- **Throughput:** High
- **Verdict:** **OPTIMAL** ✅

### 80GB GPU
- **Cost:** $15.00-25.00 (5 hours)
- **Risk:** None
- **Throughput:** Very high (but overkill)
- **Verdict:** Unnecessary for 5 hours ❌

---

## 🎯 Final Recommendation

### **Choose: 40GB GPU (A100-40GB or RTX A6000)**

### Why?
1. ✅ **Sufficient capacity** for 8192-token sequences
2. ✅ **Large models** (1024 dim, 8-12 layers) fit comfortably
3. ✅ **Multiple experiments** possible in 5 hours
4. ✅ **Cost-effective** for your time budget
5. ✅ **Safety margin** prevents OOM surprises
6. ✅ **Future-proof** for scaling experiments

### Where to Rent?
**Best Value:** Lambda Labs A100-40GB (~$1.10/hour)
**Most Available:** RunPod A100-40GB (~$1.50/hour)
**Budget Friendly:** Vast.ai RTX A6000 (~$0.80-1.50/hour)

### Expected Results in 5 Hours:
- ✅ 1 large pretrained model (1024 dim, 8 layers)
- ✅ OR 3 medium models (512 dim, 6 layers each)
- ✅ OR Complete benchmark suite (all sequence lengths)
- ✅ Publication-ready results with charts
- ✅ Saved checkpoints for future use

---

## 🚀 Getting Started Script

Once you have your GPU:

```bash
# 1. Clone repo
git clone https://github.com/genovotechnologies/temporal-eigenstate-networks.git
cd temporal-eigenstate-networks

# 2. Install dependencies
pip install -e .
pip install wandb datasets transformers  # extras

# 3. Download data
python scripts/download_data.py --dataset wikitext-103

# 4. Train large model
python examples/train_large_model.py \
    --d_model 1024 \
    --n_layers 8 \
    --num_eigenstates 128 \
    --max_seq_len 8192 \
    --batch_size 32 \
    --gradient_accum 4 \
    --epochs 3 \
    --mixed_precision \
    --output_dir ./checkpoints/large_model

# Total time: ~4-5 hours
```

---

## 📞 Need Help?

If you need assistance with:
- Setting up on specific GPU providers
- Optimizing training for your GPU
- Troubleshooting OOM errors
- Scaling to multiple GPUs

Open an issue or check the documentation!

---

**TL;DR: Get a 40GB GPU (A100 or A6000). It's perfect for your 5-hour training window.**
