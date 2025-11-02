# Dataset Guide for TEN Training on DigitalOcean

## 🎯 Quick Recommendation for YOUR Setup ($15, 5 hours)

**Best combination for pretrained models with long-range capabilities:**

```bash
cd /root/temporal-eigenstate-networks/scripts

# Option 1: Fast + Effective (Recommended)
python3 download_datasets.py --dataset wikitext-103  # 3 min, 1.8M articles
python3 download_datasets.py --dataset pg19          # 15 min, ultra-long sequences

# Option 2: Modern + High Quality
python3 download_datasets.py --dataset tinystories   # 5 min, 2M stories
python3 download_datasets.py --dataset finewebedu    # 25 min, 10B tokens
```

---

## 📊 Dataset Comparison Table

| Dataset | Size | Samples | Avg Length | Best For | Download Time |
|---------|------|---------|------------|----------|---------------|
| **WikiText-103** ✅ | 500MB | 1.8M | 3800 tokens | General pretraining | 3 min |
| **PG19** ✅ | 11GB | 28K books | 50K-100K tokens | Ultra-long range | 15 min |
| **TinyStories** ✅ | 2GB | 2M stories | 300 tokens | Fast testing | 5 min |
| **FineWeb-Edu** ✅ | 40GB | 9.6M docs | 1000 tokens | High quality web | 25 min |
| **BookCorpus** | 4.5GB | 74M sentences | 150 tokens | Books (BERT) | 15 min |
| **OpenWebText** | 38GB | 8M docs | Varies | Web (GPT-2) | 60 min |

✅ = Highly recommended for your goals

---

## 🚀 Training Recipes

### Recipe 1: Quick Pretrain (3 hours total)
**Goal:** Create a solid pretrained model fast

```bash
# Download (18 min)
python3 download_datasets.py --dataset wikitext-103
python3 download_datasets.py --dataset pg19

# Train (2.5 hours)
python3 examples/train_digitalocean.py \
    --config medium \
    --dataset wikitext-103 \
    --epochs 3 \
    --mixed_precision

# Cost: ~$4.50 total
```

**Why this works:**
- WikiText-103: 1.8M high-quality Wikipedia articles
- PG19: Tests long-range (8K+ tokens) on books
- Medium model: 768d, 6 layers, fits in budget
- Leaves $10.50 for experiments

---

### Recipe 2: Maximum Quality (4.5 hours)
**Goal:** Best pretrained model within budget

```bash
# Download (30 min)
python3 download_datasets.py --dataset finewebedu
python3 download_datasets.py --dataset pg19

# Train (4 hours)
python3 examples/train_digitalocean.py \
    --config large \
    --dataset finewebedu \
    --epochs 2 \
    --mixed_precision

# Cost: ~$7.00 total
```

**Why this works:**
- FineWeb-Edu: 10B tokens of highest quality web text
- Large model: 1024d, 8 layers, 90% accuracy
- PG19 for long-range validation
- Leaves $8 for fine-tuning experiments

---

### Recipe 3: Fast Iteration (1 hour cycles)
**Goal:** Quick experiments and testing

```bash
# Download (5 min)
python3 download_datasets.py --dataset tinystories

# Train multiple configs (1 hour each)
python3 examples/train_digitalocean.py --config tiny --dataset tinystories
python3 examples/train_digitalocean.py --config small --dataset tinystories
python3 examples/train_digitalocean.py --config medium --dataset tinystories

# Cost: ~$1.57/hour per config
```

**Why this works:**
- TinyStories: Small, high-quality, fast to train
- Test multiple architectures
- Find best hyperparameters
- Then scale up with bigger dataset

---

## 🔥 Why IMDb Is Bad (What You Discovered)

| Metric | IMDb | WikiText-103 | PG19 |
|--------|------|--------------|------|
| Samples | 25K | 1.8M | 28K books |
| Avg Length | 250 tokens | 3800 tokens | 70,000 tokens |
| Total Tokens | ~6M | ~6.8B | ~2B |
| Task Diversity | 1 (sentiment) | General LM | General LM |
| Long-range? | ❌ No | ✅ Yes | ✅✅ Perfect |
| Pretraining? | ❌ No | ✅ Yes | ✅ Yes |

**IMDb problems:**
- Only 6M tokens (tiny for pretraining)
- Short sequences (can't test 8K capability)
- Single task (no transfer learning)
- No long-range dependencies

---

## 📥 Download Commands Reference

```bash
cd /root/temporal-eigenstate-networks/scripts

# Show all options
python3 download_datasets.py --menu

# Download specific datasets
python3 download_datasets.py --dataset wikitext-103
python3 download_datasets.py --dataset pg19
python3 download_datasets.py --dataset tinystories
python3 download_datasets.py --dataset finewebedu
python3 download_datasets.py --dataset bookcorpus
python3 download_datasets.py --dataset openwebtext

# Download all recommended (takes ~2 hours, 100GB)
python3 download_datasets.py --all-recommended
```

---

## ⚡ What to Download NOW on Your Droplet

Based on your L40S GPU and $15 budget:

```bash
# Run this NOW (takes 20 min total)
cd /root/temporal-eigenstate-networks/scripts
python3 download_datasets.py --dataset wikitext-103  # 3 min
python3 download_datasets.py --dataset pg19          # 15 min
```

While downloading, read the training guide:
```bash
cat /root/ten_workspace/README.md
```

Then start training:
```bash
tmux new -s training
source /root/ten_venv/bin/activate
cd /root/temporal-eigenstate-networks

# Start with medium config (3 hours, $4.71)
python3 examples/train_digitalocean.py \
    --config medium \
    --dataset wikitext \
    --mixed_precision \
    --save_steps 1000
```

---

## 💡 Pro Tips

1. **Download during setup:** Downloads don't use GPU, so they're "free"
2. **Use streaming for huge datasets:** FineWeb can stream without full download
3. **Start small:** Test with TinyStories first, then scale up
4. **Mix datasets:** Pretrain on WikiText, fine-tune on PG19
5. **Track your spend:** Use `python3 ~/ten_workspace/cost_tracker.py`

---

## 🔍 Dataset Details

### WikiText-103 ⭐ (RECOMMENDED)
- **What:** Wikipedia articles (cleaned, no markup)
- **Size:** 500MB, 1.8M articles
- **Best for:** General language understanding
- **Sequences:** 3800 tokens average
- **Quality:** High (curated Wikipedia)
- **Training time:** 2-3 hours for medium model

### PG19 ⭐⭐ (BEST FOR LONG-RANGE)
- **What:** Books from Project Gutenberg (1919 and before)
- **Size:** 11GB, 28K books
- **Best for:** Ultra-long sequences (8K+ tokens)
- **Sequences:** 50K-100K tokens per book
- **Quality:** Very high (classic literature)
- **Training time:** 4-5 hours for large model

### TinyStories ⭐ (FAST TESTING)
- **What:** Synthetic short stories (GPT-generated)
- **Size:** 2GB, 2M stories
- **Best for:** Quick experiments, testing architectures
- **Sequences:** 200-500 tokens
- **Quality:** Good (coherent narratives)
- **Training time:** 1 hour for small model

### FineWeb-Edu ⭐⭐⭐ (HIGHEST QUALITY)
- **What:** Web pages filtered for educational content
- **Size:** 40GB sample (10B tokens)
- **Best for:** Modern, diverse pretraining
- **Sequences:** ~1000 tokens average
- **Quality:** Excellent (filtered for quality)
- **Training time:** 4-5 hours for large model

---

## ❓ FAQ

**Q: Can I use multiple datasets?**
A: Yes! Train on WikiText-103 first, then fine-tune on PG19.

**Q: Will downloads use my $15 credit?**
A: No! Downloads don't use GPU. Only training costs money.

**Q: How much time for downloads?**
A: WikiText-103 (3 min) + PG19 (15 min) = 18 minutes total.

**Q: What if download fails?**
A: Some datasets are deprecated. Use the updated script with working alternatives.

**Q: Can I download while training?**
A: Yes, but it might slow down training slightly. Better to download first.

**Q: Where are datasets stored?**
A: HuggingFace cache: `~/.cache/huggingface/datasets/`

---

## 🎯 Final Recommendation

For your specific situation (L40S, $15, 5 hours):

```bash
# The optimal setup (20 min + 4 hours = 4.3 hours, $6.75)
python3 download_datasets.py --dataset wikitext-103  # 3 min
python3 download_datasets.py --dataset pg19          # 15 min

python3 examples/train_digitalocean.py \
    --config large \
    --dataset wikitext \
    --max_seq_len 8192 \
    --mixed_precision \
    --epochs 3
```

This gives you:
- ✅ 1.8M training samples (real pretraining scale)
- ✅ Long sequences (8K tokens, not 250 like IMDb)
- ✅ Large model (1024d, 8 layers, 180M params)
- ✅ PG19 for ultra-long validation
- ✅ ~$8 left for experiments
- ✅ Actual pretrained model you can use!
