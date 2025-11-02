# Google Colab Training Guide

## 🚀 Quick Start for Google Colab Training

This guide helps you train and benchmark **Temporal Eigenstate Networks (TEN)** on Google Colab's free T4 GPU.

---

## 📋 Prerequisites

1. **Google Account** - For accessing Google Colab
2. **GitHub Access** - To clone the repository (public or with token for private repos)
3. **Basic Python Knowledge** - Understanding of neural networks is helpful

---

## 🎯 Step-by-Step Instructions

### 1. Open the Notebook in Google Colab

**Option A: Upload Manually**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Upload `google_colab_training.ipynb`

**Option B: Open from GitHub**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Open notebook`
3. Select `GitHub` tab
4. Paste: `https://github.com/genovotechnologies/temporal-eigenstate-networks`
5. Select `examples/google_colab_training.ipynb`

### 2. Enable GPU

**CRITICAL:** You must enable GPU for reasonable training speed.

1. Click `Runtime` → `Change runtime type`
2. Select `T4 GPU` under "Hardware accelerator"
3. Click `Save`
4. Verify by running the GPU verification cell

### 3. Run the Notebook

**Recommended:** Run cells sequentially (Shift+Enter on each cell)

**Or:** Run all at once with `Runtime` → `Run all`

The notebook will:
- ✅ Install dependencies
- ✅ Clone the repository
- ✅ Download the IMDb dataset
- ✅ Train the model
- ✅ Generate benchmarks
- ✅ Create visualizations

---

## 🔑 Private Repository Access

If the repository is private, you need authentication:

### Step 1: Create GitHub Personal Access Token

1. Go to GitHub → Settings → Developer settings
2. Click "Personal access tokens" → "Tokens (classic)"
3. Click "Generate new token (classic)"
4. Select scope: `repo` (Full control of private repositories)
5. Copy the token (you won't see it again!)

### Step 2: Add Token to Colab

**Option A: Use Colab Secrets (Recommended)**
1. In Colab, click the 🔑 key icon in left sidebar
2. Click "Add new secret"
3. Name: `GITHUB_TOKEN`
4. Value: Paste your token
5. Enable notebook access

**Option B: Manual Entry**
- Uncomment the authentication cell in section 3.1
- Enter token when prompted (it will be hidden)

---

## ⚙️ Configuration Options

### Model Size (Section 5)

Adjust these for different model sizes:

```python
config = TemporalEigenstateConfig(
    d_model=256,           # Try: 128, 256, 512
    n_heads=8,             # Must divide d_model
    n_layers=4,            # Try: 2, 4, 6, 8
    num_eigenstates=64,    # Try: 32, 64, 128
    # ...
)
```

**Memory Guide:**
- **Tiny** (T4 safe): `d_model=128, n_layers=2, batch_size=32`
- **Small** (default): `d_model=256, n_layers=4, batch_size=16`
- **Medium**: `d_model=512, n_layers=6, batch_size=8`

### Training Settings (Section 6)

```python
BATCH_SIZE = 16         # Reduce if OOM errors
NUM_EPOCHS = 3          # Increase for better results
LEARNING_RATE = 3e-4    # Standard for Adam
MAX_SEQ_LENGTH = 512    # Reduce to 256 if needed
```

### Dataset Options (Section 4)

Use different datasets from Hugging Face:

```python
# Sentiment Analysis
dataset = load_dataset("imdb")  # Default

# Question Answering
dataset = load_dataset("squad")

# Text Classification
dataset = load_dataset("ag_news")

# Translation
dataset = load_dataset("wmt14", "de-en")
```

---

## 📊 Expected Results

### Training Performance (T4 GPU)

| Configuration | Training Time | Test Accuracy | Memory Usage |
|--------------|---------------|---------------|--------------|
| Tiny (2 layers) | ~5 min/epoch | ~82-85% | ~1.5 GB |
| Small (4 layers) | ~10 min/epoch | ~85-88% | ~2.5 GB |
| Medium (6 layers) | ~20 min/epoch | ~88-90% | ~4.5 GB |

### Benchmark Results (512 tokens)

- **Inference Time**: 10-20ms per batch
- **Memory Efficiency**: ~85% less than standard Transformer
- **Speedup**: 3-5× faster than equivalent Transformer

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory Error

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce `BATCH_SIZE` (try 8 or 4)
2. Reduce `MAX_SEQ_LENGTH` (try 256)
3. Reduce model size (`d_model=128, n_layers=2`)
4. Clear cache: `torch.cuda.empty_cache()`

### Issue 2: Import Errors

```
ModuleNotFoundError: No module named 'src'
```

**Solutions:**
1. Run the installation cell again
2. Restart runtime: `Runtime` → `Restart runtime`
3. Manually add to path:
   ```python
   import sys
   sys.path.insert(0, '/content/temporal-eigenstate-networks/src')
   ```

### Issue 3: No GPU Detected

```
CUDA available: False
```

**Solutions:**
1. Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`
2. Restart runtime
3. Check Colab quota (free tier has limits)

### Issue 4: Runtime Disconnected

**Prevention:**
1. Mount Google Drive (see troubleshooting section)
2. Models auto-save to Drive
3. Keep browser tab active
4. Consider Colab Pro for longer runtimes

### Issue 5: Slow Training

**Solutions:**
1. Verify GPU is being used (run `nvidia-smi`)
2. Increase `BATCH_SIZE` if memory allows
3. Use smaller dataset subset
4. Reduce `num_workers` in DataLoader

### Issue 6: GitHub Clone Failed

```
fatal: could not read Username
```

**For Public Repo:**
- Check repository URL is correct
- Ensure repository is public or you have access

**For Private Repo:**
- Use authentication (see section above)
- Verify token has `repo` scope
- Check token hasn't expired

---

## 💡 Tips & Best Practices

### 1. Start Small, Scale Up
- Begin with tiny model to verify everything works
- Gradually increase size after confirming success

### 2. Use Subset for Experimentation
```python
USE_SUBSET = True
SUBSET_SIZE = 1000  # Quick experiments
```

### 3. Save Checkpoints Regularly
- The notebook auto-saves after training
- Mount Google Drive for persistence

### 4. Monitor GPU Usage
```python
!nvidia-smi
```

### 5. Experiment with Hyperparameters
- Learning rate: Try 1e-4 to 5e-4
- Warmup steps: Add learning rate warmup
- Dropout: Adjust for regularization

### 6. Compare with Baseline
- Train a simple LSTM for comparison
- Use the benchmarking tools provided

---

## 📈 Advanced Usage

### Mixed Precision Training

Add AMP for faster training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

For larger effective batch size:

```python
ACCUMULATION_STEPS = 4

for i, batch in enumerate(train_loader):
    loss = loss / ACCUMULATION_STEPS
    loss.backward()
    
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Custom Datasets

```python
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        # Your custom logic here
        pass
```

---

## 📚 Additional Resources

### Documentation
- Main README: `/README.md`
- Architecture Details: `/docs/ARCHITECTURE_EVALUATION.md`
- Installation Guide: `/docs/INSTALLATION.md`

### Examples
- Text Classification: `/examples/text_classification_example.py`
- Sequence Labeling: `/examples/sequence_labeling_example.py`
- Time Series: `/examples/timeseries_example.py`

### Research Paper
- LaTeX source: `/paper/paper.tex`
- Theory and proofs included

---

## 🎓 Learning Path

1. **Beginner**: Run notebook with default settings
2. **Intermediate**: Adjust hyperparameters, try different datasets
3. **Advanced**: Modify architecture, implement new features
4. **Expert**: Contribute improvements, publish results

---

## 📞 Support

- **Issues**: Open GitHub issue with error details
- **Questions**: Include Colab output and configuration
- **Contributions**: Submit pull requests with improvements

---

## 📝 Citation

If you use this notebook in your research:

```bibtex
@article{afolabi2025ten,
  title={Temporal Eigenstate Networks: Linear-Complexity Sequence Modeling via Spectral Decomposition},
  author={Afolabi, Oluwatosin},
  year={2025},
  organization={Genovo Technologies}
}
```

---

## ⚖️ License

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.

**PROPRIETARY AND CONFIDENTIAL** - Internal Use Only

---

**Happy Training! 🚀**
