# Quick Reference: Updated API

## Configuration Changes

### New Parameter Names (use these)
```python
config = TemporalEigenstateConfig(
    vocab_size=50000,                      # Vocabulary size
    dim=512,                                # Hidden dimension (was d_model)
    n_layers=6,                             # Number of layers
    num_eigenstates=128,                    # Number of eigenstates per cell
    num_cells=2,                            # Number of cells per block
    max_seq_len=2048,                       # Maximum sequence length
    dropout=0.1,                            # Dropout rate
    chunk_size=64,                          # Chunking for memory efficiency
    use_gradient_checkpointing=False,       # ⚠️ Default changed to False
    use_resonance=True,                     # Eigenstate coupling
    ffn_multiplier=4.0,                     # FFN dimension multiplier
    pos_emb_type="learned",                 # "learned" or "sinusoidal"
    magnitude_reg_weight=0.01,              # ✨ Renamed from energy_reg_weight
    use_hten=False,                         # Hierarchical TEN
    hten_scales=[1, 2, 4, 8],              # Multi-scale factors
    eigenvalue_clip=0.99,                   # Max eigenvalue magnitude
    eigenvalue_min=0.1,                     # Min eigenvalue magnitude
    resonance_epsilon=0.01,                 # Resonance constraint
    init_std=0.02,                          # Weight initialization std
)
```

## Loss Dictionary Keys

### New Keys (use these)
```python
loss_dict = model.compute_loss(x, targets, return_dict=True)

# Access loss components
total_loss = loss_dict['loss']              # Total loss (unchanged)
ce_loss = loss_dict['ce_loss']              # Cross-entropy loss (unchanged)
magnitude_loss = loss_dict['magnitude_loss'] # ✨ Was 'energy_loss'
```

## What Changed?

| Old Name | New Name | Reason |
|----------|----------|--------|
| `energy_reg_weight` | `magnitude_reg_weight` | More accurate - penalizes eigenvalue magnitudes, not true energy |
| `'energy_loss'` | `'magnitude_loss'` | Consistent with parameter rename |
| `use_gradient_checkpointing=True` | `use_gradient_checkpointing=False` | Incompatible with stateful operations |

## Common Configurations

### Small Model (for testing/demos)
```python
config = TemporalEigenstateConfig(
    vocab_size=5000,
    dim=256,
    n_layers=4,
    num_eigenstates=64,
    max_seq_len=128,
)
```

### Medium Model (for training)
```python
config = TemporalEigenstateConfig(
    vocab_size=50000,
    dim=512,
    n_layers=6,
    num_eigenstates=128,
    max_seq_len=2048,
    magnitude_reg_weight=0.01,
)
```

### Large Model (for production)
```python
config = TemporalEigenstateConfig(
    vocab_size=50257,
    dim=1024,
    n_layers=12,
    num_eigenstates=256,
    max_seq_len=4096,
    chunk_size=64,
    magnitude_reg_weight=0.01,
)
```

### HTEN Model (multi-scale)
```python
config = TemporalEigenstateConfig(
    vocab_size=50000,
    dim=512,
    n_layers=6,
    num_eigenstates=128,
    use_hten=True,
    hten_scales=[1, 2, 4, 8],
    max_seq_len=8192,  # Supports longer sequences
)
```

## Training Example

```python
from src.model import TemporalEigenstateNetwork, TemporalEigenstateConfig
import torch

# Create model
config = TemporalEigenstateConfig(
    vocab_size=50000,
    dim=512,
    n_layers=6,
    num_eigenstates=128,
    magnitude_reg_weight=0.01,  # ✨ New name
)
model = TemporalEigenstateNetwork(config)

# Training step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randint(0, 50000, (8, 512))
targets = torch.randint(0, 50000, (8, 512))

# Get loss with components
loss_dict = model.compute_loss(x, targets, return_dict=True)

# Print losses
print(f"Total: {loss_dict['loss']:.4f}")
print(f"CE: {loss_dict['ce_loss']:.4f}")
print(f"Magnitude: {loss_dict['magnitude_loss']:.4f}")  # ✨ New name

# Backward pass
optimizer.zero_grad()
loss_dict['loss'].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Generation Example

```python
model.eval()
start_tokens = torch.randint(0, 50000, (1, 10))

with torch.no_grad():
    generated = model.generate(
        start_tokens,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        use_cache=True,  # Enable state caching for efficiency
    )

print(f"Generated sequence shape: {generated.shape}")
```

## Key Points

1. **Always use `magnitude_reg_weight`** instead of `energy_reg_weight`
2. **Access `'magnitude_loss'`** in loss dict instead of `'energy_loss'`
3. **Don't enable `use_gradient_checkpointing`** - it causes issues with states
4. **Use `chunk_size` for memory efficiency** instead of checkpointing
5. **Enable state caching in generation** with `use_cache=True`

## Verification

Test your configuration:
```python
# Quick test
model = TemporalEigenstateNetwork(config)
x = torch.randint(0, config.vocab_size, (2, 64))
targets = torch.randint(0, config.vocab_size, (2, 64))

# Should work without errors
loss_dict = model.compute_loss(x, targets, return_dict=True)
print(f"✅ Config works! Keys: {list(loss_dict.keys())}")
```

Expected output:
```
✅ Config works! Keys: ['loss', 'ce_loss', 'magnitude_loss']
```

---

**Last Updated:** November 4, 2025  
**See Also:** `SCRIPT_UPDATES.md` for detailed migration guide
