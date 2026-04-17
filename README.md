# Temporal Eigenstate Networks (TEN)

Linear-complexity sequence modeling via spectral decomposition. Accepted at AAAI-26 AIDD Workshop (original TEN paper). This repo contains the improved implementation with FFT-based and chunked matmul backends.

**Paper:** [Temporal Eigenstate Networks: Linear-Complexity Sequence Modeling via Spectral Decomposition](paper/ten.pdf)

## Results

Inference time (ms) on NVIDIA A100 40GB, d=512, 6 layers, ~42M params:

| Model | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
|---|---|---|---|---|---|
| **TEN-FFT** | **40** | **79** | **153** | **154** | **154** |
| Transformer (SDPA) | 50 | 114 | 307 | 458 | OOM |
| Speedup | 1.2x | 1.5x | 2.0x | 3.0x | -- |

TEN-FFT inference time is nearly constant from T=2048 to T=8192 because the O(T log T) FFT cost is negligible relative to the projection overhead.

## Quick Start

```python
from ten import TENFastModel

model = TENFastModel(
    vocab_size=50257,
    d_model=512,
    n_layers=6,
    k_eigenstates=64,
    n_heads=4,
)

input_ids = torch.randint(0, 50257, (1, 2048))
logits = model(input_ids)  # (1, 2048, 50257)
```

## Architecture Variants

| Variant | Backend | Best for |
|---|---|---|
| `TENFastModel` | FFT convolution | Fixed eigenvalues, maximum speed |
| `TENChunkedModel` | Chunked matmul | GPU-friendly, good hardware utilization |
| `TENProModel` | FFT + cross-layer memory | Best quality, adaptive depth |

## How It Works

TEN replaces attention with learned eigenstate decomposition:

1. **Project** input to K complex-valued eigenstate amplitudes
2. **Evolve** each eigenstate via diagonal recurrence: `c_k(t) = λ_k · c_k(t-1) + β_k(t)`
3. **Mix** eigenstates through block-diagonal coupling
4. **Reconstruct** hidden state via gated output projection

For fixed eigenvalues, step 2 is a causal convolution computed via FFT in O(T log T).

## Training

```bash
# Install
pip install -e ".[train]"

# Train on WikiText-2 (quick validation)
python scripts/run_all.py

# Train on WikiText-103 (full)
python scripts/train.py --model ten_fft --seq_len 2048 --batch_size 16 --hours 8 --bf16
```

## Citation

```bibtex
@inproceedings{afolabi2026ten,
  title={Temporal Eigenstate Networks: Linear-Complexity Sequence Modeling via Spectral Decomposition},
  author={Afolabi, Oluwatosin},
  booktitle={AAAI-26 AIDD Workshop},
  year={2026}
}
```

## License

MIT
