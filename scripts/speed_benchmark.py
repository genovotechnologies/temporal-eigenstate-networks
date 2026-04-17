"""Full benchmark: TEN-Chunked vs TEN-FFT vs Transformer vs FlashAttention Transformer"""
import torch, torch.nn as nn, time, sys, math
sys.path.insert(0, '/home/USER/ten/experiments')

def time_model(name, model, x, n_warmup=5, n_measure=30):
    model.eval()
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_measure):
        with torch.no_grad():
            model(x)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n_measure
    mem = torch.cuda.max_memory_allocated() / 1e6
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'{name:40s} | {ms:9.2f}ms | {mem:8.0f}MB | {params:.1f}M')
    return ms

# Check FlashAttention availability
try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
    print("FlashAttention-2 available")
except ImportError:
    HAS_FLASH = False
    print("FlashAttention-2 NOT available (install: pip install flash-attn --no-build-isolation)")

print()
print('=' * 80)
print(f'{"Model":40s} | {"Time":>9s} | {"Mem":>8s} | Params')
print('=' * 80)

D_MODEL = 512
N_LAYERS = 6
K = 64
N_HEADS = 8

for T in [512, 1024, 2048, 4096, 8192]:
    B = max(1, min(16, 32768 // T))
    print(f'\n--- T={T}, B={B} (d={D_MODEL}, L={N_LAYERS}) ---')
    x = torch.randint(0, 50257, (B, T)).cuda()

    # TEN-Chunked (our best)
    from models.ten_chunked import TENChunkedModel
    for cs in [64]:
        m = TENChunkedModel(50257, d_model=D_MODEL, n_layers=N_LAYERS, k_eigenstates=K,
                            n_heads=4, chunk_size=cs, max_seq_len=max(T+1, 256)).cuda().float()
        try:
            time_model(f'TEN-Chunked (C={cs})', m, x)
        except Exception as ex:
            print(f'TEN-Chunked (C={cs}): ERROR - {ex}')
        del m; torch.cuda.empty_cache()

    # TEN-FFT
    from models.ten_fast import TENFastModel
    m = TENFastModel(50257, d_model=D_MODEL, n_layers=N_LAYERS, k_eigenstates=K,
                     n_heads=4, selective=False, max_seq_len=max(T+1, 256)).cuda().float()
    try:
        time_model('TEN-FFT', m, x)
    except Exception as ex:
        print(f'TEN-FFT: ERROR - {ex}')
    del m; torch.cuda.empty_cache()

    # Transformer (PyTorch SDPA — dispatches to FlashAttn if available)
    class TransformerLM(nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.emb = nn.Embedding(50257, D_MODEL)
            self.pos = nn.Embedding(seq, D_MODEL)
            layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL*4, 0.1,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, N_LAYERS)
            self.norm = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, 50257, bias=False)
            self.head.weight = self.emb.weight
        def forward(self, ids):
            b, t = ids.shape
            h = self.emb(ids) + self.pos(torch.arange(t, device=ids.device))
            mask = nn.Transformer.generate_square_subsequent_mask(t, device=ids.device)
            return self.head(self.norm(self.enc(h, mask=mask, is_causal=True)))

    m = TransformerLM(max(T+1, 256)).cuda().float()
    try:
        time_model('Transformer (SDPA)', m, x)
    except Exception as ex:
        print(f'Transformer: ERROR - {ex}')
    del m; torch.cuda.empty_cache()

print('\nDone.')
