"""
Definitive benchmark: prove TEN's speedup claims.

Tests:
1. TEN-FFT vs Naive Attention vs SDPA at T=512, 2048, 8192, 32768, 65536
2. Triton kernel vs FFT vs naive sequential
3. Attention-specific memory measurement (not total GPU memory)
4. Memory scaling analysis
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, json, sys, math, gc
sys.path.insert(0, '/home/USER/ten/experiments')

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

D_MODEL = 512
N_LAYERS = 6
K = 64
RESULTS = {}

def time_fn(fn, n_warmup=3, n_measure=10):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_measure): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_measure, torch.cuda.max_memory_allocated() / 1e6


# ============================================================================
# 1. NAIVE ATTENTION (materializes T×T matrix — the original paper's baseline)
# ============================================================================

class NaiveAttention(nn.Module):
    """Deliberately naive attention that materializes the full T×T matrix."""
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.hd = d // heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.scale = self.hd ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.hd).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Explicitly compute and store the T×T attention matrix
        attn = (q @ k.transpose(-2,-1)) * self.scale  # (B, H, T, T) — THIS is the bottleneck
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, T, D)
        return self.out(out)


class NaiveTransformerLM(nn.Module):
    def __init__(self, vocab, d, L, heads, max_T):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_T, d)
        self.blocks = nn.ModuleList()
        for _ in range(L):
            self.blocks.append(nn.ModuleDict({
                'n1': nn.LayerNorm(d), 'n2': nn.LayerNorm(d),
                'attn': NaiveAttention(d, heads),
                'mlp': nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d)),
            }))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids):
        B, T = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            x = x + b['attn'](b['n1'](x))
            x = x + b['mlp'](b['n2'](x))
        return self.head(self.norm(x))


# ============================================================================
# 2. SDPA TRANSFORMER (optimized baseline)
# ============================================================================

class SDPATransformerLM(nn.Module):
    def __init__(self, vocab, d, L, heads, max_T):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_T, d)
        self.blocks = nn.ModuleList()
        for _ in range(L):
            self.blocks.append(nn.ModuleDict({
                'n1': nn.LayerNorm(d), 'n2': nn.LayerNorm(d),
                'qkv': nn.Linear(d, 3*d, bias=False),
                'out': nn.Linear(d, d, bias=False),
                'mlp': nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d)),
            }))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.heads = heads
        self.hd = d // heads

    def forward(self, ids):
        B, T = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            h = b['n1'](x)
            qkv = b['qkv'](h).reshape(B, T, 3, self.heads, self.hd).permute(2,0,3,1,4)
            out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
            x = x + b['out'](out.transpose(1,2).reshape(B, T, -1))
            x = x + b['mlp'](b['n2'](x))
        return self.head(self.norm(x))


# ============================================================================
# 3. MEASURE ATTENTION-SPECIFIC MEMORY
# ============================================================================

def measure_attention_memory(T, B=1):
    """Measure memory used by T×T attention matrix alone."""
    # Attention matrix: (B, H, T, T) in float32
    attn_bytes = B * 8 * T * T * 4  # 8 heads, float32
    attn_mb = attn_bytes / 1e6

    # TEN eigenstate memory: (B, T, K) complex = (B, T, 2K) real
    ten_bytes = B * T * K * 2 * 4  # 2K for complex, float32
    ten_mb = ten_bytes / 1e6

    ratio = attn_mb / ten_mb if ten_mb > 0 else float('inf')
    return attn_mb, ten_mb, ratio


# ============================================================================
# 4. TRITON KERNEL BENCHMARK
# ============================================================================

def benchmark_triton():
    print('\n' + '='*80)
    print('TRITON KERNEL BENCHMARK')
    print('='*80)
    try:
        from models.triton_scan import triton_eigenstate_scan, HAS_TRITON
        if not HAS_TRITON:
            print('Triton not available, skipping')
            return {}
    except ImportError:
        print('Triton kernel not found, skipping')
        return {}

    from models.ten_fast import eigenstate_fft

    results = {}
    log_d = torch.randn(K).cuda().float() * -1.5
    freq = torch.linspace(0, 2*math.pi*0.99, K).cuda().float()

    for T in [512, 2048, 8192, 32768]:
        B = max(1, min(8, 32768 // T))
        br = torch.randn(B, T, K).cuda().float()
        bi = torch.randn(B, T, K).cuda().float()

        # FFT
        ms_fft, _ = time_fn(lambda: eigenstate_fft(log_d, freq, br, bi))
        # Triton
        ms_tri, _ = time_fn(lambda: triton_eigenstate_scan(log_d, freq, br, bi))

        results[T] = {'fft_ms': round(ms_fft, 2), 'triton_ms': round(ms_tri, 2),
                      'speedup': round(ms_fft / ms_tri, 2) if ms_tri > 0 else 0}
        print(f'  T={T:>6d} | FFT: {ms_fft:.2f}ms | Triton: {ms_tri:.2f}ms | '
              f'Speedup: {ms_fft/ms_tri:.1f}x')

    return results


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

print('\n' + '='*80)
print('DEFINITIVE SPEED + MEMORY BENCHMARK')
print('='*80)
print(f'Config: d={D_MODEL}, L={N_LAYERS}, K={K}')
print(f'{"":>6s} | {"TEN-FFT":>10s} | {"Naive Attn":>10s} | {"SDPA":>10s} | {"vs Naive":>8s} | {"vs SDPA":>8s} | {"Mem Ratio":>9s}')
print('-' * 80)

speed_results = {}

for T in [512, 2048, 8192, 16384, 32768, 65536]:
    B = max(1, min(8, 65536 // T))
    x = torch.randint(0, 50257, (B, T)).cuda()
    speed_results[T] = {}

    # Memory analysis (theoretical)
    attn_mem, ten_mem, mem_ratio = measure_attention_memory(T, B)

    # TEN-FFT
    from models.ten_fast import TENFastModel
    m = TENFastModel(50257, d_model=D_MODEL, n_layers=N_LAYERS, k_eigenstates=K,
                    n_heads=4, selective=False, max_seq_len=max(T+1,256)).cuda().float()
    try:
        ms_ten, mem_ten = time_fn(lambda: m(x))
        speed_results[T]['TEN-FFT'] = {'ms': round(ms_ten, 1), 'mem': mem_ten}
    except Exception as e:
        ms_ten = float('inf')
        speed_results[T]['TEN-FFT'] = {'error': str(e)[:50]}
    del m; torch.cuda.empty_cache(); gc.collect()

    # Naive Transformer
    try:
        m = NaiveTransformerLM(50257, D_MODEL, N_LAYERS, 8, max(T+1,256)).cuda().float()
        ms_naive, mem_naive = time_fn(lambda: m(x))
        speed_results[T]['Naive'] = {'ms': round(ms_naive, 1), 'mem': mem_naive}
        del m; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        ms_naive = float('inf')
        speed_results[T]['Naive'] = {'ms': 'OOM', 'mem': 'OOM'}
        torch.cuda.empty_cache(); gc.collect()

    # SDPA Transformer
    try:
        m = SDPATransformerLM(50257, D_MODEL, N_LAYERS, 8, max(T+1,256)).cuda().float()
        ms_sdpa, mem_sdpa = time_fn(lambda: m(x))
        speed_results[T]['SDPA'] = {'ms': round(ms_sdpa, 1), 'mem': mem_sdpa}
        del m; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        ms_sdpa = float('inf')
        speed_results[T]['SDPA'] = {'ms': 'OOM', 'mem': 'OOM'}
        torch.cuda.empty_cache(); gc.collect()

    vs_naive = f'{ms_naive/ms_ten:.1f}x' if ms_ten < float('inf') and ms_naive < float('inf') else 'N/A'
    vs_sdpa = f'{ms_sdpa/ms_ten:.1f}x' if ms_ten < float('inf') and ms_sdpa < float('inf') else 'N/A'

    ten_str = f'{ms_ten:.1f}ms' if ms_ten < float('inf') else 'ERR'
    naive_str = f'{ms_naive:.1f}ms' if ms_naive < float('inf') else 'OOM'
    sdpa_str = f'{ms_sdpa:.1f}ms' if ms_sdpa < float('inf') else 'OOM'

    print(f'T={T:>5d} | {ten_str:>10s} | {naive_str:>10s} | {sdpa_str:>10s} | {vs_naive:>8s} | {vs_sdpa:>8s} | {mem_ratio:>8.0f}x')

RESULTS['speed'] = speed_results

# Memory scaling table
print('\n' + '='*80)
print('ATTENTION-SPECIFIC MEMORY (theoretical, per layer)')
print('='*80)
print(f'{"T":>8s} | {"Attn Matrix":>12s} | {"TEN States":>12s} | {"Ratio":>8s}')
for T in [512, 2048, 8192, 32768, 65536, 131072]:
    attn_mb, ten_mb, ratio = measure_attention_memory(T, 1)
    print(f'{T:>8d} | {attn_mb:>10.1f}MB | {ten_mb:>10.1f}MB | {ratio:>7.0f}x')
RESULTS['memory_analysis'] = {T: measure_attention_memory(T, 1) for T in [512, 2048, 8192, 32768, 65536, 131072]}

# Triton kernel
triton_results = benchmark_triton()
RESULTS['triton'] = triton_results

# Save
with open('/home/USER/ten/experiments/definitive_results.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)

print('\n' + '='*80)
print('DEFINITIVE BENCHMARK COMPLETE')
print('='*80)
print('Results saved to definitive_results.json')
