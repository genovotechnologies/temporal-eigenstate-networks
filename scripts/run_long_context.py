"""
Long-context benchmark: TEN-FFT vs TEN-Pro vs Transformer at T=8K-131K.
Tests whether TEN-Pro's cross-layer memory helps at very long sequences.
Also tests the Triton kernel at short sequences.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, json, sys, math, gc
sys.path.insert(0, '/home/USER/ten/experiments')

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

D, L, K = 512, 6, 64

def time_model(name, model, x, n_warmup=3, n_measure=10):
    model.eval()
    for _ in range(n_warmup):
        with torch.no_grad(): model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_measure):
        with torch.no_grad(): model(x)
    e.record()
    torch.cuda.synchronize()
    return round(s.elapsed_time(e)/n_measure, 1), round(torch.cuda.max_memory_allocated()/1e6, 0)

# SDPA Transformer for comparison
class SDPABlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads, self.hd = heads, d//heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
    def forward(self, x):
        B, T, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.heads, self.hd).permute(2,0,3,1,4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + self.out(out.transpose(1,2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))

class SDPALM(nn.Module):
    def __init__(self, vocab, d, layers, heads, max_T):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_T, d)
        self.blocks = nn.ModuleList([SDPABlock(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
    def forward(self, ids):
        B, T = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))

# ============================================================================
# 1. Triton vs FFT at short sequences
# ============================================================================
print('\n' + '='*80)
print('1. TRITON vs FFT (short sequences)')
print('='*80)

try:
    from models.triton_scan import triton_eigenstate_scan, HAS_TRITON
    from models.ten_fast import eigenstate_fft
    if HAS_TRITON:
        log_d = torch.randn(K).cuda() * -1.5
        freq = torch.linspace(0, 6.2, K).cuda()
        for T in [64, 128, 256, 512]:
            B = 16
            br = torch.randn(B, T, K).cuda()
            bi = torch.randn(B, T, K).cuda()
            # FFT
            for _ in range(3): eigenstate_fft(log_d, freq, br, bi)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(20): eigenstate_fft(log_d, freq, br, bi)
            e.record(); torch.cuda.synchronize()
            fft_ms = s.elapsed_time(e) / 20
            # Triton
            for _ in range(3): triton_eigenstate_scan(log_d, freq, br, bi)
            torch.cuda.synchronize()
            s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
            s2.record()
            for _ in range(20): triton_eigenstate_scan(log_d, freq, br, bi)
            e2.record(); torch.cuda.synchronize()
            tri_ms = s2.elapsed_time(e2) / 20
            print(f'  T={T:>4d} | FFT: {fft_ms:.2f}ms | Triton: {tri_ms:.2f}ms | Speedup: {fft_ms/tri_ms:.1f}x')
    else:
        print('  Triton not available')
except Exception as ex:
    print(f'  Triton test failed: {ex}')

# ============================================================================
# 2. TEN-FFT vs TEN-Pro vs Transformer at VERY long sequences
# ============================================================================
print('\n' + '='*80)
print('2. LONG-CONTEXT BENCHMARK (d=512, L=6)')
print('='*80)
print(f'{"T":>8s} | {"TEN-FFT":>10s} {"mem":>7s} | {"TEN-Pro":>10s} {"mem":>7s} | {"SDPA":>10s} {"mem":>7s}')
print('-' * 85)

results = {}

for T in [2048, 8192, 16384, 32768, 65536, 131072]:
    B = max(1, min(4, 131072 // T))
    x = torch.randint(0, 50257, (B, T)).cuda()
    results[T] = {}

    # TEN-FFT
    from models.ten_fast import TENFastModel
    m = TENFastModel(50257, d_model=D, n_layers=L, k_eigenstates=K,
                    n_heads=4, selective=False, max_seq_len=max(T+1,256)).cuda().float()
    try:
        ms, mem = time_model('TEN-FFT', m, x)
        results[T]['TEN-FFT'] = {'ms': ms, 'mem': mem}
        fft_str = f'{ms:>8.1f}ms {mem:>6.0f}MB'
    except Exception as e:
        fft_str = f'{"ERR":>8s} {"":>6s}  '
        results[T]['TEN-FFT'] = {'error': str(e)[:40]}
    del m; torch.cuda.empty_cache(); gc.collect()

    # TEN-Pro
    from models.ten_pro import TENProModel
    m = TENProModel(50257, d_model=D, n_layers=L, k_eigenstates=K,
                   n_heads=4, max_seq_len=max(T+1,256), use_cross_layer_memory=True).cuda().float()
    try:
        ms, mem = time_model('TEN-Pro', m, x)
        results[T]['TEN-Pro'] = {'ms': ms, 'mem': mem}
        pro_str = f'{ms:>8.1f}ms {mem:>6.0f}MB'
    except Exception as e:
        pro_str = f'{"ERR":>8s} {"":>6s}  '
        results[T]['TEN-Pro'] = {'error': str(e)[:40]}
    del m; torch.cuda.empty_cache(); gc.collect()

    # Transformer (SDPA)
    try:
        m = SDPALM(50257, D, L, 8, max(T+1,256)).cuda().float()
        ms, mem = time_model('SDPA', m, x)
        results[T]['SDPA'] = {'ms': ms, 'mem': mem}
        sdpa_str = f'{ms:>8.1f}ms {mem:>6.0f}MB'
        del m
    except Exception as e:
        sdpa_str = f'{"OOM":>8s} {"":>6s}  '
        results[T]['SDPA'] = {'ms': 'OOM'}
    torch.cuda.empty_cache(); gc.collect()

    print(f'{T:>8d} | {fft_str} | {pro_str} | {sdpa_str}')

# ============================================================================
# 3. Training comparison at long context (T=2048)
# ============================================================================
print('\n' + '='*80)
print('3. TRAINING at T=2048 (TEN-Pro vs TEN-FFT, 500 steps)')
print('='*80)

from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
text = ' '.join([t for t in ds['text'] if len(t) > 0])
tokens = tok.encode(text[:2000000])
SEQ = 2049  # 2048 + 1 for target
tokens = tokens[:len(tokens)//SEQ*SEQ]
data = torch.tensor(tokens).reshape(-1, SEQ).cuda()
print(f'Data: {data.shape[0]} seqs x {SEQ}')

def qtrain(name, model, steps=500, bs=4):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    t0 = time.time()
    final_loss = 99
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (bs,))
        batch = data[idx]
        logits = model(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            ppl = math.exp(min(loss.item(), 20))
            print(f'  [{name}] step {step} | loss {loss.item():.3f} | ppl {ppl:.1f} | {time.time()-t0:.0f}s', flush=True)
        final_loss = loss.item()
    ppl = math.exp(min(final_loss, 20))
    print(f'  [{name}] FINAL: loss={final_loss:.3f}, ppl={ppl:.1f}, {time.time()-t0:.0f}s')
    return {'loss': round(final_loss, 4), 'ppl': round(ppl, 1)}

train_results = {}

# TEN-FFT at T=2048
m = TENFastModel(tok.vocab_size, d_model=256, n_layers=4, k_eigenstates=32,
                n_heads=4, selective=False, max_seq_len=SEQ).cuda().float()
train_results['TEN-FFT'] = qtrain('TEN-FFT', m)
del m; torch.cuda.empty_cache()

# TEN-Pro at T=2048
m = TENProModel(tok.vocab_size, d_model=256, n_layers=4, k_eigenstates=32,
               n_heads=4, max_seq_len=SEQ, use_cross_layer_memory=True).cuda().float()
train_results['TEN-Pro'] = qtrain('TEN-Pro', m)
del m; torch.cuda.empty_cache()

results['training_T2048'] = train_results

# Save
with open('/home/USER/ten/experiments/long_context_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('\n' + '='*80)
print('LONG-CONTEXT BENCHMARK COMPLETE')
print('='*80)
print('Results saved to long_context_results.json')
