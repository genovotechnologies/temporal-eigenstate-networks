"""
Unified benchmark: TEN vs all baselines on WikiText-103.
Reports perplexity, throughput (tokens/sec), peak memory, and time/batch.

Usage:
    python benchmark.py --model ten --seq_len 2048 --batch_size 32
    python benchmark.py --model transformer_flash --seq_len 2048 --batch_size 32
    python benchmark.py --model mamba --seq_len 2048 --batch_size 32
    python benchmark.py --model all --seq_len 2048 --batch_size 32
"""

import argparse
import json
import time
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# ============================================================================
# Model factories
# ============================================================================

def build_ten(vocab_size, d_model=512, n_layers=6, k=64, **kw):
    from models.ten import TENModel
    return TENModel(vocab_size, d_model=d_model, n_layers=n_layers,
                    k_eigenstates=k, max_seq_len=kw.get('max_seq_len', 8192))


def build_transformer_flash(vocab_size, d_model=512, n_layers=6, n_heads=8, **kw):
    """Standard transformer with FlashAttention-2."""
    from torch.nn import TransformerEncoderLayer, TransformerEncoder

    # Use PyTorch's native SDPA which dispatches to FlashAttention-2
    # when flash-attn is installed
    config = {
        'd_model': d_model, 'nhead': n_heads, 'dim_feedforward': d_model * 4,
        'dropout': 0.1, 'batch_first': True, 'norm_first': True,
    }
    encoder_layer = TransformerEncoderLayer(**config)
    encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

    class TransformerLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(kw.get('max_seq_len', 8192), d_model)
            self.encoder = encoder
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight
            self.d_model = d_model

        def forward(self, input_ids):
            B, T = input_ids.shape
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = self.token_emb(input_ids) + self.pos_emb(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
            x = self.encoder(x, mask=mask, is_causal=True)
            return self.lm_head(self.norm(x))

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return TransformerLM()


def build_mamba(vocab_size, d_model=512, n_layers=6, **kw):
    """Mamba (S6) baseline using mamba-ssm package."""
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig

        config = MambaConfig(
            d_model=d_model,
            n_layer=n_layers,
            vocab_size=vocab_size,
        )
        model = MambaLMHeadModel(config)
        model.count_parameters = lambda: sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model
    except ImportError:
        print("ERROR: mamba-ssm not installed. Run: pip install mamba-ssm causal-conv1d")
        sys.exit(1)


MODEL_REGISTRY = {
    'ten': build_ten,
    'transformer_flash': build_transformer_flash,
    'mamba': build_mamba,
}


# ============================================================================
# Data
# ============================================================================

def get_wikitext103_dataloader(seq_len: int, batch_size: int, split: str = 'test'):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    text = '\n'.join([t for t in dataset['text'] if len(t) > 0])
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Chunk into sequences
    n_chunks = len(tokens) // seq_len
    tokens = tokens[:n_chunks * seq_len]
    token_tensor = torch.tensor(tokens).reshape(n_chunks, seq_len)

    loader = DataLoader(token_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader, tokenizer.vocab_size


# ============================================================================
# Measurement utilities
# ============================================================================

def measure_throughput(model, input_ids, n_warmup=3, n_measure=10):
    """Measure tokens/sec and peak memory using CUDA events."""
    device = input_ids.device
    B, T = input_ids.shape

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)

    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_measure):
        with torch.no_grad():
            _ = model(input_ids)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    time_per_batch_ms = elapsed_ms / n_measure
    tokens_per_sec = (B * T * 1000) / time_per_batch_ms
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        'time_per_batch_ms': round(time_per_batch_ms, 1),
        'tokens_per_sec': round(tokens_per_sec, 0),
        'peak_memory_mb': round(peak_memory_mb, 1),
    }


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, max_batches=50):
    """Evaluate perplexity on WikiText-103 test set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return round(perplexity, 2)


# ============================================================================
# Main
# ============================================================================

def run_benchmark(model_name, seq_len, batch_size, device='cuda'):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} | seq_len={seq_len} | batch={batch_size}")
    print(f"{'='*60}")

    # Load data
    print("Loading WikiText-103...")
    dataloader, vocab_size = get_wikitext103_dataloader(seq_len + 1, batch_size, split='test')

    # Build model
    print(f"Building {model_name}...")
    build_fn = MODEL_REGISTRY[model_name]
    model = build_fn(vocab_size, max_seq_len=seq_len + 1).to(device)

    n_params = model.count_parameters()
    print(f"Parameters: {n_params / 1e6:.1f}M")

    # Throughput measurement
    print("Measuring throughput...")
    sample_batch = next(iter(dataloader))[:, :-1].to(device)
    try:
        perf = measure_throughput(model, sample_batch)
        print(f"  Time/batch: {perf['time_per_batch_ms']}ms")
        print(f"  Tokens/sec: {perf['tokens_per_sec']:,.0f}")
        print(f"  Peak memory: {perf['peak_memory_mb']:.0f}MB")
    except torch.cuda.OutOfMemoryError:
        perf = {'time_per_batch_ms': 'OOM', 'tokens_per_sec': 'OOM', 'peak_memory_mb': 'OOM'}
        print(f"  OOM at batch_size={batch_size}, seq_len={seq_len}")

    # Perplexity (on untrained model — just validates forward pass works)
    # For real results, train first then evaluate
    print("Evaluating perplexity (untrained — forward pass validation)...")
    try:
        ppl = evaluate_perplexity(model, dataloader, device, max_batches=5)
        print(f"  Perplexity (untrained): {ppl}")
    except torch.cuda.OutOfMemoryError:
        ppl = 'OOM'
        print(f"  OOM during evaluation")

    results = {
        'model': model_name,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'params_M': round(n_params / 1e6, 1),
        'perplexity': ppl,
        **perf,
        'gpu': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }

    print(f"\nResults: {json.dumps(results, indent=2)}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ten', choices=list(MODEL_REGISTRY.keys()) + ['all'])
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='results.json')
    args = parser.parse_args()

    all_results = []

    if args.model == 'all':
        for name in MODEL_REGISTRY:
            try:
                result = run_benchmark(name, args.seq_len, args.batch_size, args.device)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR running {name}: {e}")
                all_results.append({'model': name, 'error': str(e)})
            torch.cuda.empty_cache()
    else:
        result = run_benchmark(args.model, args.seq_len, args.batch_size, args.device)
        all_results.append(result)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
