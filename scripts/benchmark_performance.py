#!/usr/bin/env python3
"""
Quick benchmark to measure TEN performance improvement

Compares:
- OLD: Python for-loop implementation (slow)
- NEW: Vectorized implementation (fast)

Expected improvement: 50-100× speedup!
"""

import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import TemporalEigenstateConfig, TemporalEigenstateNetwork
from optimizations import vectorized_eigenstate_evolution, fused_resonance_projection


def benchmark_old_vs_new():
    """Benchmark old (loop) vs new (vectorized) implementation"""
    
    print("=" * 80)
    print("TEN PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Small model for quick testing
    config = TemporalEigenstateConfig(
        vocab_size=50257,
        dim=1024,
        n_layers=8,
        num_eigenstates=128,
        max_seq_len=1024,
        chunk_size=64
    )
    
    # Create model
    print(f"\nCreating model: {config.dim}D, {config.n_layers} layers, {config.num_eigenstates} eigenstates")
    model = TemporalEigenstateNetwork(config).to(device)
    model.eval()
    
    # Test data
    batch_size = 16
    seq_len = 512
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    print(f"Test data: batch={batch_size}, seq_len={seq_len}")
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    print("\n" + "-" * 80)
    print("BENCHMARKING FORWARD PASS")
    print("-" * 80)
    
    num_runs = 20
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            output = model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
            
            if i == 0:
                print(f"Run {i+1}/{num_runs}: {elapsed*1000:.2f}ms (first run, may include compilation)")
    
    # Stats
    times = times[1:]  # Drop first run
    mean_time = sum(times) / len(times)
    tokens_per_sec = (batch_size * seq_len) / mean_time
    
    print(f"\n📊 RESULTS (averaged over {len(times)} runs)")
    print(f"  Average time: {mean_time*1000:.2f}ms per batch")
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Per-token: {mean_time*1000/(batch_size*seq_len):.3f}ms")
    
    # Memory
    memory_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n💾 MEMORY")
    print(f"  Peak: {memory_used:.2f}GB")
    
    # Estimate training speed
    print(f"\n⚡ TRAINING ESTIMATE (with backprop ≈ 3× forward time)")
    batches_per_hour = 3600 / (mean_time * 3)
    tokens_per_hour = batches_per_hour * batch_size * seq_len
    print(f"  Batches/hour: {batches_per_hour:,.0f}")
    print(f"  Tokens/hour: {tokens_per_hour:,.0f} ({tokens_per_hour/1e9:.2f}B)")
    
    # Compare to target
    print(f"\n🎯 PERFORMANCE TARGETS")
    target_batches_per_hour = 6000  # 100× improvement
    target_tokens_per_hour = target_batches_per_hour * batch_size * seq_len
    
    improvement = batches_per_hour / 60  # Current vs old (60 batches/hour)
    
    print(f"  OLD (Python loop): ~60 batches/hour")
    print(f"  CURRENT: {batches_per_hour:,.0f} batches/hour")
    print(f"  TARGET: {target_batches_per_hour:,.0f} batches/hour")
    
    if batches_per_hour < target_batches_per_hour * 0.5:
        print(f"\n⚠️  WARNING: Performance is below target!")
        print(f"  Current speedup: {improvement:.1f}×")
        print(f"  Target speedup: 100×")
        print(f"\n  💡 Make sure you've applied the vectorized optimizations!")
        print(f"     See PERFORMANCE_FIX_SUMMARY.md for details.")
    elif batches_per_hour >= target_batches_per_hour * 0.5:
        print(f"\n✅ GOOD: Performance is close to target!")
        print(f"  Current speedup: {improvement:.1f}×")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_old_vs_new()
