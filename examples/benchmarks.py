"""Benchmarking script for Temporal Eigenstate Networks."""

import torch
import time
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import TemporalEigenstateNetwork, TemporalEigenstateConfig
from eval import Evaluator


class StandardAttention(torch.nn.Module):
    """Standard scaled dot-product attention for comparison."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


def benchmark_attention_mechanisms(
    seq_lengths: List[int],
    d_model: int = 512,
    n_heads: int = 8,
    batch_size: int = 16,
    num_runs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, List[float]]:
    """Compare TEN attention with standard attention."""
    
    print(f"Running benchmarks on {device}...")
    
    # Create models
    config = TemporalEigenstateConfig(d_model=d_model, n_heads=n_heads, n_layers=1)
    ten_model = TemporalEigenstateNetwork(config).to(device)
    standard_model = StandardAttention(d_model, n_heads).to(device)
    
    results = {
        "seq_lengths": seq_lengths,
        "ten_time": [],
        "standard_time": [],
        "ten_memory": [],
        "standard_memory": [],
    }
    
    for seq_len in seq_lengths:
        print(f"\nBenchmarking sequence length: {seq_len}")
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # Benchmark TEN
        ten_model.eval()
        with torch.no_grad():
            # Warm-up
            _ = ten_model(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = ten_model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            results["ten_time"].append(np.mean(times))
            
            if device == "cuda":
                memory = torch.cuda.max_memory_allocated() / 1024**2
                results["ten_memory"].append(memory)
        
        # Benchmark Standard Attention
        standard_model.eval()
        with torch.no_grad():
            # Warm-up
            _ = standard_model(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = standard_model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            results["standard_time"].append(np.mean(times))
            
            if device == "cuda":
                memory = torch.cuda.max_memory_allocated() / 1024**2
                results["standard_memory"].append(memory)
        
        print(f"  TEN: {results['ten_time'][-1]:.4f}s")
        print(f"  Standard: {results['standard_time'][-1]:.4f}s")
        print(f"  Speedup: {results['standard_time'][-1] / results['ten_time'][-1]:.2f}x")
    
    return results


def plot_benchmark_results(results: Dict[str, List[float]]):
    """Plot benchmark comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    seq_lengths = results["seq_lengths"]
    
    # Time comparison
    ax1.plot(seq_lengths, results["ten_time"], marker='o', label='TEN', linewidth=2)
    ax1.plot(seq_lengths, results["standard_time"], marker='s', label='Standard', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Forward Pass Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup
    speedup = [s / t for s, t in zip(results["standard_time"], results["ten_time"])]
    ax2.plot(seq_lengths, speedup, marker='o', color='green', linewidth=2)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('TEN Speedup over Standard Attention')
    ax2.grid(True, alpha=0.3)
    
    # Memory comparison
    if results["ten_memory"] and results["standard_memory"]:
        ax3.plot(seq_lengths, results["ten_memory"], marker='o', label='TEN', linewidth=2)
        ax3.plot(seq_lengths, results["standard_memory"], marker='s', label='Standard', linewidth=2)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('Memory Usage Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Memory savings
        memory_savings = [100 * (1 - t / s) for s, t in zip(results["standard_memory"], results["ten_memory"])]
        ax4.plot(seq_lengths, memory_savings, marker='o', color='purple', linewidth=2)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Memory Savings (%)')
        ax4.set_title('TEN Memory Savings vs Standard Attention')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'benchmark_results.png'")
    plt.show()


def main():
    """Run comprehensive benchmarks."""
    print("=" * 60)
    print("Temporal Eigenstate Networks - Comprehensive Benchmark")
    print("=" * 60)
    
    # Configuration
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    d_model = 512
    n_heads = 8
    batch_size = 16
    num_runs = 10
    
    # Run benchmarks
    results = benchmark_attention_mechanisms(
        seq_lengths=seq_lengths,
        d_model=d_model,
        n_heads=n_heads,
        batch_size=batch_size,
        num_runs=num_runs,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for i, seq_len in enumerate(results["seq_lengths"]):
        speedup = results["standard_time"][i] / results["ten_time"][i]
        print(f"\nSequence Length: {seq_len}")
        print(f"  TEN Time:       {results['ten_time'][i]:.4f}s")
        print(f"  Standard Time:  {results['standard_time'][i]:.4f}s")
        print(f"  Speedup:        {speedup:.2f}x")
        
        if results["ten_memory"] and results["standard_memory"]:
            memory_saved = results["standard_memory"][i] - results["ten_memory"][i]
            memory_percent = 100 * memory_saved / results["standard_memory"][i]
            print(f"  Memory Saved:   {memory_saved:.2f}MB ({memory_percent:.1f}%)")
    
    # Plot results
    plot_benchmark_results(results)


if __name__ == "__main__":
    main()
