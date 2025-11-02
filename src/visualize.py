"""
Visualization utilities for Temporal Eigenstate Networks.

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL - Internal Use Only

This module provides tools for visualizing:
- Eigenstate dynamics over time
- Eigenvalue distributions
- Temporal flow patterns
- Model activations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import seaborn as sns

from .model import TemporalEigenstateNetwork, TemporalFlowCell


def plot_eigenvalue_spectrum(
    model: TemporalEigenstateNetwork,
    block_idx: int = 0,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
):
    """
    Visualize eigenvalue spectrum (magnitude and phase) for a specific block.
    
    Args:
        model: TEN model
        block_idx: Which block to visualize
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    block = model.blocks[block_idx]
    
    # Collect eigenvalues from all cells in the block
    all_magnitudes = []
    all_phases = []
    
    for cell in block.cells:
        magnitude, phase = cell.get_eigenvalues()
        all_magnitudes.append(magnitude.detach().cpu().numpy())
        all_phases.append(phase.detach().cpu().numpy())
    
    # Plot magnitudes
    axes[0].set_title(f'Eigenvalue Magnitudes (Block {block_idx})')
    for i, mag in enumerate(all_magnitudes):
        axes[0].plot(mag, label=f'Cell {i}', marker='o', markersize=4)
    axes[0].set_xlabel('Eigenstate Index')
    axes[0].set_ylabel('Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Plot phases
    axes[1].set_title(f'Eigenvalue Phases (Block {block_idx})')
    for i, phase in enumerate(all_phases):
        axes[1].plot(phase, label=f'Cell {i}', marker='o', markersize=4)
    axes[1].set_xlabel('Eigenstate Index')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].axhline(y=np.pi, color='k', linestyle='--', alpha=0.3)
    axes[1].axhline(y=2*np.pi, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvalue spectrum to {save_path}")
    
    return fig


def plot_eigenstate_trajectory(
    model: TemporalEigenstateNetwork,
    tokens: torch.Tensor,
    block_idx: int = 0,
    cell_idx: int = 0,
    max_states: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot eigenstate activations over time for a given sequence.
    
    Args:
        model: TEN model
        tokens: Input token sequence (1, seq_len)
        block_idx: Which block to visualize
        cell_idx: Which cell within the block
        max_states: Maximum number of eigenstates to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    model.eval()
    
    # Track state evolution
    states_real = []
    states_imag = []
    
    cell = model.blocks[block_idx].cells[cell_idx]
    
    # Hook to capture states
    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            out, (real, imag) = output
            states_real.append(real[0].detach().cpu().numpy())  # First batch element
            states_imag.append(imag[0].detach().cpu().numpy())
    
    handle = cell.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(tokens)
    
    handle.remove()
    
    if not states_real:
        print("No states captured")
        return None
    
    # Convert to arrays
    states_real = np.array(states_real)  # (seq_len, num_eigenstates)
    states_imag = np.array(states_imag)
    
    # Compute magnitude
    states_mag = np.sqrt(states_real**2 + states_imag**2)
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Select top eigenstates by average magnitude
    top_indices = states_mag.mean(axis=0).argsort()[-max_states:][::-1]
    
    # Plot real part
    axes[0].set_title(f'Real Part of Eigenstates (Block {block_idx}, Cell {cell_idx})')
    for idx in top_indices:
        axes[0].plot(states_real[:, idx], label=f'State {idx}', alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Real Component')
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot imaginary part
    axes[1].set_title(f'Imaginary Part of Eigenstates')
    for idx in top_indices:
        axes[1].plot(states_imag[:, idx], label=f'State {idx}', alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Imaginary Component')
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Plot magnitude
    axes[2].set_title(f'Magnitude of Eigenstates')
    for idx in top_indices:
        axes[2].plot(states_mag[:, idx], label=f'State {idx}', alpha=0.7, linewidth=2)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Magnitude')
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenstate trajectory to {save_path}")
    
    return fig


def plot_eigenstate_heatmap(
    model: TemporalEigenstateNetwork,
    tokens: torch.Tensor,
    block_idx: int = 0,
    cell_idx: int = 0,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Create a heatmap of all eigenstate activations over time.
    
    Args:
        model: TEN model
        tokens: Input token sequence (1, seq_len)
        block_idx: Which block to visualize
        cell_idx: Which cell within the block
        figsize: Figure size
        save_path: Optional path to save figure
    """
    model.eval()
    
    states_real = []
    states_imag = []
    
    cell = model.blocks[block_idx].cells[cell_idx]
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            out, (real, imag) = output
            states_real.append(real[0].detach().cpu().numpy())
            states_imag.append(imag[0].detach().cpu().numpy())
    
    handle = cell.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(tokens)
    
    handle.remove()
    
    if not states_real:
        return None
    
    states_real = np.array(states_real).T  # (num_eigenstates, seq_len)
    states_imag = np.array(states_imag).T
    states_mag = np.sqrt(states_real**2 + states_imag**2)
    
    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Real part heatmap
    sns.heatmap(
        states_real,
        ax=axes[0],
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Activation'},
        xticklabels=max(1, states_real.shape[1] // 10),
        yticklabels=max(1, states_real.shape[0] // 10)
    )
    axes[0].set_title(f'Real Part (Block {block_idx}, Cell {cell_idx})')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Eigenstate Index')
    
    # Magnitude heatmap
    sns.heatmap(
        states_mag,
        ax=axes[1],
        cmap='viridis',
        cbar_kws={'label': 'Magnitude'},
        xticklabels=max(1, states_mag.shape[1] // 10),
        yticklabels=max(1, states_mag.shape[0] // 10)
    )
    axes[1].set_title(f'Magnitude')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Eigenstate Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenstate heatmap to {save_path}")
    
    return fig


def compare_model_scales(
    results: Dict[str, Dict],
    metric: str = 'forward_time',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Compare efficiency metrics across different model configurations.
    
    Args:
        results: Dictionary mapping model names to their efficiency results
        metric: Which metric to plot ('forward_time' or 'memory_usage')
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, result in results.items():
        seq_lengths = result['seq_lengths']
        values = result[metric]
        ax.plot(seq_lengths, values, marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    
    if metric == 'forward_time':
        ax.set_ylabel('Forward Time (seconds)', fontsize=12)
        ax.set_title('Computational Efficiency Comparison', fontsize=14)
    else:
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Efficiency Comparison', fontsize=14)
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    from .model import TemporalEigenstateConfig
    import os
    
    print("Testing TEN Visualization Tools...")
    
    # Create a small model for testing
    config = TemporalEigenstateConfig(
        d_model=128,
        n_layers=2,
        num_eigenstates=32,
        vocab_size=1000,
        max_seq_len=64,
    )
    
    model = TemporalEigenstateNetwork(config)
    
    # Create dummy input
    tokens = torch.randint(0, 1000, (1, 32))
    
    # Test visualizations
    print("\n1. Plotting eigenvalue spectrum...")
    fig1 = plot_eigenvalue_spectrum(model, block_idx=0)
    plt.close(fig1)
    print("✓ Eigenvalue spectrum created")
    
    print("\n2. Plotting eigenstate trajectory...")
    fig2 = plot_eigenstate_trajectory(model, tokens, block_idx=0, cell_idx=0)
    if fig2:
        plt.close(fig2)
        print("✓ Eigenstate trajectory created")
    
    print("\n3. Creating eigenstate heatmap...")
    fig3 = plot_eigenstate_heatmap(model, tokens, block_idx=0, cell_idx=0)
    if fig3:
        plt.close(fig3)
        print("✓ Eigenstate heatmap created")
    
    print("\n✓ All visualization tests passed!")
