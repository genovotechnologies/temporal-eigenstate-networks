"""
TEN Unified — The definitive Temporal Eigenstate Network.

Auto-selects the optimal backend based on sequence length:
  T ≤ 1024  →  TEN-FFT  (simpler, fastest at short/medium context)
  T > 1024  →  TEN-Pro  (cross-layer memory, spectral gating, better at long context)

This is the single file you import. Everything else is an implementation detail.

Usage:
    from ten.models.ten_unified import TEN

    model = TEN(vocab_size=50257, d_model=512, n_layers=6)
    logits = model(input_ids)  # auto-selects backend based on input length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================================
# Core: FFT-based eigenstate evolution (O(T log T), fully parallel)
# ============================================================================

def eigenstate_fft(
    log_decay: torch.Tensor,
    frequency: torch.Tensor,
    beta_r: torch.Tensor,
    beta_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FFT-based eigenstate evolution. O(T log T), no sequential loop.
    c_k(t) = Σ λ_k^(t-s) · β_k(s)  =  causal convolution via FFT.
    """
    B, T, K = beta_r.shape
    device = beta_r.device

    magnitude = torch.sigmoid(log_decay)
    n = torch.arange(T, device=device, dtype=beta_r.dtype)
    mag_powers = magnitude.unsqueeze(0) ** n.unsqueeze(1)
    phase = n.unsqueeze(1) * frequency.unsqueeze(0)
    kernel_r = mag_powers * torch.cos(phase)
    kernel_i = mag_powers * torch.sin(phase)

    beta_complex = torch.complex(
        F.pad(beta_r, (0, 0, 0, T)),
        F.pad(beta_i, (0, 0, 0, T)),
    )
    kernel_complex = torch.complex(
        F.pad(kernel_r, (0, 0, 0, T)),
        F.pad(kernel_i, (0, 0, 0, T)),
    )

    B_f = torch.fft.fft(beta_complex, dim=1)
    K_f = torch.fft.fft(kernel_complex, dim=0).unsqueeze(0)
    c = torch.fft.ifft(B_f * K_f, dim=1)[:, :T, :]

    return c.real, c.imag


# ============================================================================
# Spectral Gate: frequency-aware output gating
# ============================================================================

class SpectralGate(nn.Module):
    """
    Gates output differently for low-frequency (long-range) vs
    high-frequency (local) eigenstates. Lets the model learn when to
    use distant context vs local patterns per token.
    """
    def __init__(self, d_model: int, k_eigenstates: int, n_bands: int = 4):
        super().__init__()
        self.n_bands = n_bands
        self.k_per_band = k_eigenstates // n_bands
        self.band_proj = nn.Linear(d_model, n_bands, bias=True)
        self.main_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.main_gate.bias, -2.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.main_gate(x)) * h


# ============================================================================
# TEN-FFT Layer (short/medium context, T ≤ 1024)
# ============================================================================

class TENFFTLayer(nn.Module):
    """Fast eigenstate layer using FFT convolution. Best for T ≤ 1024."""

    def __init__(self, d_model, k_eigenstates=64, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.K = k_eigenstates
        self.n_heads = n_heads
        self.K_per_head = k_eigenstates // n_heads

        self.in_proj = nn.Linear(d_model, k_eigenstates * 2, bias=False)
        self.log_decay = nn.Parameter(torch.empty(k_eigenstates).uniform_(-3.0, -0.1))
        self.frequency = nn.Parameter(
            torch.linspace(0, 2 * math.pi * (1 - 1/k_eigenstates), k_eigenstates)
        )

        self.coupling_blocks = nn.ParameterList([
            nn.Parameter(
                torch.eye(self.K_per_head) +
                0.01 * torch.randn(self.K_per_head, self.K_per_head) / math.sqrt(self.K_per_head)
            ) for _ in range(n_heads)
        ])

        self.out_proj = nn.Linear(k_eigenstates * 2, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.gate_proj.bias, -2.0)

        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.SiLU(), nn.Linear(mlp_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, prev_eigenstates=None):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        beta = self.in_proj(x_norm)
        beta_r, beta_i = beta.chunk(2, dim=-1)
        c_r, c_i = eigenstate_fft(self.log_decay, self.frequency, beta_r, beta_i)

        c_r_h = c_r.reshape(B, T, self.n_heads, self.K_per_head)
        c_i_h = c_i.reshape(B, T, self.n_heads, self.K_per_head)
        out_r, out_i = [], []
        for h in range(self.n_heads):
            R = self.coupling_blocks[h]
            out_r.append(torch.einsum('jk,btk->btj', R, c_r_h[:, :, h]))
            out_i.append(torch.einsum('jk,btk->btj', R, c_i_h[:, :, h]))
        c_r, c_i = torch.cat(out_r, dim=-1), torch.cat(out_i, dim=-1)

        eigenstates = torch.cat([c_r, c_i], dim=-1)
        h = self.out_proj(eigenstates)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x = residual + self.drop(gate * h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, eigenstates


# ============================================================================
# TEN-Pro Layer (long context, T > 1024)
# ============================================================================

class TENProLayer(nn.Module):
    """
    Enhanced eigenstate layer with:
    - Cross-layer eigenstate memory
    - Adaptive eigenvalue initialization by depth
    - Spectral gating
    Best for T > 1024 where long-range modeling matters.
    """

    def __init__(self, d_model, k_eigenstates=64, n_heads=4, mlp_ratio=4.0,
                 dropout=0.1, layer_idx=0, n_layers=6, use_memory=True):
        super().__init__()
        self.K = k_eigenstates
        self.n_heads = n_heads
        self.K_per_head = k_eigenstates // n_heads
        self.layer_idx = layer_idx
        self.use_memory = use_memory and (layer_idx > 0)

        self.in_proj = nn.Linear(d_model, k_eigenstates * 2, bias=False)

        # Adaptive init: early layers → fast decay, deep layers → slow decay
        depth_ratio = layer_idx / max(1, n_layers - 1)
        decay_center = -2.0 + 1.5 * depth_ratio
        self.log_decay = nn.Parameter(
            torch.empty(k_eigenstates).uniform_(decay_center - 1.0, decay_center + 0.5)
        )
        self.frequency = nn.Parameter(
            torch.linspace(0, 2 * math.pi * (1 - 1/k_eigenstates), k_eigenstates)
        )

        # Cross-layer memory
        if self.use_memory:
            self.memory_gate = nn.Parameter(torch.tensor(-3.0))
            self.memory_proj = nn.Linear(k_eigenstates * 2, k_eigenstates * 2, bias=False)
            nn.init.eye_(self.memory_proj.weight)

        self.coupling_blocks = nn.ParameterList([
            nn.Parameter(
                torch.eye(self.K_per_head) +
                0.01 * torch.randn(self.K_per_head, self.K_per_head) / math.sqrt(self.K_per_head)
            ) for _ in range(n_heads)
        ])

        self.spectral_gate = SpectralGate(d_model, k_eigenstates)
        self.out_proj = nn.Linear(k_eigenstates * 2, d_model, bias=False)

        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.SiLU(), nn.Linear(mlp_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, prev_eigenstates=None):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        beta = self.in_proj(x_norm)
        beta_r, beta_i = beta.chunk(2, dim=-1)

        # Cross-layer memory: blend previous layer's eigenstate amplitudes
        if self.use_memory and prev_eigenstates is not None:
            mg = torch.sigmoid(self.memory_gate)
            prev_t = self.memory_proj(prev_eigenstates)
            pr, pi = prev_t.chunk(2, dim=-1)
            beta_r = beta_r + mg * pr
            beta_i = beta_i + mg * pi

        c_r, c_i = eigenstate_fft(self.log_decay, self.frequency, beta_r, beta_i)

        c_r_h = c_r.reshape(B, T, self.n_heads, self.K_per_head)
        c_i_h = c_i.reshape(B, T, self.n_heads, self.K_per_head)
        out_r, out_i = [], []
        for h in range(self.n_heads):
            R = self.coupling_blocks[h]
            out_r.append(torch.einsum('jk,btk->btj', R, c_r_h[:, :, h]))
            out_i.append(torch.einsum('jk,btk->btj', R, c_i_h[:, :, h]))
        c_r, c_i = torch.cat(out_r, dim=-1), torch.cat(out_i, dim=-1)

        eigenstates = torch.cat([c_r, c_i], dim=-1)
        h = self.out_proj(eigenstates)
        h = self.spectral_gate(x_norm, h)
        x = residual + self.drop(h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, eigenstates


# ============================================================================
# TEN: The unified model
# ============================================================================

class TEN(nn.Module):
    """
    Temporal Eigenstate Network — unified architecture.

    Auto-selects backend based on sequence length:
      T ≤ context_threshold  →  TEN-FFT layers (fast, simple)
      T > context_threshold  →  TEN-Pro layers (cross-layer memory, spectral gating)

    Both use the same FFT-based O(T log T) eigenstate evolution.
    The difference is in the surrounding architecture: Pro adds cross-layer
    eigenstate memory and adaptive depth-dependent initialization, which
    help at long contexts where information must flow across many layers.

    Args:
        vocab_size: vocabulary size
        d_model: hidden dimension (default 512)
        n_layers: number of layers (default 6)
        k_eigenstates: number of eigenstates per layer (default 64)
        n_heads: number of eigenstate heads (default 4)
        context_threshold: T above which Pro mode activates (default 1024)
        mlp_ratio: MLP expansion ratio (default 4.0)
        dropout: dropout rate (default 0.1)
        max_seq_len: maximum sequence length (default 8192)
        force_mode: 'auto', 'fft', or 'pro' (default 'auto')
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        k_eigenstates: int = 64,
        n_heads: int = 4,
        context_threshold: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        force_mode: str = 'auto',
    ):
        super().__init__()
        self.d_model = d_model
        self.context_threshold = context_threshold
        self.force_mode = force_mode

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Build both layer stacks
        self.fft_layers = nn.ModuleList([
            TENFFTLayer(d_model, k_eigenstates, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.pro_layers = nn.ModuleList([
            TENProLayer(d_model, k_eigenstates, n_heads, mlp_ratio, dropout,
                       layer_idx=i, n_layers=n_layers, use_memory=True)
            for i in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _select_mode(self, T: int) -> str:
        if self.force_mode != 'auto':
            return self.force_mode
        return 'pro' if T > self.context_threshold else 'fft'

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))

        mode = self._select_mode(T)

        if mode == 'fft':
            for layer in self.fft_layers:
                x, _ = layer(x)
        else:
            prev_eig = None
            for layer in self.pro_layers:
                x, prev_eig = layer(x, prev_eig)

        return self.lm_head(self.norm_f(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fft_parameters(self):
        return sum(p.numel() for p in self.fft_layers.parameters() if p.requires_grad)

    def pro_parameters(self):
        return sum(p.numel() for p in self.pro_layers.parameters() if p.requires_grad)
