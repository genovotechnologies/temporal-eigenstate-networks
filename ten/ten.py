"""
TEN — Temporal Eigenstate Networks
===================================

Single-file implementation. Import and use:

    from ten import TEN
    model = TEN(vocab_size=50257, d_model=512, n_layers=6)
    logits = model(input_ids)

Auto-selects the best backend:
  T ≤ 1024  →  FFT mode (fastest for short/medium context)
  T > 1024  →  Pro mode (cross-layer memory for long context)

Architecture: spectral decomposition of hidden states into K complex-valued
eigenstates that evolve via diagonal recurrence, computed in O(T log T) via
FFT convolution. See: "Temporal Eigenstate Networks: Linear-Complexity
Sequence Modeling via Spectral Decomposition" (AAAI-26 / NeurIPS submission).

Author: Oluwatosin Afolabi <afolabi@genovotech.com>
License: MIT
"""

__version__ = "0.2.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================================
# Core: FFT-based eigenstate evolution
# ============================================================================

def eigenstate_fft(
    log_decay: torch.Tensor,      # (K,)
    frequency: torch.Tensor,      # (K,)
    beta_r: torch.Tensor,         # (B, T, K)
    beta_i: torch.Tensor,         # (B, T, K)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigenstate evolution via FFT convolution. O(T log T), fully parallel.

    The recurrence c_k(t) = λ_k · c_k(t-1) + β_k(t) is a causal convolution
    with kernel h_k(n) = λ_k^n. We compute it via FFT: C = IFFT(FFT(β) · FFT(h)).
    """
    B, T, K = beta_r.shape
    device = beta_r.device

    mag = torch.sigmoid(log_decay)
    n = torch.arange(T, device=device, dtype=beta_r.dtype)
    mag_pow = mag.unsqueeze(0) ** n.unsqueeze(1)       # (T, K)
    phase = n.unsqueeze(1) * frequency.unsqueeze(0)     # (T, K)
    kern_r = mag_pow * torch.cos(phase)
    kern_i = mag_pow * torch.sin(phase)

    beta_c = torch.complex(F.pad(beta_r, (0,0,0,T)), F.pad(beta_i, (0,0,0,T)))
    kern_c = torch.complex(F.pad(kern_r, (0,0,0,T)), F.pad(kern_i, (0,0,0,T)))

    c = torch.fft.ifft(torch.fft.fft(beta_c, dim=1) *
                        torch.fft.fft(kern_c, dim=0).unsqueeze(0), dim=1)[:, :T, :]
    return c.real, c.imag


def ssd_eigenstate_evolution(
    log_decay: torch.Tensor,
    frequency: torch.Tensor,
    beta_r: torch.Tensor,
    beta_i: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SSD (State-Space Duality) eigenstate evolution via chunked matmul.
    Within each chunk: eigenstate evolution = lower-triangular Toeplitz matmul.
    Between chunks: propagate K-dimensional state.
    """
    B, T, K = beta_r.shape
    C = chunk_size
    device = beta_r.device
    dtype = beta_r.dtype

    T_pad = ((T + C - 1) // C) * C
    if T_pad > T:
        beta_r = F.pad(beta_r, (0, 0, 0, T_pad - T))
        beta_i = F.pad(beta_i, (0, 0, 0, T_pad - T))
    n_chunks = T_pad // C

    # Build Toeplitz decay matrix
    mag = torch.sigmoid(log_decay)
    pos = torch.arange(C, device=device, dtype=dtype)
    mag_pow = mag.unsqueeze(0) ** pos.unsqueeze(1)
    phases = pos.unsqueeze(1) * frequency.unsqueeze(0)
    kern_r = mag_pow * torch.cos(phases)
    kern_i = mag_pow * torch.sin(phases)

    diffs = pos.unsqueeze(0) - pos.unsqueeze(1)
    mask = (diffs >= 0).float()
    diffs_c = diffs.clamp(min=0).long()
    L_r = kern_r[diffs_c] * mask.unsqueeze(-1)
    L_i = kern_i[diffs_c] * mask.unsqueeze(-1)

    # Reshape to chunks
    br = beta_r.reshape(B, n_chunks, C, K).permute(0,1,3,2).reshape(B*n_chunks, K, C)
    bi = beta_i.reshape(B, n_chunks, C, K).permute(0,1,3,2).reshape(B*n_chunks, K, C)

    intra_r = (torch.einsum('kjs,bks->bkj', L_r, br) -
               torch.einsum('kjs,bks->bkj', L_i, bi)).reshape(B, n_chunks, K, C).permute(0,1,3,2)
    intra_i = (torch.einsum('kjs,bks->bkj', L_r, bi) +
               torch.einsum('kjs,bks->bkj', L_i, br)).reshape(B, n_chunks, K, C).permute(0,1,3,2)

    # Inter-chunk propagation
    prop_r = L_r[:, 0, :]
    prop_i = L_i[:, 0, :]

    state_r = torch.zeros(B, K, device=device, dtype=dtype)
    state_i = torch.zeros(B, K, device=device, dtype=dtype)
    out_r = torch.empty_like(intra_r)
    out_i = torch.empty_like(intra_i)

    for ci in range(n_chunks):
        corr_r = prop_r.unsqueeze(0) * state_r.unsqueeze(1) - prop_i.unsqueeze(0) * state_i.unsqueeze(1)
        corr_i = prop_r.unsqueeze(0) * state_i.unsqueeze(1) + prop_i.unsqueeze(0) * state_r.unsqueeze(1)
        out_r[:, ci] = intra_r[:, ci] + corr_r
        out_i[:, ci] = intra_i[:, ci] + corr_i
        state_r = out_r[:, ci, -1, :]
        state_i = out_i[:, ci, -1, :]

    return out_r.reshape(B, T_pad, K)[:,:T], out_i.reshape(B, T_pad, K)[:,:T]


# ============================================================================
# Spectral Gate
# ============================================================================

class SpectralGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * h


# ============================================================================
# TEN Layer (FFT mode — for T ≤ 1024)
# ============================================================================

class TENFFTLayer(nn.Module):
    def __init__(self, d_model, K=64, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.K, self.n_heads = K, n_heads
        self.Kh = K // n_heads
        self.in_proj = nn.Linear(d_model, K * 2, bias=False)
        self.log_decay = nn.Parameter(torch.empty(K).uniform_(-3.0, -0.1))
        self.frequency = nn.Parameter(torch.linspace(0, 2*math.pi*(1-1/K), K))
        self.coupling = nn.ParameterList([
            nn.Parameter(torch.eye(self.Kh) + 0.01*torch.randn(self.Kh, self.Kh)/math.sqrt(self.Kh))
            for _ in range(n_heads)])
        self.out_proj = nn.Linear(K * 2, d_model, bias=False)
        self.gate = SpectralGate(d_model)
        m = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, m), nn.SiLU(), nn.Linear(m, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, prev_eig=None):
        B, T, D = x.shape
        xn = self.n1(x)
        beta = self.in_proj(xn)
        br, bi = beta.chunk(2, dim=-1)
        cr, ci = eigenstate_fft(self.log_decay, self.frequency, br, bi)
        # Per-head coupling
        cr_h, ci_h = cr.reshape(B,T,self.n_heads,self.Kh), ci.reshape(B,T,self.n_heads,self.Kh)
        or_, oi_ = [], []
        for h in range(self.n_heads):
            R = self.coupling[h]
            or_.append(torch.einsum('jk,btk->btj', R, cr_h[:,:,h]))
            oi_.append(torch.einsum('jk,btk->btj', R, ci_h[:,:,h]))
        cr, ci = torch.cat(or_, -1), torch.cat(oi_, -1)
        eig = torch.cat([cr, ci], -1)
        h = self.gate(xn, self.out_proj(eig))
        x = x + self.drop(h)
        x = x + self.drop(self.mlp(self.n2(x)))
        return x, eig


# ============================================================================
# TEN Pro Layer (for T > 1024 — adds cross-layer memory + depth-adaptive init)
# ============================================================================

class TENProLayer(nn.Module):
    def __init__(self, d_model, K=64, n_heads=4, mlp_ratio=4.0, dropout=0.1,
                 layer_idx=0, n_layers=6, use_short_conv=True, conv_kernel=4):
        super().__init__()
        self.K, self.n_heads = K, n_heads
        self.Kh = K // n_heads
        self.layer_idx = layer_idx
        self.use_short_conv = use_short_conv

        self.in_proj = nn.Linear(d_model, K * 2, bias=False)
        if use_short_conv:
            self.conv = nn.Conv1d(K*2, K*2, conv_kernel, padding=conv_kernel-1, groups=K*2)

        depth = layer_idx / max(1, n_layers - 1)
        center = -2.0 + 1.5 * depth
        self.log_decay = nn.Parameter(torch.empty(K).uniform_(center-1.0, center+0.5))
        self.frequency = nn.Parameter(torch.linspace(0, 2*math.pi*(1-1/K), K))

        # Cross-layer memory (from layer 1 onward)
        self.has_memory = layer_idx > 0
        if self.has_memory:
            self.mem_gate = nn.Parameter(torch.tensor(-3.0))
            self.mem_proj = nn.Linear(K*2, K*2, bias=False)
            nn.init.eye_(self.mem_proj.weight)

        self.coupling = nn.ParameterList([
            nn.Parameter(torch.eye(self.Kh) + 0.01*torch.randn(self.Kh, self.Kh)/math.sqrt(self.Kh))
            for _ in range(n_heads)])
        self.out_proj = nn.Linear(K*2, d_model, bias=False)
        self.gate = SpectralGate(d_model)
        m = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, m), nn.SiLU(), nn.Linear(m, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, prev_eig=None):
        B, T, D = x.shape
        xn = self.n1(x)
        beta = self.in_proj(xn)
        if self.use_short_conv:
            beta = F.silu(self.conv(beta.transpose(1,2))[:,:,:T].transpose(1,2))
        br, bi = beta.chunk(2, dim=-1)

        if self.has_memory and prev_eig is not None:
            mg = torch.sigmoid(self.mem_gate)
            pt = self.mem_proj(prev_eig)
            pr, pi = pt.chunk(2, dim=-1)
            br, bi = br + mg*pr, bi + mg*pi

        cr, ci = eigenstate_fft(self.log_decay, self.frequency, br, bi)

        cr_h, ci_h = cr.reshape(B,T,self.n_heads,self.Kh), ci.reshape(B,T,self.n_heads,self.Kh)
        or_, oi_ = [], []
        for h in range(self.n_heads):
            R = self.coupling[h]
            or_.append(torch.einsum('jk,btk->btj', R, cr_h[:,:,h]))
            oi_.append(torch.einsum('jk,btk->btj', R, ci_h[:,:,h]))
        cr, ci = torch.cat(or_, -1), torch.cat(oi_, -1)

        eig = torch.cat([cr, ci], -1)
        h = self.gate(xn, self.out_proj(eig))
        x = x + self.drop(h)
        x = x + self.drop(self.mlp(self.n2(x)))
        return x, eig


# ============================================================================
# TEN: The unified model
# ============================================================================

class TEN(nn.Module):
    """
    Temporal Eigenstate Network.

    Auto-selects backend based on sequence length:
      T ≤ context_threshold  →  FFT layers
      T > context_threshold  →  Pro layers (+ cross-layer memory, short conv, depth-adaptive init)

    Args:
        vocab_size: vocabulary size
        d_model: hidden dimension (default 512)
        n_layers: number of layers (default 6)
        K: number of eigenstates per layer (default 64)
        n_heads: eigenstate heads (default 4)
        context_threshold: T above which Pro mode activates (default 1024)
        force_mode: 'auto', 'fft', or 'pro' to override auto-selection
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        K: int = 64,
        n_heads: int = 4,
        context_threshold: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 32768,
        force_mode: str = 'auto',
        use_short_conv: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.threshold = context_threshold
        self.force_mode = force_mode

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.fft_layers = nn.ModuleList([
            TENFFTLayer(d_model, K, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.pro_layers = nn.ModuleList([
            TENProLayer(d_model, K, n_heads, mlp_ratio, dropout, i, n_layers, use_short_conv)
            for i in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.emb_drop(self.tok_emb(input_ids) +
                          self.pos_emb(torch.arange(T, device=input_ids.device)))

        mode = self.force_mode if self.force_mode != 'auto' else ('pro' if T > self.threshold else 'fft')
        layers = self.pro_layers if mode == 'pro' else self.fft_layers

        prev = None
        for layer in layers:
            x, prev = layer(x, prev)

        return self.lm_head(self.norm_f(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def active_parameters(self, T=512):
        """Parameters used at sequence length T (FFT vs Pro layers)."""
        mode = 'pro' if T > self.threshold else 'fft'
        layers = self.pro_layers if mode == 'pro' else self.fft_layers
        shared = sum(p.numel() for n, p in self.named_parameters()
                     if 'fft_layers' not in n and 'pro_layers' not in n and p.requires_grad)
        layer_params = sum(p.numel() for p in layers.parameters() if p.requires_grad)
        return shared + layer_params
