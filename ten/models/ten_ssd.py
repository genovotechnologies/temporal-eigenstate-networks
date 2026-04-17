"""
TEN-SSD: State-Space Duality implementation for Temporal Eigenstate Networks.

Converts the eigenstate recurrence into hardware-friendly matmuls using the
Mamba-2 SSD technique. Within each chunk of C timesteps:

    Y = (L ⊙ C·B^T) · X

where L is a lower-triangular matrix of cumulative eigenvalue products.
This is a structured matmul that saturates tensor cores.

Expected 10-20x speedup over FFT, 30-50x over naive sequential.

Key insight from Mamba-2: the linear recurrence c(t) = a(t)*c(t-1) + b(t)
is mathematically equivalent to a masked matrix multiplication when computed
in chunks. The mask L encodes the decay structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def build_decay_mask(log_decay: torch.Tensor, frequency: torch.Tensor,
                     chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the lower-triangular decay mask L for SSD computation.

    L[i,j,k] = λ_k^(i-j) for i >= j, 0 otherwise

    where λ_k = sigmoid(log_decay_k) * exp(i * frequency_k)

    Returns (L_real, L_imag): (C, C, K)
    """
    C = chunk_size
    K = log_decay.shape[0]
    device = log_decay.device
    dtype = log_decay.dtype

    mag = torch.sigmoid(log_decay)  # (K,)
    positions = torch.arange(C, device=device, dtype=dtype)  # (C,)

    # Compute λ^n for n = 0..C-1
    # mag_powers[n, k] = mag_k^n
    mag_powers = mag.unsqueeze(0) ** positions.unsqueeze(1)  # (C, K)
    phases = positions.unsqueeze(1) * frequency.unsqueeze(0)  # (C, K)

    kernel_r = mag_powers * torch.cos(phases)  # (C, K)
    kernel_i = mag_powers * torch.sin(phases)  # (C, K)

    # Build lower-triangular Toeplitz: L[i,j] = kernel[i-j] for i >= j
    diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # (C, C)
    mask = (diffs >= 0)  # lower triangular
    diffs_clamped = diffs.clamp(min=0).long()

    # L[i, j, k] = kernel_r[i-j, k] if i >= j, else 0
    L_r = kernel_r[diffs_clamped] * mask.unsqueeze(-1).float()  # (C, C, K)
    L_i = kernel_i[diffs_clamped] * mask.unsqueeze(-1).float()

    return L_r, L_i


def ssd_eigenstate_evolution(
    log_decay: torch.Tensor,     # (K,)
    frequency: torch.Tensor,     # (K,)
    B_proj: torch.Tensor,        # (B, T, K) input projection (real part)
    B_proj_i: torch.Tensor,      # (B, T, K) input projection (imag part)
    C_proj: torch.Tensor,        # (B, T, K) output projection (real part)
    C_proj_i: torch.Tensor,      # (B, T, K) output projection (imag part)
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    SSD-based eigenstate evolution.

    Instead of computing eigenstate amplitudes then projecting back,
    SSD computes the full input→output map as a structured matmul:

        Y[t] = Σ_k C_k[t] · (Σ_{s≤t} λ_k^(t-s) · B_k[s] · X[s])

    Within a chunk of C timesteps, this becomes:
        Y_chunk = C_chunk · (L ⊙ (B_chunk^T · X_chunk))

    where L is the decay mask and ⊙ is element-wise multiply.

    This formulation uses matmuls everywhere — perfect for tensor cores.
    """
    B_batch, T, K = B_proj.shape
    C = chunk_size
    device = B_proj.device
    dtype = B_proj.dtype

    # Pad T to multiple of C
    T_pad = ((T + C - 1) // C) * C
    if T_pad > T:
        pad = T_pad - T
        B_proj = F.pad(B_proj, (0, 0, 0, pad))
        B_proj_i = F.pad(B_proj_i, (0, 0, 0, pad))
        C_proj = F.pad(C_proj, (0, 0, 0, pad))
        C_proj_i = F.pad(C_proj_i, (0, 0, 0, pad))

    n_chunks = T_pad // C

    # Build decay mask: (C, C, K)
    L_r, L_i = build_decay_mask(log_decay, frequency, C)

    # Reshape into chunks: (B, n_chunks, C, K)
    B_r = B_proj.reshape(B_batch, n_chunks, C, K)
    B_i = B_proj_i.reshape(B_batch, n_chunks, C, K)
    C_r = C_proj.reshape(B_batch, n_chunks, C, K)
    C_i = C_proj_i.reshape(B_batch, n_chunks, C, K)

    # Inter-chunk decay: λ^C
    mag = torch.sigmoid(log_decay)
    mag_C = mag ** C
    phase_C = frequency * C
    decay_C_r = mag_C * torch.cos(phase_C)  # (K,)
    decay_C_i = mag_C * torch.sin(phase_C)

    # =========================================================================
    # Intra-chunk computation (the SSD matmul)
    # =========================================================================
    # For each chunk, compute: output[i] = Σ_j L[i,j,k] * B[j,k] * C[i,k]
    # This is: (C_chunk) · (L ⊙ B_chunk) summed over K
    #
    # Step 1: Apply decay mask to input: M[i,j,k] = L[i,j,k] * B[j,k]
    #         Shape: (B, n_chunks, C_i, C_j, K) — but this materializes C²K
    #         Instead, compute per-eigenstate and accumulate
    #
    # For efficiency, compute the C×C attention-like matrix per eigenstate:
    #   A_k[i,j] = C_k[i] * L_k[i,j] * B_k[j]
    # Then: output[i] = Σ_k Σ_j A_k[i,j] * input[j]
    #
    # Since we're doing eigenstate evolution (not full SSM), the output IS
    # the eigenstate amplitudes. Let's compute them directly.

    # Intra-chunk eigenstate evolution via batched matmul with decay mask
    # For each eigenstate k: c_k = L_k @ beta_k  (C×C matmul with C-dim vectors)
    # This is: einsum('ijk,bnjk->bnik', L, B)  — but L is (C,C,K) and B is (B,n,C,K)

    # Complex multiply: (L_r + i*L_i) * (B_r + i*B_i)
    # Real part: L_r*B_r - L_i*B_i
    # Imag part: L_r*B_i + L_i*B_r

    # Vectorized over K using einsum
    # intra_r[b,n,i,k] = Σ_j L_r[i,j,k] * B_r[b,n,j,k] - L_i[i,j,k] * B_i[b,n,j,k]
    intra_r = torch.einsum('ijk,bnjk->bnik', L_r, B_r) - \
              torch.einsum('ijk,bnjk->bnik', L_i, B_i)  # (B, n, C, K)
    intra_i = torch.einsum('ijk,bnjk->bnik', L_r, B_i) + \
              torch.einsum('ijk,bnjk->bnik', L_i, B_r)

    # =========================================================================
    # Inter-chunk state propagation (sequential over n_chunks, but n_chunks is small)
    # =========================================================================
    # Typically T/C = 32-128 chunks, so this loop is fast

    # Decay propagation column (how initial state decays within a chunk)
    # First column of L: L[:, 0, :] = [1, λ, λ², ..., λ^(C-1)]
    prop_r = L_r[:, 0, :]  # (C, K)
    prop_i = L_i[:, 0, :]

    state_r = torch.zeros(B_batch, K, device=device, dtype=dtype)
    state_i = torch.zeros(B_batch, K, device=device, dtype=dtype)

    output_r = torch.empty_like(intra_r)
    output_i = torch.empty_like(intra_i)

    for chunk_idx in range(n_chunks):
        # Add state contribution: state decayed through the chunk
        # correction[i, k] = prop[i, k] * state[k]  (complex multiply)
        corr_r = prop_r.unsqueeze(0) * state_r.unsqueeze(1) - \
                 prop_i.unsqueeze(0) * state_i.unsqueeze(1)  # (B, C, K)
        corr_i = prop_r.unsqueeze(0) * state_i.unsqueeze(1) + \
                 prop_i.unsqueeze(0) * state_r.unsqueeze(1)

        output_r[:, chunk_idx] = intra_r[:, chunk_idx] + corr_r
        output_i[:, chunk_idx] = intra_i[:, chunk_idx] + corr_i

        # Update state: last position of this chunk
        state_r = output_r[:, chunk_idx, -1, :]
        state_i = output_i[:, chunk_idx, -1, :]

    # Reshape back: (B, T_pad, K) -> trim to (B, T, K)
    c_r = output_r.reshape(B_batch, T_pad, K)[:, :T]
    c_i = output_i.reshape(B_batch, T_pad, K)[:, :T]

    return c_r, c_i


class TENSSDLayer(nn.Module):
    """TEN layer using SSD matmul-based evolution."""

    def __init__(self, d_model, k_eigenstates=64, n_heads=4, chunk_size=64,
                 mlp_ratio=4.0, dropout=0.1, layer_idx=0, n_layers=6,
                 use_memory=True, use_short_conv=True, conv_kernel=4):
        super().__init__()
        self.K = k_eigenstates
        self.n_heads = n_heads
        self.K_per_head = k_eigenstates // n_heads
        self.chunk_size = chunk_size
        self.use_memory = use_memory and (layer_idx > 0)
        self.use_short_conv = use_short_conv

        # Input projection
        self.in_proj = nn.Linear(d_model, k_eigenstates * 2, bias=False)

        # Short convolution before recurrence (Hyena/Mamba pattern)
        if use_short_conv:
            self.short_conv = nn.Conv1d(
                k_eigenstates * 2, k_eigenstates * 2,
                kernel_size=conv_kernel, padding=conv_kernel - 1,
                groups=k_eigenstates * 2,  # depthwise
            )

        # Adaptive eigenvalues by depth
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

        # Per-head coupling
        self.coupling_blocks = nn.ParameterList([
            nn.Parameter(
                torch.eye(self.K_per_head) +
                0.01 * torch.randn(self.K_per_head, self.K_per_head) / math.sqrt(self.K_per_head)
            ) for _ in range(n_heads)
        ])

        # Output
        self.out_proj = nn.Linear(k_eigenstates * 2, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.gate_proj.bias, -2.0)

        # MLP
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

        # Project to eigenstate space
        beta = self.in_proj(x_norm)  # (B, T, 2K)

        # Short convolution (captures local patterns, smooths input for recurrence)
        if self.use_short_conv:
            beta = self.short_conv(beta.transpose(1, 2))[:, :, :T].transpose(1, 2)
            beta = F.silu(beta)  # nonlinearity after conv

        beta_r, beta_i = beta.chunk(2, dim=-1)

        # Cross-layer memory
        if self.use_memory and prev_eigenstates is not None:
            mg = torch.sigmoid(self.memory_gate)
            prev_t = self.memory_proj(prev_eigenstates)
            pr, pi = prev_t.chunk(2, dim=-1)
            beta_r = beta_r + mg * pr
            beta_i = beta_i + mg * pi

        # SSD eigenstate evolution
        c_r, c_i = ssd_eigenstate_evolution(
            self.log_decay, self.frequency,
            beta_r, beta_i, beta_r, beta_i,  # B=C for now (symmetric projection)
            chunk_size=self.chunk_size,
        )

        # Per-head coupling
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


class TENSSDModel(nn.Module):
    """TEN with SSD backend + short conv + cross-layer memory."""

    def __init__(self, vocab_size, d_model=512, n_layers=6, k_eigenstates=64,
                 n_heads=4, chunk_size=64, mlp_ratio=4.0, dropout=0.1,
                 max_seq_len=8192, use_short_conv=True, conv_kernel=4):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TENSSDLayer(d_model, k_eigenstates, n_heads, chunk_size,
                       mlp_ratio, dropout, i, n_layers,
                       use_memory=True, use_short_conv=use_short_conv,
                       conv_kernel=conv_kernel)
            for i in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.emb_drop(self.token_emb(input_ids) +
                          self.pos_emb(torch.arange(T, device=input_ids.device)))
        prev = None
        for layer in self.layers:
            x, prev = layer(x, prev)
        return self.lm_head(self.norm_f(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
