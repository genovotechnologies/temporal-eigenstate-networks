"""
TEN-Chunked: Matmul-based eigenstate evolution via chunked computation.

Key idea: The eigenstate recurrence c(t) = λ·c(t-1) + β(t) can be computed
as a matrix multiplication within chunks of size C, with only a K-dimensional
state propagated between chunks.

Within chunk: c = L·β  (matmul, GPU-friendly)
Between chunks: state *= λ^C  (tiny multiply)

Total: O(T·C·K + T·K·d) using matmuls that saturate tensor cores.

This is the same mathematical trick behind Mamba-2's SSD (State Space Duality),
applied to TEN's eigenstate decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def build_decay_matrix(log_decay: torch.Tensor, frequency: torch.Tensor, chunk_size: int):
    """
    Build the lower-triangular Toeplitz matrix L for within-chunk computation.

    L[j, s] = λ^(j-s) for s ≤ j, 0 otherwise
    where λ = sigmoid(log_decay) * exp(i * frequency)

    Returns real and imaginary parts separately for matmul compatibility.
    Shape: (K, C, C) — one C×C matrix per eigenstate
    """
    K = log_decay.shape[0]
    C = chunk_size

    magnitude = torch.sigmoid(log_decay)  # (K,)
    device = log_decay.device

    # Powers: [0, 1, 2, ..., C-1]
    powers = torch.arange(C, device=device, dtype=log_decay.dtype)  # (C,)

    # Magnitude decay: |λ|^n for each eigenstate
    mag_powers = magnitude.unsqueeze(1) ** powers.unsqueeze(0)  # (K, C)

    # Phase: n * ω for each eigenstate
    phases = frequency.unsqueeze(1) * powers.unsqueeze(0)  # (K, C)

    # Kernel values: λ^n = |λ|^n * (cos(nω) + i·sin(nω))
    kernel_r = mag_powers * torch.cos(phases)  # (K, C)
    kernel_i = mag_powers * torch.sin(phases)  # (K, C)

    # Build Toeplitz lower-triangular matrix from kernel
    # L[j, s] = kernel[j - s] for j >= s, 0 otherwise
    # This is equivalent to: for each row j, L[j, :] = [kernel[j], kernel[j-1], ..., kernel[0], 0, ..., 0]

    # Efficient construction using indexing
    indices = powers.unsqueeze(0) - powers.unsqueeze(1)  # (C, C): indices[j,s] = j - s
    mask = indices >= 0  # lower triangular

    # Gather kernel values
    indices_clamped = indices.clamp(min=0).long()  # clamp negatives to 0 (will be masked)

    L_r = kernel_r[:, indices_clamped] * mask.unsqueeze(0).float()  # (K, C, C)
    L_i = kernel_i[:, indices_clamped] * mask.unsqueeze(0).float()  # (K, C, C)

    return L_r, L_i


def chunked_eigenstate_evolution(
    log_decay: torch.Tensor,     # (K,)
    frequency: torch.Tensor,     # (K,)
    beta_r: torch.Tensor,        # (B, T, K)
    beta_i: torch.Tensor,        # (B, T, K)
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigenstate evolution using chunked matmul.

    1. Split sequence into T/C chunks of size C
    2. Within each chunk: c = L·β (matmul with lower-triangular Toeplitz)
    3. Between chunks: propagate K-dim state via multiply

    Total cost: O(T/C · C² · K + T · K · d) ≈ O(T · C · K + T · K · d)
    All operations are matmuls that tensor cores can accelerate.
    """
    B, T, K = beta_r.shape
    C = chunk_size
    device = beta_r.device
    dtype = beta_r.dtype

    # Pad T to multiple of C
    T_pad = ((T + C - 1) // C) * C
    if T_pad > T:
        beta_r = F.pad(beta_r, (0, 0, 0, T_pad - T))
        beta_i = F.pad(beta_i, (0, 0, 0, T_pad - T))

    n_chunks = T_pad // C

    # Build the C×C decay matrix (shared across all chunks for fixed eigenvalues)
    L_r, L_i = build_decay_matrix(log_decay, frequency, C)  # (K, C, C)

    # Reshape input into chunks: (B, n_chunks, C, K)
    beta_r_chunks = beta_r.reshape(B, n_chunks, C, K)
    beta_i_chunks = beta_i.reshape(B, n_chunks, C, K)

    # Compute inter-chunk decay factor: λ^C for propagating state between chunks
    magnitude = torch.sigmoid(log_decay)
    mag_C = magnitude ** C  # (K,)
    phase_C = frequency * C  # (K,)
    decay_r = mag_C * torch.cos(phase_C)  # (K,)
    decay_i = mag_C * torch.sin(phase_C)  # (K,)

    # Within-chunk computation via matmul
    # For each chunk, c_chunk = L · β_chunk (complex matmul)
    # c_r = L_r · β_r - L_i · β_i
    # c_i = L_r · β_i + L_i · β_r
    # Shape: (K, C, C) × (B, n_chunks, C, K) -> need to rearrange

    # Rearrange for batched matmul: (B*n_chunks, K, C) for beta
    beta_r_flat = beta_r_chunks.permute(0, 1, 3, 2).reshape(B * n_chunks, K, C)  # (B*n, K, C)
    beta_i_flat = beta_i_chunks.permute(0, 1, 3, 2).reshape(B * n_chunks, K, C)

    # L is (K, C, C), beta is (B*n, K, C)
    # We want: for each k, result[k] = L[k] @ beta[k]
    # This is a batched matmul over K

    # Expand L for batch: (1, K, C, C) -> broadcast with (B*n, K, C, 1)
    # Use einsum for clarity: result[b,k,j] = sum_s L[k,j,s] * beta[b,k,s]
    intra_r = torch.einsum('kjs,bks->bkj', L_r, beta_r_flat) - \
              torch.einsum('kjs,bks->bkj', L_i, beta_i_flat)  # (B*n, K, C)
    intra_i = torch.einsum('kjs,bks->bkj', L_r, beta_i_flat) + \
              torch.einsum('kjs,bks->bkj', L_i, beta_r_flat)  # (B*n, K, C)

    # Reshape back: (B, n_chunks, K, C) -> (B, n_chunks, C, K)
    intra_r = intra_r.reshape(B, n_chunks, K, C).permute(0, 1, 3, 2)  # (B, n, C, K)
    intra_i = intra_i.reshape(B, n_chunks, K, C).permute(0, 1, 3, 2)

    # Inter-chunk state propagation
    # Each chunk's output needs to include the contribution from previous chunks' final state
    # final_state of chunk i is intra[:, i, C-1, :] (the last timestep in the chunk)
    # This state gets multiplied by λ^C and added to the next chunk

    # Build correction: for each chunk after the first, add λ^C * prev_state propagated
    # through the within-chunk decay matrix's first column

    # The correction for chunk n at position j is: λ^(j) * accumulated_state
    # where accumulated_state = λ^C * prev_chunk_final_state

    # First column of L gives the decay from chunk start: L[:, :, 0] = [1, λ, λ², ..., λ^(C-1)]
    first_col_r = L_r[:, :, 0]  # (K, C)
    first_col_i = L_i[:, :, 0]  # (K, C)

    # Process chunks sequentially (only n_chunks iterations, typically T/64 ≈ 32)
    state_r = torch.zeros(B, K, device=device, dtype=dtype)
    state_i = torch.zeros(B, K, device=device, dtype=dtype)

    output_r = torch.zeros_like(intra_r)
    output_i = torch.zeros_like(intra_i)

    for chunk_idx in range(n_chunks):
        # Add state contribution: state propagated through within-chunk decay
        # correction[j, k] = first_col[k, j] * state[k] (complex multiply)
        corr_r = first_col_r.unsqueeze(0) * state_r.unsqueeze(2) - \
                 first_col_i.unsqueeze(0) * state_i.unsqueeze(2)  # (B, K, C)
        corr_i = first_col_r.unsqueeze(0) * state_i.unsqueeze(2) + \
                 first_col_i.unsqueeze(0) * state_r.unsqueeze(2)

        # Add correction to intra-chunk result
        output_r[:, chunk_idx] = intra_r[:, chunk_idx] + corr_r.permute(0, 2, 1)  # (B, C, K)
        output_i[:, chunk_idx] = intra_i[:, chunk_idx] + corr_i.permute(0, 2, 1)

        # Update state: state = output at last position of this chunk
        state_r = output_r[:, chunk_idx, -1, :]  # (B, K)
        state_i = output_i[:, chunk_idx, -1, :]

    # Reshape to (B, T_pad, K) and trim
    c_r = output_r.reshape(B, T_pad, K)[:, :T]
    c_i = output_i.reshape(B, T_pad, K)[:, :T]

    return c_r, c_i


class TENChunkedLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        k_eigenstates: int = 64,
        n_heads: int = 4,
        chunk_size: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = k_eigenstates
        self.n_heads = n_heads
        self.K_per_head = k_eigenstates // n_heads
        self.chunk_size = chunk_size

        assert k_eigenstates % n_heads == 0

        # Input/output projections
        self.in_proj = nn.Linear(d_model, k_eigenstates * 2, bias=False)
        self.out_proj = nn.Linear(k_eigenstates * 2, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # Eigenvalues
        self.log_decay = nn.Parameter(torch.empty(k_eigenstates).uniform_(-3.0, -0.1))
        self.frequency = nn.Parameter(
            torch.linspace(0, 2 * math.pi * (1 - 1/k_eigenstates), k_eigenstates)
        )

        # Per-head coupling
        self.coupling_blocks = nn.ParameterList([
            nn.Parameter(
                torch.eye(self.K_per_head) + 0.01 * torch.randn(self.K_per_head, self.K_per_head) / math.sqrt(self.K_per_head)
            )
            for _ in range(n_heads)
        ])

        # MLP
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.SiLU(), nn.Linear(mlp_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        # Project to eigenstate space
        beta = self.in_proj(x_norm)
        beta_r, beta_i = beta.chunk(2, dim=-1)

        # Chunked matmul evolution
        c_r, c_i = chunked_eigenstate_evolution(
            self.log_decay, self.frequency, beta_r, beta_i,
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
        c_r = torch.cat(out_r, dim=-1)
        c_i = torch.cat(out_i, dim=-1)

        # Gated output
        h = self.out_proj(torch.cat([c_r, c_i], dim=-1))
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x = residual + self.drop(gate * h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TENChunkedModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        k_eigenstates: int = 64,
        n_heads: int = 4,
        chunk_size: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TENChunkedLayer(d_model, k_eigenstates, n_heads, chunk_size, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for layer in self.layers:
            nn.init.constant_(layer.gate_proj.bias, -2.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm_f(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
