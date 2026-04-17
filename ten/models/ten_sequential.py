"""
Temporal Eigenstate Networks (TEN) — clean reference implementation.
Matches paper Section 3 exactly. No overclaiming in comments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TENLayer(nn.Module):
    """Single TEN layer: eigenstate evolution + resonance coupling + MLP."""

    def __init__(self, d_model: int, k_eigenstates: int = 64, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.K = k_eigenstates

        # Eigenvectors: learned basis (complex-valued, stored as real pairs)
        self.eigvec_real = nn.Parameter(torch.randn(k_eigenstates, d_model) * 0.02 / math.sqrt(d_model))
        self.eigvec_imag = nn.Parameter(torch.randn(k_eigenstates, d_model) * 0.02 / math.sqrt(d_model))

        # Eigenvalues: parameterized to ensure |λ_k| ≤ 1
        # λ_k = sigmoid(alpha_k) * exp(i * omega_k)
        self.alpha = nn.Parameter(torch.empty(k_eigenstates).uniform_(-3.0, 0.0))
        self.omega = nn.Parameter(torch.linspace(0, 2 * math.pi * (1 - 1/k_eigenstates), k_eigenstates))

        # Resonance coupling matrix: R = I + eps*M (linear mixing, NOT nonlinear)
        self.resonance_eps = nn.Parameter(torch.tensor(0.01))
        self.resonance_M = nn.Parameter(torch.randn(k_eigenstates, k_eigenstates) / math.sqrt(k_eigenstates))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # MLP (this is where nonlinearity comes from, not R)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_eigenvalues(self):
        """Compute complex eigenvalues with |λ_k| ≤ 1."""
        magnitude = torch.sigmoid(self.alpha)  # (K,)
        cos_w = torch.cos(self.omega)
        sin_w = torch.sin(self.omega)
        lambda_real = magnitude * cos_w  # (K,)
        lambda_imag = magnitude * sin_w  # (K,)
        return lambda_real, lambda_imag

    def _get_resonance(self):
        """Compute resonance matrix R = I + eps*M."""
        R = torch.eye(self.K, device=self.resonance_M.device) + self.resonance_eps * self.resonance_M
        return R

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            state: optional (batch, K, 2) initial eigenstate amplitudes
        Returns:
            output: (batch, seq_len, d_model)
            final_state: (batch, K, 2)
        """
        B, T, D = x.shape
        residual = x
        x = self.norm1(x)

        # Get eigenvalues and resonance matrix
        lam_r, lam_i = self._get_eigenvalues()  # (K,)
        R = self._get_resonance()  # (K, K)

        # Project input onto eigenvector conjugates: beta_k(t) = <x_t, v_k*>
        # beta_real = x @ eigvec_real.T + 0 (real part of inner product with conjugate)
        # beta_imag = -x @ eigvec_imag.T (conjugate flips imaginary sign)
        beta_real = torch.einsum('btd,kd->btk', x, self.eigvec_real)  # (B, T, K)
        beta_imag = torch.einsum('btd,kd->btk', x, -self.eigvec_imag)  # (B, T, K)

        # Initialize eigenstate amplitudes
        if state is None:
            c_real = torch.zeros(B, self.K, device=x.device, dtype=x.dtype)
            c_imag = torch.zeros(B, self.K, device=x.device, dtype=x.dtype)
        else:
            c_real = state[:, :, 0]
            c_imag = state[:, :, 1]

        # Eigenstate evolution (sequential over time, parallel over K)
        outputs_real = []
        outputs_imag = []

        for t in range(T):
            # c_k(t+1) = lambda_k * c_k(t) + beta_k(t)
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            new_c_real = lam_r * c_real - lam_i * c_imag + beta_real[:, t, :]
            new_c_imag = lam_r * c_imag + lam_i * c_real + beta_imag[:, t, :]
            c_real = new_c_real
            c_imag = new_c_imag

            # Resonance coupling (linear mixing across eigenstates)
            c_coupled_real = torch.einsum('kj,bj->bk', R, c_real)
            c_coupled_imag = torch.einsum('kj,bj->bk', R, c_imag)

            outputs_real.append(c_coupled_real)
            outputs_imag.append(c_coupled_imag)

        # Stack: (B, T, K)
        c_seq_real = torch.stack(outputs_real, dim=1)
        c_seq_imag = torch.stack(outputs_imag, dim=1)

        # Reconstruct: h_t = Re[sum_k c_k(t) * v_k]
        # Re[(a+bi)(c+di)] = ac - bd
        h = torch.einsum('btk,kd->btd', c_seq_real, self.eigvec_real) - \
            torch.einsum('btk,kd->btd', c_seq_imag, self.eigvec_imag)

        h = self.out_proj(h)
        h = self.dropout(h)
        x = residual + h

        # MLP block (nonlinearity is here)
        x = x + self.mlp(self.norm2(x))

        # Final state for autoregressive generation
        final_state = torch.stack([c_real, c_imag], dim=-1)  # (B, K, 2)

        return x, final_state


class TENModel(nn.Module):
    """Full TEN language model."""

    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 k_eigenstates: int = 64, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, max_seq_len: int = 8192):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TENLayer(d_model, k_eigenstates, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(pos))

        for layer in self.layers:
            x, _ = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
