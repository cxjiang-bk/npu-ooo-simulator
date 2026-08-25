from __future__ import annotations

import torch


class TwoMatmul(torch.nn.Module):
    """Small two-Matmul graph used by runtime and scheduler tests."""

    def forward(self, lhs, middle, rhs):
        return torch.matmul(torch.matmul(lhs, middle), rhs)


class ResidualAdd(torch.nn.Module):
    """Minimal ARU-bound pointwise module."""

    def forward(self, lhs, rhs):
        return lhs + rhs


class AttentionBlock(torch.nn.Module):
    """Small pre-norm attention block without head reshape."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)
        self.out_proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        probabilities = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)),
            dim=-1,
        )
        return x + self.out_proj(torch.matmul(probabilities, v))


class AttentionMicrograph(torch.nn.Module):
    """QK-softmax-PV subset currently verified with Torch-XLA."""

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1))
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v)


class MultiHeadAttentionBlock(torch.nn.Module):
    """Two-head attention with additive mask, output projection and residual."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)
        self.out_proj = torch.nn.Linear(8, 8)

    def forward(self, x, attention_mask):
        batch, sequence, _hidden = x.shape
        q = self.q_proj(x).reshape(batch, sequence, 2, 4).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, sequence, 2, 4).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, sequence, 2, 4).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * 0.5
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        context = torch.matmul(probabilities, v)
        merged = context.permute(0, 2, 1, 3).reshape(batch, sequence, 8)
        return x + self.out_proj(merged)
