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


class PreNormDecoderBlock(torch.nn.Module):
    """Compact pre-norm decoder block built only from PyTorch operators."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2, intermediate_size: int = 16) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = torch.nn.RMSNorm(hidden_size)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.norm2 = torch.nn.RMSNorm(hidden_size)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size)

    def forward(self, x, attention_mask):
        batch, sequence, _hidden = x.shape
        normalized = self.norm1(x)
        q = self.q_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        k = self.k_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        v = self.v_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        context = torch.matmul(probabilities, v)
        merged = context.permute(0, 2, 1, 3).reshape(batch, sequence, self.hidden_size)
        residual = x + self.out_proj(merged)

        mlp_input = self.norm2(residual)
        gated = torch.nn.functional.silu(self.gate_proj(mlp_input)) * self.up_proj(mlp_input)
        return residual + self.down_proj(gated)
