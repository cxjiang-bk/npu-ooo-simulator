"""Shared PyTorch building blocks used by individual paper model files."""

from __future__ import annotations

import math
from typing import Type

import torch

from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class PaperTransformerBlock(torch.nn.Module):
    """One transformer block expressed with ordinary PyTorch operators.

    ``activation='gelu_tanh'`` is the standard tanh GELU approximation. It
    avoids the ``aten.gelu`` custom call emitted by some Torch-XLA versions.
    ``gated=True`` models the SiLU-gated MLP used by LLaMA and DeepSeek.
    """

    def __init__(
        self,
        hidden_size: int = 8,
        num_heads: int = 2,
        intermediate_size: int = 16,
        *,
        norm: str = "rmsnorm",
        activation: str = "silu",
        gated: bool = True,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if norm not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm must be 'rmsnorm' or 'layernorm'")
        if activation not in {"silu", "gelu_tanh"}:
            raise ValueError("activation must be 'silu' or 'gelu_tanh'")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.activation_name = activation
        self.gated = gated
        self.rotary = rotary
        if rotary:
            if self.head_dim % 2:
                raise ValueError("rotary embedding requires an even head dimension")
            rotation = torch.zeros(self.head_dim, self.head_dim)
            for index in range(0, self.head_dim, 2):
                rotation[index, index + 1] = 1.0
                rotation[index + 1, index] = -1.0
            self.register_buffer("rope_permutation", rotation)
        self.norm1 = torch.nn.RMSNorm(hidden_size) if norm == "rmsnorm" else torch.nn.LayerNorm(hidden_size)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.norm2 = torch.nn.RMSNorm(hidden_size) if norm == "rmsnorm" else torch.nn.LayerNorm(hidden_size)
        if gated:
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size)
            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        else:
            self.ff_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size)

    def _activation(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "silu":
            return torch.nn.functional.silu(value)
        coefficient = 0.7978845608028654
        return 0.5 * value * (1.0 + torch.tanh(coefficient * (value + 0.044715 * value**3)))

    def _apply_rotary(
        self,
        value: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        rotated = torch.matmul(value, self.rope_permutation)
        return value * rope_cos + rotated * rope_sin

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        normalized = self.norm1(x)
        q = self.q_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        k = self.k_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        v = self.v_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.rotary:
            if rope_cos is None or rope_sin is None:
                raise ValueError("rotary transformer requires rope_cos and rope_sin inputs")
            q = self._apply_rotary(q, rope_cos, rope_sin)
            k = self._apply_rotary(k, rope_cos, rope_sin)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        context = torch.matmul(probabilities, v)
        merged = context.permute(0, 2, 1, 3).reshape(batch, sequence, self.hidden_size)
        residual = x + self.out_proj(merged)
        feedforward_input = self.norm2(residual)
        if self.gated:
            feedforward = self._activation(self.gate_proj(feedforward_input)) * self.up_proj(feedforward_input)
        else:
            feedforward = self._activation(self.ff_proj(feedforward_input))
        return residual + self.down_proj(feedforward)


def transformer_workload(
    spec: PaperBenchmarkSpec,
    module_type: Type[PaperTransformerBlock],
    *,
    variant: str,
    dtype: torch.dtype | None,
) -> PaperBenchmarkWorkload:
    if spec.sequence_length is None:
        raise ValueError("transformer workloads require a sequence length")
    if variant == "micro":
        batch, sequence, hidden = 1, min(spec.sequence_length, 4), 8
    elif variant == "paper_shape":
        batch, sequence, hidden = spec.batch_size, spec.sequence_length, 8
    else:
        raise ValueError("variant must be 'micro' or 'paper_shape'")

    requested_dtype = dtype or getattr(torch, spec.dtype)
    selected_dtype = torch.float32 if requested_dtype == torch.bfloat16 else requested_dtype
    torch.manual_seed(0)
    module = module_type().eval().to(dtype=selected_dtype)
    inputs = (
        torch.randn(batch, sequence, hidden, dtype=selected_dtype),
        torch.zeros(batch, 1, sequence, sequence, dtype=selected_dtype),
    )
    if getattr(module, "rotary", False):
        positions = torch.arange(sequence, dtype=selected_dtype).reshape(sequence, 1)
        frequencies = torch.arange(0, module.head_dim, 2, dtype=selected_dtype)
        frequencies = torch.exp(-math.log(10000.0) * frequencies / module.head_dim)
        angles = positions * frequencies.reshape(1, -1)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        inputs = (*inputs, cos.reshape(1, 1, sequence, module.head_dim), sin.reshape(1, 1, sequence, module.head_dim))
    return PaperBenchmarkWorkload(
        spec=spec,
        module=module,
        inputs=inputs,
        variant=variant,
        attributes={
            "paper_reference_only": True,
            "simulation_dimensions": "scaled" if variant == "micro" else "paper_batch_sequence_scaled_hidden",
            "requested_dtype": str(requested_dtype).removeprefix("torch."),
            "simulation_dtype": str(selected_dtype).removeprefix("torch."),
            "dtype_fallback": requested_dtype != selected_dtype,
            "full_model_materialized": False,
            "compiler_route": "torch.export->torch-xla->official-stablehlo",
            "rotary_embedding": bool(getattr(module, "rotary", False)),
            "rotary_inputs": "explicit_cos_sin" if getattr(module, "rotary", False) else None,
        },
    )
