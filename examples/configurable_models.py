"""Small configurable PyTorch models for the generic JSON workload path.

These are executable model proxies built from ordinary PyTorch operators.  They
are not registered paper benchmarks and do not download weights.
"""

from __future__ import annotations

import torch

from npu_ooo.frontend import Workload, make_workload


class Matmul(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)


class MultiInputAttention(torch.nn.Module):
    """QK-softmax-PV attention with an explicit additive mask."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        return torch.matmul(probabilities, v)


class InputContractProbe(torch.nn.Module):
    """Forward signature used to exercise mixed and nested static inputs."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.width = width

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        optional: None,
        nested: tuple[list[torch.Tensor], dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        value = token_ids.to(dtype=torch.float32) * scale
        value = value + mask.to(dtype=torch.float32)
        return value + nested[0][0] + nested[1]["bias"]


class ScalarTensorProbe(torch.nn.Module):
    def forward(self, value: torch.Tensor, scale: float) -> torch.Tensor:
        return value * scale


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size)

    @staticmethod
    def _gelu_tanh(value: torch.Tensor) -> torch.Tensor:
        coefficient = 0.7978845608028654
        return 0.5 * value * (
            1.0 + torch.tanh(coefficient * (value + 0.044715 * value**3))
        )

    def forward(self, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, sequence, _hidden = value.shape
        normalized = self.norm1(value)
        q = self.q_proj(normalized).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(normalized).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(normalized).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        context = torch.matmul(probabilities, v)
        merged = context.permute(0, 2, 1, 3).reshape(
            batch, sequence, self.hidden_size
        )
        residual = value + self.out_proj(merged)
        feedforward = self.down_proj(self._gelu_tanh(self.up_proj(self.norm2(residual))))
        return residual + feedforward


class ConfigurableBertProxy(torch.nn.Module):
    """Parameterizable encoder shell with real embeddings and repeated layers."""

    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 8,
        num_heads: int = 2,
        intermediate_size: int = 16,
        vocab_size: int = 32,
        max_position_embeddings: int = 16,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if vocab_size <= 0 or max_position_embeddings <= 0:
            raise ValueError("embedding sizes must be positive")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = torch.nn.Embedding(2, hidden_size)
        self.layers = torch.nn.ModuleList(
            EncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        value = (
            self.token_embedding(input_ids)
            + self.position_embedding(position_ids)
            + self.token_type_embedding(token_type_ids)
        )
        for layer in self.layers:
            value = layer(value, attention_mask)
        return value


def build_attention_workload(
    batch_size: int = 1,
    num_heads: int = 2,
    sequence_length: int = 4,
    head_dim: int = 4,
    dtype: str = "float32",
) -> Workload:
    """Python workload factory example for a structured causal mask."""

    selected_dtype = getattr(torch, dtype)
    shape = (batch_size, num_heads, sequence_length, head_dim)
    module = MultiInputAttention().eval()
    q = torch.randn(*shape, dtype=selected_dtype)
    k = torch.randn(*shape, dtype=selected_dtype)
    v = torch.randn(*shape, dtype=selected_dtype)
    mask = torch.triu(
        torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            dtype=selected_dtype,
        ),
        diagonal=1,
    ).reshape(1, 1, sequence_length, sequence_length)
    return make_workload(
        module,
        (q, k, v),
        kwargs={"attention_mask": mask},
        experiment_id="configurable-attention",
        provenance={
            "input_initializers": {
                "q": "randn",
                "k": "randn",
                "v": "randn",
                "attention_mask": "causal_additive_upper_triangle",
            },
            "static_dimensions": {
                "batch_size": batch_size,
                "num_heads": num_heads,
                "sequence_length": sequence_length,
                "head_dim": head_dim,
            },
        },
    )


def build_paper_model_workload(
    case_id: str,
    variant: str = "micro",
    layer_count: int = 1,
    model_scope: str = "one_block",
    deepseek_mode: str = "dense",
    dtype: str | None = None,
) -> Workload:
    """Expose a paper example through the generic Workload factory contract.

    The paper registry remains responsible only for selecting its model proxy
    and experiment scale.  The returned value is the same benchmark-independent
    Workload consumed by the normal JSON compiler path.
    """

    from examples.paper_benchmarks import build_paper_benchmark

    paper = build_paper_benchmark(
        case_id,
        variant=variant,
        dtype=getattr(torch, dtype) if dtype is not None else None,
        layer_count=layer_count,
        model_scope=model_scope,
        deepseek_mode=deepseek_mode,
        seed=None,
    )
    workload = paper.workload
    return make_workload(
        workload.module,
        workload.args,
        kwargs=workload.kwargs,
        experiment_id=workload.experiment_id,
        provenance={
            **dict(workload.provenance),
            "paper_benchmark": {
                "spec": paper.spec.to_dict(),
                "variant": paper.variant,
                "attributes": dict(paper.attributes),
            },
        },
    )


__all__ = [
    "ConfigurableBertProxy",
    "EncoderLayer",
    "InputContractProbe",
    "Matmul",
    "MultiInputAttention",
    "ScalarTensorProbe",
    "build_attention_workload",
    "build_paper_model_workload",
]
