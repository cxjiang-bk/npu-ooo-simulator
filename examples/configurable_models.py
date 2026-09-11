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


class FlashAttention(torch.nn.Module):
    """Exact block-streaming attention with online softmax state.

    Only one ``QK`` score block is materialized at a time.  The running row
    maximum, normalization sum and output accumulator follow the FlashAttention
    recurrence, and the Python loop is statically unrolled by ``torch.export``.
    """

    def __init__(
        self,
        query_block_size: int = 4,
        kv_block_size: int = 4,
    ) -> None:
        super().__init__()
        if query_block_size <= 0 or kv_block_size <= 0:
            raise ValueError("FlashAttention block sizes must be positive")
        self.query_block_size = query_block_size
        self.kv_block_size = kv_block_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("FlashAttention expects rank-4 [batch, heads, sequence, head] tensors")
        if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
            raise ValueError("FlashAttention Q/K/V shapes are incompatible")
        if attention_mask.shape[-2:] != (q.shape[-2], k.shape[-2]):
            raise ValueError("attention_mask must cover query and key sequence dimensions")

        scale = q.shape[-1] ** -0.5
        output_blocks: list[torch.Tensor] = []
        for query_start in range(0, q.shape[-2], self.query_block_size):
            query_stop = min(query_start + self.query_block_size, q.shape[-2])
            query_block = q[..., query_start:query_stop, :]
            running_max: torch.Tensor | None = None
            running_sum: torch.Tensor | None = None
            output_accumulator: torch.Tensor | None = None
            for key_start in range(0, k.shape[-2], self.kv_block_size):
                key_stop = min(key_start + self.kv_block_size, k.shape[-2])
                key_block = k[..., key_start:key_stop, :]
                value_block = v[..., key_start:key_stop, :]
                scores = torch.matmul(query_block, key_block.transpose(-2, -1)) * scale
                scores = scores + attention_mask[
                    ..., query_start:query_stop, key_start:key_stop
                ]
                block_max = torch.amax(scores, dim=-1, keepdim=True)
                if running_max is None:
                    running_max = block_max
                    probabilities = torch.exp(scores - running_max)
                    running_sum = torch.sum(probabilities, dim=-1, keepdim=True)
                    output_accumulator = torch.matmul(probabilities, value_block)
                    continue
                new_max = torch.maximum(running_max, block_max)
                previous_scale = torch.exp(running_max - new_max)
                probabilities = torch.exp(scores - new_max)
                assert running_sum is not None and output_accumulator is not None
                running_sum = running_sum * previous_scale + torch.sum(
                    probabilities, dim=-1, keepdim=True
                )
                output_accumulator = output_accumulator * previous_scale + torch.matmul(
                    probabilities, value_block
                )
                running_max = new_max
            if running_sum is None or output_accumulator is None:
                raise ValueError("FlashAttention requires a non-empty key sequence")
            output_blocks.append(output_accumulator / running_sum)
        if not output_blocks:
            raise ValueError("FlashAttention requires a non-empty query sequence")
        return torch.cat(output_blocks, dim=-2)


class Top2MoE(torch.nn.Module):
    """Router-owned normalized top-2 MoE with SwiGLU experts.

    Selection is internal to the operator; callers do not supply a routing
    mask.  All expert branches remain in the exported graph and are combined
    with exact sparse weights.  Dynamic token compaction and expert capacity
    are deliberately separate backend/runtime contracts.
    """

    def __init__(
        self,
        hidden_size: int = 8,
        intermediate_size: int = 16,
        expert_count: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("MoE hidden and intermediate sizes must be positive")
        if expert_count < 2:
            raise ValueError("Top2MoE requires at least two experts")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_count = expert_count
        self.router = torch.nn.Linear(hidden_size, expert_count, bias=False)
        self.gate_proj = torch.nn.ModuleList(
            torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(expert_count)
        )
        self.up_proj = torch.nn.ModuleList(
            torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(expert_count)
        )
        self.down_proj = torch.nn.ModuleList(
            torch.nn.Linear(intermediate_size, hidden_size, bias=False)
            for _ in range(expert_count)
        )
        self.register_buffer(
            "ranking_tie_break",
            torch.arange(expert_count, dtype=torch.float32) * 1e-7,
        )

    def routing_weights(self, value: torch.Tensor) -> torch.Tensor:
        logits = self.router(value)
        probabilities = torch.softmax(logits, dim=-1)
        ranking = logits + self.ranking_tie_break
        first = torch.amax(ranking, dim=-1, keepdim=True)
        first_mask = ranking == first
        remaining = torch.where(
            first_mask,
            torch.full_like(ranking, float("-inf")),
            ranking,
        )
        second = torch.amax(remaining, dim=-1, keepdim=True)
        second_mask = remaining == second
        selected = first_mask.to(dtype=value.dtype) + second_mask.to(dtype=value.dtype)
        sparse_weights = probabilities * selected
        return sparse_weights / torch.sum(sparse_weights, dim=-1, keepdim=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.hidden_size:
            raise ValueError("Top2MoE input hidden dimension does not match hidden_size")
        weights = self.routing_weights(value)
        combined = torch.zeros_like(value)
        for expert in range(self.expert_count):
            gate = torch.nn.functional.silu(self.gate_proj[expert](value))
            hidden = gate * self.up_proj[expert](value)
            expert_output = self.down_proj[expert](hidden)
            combined = combined + weights[..., expert : expert + 1] * expert_output
        return combined


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


def build_flash_attention_workload(
    batch_size: int = 1,
    num_heads: int = 2,
    query_length: int = 8,
    key_length: int = 8,
    head_dim: int = 4,
    query_block_size: int = 4,
    kv_block_size: int = 4,
    causal: bool = True,
    dtype: str = "float32",
) -> Workload:
    """Build a true online-softmax FlashAttention workload."""

    selected_dtype = getattr(torch, dtype)
    query_shape = (batch_size, num_heads, query_length, head_dim)
    key_shape = (batch_size, num_heads, key_length, head_dim)
    module = FlashAttention(
        query_block_size=query_block_size,
        kv_block_size=kv_block_size,
    ).eval()
    q = torch.randn(*query_shape, dtype=selected_dtype)
    k = torch.randn(*key_shape, dtype=selected_dtype)
    v = torch.randn(*key_shape, dtype=selected_dtype)
    if causal:
        query_positions = torch.arange(query_length).reshape(query_length, 1)
        key_positions = torch.arange(key_length).reshape(1, key_length)
        visible = key_positions <= query_positions + max(0, key_length - query_length)
        mask = torch.where(
            visible,
            torch.tensor(0.0, dtype=selected_dtype),
            torch.tensor(float("-inf"), dtype=selected_dtype),
        ).reshape(1, 1, query_length, key_length)
    else:
        mask = torch.zeros(1, 1, query_length, key_length, dtype=selected_dtype)
    return make_workload(
        module,
        (q, k, v, mask),
        experiment_id="flash-attention-online",
        provenance={
            "algorithm": "flash_attention_online_softmax",
            "query_block_size": query_block_size,
            "kv_block_size": kv_block_size,
            "materializes_full_attention_matrix": False,
            "causal": causal,
            "static_dimensions": {
                "batch_size": batch_size,
                "num_heads": num_heads,
                "query_length": query_length,
                "key_length": key_length,
                "head_dim": head_dim,
            },
        },
    )


def build_top2_moe_workload(
    batch_size: int = 1,
    sequence_length: int = 8,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    expert_count: int = 4,
    dtype: str = "float32",
) -> Workload:
    """Build an internally routed top-2 MoE functional workload."""

    selected_dtype = getattr(torch, dtype)
    module = Top2MoE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        expert_count=expert_count,
    ).eval().to(dtype=selected_dtype)
    value = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        dtype=selected_dtype,
    )
    return make_workload(
        module,
        (value,),
        experiment_id="top2-moe",
        provenance={
            "algorithm": "normalized_top2_router_swiglu_experts",
            "routing_contract": "internal_router+internal_top2",
            "top_k": 2,
            "expert_count": expert_count,
            "dispatch_contract": "sparse_weights+dense_reference_expert_branches",
            "dynamic_token_compaction": False,
            "expert_capacity": None,
            "static_dimensions": {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
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
    "FlashAttention",
    "Top2MoE",
    "build_attention_workload",
    "build_flash_attention_workload",
    "build_paper_model_workload",
    "build_top2_moe_workload",
]
