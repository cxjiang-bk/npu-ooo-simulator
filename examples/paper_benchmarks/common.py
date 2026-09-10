"""Shared PyTorch building blocks used by individual paper model files."""

from __future__ import annotations

import math
from typing import Type

import torch

from npu_ooo.frontend import make_workload

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
        moe_expert_count: int = 0,
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
        if moe_expert_count < 0:
            raise ValueError("moe_expert_count must be non-negative")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.activation_name = activation
        self.gated = gated
        self.rotary = rotary
        self.moe_expert_count = moe_expert_count
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
        if moe_expert_count:
            self.router = torch.nn.Linear(hidden_size, moe_expert_count, bias=False)
            self.expert_gate_proj = torch.nn.ModuleList(
                torch.nn.Linear(hidden_size, intermediate_size)
                for _ in range(moe_expert_count)
            )
            self.expert_up_proj = torch.nn.ModuleList(
                torch.nn.Linear(hidden_size, intermediate_size)
                for _ in range(moe_expert_count)
            )
            self.expert_down_proj = torch.nn.ModuleList(
                torch.nn.Linear(intermediate_size, hidden_size)
                for _ in range(moe_expert_count)
            )
        elif gated:
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size)
            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        else:
            self.ff_proj = torch.nn.Linear(hidden_size, intermediate_size)
        if not moe_expert_count:
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

    def _feed_forward(
        self,
        value: torch.Tensor,
        routing_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.moe_expert_count:
            if routing_mask is None:
                raise ValueError("MoE block requires routing_mask")
            if routing_mask.shape[-1] != self.moe_expert_count:
                raise ValueError("routing_mask expert dimension does not match the block")
            routing_weights = torch.softmax(self.router(value), dim=-1) * routing_mask
            combined = torch.zeros_like(value)
            for expert in range(self.moe_expert_count):
                activated = self._activation(self.expert_gate_proj[expert](value))
                hidden = activated * self.expert_up_proj[expert](value)
                expert_output = self.expert_down_proj[expert](hidden)
                combined = combined + routing_weights[..., expert : expert + 1] * expert_output
            return combined
        if self.gated:
            hidden = self._activation(self.gate_proj(value)) * self.up_proj(value)
        else:
            hidden = self._activation(self.ff_proj(value))
        return self.down_proj(hidden)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        routing_mask: torch.Tensor | None = None,
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
        return residual + self._feed_forward(feedforward_input, routing_mask)


class RepeatedTransformer(torch.nn.Module):
    """A sequential stack of separately parameterized transformer blocks."""

    def __init__(
        self,
        block_type: Type[PaperTransformerBlock],
        layer_count: int,
    ) -> None:
        super().__init__()
        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        self.layers = torch.nn.ModuleList(block_type() for _ in range(layer_count))
        first = self.layers[0]
        self.layer_count = layer_count
        self.hidden_size = first.hidden_size
        self.rotary = first.rotary
        self.head_dim = first.head_dim
        self.num_heads = first.num_heads
        self.moe_expert_count = first.moe_expert_count

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        routing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask, rope_cos, rope_sin, routing_mask)
        return x


class PaperTransformerModelProxy(torch.nn.Module):
    """Scaled model shell around a one-block or repeated-block backbone.

    Token/position embeddings and masks are real PyTorch operations.  MoE
    routing weights remain explicit inputs so a request can change routing
    without recompiling the model; expert compaction is intentionally outside
    this proxy contract.
    """

    def __init__(
        self,
        backbone: PaperTransformerBlock | RepeatedTransformer,
        *,
        sequence_length: int,
        vocabulary_size: int = 32,
        learned_position_embedding: bool = False,
        token_type_embedding: bool = False,
        causal_mask: bool = False,
        output_head: bool = False,
    ) -> None:
        super().__init__()
        if sequence_length <= 0 or vocabulary_size <= 0:
            raise ValueError("sequence_length and vocabulary_size must be positive")
        self.backbone = backbone
        self.hidden_size = backbone.hidden_size
        self.num_heads = backbone.num_heads
        self.head_dim = backbone.head_dim
        self.rotary = backbone.rotary
        self.moe_expert_count = backbone.moe_expert_count
        self.token_embedding = torch.nn.Embedding(vocabulary_size, self.hidden_size)
        self.position_embedding = (
            torch.nn.Embedding(sequence_length, self.hidden_size)
            if learned_position_embedding
            else None
        )
        self.token_type_embedding = (
            torch.nn.Embedding(2, self.hidden_size)
            if token_type_embedding
            else None
        )
        self.output_head = (
            torch.nn.Linear(self.hidden_size, vocabulary_size, bias=False)
            if output_head
            else None
        )
        if causal_mask:
            mask = torch.full((sequence_length, sequence_length), float("-inf"))
            mask = torch.triu(mask, diagonal=1).reshape(1, 1, sequence_length, sequence_length)
        else:
            mask = torch.zeros(1, 1, sequence_length, sequence_length)
        self.register_buffer("attention_mask", mask)
        if self.rotary:
            positions = torch.arange(sequence_length, dtype=torch.float32).reshape(sequence_length, 1)
            frequencies = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
            frequencies = torch.exp(-math.log(10000.0) * frequencies / self.head_dim)
            angles = positions * frequencies.reshape(1, -1)
            cos = torch.cos(angles).repeat_interleave(2, dim=-1)
            sin = torch.sin(angles).repeat_interleave(2, dim=-1)
            self.register_buffer("rope_cos", cos.reshape(1, 1, sequence_length, self.head_dim))
            self.register_buffer("rope_sin", sin.reshape(1, 1, sequence_length, self.head_dim))
        else:
            self.rope_cos = None
            self.rope_sin = None

    def forward(
        self,
        token_ids: torch.Tensor,
        auxiliary0: torch.Tensor | None = None,
        auxiliary1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            if auxiliary0 is None:
                raise ValueError("learned position embedding requires position_ids")
            value = value + self.position_embedding(auxiliary0)
        if self.token_type_embedding is not None:
            if auxiliary1 is None:
                raise ValueError("token type embedding requires token_type_ids")
            value = value + self.token_type_embedding(auxiliary1)
        routing_mask = auxiliary0 if self.moe_expert_count else None
        value = self.backbone(
            value,
            self.attention_mask,
            self.rope_cos,
            self.rope_sin,
            routing_mask,
        )
        return self.output_head(value) if self.output_head is not None else value


def transformer_workload(
    spec: PaperBenchmarkSpec,
    module_type: Type[PaperTransformerBlock],
    *,
    variant: str,
    dtype: torch.dtype | None,
    layer_count: int = 1,
    model_scope: str = "one_block",
    seed: int | None = 0,
) -> PaperBenchmarkWorkload:
    if spec.sequence_length is None:
        raise ValueError("transformer workloads require a sequence length")
    if variant == "micro":
        batch, sequence, hidden = 1, min(spec.sequence_length, 4), 8
    elif variant == "paper_shape":
        batch, sequence, hidden = spec.batch_size, spec.sequence_length, 8
    else:
        raise ValueError("variant must be 'micro' or 'paper_shape'")

    if model_scope not in {"one_block", "model_proxy"}:
        raise ValueError("model_scope must be 'one_block' or 'model_proxy'")
    requested_dtype = dtype or getattr(torch, spec.dtype)
    selected_dtype = torch.float32 if requested_dtype == torch.bfloat16 else requested_dtype
    if seed is not None:
        torch.manual_seed(seed)
    backbone = (
        module_type()
        if layer_count == 1
        else RepeatedTransformer(module_type, layer_count)
    )
    causal = spec.model_family != "encoder_transformer"
    attention_mask = torch.zeros(batch, 1, sequence, sequence, dtype=selected_dtype)
    if causal:
        attention_mask = torch.triu(
            torch.full((sequence, sequence), float("-inf"), dtype=selected_dtype),
            diagonal=1,
        ).reshape(1, 1, sequence, sequence).expand(batch, 1, sequence, sequence)
    routing_mask = None
    if backbone.moe_expert_count:
        routing_mask = torch.zeros(
            batch,
            sequence,
            backbone.moe_expert_count,
            dtype=selected_dtype,
        )
        for token in range(sequence):
            first = token % backbone.moe_expert_count
            second = (token + 1) % backbone.moe_expert_count
            routing_mask[:, token, first] = 1.0
            routing_mask[:, token, second] = 1.0
    if model_scope == "model_proxy":
        learned_position = spec.model_family == "encoder_transformer"
        token_type = spec.model_family == "encoder_transformer"
        module = PaperTransformerModelProxy(
            backbone,
            sequence_length=sequence,
            learned_position_embedding=learned_position,
            token_type_embedding=token_type,
            causal_mask=causal,
            output_head=spec.model_family != "encoder_transformer",
        ).eval().to(dtype=selected_dtype)
        token_ids = torch.arange(batch * sequence, dtype=torch.int64).reshape(batch, sequence) % 32
        inputs: tuple[torch.Tensor, ...] = (token_ids,)
        if learned_position:
            inputs = (
                *inputs,
                torch.arange(sequence, dtype=torch.int64).reshape(1, sequence).expand(batch, sequence),
            )
        if token_type:
            inputs = (*inputs, torch.zeros(batch, sequence, dtype=torch.int64))
        if routing_mask is not None:
            inputs = (*inputs, routing_mask)
    else:
        module = backbone.eval().to(dtype=selected_dtype)
        inputs = (
            torch.randn(batch, sequence, hidden, dtype=selected_dtype),
            attention_mask,
        )
    if model_scope == "one_block" and getattr(module, "rotary", False):
        positions = torch.arange(sequence, dtype=selected_dtype).reshape(sequence, 1)
        frequencies = torch.arange(0, module.head_dim, 2, dtype=selected_dtype)
        frequencies = torch.exp(-math.log(10000.0) * frequencies / module.head_dim)
        angles = positions * frequencies.reshape(1, -1)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        inputs = (*inputs, cos.reshape(1, 1, sequence, module.head_dim), sin.reshape(1, 1, sequence, module.head_dim))
    if model_scope == "one_block" and routing_mask is not None:
        inputs = (*inputs, routing_mask)
    remaining_features = set(spec.unsupported_features)
    if model_scope == "model_proxy":
        remaining_features.discard("token_embedding")
        remaining_features.discard("position_embedding")
    if getattr(backbone, "rotary", False):
        remaining_features.discard("rotary_embedding")
    if backbone.moe_expert_count:
        remaining_features.discard("moe_routing")
        remaining_features.add("dynamic_topk_selection")
        remaining_features.add("token_compaction_and_capacity")
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )
    input_bytes = sum(value.numel() * value.element_size() for value in inputs)
    return PaperBenchmarkWorkload(
        spec=spec,
        workload=make_workload(
            module,
            inputs,
            experiment_id=spec.case_id,
            provenance={"construction": "paper_benchmark_builder", "variant": variant},
        ),
        variant=variant,
        attributes={
            "paper_reference_only": True,
            "simulation_dimensions": "scaled" if variant == "micro" else "paper_batch_sequence_scaled_hidden",
            "requested_dtype": str(requested_dtype).removeprefix("torch."),
            "simulation_dtype": str(selected_dtype).removeprefix("torch."),
            "dtype_fallback": requested_dtype != selected_dtype,
            "full_model_materialized": False,
            "layer_count": layer_count,
            "materialized_scope": model_scope,
            "model_depth_proxy": "one_block" if layer_count == 1 else "sequential_repeated_blocks",
            "paper_evaluation_scope": spec.paper_evaluation_scope,
            "reference_layer_count": spec.reference_layer_count,
            "remaining_features": sorted(remaining_features),
            "causal_mask": "additive_upper_triangle" if causal else "bidirectional_zero_mask",
            "embedding": "token+learned_position+token_type" if model_scope == "model_proxy" and learned_position else (
                "token" if model_scope == "model_proxy" else "preembedded_input"
            ),
            "output_head": model_scope == "model_proxy" and spec.model_family != "encoder_transformer",
            "moe": {
                "enabled": bool(backbone.moe_expert_count),
                "expert_count": backbone.moe_expert_count,
                "top_k": 2 if backbone.moe_expert_count else 0,
                "routing_contract": "internal_softmax+external_topk_mask" if backbone.moe_expert_count else None,
                "dispatch_contract": "mask_weighted_all_experts" if backbone.moe_expert_count else None,
            },
            "model_components": {
                "token_embedding": model_scope == "model_proxy",
                "learned_position_embedding": model_scope == "model_proxy" and learned_position,
                "token_type_embedding": model_scope == "model_proxy" and token_type,
                "rotary_embedding": bool(getattr(backbone, "rotary", False)),
                "causal_mask": causal,
                "transformer_blocks": layer_count,
                "output_head": model_scope == "model_proxy" and spec.model_family != "encoder_transformer",
                "moe_experts": backbone.moe_expert_count,
            },
            "model_statistics": {
                "parameter_count": parameter_count,
                "parameter_bytes": parameter_bytes,
                "input_bytes": input_bytes,
                "input_tensor_count": len(inputs),
                "routing_nonzero_count": (
                    int(torch.count_nonzero(routing_mask).item())
                    if routing_mask is not None
                    else 0
                ),
            },
            "compiler_route": "torch.export->torch-xla->official-stablehlo",
            "rotary_embedding": bool(getattr(module, "rotary", False)),
            "rotary_inputs": "explicit_cos_sin" if getattr(module, "rotary", False) else None,
        },
    )
