"""DeepSeek-R1-16B prefill/decode benchmark rows and one-block workload."""

from __future__ import annotations

import torch

from npu_ooo.frontend import make_workload

from .common import PaperTransformerBlock, transformer_workload
from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class DeepSeekR1OneBlock(PaperTransformerBlock):
    def __init__(self) -> None:
        super().__init__(
            norm="rmsnorm",
            activation="silu",
            gated=True,
            rotary=True,
        )


class DeepSeekR1MoEProxyBlock(PaperTransformerBlock):
    """Scaled expert block with externally supplied sparse routing weights."""

    def __init__(self) -> None:
        super().__init__(
            norm="rmsnorm",
            activation="silu",
            gated=True,
            rotary=True,
            moe_expert_count=4,
        )


class DeepSeekR1DecodeProxy(torch.nn.Module):
    """One-token decode with explicit key/value cache state."""

    def __init__(self, *, mode: str, cache_window: int) -> None:
        super().__init__()
        if cache_window <= 1:
            raise ValueError("cache_window must exceed one")
        self.block = (
            DeepSeekR1MoEProxyBlock()
            if mode == "moe_proxy"
            else DeepSeekR1OneBlock()
        )
        self.cache_window = cache_window
        self.hidden_size = self.block.hidden_size
        self.num_heads = self.block.num_heads
        self.head_dim = self.block.head_dim
        self.moe_expert_count = self.block.moe_expert_count

    def forward(
        self,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        routing_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = value.shape
        if sequence != 1:
            raise ValueError("DeepSeek decode proxy expects one token per invocation")
        normalized = self.block.norm1(value)
        query = self.block.q_proj(normalized).reshape(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        )
        key = self.block.k_proj(normalized).reshape(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        )
        current_value = self.block.v_proj(normalized).reshape(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        )
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        current_value = current_value.permute(0, 2, 1, 3)
        query = self.block._apply_rotary(query, rope_cos, rope_sin)
        key = self.block._apply_rotary(key, rope_cos, rope_sin)
        updated_key_cache = torch.cat((key_cache[..., 1:, :], key), dim=-2)
        updated_value_cache = torch.cat(
            (value_cache[..., 1:, :], current_value),
            dim=-2,
        )
        scores = torch.matmul(query, updated_key_cache.transpose(-2, -1))
        scores = scores * (self.head_dim**-0.5)
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        context = torch.matmul(probabilities, updated_value_cache)
        merged = context.permute(0, 2, 1, 3).reshape(
            batch,
            sequence,
            self.hidden_size,
        )
        residual = value + self.block.out_proj(merged)
        feedforward_input = self.block.norm2(residual)
        output = residual + self.block._feed_forward(
            feedforward_input,
            routing_mask,
        )
        return output, updated_key_cache, updated_value_cache


class DeepSeekR1DecodeModelProxy(torch.nn.Module):
    """Token embedding and output head around the stateful decode proxy."""

    def __init__(self, *, mode: str, cache_window: int, vocabulary_size: int = 32) -> None:
        super().__init__()
        self.decode = DeepSeekR1DecodeProxy(mode=mode, cache_window=cache_window)
        self.token_embedding = torch.nn.Embedding(vocabulary_size, self.decode.hidden_size)
        self.output_head = torch.nn.Linear(
            self.decode.hidden_size,
            vocabulary_size,
            bias=False,
        )
        self.hidden_size = self.decode.hidden_size
        self.num_heads = self.decode.num_heads
        self.head_dim = self.decode.head_dim
        self.moe_expert_count = self.decode.moe_expert_count

    def forward(
        self,
        token_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        routing_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, updated_key_cache, updated_value_cache = self.decode(
            self.token_embedding(token_ids),
            key_cache,
            value_cache,
            rope_cos,
            rope_sin,
            attention_mask,
            routing_mask,
        )
        return self.output_head(output), updated_key_cache, updated_value_cache


_UNSUPPORTED = ("token_embedding", "kv_cache", "moe_routing", "full_model_depth")

PREFILL_SPEC = PaperBenchmarkSpec(
    "deepseek-r1-16b-prefill", "DeepSeek-R1-16B", "decoder_reasoning", "prefill", "bfloat16", 50, 100, None,
    5120, 40, 13824, 213.5, 412.3, 1.93, "transformer_one_block", _UNSUPPORTED,
    "full_model", None,
)
DECODE_SPEC = PaperBenchmarkSpec(
    "deepseek-r1-16b-decode", "DeepSeek-R1-16B", "decoder_reasoning", "decode", "bfloat16", 50, 700, None,
    5120, 40, 13824, 51.2, 69.0, 1.35, "transformer_one_block", _UNSUPPORTED,
    "full_model", None,
)


def _decode_workload(
    spec: PaperBenchmarkSpec,
    *,
    variant: str,
    dtype: torch.dtype | None,
    model_scope: str,
    mode: str,
    seed: int | None,
) -> PaperBenchmarkWorkload:
    if variant == "micro":
        batch, cache_window = 1, 4
    elif variant == "paper_shape":
        batch, cache_window = spec.batch_size, int(spec.sequence_length or 1)
    else:
        raise ValueError("variant must be 'micro' or 'paper_shape'")
    requested_dtype = dtype or getattr(torch, spec.dtype)
    selected_dtype = torch.float32 if requested_dtype == torch.bfloat16 else requested_dtype
    if seed is not None:
        torch.manual_seed(seed)
    module: torch.nn.Module
    if model_scope == "model_proxy":
        module = DeepSeekR1DecodeModelProxy(mode=mode, cache_window=cache_window)
        primary_input = torch.arange(batch, dtype=torch.int64).reshape(batch, 1) % 32
    else:
        module = DeepSeekR1DecodeProxy(mode=mode, cache_window=cache_window)
        primary_input = torch.randn(batch, 1, module.hidden_size, dtype=selected_dtype)
    module = module.eval().to(dtype=selected_dtype)
    num_heads = module.num_heads
    head_dim = module.head_dim
    inputs: tuple[torch.Tensor, ...] = (
        primary_input,
        torch.randn(batch, num_heads, cache_window, head_dim, dtype=selected_dtype),
        torch.randn(batch, num_heads, cache_window, head_dim, dtype=selected_dtype),
        torch.ones(1, 1, 1, head_dim, dtype=selected_dtype),
        torch.zeros(1, 1, 1, head_dim, dtype=selected_dtype),
        torch.zeros(batch, 1, 1, cache_window, dtype=selected_dtype),
    )
    routing_mask = None
    if module.moe_expert_count:
        routing_mask = torch.zeros(
            batch,
            1,
            module.moe_expert_count,
            dtype=selected_dtype,
        )
        routing_mask[..., 0] = 1.0
        routing_mask[..., 1] = 1.0
        inputs = (*inputs, routing_mask)
    remaining_features = set(spec.unsupported_features)
    remaining_features.discard("rotary_embedding")
    remaining_features.discard("kv_cache")
    if model_scope == "model_proxy":
        remaining_features.discard("token_embedding")
    if mode == "moe_proxy":
        remaining_features.discard("moe_routing")
        remaining_features.add("dynamic_topk_selection")
        remaining_features.add("token_compaction_and_capacity")
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
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
            "phase": "decode",
            "simulation_dimensions": (
                "scaled_one_token_fixed_window"
                if variant == "micro"
                else "paper_batch_context_scaled_hidden_one_token"
            ),
            "requested_dtype": str(requested_dtype).removeprefix("torch."),
            "simulation_dtype": str(selected_dtype).removeprefix("torch."),
            "dtype_fallback": requested_dtype != selected_dtype,
            "full_model_materialized": False,
            "layer_count": 1,
            "materialized_scope": model_scope,
            "model_depth_proxy": "one_block",
            "paper_evaluation_scope": spec.paper_evaluation_scope,
            "reference_layer_count": spec.reference_layer_count,
            "remaining_features": sorted(remaining_features),
            "causal_mask": "past_and_current_cache_window",
            "embedding": "token" if model_scope == "model_proxy" else "preembedded_input",
            "output_head": model_scope == "model_proxy",
            "rotary_embedding": True,
            "rotary_inputs": "explicit_cos_sin",
            "kv_cache": {
                "state_ids": ("key_cache", "value_cache"),
                "cache_window": cache_window,
                "update_length": 1,
                "transition": "drop_oldest_append_new",
                "contract": "persistent_buffer_v1",
            },
            "moe": {
                "enabled": mode == "moe_proxy",
                "expert_count": module.moe_expert_count,
                "top_k": 2 if mode == "moe_proxy" else 0,
                "routing_contract": "internal_softmax+external_topk_mask" if mode == "moe_proxy" else None,
                "dispatch_contract": "mask_weighted_all_experts" if mode == "moe_proxy" else None,
            },
            "model_components": {
                "token_embedding": model_scope == "model_proxy",
                "rotary_embedding": True,
                "causal_mask": True,
                "transformer_blocks": 1,
                "kv_cache": True,
                "output_head": model_scope == "model_proxy",
                "moe_experts": module.moe_expert_count,
            },
            "model_statistics": {
                "parameter_count": parameter_count,
                "parameter_bytes": sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in module.parameters()
                ),
                "input_bytes": sum(value.numel() * value.element_size() for value in inputs),
                "input_tensor_count": len(inputs),
                "routing_nonzero_count": (
                    int(torch.count_nonzero(routing_mask).item())
                    if routing_mask is not None
                    else 0
                ),
            },
            "compiler_route": "torch.export->torch-xla->official-stablehlo",
        },
    )


def build(
    case_id: str,
    variant: str = "micro",
    dtype: torch.dtype | None = None,
    *,
    layer_count: int = 1,
    model_scope: str = "one_block",
    mode: str = "dense",
    seed: int | None = 0,
) -> PaperBenchmarkWorkload:
    if case_id == PREFILL_SPEC.case_id:
        spec = PREFILL_SPEC
    elif case_id == DECODE_SPEC.case_id:
        spec = DECODE_SPEC
    else:
        raise ValueError(f"unknown DeepSeek benchmark case '{case_id}'")
    if mode not in {"dense", "moe_proxy"}:
        raise ValueError("DeepSeek mode must be 'dense' or 'moe_proxy'")
    if spec is DECODE_SPEC:
        if layer_count != 1:
            raise ValueError(
                "DeepSeek decode proxy currently requires layer_count=1 because each layer owns distinct KV state"
            )
        return _decode_workload(
            spec,
            variant=variant,
            dtype=dtype,
            model_scope=model_scope,
            mode=mode,
            seed=seed,
        )
    return transformer_workload(
        spec,
        DeepSeekR1MoEProxyBlock if mode == "moe_proxy" else DeepSeekR1OneBlock,
        variant=variant,
        dtype=dtype,
        layer_count=layer_count,
        model_scope=model_scope,
        seed=seed,
    )
