"""LLaMA2-13B benchmark row and one-block workload."""

from __future__ import annotations

import torch

from .common import PaperTransformerBlock, transformer_workload
from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class LLaMA2OneBlock(PaperTransformerBlock):
    def __init__(self) -> None:
        super().__init__(norm="rmsnorm", activation="silu", gated=True, rotary=True)


class LLaMA2DecodeOneBlock(torch.nn.Module):
    """Scaled LLaMA2 decode block with an explicit fixed-window KV cache.

    The cache contract intentionally mirrors the compiler's currently proven
    StableHLO pattern: ``slice(cache[..., 1:, :])`` followed by
    ``concatenate(new_token, dim=-2)``.  This is a real PyTorch workload for
    frontend/runtime experiments, not a full model implementation.
    """

    def __init__(
        self,
        hidden_size: int = 8,
        num_heads: int = 2,
        intermediate_size: int = 16,
        cache_window: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if intermediate_size <= 0 or cache_window <= 1:
            raise ValueError("intermediate_size must be positive and cache_window must exceed one")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.cache_window = cache_window
        if self.head_dim % 2:
            raise ValueError("rotary embedding requires an even head dimension")

        self.norm1 = torch.nn.RMSNorm(hidden_size)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.norm2 = torch.nn.RMSNorm(hidden_size)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size)

        rotation = torch.zeros(self.head_dim, self.head_dim)
        for index in range(0, self.head_dim, 2):
            rotation[index, index + 1] = 1.0
            rotation[index + 1, index] = -1.0
        self.register_buffer("rope_permutation", rotation)

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
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = x.shape
        if sequence != 1:
            raise ValueError("LLaMA2 decode workload expects one token per invocation")
        normalized = self.norm1(x)
        q = self.q_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        k = self.k_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        v = self.v_proj(normalized).reshape(batch, sequence, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q = self._apply_rotary(q, rope_cos, rope_sin)
        k = self._apply_rotary(k, rope_cos, rope_sin)

        updated_key_cache = torch.cat((key_cache[..., 1:, :], k), dim=-2)
        updated_value_cache = torch.cat((value_cache[..., 1:, :], v), dim=-2)
        scores = torch.matmul(q, updated_key_cache.transpose(-2, -1)) * (self.head_dim**-0.5)
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        context = torch.matmul(probabilities, updated_value_cache)
        merged = context.permute(0, 2, 1, 3).reshape(batch, sequence, self.hidden_size)
        residual = x + self.out_proj(merged)

        mlp_input = self.norm2(residual)
        gated = torch.nn.functional.silu(self.gate_proj(mlp_input)) * self.up_proj(mlp_input)
        output = residual + self.down_proj(gated)
        return output, updated_key_cache, updated_value_cache


SPEC = PaperBenchmarkSpec(
    "llama2-13b-oneblk", "LLaMA2-13B", "decoder_transformer", "prefill", "float16", 1, 512, None,
    5120, 40, 13824, 54.0, 77.1, 1.43, "transformer_one_block",
    ("token_embedding", "kv_cache", "full_model_depth"),
)


def build(
    variant: str = "micro",
    dtype: torch.dtype | None = None,
    *,
    layer_count: int = 1,
) -> PaperBenchmarkWorkload:
    return transformer_workload(
        SPEC,
        LLaMA2OneBlock,
        variant=variant,
        dtype=dtype,
        layer_count=layer_count,
    )


def build_decode(dtype: torch.dtype | None = None) -> PaperBenchmarkWorkload:
    """Build a one-token decode workload for the fixed-window state contract."""

    requested_dtype = dtype or getattr(torch, SPEC.dtype)
    selected_dtype = torch.float32 if requested_dtype == torch.bfloat16 else requested_dtype
    torch.manual_seed(0)
    module = LLaMA2DecodeOneBlock().eval().to(dtype=selected_dtype)
    sequence = 1
    inputs = (
        torch.randn(1, sequence, module.hidden_size, dtype=selected_dtype),
        torch.randn(1, module.num_heads, module.cache_window, module.head_dim, dtype=selected_dtype),
        torch.randn(1, module.num_heads, module.cache_window, module.head_dim, dtype=selected_dtype),
        torch.ones(1, 1, sequence, module.head_dim, dtype=selected_dtype),
        torch.zeros(1, 1, sequence, module.head_dim, dtype=selected_dtype),
        torch.zeros(1, 1, sequence, module.cache_window, dtype=selected_dtype),
    )
    return PaperBenchmarkWorkload(
        spec=SPEC,
        module=module,
        inputs=inputs,
        variant="decode_micro",
        attributes={
            "paper_reference_only": True,
            "phase": "decode",
            "simulation_dimensions": "scaled_one_token_fixed_window",
            "requested_dtype": str(requested_dtype).removeprefix("torch."),
            "simulation_dtype": str(selected_dtype).removeprefix("torch."),
            "dtype_fallback": requested_dtype != selected_dtype,
            "full_model_materialized": False,
            "compiler_route": "torch.export->torch-xla->official-stablehlo",
            "rotary_embedding": True,
            "rotary_inputs": "explicit_cos_sin",
            "kv_cache": {
                "state_ids": ("key_cache", "value_cache"),
                "cache_window": module.cache_window,
                "update_length": 1,
                "transition": "drop_oldest_append_new",
                "contract": "persistent_buffer_v1",
            },
        },
    )
