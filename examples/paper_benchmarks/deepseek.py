"""DeepSeek-R1-16B prefill/decode benchmark rows and one-block workload."""

from __future__ import annotations

import torch

from .common import PaperTransformerBlock, transformer_workload
from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class DeepSeekR1OneBlock(PaperTransformerBlock):
    def __init__(self) -> None:
        super().__init__(norm="rmsnorm", activation="silu", gated=True)


_UNSUPPORTED = ("token_embedding", "rotary_embedding", "kv_cache", "moe_routing", "full_model_depth")

PREFILL_SPEC = PaperBenchmarkSpec(
    "deepseek-r1-16b-prefill", "DeepSeek-R1-16B", "decoder_reasoning", "prefill", "bfloat16", 50, 100, None,
    5120, 40, 13824, 213.5, 412.3, 1.93, "transformer_one_block", _UNSUPPORTED,
)
DECODE_SPEC = PaperBenchmarkSpec(
    "deepseek-r1-16b-decode", "DeepSeek-R1-16B", "decoder_reasoning", "decode", "bfloat16", 50, 700, None,
    5120, 40, 13824, 51.2, 69.0, 1.35, "transformer_one_block", _UNSUPPORTED,
)


def build(
    case_id: str,
    variant: str = "micro",
    dtype: torch.dtype | None = None,
    *,
    layer_count: int = 1,
) -> PaperBenchmarkWorkload:
    if case_id == PREFILL_SPEC.case_id:
        spec = PREFILL_SPEC
    elif case_id == DECODE_SPEC.case_id:
        spec = DECODE_SPEC
    else:
        raise ValueError(f"unknown DeepSeek benchmark case '{case_id}'")
    return transformer_workload(
        spec,
        DeepSeekR1OneBlock,
        variant=variant,
        dtype=dtype,
        layer_count=layer_count,
    )
