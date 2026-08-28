"""LLaMA2-13B benchmark row and one-block workload."""

from __future__ import annotations

import torch

from .common import PaperTransformerBlock, transformer_workload
from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class LLaMA2OneBlock(PaperTransformerBlock):
    def __init__(self) -> None:
        super().__init__(norm="rmsnorm", activation="silu", gated=True, rotary=True)


SPEC = PaperBenchmarkSpec(
    "llama2-13b-oneblk", "LLaMA2-13B", "decoder_transformer", "prefill", "float16", 1, 512, None,
    5120, 40, 13824, 54.0, 77.1, 1.43, "transformer_one_block",
    ("token_embedding", "kv_cache", "full_model_depth"),
)


def build(variant: str = "micro", dtype: torch.dtype | None = None) -> PaperBenchmarkWorkload:
    return transformer_workload(SPEC, LLaMA2OneBlock, variant=variant, dtype=dtype)
