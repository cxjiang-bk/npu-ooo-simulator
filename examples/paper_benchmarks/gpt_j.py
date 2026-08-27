"""GPT-J-6B benchmark row and one-block workload."""

from __future__ import annotations

import torch

from .common import PaperTransformerBlock, transformer_workload
from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class GPTJ6BOneBlock(PaperTransformerBlock):
    def __init__(self) -> None:
        super().__init__(norm="layernorm", activation="gelu_tanh", gated=False)


SPEC = PaperBenchmarkSpec(
    "gpt-j-6b-oneblk", "GPT-J-6B", "decoder_transformer", "prefill", "float16", 1, 512, None,
    4096, 16, 16384, 29.9, 37.3, 1.25, "transformer_one_block",
    ("token_embedding", "rotary_embedding", "full_model_depth", "exact_gelu"),
)


def build(variant: str = "micro", dtype: torch.dtype | None = None) -> PaperBenchmarkWorkload:
    return transformer_workload(SPEC, GPTJ6BOneBlock, variant=variant, dtype=dtype)
