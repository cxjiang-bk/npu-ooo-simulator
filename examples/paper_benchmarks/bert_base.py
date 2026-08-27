"""BERT-Base benchmark row and one-block workload."""

from __future__ import annotations

import torch

from .common import PaperTransformerBlock, transformer_workload
from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class BertBaseOneBlock(PaperTransformerBlock):
    def __init__(self) -> None:
        super().__init__(norm="layernorm", activation="gelu_tanh", gated=False)


SPEC = PaperBenchmarkSpec(
    "bert-base", "BERT-Base", "encoder_transformer", "inference", "float16", 64, 128, None,
    768, 12, 3072, 7.5, 9.8, 1.31, "transformer_one_block",
    ("token_embedding", "position_embedding", "full_model_depth", "exact_gelu"),
)


def build(variant: str = "micro", dtype: torch.dtype | None = None) -> PaperBenchmarkWorkload:
    return transformer_workload(SPEC, BertBaseOneBlock, variant=variant, dtype=dtype)
