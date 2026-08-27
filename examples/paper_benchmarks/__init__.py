"""Paper Table IX benchmark registry.

Each model family lives in its own module.  The public functions below keep
the original registry API stable while making model definitions independently
readable and extensible.
"""

from __future__ import annotations

import torch

from . import bert_base, deepseek, gpt_j, llama2, resnet50
from .common import PaperTransformerBlock
from .resnet50 import ResNet50BottleneckWorkload
from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload

_SPECS: tuple[PaperBenchmarkSpec, ...] = (
    resnet50.SPEC,
    bert_base.SPEC,
    gpt_j.SPEC,
    llama2.SPEC,
    deepseek.PREFILL_SPEC,
    deepseek.DECODE_SPEC,
)
_SPEC_BY_ID = {spec.case_id: spec for spec in _SPECS}


def paper_benchmark_specs() -> tuple[PaperBenchmarkSpec, ...]:
    """Return the immutable Table IX registry in paper order."""

    return _SPECS


def get_paper_benchmark(case_id: str) -> PaperBenchmarkSpec:
    try:
        return _SPEC_BY_ID[case_id]
    except KeyError as exc:
        known = ", ".join(_SPEC_BY_ID)
        raise ValueError(f"unknown paper benchmark '{case_id}'; choose one of: {known}") from exc


def build_paper_benchmark(
    case_id: str,
    *,
    variant: str = "micro",
    dtype: torch.dtype | None = None,
) -> PaperBenchmarkWorkload:
    """Build a real PyTorch workload and deterministic example inputs."""

    if case_id == resnet50.SPEC.case_id:
        return resnet50.build(variant=variant, dtype=dtype)
    if case_id == bert_base.SPEC.case_id:
        return bert_base.build(variant=variant, dtype=dtype)
    if case_id == gpt_j.SPEC.case_id:
        return gpt_j.build(variant=variant, dtype=dtype)
    if case_id == llama2.SPEC.case_id:
        return llama2.build(variant=variant, dtype=dtype)
    return deepseek.build(case_id, variant=variant, dtype=dtype)


__all__ = [
    "PaperBenchmarkSpec",
    "PaperBenchmarkWorkload",
    "PaperTransformerBlock",
    "ResNet50BottleneckWorkload",
    "build_paper_benchmark",
    "get_paper_benchmark",
    "paper_benchmark_specs",
]
