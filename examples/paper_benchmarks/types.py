"""Shared data contracts for the paper benchmark examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class PaperBenchmarkSpec:
    """A row from the paper plus the semantic workload contract."""

    case_id: str
    model_name: str
    model_family: str
    phase: str
    dtype: str
    batch_size: int
    sequence_length: int | None
    image_size: tuple[int, int] | None
    hidden_size: int | None
    num_heads: int | None
    intermediate_size: int | None
    reference_epoch_ms: float
    reference_a100_ms: float
    reference_speedup: float
    workload_kind: str
    unsupported_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "phase": self.phase,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "image_size": list(self.image_size) if self.image_size else None,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "reference": {
                "epoch_ms": self.reference_epoch_ms,
                "a100_ms": self.reference_a100_ms,
                "speedup": self.reference_speedup,
                "source": "TISA Table IX",
            },
            "workload_kind": self.workload_kind,
            "unsupported_features": list(self.unsupported_features),
        }


@dataclass(frozen=True)
class PaperBenchmarkWorkload:
    """A concrete module/input tuple ready for ``compile_torch_module``."""

    spec: PaperBenchmarkSpec
    module: torch.nn.Module
    inputs: tuple[torch.Tensor, ...]
    variant: str
    attributes: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "variant": self.variant,
            "input_shapes": [list(value.shape) for value in self.inputs],
            "input_dtypes": [str(value.dtype).removeprefix("torch.") for value in self.inputs],
            "module": type(self.module).__name__,
            "attributes": dict(self.attributes),
        }
