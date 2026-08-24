"""External systolic-MXU timing profile adapter.

The adapter deliberately consumes a small normalized profile instead of
calling SCALE-Sim at scheduling time.  A SCALE-Sim, RTL, or hardware-counter
exporter can produce this profile offline; the common TISA scheduler then
replays those MXU timings alongside the existing DMA/vector model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import ExecutionTask
from npu_ooo.simulator.core import AnalyticalTimingModel, TaskTimingSpec, TimingModel

from .contracts import BackendCapabilities


_PROFILE_FORMAT = "npu_ooo.systolic_mxu_profile.v1"


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_float(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(value)


@dataclass(frozen=True)
class SystolicMXUProfileEntry:
    """One calibrated execution point for a GEMM tile shape."""

    batch: int
    m: int
    n: int
    k: int
    timing: TaskTimingSpec
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.batch, self.m, self.n, self.k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": {"batch": self.batch, "m": self.m, "n": self.n, "k": self.k},
            "duration_cycles": self.timing.duration_cycles,
            "initiation_interval_cycles": self.timing.initiation_interval_cycles,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class SystolicMXUProfileTimingProvider:
    """Use exact externally calibrated MXU timings and explicit fallbacks.

    Non-matmul primitives always delegate to ``fallback`` because this is an
    MXU-only adapter.  For a matmul tile absent from the profile,
    ``unmatched_matmul`` selects either the same analytical fallback or a
    deterministic error.  The selected policy is exported in metadata so a
    mixed calibrated/analytical result cannot be mistaken for a fully
    calibrated simulation.
    """

    entries: tuple[SystolicMXUProfileEntry, ...]
    name: str = "systolic_mxu_profile"
    calibration_status: str = "source-derived"
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    unmatched_matmul: str = "analytical"
    profile_path: str | None = None
    fallback: TimingModel = field(default_factory=AnalyticalTimingModel)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SystolicMXUProfileTimingProvider":
        if not isinstance(payload, Mapping):
            raise ValueError("systolic MXU profile payload must be an object")
        if payload.get("format") != _PROFILE_FORMAT:
            raise ValueError(
                f"systolic MXU profile format must be '{_PROFILE_FORMAT}'"
            )
        name = payload.get("name", "systolic_mxu_profile")
        if not isinstance(name, str) or not name:
            raise ValueError("systolic MXU profile name must be a non-empty string")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("systolic MXU profile metadata must be an object")
        calibration_status = metadata.get("calibration_status", "source-derived")
        if not isinstance(calibration_status, str) or not calibration_status:
            raise ValueError("systolic MXU profile calibration_status must be a non-empty string")
        unmatched_matmul = payload.get("unmatched_matmul", "analytical")
        if unmatched_matmul not in {"analytical", "error"}:
            raise ValueError("systolic MXU profile unmatched_matmul must be analytical or error")
        raw_entries = payload.get("matmul_profiles")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("systolic MXU profile matmul_profiles must be a non-empty list")

        entries: list[SystolicMXUProfileEntry] = []
        keys: set[tuple[int, int, int, int]] = set()
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"matmul_profiles[{index}] must be an object")
            shape = raw_entry.get("shape")
            if not isinstance(shape, Mapping):
                raise ValueError(f"matmul_profiles[{index}].shape must be an object")
            attributes = raw_entry.get("attributes", {})
            if not isinstance(attributes, Mapping):
                raise ValueError(f"matmul_profiles[{index}].attributes must be an object")
            entry = SystolicMXUProfileEntry(
                batch=_positive_int(shape.get("batch", 1), field_name=f"matmul_profiles[{index}].shape.batch"),
                m=_positive_int(shape.get("m"), field_name=f"matmul_profiles[{index}].shape.m"),
                n=_positive_int(shape.get("n"), field_name=f"matmul_profiles[{index}].shape.n"),
                k=_positive_int(shape.get("k"), field_name=f"matmul_profiles[{index}].shape.k"),
                timing=TaskTimingSpec(
                    duration_cycles=_positive_float(
                        raw_entry.get("duration_cycles"),
                        field_name=f"matmul_profiles[{index}].duration_cycles",
                    ),
                    initiation_interval_cycles=_positive_float(
                        raw_entry.get("initiation_interval_cycles"),
                        field_name=f"matmul_profiles[{index}].initiation_interval_cycles",
                    ),
                ),
                attributes=dict(attributes),
            )
            if entry.key in keys:
                raise ValueError(
                    "systolic MXU profile contains duplicate shape "
                    f"batch={entry.batch}, m={entry.m}, n={entry.n}, k={entry.k}"
                )
            keys.add(entry.key)
            entries.append(entry)
        return cls(
            entries=tuple(entries),
            name=name,
            calibration_status=calibration_status,
            source_metadata=dict(metadata),
            unmatched_matmul=unmatched_matmul,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "SystolicMXUProfileTimingProvider":
        profile_path = Path(path)
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load systolic MXU profile '{profile_path}': {exc}") from exc
        provider = cls.from_dict(payload)
        return cls(
            entries=provider.entries,
            name=provider.name,
            calibration_status=provider.calibration_status,
            source_metadata=provider.source_metadata,
            unmatched_matmul=provider.unmatched_matmul,
            profile_path=str(profile_path),
            fallback=provider.fallback,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            calibration_status=(
                f"mixed:{self.calibration_status}+"
                f"{getattr(self.fallback, 'name', 'analytical')}-fallback"
            ),
            attributes={
                "format": _PROFILE_FORMAT,
                "calibrated_primitives": ["matmul"],
                "profile_calibration_status": self.calibration_status,
                "unmatched_matmul": self.unmatched_matmul,
                "non_matmul_fallback": getattr(self.fallback, "name", "analytical"),
            },
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "format": _PROFILE_FORMAT,
            "profile_count": len(self.entries),
            "unmatched_matmul": self.unmatched_matmul,
            "non_matmul_fallback": getattr(self.fallback, "name", "analytical"),
            "profile_path": self.profile_path,
            "profile_calibration_status": self.calibration_status,
            **dict(self.source_metadata),
        }

    def timing(self, task: ExecutionTask, machine: MachineConfig) -> TaskTimingSpec:
        if task.primitive != "matmul":
            return self.fallback.timing(task, machine)
        shape = self._task_shape(task)
        if shape is not None:
            for entry in self.entries:
                if entry.key == shape:
                    return entry.timing
        if self.unmatched_matmul == "analytical":
            return self.fallback.timing(task, machine)
        shape_text = "unknown" if shape is None else ", ".join(
            f"{name}={value}" for name, value in zip(("batch", "m", "n", "k"), shape)
        )
        raise ValueError(
            f"systolic MXU profile '{self.name}' has no calibrated matmul tile ({shape_text}); "
            "set unmatched_matmul to 'analytical' to allow an explicit fallback"
        )

    @staticmethod
    def _task_shape(task: ExecutionTask) -> tuple[int, int, int, int] | None:
        m = task.attributes.get("m_tile")
        n = task.attributes.get("n_tile")
        k = task.attributes.get("k_tile")
        batch_tile = task.attributes.get("batch_tile", ())
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (m, n, k)):
            return None
        if not isinstance(batch_tile, (tuple, list)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in batch_tile
        ):
            return None
        return (math.prod(batch_tile) if batch_tile else 1, m, n, k)


__all__ = [
    "SystolicMXUProfileEntry",
    "SystolicMXUProfileTimingProvider",
]
