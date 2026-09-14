"""Replayable per-task GM latency for simulation-only sensitivity studies."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import BackendArtifact, ExecutionTask
from npu_ooo.simulator.core import (
    AnalyticalTimingModel,
    TaskTimingSpec,
    TimingModel,
)

from .contracts import BackendCapabilities
from .registry import analytical_capabilities


TRACE_FORMAT = "npu_ooo.gm_latency_trace.v1"


def _non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _probability(value: Any, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError(f"{field_name} must be between 0 and 1")
    return float(value)


def _gm_bytes(task: ExecutionTask, memory: str) -> tuple[int, int, int]:
    read_bytes = sum(
        int(region.valid_bytes or region.size_bytes)
        for region in task.reads
        if region.memory == memory
    )
    write_bytes = sum(
        int(region.valid_bytes or region.size_bytes)
        for region in task.writes
        if region.memory == memory
    )
    return read_bytes, write_bytes, read_bytes + write_bytes


def _touches_memory(task: ExecutionTask, memory: str) -> bool:
    return any(
        region.memory == memory
        for region in (*task.reads, *task.writes)
    )


@dataclass(frozen=True)
class GMLatencyTraceTimingProvider:
    """Add fixed, replayable latency samples to tasks that touch one memory."""

    entries: Mapping[str, int]
    memory: str = "GM"
    name: str = "gm_latency_trace"
    seed: int | None = None
    distribution: Mapping[str, Any] = field(default_factory=dict)
    trace_path: str | None = None
    fallback: TimingModel = field(default_factory=AnalyticalTimingModel)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "GMLatencyTraceTimingProvider":
        if not isinstance(payload, Mapping):
            raise ValueError("GM latency trace payload must be an object")
        if payload.get("format") != TRACE_FORMAT:
            raise ValueError(f"GM latency trace format must be '{TRACE_FORMAT}'")
        memory = payload.get("memory", "GM")
        if not isinstance(memory, str) or not memory:
            raise ValueError("GM latency trace memory must be a non-empty string")
        name = payload.get("name", "gm_latency_trace")
        if not isinstance(name, str) or not name:
            raise ValueError("GM latency trace name must be a non-empty string")
        raw_entries = payload.get("requests", {})
        if not isinstance(raw_entries, Mapping):
            raise ValueError("GM latency trace requests must be an object")
        entries: dict[str, int] = {}
        for task_id, raw_entry in raw_entries.items():
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("GM latency trace request ids must be non-empty strings")
            if not isinstance(raw_entry, Mapping):
                raise ValueError(
                    f"GM latency trace request '{task_id}' must be an object"
                )
            extra = _non_negative_integer(
                raw_entry.get("extra_latency_cycles"),
                field_name=f"GM latency trace request '{task_id}'.extra_latency_cycles",
            )
            entries[task_id] = extra
        raw_seed = payload.get("seed")
        seed = (
            None
            if raw_seed is None
            else _integer(raw_seed, field_name="GM latency trace seed")
        )
        distribution = payload.get("distribution", {})
        if not isinstance(distribution, Mapping):
            raise ValueError("GM latency trace distribution must be an object")
        if "extra_latency_probability" in distribution:
            _probability(
                distribution["extra_latency_probability"],
                field_name="GM latency trace extra_latency_probability",
            )
        return cls(
            entries=entries,
            memory=memory,
            name=name,
            seed=seed,
            distribution=dict(distribution),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "GMLatencyTraceTimingProvider":
        trace_path = Path(path)
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load GM latency trace '{trace_path}': {exc}") from exc
        provider = cls.from_dict(payload)
        return cls(
            entries=provider.entries,
            memory=provider.memory,
            name=provider.name,
            seed=provider.seed,
            distribution=provider.distribution,
            trace_path=str(trace_path),
            fallback=provider.fallback,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            supported_primitives=analytical_capabilities().supported_primitives,
            calibration_status="synthetic-random",
            attributes={
                "format": TRACE_FORMAT,
                "memory": self.memory,
                "latency_semantics": "extra_latency_added_to_analytical_task_duration",
                "trace_path": self.trace_path,
            },
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "format": TRACE_FORMAT,
            "memory": self.memory,
            "request_count": len(self.entries),
            "nonideal_request_count": sum(
                extra > 0 for extra in self.entries.values()
            ),
            "seed": self.seed,
            "distribution": dict(self.distribution),
            "trace_path": self.trace_path,
        }

    def timing(self, task: ExecutionTask, machine: MachineConfig) -> TaskTimingSpec:
        base = self.fallback.timing(task, machine)
        if not _touches_memory(task, self.memory):
            return base
        try:
            extra = self.entries[task.task_id]
        except KeyError as exc:
            raise ValueError(
                f"GM latency trace '{self.name}' is missing request '{task.task_id}'"
            ) from exc
        return TaskTimingSpec(
            duration_cycles=base.duration_cycles + extra,
            initiation_interval_cycles=base.initiation_interval_cycles,
        )

    def coverage(self, tasks: Iterable[ExecutionTask]) -> Mapping[str, int]:
        gm_requests = 0
        traced = 0
        nonideal = 0
        missing = 0
        extra_total = 0
        gm_read_bytes = 0
        gm_write_bytes = 0
        for task in tasks:
            if not _touches_memory(task, self.memory):
                continue
            gm_requests += 1
            read_bytes, write_bytes, _total = _gm_bytes(task, self.memory)
            gm_read_bytes += read_bytes
            gm_write_bytes += write_bytes
            if task.task_id in self.entries:
                traced += 1
                extra_total += self.entries[task.task_id]
                nonideal += self.entries[task.task_id] > 0
            else:
                missing += 1
        return {
            "gm_request_count": gm_requests,
            "gm_trace_hit_count": traced,
            "gm_nonideal_hit_count": nonideal,
            "gm_trace_missing_count": missing,
            "gm_extra_latency_cycles": extra_total,
            "gm_read_bytes": gm_read_bytes,
            "gm_write_bytes": gm_write_bytes,
        }


def build_gm_latency_trace(
    artifact: BackendArtifact,
    machine: MachineConfig,
    *,
    seed: int,
    min_extra_latency_cycles: int,
    max_extra_latency_cycles: int,
    extra_latency_probability: float = 0.1,
    memory: str = "GM",
    name: str | None = None,
) -> dict[str, Any]:
    """Generate a centralized, replayable trace for GM-facing tasks."""

    seed = _integer(seed, field_name="seed")
    min_extra_latency_cycles = _non_negative_integer(
        min_extra_latency_cycles,
        field_name="min_extra_latency_cycles",
    )
    max_extra_latency_cycles = _non_negative_integer(
        max_extra_latency_cycles,
        field_name="max_extra_latency_cycles",
    )
    extra_latency_probability = _probability(
        extra_latency_probability,
        field_name="extra_latency_probability",
    )
    if min_extra_latency_cycles > max_extra_latency_cycles:
        raise ValueError(
            "min_extra_latency_cycles must not exceed max_extra_latency_cycles"
        )
    if not isinstance(memory, str) or not memory:
        raise ValueError("memory must be a non-empty string")
    rng = random.Random(seed)
    requests: dict[str, dict[str, Any]] = {}
    nonideal_request_count = 0
    analytical = AnalyticalTimingModel()
    for task in artifact.execution_graph.tasks:
        if not _touches_memory(task, memory):
            continue
        read_bytes, write_bytes, total_bytes = _gm_bytes(task, memory)
        if rng.random() < extra_latency_probability:
            extra = rng.randint(min_extra_latency_cycles, max_extra_latency_cycles)
            if extra > 0:
                nonideal_request_count += 1
        else:
            extra = 0
        requests[task.task_id] = {
            "extra_latency_cycles": extra,
            "base_duration_cycles": analytical.timing(task, machine).duration_cycles,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "total_bytes": total_bytes,
        }
    topology_hash = (
        artifact.memory_plan.machine_topology_hash
        if artifact.memory_plan is not None
        else machine.topology_hash()
    )
    return {
        "format": TRACE_FORMAT,
        "name": name or f"gm_latency_seed_{seed}",
        "memory": memory,
        "seed": seed,
        "distribution": {
            "type": "uniform_integer",
            "min_extra_latency_cycles": min_extra_latency_cycles,
            "max_extra_latency_cycles": max_extra_latency_cycles,
            "extra_latency_probability": extra_latency_probability,
        },
        "artifact_id": artifact.artifact_id,
        "program_id": artifact.program.program_id,
        "machine_topology_hash": topology_hash,
        "request_count": len(requests),
        "nonideal_request_count": nonideal_request_count,
        "requests": requests,
    }


__all__ = [
    "GMLatencyTraceTimingProvider",
    "TRACE_FORMAT",
    "build_gm_latency_trace",
]
