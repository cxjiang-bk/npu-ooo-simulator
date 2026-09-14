"""Event-driven conservative buffer lifetime and occupancy analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from npu_ooo.ir import BackendArtifact, LoadedDeviceProgram
from npu_ooo.simulator.core import SimulationResult


@dataclass(frozen=True)
class BufferLifecycleReport:
    workload_hash: str | None
    total_cycles: float
    memories: Mapping[str, Any]
    versions: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    assumptions: Mapping[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload_hash": self.workload_hash,
            "total_cycles": self.total_cycles,
            "memories": dict(self.memories),
            "versions": [dict(item) for item in self.versions],
            "events": [dict(item) for item in self.events],
            "assumptions": dict(self.assumptions),
        }


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _curve(
    intervals_by_allocation: Mapping[str, tuple[int, list[tuple[float, float]]]],
) -> tuple[list[dict[str, float]], int]:
    changes: dict[float, int] = {}
    for size, intervals in intervals_by_allocation.values():
        for start, end in _merge(intervals):
            changes[start] = changes.get(start, 0) + size
            changes[end] = changes.get(end, 0) - size
    current = peak = 0
    result: list[dict[str, float]] = []
    for cycle in sorted(changes):
        current += changes[cycle]
        peak = max(peak, current)
        result.append({"cycle": cycle, "bytes": current})
    return result, peak


def build_buffer_lifecycle(
    artifact: BackendArtifact,
    loaded: LoadedDeviceProgram,
    result: SimulationResult,
    machine,
) -> BufferLifecycleReport:
    """Map actual instruction timing to allocation-level conservative lifetimes.

    The analysis is event-driven at TISA/allocation granularity. It does not
    model byte-by-byte fill, dynamic allocation or exact DRAM transactions.
    """

    if artifact.memory_plan is None:
        raise ValueError("buffer lifecycle analysis requires MemoryPlan")
    timings = {item.task_id: item for item in result.instruction_timings}
    physical_done = {
        event.task_id: event.timestamp
        for event in result.events
        if event.event == "TISA_EXECUTION_DONE"
    }
    plan_by_buffer = {item.buffer_id: item for item in artifact.memory_plan.buffers}
    allocations: dict[str, dict[str, Any]] = {}
    for buffer in artifact.memory_plan.buffers:
        current = allocations.setdefault(
            buffer.allocation_id,
            {
                "allocation_id": buffer.allocation_id,
                "memory": buffer.memory,
                "allocation_bytes": buffer.allocation_bytes,
                "valid_bytes": 0,
                "buffers": set(),
                "persistent": False,
                "accesses": [],
            },
        )
        current["allocation_bytes"] = max(
            current["allocation_bytes"], buffer.allocation_bytes
        )
        current["valid_bytes"] = max(current["valid_bytes"], buffer.valid_bytes)
        current["buffers"].add(buffer.buffer_id)
        current["persistent"] = current["persistent"] or bool(
            buffer.attributes.get("persistent", False)
        )
    for descriptor in loaded.descriptors:
        tisa_id = descriptor.instruction.tisa_id
        if tisa_id not in timings:
            continue
        instruction_operands = {
            item.name: item for item in descriptor.instruction.operands
        }
        for operand in descriptor.operands:
            logical = instruction_operands.get(operand.operand_name)
            if logical is None or logical.tile_mem.buffer_id not in plan_by_buffer:
                continue
            planned = plan_by_buffer[logical.tile_mem.buffer_id]
            allocations[planned.allocation_id]["accesses"].append(
                {
                    "tisa_id": tisa_id,
                    "buffer_id": planned.buffer_id,
                    "access": operand.access_type,
                    "issue": timings[tisa_id].issue,
                    "complete": timings[tisa_id].finish,
                    "physical_done": physical_done.get(
                        tisa_id, timings[tisa_id].finish
                    ),
                    "address": operand.address,
                    "size_bytes": operand.size_bytes,
                }
            )

    versions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    protected_by_memory: dict[str, dict[str, tuple[int, list[tuple[float, float]]]]] = {}
    retained_by_memory: dict[str, dict[str, tuple[int, list[tuple[float, float]]]]] = {}
    for allocation_id, allocation in allocations.items():
        accesses = sorted(
            allocation["accesses"], key=lambda item: (item["issue"], item["tisa_id"])
        )
        writes = [
            item for item in accesses if item["access"] in {"write", "read_write"}
        ]
        memory = allocation["memory"]
        protected_intervals: list[tuple[float, float]] = []
        retained_intervals: list[tuple[float, float]] = []
        if writes:
            for generation, write in enumerate(writes):
                next_write = (
                    writes[generation + 1]["issue"]
                    if generation + 1 < len(writes)
                    else result.total_cycles
                )
                readers = [
                    item
                    for item in accesses
                    if item["access"] in {"read", "read_write"}
                    and item["issue"] >= write["issue"]
                    and item["issue"] < next_write
                    and item["tisa_id"] != write["tisa_id"]
                ]
                release = max(
                    [write["physical_done"], *(item["complete"] for item in readers)]
                )
                if allocation["persistent"]:
                    release = result.total_cycles
                version_id = f"{allocation_id}:g{generation}"
                version = {
                    "version_id": version_id,
                    "allocation_id": allocation_id,
                    "memory": memory,
                    "buffer_id": write["buffer_id"],
                    "producer": write["tisa_id"],
                    "consumers": sorted({item["tisa_id"] for item in readers}),
                    "protect_start": write["issue"],
                    "valid_cycle": write["physical_done"],
                    "release_cycle": release,
                    "allocation_bytes": allocation["allocation_bytes"],
                    "valid_bytes": allocation["valid_bytes"],
                    "persistent": allocation["persistent"],
                }
                versions.append(version)
                protected_intervals.append((write["issue"], release))
                retained_intervals.append((write["physical_done"], release))
                for event_name, cycle in (
                    ("BUFFER_PROTECT", write["issue"]),
                    ("BUFFER_VALID", write["physical_done"]),
                    ("BUFFER_RELEASE", release),
                ):
                    events.append(
                        {
                            "cycle": cycle,
                            "event": event_name,
                            "version_id": version_id,
                            "allocation_id": allocation_id,
                            "buffer_id": write["buffer_id"],
                            "memory": memory,
                            "producer": write["tisa_id"],
                            "consumers": version["consumers"],
                        }
                    )
        elif accesses:
            readers = [
                item for item in accesses if item["access"] in {"read", "read_write"}
            ]
            if readers:
                release = (
                    result.total_cycles
                    if allocation["persistent"]
                    else max(item["complete"] for item in readers)
                )
                # External/read-only data is already valid on invocation
                # entry and cannot be overwritten before its first reader.
                protected_intervals.append((0.0, release))
                retained_intervals.append((0.0, release))
        if allocation["persistent"]:
            # Persistent state is already valid on invocation entry and cannot
            # be reclaimed at an inter-invocation gap.
            protected_intervals.append((0.0, result.total_cycles))
            retained_intervals.append((0.0, result.total_cycles))
        protected_by_memory.setdefault(memory, {})[allocation_id] = (
            allocation["allocation_bytes"],
            protected_intervals,
        )
        retained_by_memory.setdefault(memory, {})[allocation_id] = (
            allocation["allocation_bytes"],
            retained_intervals,
        )

    capacity = {item.name: item.capacity_bytes for item in machine.memory_levels}
    memories: dict[str, Any] = {}
    for memory in sorted({item["memory"] for item in allocations.values()}):
        allocated = sum(
            item["allocation_bytes"]
            for item in allocations.values()
            if item["memory"] == memory
        )
        valid = sum(
            item["valid_bytes"]
            for item in allocations.values()
            if item["memory"] == memory
        )
        protected_curve, protected_peak = _curve(
            protected_by_memory.get(memory, {})
        )
        retained_curve, retained_peak = _curve(retained_by_memory.get(memory, {}))
        memories[memory] = {
            "allocated_bytes": allocated,
            "valid_bytes": valid,
            "padding_bytes": max(0, allocated - valid),
            "capacity_bytes": capacity.get(memory),
            "protected_peak_bytes": protected_peak,
            "retained_peak_bytes": retained_peak,
            "protected_occupancy": protected_curve,
            "retained_occupancy": retained_curve,
            "allocation_count": sum(
                item["memory"] == memory for item in allocations.values()
            ),
        }
    return BufferLifecycleReport(
        workload_hash=artifact.attributes.get("shared_workload_hash"),
        total_cycles=result.total_cycles,
        memories=memories,
        versions=tuple(sorted(versions, key=lambda item: (item["protect_start"], item["version_id"]))),
        events=tuple(sorted(events, key=lambda item: (item["cycle"], item["event"]))),
        assumptions={
            "allocation": "MemoryPlan fixed; no dynamic malloc/free",
            "default_quantity": "temporarily_not_reusable",
            "valid_quantity": "produced_and_needed",
            "granularity": "conservative allocation/TISA",
            "physical_done": "BUFFER_VALID",
            "reuse_ready": "last observed consumer completion",
            "alias_counting": "unique allocation_id",
            "partial_regions": "whole allocation conservative",
        },
    )


def combine_buffer_lifecycles(
    reports: tuple[tuple[str, float, BufferLifecycleReport], ...],
    *,
    total_cycles: float,
) -> BufferLifecycleReport:
    """Place per-invocation reports on one sequential absolute timeline."""

    if not reports:
        raise ValueError("buffer lifecycle sequence requires at least one invocation")
    memories: dict[str, dict[str, Any]] = {}
    versions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for invocation_id, offset, report in reports:
        for memory, item in report.memories.items():
            current = memories.setdefault(
                memory,
                {
                    key: item[key]
                    for key in (
                        "allocated_bytes",
                        "valid_bytes",
                        "padding_bytes",
                        "capacity_bytes",
                        "allocation_count",
                    )
                },
            )
            current["protected_peak_bytes"] = max(
                current.get("protected_peak_bytes", 0), item["protected_peak_bytes"]
            )
            current["retained_peak_bytes"] = max(
                current.get("retained_peak_bytes", 0), item["retained_peak_bytes"]
            )
            current.setdefault("protected_occupancy", []).extend(
                {"cycle": row["cycle"] + offset, "bytes": row["bytes"]}
                for row in item["protected_occupancy"]
            )
            current.setdefault("retained_occupancy", []).extend(
                {"cycle": row["cycle"] + offset, "bytes": row["bytes"]}
                for row in item["retained_occupancy"]
            )
        for item in report.versions:
            release_cycle = (
                total_cycles
                if item.get("persistent")
                else item["release_cycle"] + offset
            )
            versions.append(
                {
                    **dict(item),
                    "version_id": f"{invocation_id}/{item['version_id']}",
                    "invocation_id": invocation_id,
                    "protect_start": item["protect_start"] + offset,
                    "valid_cycle": item["valid_cycle"] + offset,
                    "release_cycle": release_cycle,
                }
            )
        for item in report.events:
            source_version = next(
                (
                    version
                    for version in report.versions
                    if version["version_id"] == item["version_id"]
                ),
                None,
            )
            event_cycle = (
                total_cycles
                if item["event"] == "BUFFER_RELEASE"
                and source_version is not None
                and source_version.get("persistent")
                else item["cycle"] + offset
            )
            events.append(
                {
                    **dict(item),
                    "version_id": f"{invocation_id}/{item['version_id']}",
                    "invocation_id": invocation_id,
                    "cycle": event_cycle,
                }
            )
    return BufferLifecycleReport(
        workload_hash=reports[0][2].workload_hash,
        total_cycles=total_cycles,
        memories=memories,
        versions=tuple(sorted(versions, key=lambda item: item["protect_start"])),
        events=tuple(sorted(events, key=lambda item: (item["cycle"], item["event"]))),
        assumptions={
            **dict(reports[0][2].assumptions),
            "sequence": "per-invocation lifetimes shifted onto sequential runtime timeline",
            "persistent_state": "persistent allocations remain retained through invocation boundary",
        },
    )


__all__ = [
    "BufferLifecycleReport",
    "build_buffer_lifecycle",
    "combine_buffer_lifecycles",
]
