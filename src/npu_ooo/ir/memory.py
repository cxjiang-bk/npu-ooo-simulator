"""Compiler-owned memory-copy and physical-allocation plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from npu_ooo.arch import MachineConfig


MEMORY_PLAN_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MemoryBuffer:
    buffer_id: str
    allocation_id: str
    tensor: str
    memory: str
    offset_bytes: int
    allocation_bytes: int
    valid_bytes: int
    alignment_bytes: int
    layout: str
    strides_bytes: tuple[int, ...] | None = None
    tile_id: str | None = None
    role: str | None = None
    slot: int | None = None
    external: bool = False
    lifetime_start: int = 0
    lifetime_end: int = 0
    alias_of: str | None = None
    dtype: str = "fp16"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.buffer_id or not self.allocation_id or not self.tensor or not self.memory or not self.dtype:
            issues.append("memory buffer identities, tensor and memory must not be empty")
        if self.offset_bytes < 0:
            issues.append(f"memory buffer '{self.buffer_id}' offset must be non-negative")
        if self.allocation_bytes <= 0 or self.valid_bytes <= 0:
            issues.append(f"memory buffer '{self.buffer_id}' byte sizes must be positive")
        if self.valid_bytes > self.allocation_bytes:
            issues.append(
                f"memory buffer '{self.buffer_id}' valid bytes exceed allocation span"
            )
        if self.alignment_bytes <= 0 or self.offset_bytes % self.alignment_bytes:
            issues.append(f"memory buffer '{self.buffer_id}' offset violates alignment")
        if not self.layout:
            issues.append(f"memory buffer '{self.buffer_id}' layout must not be empty")
        if self.slot is not None and self.slot < 0:
            issues.append(f"memory buffer '{self.buffer_id}' slot must be non-negative")
        if self.lifetime_start < 0 or self.lifetime_end < self.lifetime_start:
            issues.append(f"memory buffer '{self.buffer_id}' lifetime is invalid")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer_id": self.buffer_id,
            "allocation_id": self.allocation_id,
            "tensor": self.tensor,
            "memory": self.memory,
            "offset_bytes": self.offset_bytes,
            "allocation_bytes": self.allocation_bytes,
            "valid_bytes": self.valid_bytes,
            "alignment_bytes": self.alignment_bytes,
            "layout": self.layout,
            "strides_bytes": list(self.strides_bytes) if self.strides_bytes is not None else None,
            "tile_id": self.tile_id,
            "role": self.role,
            "slot": self.slot,
            "external": self.external,
            "lifetime_start": self.lifetime_start,
            "lifetime_end": self.lifetime_end,
            "alias_of": self.alias_of,
            "dtype": self.dtype,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryBuffer":
        if not isinstance(payload, Mapping):
            raise ValueError("memory buffer must be a JSON object")
        try:
            result = cls(
                buffer_id=str(payload["buffer_id"]),
                allocation_id=str(payload["allocation_id"]),
                tensor=str(payload["tensor"]),
                memory=str(payload["memory"]),
                offset_bytes=int(payload["offset_bytes"]),
                allocation_bytes=int(payload["allocation_bytes"]),
                valid_bytes=int(payload["valid_bytes"]),
                alignment_bytes=int(payload["alignment_bytes"]),
                layout=str(payload["layout"]),
                strides_bytes=(
                    tuple(int(item) for item in payload["strides_bytes"])
                    if payload.get("strides_bytes") is not None
                    else None
                ),
                tile_id=(str(payload["tile_id"]) if payload.get("tile_id") is not None else None),
                role=(str(payload["role"]) if payload.get("role") is not None else None),
                slot=(int(payload["slot"]) if payload.get("slot") is not None else None),
                external=bool(payload.get("external", False)),
                lifetime_start=int(payload.get("lifetime_start", 0)),
                lifetime_end=int(payload.get("lifetime_end", 0)),
                alias_of=(str(payload["alias_of"]) if payload.get("alias_of") is not None else None),
                dtype=str(payload.get("dtype", "fp16")),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid memory buffer") from exc
        issues = result.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return result


@dataclass(frozen=True)
class MemoryPlan:
    plan_id: str
    machine_config_id: str
    machine_topology_hash: str
    buffers: tuple[MemoryBuffer, ...]
    schema_version: int = MEMORY_PLAN_SCHEMA_VERSION
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, machine: "MachineConfig | None" = None) -> tuple[str, ...]:
        issues: list[str] = []
        if self.schema_version != MEMORY_PLAN_SCHEMA_VERSION:
            issues.append(
                f"memory plan schema {self.schema_version} is unsupported; expected "
                f"{MEMORY_PLAN_SCHEMA_VERSION}"
            )
        if not self.plan_id or not self.machine_config_id or not self.machine_topology_hash:
            issues.append("memory plan identities must not be empty")
        ids = [item.buffer_id for item in self.buffers]
        if len(set(ids)) != len(ids):
            issues.append("memory plan buffer ids must be unique")
        for item in self.buffers:
            issues.extend(item.validate())
        allocations: dict[str, tuple[str, int, int]] = {}
        for item in self.buffers:
            value = (item.memory, item.offset_bytes, item.allocation_bytes)
            previous = allocations.setdefault(item.allocation_id, value)
            if previous != value:
                issues.append(
                    f"allocation '{item.allocation_id}' has inconsistent memory/range"
                )
        by_memory: dict[str, list[tuple[str, int, int]]] = {}
        for allocation_id, (memory, offset, size) in allocations.items():
            by_memory.setdefault(memory, []).append(
                (allocation_id, offset, offset + size)
            )
        for memory, ranges in by_memory.items():
            ordered = sorted(ranges, key=lambda item: (item[1], item[2], item[0]))
            for left, right in zip(ordered, ordered[1:]):
                if left[2] > right[1]:
                    issues.append(
                        f"allocations '{left[0]}' and '{right[0]}' overlap in '{memory}'"
                    )
        by_id = {item.buffer_id: item for item in self.buffers}
        for item in self.buffers:
            if item.alias_of is None:
                continue
            target = by_id.get(item.alias_of)
            if target is None:
                issues.append(
                    f"memory buffer '{item.buffer_id}' aliases unknown buffer '{item.alias_of}'"
                )
            elif target.allocation_id != item.allocation_id:
                issues.append(
                    f"memory buffer '{item.buffer_id}' alias does not share allocation "
                    f"with '{item.alias_of}'"
                )
        if machine is not None:
            memories = {item.name: item for item in machine.memory_levels}
            for item in self.buffers:
                if item.memory not in memories:
                    issues.append(
                        f"memory buffer '{item.buffer_id}' references unknown memory '{item.memory}'"
                    )
            for memory, level in memories.items():
                if level.capacity_bytes is None:
                    continue
                end = max(
                    (
                        item.offset_bytes + item.allocation_bytes
                        for item in self.buffers
                        if item.memory == memory
                    ),
                    default=0,
                )
                if end > level.capacity_bytes:
                    issues.append(
                        f"memory plan requires {end} bytes in '{memory}', capacity is "
                        f"{level.capacity_bytes}"
                    )
        return tuple(issues)

    def buffer(self, buffer_id: str) -> MemoryBuffer:
        for item in self.buffers:
            if item.buffer_id == buffer_id:
                return item
        raise KeyError(buffer_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "machine_config_id": self.machine_config_id,
            "machine_topology_hash": self.machine_topology_hash,
            "buffers": [item.to_dict() for item in self.buffers],
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("memory plan must be a JSON object")
        try:
            result = cls(
                plan_id=str(payload["plan_id"]),
                machine_config_id=str(payload["machine_config_id"]),
                machine_topology_hash=str(payload["machine_topology_hash"]),
                buffers=tuple(
                    MemoryBuffer.from_dict(item) for item in payload.get("buffers", ())
                ),
                schema_version=int(payload.get("schema_version", 0)),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid memory plan") from exc
        issues = result.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return result
