"""Public scheduler helpers over bound device descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.execution import ExecutionBackend
from npu_ooo.ir import AccessType, BoundTISADescriptor, RuntimeOperandBinding


@dataclass(frozen=True)
class MemoryAccess:
    memory: str
    bank: int
    mode: str


@dataclass(frozen=True)
class SemanticConflict:
    """One conservative Algorithm-2 conflict against a local in-flight entry."""

    predecessor: str
    successor: str
    kind: str
    memory: str
    address: int
    size_bytes: int
    predecessor_operand: str
    successor_operand: str
    predecessor_op_type: str
    successor_op_type: str
    predecessor_logical_scope: str
    successor_logical_scope: str
    declared_dependency: bool
    rule: str = "tilemem_overlap_without_safe_semantic_override"

    def to_dict(self) -> dict[str, Any]:
        return {
            "predecessor": self.predecessor,
            "successor": self.successor,
            "kind": self.kind,
            "memory": self.memory,
            "address": self.address,
            "size_bytes": self.size_bytes,
            "predecessor_operand": self.predecessor_operand,
            "successor_operand": self.successor_operand,
            "predecessor_op_type": self.predecessor_op_type,
            "successor_op_type": self.successor_op_type,
            "predecessor_logical_scope": self.predecessor_logical_scope,
            "successor_logical_scope": self.successor_logical_scope,
            "declared_dependency": self.declared_dependency,
            "rule": self.rule,
        }


def access_banks(
    address: int,
    size: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    *,
    bank_width: int,
    bank_count: int,
) -> tuple[int, ...]:
    def span_banks(start: int, byte_count: int) -> set[int]:
        first = start // bank_width
        last = (start + max(byte_count - 1, 0)) // bank_width
        return {
            (first + index) % bank_count
            for index in range(min(last - first + 1, bank_count))
        }

    if (
        not shape
        or len(shape) != len(strides)
        or any(value <= 0 for value in shape)
        or any(value < 0 for value in strides)
    ):
        return tuple(sorted(span_banks(address, size)))
    element_count = math.prod(shape)
    if element_count > 4096:
        return tuple(sorted(span_banks(address, size)))
    element_bytes = size - sum(
        (extent - 1) * stride for extent, stride in zip(shape, strides)
    )
    if element_bytes <= 0:
        return tuple(sorted(span_banks(address, size)))
    banks: set[int] = set()
    for indices in product(*(range(extent) for extent in shape)):
        start = address + sum(
            index * stride for index, stride in zip(indices, strides)
        )
        banks.update(span_banks(start, element_bytes))
    return tuple(sorted(banks))


def memory_accesses(
    descriptor: BoundTISADescriptor,
    machine: MachineConfig,
) -> tuple[MemoryAccess, ...]:
    semantic_operands = {
        operand.name: operand for operand in descriptor.instruction.operands
    }
    accesses: set[MemoryAccess] = set()
    for operand in descriptor.operands:
        semantic = semantic_operands[operand.operand_name]
        dynamic_region = operand.attributes.get("dynamic_region")
        shape = tuple(
            dynamic_region.get("shape", ())
            if isinstance(dynamic_region, Mapping)
            else semantic.tile_shape
        )
        strides = tuple(
            operand.attributes.get("runtime_strides_bytes")
            or semantic.tile_mem.strides_bytes
            or ()
        )
        try:
            level = machine.memory(operand.physical_scope)
        except KeyError:
            continue
        banks = access_banks(
            operand.address,
            operand.size_bytes,
            shape,
            strides,
            bank_width=level.bank_width_bytes or 1,
            bank_count=level.bank_count,
        )
        modes = (
            ("read", "write")
            if operand.access_type == AccessType.READ_WRITE.value
            else ("read",)
            if operand.access_type == AccessType.READ.value
            else ("write",)
        )
        for bank in banks:
            for mode in modes:
                accesses.add(MemoryAccess(operand.physical_scope, bank, mode))
    return tuple(sorted(accesses, key=lambda item: (item.memory, item.bank, item.mode)))


def memory_port_conflict(
    active_accesses: Mapping[str, tuple[MemoryAccess, ...]],
    candidate: tuple[MemoryAccess, ...],
    machine: MachineConfig,
) -> tuple[str, int, str] | None:
    active_counts: dict[tuple[str, int, str], int] = {}
    for accesses in active_accesses.values():
        for access in accesses:
            key = (access.memory, access.bank, access.mode)
            active_counts[key] = active_counts.get(key, 0) + 1
    for access in candidate:
        level = machine.memory(access.memory)
        limit = level.read_ports if access.mode == "read" else level.write_ports
        if active_counts.get((access.memory, access.bank, access.mode), 0) >= limit:
            return access.memory, access.bank, access.mode
    return None


def _reads(operand: RuntimeOperandBinding) -> bool:
    return operand.access_type in {AccessType.READ.value, AccessType.READ_WRITE.value}


def _writes(operand: RuntimeOperandBinding) -> bool:
    return operand.access_type in {AccessType.WRITE.value, AccessType.READ_WRITE.value}


def _operand_conflict(
    older: BoundTISADescriptor,
    younger: BoundTISADescriptor,
) -> tuple[str, RuntimeOperandBinding, RuntimeOperandBinding] | None:
    for left in older.operands:
        for right in younger.operands:
            if left.physical_scope != right.physical_scope:
                continue
            left_identity = left.attributes.get("allocation_identity")
            right_identity = right.attributes.get("allocation_identity")
            if (
                left_identity is not None
                and right_identity is not None
                and left_identity != right_identity
            ):
                continue
            if not (
                left.address < right.address + right.size_bytes
                and right.address < left.address + left.size_bytes
            ):
                continue
            if _writes(left) and _reads(right):
                return "RAW", left, right
            if _reads(left) and _writes(right):
                return "WAR", left, right
            if _writes(left) and _writes(right):
                return "WAW", left, right
    return None


def address_conflict(
    older: BoundTISADescriptor,
    younger: BoundTISADescriptor,
) -> tuple[str, RuntimeOperandBinding] | None:
    """Return the first conservative physical-address conflict in program order."""

    conflict = _operand_conflict(older, younger)
    if conflict is None:
        return None
    kind, older_operand, _younger_operand = conflict
    return kind, older_operand


def semantic_conflict(
    active: BoundTISADescriptor,
    candidate: BoundTISADescriptor,
) -> SemanticConflict | None:
    """Apply the implemented subset of TISA Algorithm 2.

    The runtime-bound memory space and allocation identity establish the
    alias domain.  Range overlap plus a write-bearing access establishes a
    RAW/WAR/WAW conflict.  No OpType pair is assumed reorder-safe without an
    explicit project contract, so reductions, accumulation and atomics remain
    conservative.  Cross-unit declared dependency readiness is handled by the
    completion-tag broadcast; this function checks only the entries supplied
    by the caller's local ``Fu[u]``.
    """

    conflict = _operand_conflict(active, candidate)
    if conflict is None:
        return None
    kind, predecessor_operand, successor_operand = conflict
    overlap_start = max(predecessor_operand.address, successor_operand.address)
    overlap_end = min(
        predecessor_operand.address + predecessor_operand.size_bytes,
        successor_operand.address + successor_operand.size_bytes,
    )
    predecessor_id = active.instruction.tisa_id
    return SemanticConflict(
        predecessor=predecessor_id,
        successor=candidate.instruction.tisa_id,
        kind=kind,
        memory=predecessor_operand.physical_scope,
        address=overlap_start,
        size_bytes=overlap_end - overlap_start,
        predecessor_operand=predecessor_operand.operand_name,
        successor_operand=successor_operand.operand_name,
        predecessor_op_type=active.instruction.op_type,
        successor_op_type=candidate.instruction.op_type,
        predecessor_logical_scope=predecessor_operand.logical_scope,
        successor_logical_scope=successor_operand.logical_scope,
        declared_dependency=any(
            dependency.source.tisa_id == predecessor_id
            for dependency in candidate.dependencies
        ),
    )


def dependency_details(descriptor: BoundTISADescriptor) -> list[dict[str, Any]]:
    return [
        {
            "predecessor": dependency.source.tisa_id,
            "completion_token": dependency.source.token_id,
            "kind": dependency.kind,
            "condition": dependency.condition,
            "provenance": dict(dependency.provenance),
        }
        for dependency in descriptor.dependencies
    ]


def address_observation(
    predecessor: BoundTISADescriptor,
    successor: BoundTISADescriptor,
    kind: str,
    region: RuntimeOperandBinding,
) -> dict[str, Any]:
    dependency = next(
        (
            item
            for item in successor.dependencies
            if item.source.tisa_id == predecessor.instruction.tisa_id
        ),
        None,
    )
    result: dict[str, Any] = {
        "predecessor": predecessor.instruction.tisa_id,
        "successor": successor.instruction.tisa_id,
        "kind": kind,
        "tensor": region.tensor,
        "memory": region.physical_scope,
        "address": region.address,
        "size_bytes": region.size_bytes,
        "condition": dependency.condition if dependency else "address_overlap",
        "provenance": {
            "source": "bounded_address_scoreboard",
            "dependency": dependency.to_dict() if dependency else None,
        },
    }
    for key in (
        "address_source",
        "dynamic_region",
        "resolved_index_values",
        "resolved_offset_bytes",
    ):
        if key in region.attributes:
            result[key] = region.attributes[key]
    return result


def critical_path_lengths(
    descriptors: tuple[BoundTISADescriptor, ...],
    execution: ExecutionBackend,
) -> dict[str, float]:
    successors = {
        descriptor.instruction.tisa_id: [] for descriptor in descriptors
    }
    for descriptor in descriptors:
        for dependency in descriptor.dependencies:
            successors[dependency.source.tisa_id].append(
                descriptor.instruction.tisa_id
            )
    lengths: dict[str, float] = {}
    for descriptor in reversed(descriptors):
        tisa_id = descriptor.instruction.tisa_id
        lengths[tisa_id] = execution.estimate(
            descriptor.payload_handle
        ).duration_cycles + max(
            (lengths[successor] for successor in successors[tisa_id]),
            default=0.0,
        )
    return lengths


def unit_map_matches(descriptor: BoundTISADescriptor, resource: str) -> bool:
    requested = descriptor.instruction.unit_map.unit.lower()
    physical = resource.lower()
    if requested == physical:
        return True
    aliases = {
        "dma": {"dma", "gdma", "ldma", "de", "copy"},
        "tensor": {"mxu", "me", "tensor", "matrix"},
        "vector": {"aru", "ve", "vector", "vu"},
        "scalar": {"scalar", "cpu"},
    }
    return physical in aliases.get(requested, set())


__all__ = [
    "MemoryAccess",
    "SemanticConflict",
    "access_banks",
    "address_conflict",
    "address_observation",
    "critical_path_lengths",
    "dependency_details",
    "memory_accesses",
    "memory_port_conflict",
    "semantic_conflict",
    "unit_map_matches",
]
