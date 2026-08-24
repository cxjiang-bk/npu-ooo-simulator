"""Runtime binding contracts between compiled TISA and device submission.

The compiler deliberately leaves ``TileMem`` logical.  This module owns the
one execution-specific step that follows code generation: assigning physical
buffer ranges and packaging TISA ids into software command-buffer chunks.
Device scheduling is intentionally outside this module; a submission records
software order but does not prescribe device issue order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

from .operator import TensorSpec
from .tisa import BackendArtifact, TISAProgram


_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int32": 4,
    "float32": 4,
    "fp32": 4,
    "int64": 8,
    "float64": 8,
    "fp64": 8,
}


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype.lower().replace("torch.", ""), 2)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class BufferBinding:
    """Physical allocation for one logical tensor and scope."""

    tensor: str
    base_address: int
    size_bytes: int
    memory: str = "DRAM"
    logical_scope: str = "logical"
    dtype: str = "fp16"
    alignment_bytes: int = 1
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.tensor:
            issues.append("runtime buffer tensor must not be empty")
        if not self.memory or not self.logical_scope:
            issues.append(f"runtime buffer '{self.tensor}' memory and logical_scope must not be empty")
        if isinstance(self.base_address, bool) or not isinstance(self.base_address, int) or self.base_address < 0:
            issues.append(f"runtime buffer '{self.tensor}' base_address must be a non-negative integer")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            issues.append(f"runtime buffer '{self.tensor}' size_bytes must be positive")
        if isinstance(self.alignment_bytes, bool) or not isinstance(self.alignment_bytes, int) or self.alignment_bytes <= 0:
            issues.append(f"runtime buffer '{self.tensor}' alignment_bytes must be positive")
        elif self.base_address % self.alignment_bytes:
            issues.append(
                f"runtime buffer '{self.tensor}' base_address is not aligned to "
                f"{self.alignment_bytes} bytes"
            )
        return tuple(issues)

    @property
    def end_address(self) -> int:
        return self.base_address + self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "base_address": self.base_address,
            "end_address": self.end_address,
            "size_bytes": self.size_bytes,
            "memory": self.memory,
            "logical_scope": self.logical_scope,
            "dtype": self.dtype,
            "alignment_bytes": self.alignment_bytes,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class RuntimeOperandBinding:
    """Concrete range selected for one TISA operand."""

    tisa_id: str
    operand_name: str
    tensor: str
    logical_scope: str
    physical_scope: str
    address: int
    size_bytes: int
    access_type: str
    offset_bytes: int
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.tisa_id or not self.operand_name or not self.tensor:
            issues.append("runtime operand binding identifiers must not be empty")
        if not self.logical_scope or not self.physical_scope:
            issues.append(f"runtime operand '{self.operand_name}' scopes must not be empty")
        if self.address < 0 or self.offset_bytes < 0:
            issues.append(f"runtime operand '{self.operand_name}' address and offset must be non-negative")
        if self.size_bytes <= 0:
            issues.append(f"runtime operand '{self.operand_name}' size_bytes must be positive")
        if self.access_type not in {"read", "write", "read_write"}:
            issues.append(
                f"runtime operand '{self.operand_name}' has invalid access '{self.access_type}'"
            )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tisa_id": self.tisa_id,
            "operand_name": self.operand_name,
            "tensor": self.tensor,
            "logical_scope": self.logical_scope,
            "physical_scope": self.physical_scope,
            "address": self.address,
            "end_address": self.address + self.size_bytes,
            "size_bytes": self.size_bytes,
            "access_type": self.access_type,
            "offset_bytes": self.offset_bytes,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class RuntimeCommandChunk:
    """A software submission chunk; device issue may reorder its descriptors."""

    chunk_id: str
    queue: str
    submission_order: int
    tisa_ids: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.chunk_id or not self.queue:
            issues.append("runtime command chunk identifiers must not be empty")
        if self.submission_order < 0:
            issues.append(f"runtime command chunk '{self.chunk_id}' order must be non-negative")
        if not self.tisa_ids:
            issues.append(f"runtime command chunk '{self.chunk_id}' must contain TISA ids")
        if len(set(self.tisa_ids)) != len(self.tisa_ids):
            issues.append(f"runtime command chunk '{self.chunk_id}' TISA ids must be unique")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "queue": self.queue,
            "submission_order": self.submission_order,
            "tisa_ids": list(self.tisa_ids),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class RuntimeSubmission:
    """One software submission of a compiled TISA program."""

    submission_id: str
    program_id: str
    artifact_id: str | None
    policy: str
    buffers: tuple[BufferBinding, ...]
    operands: tuple[RuntimeOperandBinding, ...]
    commands: tuple[RuntimeCommandChunk, ...]
    launch_latency_cycles: float = 0.0
    synchronization_cycles: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, program: TISAProgram | None = None) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.submission_id or not self.program_id:
            issues.append("runtime submission identifiers must not be empty")
        if self.policy not in {"static", "dynamic_ready_queue"}:
            issues.append(f"runtime policy '{self.policy}' is unsupported")
        if self.launch_latency_cycles < 0 or self.synchronization_cycles < 0:
            issues.append("runtime latency values must be non-negative")
        buffer_keys = {(item.tensor, item.logical_scope) for item in self.buffers}
        if len(buffer_keys) != len(self.buffers):
            issues.append("runtime buffer bindings must be unique by tensor and logical_scope")
        for item in self.buffers:
            issues.extend(item.validate())
        operand_keys = {(item.tisa_id, item.operand_name) for item in self.operands}
        if len(operand_keys) != len(self.operands):
            issues.append("runtime operand bindings must be unique by TISA id and operand name")
        for item in self.operands:
            issues.extend(item.validate())
            matches = [
                buffer
                for buffer in self.buffers
                if buffer.tensor == item.tensor
                and buffer.logical_scope == item.logical_scope
                and buffer.memory == item.physical_scope
            ]
            if len(matches) != 1:
                issues.append(
                    f"runtime operand '{item.operand_name}' does not resolve to one physical buffer"
                )
            elif item.address < matches[0].base_address:
                issues.append(
                    f"runtime operand '{item.operand_name}' offset exceeds buffer "
                    f"'{item.tensor}' allocation"
                )
            elif item.address + item.size_bytes > matches[0].end_address:
                issues.append(
                    f"runtime operand '{item.operand_name}' offset exceeds buffer "
                    f"'{item.tensor}' allocation"
                )
        chunk_orders = [chunk.submission_order for chunk in self.commands]
        if chunk_orders != list(range(len(self.commands))):
            issues.append("runtime command chunks must have contiguous submission_order")
        command_ids: list[str] = []
        for chunk in self.commands:
            issues.extend(chunk.validate())
            command_ids.extend(chunk.tisa_ids)
        if len(set(command_ids)) != len(command_ids):
            issues.append("runtime command chunks must not repeat TISA ids")
        if program is not None:
            if self.program_id != program.program_id:
                issues.append(
                    f"runtime submission program '{self.program_id}' does not match '{program.program_id}'"
                )
            program_ids = [instruction.tisa_id for instruction in program.instructions]
            if sorted(command_ids) != sorted(program_ids):
                issues.append("runtime command chunks must cover every TISA instruction exactly once")
            else:
                submission_index = {tisa_id: index for index, tisa_id in enumerate(command_ids)}
                for instruction in program.instructions:
                    for dependency in instruction.dependencies:
                        if submission_index[dependency.source] >= submission_index[instruction.tisa_id]:
                            issues.append(
                                f"runtime submission sends '{instruction.tisa_id}' before dependency "
                                f"'{dependency.source}'"
                            )
            expected_operands = {
                (instruction.tisa_id, operand.name)
                for instruction in program.instructions
                for operand in instruction.operands
            }
            if operand_keys != expected_operands:
                missing = sorted(expected_operands - operand_keys)
                extra = sorted(operand_keys - expected_operands)
                if missing:
                    issues.append("runtime operand bindings are missing: " + ", ".join(map(str, missing[:8])))
                if extra:
                    issues.append("runtime operand bindings are unknown: " + ", ".join(map(str, extra[:8])))
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "program_id": self.program_id,
            "artifact_id": self.artifact_id,
            "policy": self.policy,
            "buffers": [item.to_dict() for item in self.buffers],
            "operands": [item.to_dict() for item in self.operands],
            "commands": [item.to_dict() for item in self.commands],
            "launch_latency_cycles": self.launch_latency_cycles,
            "synchronization_cycles": self.synchronization_cycles,
            "attributes": dict(self.attributes),
        }


def allocate_buffer_bindings(
    tensors: Sequence[TensorSpec],
    *,
    base_address: int = 0x10000000,
    memory: str = "DRAM",
    logical_scope: str = "logical",
    alignment_bytes: int = 256,
) -> tuple[BufferBinding, ...]:
    """Allocate non-overlapping physical ranges for resolved tensor specs."""

    if base_address < 0 or alignment_bytes <= 0:
        raise ValueError("base_address must be non-negative and alignment_bytes must be positive")
    bindings: list[BufferBinding] = []
    cursor = base_address
    for tensor in tensors:
        if any(not isinstance(value, int) or value <= 0 for value in tensor.shape):
            raise ValueError(
                f"tensor '{tensor.name}' must have resolved positive integer shape before runtime binding"
            )
        size_bytes = math.prod(tensor.shape) * _dtype_bytes(tensor.dtype)
        cursor = _align(cursor, alignment_bytes)
        binding = BufferBinding(
            tensor=tensor.name,
            base_address=cursor,
            size_bytes=size_bytes,
            memory=memory,
            logical_scope=logical_scope,
            dtype=tensor.dtype,
            alignment_bytes=alignment_bytes,
        )
        issues = binding.validate()
        if issues:
            raise ValueError("invalid runtime allocation: " + "; ".join(issues))
        bindings.append(binding)
        cursor += size_bytes
    return tuple(bindings)


def _buffer_lookup(buffers: Sequence[BufferBinding]) -> dict[tuple[str, str], BufferBinding]:
    lookup: dict[tuple[str, str], BufferBinding] = {}
    for binding in buffers:
        key = (binding.tensor, binding.logical_scope)
        if key in lookup:
            raise ValueError(f"duplicate runtime buffer binding for {key}")
        lookup[key] = binding
    return lookup


def _submission_order(program: TISAProgram, policy: str) -> tuple[str, ...]:
    """Return software submission order without changing the compiled program."""

    if policy == "static":
        return tuple(instruction.tisa_id for instruction in program.instructions)
    if policy != "dynamic_ready_queue":
        raise ValueError(f"runtime policy '{policy}' is unsupported")
    instructions = {instruction.tisa_id: instruction for instruction in program.instructions}
    source_order = {
        instruction.tisa_id: index for index, instruction in enumerate(program.instructions)
    }
    successors = {instruction.tisa_id: [] for instruction in program.instructions}
    remaining = {instruction.tisa_id: set() for instruction in program.instructions}
    for instruction in program.instructions:
        for dependency in instruction.dependencies:
            if dependency.source not in instructions:
                raise ValueError(
                    f"runtime submission references unknown TISA dependency '{dependency.source}'"
                )
            remaining[instruction.tisa_id].add(dependency.source)
            successors[dependency.source].append(instruction.tisa_id)
    ready = [tisa_id for tisa_id, dependencies in remaining.items() if not dependencies]
    ordered: list[str] = []
    last_tile: str | None = None
    while ready:
        # Keep the remaining stages of an admitted tile together when they are
        # ready.  This avoids filling a finite device tile window with partial
        # packets while retaining fanout-first selection between tile packets.
        ready.sort(
            key=lambda item: (
                instructions[item].tile_id != last_tile,
                -len(successors[item]),
                source_order[item],
                item,
            )
        )
        current = ready.pop(0)
        ordered.append(current)
        last_tile = instructions[current].tile_id
        for successor in successors[current]:
            remaining[successor].discard(current)
            if not remaining[successor] and successor not in ordered and successor not in ready:
                ready.append(successor)
    if len(ordered) != len(program.instructions):
        raise ValueError("TISA dependency graph contains a cycle during runtime submission")
    return tuple(ordered)


def create_runtime_submission(
    program_or_artifact: TISAProgram | BackendArtifact,
    buffers: Iterable[BufferBinding],
    *,
    submission_id: str | None = None,
    policy: str = "static",
    chunk_size: int | None = None,
    queue: str = "device",
    artifact_id: str | None = None,
    operand_offsets: Mapping[tuple[str, str], int] | None = None,
    operand_sizes: Mapping[tuple[str, str], int] | None = None,
    launch_latency_cycles: float = 0.0,
    synchronization_cycles: float = 0.0,
) -> RuntimeSubmission:
    """Bind a TISA program to physical buffers and command chunks.

    ``operand_offsets``/``operand_sizes`` are optional runtime bindings for
    descriptor replay, buffer reuse, or compiler-side symbolic addresses.
    An explicitly supplied runtime offset takes precedence; otherwise a
    concrete offset stored in ``TileMem`` is reused.
    """

    artifact = program_or_artifact if isinstance(program_or_artifact, BackendArtifact) else None
    program = artifact.program if artifact is not None else program_or_artifact
    if not isinstance(program, TISAProgram):
        raise TypeError("program_or_artifact must be a TISAProgram or BackendArtifact")
    normalized_buffers = tuple(buffers)
    buffer_lookup = _buffer_lookup(normalized_buffers)
    offsets = operand_offsets or {}
    sizes = operand_sizes or {}
    operand_bindings: list[RuntimeOperandBinding] = []
    for instruction in program.instructions:
        for operand in instruction.operands:
            tensor = operand.tile_mem.tensor or operand.tile_mem.base
            logical_scope = operand.tile_mem.scope
            binding = buffer_lookup.get((tensor, logical_scope))
            if binding is None:
                candidates = [item for item in normalized_buffers if item.tensor == tensor]
                if len(candidates) == 1:
                    binding = candidates[0]
            if binding is None:
                raise ValueError(
                    f"no runtime buffer binding for operand '{operand.name}' "
                    f"({tensor}, {logical_scope})"
                )
            key = (instruction.tisa_id, operand.name)
            requested_offset = offsets.get(key)
            offset = requested_offset
            if offset is None:
                offset = operand.tile_mem.offset_bytes
            if offset is None:
                offset = 0
            if offset < 0:
                raise ValueError(f"runtime operand offset for {key} must be non-negative")
            size_source = "tile_mem"
            size = operand.tile_mem.size_bytes
            if size is None:
                size = sizes.get(key)
                if size is not None:
                    size_source = "runtime_binding"
            if size is None:
                requested_size = math.prod(operand.tile_shape) * _dtype_bytes(binding.dtype)
                available_size = binding.size_bytes - offset
                if available_size <= 0:
                    raise ValueError(
                        f"runtime operand {key} offset exceeds buffer '{binding.tensor}' allocation"
                    )
                # Keep a conservative allocation only for an unresolved
                # dynamic shape or an operator geometry not covered by the
                # current compiler region legalizer; record that approximation
                # explicitly instead of presenting it as a precise tile range.
                size = min(requested_size, available_size)
                size_source = (
                    "tile_shape"
                    if requested_size <= available_size
                    else "buffer_capacity_fallback"
                )
            if size <= 0:
                raise ValueError(f"runtime operand size for {key} must be positive")
            operand_bindings.append(
                RuntimeOperandBinding(
                    tisa_id=instruction.tisa_id,
                    operand_name=operand.name,
                    tensor=tensor,
                    logical_scope=logical_scope,
                    physical_scope=binding.memory,
                    address=binding.base_address + offset,
                    size_bytes=size,
                    access_type=operand.normalized_access,
                    offset_bytes=offset,
                    attributes={
                        "address_source": (
                            "runtime_binding"
                            if requested_offset is not None
                            else "tile_mem"
                            if operand.tile_mem.offset_bytes is not None
                            else "runtime_binding"
                        ),
                        "size_source": size_source,
                    },
                )
            )
    effective_chunk_size = (
        len(program.instructions) or 1 if chunk_size is None else chunk_size
    )
    if effective_chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    submission_order = _submission_order(program, policy)
    commands = tuple(
        RuntimeCommandChunk(
            chunk_id=f"{submission_id or program.program_id}.chunk{index:04d}",
            queue=queue,
            submission_order=index,
            tisa_ids=tuple(
                tisa_id
                for tisa_id in submission_order[start : start + effective_chunk_size]
            ),
        )
        for index, start in enumerate(range(0, len(program.instructions), effective_chunk_size))
    )
    submission = RuntimeSubmission(
        submission_id=submission_id or f"submission.{program.program_id}",
        program_id=program.program_id,
        artifact_id=artifact_id or (artifact.artifact_id if artifact is not None else None),
        policy=policy,
        buffers=normalized_buffers,
        operands=tuple(operand_bindings),
        commands=commands,
        launch_latency_cycles=launch_latency_cycles,
        synchronization_cycles=synchronization_cycles,
        attributes={
            "source": "runtime-buffer-binding",
            "device_issue_order": "independent",
            "submission_order_kind": (
                "program_order"
                if policy == "static"
                else "dependency_ready_tile_affine_fanout"
            ),
            "command_chunk_size": effective_chunk_size,
        },
    )
    issues = submission.validate(program)
    if issues:
        raise ValueError("runtime submission validation failed: " + "; ".join(issues))
    return submission


__all__ = [
    "BufferBinding",
    "RuntimeCommandChunk",
    "RuntimeOperandBinding",
    "RuntimeSubmission",
    "allocate_buffer_bindings",
    "create_runtime_submission",
]
