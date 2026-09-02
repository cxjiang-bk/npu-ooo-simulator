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
from .index import DynamicIndexBinding, resolve_dynamic_index
from .tisa import BackendArtifact, TISAProgram
from .dtype import dtype_bytes as _shared_dtype_bytes
from .layout import resolve_layout, tensor_layout


def _dtype_bytes(dtype: str) -> int:
    return _shared_dtype_bytes(dtype, default=2)


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
class RuntimeLayoutBinding:
    """Per-invocation concrete layout supplied by the runtime.

    The compiler keeps symbolic/opaque layout metadata in ``TensorSpec`` and
    ``TileMem``.  This object binds a concrete shape and byte-stride vector at
    submission time, which is required for paged KV cache and externally
    allocated non-contiguous tensors.
    """

    tensor: str
    shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    layout: str = "runtime"
    offset_bytes: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.tensor:
            issues.append("runtime layout tensor must not be empty")
        if len(self.shape) != len(self.strides_bytes):
            issues.append(f"runtime layout '{self.tensor}' shape and strides must have equal rank")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.shape):
            issues.append(f"runtime layout '{self.tensor}' shape must contain positive integers")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.strides_bytes):
            issues.append(f"runtime layout '{self.tensor}' strides_bytes must be non-negative integers")
        if not self.layout:
            issues.append(f"runtime layout '{self.tensor}' layout must not be empty")
        if isinstance(self.offset_bytes, bool) or not isinstance(self.offset_bytes, int) or self.offset_bytes < 0:
            issues.append(f"runtime layout '{self.tensor}' offset_bytes must be non-negative")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "shape": list(self.shape),
            "strides_bytes": list(self.strides_bytes),
            "layout": self.layout,
            "offset_bytes": self.offset_bytes,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class RuntimeStateRegistry:
    """Stable physical bindings for persistent runtime state.

    A compiler may expose more than one logical tensor for a state buffer
    (for example, a cache input and its output alias).  The registry indexes
    those bindings by ``state_id`` and requires every alias to resolve to the
    same physical range.  Temporary tensor allocation is intentionally not
    part of this contract.
    """

    bindings: tuple[BufferBinding, ...] = ()
    buffers: tuple[BufferBinding, ...] = ()

    @classmethod
    def from_bindings(cls, bindings: Iterable[BufferBinding]) -> "RuntimeStateRegistry":
        normalized = tuple(bindings)
        state_bindings = tuple(
            binding
            for binding in normalized
            if binding.attributes.get("persistent")
        )
        registry = cls(state_bindings, normalized)
        issues = registry.validate()
        if issues:
            raise ValueError("runtime state registry validation failed: " + "; ".join(issues))
        return registry

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        by_state: dict[str, list[BufferBinding]] = {}
        for binding in self.bindings:
            state_id = binding.attributes.get("state_id")
            if not isinstance(state_id, str) or not state_id:
                issues.append(
                    f"persistent buffer '{binding.tensor}' is missing state_id"
                )
                continue
            if not binding.attributes.get("persistent"):
                issues.append(
                    f"state buffer '{binding.tensor}' must be marked persistent"
                )
            issues.extend(binding.validate())
            by_state.setdefault(state_id, []).append(binding)
        for state_id, bindings in by_state.items():
            first = bindings[0]
            if any(
                (
                    item.base_address != first.base_address
                    or item.memory != first.memory
                )
                for item in bindings[1:]
            ):
                issues.append(
                    f"state '{state_id}' bindings must share one physical address and memory"
                )
        return tuple(issues)

    def state_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(binding.attributes["state_id"])
                    for binding in self.bindings
                    if isinstance(binding.attributes.get("state_id"), str)
                }
            )
        )

    def runtime_buffers(self) -> tuple[BufferBinding, ...]:
        """Return all bindings needed to construct a submission."""

        return self.buffers or self.bindings

    def binding(self, state_id: str) -> BufferBinding:
        matches = [
            binding
            for binding in self.bindings
            if binding.attributes.get("state_id") == state_id
        ]
        if not matches:
            raise KeyError(state_id)
        return max(matches, key=lambda item: (item.size_bytes, item.tensor))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "persistent_buffer_v1",
            "state_ids": list(self.state_ids()),
            "bindings": [binding.to_dict() for binding in self.bindings],
            "buffers": [binding.to_dict() for binding in self.runtime_buffers()],
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
    availability_cycle: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.chunk_id or not self.queue:
            issues.append("runtime command chunk identifiers must not be empty")
        if self.submission_order < 0:
            issues.append(f"runtime command chunk '{self.chunk_id}' order must be non-negative")
        if not math.isfinite(self.availability_cycle) or self.availability_cycle < 0:
            issues.append(f"runtime command chunk '{self.chunk_id}' availability must be non-negative")
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
            "availability_cycle": self.availability_cycle,
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
    dynamic_indices: tuple[DynamicIndexBinding, ...] = ()
    dynamic_layouts: tuple[RuntimeLayoutBinding, ...] = ()

    def validate(self, program: TISAProgram | None = None) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.submission_id or not self.program_id:
            issues.append("runtime submission identifiers must not be empty")
        if self.policy not in {"static", "dynamic_ready_queue"}:
            issues.append(f"runtime policy '{self.policy}' is unsupported")
        if (
            not math.isfinite(self.launch_latency_cycles)
            or not math.isfinite(self.synchronization_cycles)
            or self.launch_latency_cycles < 0
            or self.synchronization_cycles < 0
        ):
            issues.append("runtime latency values must be non-negative")
        dynamic_binding_ids = [binding.expression_id for binding in self.dynamic_indices]
        if len(set(dynamic_binding_ids)) != len(dynamic_binding_ids):
            issues.append("runtime dynamic index bindings must be unique by expression_id")
        for binding in self.dynamic_indices:
            issues.extend(binding.validate())
        layout_ids = [binding.tensor for binding in self.dynamic_layouts]
        if len(set(layout_ids)) != len(layout_ids):
            issues.append("runtime layout bindings must be unique by tensor")
        for binding in self.dynamic_layouts:
            issues.extend(binding.validate())
        buffer_keys = {(item.tensor, item.logical_scope) for item in self.buffers}
        if len(buffer_keys) != len(self.buffers):
            issues.append("runtime buffer bindings must be unique by tensor and logical_scope")
        for item in self.buffers:
            issues.extend(item.validate())
        buffer_tensor_names = {item.tensor for item in self.buffers}
        for binding in self.dynamic_layouts:
            if binding.tensor not in buffer_tensor_names:
                issues.append(
                    f"runtime layout binding '{binding.tensor}' does not match a runtime buffer"
                )
        state_contract = ()
        if program is not None:
            try:
                state_contract = _state_contract(program)
            except ValueError as exc:
                issues.append(str(exc))
        for descriptor in state_contract:
            matches = [
                item
                for item in self.buffers
                if item.tensor == descriptor["tensor"]
                and bool(item.attributes.get("persistent"))
            ]
            if len(matches) != 1:
                issues.append(
                    f"state buffer '{descriptor['tensor']}' for state_id "
                    f"'{descriptor['state_id']}' must have exactly one persistent binding"
                )
            elif matches[0].attributes.get("state_id", descriptor["state_id"]) != descriptor["state_id"]:
                issues.append(
                    f"state buffer '{descriptor['tensor']}' binding state_id does not match "
                    f"'{descriptor['state_id']}'"
                )
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
            expected_dynamic_indices = {
                str(instruction.attributes["dynamic_index"]["expression_id"])
                for instruction in program.instructions
                if isinstance(instruction.attributes.get("dynamic_index"), Mapping)
                and instruction.attributes["dynamic_index"].get("expression_id")
            }
            actual_dynamic_indices = set(dynamic_binding_ids)
            if actual_dynamic_indices - expected_dynamic_indices:
                issues.append(
                    "runtime dynamic index bindings are unknown: "
                    + ", ".join(sorted(actual_dynamic_indices - expected_dynamic_indices))
                )
            if expected_dynamic_indices - actual_dynamic_indices:
                issues.append(
                    "runtime dynamic index bindings are missing: "
                    + ", ".join(sorted(expected_dynamic_indices - actual_dynamic_indices))
                )
            dynamic_contracts = {
                str(instruction.attributes["dynamic_index"]["expression_id"]): instruction.attributes["dynamic_index"]
                for instruction in program.instructions
                if isinstance(instruction.attributes.get("dynamic_index"), Mapping)
                and instruction.attributes["dynamic_index"].get("expression_id")
            }
            for binding in self.dynamic_indices:
                contract = dynamic_contracts.get(binding.expression_id)
                if contract is not None:
                    rank = contract.get("index_rank")
                    if isinstance(rank, int) and len(binding.values) != rank:
                        issues.append(
                            f"runtime dynamic index binding '{binding.expression_id}' expects {rank} values"
                        )
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
            "dynamic_indices": [item.to_dict() for item in self.dynamic_indices],
            "dynamic_layouts": [item.to_dict() for item in self.dynamic_layouts],
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class RuntimeStateDependency:
    """Dependency from one invocation's completed state to the next."""

    source_invocation: str
    target_invocation: str
    state_ids: tuple[str, ...]
    condition: str = "state_complete"

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.source_invocation or not self.target_invocation:
            issues.append("runtime state dependency invocation ids must not be empty")
        if self.source_invocation == self.target_invocation:
            issues.append("runtime state dependency cannot be self-referential")
        if not self.state_ids:
            issues.append("runtime state dependency must name at least one state_id")
        if len(set(self.state_ids)) != len(self.state_ids):
            issues.append("runtime state dependency state_ids must be unique")
        if self.condition != "state_complete":
            issues.append(
                f"runtime state dependency condition '{self.condition}' is unsupported"
            )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_invocation": self.source_invocation,
            "target_invocation": self.target_invocation,
            "state_ids": list(self.state_ids),
            "condition": self.condition,
        }


@dataclass(frozen=True)
class RuntimeSequence:
    """Ordered invocations sharing a persistent state registry."""

    sequence_id: str
    program_id: str
    artifact_id: str | None
    invocations: tuple[RuntimeSubmission, ...]
    state_registry: RuntimeStateRegistry
    dependencies: tuple[RuntimeStateDependency, ...] = ()
    inter_invocation_gap_cycles: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, program: TISAProgram | None = None) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.sequence_id or not self.program_id:
            issues.append("runtime sequence identifiers must not be empty")
        if not self.invocations:
            issues.append("runtime sequence must contain at least one invocation")
        if (
            isinstance(self.inter_invocation_gap_cycles, bool)
            or not math.isfinite(self.inter_invocation_gap_cycles)
            or self.inter_invocation_gap_cycles < 0
        ):
            issues.append("runtime sequence inter-invocation gap must be non-negative")
        issues.extend(self.state_registry.validate())
        invocation_ids = [item.submission_id for item in self.invocations]
        registry_states = set(self.state_registry.state_ids())
        if len(set(invocation_ids)) != len(invocation_ids):
            issues.append("runtime sequence invocation ids must be unique")
        if self.artifact_id is not None:
            for item in self.invocations:
                if item.artifact_id not in {None, self.artifact_id}:
                    issues.append(
                        f"invocation '{item.submission_id}' artifact does not match sequence"
                    )
        for item in self.invocations:
            if item.program_id != self.program_id:
                issues.append(
                    f"invocation '{item.submission_id}' program does not match sequence"
                )
            issues.extend(item.validate(program))
            for descriptor in item.attributes.get("state_buffers", ()):
                state_id = descriptor.get("state_id") if isinstance(descriptor, Mapping) else None
                if state_id not in registry_states:
                    issues.append(
                        f"invocation '{item.submission_id}' references unregistered state '{state_id}'"
                    )
            for state_id in registry_states:
                registry_binding = self.state_registry.binding(state_id)
                matching = [
                    binding
                    for binding in item.buffers
                    if binding.attributes.get("state_id") == state_id
                    and binding.attributes.get("persistent")
                ]
                if not matching:
                    issues.append(
                        f"invocation '{item.submission_id}' must expose a persistent binding "
                        f"for state '{state_id}'"
                    )
                elif any(
                    binding.base_address != registry_binding.base_address
                    or binding.memory != registry_binding.memory
                    or binding.end_address > registry_binding.end_address
                    for binding in matching
                ):
                    issues.append(
                        f"invocation '{item.submission_id}' changes physical binding for state "
                        f"'{state_id}'"
                    )
        for dependency in self.dependencies:
            issues.extend(dependency.validate())
            if dependency.source_invocation not in invocation_ids:
                issues.append(
                    f"state dependency source '{dependency.source_invocation}' is unknown"
                )
            if dependency.target_invocation not in invocation_ids:
                issues.append(
                    f"state dependency target '{dependency.target_invocation}' is unknown"
                )
            if any(state_id not in self.state_registry.state_ids() for state_id in dependency.state_ids):
                issues.append("state dependency references an unregistered state")
        expected_dependency_count = max(0, len(self.invocations) - 1) if registry_states else 0
        if len(self.dependencies) != expected_dependency_count:
            issues.append(
                "runtime sequence must contain one state dependency per adjacent invocation "
                "when persistent state is present"
            )
        for index, dependency in enumerate(self.dependencies):
            if index + 1 >= len(self.invocations):
                continue
            if (
                dependency.source_invocation != self.invocations[index].submission_id
                or dependency.target_invocation != self.invocations[index + 1].submission_id
                or dependency.state_ids != self.state_registry.state_ids()
            ):
                issues.append(
                    "runtime sequence state dependencies must connect adjacent invocations "
                    "for every registered state"
                )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "program_id": self.program_id,
            "artifact_id": self.artifact_id,
            "invocations": [item.to_dict() for item in self.invocations],
            "state_registry": self.state_registry.to_dict(),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "inter_invocation_gap_cycles": self.inter_invocation_gap_cycles,
            "attributes": dict(self.attributes),
        }


def allocate_buffer_bindings(
    tensors: Sequence[TensorSpec],
    *,
    base_address: int = 0x10000000,
    memory: str = "DRAM",
    logical_scope: str = "logical",
    alignment_bytes: int = 256,
    lifetimes: Mapping[str, tuple[int, int]] | None = None,
    reuse_buffers: bool = False,
    reuse_pairs: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> tuple[BufferBinding, ...]:
    """Allocate physical ranges for resolved tensor specs.

    The default is a monotonic allocation, preserving the original runtime
    behavior.  ``reuse_buffers`` enables lifetime-aware reuse, but only for
    explicitly proven ``reuse_pairs``.  This keeps an out-of-order device from
    observing two logically live tensors at the same physical address merely
    because their compiler program-order intervals do not overlap.
    """

    if base_address < 0 or alignment_bytes <= 0:
        raise ValueError("base_address must be non-negative and alignment_bytes must be positive")
    if reuse_buffers and lifetimes is None:
        raise ValueError("reuse_buffers requires explicit tensor lifetimes")
    if lifetimes is not None:
        tensor_names = {tensor.name for tensor in tensors}
        unknown_lifetimes = sorted(set(lifetimes) - tensor_names)
        if unknown_lifetimes:
            raise ValueError(
                "lifetimes reference unknown tensors: " + ", ".join(unknown_lifetimes[:8])
            )
        for tensor, lifetime in lifetimes.items():
            if len(lifetime) != 2 or lifetime[0] < 0 or lifetime[1] < lifetime[0]:
                raise ValueError(f"invalid lifetime for tensor '{tensor}'")
    bindings_by_name: dict[str, BufferBinding] = {}

    def tensor_layout_metadata(tensor: TensorSpec) -> dict[str, Any]:
        layout_info = tensor_layout(tensor)
        return {
            "shape": list(layout_info.shape),
            "strides_bytes": (
                list(layout_info.strides_bytes)
                if layout_info.strides_bytes is not None
                else None
            ),
            "layout": layout_info.layout,
            "layout_source": layout_info.source,
            "layout_encoding": layout_info.encoding,
            "layout_resolution": (
                "concrete" if layout_info.concrete else "opaque_conservative"
            ),
            "layout_offset_bytes": layout_info.offset_bytes,
        }
    allocation_blocks: list[dict[str, Any]] = []
    cursor = base_address
    tensor_order = {tensor.name: index for index, tensor in enumerate(tensors)}
    allocation_order = tuple(
        sorted(
            tensors,
            key=lambda tensor: (
                lifetimes.get(tensor.name, (tensor_order[tensor.name],))[0]
                if lifetimes is not None
                else tensor_order[tensor.name],
                tensor_order[tensor.name],
            ),
        )
    )
    for tensor in allocation_order:
        if any(not isinstance(value, int) or value <= 0 for value in tensor.shape):
            raise ValueError(
                f"tensor '{tensor.name}' must have resolved positive integer shape before runtime binding"
            )
        size_bytes = tensor_layout(tensor).allocation_size_bytes
        alias_of = tensor.attributes.get("alias_of")
        if alias_of is not None:
            if not isinstance(alias_of, str) or not alias_of:
                raise ValueError(f"tensor '{tensor.name}' alias_of must be a non-empty tensor name")
            aliased = bindings_by_name.get(alias_of)
            if aliased is None:
                raise ValueError(
                    f"tensor '{tensor.name}' aliases '{alias_of}', which has not been allocated"
                )
            if size_bytes > aliased.size_bytes:
                raise ValueError(
                    f"tensor '{tensor.name}' alias size {size_bytes} exceeds '{alias_of}' allocation "
                    f"{aliased.size_bytes}"
                )
            binding = BufferBinding(
                tensor=tensor.name,
                base_address=aliased.base_address,
                size_bytes=size_bytes,
                memory=aliased.memory,
                logical_scope=logical_scope,
                dtype=tensor.dtype,
                alignment_bytes=aliased.alignment_bytes,
                attributes={
                    "allocation_policy": "alias",
                    "alias_of": alias_of,
                    **tensor_layout_metadata(tensor),
                    **(
                        {
                            "persistent": True,
                            "state_id": tensor.attributes.get("state_id", alias_of),
                            "state_buffer": tensor.attributes.get("state_buffer", alias_of),
                        }
                        if tensor.attributes.get("persistent")
                        else {}
                    ),
                },
            )
            issues = binding.validate()
            if issues:
                raise ValueError("invalid runtime alias allocation: " + "; ".join(issues))
            bindings_by_name[tensor.name] = binding
            continue
        lifetime = lifetimes.get(tensor.name) if lifetimes is not None else None
        persistent = bool(tensor.attributes.get("persistent"))
        selected_block: dict[str, Any] | None = None
        if reuse_buffers and lifetime is not None and not persistent:
            reusable = [
                block
                for block in allocation_blocks
                if block["lifetime_end"] < lifetime[0]
                and block["capacity_bytes"] >= size_bytes
                and (block["owner"], tensor.name) in reuse_pairs
            ]
            if reusable:
                selected_block = min(
                    reusable,
                    key=lambda block: (
                        block["capacity_bytes"] - size_bytes,
                        block["base_address"],
                    ),
                )
        if selected_block is None:
            cursor = _align(cursor, alignment_bytes)
            base = cursor
            cursor += size_bytes
        else:
            base = selected_block["base_address"]
        binding = BufferBinding(
            tensor=tensor.name,
            base_address=base,
            size_bytes=size_bytes,
            memory=memory,
            logical_scope=logical_scope,
            dtype=tensor.dtype,
            alignment_bytes=alignment_bytes,
            attributes={
                "allocation_policy": "lifetime_reuse" if reuse_buffers else "linear",
                **tensor_layout_metadata(tensor),
                **(
                    {
                        "persistent": True,
                        "state_id": tensor.attributes.get("state_id", tensor.name),
                        "state_buffer": tensor.attributes.get("state_buffer", tensor.name),
                    }
                    if tensor.attributes.get("persistent")
                    else {}
                ),
                **(
                    {
                        "lifetime_start": lifetime[0],
                        "lifetime_end": lifetime[1],
                        "reused_from": selected_block["owner"] if selected_block is not None else None,
                    }
                    if lifetime is not None
                    else {}
                ),
            },
        )
        issues = binding.validate()
        if issues:
            raise ValueError("invalid runtime allocation: " + "; ".join(issues))
        bindings_by_name[tensor.name] = binding
        if reuse_buffers and lifetime is not None and not persistent:
            if selected_block is None:
                allocation_blocks.append(
                    {
                        "owner": tensor.name,
                        "base_address": base,
                        "capacity_bytes": size_bytes,
                        "lifetime_end": lifetime[1],
                    }
                )
            else:
                selected_block.update(
                    owner=tensor.name,
                    lifetime_end=lifetime[1],
                )
    return tuple(bindings_by_name[tensor.name] for tensor in tensors)


def derive_tensor_lifetimes(program: TISAProgram) -> dict[str, tuple[int, int]]:
    """Return first/last TISA program-order use for every logical tensor."""

    lifetimes: dict[str, list[int]] = {}
    for index, instruction in enumerate(program.instructions):
        for operand in instruction.operands:
            tensor = operand.tile_mem.tensor or operand.tile_mem.base
            current = lifetimes.setdefault(tensor, [index, index])
            current[0] = min(current[0], index)
            current[1] = max(current[1], index)
    return {tensor: (bounds[0], bounds[1]) for tensor, bounds in lifetimes.items()}


def derive_tensor_reuse_pairs(program: TISAProgram) -> frozenset[tuple[str, str]]:
    """Prove tensor pairs whose uses are ordered by the TISA dependency DAG."""

    uses: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {
        instruction.tisa_id: set() for instruction in program.instructions
    }
    for instruction in program.instructions:
        for operand in instruction.operands:
            tensor = operand.tile_mem.tensor or operand.tile_mem.base
            uses.setdefault(tensor, set()).add(instruction.tisa_id)
        for dependency in instruction.dependencies:
            successors.setdefault(dependency.source, set()).add(instruction.tisa_id)

    descendants: dict[str, set[str]] = {}
    for source in successors:
        seen: set[str] = set()
        pending = list(successors[source])
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(successors.get(current, ()))
        descendants[source] = seen

    pairs: set[tuple[str, str]] = set()
    tensors = tuple(sorted(uses))
    for left in tensors:
        for right in tensors:
            if left == right:
                continue
            if all(
                right_use in descendants.get(left_use, set())
                for left_use in uses[left]
                for right_use in uses[right]
            ):
                pairs.add((left, right))
    return frozenset(pairs)


def _state_contract(program: TISAProgram) -> tuple[dict[str, Any], ...]:
    """Collect and validate persistent state metadata carried by TISA."""

    contracts: dict[str, dict[str, Any]] = {}
    for instruction in program.instructions:
        attributes = instruction.attributes
        if not attributes.get("stateful"):
            continue
        state_id = attributes.get("state_id")
        state_buffer = attributes.get("state_buffer")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(
                f"stateful TISA instruction '{instruction.tisa_id}' is missing state_id"
            )
        if not isinstance(state_buffer, str) or not state_buffer:
            raise ValueError(
                f"stateful TISA instruction '{instruction.tisa_id}' is missing state_buffer"
            )
        descriptor = {
            "state_id": state_id,
            "tensor": state_buffer,
            **{
                key: attributes[key]
                for key in (
                    "cache_axis",
                    "cache_window",
                    "update_length",
                    "slice_start",
                    "state_transition",
                )
                if key in attributes
            },
        }
        previous = contracts.get(state_id)
        if previous is not None and previous != descriptor:
            raise ValueError(
                f"stateful TISA instructions disagree on state contract '{state_id}'"
            )
        contracts[state_id] = descriptor
    return tuple(contracts[key] for key in sorted(contracts))


def _buffer_lookup(buffers: Sequence[BufferBinding]) -> dict[tuple[str, str], BufferBinding]:
    lookup: dict[tuple[str, str], BufferBinding] = {}
    for binding in buffers:
        key = (binding.tensor, binding.logical_scope)
        if key in lookup:
            raise ValueError(f"duplicate runtime buffer binding for {key}")
        lookup[key] = binding
    return lookup


def _binding_shape(binding: BufferBinding) -> tuple[int, ...]:
    shape = binding.attributes.get("shape")
    if not isinstance(shape, (tuple, list)) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in shape
    ):
        raise ValueError(
            f"runtime buffer '{binding.tensor}' must carry a resolved positive shape"
        )
    return tuple(int(value) for value in shape)


def _binding_strides(binding: BufferBinding, shape: tuple[int, ...]) -> tuple[int, ...]:
    layout_info = resolve_layout(
        shape,
        binding.dtype,
        layout=str(binding.attributes.get("layout", "dense")),
        attributes=binding.attributes,
    )
    if layout_info.strides_bytes is None:
        # Runtime allocation reserves the complete logical tensor when the
        # encoding is opaque.  Dense strides provide a deterministic address
        # fallback for APIs that require a concrete descriptor.
        return resolve_layout(shape, binding.dtype).strides_bytes or ()
    return layout_info.strides_bytes


def _dynamic_operand_resolution(
    instruction: Any,
    operand: Any,
    binding: BufferBinding,
    dynamic_bindings: Mapping[str, DynamicIndexBinding],
) -> tuple[int, int, dict[str, Any]] | None:
    """Resolve a dynamic operand window into a physical offset and span."""

    metadata = instruction.attributes.get("dynamic_index")
    if not isinstance(metadata, Mapping):
        return None
    expression_id = metadata.get("expression_id")
    runtime_binding = dynamic_bindings.get(str(expression_id))
    if runtime_binding is None:
        return None
    expression_attributes = metadata.get("attributes", {})
    operation = (
        expression_attributes.get("operation")
        if isinstance(expression_attributes, Mapping)
        else None
    )
    source_tensor = str(metadata.get("source_tensor", ""))
    is_state_target = False
    if operation == "dynamic_slice":
        if operand.tile_mem.tensor != source_tensor:
            return None
        raw_window = expression_attributes.get("sizes", ())
    elif operation == "dynamic_update_slice":
        state_buffer = str(instruction.attributes.get("state_buffer", source_tensor))
        alias_of = binding.attributes.get("alias_of")
        is_state_target = operand.normalized_access in {"write", "read_write"} and (
            operand.tile_mem.tensor == state_buffer or alias_of == state_buffer
        )
        if not is_state_target:
            return None
        raw_window = expression_attributes.get("update_shape", ())
    else:
        return None
    if not isinstance(raw_window, (tuple, list)):
        return None
    window_shape = tuple(int(value) for value in raw_window)
    source_shape = _binding_shape(binding)
    if len(window_shape) != len(source_shape):
        raise ValueError(
            f"dynamic index expression '{expression_id}' window rank does not match buffer "
            f"'{binding.tensor}'"
        )
    resolved = resolve_dynamic_index(metadata, runtime_binding, source_shape, window_shape)
    strides = _binding_strides(binding, source_shape)
    if len(strides) != len(source_shape):
        raise ValueError(
            f"dynamic operand '{operand.name}' strides rank does not match source shape"
        )
    layout = resolve_layout(
        source_shape,
        binding.dtype,
        layout=str(binding.attributes.get("layout", "dense")),
        attributes=binding.attributes,
    )
    offset = layout.offset_bytes + sum(
        index * stride for index, stride in zip(resolved, strides)
    )
    element_size = _dtype_bytes(binding.dtype)
    size = element_size + sum(
        (extent - 1) * stride for extent, stride in zip(window_shape, strides)
    )
    region = {
        "tensor": source_tensor if operation == "dynamic_update_slice" else operand.tile_mem.tensor,
        "starts": list(resolved),
        "shape": list(window_shape),
        "operation": operation,
        "expression_id": expression_id,
        "address_source": "dynamic_index_binding",
        "resolved_index_values": list(resolved),
        "resolved_offset_bytes": offset,
        "strides_bytes": list(strides),
    }
    return offset, size, region


def _runtime_layout_operand_resolution(
    operand: Any,
    binding: BufferBinding,
) -> tuple[int, int, dict[str, Any]] | None:
    """Project a compiled logical operand region through a runtime layout."""

    dynamic_layout = binding.attributes.get("dynamic_layout")
    starts = operand.tile_mem.logical_starts
    shape = operand.tile_mem.logical_shape
    if not isinstance(dynamic_layout, Mapping) or starts is None or shape is None:
        return None
    buffer_shape = _binding_shape(binding)
    layout = resolve_layout(
        buffer_shape,
        binding.dtype,
        layout=str(binding.attributes.get("layout", "dense")),
        attributes=binding.attributes,
    )
    interval = layout.interval(starts, shape)
    if interval is None:
        return None
    offset, size = interval
    return offset, size, {
        "tensor": binding.tensor,
        "starts": list(starts),
        "shape": list(shape),
        "address_source": "runtime_layout_binding",
        "layout": layout.layout,
        "strides_bytes": list(layout.strides_bytes or ()),
        "resolved_offset_bytes": offset,
    }


def _submission_order(
    program: TISAProgram,
    policy: str,
    descriptor_available_cycles: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Return software submission order without changing the compiled program."""

    instructions = {instruction.tisa_id: instruction for instruction in program.instructions}
    available_cycles = dict(descriptor_available_cycles or {})
    unknown_availability = sorted(set(available_cycles) - set(instructions))
    if unknown_availability:
        raise ValueError(
            "runtime availability references unknown TISA ids: "
            + ", ".join(unknown_availability[:8])
        )
    if any(
        isinstance(cycle, bool)
        or not isinstance(cycle, (int, float))
        or not math.isfinite(cycle)
        or cycle < 0
        for cycle in available_cycles.values()
    ):
        raise ValueError("runtime descriptor availability cycles must be finite and non-negative")
    if policy == "static":
        return tuple(instruction.tisa_id for instruction in program.instructions)
    if policy != "dynamic_ready_queue":
        raise ValueError(f"runtime policy '{policy}' is unsupported")
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
    availability_cursor = 0.0
    while ready:
        # Keep the remaining stages of an admitted tile together when they are
        # ready.  This avoids filling a finite device tile window with partial
        # packets while retaining fanout-first selection between tile packets.
        earliest_available = min(available_cycles.get(item, 0.0) for item in ready)
        availability_cursor = max(availability_cursor, earliest_available)
        available = [
            item
            for item in ready
            if available_cycles.get(item, 0.0) <= availability_cursor
        ]
        available.sort(
            key=lambda item: (
                instructions[item].tile_id != last_tile,
                -len(successors[item]),
                source_order[item],
                item,
            )
        )
        current = available[0]
        ready.remove(current)
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
    descriptor_available_cycles: Mapping[str, float] | None = None,
    launch_latency_cycles: float = 0.0,
    synchronization_cycles: float = 0.0,
    dynamic_index_bindings: Iterable[DynamicIndexBinding] = (),
    dynamic_layout_bindings: Iterable[RuntimeLayoutBinding] = (),
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
    normalized_layouts = tuple(dynamic_layout_bindings)
    layout_by_tensor = {binding.tensor: binding for binding in normalized_layouts}
    if len(layout_by_tensor) != len(normalized_layouts):
        raise ValueError("runtime layout bindings must be unique by tensor")
    for layout_binding in normalized_layouts:
        issues = layout_binding.validate()
        if issues:
            raise ValueError("invalid runtime layout binding: " + "; ".join(issues))
    supplied_buffers = tuple(buffers)
    unknown_layout_tensors = sorted(set(layout_by_tensor) - {item.tensor for item in supplied_buffers})
    if unknown_layout_tensors:
        raise ValueError(
            "runtime layout bindings reference unknown buffers: "
            + ", ".join(unknown_layout_tensors)
        )
    normalized_buffers_list: list[BufferBinding] = []
    for buffer in supplied_buffers:
        layout_binding = layout_by_tensor.get(buffer.tensor)
        if layout_binding is None:
            normalized_buffers_list.append(buffer)
            continue
        attributes = {
            **dict(buffer.attributes),
            "shape": list(layout_binding.shape),
            "strides_bytes": list(layout_binding.strides_bytes),
            "layout": layout_binding.layout,
            "layout_source": "runtime_binding",
            "layout_offset_bytes": layout_binding.offset_bytes,
            "dynamic_layout": layout_binding.to_dict(),
        }
        resolved_layout = resolve_layout(
            layout_binding.shape,
            buffer.dtype,
            layout=layout_binding.layout,
            attributes=attributes,
        )
        if resolved_layout.allocation_size_bytes > buffer.size_bytes:
            raise ValueError(
                f"runtime layout for '{buffer.tensor}' requires "
                f"{resolved_layout.allocation_size_bytes} bytes, allocation has {buffer.size_bytes}"
            )
        normalized_buffers_list.append(
            BufferBinding(
                tensor=buffer.tensor,
                base_address=buffer.base_address,
                size_bytes=buffer.size_bytes,
                memory=buffer.memory,
                logical_scope=buffer.logical_scope,
                dtype=buffer.dtype,
                alignment_bytes=buffer.alignment_bytes,
                attributes=attributes,
            )
        )
    normalized_buffers = tuple(normalized_buffers_list)
    normalized_dynamic_indices = tuple(dynamic_index_bindings)
    dynamic_bindings_by_id = {
        binding.expression_id: binding for binding in normalized_dynamic_indices
    }
    buffer_lookup = _buffer_lookup(normalized_buffers)
    state_contract = _state_contract(program)
    for descriptor in state_contract:
        state_binding = next(
            (
                item
                for item in normalized_buffers
                if item.tensor == descriptor["tensor"]
                and bool(item.attributes.get("persistent"))
            ),
            None,
        )
        if state_binding is None:
            raise ValueError(
                f"state buffer '{descriptor['tensor']}' for state_id "
                f"'{descriptor['state_id']}' must be bound as persistent"
            )
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
            dynamic_resolution = _dynamic_operand_resolution(
                instruction,
                operand,
                binding,
                dynamic_bindings_by_id,
            )
            layout_resolution = _runtime_layout_operand_resolution(operand, binding)
            if requested_offset is None:
                if dynamic_resolution is not None:
                    offset = dynamic_resolution[0]
                elif layout_resolution is not None:
                    offset = layout_resolution[0]
            size_source = "tile_mem"
            size = operand.tile_mem.size_bytes
            requested_size = sizes.get(key)
            if requested_size is not None:
                size = requested_size
                size_source = "runtime_binding"
            elif dynamic_resolution is not None:
                size = dynamic_resolution[1]
                size_source = "dynamic_index_binding"
            elif layout_resolution is not None:
                size = layout_resolution[1]
                size_source = "dynamic_layout_binding"
            if size is None:
                shape_size = math.prod(operand.tile_shape) * _dtype_bytes(binding.dtype)
                available_size = binding.size_bytes - offset
                if available_size <= 0:
                    raise ValueError(
                        f"runtime operand {key} offset exceeds buffer '{binding.tensor}' allocation"
                    )
                # Keep a conservative allocation only for an unresolved
                # dynamic shape or an operator geometry not covered by the
                # current compiler region legalizer; record that approximation
                # explicitly instead of presenting it as a precise tile range.
                size = min(shape_size, available_size)
                size_source = (
                    "tile_shape"
                    if shape_size <= available_size
                    else "buffer_capacity_fallback"
                )
            if size <= 0:
                raise ValueError(f"runtime operand size for {key} must be positive")
            operand_attributes = {
                "address_source": (
                    "runtime_binding"
                    if requested_offset is not None
                    else "dynamic_index_binding"
                    if dynamic_resolution is not None
                    else "dynamic_layout_binding"
                    if layout_resolution is not None
                    else "tile_mem"
                    if operand.tile_mem.offset_bytes is not None
                    else "runtime_binding"
                ),
                "size_source": size_source,
                "dynamic_index": instruction.attributes.get("dynamic_index"),
                "dynamic_index_values": (
                    list(dynamic_bindings_by_id[instruction.attributes["dynamic_index"]["expression_id"]].values)
                    if isinstance(instruction.attributes.get("dynamic_index"), Mapping)
                    and instruction.attributes["dynamic_index"].get("expression_id") in dynamic_bindings_by_id
                    else None
                ),
                "dynamic_layout": binding.attributes.get("dynamic_layout"),
                "runtime_strides_bytes": binding.attributes.get("strides_bytes"),
                "runtime_shape": binding.attributes.get("shape"),
            }
            if dynamic_resolution is not None:
                operand_attributes["dynamic_region"] = dynamic_resolution[2]
                operand_attributes["resolved_index_values"] = dynamic_resolution[2][
                    "resolved_index_values"
                ]
                operand_attributes["resolved_offset_bytes"] = dynamic_resolution[2][
                    "resolved_offset_bytes"
                ]
            elif layout_resolution is not None:
                operand_attributes["dynamic_region"] = layout_resolution[2]
                operand_attributes["resolved_offset_bytes"] = layout_resolution[2][
                    "resolved_offset_bytes"
                ]
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
                    attributes=operand_attributes,
                )
            )
    effective_chunk_size = (
        len(program.instructions) or 1 if chunk_size is None else chunk_size
    )
    if effective_chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    available_cycles = dict(descriptor_available_cycles or {})
    submission_order = _submission_order(program, policy, available_cycles)
    commands = tuple(
        RuntimeCommandChunk(
            chunk_id=f"{submission_id or program.program_id}.chunk{index:04d}",
            queue=queue,
            submission_order=index,
            tisa_ids=tuple(
                tisa_id
                for tisa_id in submission_order[start : start + effective_chunk_size]
            ),
            availability_cycle=max(
                (
                    available_cycles.get(tisa_id, 0.0)
                    for tisa_id in submission_order[start : start + effective_chunk_size]
                ),
                default=0.0,
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
        dynamic_indices=normalized_dynamic_indices,
        dynamic_layouts=normalized_layouts,
        attributes={
            "source": "runtime-buffer-binding",
            "device_issue_order": "independent",
            "submission_order_kind": (
                "program_order"
                if policy == "static"
                else "dependency_ready_tile_affine_fanout"
            ),
            "command_chunk_size": effective_chunk_size,
            "descriptor_availability_count": len(available_cycles),
            "state_contract": "persistent_buffer_v1" if state_contract else None,
            "state_buffers": list(state_contract),
            "dynamic_index_binding_count": len(normalized_dynamic_indices),
            "dynamic_layout_binding_count": len(normalized_layouts),
        },
    )
    issues = submission.validate(program)
    if issues:
        raise ValueError("runtime submission validation failed: " + "; ".join(issues))
    return submission


def create_runtime_state_registry(
    program_or_artifact: TISAProgram | BackendArtifact,
    buffers: Iterable[BufferBinding],
) -> RuntimeStateRegistry:
    """Create a state registry from a compiled program and its bindings."""

    artifact = program_or_artifact if isinstance(program_or_artifact, BackendArtifact) else None
    program = artifact.program if artifact is not None else program_or_artifact
    if not isinstance(program, TISAProgram):
        raise TypeError("program_or_artifact must be a TISAProgram or BackendArtifact")
    normalized = tuple(buffers)
    contracts = _state_contract(program)
    persistent: list[BufferBinding] = []
    for descriptor in contracts:
        matches = [
            binding
            for binding in normalized
            if binding.tensor == descriptor["tensor"]
            and binding.attributes.get("persistent")
            and binding.attributes.get("state_id", descriptor["state_id"]) == descriptor["state_id"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"state buffer '{descriptor['tensor']}' for state_id "
                f"'{descriptor['state_id']}' must have exactly one persistent binding"
            )
        persistent.append(matches[0])
    registry = RuntimeStateRegistry(tuple(persistent), normalized)
    issues = registry.validate()
    if issues:
        raise ValueError("runtime state registry validation failed: " + "; ".join(issues))
    return registry


def create_runtime_sequence(
    program_or_artifact: TISAProgram | BackendArtifact,
    state_registry: RuntimeStateRegistry,
    *,
    invocation_count: int,
    sequence_id: str | None = None,
    invocation_buffers: Sequence[Iterable[BufferBinding]] | None = None,
    policy: str = "static",
    chunk_size: int | None = None,
    queue: str = "device",
    operand_offsets: Mapping[tuple[str, str], int] | None = None,
    operand_sizes: Mapping[tuple[str, str], int] | None = None,
    descriptor_available_cycles: Mapping[str, float] | None = None,
    launch_latency_cycles: float = 0.0,
    synchronization_cycles: float = 0.0,
    inter_invocation_gap_cycles: float = 0.0,
    invocation_dynamic_indices: Sequence[Iterable[DynamicIndexBinding]] | None = None,
    invocation_dynamic_layouts: Sequence[Iterable[RuntimeLayoutBinding]] | None = None,
) -> RuntimeSequence:
    """Build repeated submissions sharing stable persistent state buffers.

    ``invocation_buffers`` can replace temporary/input bindings per step, but
    each entry must still contain the registry's persistent bindings.  The
    same compiled TISA program is used for every invocation; only submission
    identity and state-completion edges vary.
    """

    artifact = program_or_artifact if isinstance(program_or_artifact, BackendArtifact) else None
    program = artifact.program if artifact is not None else program_or_artifact
    if not isinstance(program, TISAProgram):
        raise TypeError("program_or_artifact must be a TISAProgram or BackendArtifact")
    if isinstance(invocation_count, bool) or not isinstance(invocation_count, int) or invocation_count <= 0:
        raise ValueError("invocation_count must be a positive integer")
    if not isinstance(state_registry, RuntimeStateRegistry):
        raise TypeError("state_registry must be a RuntimeStateRegistry")
    registry_issues = state_registry.validate()
    if registry_issues:
        raise ValueError("runtime state registry validation failed: " + "; ".join(registry_issues))
    if invocation_buffers is not None and len(invocation_buffers) != invocation_count:
        raise ValueError("invocation_buffers length must equal invocation_count")
    if (
        invocation_dynamic_indices is not None
        and len(invocation_dynamic_indices) != invocation_count
    ):
        raise ValueError("invocation_dynamic_indices length must equal invocation_count")
    if (
        invocation_dynamic_layouts is not None
        and len(invocation_dynamic_layouts) != invocation_count
    ):
        raise ValueError("invocation_dynamic_layouts length must equal invocation_count")
    if (
        isinstance(inter_invocation_gap_cycles, bool)
        or not math.isfinite(inter_invocation_gap_cycles)
        or inter_invocation_gap_cycles < 0
    ):
        raise ValueError("inter_invocation_gap_cycles must be non-negative")

    prefix = sequence_id or f"sequence.{program.program_id}"
    buffers_by_invocation = (
        tuple(tuple(items) for items in invocation_buffers)
        if invocation_buffers is not None
        else tuple(state_registry.runtime_buffers() for _ in range(invocation_count))
    )
    invocations: list[RuntimeSubmission] = []
    for ordinal in range(invocation_count):
        invocations.append(
            create_runtime_submission(
                program_or_artifact,
                buffers_by_invocation[ordinal],
                submission_id=f"{prefix}.invocation{ordinal:04d}",
                policy=policy,
                chunk_size=chunk_size,
                queue=queue,
                operand_offsets=operand_offsets,
                operand_sizes=operand_sizes,
                descriptor_available_cycles=descriptor_available_cycles,
                launch_latency_cycles=launch_latency_cycles,
                synchronization_cycles=synchronization_cycles,
                dynamic_index_bindings=(
                    invocation_dynamic_indices[ordinal]
                    if invocation_dynamic_indices is not None
                    else ()
                ),
                dynamic_layout_bindings=(
                    invocation_dynamic_layouts[ordinal]
                    if invocation_dynamic_layouts is not None
                    else ()
                ),
            )
        )
    dependencies = tuple(
        RuntimeStateDependency(
            source_invocation=invocations[ordinal - 1].submission_id,
            target_invocation=invocations[ordinal].submission_id,
            state_ids=state_registry.state_ids(),
        )
        for ordinal in range(1, invocation_count)
        if state_registry.state_ids()
    )
    sequence = RuntimeSequence(
        sequence_id=prefix,
        program_id=program.program_id,
        artifact_id=artifact.artifact_id if artifact is not None else None,
        invocations=tuple(invocations),
        state_registry=state_registry,
        dependencies=dependencies,
        inter_invocation_gap_cycles=inter_invocation_gap_cycles,
        attributes={
            "state_contract": "persistent_buffer_v1" if state_registry.state_ids() else None,
            "state_ids": list(state_registry.state_ids()),
            "invocation_count": invocation_count,
            "dependency_kind": "state_complete",
            "compiled_program_reused": True,
        },
    )
    issues = sequence.validate(program)
    if issues:
        raise ValueError("runtime sequence validation failed: " + "; ".join(issues))
    return sequence


__all__ = [
    "BufferBinding",
    "RuntimeLayoutBinding",
    "RuntimeCommandChunk",
    "RuntimeOperandBinding",
    "RuntimeStateDependency",
    "RuntimeStateRegistry",
    "RuntimeSequence",
    "RuntimeSubmission",
    "allocate_buffer_bindings",
    "create_runtime_state_registry",
    "create_runtime_sequence",
    "create_runtime_submission",
    "derive_tensor_lifetimes",
    "derive_tensor_reuse_pairs",
]
