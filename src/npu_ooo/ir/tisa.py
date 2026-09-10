"""Tile-level semantic instruction contracts used by the compiler/backend boundary.

The objects in this module deliberately sit between :class:`TileInstance` and
backend-specific :class:`ExecutionTask`.  A TISA instruction is one semantic
tile operation; its backend payload may contain several primitive operations,
but the device scheduler observes the instruction as one run-to-complete unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

from .execution import AccessType, ExecutionGraph
from .memory import MemoryPlan

if TYPE_CHECKING:
    from .static import StaticControlProgram
    from .target import TargetPlan


@dataclass(frozen=True)
class TileMem:
    """Symbolic or target-materialized memory descriptor.

    FC uses the logical fields; Codegen adds ``buffer_id``, ``allocation_id``
    and a concrete memory scope. ``address_expr`` keeps the logical slice
    auditable after target materialization.
    """

    base: str
    scope: str = "local"
    tensor: str | None = None
    dtype: str = "fp16"
    offset_bytes: int | None = None
    size_bytes: int | None = None
    address_expr: str | None = None
    strides_bytes: tuple[int, ...] | None = None
    stride_expr: str | None = None
    layout: str = "dense"
    logical_starts: tuple[int, ...] | None = None
    logical_shape: tuple[int, ...] | None = None
    memory_space: str | None = None
    visibility: str | None = None
    role: str | None = None
    owner: str | None = None
    domain: str | None = None
    symbolic_buffer_id: str | None = None
    buffer_id: str | None = None
    allocation_id: str | None = None
    valid_bytes: int | None = None

    @property
    def physical_space(self) -> str:
        """Return the target memory, falling back for legacy descriptors."""

        return self.memory_space or self.scope

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.base:
            issues.append("TISA TileMem base must not be empty")
        if not self.scope:
            issues.append("TISA TileMem scope must not be empty")
        if not self.dtype:
            issues.append("TISA TileMem dtype must not be empty")
        if self.offset_bytes is not None and self.offset_bytes < 0:
            issues.append("TISA TileMem offset_bytes must be non-negative")
        if self.size_bytes is not None and self.size_bytes <= 0:
            issues.append("TISA TileMem size_bytes must be positive")
        if self.buffer_id is not None and not self.buffer_id:
            issues.append("TISA TileMem buffer_id must not be blank")
        if self.allocation_id is not None and not self.allocation_id:
            issues.append("TISA TileMem allocation_id must not be blank")
        for label, value in (
            ("memory_space", self.memory_space),
            ("visibility", self.visibility),
            ("role", self.role),
            ("owner", self.owner),
            ("domain", self.domain),
            ("symbolic_buffer_id", self.symbolic_buffer_id),
        ):
            if value is not None and not value.strip():
                issues.append(f"TISA TileMem {label} must not be blank")
        if self.valid_bytes is not None and (
            self.valid_bytes <= 0
            or (self.size_bytes is not None and self.valid_bytes > self.size_bytes)
        ):
            issues.append("TISA TileMem valid_bytes is invalid")
        if self.address_expr is not None and not self.address_expr.strip():
            issues.append("TISA TileMem address_expr must not be blank")
        if not self.layout or not self.layout.strip():
            issues.append("TISA TileMem layout must not be blank")
        if self.strides_bytes is not None:
            # A rank-0 tensor has no stride axes; ``()`` is its complete
            # concrete stride metadata and is distinct from missing metadata.
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.strides_bytes
            ):
                issues.append("TISA TileMem strides_bytes must contain non-negative integers")
        if self.stride_expr is not None and not self.stride_expr.strip():
            issues.append("TISA TileMem stride_expr must not be blank")
        if (self.logical_starts is None) != (self.logical_shape is None):
            issues.append("TISA TileMem logical starts and shape must be provided together")
        if self.logical_starts is not None and self.logical_shape is not None:
            if len(self.logical_starts) != len(self.logical_shape):
                issues.append("TISA TileMem logical starts and shape must have equal rank")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.logical_starts
            ):
                issues.append("TISA TileMem logical starts must be non-negative integers")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.logical_shape
            ):
                issues.append("TISA TileMem logical shape must contain positive integers")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "scope": self.scope,
            "tensor": self.tensor,
            "dtype": self.dtype,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "address_expr": self.address_expr,
            "strides_bytes": list(self.strides_bytes) if self.strides_bytes is not None else None,
            "stride_expr": self.stride_expr,
            "layout": self.layout,
            "logical_starts": list(self.logical_starts) if self.logical_starts is not None else None,
            "logical_shape": list(self.logical_shape) if self.logical_shape is not None else None,
            "memory_space": self.memory_space,
            "visibility": self.visibility,
            "role": self.role,
            "owner": self.owner,
            "domain": self.domain,
            "symbolic_buffer_id": self.symbolic_buffer_id,
            "buffer_id": self.buffer_id,
            "allocation_id": self.allocation_id,
            "valid_bytes": self.valid_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TileMem":
        if not isinstance(payload, Mapping):
            raise ValueError("TISA TileMem payload must be an object")
        try:
            value = cls(
                base=str(payload["base"]),
                scope=str(payload.get("scope", "local")),
                tensor=(str(payload["tensor"]) if payload.get("tensor") is not None else None),
                dtype=str(payload.get("dtype", "fp16")),
                offset_bytes=(int(payload["offset_bytes"]) if payload.get("offset_bytes") is not None else None),
                size_bytes=(int(payload["size_bytes"]) if payload.get("size_bytes") is not None else None),
                address_expr=(str(payload["address_expr"]) if payload.get("address_expr") is not None else None),
                strides_bytes=(
                    tuple(int(item) for item in payload["strides_bytes"])
                    if payload.get("strides_bytes") is not None
                    else None
                ),
                stride_expr=(str(payload["stride_expr"]) if payload.get("stride_expr") is not None else None),
                layout=str(payload.get("layout", "dense")),
                logical_starts=(
                    tuple(int(item) for item in payload["logical_starts"])
                    if payload.get("logical_starts") is not None
                    else None
                ),
                logical_shape=(
                    tuple(int(item) for item in payload["logical_shape"])
                    if payload.get("logical_shape") is not None
                    else None
                ),
                memory_space=(str(payload["memory_space"]) if payload.get("memory_space") is not None else None),
                visibility=(str(payload["visibility"]) if payload.get("visibility") is not None else None),
                role=(str(payload["role"]) if payload.get("role") is not None else None),
                owner=(str(payload["owner"]) if payload.get("owner") is not None else None),
                domain=(str(payload["domain"]) if payload.get("domain") is not None else None),
                symbolic_buffer_id=(
                    str(payload["symbolic_buffer_id"])
                    if payload.get("symbolic_buffer_id") is not None
                    else None
                ),
                buffer_id=(str(payload["buffer_id"]) if payload.get("buffer_id") is not None else None),
                allocation_id=(str(payload["allocation_id"]) if payload.get("allocation_id") is not None else None),
                valid_bytes=(int(payload["valid_bytes"]) if payload.get("valid_bytes") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid TISA TileMem payload") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid TISA TileMem: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class TISAOperand:
    """A tile operand: shape, memory range and access mode."""

    name: str
    tile_shape: tuple[int, ...]
    tile_mem: TileMem
    access_type: AccessType | str

    @property
    def normalized_access(self) -> str:
        return self.access_type.value if isinstance(self.access_type, AccessType) else str(self.access_type)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name:
            issues.append("TISA operand name must not be empty")
        if any(value <= 0 for value in self.tile_shape):
            issues.append(f"TISA operand '{self.name}' tile_shape values must be positive")
        if self.normalized_access not in {item.value for item in AccessType}:
            issues.append(f"TISA operand '{self.name}' has invalid access type '{self.normalized_access}'")
        issues.extend(self.tile_mem.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tile_shape": list(self.tile_shape),
            "tile_mem": self.tile_mem.to_dict(),
            "access_type": self.normalized_access,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TISAOperand":
        if not isinstance(payload, Mapping):
            raise ValueError("TISA operand payload must be an object")
        try:
            value = cls(
                name=str(payload["name"]),
                tile_shape=tuple(int(item) for item in payload["tile_shape"]),
                tile_mem=TileMem.from_dict(payload["tile_mem"]),
                access_type=payload["access_type"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid TISA operand payload") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid TISA operand: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class UnitMap:
    """Resource class requested by one semantic tile instruction."""

    unit: str
    quantity: int = 1
    affinity: str | None = None

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.unit:
            issues.append("TISA UnitMap unit must not be empty")
        if isinstance(self.quantity, bool) or self.quantity <= 0:
            issues.append("TISA UnitMap quantity must be positive")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "quantity": self.quantity, "affinity": self.affinity}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitMap":
        if not isinstance(payload, Mapping):
            raise ValueError("TISA UnitMap payload must be an object")
        try:
            value = cls(
                unit=str(payload["unit"]),
                quantity=int(payload.get("quantity", 1)),
                affinity=(str(payload["affinity"]) if payload.get("affinity") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid TISA UnitMap payload") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid TISA UnitMap: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class TISADependency:
    """Typed dependency between semantic tile instructions."""

    source: str
    kind: str = "RAW"
    condition: str = "full_region_ready"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.source:
            issues.append("TISA dependency source must not be empty")
        if self.kind not in {
            "RAW",
            "WAR",
            "WAW",
            "STATE",
            "ACCUMULATE",
            "BUFFER_REUSE",
            "CONTROL",
        }:
            issues.append(f"TISA dependency kind '{self.kind}' is unsupported")
        if not self.condition:
            issues.append("TISA dependency condition must not be empty")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "condition": self.condition,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TISADependency":
        if not isinstance(payload, Mapping):
            raise ValueError("TISA dependency payload must be an object")
        try:
            value = cls(
                source=str(payload["source"]),
                kind=str(payload.get("kind", "RAW")),
                condition=str(payload.get("condition", "full_region_ready")),
                provenance=payload.get("provenance", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid TISA dependency payload") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid TISA dependency: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class TISAInstruction:
    """One scheduler-visible semantic tile instruction."""

    tisa_id: str
    tile_id: str
    operator_id: str
    op_type: str
    operands: tuple[TISAOperand, ...]
    unit_map: UnitMap
    dependencies: tuple[TISADependency, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    payload_ref: str | None = None

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.tisa_id or not self.tile_id or not self.operator_id or not self.op_type:
            issues.append("TISA instruction identifiers and op_type must not be empty")
        if not self.operands:
            issues.append(f"TISA instruction '{self.tisa_id}' must have operands")
        for operand in self.operands:
            issues.extend(operand.validate())
        issues.extend(self.unit_map.validate())
        for dependency in self.dependencies:
            issues.extend(dependency.validate())
            if dependency.source == self.tisa_id:
                issues.append(f"TISA instruction '{self.tisa_id}' cannot depend on itself")
        if len({dependency.source for dependency in self.dependencies}) != len(self.dependencies):
            issues.append(f"TISA instruction '{self.tisa_id}' dependencies must be unique by source")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tisa_id": self.tisa_id,
            "tile_id": self.tile_id,
            "operator_id": self.operator_id,
            "op_type": self.op_type,
            "operands": [operand.to_dict() for operand in self.operands],
            "unit_map": self.unit_map.to_dict(),
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "attributes": dict(self.attributes),
            "payload_ref": self.payload_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TISAInstruction":
        if not isinstance(payload, Mapping):
            raise ValueError("TISA instruction payload must be an object")
        try:
            value = cls(
                tisa_id=str(payload["tisa_id"]),
                tile_id=str(payload["tile_id"]),
                operator_id=str(payload["operator_id"]),
                op_type=str(payload["op_type"]),
                operands=tuple(TISAOperand.from_dict(item) for item in payload.get("operands", ())),
                unit_map=UnitMap.from_dict(payload["unit_map"]),
                dependencies=tuple(
                    TISADependency.from_dict(item) for item in payload.get("dependencies", ())
                ),
                attributes=payload.get("attributes", {}),
                payload_ref=(str(payload["payload_ref"]) if payload.get("payload_ref") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid TISA instruction payload") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid TISA instruction: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class TISAProgram:
    """A deterministic stream of semantic tile descriptors."""

    program_id: str
    instructions: tuple[TISAInstruction, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.program_id:
            issues.append("TISA program id must not be empty")
        ids = {instruction.tisa_id for instruction in self.instructions}
        instruction_index = {
            instruction.tisa_id: index for index, instruction in enumerate(self.instructions)
        }
        if len(ids) != len(self.instructions):
            issues.append("TISA instruction ids must be unique")
        for instruction in self.instructions:
            issues.extend(instruction.validate())
            for dependency in instruction.dependencies:
                if dependency.source not in ids:
                    issues.append(
                        f"TISA instruction '{instruction.tisa_id}' references unknown dependency '{dependency.source}'"
                    )
                elif instruction_index[dependency.source] >= instruction_index[instruction.tisa_id]:
                    issues.append(
                        f"TISA instruction '{instruction.tisa_id}' dependency '{dependency.source}' "
                        "must precede it in program order"
                    )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TISAProgram":
        if not isinstance(payload, Mapping):
            raise ValueError("TISA program payload must be an object")
        try:
            value = cls(
                program_id=str(payload["program_id"]),
                instructions=tuple(
                    TISAInstruction.from_dict(item) for item in payload.get("instructions", ())
                ),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid TISA program payload") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid TISA program: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class BackendArtifact:
    """TISA descriptors plus backend payload association.

    ``execution_graph`` is intentionally retained for the current analytical
    simulator.  A future device backend can replace it with native binary or
    a cycle model without changing the TISA program contract.
    """

    artifact_id: str
    program: TISAProgram
    execution_graph: ExecutionGraph
    payloads: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    backend: str = "analytical"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    memory_plan: MemoryPlan | None = None
    target_plan: TargetPlan | None = None
    static_control: StaticControlProgram | None = None

    def validate(self) -> tuple[str, ...]:
        issues = list(self.program.validate())
        issues.extend(self.execution_graph.validate())
        instructions = {
            instruction.tisa_id: instruction for instruction in self.program.instructions
        }
        instruction_ids = set(instructions)
        task_owners: dict[str, str] = {}
        for tisa_id, task_ids in self.payloads.items():
            if tisa_id not in instruction_ids:
                issues.append(f"backend payload references unknown TISA instruction '{tisa_id}'")
                continue
            if not task_ids:
                issues.append(f"backend payload for TISA instruction '{tisa_id}' must not be empty")
                continue
            payload_resources: set[str] = set()
            for task_id in task_ids:
                previous_owner = task_owners.get(task_id)
                if previous_owner is not None:
                    issues.append(
                        f"backend task '{task_id}' belongs to both '{previous_owner}' and '{tisa_id}'"
                    )
                else:
                    task_owners[task_id] = tisa_id
                try:
                    task = self.execution_graph.task(task_id)
                except KeyError:
                    issues.append(f"backend payload references unknown task '{task_id}'")
                    continue
                payload_resources.add(task.resource)
                if task.tile_id != instructions[tisa_id].tile_id:
                    issues.append(f"backend payload task '{task_id}' is attached to the wrong TISA tile")
            if len(payload_resources) > 1:
                issues.append(
                    f"backend payload for TISA instruction '{tisa_id}' spans multiple resources: "
                    + ", ".join(sorted(payload_resources))
                )
        for instruction in self.program.instructions:
            if instruction.payload_ref is not None and instruction.tisa_id not in self.payloads:
                issues.append(
                    f"TISA instruction '{instruction.tisa_id}' has no backend payload"
                )
        for task in self.execution_graph.tasks:
            if task.task_id not in task_owners:
                issues.append(f"backend task '{task.task_id}' is not owned by a TISA payload")

        dependency_sources = {
            instruction.tisa_id: {
                dependency.source for dependency in instruction.dependencies
            }
            for instruction in self.program.instructions
        }
        task_by_id = {task.task_id: task for task in self.execution_graph.tasks}
        composite_types = {"softmax", "rmsnorm", "layernorm"}

        # A primitive payload may expose an edge that skips one or more
        # semantic stages.  Validation checks that the owner-to-owner ordering
        # is reachable in the TISA dependency DAG, rather than requiring every
        # backend-internal edge to be repeated as a direct TISA edge.
        dependency_closure: dict[str, set[str]] = {}

        def ancestors(tisa_id: str, active: set[str] | None = None) -> set[str]:
            cached = dependency_closure.get(tisa_id)
            if cached is not None:
                return set(cached)
            visiting = set() if active is None else active
            if tisa_id in visiting:
                return set()
            visiting.add(tisa_id)
            result: set[str] = set()
            for source in dependency_sources.get(tisa_id, set()):
                if source not in instructions:
                    continue
                result.add(source)
                result.update(ancestors(source, visiting))
            visiting.remove(tisa_id)
            dependency_closure[tisa_id] = set(result)
            return result

        for tisa_id in instructions:
            ancestors(tisa_id)
        for task in self.execution_graph.tasks:
            owner = task_owners.get(task.task_id)
            if owner is None:
                continue
            for predecessor_id in task.predecessors:
                predecessor_owner = task_owners.get(predecessor_id)
                if predecessor_owner is None or predecessor_owner == owner:
                    continue

                # Composite lowering may keep a materialized primitive DAG
                # across tiles (for example, row-wise softmax's max and sum
                # finalization).  Those edges are implementation details of
                # the payload and are intentionally not promoted to global
                # TISA dependencies.  The semantic instruction is still
                # ordered at the tile boundary; cross-operator edges remain
                # subject to the strict check below.
                predecessor_task = task_by_id.get(predecessor_id)
                owner_instruction = instructions[owner]
                predecessor_instruction = instructions[predecessor_owner]
                same_operator = (
                    predecessor_task is not None
                    and predecessor_task.operator_id == task.operator_id
                    and task.operator_id == owner_instruction.operator_id
                )
                same_composite_operator = (
                    owner_instruction.attributes.get("semantic_op_type")
                    in composite_types
                    and predecessor_instruction.attributes.get("semantic_op_type")
                    == owner_instruction.attributes.get("semantic_op_type")
                    and same_operator
                )
                if same_composite_operator:
                    continue
                if predecessor_owner not in dependency_closure.get(owner, set()):
                    issues.append(
                        f"backend task edge '{predecessor_id}' -> '{task.task_id}' is not "
                        f"ordered by TISA dependency '{predecessor_owner}' -> '{owner}'"
                )
        if self.memory_plan is not None:
            issues.extend(self.memory_plan.validate())
            planned = {item.buffer_id: item for item in self.memory_plan.buffers}
            for instruction in self.program.instructions:
                for operand in instruction.operands:
                    memory = operand.tile_mem
                    if memory.buffer_id not in planned:
                        issues.append(
                            f"TISA operand '{instruction.tisa_id}:{operand.name}' has no planned buffer"
                        )
                        continue
                    target = planned[memory.buffer_id]
                    if memory.physical_space != target.memory:
                        issues.append(
                            f"TISA operand '{instruction.tisa_id}:{operand.name}' memory "
                            f"'{memory.physical_space}' differs from plan '{target.memory}'"
                        )
                    if memory.allocation_id != target.allocation_id:
                        issues.append(
                            f"TISA operand '{instruction.tisa_id}:{operand.name}' allocation id "
                            "differs from memory plan"
                        )
                    if (
                        memory.offset_bytes is None
                        or memory.size_bytes is None
                        or memory.offset_bytes + memory.size_bytes > target.allocation_bytes
                    ):
                        issues.append(
                            f"TISA operand '{instruction.tisa_id}:{operand.name}' exceeds planned allocation"
                        )
            for task in self.execution_graph.tasks:
                for region in (*task.reads, *task.writes):
                    if region.buffer_id not in planned:
                        issues.append(
                            f"backend region '{task.task_id}:{region.tensor}' has no planned buffer"
                        )
                    elif region.memory != planned[region.buffer_id].memory:
                        issues.append(
                            f"backend region '{task.task_id}:{region.tensor}' memory differs from plan"
                        )
            readable = {"read", "read_write"}
            writable = {"write", "read_write"}
            for instruction in self.program.instructions:
                operands = instruction.operands
                for task_id in self.payloads.get(instruction.tisa_id, ()):
                    task = task_by_id.get(task_id)
                    if task is None:
                        continue
                    for regions, allowed in (
                        (task.reads, readable),
                        (task.writes, writable),
                    ):
                        for region in regions:
                            covered = any(
                                operand.tile_mem.buffer_id == region.buffer_id
                                and operand.tile_mem.physical_space == region.memory
                                and operand.tile_mem.offset_bytes is not None
                                and operand.tile_mem.size_bytes is not None
                                and operand.tile_mem.offset_bytes <= region.offset_bytes
                                and operand.tile_mem.offset_bytes + operand.tile_mem.size_bytes
                                >= region.offset_bytes + region.size_bytes
                                and operand.normalized_access in allowed
                                for operand in operands
                            )
                            if not covered:
                                issues.append(
                                    f"TISA instruction '{instruction.tisa_id}' does not cover "
                                    f"payload access '{task_id}:{region.buffer_id}'"
                                )
        if self.target_plan is not None:
            issues.extend(self.target_plan.validate())
            target_program = self.target_plan.program
            if target_program.to_dict() != self.program.to_dict():
                issues.append("backend target plan program differs from final TISA program")
            if (
                self.memory_plan is not None
                and self.target_plan.memory_plan is not None
                and self.target_plan.memory_plan.to_dict() != self.memory_plan.to_dict()
            ):
                issues.append("backend TargetPlan and MemoryPlan disagree")
        if self.static_control is not None:
            issues.extend(self.static_control.validate(set(instruction_ids)))
            issues.extend(
                self.static_control.validate_dependencies(self.program.instructions)
            )
            if self.static_control.workload_program_id != self.program.program_id:
                issues.append("static control targets a different workload program")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "program": self.program.to_dict(),
            "execution_graph": self.execution_graph.to_dict(),
            "payloads": {key: list(value) for key, value in self.payloads.items()},
            "backend": self.backend,
            "attributes": dict(self.attributes),
            "memory_plan": self.memory_plan.to_dict() if self.memory_plan is not None else None,
            "target_plan": self.target_plan.to_dict() if self.target_plan is not None else None,
            "static_control": (
                self.static_control.to_dict()
                if self.static_control is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackendArtifact":
        if not isinstance(payload, Mapping):
            raise ValueError("backend artifact payload must be an object")
        try:
            from .static import StaticControlProgram
            from .target import TargetPlan

            value = cls(
                artifact_id=str(payload["artifact_id"]),
                program=TISAProgram.from_dict(payload["program"]),
                execution_graph=ExecutionGraph.from_dict(payload["execution_graph"]),
                payloads={
                    str(key): tuple(str(item) for item in items)
                    for key, items in payload.get("payloads", {}).items()
                },
                backend=str(payload.get("backend", "analytical")),
                attributes=payload.get("attributes", {}),
                memory_plan=(
                    MemoryPlan.from_dict(payload["memory_plan"])
                    if payload.get("memory_plan") is not None
                    else None
                ),
                target_plan=(
                    TargetPlan.from_dict(payload["target_plan"])
                    if payload.get("target_plan") is not None
                    else None
                ),
                static_control=(
                    StaticControlProgram.from_dict(payload["static_control"])
                    if payload.get("static_control") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid backend artifact payload") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid backend artifact: " + "; ".join(issues))
        return value
