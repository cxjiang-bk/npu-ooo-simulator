"""Tile-level semantic instruction contracts used by the compiler/backend boundary.

The objects in this module deliberately sit between :class:`TileInstance` and
backend-specific :class:`ExecutionTask`.  A TISA instruction is one semantic
tile operation; its backend payload may contain several primitive operations,
but the device scheduler observes the instruction as one run-to-complete unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .execution import AccessType, ExecutionGraph


@dataclass(frozen=True)
class TileMem:
    """Logical memory descriptor used by TISA dependency checks."""

    base: str
    scope: str = "local"
    tensor: str | None = None
    offset_bytes: int | None = None
    size_bytes: int | None = None

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.base:
            issues.append("TISA TileMem base must not be empty")
        if not self.scope:
            issues.append("TISA TileMem scope must not be empty")
        if self.offset_bytes is not None and self.offset_bytes < 0:
            issues.append("TISA TileMem offset_bytes must be non-negative")
        if self.size_bytes is not None and self.size_bytes <= 0:
            issues.append("TISA TileMem size_bytes must be positive")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "scope": self.scope,
            "tensor": self.tensor,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
        }


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
        if not self.tile_shape or any(value <= 0 for value in self.tile_shape):
            issues.append(f"TISA operand '{self.name}' tile_shape must be positive")
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


@dataclass(frozen=True)
class TISADependency:
    """Typed dependency between semantic tile instructions."""

    source: str
    kind: str = "RAW"
    condition: str = "full_region_ready"

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.source:
            issues.append("TISA dependency source must not be empty")
        if self.kind not in {"RAW", "WAR", "WAW"}:
            issues.append(f"TISA dependency kind '{self.kind}' is unsupported")
        if not self.condition:
            issues.append("TISA dependency condition must not be empty")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "kind": self.kind, "condition": self.condition}


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
                if predecessor_owner not in dependency_closure.get(owner, set()):
                    issues.append(
                        f"backend task edge '{predecessor_id}' -> '{task.task_id}' is not "
                        f"ordered by TISA dependency '{predecessor_owner}' -> '{owner}'"
                    )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "program": self.program.to_dict(),
            "execution_graph": self.execution_graph.to_dict(),
            "payloads": {key: list(value) for key, value in self.payloads.items()},
            "backend": self.backend,
            "attributes": dict(self.attributes),
        }
