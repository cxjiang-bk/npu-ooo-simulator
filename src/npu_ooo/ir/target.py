"""Auditable mapping from virtual TISA to one concrete target program."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .memory import MemoryPlan
from .tisa import TISAInstruction, TISAProgram


TARGET_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TargetInstructionPlan:
    """One target instruction and the abstract instruction it implements."""

    abstract_tisa_id: str
    instruction: TISAInstruction
    expansion_kind: str
    abstract_operand_roles: tuple[str, ...] = ()
    route_hop: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.instruction.validate())
        if not self.abstract_tisa_id or not self.expansion_kind:
            issues.append("target instruction mapping identities must not be empty")
        if self.route_hop is not None and self.route_hop < 0:
            issues.append("target instruction route_hop must be non-negative")
        if len(set(self.abstract_operand_roles)) != len(self.abstract_operand_roles):
            issues.append("target instruction abstract operand roles must be unique")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstract_tisa_id": self.abstract_tisa_id,
            "target_tisa_id": self.instruction.tisa_id,
            "expansion_kind": self.expansion_kind,
            "abstract_operand_roles": list(self.abstract_operand_roles),
            "route_hop": self.route_hop,
            "instruction": self.instruction.to_dict(),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetInstructionPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("target instruction plan must be a JSON object")
        try:
            result = cls(
                abstract_tisa_id=str(payload["abstract_tisa_id"]),
                instruction=TISAInstruction.from_dict(payload["instruction"]),
                expansion_kind=str(payload["expansion_kind"]),
                abstract_operand_roles=tuple(
                    str(item) for item in payload.get("abstract_operand_roles", ())
                ),
                route_hop=(
                    int(payload["route_hop"])
                    if payload.get("route_hop") is not None
                    else None
                ),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid target instruction plan") from exc
        if payload.get("target_tisa_id") not in {None, result.instruction.tisa_id}:
            raise ValueError("target instruction id disagrees with embedded instruction")
        issues = result.validate()
        if issues:
            raise ValueError("invalid target instruction plan: " + "; ".join(issues))
        return result


@dataclass(frozen=True)
class TargetPlan:
    """Single source for target stages, operands, routes and provenance."""

    plan_id: str
    abstract_program_id: str
    machine_config_id: str
    machine_topology_hash: str
    instructions: tuple[TargetInstructionPlan, ...]
    symbolic_buffer_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    memory_plan: MemoryPlan | None = None
    schema_version: int = TARGET_PLAN_SCHEMA_VERSION
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def program(self) -> TISAProgram:
        return TISAProgram(
            program_id=str(
                self.attributes.get(
                    "target_program_id", f"{self.abstract_program_id}.target"
                )
            ),
            instructions=tuple(item.instruction for item in self.instructions),
            attributes={
                "source": "target-plan",
                "target_plan_id": self.plan_id,
                "abstract_program_id": self.abstract_program_id,
                "machine_topology_hash": self.machine_topology_hash,
                **dict(self.attributes.get("program_attributes", {})),
            },
        )

    def validate(self, abstract_program: TISAProgram | None = None) -> tuple[str, ...]:
        issues: list[str] = []
        if self.schema_version != TARGET_PLAN_SCHEMA_VERSION:
            issues.append(
                f"target plan schema {self.schema_version} is unsupported; expected "
                f"{TARGET_PLAN_SCHEMA_VERSION}"
            )
        if not self.plan_id or not self.abstract_program_id or not self.machine_config_id:
            issues.append("target plan identities must not be empty")
        if not self.machine_topology_hash:
            issues.append("target plan machine topology hash must not be empty")
        ids = [item.instruction.tisa_id for item in self.instructions]
        if len(set(ids)) != len(ids):
            issues.append("target instruction ids must be unique")
        for item in self.instructions:
            issues.extend(item.validate())
        if self.memory_plan is not None:
            issues.extend(self.memory_plan.validate())
            if self.memory_plan.machine_topology_hash != self.machine_topology_hash:
                issues.append("target and memory plans use different machine topologies")
        issues.extend(self.program.validate())
        if abstract_program is not None:
            if abstract_program.program_id != self.abstract_program_id:
                issues.append("target plan abstract program id does not match input")
            abstract_ids = {item.tisa_id for item in abstract_program.instructions}
            for item in self.instructions:
                if item.abstract_tisa_id not in abstract_ids:
                    issues.append(
                        f"target instruction '{item.instruction.tisa_id}' references unknown "
                        f"abstract instruction '{item.abstract_tisa_id}'"
                    )
            mapped = {item.abstract_tisa_id for item in self.instructions}
            elided = self.attributes.get("elided_abstract_instructions", {})
            if not isinstance(elided, Mapping):
                issues.append("target plan elided_abstract_instructions must be a mapping")
                elided_ids: set[str] = set()
            else:
                elided_ids = {str(item) for item in elided}
                unknown_elided = sorted(elided_ids - abstract_ids)
                if unknown_elided:
                    issues.append(
                        "target plan elides unknown abstract instructions: "
                        + ", ".join(unknown_elided[:8])
                    )
                target_ids = set(ids)
                for abstract_id, record in elided.items():
                    if not isinstance(record, Mapping):
                        issues.append(
                            f"elided abstract instruction '{abstract_id}' must be a mapping"
                        )
                        continue
                    replacements = record.get("resolved_target_tisa_ids", ())
                    if not replacements or any(
                        str(target_id) not in target_ids for target_id in replacements
                    ):
                        issues.append(
                            f"elided abstract instruction '{abstract_id}' has invalid replacement targets"
                        )
            missing = sorted(abstract_ids - mapped - elided_ids)
            if missing:
                issues.append(
                    "target plan does not map abstract instructions: "
                    + ", ".join(missing[:8])
                )
        target_buffers = {
            operand.tile_mem.buffer_id
            for item in self.instructions
            for operand in item.instruction.operands
            if operand.tile_mem.buffer_id is not None
        }
        for symbolic, targets in self.symbolic_buffer_map.items():
            if not symbolic or not targets:
                issues.append("target symbolic buffer map entries must not be empty")
            for target in targets:
                if target not in target_buffers:
                    issues.append(
                        f"symbolic buffer '{symbolic}' maps unknown target buffer '{target}'"
                    )
        return tuple(issues)

    def targets_for(self, abstract_tisa_id: str) -> tuple[TargetInstructionPlan, ...]:
        return tuple(
            item
            for item in self.instructions
            if item.abstract_tisa_id == abstract_tisa_id
        )

    def to_dict(self) -> dict[str, Any]:
        mapping: dict[str, list[str]] = {}
        for item in self.instructions:
            mapping.setdefault(item.abstract_tisa_id, []).append(
                item.instruction.tisa_id
            )
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "abstract_program_id": self.abstract_program_id,
            "machine_config_id": self.machine_config_id,
            "machine_topology_hash": self.machine_topology_hash,
            "abstract_to_target": mapping,
            "symbolic_buffer_map": {
                key: list(value) for key, value in self.symbolic_buffer_map.items()
            },
            "memory_plan": (
                self.memory_plan.to_dict() if self.memory_plan is not None else None
            ),
            "instructions": [item.to_dict() for item in self.instructions],
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("target plan must be a JSON object")
        try:
            result = cls(
                plan_id=str(payload["plan_id"]),
                abstract_program_id=str(payload["abstract_program_id"]),
                machine_config_id=str(payload["machine_config_id"]),
                machine_topology_hash=str(payload["machine_topology_hash"]),
                instructions=tuple(
                    TargetInstructionPlan.from_dict(item)
                    for item in payload.get("instructions", ())
                ),
                symbolic_buffer_map={
                    str(key): tuple(str(item) for item in value)
                    for key, value in payload.get("symbolic_buffer_map", {}).items()
                },
                memory_plan=(
                    MemoryPlan.from_dict(payload["memory_plan"])
                    if payload.get("memory_plan") is not None
                    else None
                ),
                schema_version=int(payload.get("schema_version", 0)),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid target plan") from exc
        expected_mapping: dict[str, list[str]] = {}
        for item in result.instructions:
            expected_mapping.setdefault(item.abstract_tisa_id, []).append(
                item.instruction.tisa_id
            )
        supplied_mapping = payload.get("abstract_to_target")
        if supplied_mapping is not None and supplied_mapping != expected_mapping:
            raise ValueError("target plan abstract_to_target mapping is inconsistent")
        issues = result.validate()
        if issues:
            raise ValueError("invalid target plan: " + "; ".join(issues))
        return result


__all__ = [
    "TARGET_PLAN_SCHEMA_VERSION",
    "TargetInstructionPlan",
    "TargetPlan",
]
