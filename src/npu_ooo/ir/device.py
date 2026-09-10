"""Bound device-facing TISA descriptor and feedback contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .runtime import RuntimeOperandBinding
from .static import StaticControlProgram
from .tisa import TISAInstruction


DEVICE_PROGRAM_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompletionToken:
    invocation_id: str
    tisa_id: str
    kind: str = "complete"

    @property
    def token_id(self) -> str:
        return f"{self.invocation_id}::{self.tisa_id}::{self.kind}"

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.invocation_id or not self.tisa_id or not self.kind:
            issues.append("completion token fields must not be empty")
        return tuple(issues)

    def to_dict(self) -> dict[str, str]:
        return {
            "token_id": self.token_id,
            "invocation_id": self.invocation_id,
            "tisa_id": self.tisa_id,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompletionToken":
        if not isinstance(payload, Mapping):
            raise ValueError("completion token must be a JSON object")
        try:
            result = cls(
                invocation_id=str(payload["invocation_id"]),
                tisa_id=str(payload["tisa_id"]),
                kind=str(payload.get("kind", "complete")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid completion token") from exc
        if payload.get("token_id") not in {None, result.token_id}:
            raise ValueError("completion token id is inconsistent")
        issues = result.validate()
        if issues:
            raise ValueError("invalid completion token: " + "; ".join(issues))
        return result


@dataclass(frozen=True)
class BoundDependency:
    source: CompletionToken
    kind: str
    condition: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.source.validate())
        if not self.kind or not self.condition:
            issues.append("bound dependency kind and condition must not be empty")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "kind": self.kind,
            "condition": self.condition,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoundDependency":
        if not isinstance(payload, Mapping):
            raise ValueError("bound dependency must be a JSON object")
        try:
            result = cls(
                source=CompletionToken.from_dict(payload["source"]),
                kind=str(payload["kind"]),
                condition=str(payload["condition"]),
                provenance=payload.get("provenance", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid bound dependency") from exc
        issues = result.validate()
        if issues:
            raise ValueError("invalid bound dependency: " + "; ".join(issues))
        return result


@dataclass(frozen=True)
class BoundTISADescriptor:
    """One loaded TISA instruction with invocation-specific operands."""

    descriptor_id: str
    invocation_id: str
    program_id: str
    artifact_id: str
    instruction: TISAInstruction
    operands: tuple[RuntimeOperandBinding, ...]
    dependencies: tuple[BoundDependency, ...]
    completion_token: CompletionToken
    payload_handle: str
    program_order: int
    submission_order: int
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.instruction.validate())
        issues.extend(self.completion_token.validate())
        if not all(
            (self.descriptor_id, self.invocation_id, self.program_id, self.artifact_id, self.payload_handle)
        ):
            issues.append("bound descriptor identities must not be empty")
        if self.completion_token.invocation_id != self.invocation_id:
            issues.append("descriptor completion token invocation does not match")
        if self.completion_token.tisa_id != self.instruction.tisa_id:
            issues.append("descriptor completion token instruction does not match")
        if self.program_order < 0 or self.submission_order < 0:
            issues.append("descriptor orders must be non-negative")
        expected_operands = {item.name for item in self.instruction.operands}
        actual_operands = {item.operand_name for item in self.operands}
        if expected_operands != actual_operands:
            issues.append("bound descriptor operands do not cover TISA operands")
        if len(actual_operands) != len(self.operands):
            issues.append("bound descriptor operand names must be unique")
        for operand in self.operands:
            issues.extend(operand.validate())
            if operand.tisa_id != self.instruction.tisa_id:
                issues.append("bound operand references the wrong TISA instruction")
        expected_dependencies = {
            (item.source, item.kind, item.condition)
            for item in self.instruction.dependencies
        }
        actual_dependencies = {
            (item.source.tisa_id, item.kind, item.condition)
            for item in self.dependencies
        }
        if not expected_dependencies.issubset(actual_dependencies):
            issues.append("bound descriptor drops a TISA semantic dependency")
        for dependency in self.dependencies:
            issues.extend(dependency.validate())
            if dependency.source.invocation_id != self.invocation_id:
                issues.append("bound dependency crosses invocation without a sequence token")
            if (
                (dependency.source.tisa_id, dependency.kind, dependency.condition)
                not in expected_dependencies
                and dependency.provenance.get("source") != "runtime_binding_alias"
            ):
                issues.append("extra bound dependency lacks runtime alias provenance")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_id": self.descriptor_id,
            "invocation_id": self.invocation_id,
            "program_id": self.program_id,
            "artifact_id": self.artifact_id,
            "instruction": self.instruction.to_dict(),
            "operands": [item.to_dict() for item in self.operands],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "completion_token": self.completion_token.to_dict(),
            "payload_handle": self.payload_handle,
            "program_order": self.program_order,
            "submission_order": self.submission_order,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoundTISADescriptor":
        if not isinstance(payload, Mapping):
            raise ValueError("bound descriptor must be a JSON object")
        try:
            result = cls(
                descriptor_id=str(payload["descriptor_id"]),
                invocation_id=str(payload["invocation_id"]),
                program_id=str(payload["program_id"]),
                artifact_id=str(payload["artifact_id"]),
                instruction=TISAInstruction.from_dict(payload["instruction"]),
                operands=tuple(
                    RuntimeOperandBinding.from_dict(item)
                    for item in payload.get("operands", ())
                ),
                dependencies=tuple(
                    BoundDependency.from_dict(item)
                    for item in payload.get("dependencies", ())
                ),
                completion_token=CompletionToken.from_dict(payload["completion_token"]),
                payload_handle=str(payload["payload_handle"]),
                program_order=int(payload["program_order"]),
                submission_order=int(payload["submission_order"]),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid bound descriptor") from exc
        issues = result.validate()
        if issues:
            raise ValueError("invalid bound descriptor: " + "; ".join(issues))
        return result


@dataclass(frozen=True)
class DescriptorEnvelope:
    descriptor_id: str
    chunk_id: str
    queue: str
    arrival_cycle: float
    chunk_order: int
    descriptor_order: int

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.descriptor_id or not self.chunk_id or not self.queue:
            issues.append("descriptor envelope identities must not be empty")
        if self.arrival_cycle < 0 or self.chunk_order < 0 or self.descriptor_order < 0:
            issues.append("descriptor envelope cycle/orders must be non-negative")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_id": self.descriptor_id,
            "chunk_id": self.chunk_id,
            "queue": self.queue,
            "arrival_cycle": self.arrival_cycle,
            "chunk_order": self.chunk_order,
            "descriptor_order": self.descriptor_order,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DescriptorEnvelope":
        if not isinstance(payload, Mapping):
            raise ValueError("descriptor envelope must be a JSON object")
        try:
            result = cls(
                descriptor_id=str(payload["descriptor_id"]),
                chunk_id=str(payload["chunk_id"]),
                queue=str(payload["queue"]),
                arrival_cycle=float(payload["arrival_cycle"]),
                chunk_order=int(payload["chunk_order"]),
                descriptor_order=int(payload["descriptor_order"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid descriptor envelope") from exc
        issues = result.validate()
        if issues:
            raise ValueError("invalid descriptor envelope: " + "; ".join(issues))
        return result


@dataclass(frozen=True)
class StaticScheduleEntry:
    tisa_id: str
    order: int
    resource: str
    dependency_tokens: tuple[str, ...] = ()
    reservation: str = "resource_when_available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tisa_id": self.tisa_id,
            "order": self.order,
            "resource": self.resource,
            "dependency_tokens": list(self.dependency_tokens),
            "reservation": self.reservation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticScheduleEntry":
        return cls(
            tisa_id=str(payload["tisa_id"]),
            order=int(payload["order"]),
            resource=str(payload["resource"]),
            dependency_tokens=tuple(
                str(item) for item in payload.get("dependency_tokens", ())
            ),
            reservation=str(payload.get("reservation", "resource_when_available")),
        )


@dataclass(frozen=True)
class StaticSchedulePlan:
    plan_id: str
    program_id: str
    entries: tuple[StaticScheduleEntry, ...]
    policy: str = "runtime_fixed_submission_overlap"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.plan_id or not self.program_id or not self.policy:
            issues.append("static schedule plan identities must not be empty")
        ids = [item.tisa_id for item in self.entries]
        if len(set(ids)) != len(ids):
            issues.append("static schedule TISA ids must be unique")
        if [item.order for item in self.entries] != list(range(len(self.entries))):
            issues.append("static schedule orders must be contiguous")
        if any(not item.resource or not item.reservation for item in self.entries):
            issues.append("static schedule resource/reservation must not be empty")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "program_id": self.program_id,
            "policy": self.policy,
            "entries": [item.to_dict() for item in self.entries],
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticSchedulePlan":
        result = cls(
            plan_id=str(payload["plan_id"]),
            program_id=str(payload["program_id"]),
            entries=tuple(
                StaticScheduleEntry.from_dict(item)
                for item in payload.get("entries", ())
            ),
            policy=str(payload.get("policy", "runtime_fixed_submission_overlap")),
            attributes=payload.get("attributes", {}),
        )
        issues = result.validate()
        if issues:
            raise ValueError("invalid static schedule plan: " + "; ".join(issues))
        return result


@dataclass(frozen=True)
class LoadedDeviceProgram:
    program_id: str
    artifact_id: str
    invocation_id: str
    descriptors: tuple[BoundTISADescriptor, ...]
    envelopes: tuple[DescriptorEnvelope, ...]
    launch_latency_cycles: float = 0.0
    synchronization_cycles: float = 0.0
    static_schedule: StaticSchedulePlan | None = None
    static_control: StaticControlProgram | None = None
    schema_version: int = DEVICE_PROGRAM_SCHEMA_VERSION
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.schema_version != DEVICE_PROGRAM_SCHEMA_VERSION:
            issues.append("loaded device program schema is unsupported")
        if not self.program_id or not self.artifact_id or not self.invocation_id:
            issues.append("loaded device program identities must not be empty")
        descriptor_ids = [item.descriptor_id for item in self.descriptors]
        if len(set(descriptor_ids)) != len(descriptor_ids):
            issues.append("loaded descriptor ids must be unique")
        envelope_ids = [item.descriptor_id for item in self.envelopes]
        if set(envelope_ids) != set(descriptor_ids) or len(envelope_ids) != len(descriptor_ids):
            issues.append("descriptor envelopes must cover every descriptor exactly once")
        known_tisa = {
            item.instruction.tisa_id for item in self.descriptors
        }
        for descriptor in self.descriptors:
            issues.extend(descriptor.validate())
            if descriptor.program_id != self.program_id or descriptor.artifact_id != self.artifact_id:
                issues.append("loaded descriptor program/artifact identity differs")
            if descriptor.invocation_id != self.invocation_id:
                issues.append("loaded descriptor invocation differs")
            for dependency in descriptor.dependencies:
                if dependency.source.tisa_id not in known_tisa:
                    issues.append("bound descriptor references an unknown completion token")
        for envelope in self.envelopes:
            issues.extend(envelope.validate())
        if self.launch_latency_cycles < 0 or self.synchronization_cycles < 0:
            issues.append("loaded runtime latency values must be non-negative")
        if self.static_schedule is not None:
            issues.extend(self.static_schedule.validate())
            if self.static_schedule.program_id != self.program_id:
                issues.append("static schedule program id does not match loaded program")
            if {item.tisa_id for item in self.static_schedule.entries} != {
                item.instruction.tisa_id for item in self.descriptors
            }:
                issues.append("static schedule must cover every loaded descriptor")
        if self.static_control is not None:
            issues.extend(self.static_control.validate(known_tisa))
            if self.static_control.workload_program_id != self.program_id:
                issues.append("loaded static control program id does not match")
        return tuple(issues)

    def descriptor(self, descriptor_id: str) -> BoundTISADescriptor:
        for item in self.descriptors:
            if item.descriptor_id == descriptor_id:
                return item
        raise KeyError(descriptor_id)

    def descriptor_for_tisa(self, tisa_id: str) -> BoundTISADescriptor:
        for item in self.descriptors:
            if item.instruction.tisa_id == tisa_id:
                return item
        raise KeyError(tisa_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "artifact_id": self.artifact_id,
            "invocation_id": self.invocation_id,
            "descriptors": [item.to_dict() for item in self.descriptors],
            "envelopes": [item.to_dict() for item in self.envelopes],
            "launch_latency_cycles": self.launch_latency_cycles,
            "synchronization_cycles": self.synchronization_cycles,
            "static_schedule": (
                self.static_schedule.to_dict()
                if self.static_schedule is not None
                else None
            ),
            "static_control": (
                self.static_control.to_dict()
                if self.static_control is not None
                else None
            ),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoadedDeviceProgram":
        if not isinstance(payload, Mapping):
            raise ValueError("loaded device program must be a JSON object")
        try:
            result = cls(
                program_id=str(payload["program_id"]),
                artifact_id=str(payload["artifact_id"]),
                invocation_id=str(payload["invocation_id"]),
                descriptors=tuple(
                    BoundTISADescriptor.from_dict(item)
                    for item in payload.get("descriptors", ())
                ),
                envelopes=tuple(
                    DescriptorEnvelope.from_dict(item)
                    for item in payload.get("envelopes", ())
                ),
                launch_latency_cycles=float(payload.get("launch_latency_cycles", 0.0)),
                synchronization_cycles=float(payload.get("synchronization_cycles", 0.0)),
                static_schedule=(
                    StaticSchedulePlan.from_dict(payload["static_schedule"])
                    if payload.get("static_schedule") is not None
                    else None
                ),
                static_control=(
                    StaticControlProgram.from_dict(payload["static_control"])
                    if payload.get("static_control") is not None
                    else None
                ),
                schema_version=int(payload.get("schema_version", 0)),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid loaded device program") from exc
        issues = result.validate()
        if issues:
            raise ValueError("invalid loaded device program: " + "; ".join(issues))
        return result


__all__ = [
    "DEVICE_PROGRAM_SCHEMA_VERSION",
    "BoundDependency",
    "BoundTISADescriptor",
    "CompletionToken",
    "DescriptorEnvelope",
    "LoadedDeviceProgram",
    "StaticScheduleEntry",
    "StaticSchedulePlan",
]
