"""Compiler-owned static stream and synchronization contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


STATIC_CONTROL_SCHEMA_VERSION = 1
CONTROL_KINDS = frozenset({"issue", "set", "wait", "fence"})


@dataclass(frozen=True)
class StaticEvent:
    event_id: str
    source_tisa_id: str
    condition: str
    scope: str
    generation: int
    invocation_scope: str = "per_invocation"
    iteration: int | None = None
    stage_id: int | None = None
    buffer_slots: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()
    source_kind: str = "semantic_dependency"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not all((self.event_id, self.source_tisa_id, self.condition, self.scope)):
            issues.append("static event identities must not be empty")
        if self.generation < 0:
            issues.append("static event generation must be non-negative")
        if self.invocation_scope != "per_invocation":
            issues.append("static events must be scoped per invocation")
        if len(set(self.consumers)) != len(self.consumers):
            issues.append("static event consumers must be unique")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_tisa_id": self.source_tisa_id,
            "condition": self.condition,
            "scope": self.scope,
            "generation": self.generation,
            "invocation_scope": self.invocation_scope,
            "iteration": self.iteration,
            "stage_id": self.stage_id,
            "buffer_slots": list(self.buffer_slots),
            "consumers": list(self.consumers),
            "source_kind": self.source_kind,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticEvent":
        try:
            value = cls(
                event_id=str(payload["event_id"]),
                source_tisa_id=str(payload["source_tisa_id"]),
                condition=str(payload["condition"]),
                scope=str(payload["scope"]),
                generation=int(payload["generation"]),
                invocation_scope=str(payload.get("invocation_scope", "per_invocation")),
                iteration=(
                    int(payload["iteration"])
                    if payload.get("iteration") is not None
                    else None
                ),
                stage_id=(
                    int(payload["stage_id"])
                    if payload.get("stage_id") is not None
                    else None
                ),
                buffer_slots=tuple(str(item) for item in payload.get("buffer_slots", ())),
                consumers=tuple(str(item) for item in payload.get("consumers", ())),
                source_kind=str(payload.get("source_kind", "semantic_dependency")),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid static event") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid static event: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class StaticControlCommand:
    command_id: str
    kind: str
    stream_id: str
    order: int
    tisa_id: str | None = None
    event_ids: tuple[str, ...] = ()
    scope: str = "stream"
    source_kind: str = "static_stream_order"
    iteration: int | None = None
    stage_id: int | None = None
    buffer_slots: tuple[str, ...] = ()
    estimated_start: float | None = None
    estimated_finish: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.command_id or not self.stream_id or self.kind not in CONTROL_KINDS:
            issues.append("static command identity/kind is invalid")
        if self.order < 0:
            issues.append("static command order must be non-negative")
        if self.kind == "issue" and (self.tisa_id is None or self.event_ids):
            issues.append("static issue command requires tisa_id and no events")
        if self.kind == "set" and (
            self.tisa_id is None or len(self.event_ids) != 1
        ):
            issues.append("static set command requires producer tisa_id and one event")
        if self.kind == "wait" and len(self.event_ids) != 1:
            issues.append("static wait command requires exactly one event")
        if self.kind == "fence" and not self.event_ids:
            issues.append("static fence command requires at least one event")
        if self.estimated_start is not None and self.estimated_start < 0:
            issues.append("static command estimated_start must be non-negative")
        if (
            self.estimated_finish is not None
            and self.estimated_start is not None
            and self.estimated_finish < self.estimated_start
        ):
            issues.append("static command estimated finish precedes start")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "kind": self.kind,
            "stream_id": self.stream_id,
            "order": self.order,
            "tisa_id": self.tisa_id,
            "event_ids": list(self.event_ids),
            "scope": self.scope,
            "source_kind": self.source_kind,
            "iteration": self.iteration,
            "stage_id": self.stage_id,
            "buffer_slots": list(self.buffer_slots),
            "estimated_start": self.estimated_start,
            "estimated_finish": self.estimated_finish,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticControlCommand":
        try:
            value = cls(
                command_id=str(payload["command_id"]),
                kind=str(payload["kind"]),
                stream_id=str(payload["stream_id"]),
                order=int(payload["order"]),
                tisa_id=(str(payload["tisa_id"]) if payload.get("tisa_id") else None),
                event_ids=tuple(str(item) for item in payload.get("event_ids", ())),
                scope=str(payload.get("scope", "stream")),
                source_kind=str(payload.get("source_kind", "static_stream_order")),
                iteration=(
                    int(payload["iteration"])
                    if payload.get("iteration") is not None
                    else None
                ),
                stage_id=(
                    int(payload["stage_id"])
                    if payload.get("stage_id") is not None
                    else None
                ),
                buffer_slots=tuple(str(item) for item in payload.get("buffer_slots", ())),
                estimated_start=(
                    float(payload["estimated_start"])
                    if payload.get("estimated_start") is not None
                    else None
                ),
                estimated_finish=(
                    float(payload["estimated_finish"])
                    if payload.get("estimated_finish") is not None
                    else None
                ),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid static control command") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid static control command: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class StaticInstructionStream:
    stream_id: str
    resource: str
    instance: int
    commands: tuple[StaticControlCommand, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.stream_id or not self.resource or self.instance < 0:
            issues.append("static stream identity/resource/instance is invalid")
        if [item.order for item in self.commands] != list(range(len(self.commands))):
            issues.append("static stream command orders must be contiguous")
        if any(item.stream_id != self.stream_id for item in self.commands):
            issues.append("static stream contains a command owned by another stream")
        for command in self.commands:
            issues.extend(command.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "resource": self.resource,
            "instance": self.instance,
            "commands": [item.to_dict() for item in self.commands],
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticInstructionStream":
        try:
            value = cls(
                stream_id=str(payload["stream_id"]),
                resource=str(payload["resource"]),
                instance=int(payload.get("instance", 0)),
                commands=tuple(
                    StaticControlCommand.from_dict(item)
                    for item in payload.get("commands", ())
                ),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid static instruction stream") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid static stream: " + "; ".join(issues))
        return value


@dataclass(frozen=True)
class StaticControlProgram:
    control_id: str
    workload_program_id: str
    workload_hash: str
    streams: tuple[StaticInstructionStream, ...]
    events: tuple[StaticEvent, ...]
    policy: str = "compiler_list_schedule_v1"
    schema_version: int = STATIC_CONTROL_SCHEMA_VERSION
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def control_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def validate(self, workload_tisa_ids: set[str] | None = None) -> tuple[str, ...]:
        issues: list[str] = []
        if self.schema_version != STATIC_CONTROL_SCHEMA_VERSION:
            issues.append("static control schema is unsupported")
        if not self.control_id or not self.workload_program_id or not self.workload_hash:
            issues.append("static control identities must not be empty")
        stream_ids = [item.stream_id for item in self.streams]
        event_ids = [item.event_id for item in self.events]
        if len(set(stream_ids)) != len(stream_ids):
            issues.append("static stream ids must be unique")
        if len(set(event_ids)) != len(event_ids):
            issues.append("static event ids must be unique")
        event_keys = [(item.source_tisa_id, item.condition) for item in self.events]
        if len(set(event_keys)) != len(event_keys):
            issues.append("static source/condition event keys must be unique")
        command_ids = [
            command.command_id for stream in self.streams for command in stream.commands
        ]
        if len(set(command_ids)) != len(command_ids):
            issues.append("static command ids must be unique")
        for stream in self.streams:
            issues.extend(stream.validate())
        for event in self.events:
            issues.extend(event.validate())
        known_events = set(event_ids)
        referenced_events = {
            event_id
            for stream in self.streams
            for command in stream.commands
            for event_id in command.event_ids
        }
        if not referenced_events.issubset(known_events):
            issues.append("static commands reference unknown events")
        issues_tisa = [
            command.tisa_id
            for stream in self.streams
            for command in stream.commands
            if command.kind == "issue" and command.tisa_id is not None
        ]
        if len(set(issues_tisa)) != len(issues_tisa):
            issues.append("static control issues a TISA instruction more than once")
        if workload_tisa_ids is not None and set(issues_tisa) != workload_tisa_ids:
            issues.append("static control issue commands do not cover shared workload")
        return tuple(issues)

    def validate_dependencies(self, instructions: tuple[Any, ...]) -> tuple[str, ...]:
        """Check that every shared workload edge is enforced explicitly."""

        issue_position: dict[str, tuple[str, int]] = {}
        waits_before: dict[str, set[str]] = {}
        sets_after: dict[str, set[str]] = {}
        for stream in self.streams:
            for command in stream.commands:
                if command.kind == "issue" and command.tisa_id is not None:
                    issue_position[command.tisa_id] = (stream.stream_id, command.order)
                elif command.kind in {"wait", "fence"} and command.tisa_id is not None:
                    waits_before.setdefault(command.tisa_id, set()).update(
                        command.event_ids
                    )
                elif command.kind == "set" and command.tisa_id is not None:
                    sets_after.setdefault(command.tisa_id, set()).update(command.event_ids)
        event_by_source_condition = {
            (item.source_tisa_id, item.condition): item for item in self.events
        }
        issues: list[str] = []
        for instruction in instructions:
            for dependency in instruction.dependencies:
                event = event_by_source_condition.get(
                    (dependency.source, dependency.condition)
                )
                if event is None:
                    issues.append(
                        f"static control lacks event for "
                        f"{dependency.source}->{instruction.tisa_id}"
                    )
                    continue
                if event.event_id not in waits_before.get(instruction.tisa_id, set()):
                    issues.append(
                        f"static control lacks wait/fence for "
                        f"{dependency.source}->{instruction.tisa_id}"
                    )
                if event.event_id not in sets_after.get(dependency.source, set()):
                    issues.append(f"static control event '{event.event_id}' is never set")
                source_position = issue_position.get(dependency.source)
                target_position = issue_position.get(instruction.tisa_id)
                if (
                    source_position is not None
                    and target_position is not None
                    and source_position[0] == target_position[0]
                    and source_position[1] >= target_position[1]
                ):
                    issues.append(
                        "static stream orders dependency target before source: "
                        f"{dependency.source}->{instruction.tisa_id}"
                    )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "control_id": self.control_id,
            "workload_program_id": self.workload_program_id,
            "workload_hash": self.workload_hash,
            "policy": self.policy,
            "streams": [item.to_dict() for item in self.streams],
            "events": [item.to_dict() for item in self.events],
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticControlProgram":
        try:
            value = cls(
                schema_version=int(payload.get("schema_version", 1)),
                control_id=str(payload["control_id"]),
                workload_program_id=str(payload["workload_program_id"]),
                workload_hash=str(payload["workload_hash"]),
                policy=str(payload.get("policy", "compiler_list_schedule_v1")),
                streams=tuple(
                    StaticInstructionStream.from_dict(item)
                    for item in payload.get("streams", ())
                ),
                events=tuple(
                    StaticEvent.from_dict(item) for item in payload.get("events", ())
                ),
                attributes=payload.get("attributes", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid static control program") from exc
        issues = value.validate()
        if issues:
            raise ValueError("invalid static control program: " + "; ".join(issues))
        return value


__all__ = [
    "CONTROL_KINDS",
    "STATIC_CONTROL_SCHEMA_VERSION",
    "StaticControlCommand",
    "StaticControlProgram",
    "StaticEvent",
    "StaticInstructionStream",
]
