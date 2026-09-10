"""Compile final target TISA into fixed per-EU streams and synchronization."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
from dataclasses import replace
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    BackendArtifact,
    StaticControlCommand,
    StaticControlProgram,
    StaticEvent,
    StaticInstructionStream,
    TISADependency,
    TISAProgram,
)


def shared_workload_hash(artifact: BackendArtifact) -> str:
    """Hash computation/movement, payload and memory without control policy."""

    payload = {
        "program": artifact.program.to_dict(),
        "execution_graph": artifact.execution_graph.to_dict(),
        "payloads": {key: list(value) for key, value in artifact.payloads.items()},
        "memory_plan": (
            artifact.memory_plan.to_dict() if artifact.memory_plan is not None else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dynamic_control_hash(workload_hash: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": "dynamic-ready-window-v1",
                "workload_hash": workload_hash,
                "static_controls_consumed": False,
                "dependencies": "target_tisa+runtime_binding",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _iteration(tisa_id: str, attributes: Mapping[str, Any]) -> int | None:
    value = attributes.get("iteration")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    match = re.search(r"\.t(\d+)", tisa_id)
    return int(match.group(1)) if match else None


def _buffer_slots(artifact: BackendArtifact, tisa_id: str) -> tuple[str, ...]:
    instruction = next(item for item in artifact.program.instructions if item.tisa_id == tisa_id)
    planned = (
        {item.buffer_id: item for item in artifact.memory_plan.buffers}
        if artifact.memory_plan is not None
        else {}
    )
    values: set[str] = set()
    for operand in instruction.operands:
        memory = operand.tile_mem
        plan = planned.get(str(memory.buffer_id))
        if plan is not None and plan.slot is not None:
            values.add(f"{plan.memory}:slot{plan.slot}")
        elif memory.allocation_id:
            values.add(f"{memory.physical_space}:{memory.allocation_id}")
    return tuple(sorted(values))


def _duration(artifact: BackendArtifact, machine: MachineConfig, tisa_id: str) -> float:
    tasks = {
        item.task_id: item for item in artifact.execution_graph.tasks
    }
    task_ids = artifact.payloads[tisa_id]
    resource = next(
        item.unit_map.unit for item in artifact.program.instructions if item.tisa_id == tisa_id
    )
    return sum(
        float(
            tasks[task_id].duration_cycles
            if tasks[task_id].duration_cycles is not None
            else machine.unit(resource).latency_cycles
        )
        for task_id in task_ids
    )


def _dependency_source_kind(dependency: TISADependency) -> str:
    source = str(dependency.provenance.get("source", ""))
    if dependency.kind in {"BUFFER_REUSE", "WAR", "WAW"} or any(
        token in source for token in ("memory", "allocation", "alias", "reuse")
    ):
        return "buffer_reuse"
    return "true_dependency"


def _list_schedule(
    artifact: BackendArtifact,
    machine: MachineConfig,
) -> tuple[
    dict[str, tuple[str, int]],
    dict[str, float],
    dict[str, float],
]:
    instructions = tuple(artifact.program.instructions)
    by_id = {item.tisa_id: item for item in instructions}
    order = {item.tisa_id: index for index, item in enumerate(instructions)}
    remaining = {
        item.tisa_id: {dependency.source for dependency in item.dependencies}
        for item in instructions
    }
    stream_available = {
        (unit.name, instance): 0.0
        for unit in machine.execution_units
        for instance in range(unit.count)
    }
    assignment: dict[str, tuple[str, int]] = {}
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    while len(assignment) < len(instructions):
        ready = [
            tisa_id
            for tisa_id, sources in remaining.items()
            if tisa_id not in assignment and sources.issubset(assignment)
        ]
        if not ready:
            raise ValueError("cannot static-schedule cyclic TISA dependencies")
        candidates: list[tuple[float, int, int, str, int]] = []
        for tisa_id in ready:
            resource = by_id[tisa_id].unit_map.unit
            dependency_finish = max(
                (finishes[source] for source in remaining[tisa_id]), default=0.0
            )
            for instance in range(machine.unit(resource).count):
                start = max(dependency_finish, stream_available[(resource, instance)])
                candidates.append((start, order[tisa_id], instance, tisa_id, instance))
        start, _program_order, _rank, selected, instance = min(candidates)
        resource = by_id[selected].unit_map.unit
        finish = start + _duration(artifact, machine, selected)
        assignment[selected] = (resource, instance)
        starts[selected] = start
        finishes[selected] = finish
        stream_available[(resource, instance)] = finish
    return assignment, starts, finishes


def build_static_control_program(
    artifact: BackendArtifact,
    machine: MachineConfig,
) -> StaticControlProgram:
    """Generate a compiler list schedule and explicit completion synchronization."""

    workload_hash = shared_workload_hash(artifact)
    assignment, starts, finishes = _list_schedule(artifact, machine)
    instructions = {item.tisa_id: item for item in artifact.program.instructions}
    program_order = {
        item.tisa_id: index for index, item in enumerate(artifact.program.instructions)
    }
    consumers: dict[tuple[str, str], list[tuple[str, TISADependency]]] = defaultdict(list)
    for instruction in artifact.program.instructions:
        for dependency in instruction.dependencies:
            consumers[(dependency.source, dependency.condition)].append(
                (instruction.tisa_id, dependency)
            )
    events: dict[tuple[str, str], StaticEvent] = {}
    for (source, condition), targets in consumers.items():
        source_instruction = instructions[source]
        source_kind = (
            "buffer_reuse"
            if any(_dependency_source_kind(item) == "buffer_reuse" for _target, item in targets)
            else "true_dependency"
        )
        event_id = f"event.g{program_order[source]:04d}.{source}.{condition}"
        events[(source, condition)] = StaticEvent(
            event_id=event_id,
            source_tisa_id=source,
            condition=condition,
            scope="allocation" if source_kind == "buffer_reuse" else "tile",
            generation=program_order[source],
            iteration=_iteration(source, source_instruction.attributes),
            stage_id=int(source_instruction.attributes.get("stage_id", 0)),
            buffer_slots=_buffer_slots(artifact, source),
            consumers=tuple(sorted(target for target, _item in targets)),
            source_kind=source_kind,
            attributes={
                "set_timing": (
                    "matching_partial_feedback"
                    if condition.startswith("payload_ready:")
                    else "execution_done"
                ),
                "consuming_wait": False,
            },
        )

    by_stream: dict[tuple[str, int], list[str]] = defaultdict(list)
    for tisa_id, stream in assignment.items():
        by_stream[stream].append(tisa_id)
    for stream in by_stream:
        by_stream[stream].sort(key=lambda item: (starts[item], program_order[item]))

    streams: list[StaticInstructionStream] = []
    for (resource, instance), tisa_ids in sorted(by_stream.items()):
        stream_id = f"{resource}.{instance}"
        commands: list[StaticControlCommand] = []

        def append(kind: str, **kwargs: Any) -> None:
            order = len(commands)
            commands.append(
                StaticControlCommand(
                    command_id=f"{stream_id}.c{order:05d}",
                    kind=kind,
                    stream_id=stream_id,
                    order=order,
                    **kwargs,
                )
            )

        for tisa_id in tisa_ids:
            instruction = instructions[tisa_id]
            slots = _buffer_slots(artifact, tisa_id)
            true_events: list[str] = []
            fence_events: list[str] = []
            sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for dependency in instruction.dependencies:
                event = events[(dependency.source, dependency.condition)]
                if _dependency_source_kind(dependency) == "buffer_reuse":
                    fence_events.append(event.event_id)
                else:
                    true_events.append(event.event_id)
                sources[event.event_id].append(dependency.to_dict())
            for event_id in sorted(set(true_events)):
                append(
                    "wait",
                    tisa_id=tisa_id,
                    event_ids=(event_id,),
                    scope="tile",
                    source_kind="true_dependency",
                    iteration=_iteration(tisa_id, instruction.attributes),
                    stage_id=int(instruction.attributes.get("stage_id", 0)),
                    buffer_slots=slots,
                    estimated_start=starts[tisa_id],
                    estimated_finish=starts[tisa_id],
                    attributes={"dependencies": sources[event_id]},
                )
            if fence_events:
                append(
                    "fence",
                    tisa_id=tisa_id,
                    event_ids=tuple(sorted(set(fence_events))),
                    scope="allocation",
                    source_kind="buffer_reuse",
                    iteration=_iteration(tisa_id, instruction.attributes),
                    stage_id=int(instruction.attributes.get("stage_id", 0)),
                    buffer_slots=slots,
                    estimated_start=starts[tisa_id],
                    estimated_finish=starts[tisa_id],
                    attributes={
                        "dependencies": [
                            item for event_id in fence_events for item in sources[event_id]
                        ]
                    },
                )
            append(
                "issue",
                tisa_id=tisa_id,
                scope="stream",
                source_kind="static_stream_order",
                iteration=_iteration(tisa_id, instruction.attributes),
                stage_id=int(instruction.attributes.get("stage_id", 0)),
                buffer_slots=slots,
                estimated_start=starts[tisa_id],
                estimated_finish=finishes[tisa_id],
                attributes={
                    "program_order": program_order[tisa_id],
                    "estimated_duration": finishes[tisa_id] - starts[tisa_id],
                },
            )
            for event in sorted(
                (item for item in events.values() if item.source_tisa_id == tisa_id),
                key=lambda item: item.event_id,
            ):
                append(
                    "set",
                    tisa_id=tisa_id,
                    event_ids=(event.event_id,),
                    scope=event.scope,
                    source_kind=event.source_kind,
                    iteration=event.iteration,
                    stage_id=event.stage_id,
                    buffer_slots=event.buffer_slots,
                    estimated_start=finishes[tisa_id],
                    estimated_finish=finishes[tisa_id],
                    attributes={"condition": event.condition},
                )
        streams.append(
            StaticInstructionStream(
                stream_id=stream_id,
                resource=resource,
                instance=instance,
                commands=tuple(commands),
                attributes={
                    "issue_order": tisa_ids,
                    "fixed_order": True,
                },
            )
        )

    control = StaticControlProgram(
        control_id=f"{artifact.program.program_id}.static-control",
        workload_program_id=artifact.program.program_id,
        workload_hash=workload_hash,
        streams=tuple(streams),
        events=tuple(sorted(events.values(), key=lambda item: item.event_id)),
        attributes={
            "compiler": "resource_constrained_list_schedule_v1",
            "schedule_estimate_cycles": max(finishes.values(), default=0.0),
            "timing_estimate_source": "backend_payload_declared_duration",
            "timing_estimate_is_oracle": False,
            "control_cost_assumption": "configured_at_simulation",
            "initial_slot_state": "allocated_but_no_dynamic_value",
            "wait_consumes_event": False,
            "finalization": "invocation completes after all shared workload instructions",
            "dynamic_control_hash": dynamic_control_hash(workload_hash),
        },
    )
    issues = control.validate(set(instructions))
    issues += validate_static_control_dependencies(artifact.program, control)
    if issues:
        raise ValueError("invalid compiler static control: " + "; ".join(issues))
    return control


def validate_static_control_dependencies(
    program: TISAProgram,
    control: StaticControlProgram,
) -> tuple[str, ...]:
    """Use shared semantic dependencies only as a compile-time correctness oracle."""
    return control.validate_dependencies(program.instructions)


def attach_static_control(
    artifact: BackendArtifact,
    machine: MachineConfig,
) -> BackendArtifact:
    control = build_static_control_program(artifact, machine)
    return replace(
        artifact,
        static_control=control,
        attributes={
            **dict(artifact.attributes),
            "shared_workload_hash": control.workload_hash,
            "static_control_hash": control.control_hash,
            "dynamic_control_hash": control.attributes["dynamic_control_hash"],
            "static_control_schema": control.schema_version,
        },
    )


__all__ = [
    "attach_static_control",
    "build_static_control_program",
    "dynamic_control_hash",
    "shared_workload_hash",
    "validate_static_control_dependencies",
]
