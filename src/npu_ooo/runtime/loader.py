"""Join a loaded compile artifact with one RuntimeSubmission."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field, replace

from npu_ooo.ir import (
    BackendArtifact,
    BoundDependency,
    BoundTISADescriptor,
    CompletionToken,
    DescriptorEnvelope,
    LoadedDeviceProgram,
    RuntimeSubmission,
    RuntimeOperandBinding,
    StaticScheduleEntry,
    StaticSchedulePlan,
)


def load_device_program(
    artifact: BackendArtifact,
    submission: RuntimeSubmission,
    *,
    invocation_id: str | None = None,
) -> LoadedDeviceProgram:
    """Resolve scheduler-facing descriptors without exposing compiler state."""

    artifact_issues = artifact.validate()
    submission_issues = submission.validate(artifact.program)
    if artifact_issues or submission_issues:
        raise ValueError(
            "cannot load device program: "
            + "; ".join((*artifact_issues, *submission_issues))
        )
    if submission.artifact_id not in {None, artifact.artifact_id}:
        raise ValueError("runtime submission references a different backend artifact")
    invocation = invocation_id or submission.submission_id
    instructions = {
        item.tisa_id: item for item in artifact.program.instructions
    }
    program_order = {
        item.tisa_id: index for index, item in enumerate(artifact.program.instructions)
    }
    flattened = [
        (chunk, tisa_id)
        for chunk in submission.commands
        for tisa_id in chunk.tisa_ids
    ]
    submission_order = {
        tisa_id: index for index, (_chunk, tisa_id) in enumerate(flattened)
    }
    operands = {
        tisa_id: tuple(
            item for item in submission.operands if item.tisa_id == tisa_id
        )
        for tisa_id in instructions
    }
    feedback_conditions = {
        tisa_id: sorted(
            {
                dependency.condition
                for instruction in artifact.program.instructions
                for dependency in instruction.dependencies
                if dependency.source == tisa_id
                and dependency.condition.startswith("payload_ready:")
            }
        )
        for tisa_id in instructions
    }
    descriptors = tuple(
        BoundTISADescriptor(
            descriptor_id=f"{invocation}::{instruction.tisa_id}",
            invocation_id=invocation,
            program_id=artifact.program.program_id,
            artifact_id=artifact.artifact_id,
            instruction=instruction,
            operands=operands[instruction.tisa_id],
            dependencies=tuple(
                BoundDependency(
                    source=CompletionToken(invocation, dependency.source),
                    kind=dependency.kind,
                    condition=dependency.condition,
                    provenance=dependency.provenance,
                )
                for dependency in instruction.dependencies
            ),
            completion_token=CompletionToken(invocation, instruction.tisa_id),
            payload_handle=f"{artifact.artifact_id}::{instruction.tisa_id}",
            program_order=program_order[instruction.tisa_id],
            submission_order=submission_order[instruction.tisa_id],
            attributes={
                "runtime_submission_id": submission.submission_id,
                "runtime_policy": submission.policy,
                "feedback_conditions": feedback_conditions[instruction.tisa_id],
            },
        )
        for instruction in artifact.program.instructions
    )
    descriptors = _add_runtime_alias_dependencies(descriptors)
    descriptor_by_tisa = {
        item.instruction.tisa_id: item for item in descriptors
    }
    for descriptor in descriptors:
        for dependency in descriptor.dependencies:
            source = descriptor_by_tisa[dependency.source.tisa_id]
            if source.submission_order >= descriptor.submission_order:
                raise ValueError(
                    f"runtime descriptor order sends '{descriptor.instruction.tisa_id}' "
                    f"before bound dependency '{source.instruction.tisa_id}'; regenerate "
                    "submission order with physical alias dependencies"
                )
    cursor = 0.0
    envelope_by_tisa: dict[str, DescriptorEnvelope] = {}
    for chunk in submission.commands:
        start = max(cursor, chunk.availability_cycle)
        finish = start + submission.launch_latency_cycles
        cursor = finish
        for tisa_id in chunk.tisa_ids:
            descriptor_id = f"{invocation}::{tisa_id}"
            envelope_by_tisa[tisa_id] = DescriptorEnvelope(
                descriptor_id=descriptor_id,
                chunk_id=chunk.chunk_id,
                queue=chunk.queue,
                arrival_cycle=finish,
                chunk_order=chunk.submission_order,
                descriptor_order=submission_order[tisa_id],
            )
    loaded = LoadedDeviceProgram(
        program_id=artifact.program.program_id,
        artifact_id=artifact.artifact_id,
        invocation_id=invocation,
        descriptors=descriptors,
        envelopes=tuple(
            envelope_by_tisa[item.tisa_id]
            for item in artifact.program.instructions
        ),
        launch_latency_cycles=submission.launch_latency_cycles,
        synchronization_cycles=submission.synchronization_cycles,
        static_schedule=_static_schedule(artifact.program.program_id, descriptors),
        attributes={
            "runtime_submission_id": submission.submission_id,
            "runtime_policy": submission.policy,
            "command_chunk_count": len(submission.commands),
            "runtime_submit_cycles": cursor,
        },
    )
    issues = loaded.validate()
    if issues:
        raise ValueError("loaded device program is invalid: " + "; ".join(issues))
    return loaded


def _add_runtime_alias_dependencies(
    descriptors: tuple[BoundTISADescriptor, ...],
) -> tuple[BoundTISADescriptor, ...]:
    """Convert physical aliases into a minimal writer/reader frontier.

    Each disjoint address segment tracks its last writer and the readers since
    that writer.  A read waits only for the last writer; a write waits for the
    current reader frontier, or for the last writer when no reader intervened.
    This preserves RAW/WAR/WAW ordering without connecting every younger
    descriptor to every historical overlapping access.
    """

    ordered = tuple(sorted(descriptors, key=lambda item: item.program_order))
    scope_has_missing_identity: set[str] = set()
    for descriptor in ordered:
        for operand in descriptor.operands:
            identity = operand.attributes.get("allocation_identity")
            if identity is None:
                scope_has_missing_identity.add(operand.physical_scope)

    def alias_key(operand: RuntimeOperandBinding) -> tuple[str, object | None]:
        identity = operand.attributes.get("allocation_identity")
        if operand.physical_scope in scope_has_missing_identity:
            # A missing identity has the old contract of matching by physical
            # scope/address alone. Collapse the scope conservatively when
            # identified and unidentified bindings are mixed.
            identity = None
        elif identity is not None:
            identity = str(identity)
        return operand.physical_scope, identity

    boundaries: dict[tuple[str, object | None], set[int]] = defaultdict(set)
    for descriptor in ordered:
        for operand in descriptor.operands:
            key = alias_key(operand)
            boundaries[key].update((operand.address, operand.address + operand.size_bytes))
    points = {key: tuple(sorted(values)) for key, values in boundaries.items()}
    frontiers = {
        key: [_AliasFrontier() for _ in range(max(0, len(values) - 1))]
        for key, values in points.items()
    }

    updated: list[BoundTISADescriptor] = []
    priority = {"RAW": 0, "WAR": 1, "WAW": 2}
    for descriptor in ordered:
        dependencies = {
            (item.source.tisa_id, item.kind, item.condition): item
            for item in descriptor.dependencies
        }
        alias_by_source: dict[str, BoundDependency] = {}

        def add_alias(
            source: BoundTISADescriptor | None,
            kind: str,
            operand: RuntimeOperandBinding,
        ) -> None:
            if source is None or source.descriptor_id == descriptor.descriptor_id:
                return
            dependency = BoundDependency(
                source=source.completion_token,
                kind=kind,
                condition="physical_range_released",
                provenance={
                    "source": "runtime_binding_alias",
                    "frontier": "last_writer_reader",
                    "memory": operand.physical_scope,
                    "address": operand.address,
                    "size_bytes": operand.size_bytes,
                },
            )
            previous = alias_by_source.get(source.instruction.tisa_id)
            if previous is None or priority[kind] < priority[previous.kind]:
                alias_by_source[source.instruction.tisa_id] = dependency

        for operand in descriptor.operands:
            is_read = operand.access_type in {"read", "read_write"}
            is_write = operand.access_type in {"write", "read_write"}
            for state in _operand_frontiers(operand, alias_key, points, frontiers):
                if is_read:
                    add_alias(state.writer, "RAW", operand)
                if is_write:
                    if state.readers:
                        for reader in state.readers.values():
                            add_alias(reader, "WAR", operand)
                    elif state.writer is not None:
                        add_alias(
                            state.writer,
                            "RAW" if operand.access_type == "read_write" else "WAW",
                            operand,
                        )

        for dependency in alias_by_source.values():
            dependencies[
                (
                    dependency.source.tisa_id,
                    dependency.kind,
                    dependency.condition,
                )
            ] = dependency
        updated.append(replace(descriptor, dependencies=tuple(dependencies.values())))

        # Reads become the WAR frontier. Writes, including read-modify-write,
        # supersede both the previous writer and all preceding readers.
        for operand in descriptor.operands:
            if operand.access_type == "read":
                for state in _operand_frontiers(
                    operand, alias_key, points, frontiers
                ):
                    state.readers[descriptor.descriptor_id] = descriptor
        for operand in descriptor.operands:
            if operand.access_type in {"write", "read_write"}:
                for state in _operand_frontiers(
                    operand, alias_key, points, frontiers
                ):
                    state.writer = descriptor
                    state.readers.clear()
    by_id = {item.descriptor_id: item for item in updated}
    return tuple(by_id[item.descriptor_id] for item in descriptors)


@dataclass
class _AliasFrontier:
    writer: BoundTISADescriptor | None = None
    readers: dict[str, BoundTISADescriptor] = field(default_factory=dict)


def _operand_frontiers(
    operand: RuntimeOperandBinding,
    alias_key,
    points: dict[tuple[str, object | None], tuple[int, ...]],
    frontiers: dict[tuple[str, object | None], list[_AliasFrontier]],
) -> tuple[_AliasFrontier, ...]:
    key = alias_key(operand)
    boundaries = points[key]
    start = bisect_left(boundaries, operand.address)
    stop = bisect_left(boundaries, operand.address + operand.size_bytes)
    return tuple(frontiers[key][start:stop])


def _static_schedule(
    program_id: str,
    descriptors: tuple[BoundTISADescriptor, ...],
) -> StaticSchedulePlan:
    # Freeze the host-selected descriptor order for this invocation.  Using
    # compiler program order here can deadlock a finite WQ when the runtime
    # deliberately submits another legal topological order.
    ordered = tuple(sorted(descriptors, key=lambda item: item.submission_order))
    return StaticSchedulePlan(
        plan_id=f"{program_id}.static-plan",
        program_id=program_id,
        entries=tuple(
            StaticScheduleEntry(
                tisa_id=descriptor.instruction.tisa_id,
                order=index,
                resource=descriptor.instruction.unit_map.unit,
                dependency_tokens=tuple(
                    item.source.token_id for item in descriptor.dependencies
                ),
            )
            for index, descriptor in enumerate(ordered)
        ),
        policy="runtime_fixed_submission_overlap",
        attributes={
            "source": "runtime_submission_order",
            "overlap": "different resources may overlap after ordered admission",
            "calibration_status": "project_baseline",
        },
    )


def load_implicit_device_program(
    artifact: BackendArtifact,
    *,
    invocation_id: str = "implicit",
) -> LoadedDeviceProgram:
    """Build a zero-latency descriptor stream for low-level micro-tests."""

    issues = artifact.validate()
    if issues:
        raise ValueError("cannot load implicit device program: " + "; ".join(issues))
    descriptors = []
    envelopes = []
    feedback_conditions = {
        instruction.tisa_id: sorted(
            {
                dependency.condition
                for consumer in artifact.program.instructions
                for dependency in consumer.dependencies
                if dependency.source == instruction.tisa_id
                and dependency.condition.startswith("payload_ready:")
            }
        )
        for instruction in artifact.program.instructions
    }
    for order, instruction in enumerate(artifact.program.instructions):
        descriptor_id = f"{invocation_id}::{instruction.tisa_id}"
        bound_operands = tuple(
            RuntimeOperandBinding(
                tisa_id=instruction.tisa_id,
                operand_name=operand.name,
                tensor=operand.tile_mem.tensor or operand.tile_mem.base,
                logical_scope=operand.tile_mem.scope,
                physical_scope=operand.tile_mem.physical_space,
                address=int(operand.tile_mem.offset_bytes or 0),
                size_bytes=int(operand.tile_mem.size_bytes or 1),
                access_type=operand.normalized_access,
                offset_bytes=int(operand.tile_mem.offset_bytes or 0),
                buffer_id=operand.tile_mem.buffer_id,
                attributes={
                    "address_source": "implicit_tile_mem",
                    "allocation_identity": (
                        operand.tile_mem.allocation_id
                        or operand.tile_mem.buffer_id
                        or operand.tile_mem.tensor
                        or operand.tile_mem.base
                    ),
                },
            )
            for operand in instruction.operands
        )
        descriptors.append(
            BoundTISADescriptor(
                descriptor_id=descriptor_id,
                invocation_id=invocation_id,
                program_id=artifact.program.program_id,
                artifact_id=artifact.artifact_id,
                instruction=instruction,
                operands=bound_operands,
                dependencies=tuple(
                    BoundDependency(
                        CompletionToken(invocation_id, dependency.source),
                        dependency.kind,
                        dependency.condition,
                        dependency.provenance,
                    )
                    for dependency in instruction.dependencies
                ),
                completion_token=CompletionToken(invocation_id, instruction.tisa_id),
                payload_handle=f"{artifact.artifact_id}::{instruction.tisa_id}",
                program_order=order,
                submission_order=order,
                attributes={
                    "runtime_policy": "implicit_static",
                    "feedback_conditions": feedback_conditions[instruction.tisa_id],
                },
            )
        )
        envelopes.append(
            DescriptorEnvelope(
                descriptor_id=descriptor_id,
                chunk_id="implicit.chunk0000",
                queue="device",
                arrival_cycle=0.0,
                chunk_order=0,
                descriptor_order=order,
            )
        )
    descriptors = list(_add_runtime_alias_dependencies(tuple(descriptors)))
    loaded = LoadedDeviceProgram(
        program_id=artifact.program.program_id,
        artifact_id=artifact.artifact_id,
        invocation_id=invocation_id,
        descriptors=tuple(descriptors),
        envelopes=tuple(envelopes),
        static_schedule=_static_schedule(
            artifact.program.program_id, tuple(descriptors)
        ),
        attributes={"runtime_policy": "implicit_static", "command_chunk_count": 0},
    )
    loaded_issues = loaded.validate()
    if loaded_issues:
        raise ValueError("implicit device program is invalid: " + "; ".join(loaded_issues))
    return loaded
