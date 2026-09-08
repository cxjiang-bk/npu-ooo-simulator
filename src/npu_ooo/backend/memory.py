"""Target-memory materialization for analytical backend artifacts.

Target lowering has already produced target operands before this module runs.
This module binds payload regions to that plan, allocates finite target
memories, and adds physical hazard dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    BufferRegion,
    ExecutionGraph,
    MemoryBuffer,
    MemoryPlan,
    OperatorGraph,
    TISADependency,
    TISAInstruction,
    TISAProgram,
    dtype_bytes,
    tensor_layout,
)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _valid_bytes(region: BufferRegion) -> int:
    return region.valid_bytes or math.prod(region.shape) * dtype_bytes(
        region.dtype, default=2
    )


def _packed_strides(shape: tuple[int, ...], dtype: str) -> tuple[int, ...]:
    stride = dtype_bytes(dtype, default=2)
    values: list[int] = []
    for extent in reversed(shape):
        values.append(stride)
        stride *= extent
    return tuple(reversed(values))


def _normalize_region(
    region: BufferRegion,
    *,
    root_memories: set[str],
) -> BufferRegion:
    valid_bytes = _valid_bytes(region)
    size_bytes = region.size_bytes or valid_bytes
    backend_packed = region.buffer_id is None and region.memory not in root_memories
    return replace(
        region,
        offset_bytes=0 if backend_packed else region.offset_bytes,
        size_bytes=valid_bytes if backend_packed else size_bytes,
        layout="packed" if backend_packed else region.layout,
        strides_bytes=(
            _packed_strides(region.shape, region.dtype)
            if backend_packed
            else region.strides_bytes
        ),
        valid_bytes=valid_bytes,
        attributes={
            **dict(region.attributes),
            "buffer_identity_source": "target_plan",
        },
    )


def _bind_region_to_plan(
    region: BufferRegion,
    instruction: TISAInstruction,
    *,
    write: bool,
) -> BufferRegion:
    allowed = (
        {AccessType.WRITE.value, AccessType.READ_WRITE.value}
        if write
        else {AccessType.READ.value, AccessType.READ_WRITE.value}
    )
    matches = [
        operand
        for operand in instruction.operands
        if operand.tile_mem.tensor == region.tensor
        and operand.tile_mem.physical_space == region.memory
        and operand.normalized_access in allowed
        and (
            operand.tile_mem.logical_starts is None
            or operand.tile_mem.logical_starts == region.starts
        )
        and (
            operand.tile_mem.logical_shape is None
            or operand.tile_mem.logical_shape == region.shape
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"payload region '{region.tensor}@{region.memory}' in target instruction "
            f"'{instruction.tisa_id}' resolves to {len(matches)} planned operands"
        )
    planned = matches[0].tile_mem
    return replace(
        region,
        access=(AccessType.WRITE if write else AccessType.READ),
        offset_bytes=int(planned.offset_bytes or 0),
        size_bytes=int(planned.size_bytes or region.size_bytes),
        layout=planned.layout,
        strides_bytes=planned.strides_bytes,
        buffer_id=planned.buffer_id,
        valid_bytes=planned.valid_bytes or _valid_bytes(region),
        attributes={
            **dict(region.attributes),
            "buffer_identity_source": "target_plan",
            "target_plan_operand": matches[0].name,
            "symbolic_buffer_id": planned.symbolic_buffer_id,
        },
    )


def _normalize_execution_graph(
    artifact: BackendArtifact,
    machine: MachineConfig,
) -> ExecutionGraph:
    roots = {level.name for level in machine.memory_levels if level.parent is None}
    task_owner = {
        task_id: next(
            instruction
            for instruction in artifact.program.instructions
            if instruction.tisa_id == tisa_id
        )
        for tisa_id, task_ids in artifact.payloads.items()
        for task_id in task_ids
    }
    return replace(
        artifact.execution_graph,
        tasks=tuple(
            replace(
                task,
                reads=tuple(
                    _bind_region_to_plan(
                        _normalize_region(region, root_memories=roots),
                        task_owner[task.task_id],
                        write=False,
                    )
                    for region in task.reads
                ),
                writes=tuple(
                    _bind_region_to_plan(
                        _normalize_region(region, root_memories=roots),
                        task_owner[task.task_id],
                        write=True,
                    )
                    for region in task.writes
                ),
            )
            for task in artifact.execution_graph.tasks
        ),
        attributes={
            **dict(artifact.execution_graph.attributes),
            "target_memory_materialized": True,
            "operand_source": "target_plan",
        },
    )


@dataclass(frozen=True)
class _Access:
    instruction_index: int
    tisa_id: str
    buffer_id: str
    offset: int
    size: int
    read: bool
    write: bool


def _instruction_accesses(
    program: TISAProgram,
    *,
    by_allocation: bool = False,
) -> dict[str, list[_Access]]:
    result: dict[str, list[_Access]] = {}
    for index, instruction in enumerate(program.instructions):
        for operand in instruction.operands:
            memory = operand.tile_mem
            if memory.buffer_id is None or memory.offset_bytes is None or memory.size_bytes is None:
                raise ValueError(
                    f"target operand '{instruction.tisa_id}:{operand.name}' has unresolved range"
                )
            access = operand.normalized_access
            identity = (
                memory.allocation_id
                if by_allocation and memory.allocation_id is not None
                else memory.buffer_id
            )
            result.setdefault(identity, []).append(
                _Access(
                    instruction_index=index,
                    tisa_id=instruction.tisa_id,
                    buffer_id=memory.buffer_id,
                    offset=memory.offset_bytes,
                    size=memory.size_bytes,
                    read=access in {AccessType.READ.value, AccessType.READ_WRITE.value},
                    write=access in {AccessType.WRITE.value, AccessType.READ_WRITE.value},
                )
            )
    return result


def _add_physical_hazards(
    program: TISAProgram,
    *,
    by_allocation: bool = False,
) -> TISAProgram:
    additions: dict[str, dict[str, TISADependency]] = {
        instruction.tisa_id: {} for instruction in program.instructions
    }
    for buffer_id, accesses in _instruction_accesses(
        program, by_allocation=by_allocation
    ).items():
        for current_index, current in enumerate(accesses):
            for previous in accesses[:current_index]:
                if previous.tisa_id == current.tisa_id:
                    continue
                if previous.offset + previous.size <= current.offset or current.offset + current.size <= previous.offset:
                    continue
                if not (previous.write or current.write):
                    continue
                kind = (
                    "RAW"
                    if previous.write and current.read and not current.write
                    else "WAR"
                    if previous.read and current.write and not previous.write
                    else "WAW"
                )
                additions[current.tisa_id].setdefault(
                    previous.tisa_id,
                    TISADependency(
                        source=previous.tisa_id,
                        kind=kind,
                        condition="physical_range_released",
                        provenance={
                            "source": (
                                "target_allocation_hazard"
                                if by_allocation
                                else "target_memory_hazard"
                            ),
                            (
                                "allocation_id" if by_allocation else "buffer_id"
                            ): buffer_id,
                            "range": [
                                max(previous.offset, current.offset),
                                min(
                                    previous.offset + previous.size,
                                    current.offset + current.size,
                                ),
                            ],
                        },
                    ),
                )
    instructions = []
    for instruction in program.instructions:
        dependencies = {item.source: item for item in instruction.dependencies}
        for source, dependency in additions[instruction.tisa_id].items():
            dependencies.setdefault(source, dependency)
        instructions.append(
            replace(instruction, dependencies=tuple(dependencies.values()))
        )
    return replace(program, instructions=tuple(instructions))


def _allocate_plan(
    graph: OperatorGraph,
    program: TISAProgram,
    machine: MachineConfig,
) -> tuple[MemoryPlan, dict[str, str], dict[str, tuple[str, ...]]]:
    accesses = _instruction_accesses(program)
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    requirements: dict[str, dict[str, Any]] = {}
    for instruction in program.instructions:
        for operand in instruction.operands:
            memory = operand.tile_mem
            buffer_id = str(memory.buffer_id)
            current = requirements.setdefault(
                buffer_id,
                {
                    "tensor": memory.tensor or memory.base,
                    "dtype": memory.dtype,
                    "memory": memory.physical_space,
                    "allocation_bytes": 0,
                    "valid_bytes": 0,
                    "layout": memory.layout,
                    "strides_bytes": memory.strides_bytes,
                    "tile_id": instruction.tile_id,
                    "role": memory.role,
                    "visibility": memory.visibility,
                    "owner": memory.owner,
                    "domain": memory.domain,
                    "symbolic_buffer_id": memory.symbolic_buffer_id,
                    "slot": None,
                },
            )
            current["allocation_bytes"] = max(
                current["allocation_bytes"],
                int(memory.offset_bytes or 0) + int(memory.size_bytes or 0),
            )
            current["valid_bytes"] = max(
                current["valid_bytes"], int(memory.valid_bytes or memory.size_bytes or 0)
            )
            if current["layout"] != memory.layout:
                current["layout"] = "mixed"
                current["strides_bytes"] = None

    root_memories = {level.name for level in machine.memory_levels if level.parent is None}
    for buffer_id, requirement in requirements.items():
        tensor = tensors.get(requirement["tensor"])
        if requirement["memory"] in root_memories and tensor is not None:
            requirement["allocation_bytes"] = max(
                requirement["allocation_bytes"],
                tensor_layout(tensor).allocation_size_bytes,
            )
        buffer_accesses = accesses[buffer_id]
        requirement["lifetime_start"] = min(item.instruction_index for item in buffer_accesses)
        requirement["lifetime_end"] = max(item.instruction_index for item in buffer_accesses)
        marker = ".slot"
        if marker in buffer_id:
            try:
                requirement["slot"] = int(buffer_id.split(marker, 1)[1].split("@", 1)[0])
            except ValueError:
                pass

    buffers: list[MemoryBuffer] = []
    allocation_for: dict[str, str] = {}
    reuse_dependencies: dict[str, set[str]] = {}
    for memory_level in machine.memory_levels:
        memory_requirements = [
            (buffer_id, requirement)
            for buffer_id, requirement in requirements.items()
            if requirement["memory"] == memory_level.name
        ]
        memory_requirements.sort(
            key=lambda item: (
                item[1]["lifetime_start"],
                item[1]["lifetime_end"],
                item[0],
            )
        )
        cursor = 0
        blocks: list[dict[str, Any]] = []
        by_buffer: dict[str, MemoryBuffer] = {}
        for buffer_id, requirement in memory_requirements:
            tensor = tensors.get(requirement["tensor"])
            requested_alignment = memory_level.alignment_bytes
            external = memory_level.name in root_memories
            alias_tensor = (
                tensor.attributes.get("alias_of")
                if tensor is not None and isinstance(tensor.attributes.get("alias_of"), str)
                else None
            )
            alias_buffer_id = (
                f"{alias_tensor}@{memory_level.name}" if alias_tensor else None
            )
            selected: dict[str, Any] | None = None
            if alias_buffer_id in by_buffer:
                aliased = by_buffer[alias_buffer_id]
                if requirement["allocation_bytes"] > aliased.allocation_bytes:
                    raise ValueError(
                        f"alias buffer '{buffer_id}' requires {requirement['allocation_bytes']} bytes, "
                        f"but '{alias_buffer_id}' has {aliased.allocation_bytes}"
                    )
                selected = next(
                    block for block in blocks if block["allocation_id"] == aliased.allocation_id
                )
            if selected is None and not external:
                candidates = [
                    block
                    for block in blocks
                    if block["lifetime_end"] < requirement["lifetime_start"]
                    and block["size"] >= requirement["allocation_bytes"]
                ]
                if candidates:
                    selected = min(
                        candidates,
                        key=lambda block: (
                            block["size"] - requirement["allocation_bytes"],
                            block["offset"],
                        ),
                    )
                    for current_access in accesses[buffer_id]:
                        reuse_dependencies.setdefault(current_access.tisa_id, set()).update(
                            selected["all_uses"]
                        )
            if selected is None:
                offset = _align(cursor, requested_alignment)
                size = _align(requirement["allocation_bytes"], requested_alignment)
                allocation_id = f"{memory_level.name}.alloc{len(blocks):04d}"
                selected = {
                    "allocation_id": allocation_id,
                    "offset": offset,
                    "size": size,
                    "lifetime_end": requirement["lifetime_end"],
                    "all_uses": tuple(
                        sorted({item.tisa_id for item in accesses[buffer_id]})
                    ),
                    "last_buffer": buffer_id,
                }
                blocks.append(selected)
                cursor = offset + size
            else:
                selected["lifetime_end"] = max(
                    selected["lifetime_end"], requirement["lifetime_end"]
                )
                selected["all_uses"] = tuple(
                    sorted({item.tisa_id for item in accesses[buffer_id]})
                )
            allocation_for[buffer_id] = selected["allocation_id"]
            alias_of = (
                alias_buffer_id
                if alias_buffer_id in by_buffer
                else selected.get("last_buffer")
                if selected.get("last_buffer") != buffer_id
                else None
            )
            memory_buffer = MemoryBuffer(
                buffer_id=buffer_id,
                allocation_id=selected["allocation_id"],
                tensor=requirement["tensor"],
                memory=memory_level.name,
                offset_bytes=selected["offset"],
                allocation_bytes=selected["size"],
                valid_bytes=requirement["valid_bytes"],
                alignment_bytes=requested_alignment,
                layout=requirement["layout"],
                strides_bytes=requirement["strides_bytes"],
                tile_id=requirement["tile_id"],
                role=requirement["role"],
                slot=requirement["slot"],
                external=external,
                lifetime_start=requirement["lifetime_start"],
                lifetime_end=requirement["lifetime_end"],
                alias_of=alias_of,
                dtype=(tensor.dtype if tensor is not None else requirement["dtype"]),
                attributes={
                    "allocation_policy": (
                        "external_linear" if external else "dependency_guarded_reuse"
                    ),
                    "visibility": requirement["visibility"],
                    "owner": requirement["owner"],
                    "domain": requirement["domain"],
                    "symbolic_buffer_id": requirement["symbolic_buffer_id"],
                    **(
                        {
                            "shape": list(tensor.shape),
                            "logical_alias_of": alias_tensor,
                            "persistent": bool(
                                external and tensor.attributes.get("persistent", False)
                            ),
                            "state_id": tensor.attributes.get("state_id", tensor.name),
                            "state_buffer": tensor.attributes.get("state_buffer", tensor.name),
                        }
                        if tensor is not None
                        else {}
                    ),
                },
            )
            buffers.append(memory_buffer)
            by_buffer[buffer_id] = memory_buffer
            selected["last_buffer"] = buffer_id
        if memory_level.capacity_bytes is not None and cursor > memory_level.capacity_bytes:
            raise ValueError(
                f"target memory plan overflow in '{memory_level.name}': requires {cursor} bytes, "
                f"capacity is {memory_level.capacity_bytes} bytes"
            )

    plan = MemoryPlan(
        plan_id=f"{program.program_id}.{machine.config_id}.memory",
        machine_config_id=machine.config_id,
        machine_topology_hash=machine.topology_hash(),
        buffers=tuple(buffers),
        attributes={
            "owner": "codegen_backend",
            "allocation_policy": "finite_arena_with_dependency_guarded_reuse",
        },
    )
    issues = plan.validate(machine)
    if issues:
        raise ValueError("target memory plan is invalid: " + "; ".join(issues))
    return (
        plan,
        allocation_for,
        {target: tuple(sorted(sources)) for target, sources in reuse_dependencies.items()},
    )


def _apply_allocations_and_reuse(
    program: TISAProgram,
    allocation_for: Mapping[str, str],
    reuse_dependencies: Mapping[str, tuple[str, ...]],
) -> TISAProgram:
    instructions: list[TISAInstruction] = []
    for instruction in program.instructions:
        dependencies = {item.source: item for item in instruction.dependencies}
        for source in reuse_dependencies.get(instruction.tisa_id, ()):
            if source == instruction.tisa_id:
                continue
            dependencies.setdefault(
                source,
                TISADependency(
                    source=source,
                    kind="BUFFER_REUSE",
                    condition="allocation_released",
                    provenance={"source": "target_memory_allocator"},
                ),
            )
        instructions.append(
            replace(
                instruction,
                operands=tuple(
                    replace(
                        operand,
                        tile_mem=replace(
                            operand.tile_mem,
                            allocation_id=allocation_for[str(operand.tile_mem.buffer_id)],
                        ),
                    )
                    for operand in instruction.operands
                ),
                dependencies=tuple(dependencies.values()),
            )
        )
    return replace(
        program,
        instructions=tuple(instructions),
        attributes={
            **dict(program.attributes),
            "target_memory_materialized": True,
            "memory_plan_schema": 2,
        },
    )


def materialize_target_memory(
    graph: OperatorGraph,
    machine: MachineConfig,
    artifact: BackendArtifact,
) -> BackendArtifact:
    """Return a backend artifact with one auditable physical memory contract."""

    if artifact.target_plan is None:
        raise ValueError("target memory materialization requires a TargetPlan")
    execution_graph = _normalize_execution_graph(artifact, machine)
    program = artifact.program
    program = _add_physical_hazards(program)
    plan, allocation_for, reuse_dependencies = _allocate_plan(
        graph, program, machine
    )
    program = _apply_allocations_and_reuse(
        program, allocation_for, reuse_dependencies
    )
    # A second pass catches explicit aliases and distinct buffer identities
    # that share an allocation.  Existing BUFFER_REUSE edges take precedence;
    # the pass fills only missing physical hazard ordering.
    program = _add_physical_hazards(program, by_allocation=True)
    instruction_by_id = {
        instruction.tisa_id: instruction for instruction in program.instructions
    }
    target_plan = replace(
        artifact.target_plan,
        instructions=tuple(
            replace(item, instruction=instruction_by_id[item.instruction.tisa_id])
            for item in artifact.target_plan.instructions
        ),
        attributes={
            **dict(artifact.target_plan.attributes),
            "program_attributes": dict(program.attributes),
            "memory_plan_id": plan.plan_id,
            "memory_plan_schema": plan.schema_version,
        },
        memory_plan=plan,
    )
    result = replace(
        artifact,
        program=program,
        execution_graph=execution_graph,
        memory_plan=plan,
        target_plan=target_plan,
        attributes={
            **dict(artifact.attributes),
            "memory_contract": "target_memory_plan_v2",
            "machine_topology_hash": machine.topology_hash(),
        },
    )
    issues = result.validate()
    if issues:
        raise ValueError(
            "target-memory backend artifact is invalid: " + "; ".join(issues)
        )
    return result


__all__ = ["materialize_target_memory"]
