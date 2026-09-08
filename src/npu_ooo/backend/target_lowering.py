"""Lower virtual TISA into a concrete, auditable target plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    ExecutionGraph,
    OperatorGraph,
    ScheduleSpec,
    TISADependency,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TargetInstructionPlan,
    TargetPlan,
    TileGraph,
    TileInstance,
    UnitMap,
    TileMem,
    dtype_bytes,
    tensor_layout,
)


def _root_memory(machine: MachineConfig) -> str:
    roots = tuple(level.name for level in machine.memory_levels if level.parent is None)
    if len(roots) != 1:
        raise ValueError("target lowering requires exactly one root memory")
    return roots[0]


def _path(machine: MachineConfig, source: str, target: str):
    matches = tuple(
        item
        for item in machine.transfer_paths
        if item.source == source and item.target == target
    )
    if len(matches) != 1:
        raise ValueError(
            f"target lowering requires one path {source}->{target}; found {len(matches)}"
        )
    return matches[0]


def _default_local_memory(machine: MachineConfig, root: str) -> str:
    for path in machine.transfer_paths:
        if path.source == root:
            return path.target
    return root


def _packed_strides(shape: tuple[int, ...], element_bytes: int) -> tuple[int, ...]:
    stride = element_bytes
    result: list[int] = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _target_operand(
    abstract: TISAOperand,
    *,
    name: str,
    memory: str,
    buffer_id: str,
    access: AccessType,
    packed: bool,
) -> TISAOperand:
    tile_mem = abstract.tile_mem
    valid_bytes = tile_mem.valid_bytes or (
        math.prod(abstract.tile_shape) * dtype_bytes("fp16", default=2)
    )
    element_bytes = dtype_bytes(tile_mem.dtype, default=2)
    packed_strides = _packed_strides(abstract.tile_shape, element_bytes)
    return TISAOperand(
        name=name,
        tile_shape=abstract.tile_shape,
        tile_mem=replace(
            tile_mem,
            base=buffer_id,
            memory_space=memory,
            symbolic_buffer_id=(
                tile_mem.symbolic_buffer_id or tile_mem.buffer_id or tile_mem.base
            ),
            buffer_id=buffer_id,
            allocation_id=None,
            offset_bytes=0 if packed else tile_mem.offset_bytes,
            size_bytes=valid_bytes if packed else tile_mem.size_bytes,
            strides_bytes=(
                packed_strides if packed else tile_mem.strides_bytes
            ),
            stride_expr=(
                " + ".join(
                    f"i{index}*{stride}"
                    for index, stride in enumerate(
                        packed_strides
                    )
                ) or "0"
                if packed
                else tile_mem.stride_expr
            ),
            layout="packed" if packed else tile_mem.layout,
            valid_bytes=valid_bytes,
        ),
        access_type=access,
    )


def _by_role(
    instruction: TISAInstruction,
    *,
    visibility: str | None = None,
) -> dict[str, TISAOperand]:
    result: dict[str, TISAOperand] = {}
    for operand in instruction.operands:
        memory = operand.tile_mem
        if visibility is not None and memory.visibility != visibility:
            continue
        role = memory.role
        if role is None:
            continue
        result[role] = operand
    return result


def _unit_for_abstract(
    instruction: TISAInstruction,
    machine: MachineConfig,
) -> str:
    requested = instruction.unit_map.unit.lower()
    aliases = {
        "dma": {"dma", "gdma", "ldma", "de", "copy"},
        "tensor": {"mxu", "me", "tensor", "matrix"},
        "vector": {"aru", "ve", "vector", "vu"},
    }
    for unit in machine.execution_units:
        if unit.name.lower() in aliases.get(requested, {requested}):
            return unit.name
    for unit in machine.execution_units:
        if instruction.op_type in unit.supported_ops:
            return unit.name
    raise ValueError(
        f"machine '{machine.config_id}' has no target unit for abstract UnitMap "
        f"'{instruction.unit_map.unit}'"
    )


@dataclass
class _Draft:
    abstract: TISAInstruction
    instruction: TISAInstruction
    expansion_kind: str
    roles: tuple[str, ...]
    route_hop: int | None
    internal_sources: tuple[str, ...]
    attributes: Mapping[str, Any]


class TargetLowerer:
    """Expand target routes after FC and before payload generation."""

    name = "machine-config-target-lowering-v1"

    def lower(
        self,
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        tile_graph: TileGraph,
        machine: MachineConfig,
        abstract_program: TISAProgram,
    ) -> TargetPlan:
        graph_issues = graph.validate()
        schedule_issues = schedule.validate(graph)
        tile_issues = tile_graph.validate()
        machine_issues = machine.validate()
        program_issues = abstract_program.validate()
        if graph_issues or schedule_issues or tile_issues or machine_issues or program_issues:
            raise ValueError(
                "target lowering input is invalid: "
                + "; ".join(
                    (*graph_issues, *schedule_issues, *tile_issues, *machine_issues, *program_issues)
                )
            )
        root = _root_memory(machine)
        local = _default_local_memory(machine, root)
        tiles = {tile.tile_id: tile for tile in tile_graph.tiles}
        operators = {operator.op_id: operator for operator in graph.operators}
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        plan_id = f"{abstract_program.program_id}.{machine.config_id}.target-plan"
        output_slots = self._output_slots(abstract_program, schedule)
        drafts: list[_Draft] = []
        symbolic_map: dict[str, set[str]] = {}

        def record_operand(operand: TISAOperand) -> None:
            symbolic = operand.tile_mem.symbolic_buffer_id
            target = operand.tile_mem.buffer_id
            if symbolic is not None and target is not None:
                symbolic_map.setdefault(symbolic, set()).add(target)

        for abstract in abstract_program.instructions:
            operator = operators[abstract.operator_id]
            tile = tiles[abstract.tile_id]
            semantic_type = str(
                abstract.attributes.get("semantic_op_type", operator.normalized_type)
            )
            stage = str(abstract.attributes.get("tisa_stage", ""))
            if semantic_type in {"matmul", "batched_matmul", "gemv"}:
                generated = self._lower_matmul_instruction(
                    abstract,
                    tile,
                    schedule,
                    machine,
                    output_slots,
                )
            else:
                generated = (
                    self._lower_generic_instruction(
                        abstract, machine, root=root, local=local
                    ),
                )
            if not generated:
                raise ValueError(
                    f"target lowering produced no instructions for '{abstract.tisa_id}'"
                )
            for draft in generated:
                resolved_operands = []
                for operand in draft.instruction.operands:
                    memory = operand.tile_mem
                    if memory.size_bytes is None:
                        tensor = tensors[str(memory.tensor)]
                        memory = replace(
                            memory,
                            offset_bytes=0,
                            size_bytes=tensor_layout(tensor).allocation_size_bytes,
                        )
                        operand = replace(operand, tile_mem=memory)
                    resolved_operands.append(operand)
                draft.instruction = replace(
                    draft.instruction, operands=tuple(resolved_operands)
                )
                if stage and draft.abstract.attributes.get("tisa_stage") != stage:
                    raise ValueError("target draft lost its abstract stage provenance")
                for operand in draft.instruction.operands:
                    record_operand(operand)
                drafts.append(draft)

        mapped: dict[str, list[str]] = {}
        for draft in drafts:
            mapped.setdefault(draft.abstract.tisa_id, []).append(
                draft.instruction.tisa_id
            )
        terminals = {}
        for abstract_id, target_ids in mapped.items():
            consumed = {
                source
                for draft in drafts
                if draft.abstract.tisa_id == abstract_id
                for source in draft.internal_sources
            }
            terminals[abstract_id] = tuple(
                target_id for target_id in target_ids if target_id not in consumed
            )

        planned: list[TargetInstructionPlan] = []
        for draft in drafts:
            dependencies = {
                source: TISADependency(
                    source=source,
                    kind="RAW",
                    condition="target_buffer_ready",
                    provenance={
                        "source": "target_route",
                        "abstract_tisa_id": draft.abstract.tisa_id,
                    },
                )
                for source in draft.internal_sources
            }
            if not draft.internal_sources:
                for abstract_dependency in draft.abstract.dependencies:
                    for source in terminals[abstract_dependency.source]:
                        dependencies.setdefault(
                            source,
                            replace(
                                abstract_dependency,
                                source=source,
                                provenance={
                                    **dict(abstract_dependency.provenance),
                                    "expanded_from_abstract_source": abstract_dependency.source,
                                    "expanded_to_abstract_target": draft.abstract.tisa_id,
                                    "target_expansion": "abstract_dependency",
                                },
                            ),
                        )
            instruction = replace(
                draft.instruction,
                dependencies=tuple(dependencies.values()),
                attributes={
                    **dict(draft.instruction.attributes),
                    "abstract_tisa_id": draft.abstract.tisa_id,
                    "abstract_stage": draft.abstract.attributes.get("tisa_stage"),
                    "target_plan_id": plan_id,
                    "target_lowering": self.name,
                    "stage_id": tiles[draft.instruction.tile_id].stage_id,
                    "scheduler_visible": True,
                    "target_expansion_required": False,
                },
            )
            planned.append(
                TargetInstructionPlan(
                    abstract_tisa_id=draft.abstract.tisa_id,
                    instruction=instruction,
                    expansion_kind=draft.expansion_kind,
                    abstract_operand_roles=draft.roles,
                    route_hop=draft.route_hop,
                    attributes=draft.attributes,
                )
            )

        target_program_id = abstract_program.program_id.replace(
            ".virtual-tisa", ".tisa"
        )
        target_plan = TargetPlan(
            plan_id=plan_id,
            abstract_program_id=abstract_program.program_id,
            machine_config_id=machine.config_id,
            machine_topology_hash=machine.topology_hash(),
            instructions=tuple(planned),
            symbolic_buffer_map={
                key: tuple(sorted(value)) for key, value in symbolic_map.items()
            },
            attributes={
                "lowerer": self.name,
                "target_program_id": target_program_id,
                "program_attributes": {
                    **dict(abstract_program.attributes),
                    "paper_stage": "TARGET_LOWERING",
                    "virtual_program_id": abstract_program.program_id,
                    "target_plan_id": plan_id,
                    "machine_config_id": machine.config_id,
                    "machine_topology_hash": machine.topology_hash(),
                },
                "route_policy": "explicit_machine_config",
            },
        )
        issues = target_plan.validate(abstract_program)
        if issues:
            raise ValueError("target plan is invalid: " + "; ".join(issues))
        return target_plan

    @staticmethod
    def _output_slots(
        program: TISAProgram,
        schedule: ScheduleSpec,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        by_operator: dict[str, set[str]] = {}
        for instruction in program.instructions:
            if instruction.attributes.get("tisa_stage") != "compute":
                continue
            for operand in instruction.operands:
                memory = operand.tile_mem
                if memory.role == "output" and memory.symbolic_buffer_id:
                    by_operator.setdefault(instruction.operator_id, set()).add(
                        memory.symbolic_buffer_id
                    )
        for operator_id, symbolic_ids in by_operator.items():
            ping_pong = schedule.for_operator(operator_id).attributes.get(
                "ping_pong", {}
            )
            count = (
                int(ping_pong.get("buffer_count", 1))
                if isinstance(ping_pong, Mapping)
                else 1
            )
            for index, symbolic_id in enumerate(sorted(symbolic_ids)):
                result[symbolic_id] = index % max(1, count)
        return result

    def _lower_generic_instruction(
        self,
        abstract: TISAInstruction,
        machine: MachineConfig,
        *,
        root: str,
        local: str,
    ) -> _Draft:
        operands: list[TISAOperand] = []
        for operand in abstract.operands:
            memory = operand.tile_mem
            is_shared = memory.visibility == "Shared"
            target_memory = root if is_shared else local
            symbolic = memory.symbolic_buffer_id or memory.buffer_id or memory.base
            buffer_id = (
                f"{memory.tensor}@{target_memory}"
                if is_shared
                else f"{symbolic}@{target_memory}"
            )
            operands.append(
                _target_operand(
                    operand,
                    name=operand.name,
                    memory=target_memory,
                    buffer_id=buffer_id,
                    access=AccessType(operand.normalized_access),
                    packed=not is_shared,
                )
            )
        stage = str(abstract.attributes.get("tisa_stage", ""))
        if stage == "load":
            path = _path(machine, root, local)
            unit = path.engine
        elif stage == "store":
            path = _path(machine, local, root)
            unit = path.engine
        else:
            unit = _unit_for_abstract(abstract, machine)
        target_id = f"{abstract.tisa_id}.target"
        return _Draft(
            abstract=abstract,
            instruction=replace(
                abstract,
                tisa_id=target_id,
                operands=tuple(operands),
                unit_map=UnitMap(
                    unit,
                    quantity=abstract.unit_map.quantity,
                    affinity=abstract.unit_map.affinity,
                ),
                dependencies=(),
                payload_ref=f"payload:{target_id}",
                attributes={
                    **dict(abstract.attributes),
                    "target_stage_kind": "direct",
                },
            ),
            expansion_kind="direct",
            roles=tuple(
                sorted(
                    {
                        str(operand.tile_mem.role)
                        for operand in abstract.operands
                        if operand.tile_mem.role
                    }
                )
            ),
            route_hop=None,
            internal_sources=(),
            attributes={"abstract_instruction_count": 1},
        )

    def _lower_matmul_instruction(
        self,
        abstract: TISAInstruction,
        tile: TileInstance,
        schedule: ScheduleSpec,
        machine: MachineConfig,
        output_slots: Mapping[str, int],
    ) -> tuple[_Draft, ...]:
        try:
            placement = machine.placement("matmul")
        except KeyError as exc:
            raise ValueError(
                f"machine '{machine.config_id}' has no matmul target placement"
            ) from exc
        stage = str(abstract.attributes.get("tisa_stage", ""))
        if stage == "compute":
            return (
                self._matmul_compute(
                    abstract, tile, schedule, placement, output_slots
                ),
            )
        if stage == "load":
            roles = ("lhs", "rhs")
            direction = "input"
        elif stage == "store":
            roles = ("output",)
            direction = "output"
        else:
            raise ValueError(
                f"unsupported abstract Matmul stage '{stage}' in '{abstract.tisa_id}'"
            )
        role_placements = {role: placement.operand(role) for role in roles}
        local_operands = {
            str(operand.tile_mem.role): operand
            for operand in abstract.operands
            if operand.tile_mem.role is not None
            and operand.tile_mem.visibility != "Shared"
        }
        shared_operands = _by_role(abstract, visibility="Shared")
        if direction == "input":
            source_operands = shared_operands
            destination_operands = local_operands
        else:
            source_operands = local_operands
            destination_operands = shared_operands
        ping_pong = schedule.for_operator(abstract.operator_id).attributes.get(
            "ping_pong", {}
        )
        slot_count = (
            int(ping_pong.get("buffer_count", 1))
            if isinstance(ping_pong, Mapping)
            else 1
        )
        input_slot = tile.ordinal % max(1, slot_count)
        output_symbolic = (
            source_operands["output"].tile_mem.symbolic_buffer_id
            if direction == "output"
            else None
        )
        output_slot = output_slots.get(str(output_symbolic), 0)
        routes = {role: role_placements[role].route for role in roles}
        maximum_hops = max(len(route) - 1 for route in routes.values())
        role_last: dict[str, str] = {}
        drafts: list[_Draft] = []
        for hop in range(maximum_hops):
            grouped: dict[str, list[str]] = {}
            for role, route in routes.items():
                if hop + 1 >= len(route):
                    continue
                path = _path(machine, route[hop], route[hop + 1])
                grouped.setdefault(path.engine, []).append(role)
            for group_index, (engine, group_roles) in enumerate(sorted(grouped.items())):
                target_id = (
                    f"{abstract.tisa_id}.target.{direction}.h{hop:02d}.g{group_index:02d}"
                )
                operands: list[TISAOperand] = []
                transfer_pairs: list[dict[str, Any]] = []
                internal = {
                    role_last[role] for role in group_roles if role in role_last
                }
                primitives: set[str] = set()
                for role in group_roles:
                    route = routes[role]
                    source_memory, target_memory = route[hop], route[hop + 1]
                    path = _path(machine, source_memory, target_memory)
                    primitives.add(str(path.transform or ("load" if direction == "input" else "store")))
                    slot = input_slot if role in {"lhs", "rhs"} else output_slot
                    source_buffer = (
                        f"{source_operands[role].tile_mem.tensor}@{source_memory}"
                        if hop == 0 and direction == "input"
                        else f"{abstract.operator_id}.{role}.slot{slot}@{source_memory}"
                    )
                    if direction == "output" and hop == 0:
                        source_buffer = f"{abstract.operator_id}.{role}.slot{slot}@{source_memory}"
                    target_buffer = (
                        f"{destination_operands[role].tile_mem.tensor}@{target_memory}"
                        if hop == len(route) - 2 and direction == "output"
                        else f"{abstract.operator_id}.{role}.slot{slot}@{target_memory}"
                    )
                    read_template = (
                        source_operands[role]
                        if direction == "output" or hop == 0
                        else destination_operands[role]
                    )
                    write_template = (
                        destination_operands[role]
                        if direction == "input" or hop == len(route) - 2
                        else source_operands[role]
                    )
                    read = _target_operand(
                        read_template,
                        name=f"{role}.hop{hop}.source",
                        memory=source_memory,
                        buffer_id=source_buffer,
                        access=AccessType.READ,
                        packed=not (direction == "input" and hop == 0),
                    )
                    write = _target_operand(
                        write_template,
                        name=f"{role}.hop{hop}.destination",
                        memory=target_memory,
                        buffer_id=target_buffer,
                        access=AccessType.WRITE,
                        packed=not (
                            direction == "output" and hop == len(route) - 2
                        ),
                    )
                    operands.extend((read, write))
                    transfer_pairs.append(
                        {
                            "role": role,
                            "source_operand": read.name,
                            "destination_operand": write.name,
                            "source_memory": source_memory,
                            "target_memory": target_memory,
                            "engine": engine,
                            "transform": path.transform,
                            "valid_bytes": read.tile_mem.valid_bytes,
                        }
                    )
                    role_last[role] = target_id
                primitive = next(iter(primitives)) if len(primitives) == 1 else (
                    "load" if direction == "input" else "store"
                )
                drafts.append(
                    _Draft(
                        abstract=abstract,
                        instruction=TISAInstruction(
                            tisa_id=target_id,
                            tile_id=abstract.tile_id,
                            operator_id=abstract.operator_id,
                            op_type=primitive,
                            operands=tuple(operands),
                            unit_map=UnitMap(engine, affinity="data"),
                            attributes={
                                **dict(abstract.attributes),
                                "target_stage_kind": "transfer_hop",
                                "transfer_direction": direction,
                                "route_hop": hop,
                                "transfer_pairs": transfer_pairs,
                            },
                            payload_ref=f"payload:{target_id}",
                        ),
                        expansion_kind="transfer_hop",
                        roles=tuple(sorted(group_roles)),
                        route_hop=hop,
                        internal_sources=tuple(sorted(internal)),
                        attributes={"transfer_pairs": transfer_pairs},
                    )
                )
        return tuple(drafts)

    def _matmul_compute(
        self,
        abstract: TISAInstruction,
        tile: TileInstance,
        schedule: ScheduleSpec,
        placement: Any,
        output_slots: Mapping[str, int],
    ) -> _Draft:
        operands: list[TISAOperand] = []
        ping_pong = schedule.for_operator(abstract.operator_id).attributes.get(
            "ping_pong", {}
        )
        count = (
            int(ping_pong.get("buffer_count", 1))
            if isinstance(ping_pong, Mapping)
            else 1
        )
        input_slot = tile.ordinal % max(1, count)
        for operand in abstract.operands:
            role = str(operand.tile_mem.role)
            role_placement = placement.operand(role)
            if role == "output":
                symbolic = operand.tile_mem.symbolic_buffer_id or ""
                slot = output_slots.get(symbolic, 0)
            else:
                slot = input_slot
            buffer_id = f"{abstract.operator_id}.{role}.slot{slot}@{role_placement.memory}"
            operands.append(
                _target_operand(
                    operand,
                    name=operand.name,
                    memory=role_placement.memory,
                    buffer_id=buffer_id,
                    access=AccessType(operand.normalized_access),
                    packed=True,
                )
            )
        target_id = f"{abstract.tisa_id}.target.compute"
        output_operand = next(
            operand for operand in operands if operand.tile_mem.role == "output"
        )
        lhs_operand = next(
            operand for operand in operands if operand.tile_mem.role == "lhs"
        )
        return _Draft(
            abstract=abstract,
            instruction=replace(
                abstract,
                tisa_id=target_id,
                operands=tuple(operands),
                unit_map=UnitMap(placement.unit, affinity="matrix"),
                dependencies=(),
                payload_ref=f"payload:{target_id}",
                attributes={
                    **dict(abstract.attributes),
                    "target_stage_kind": "compute",
                    "operand_placement": {
                        role: placement.operand(role).memory
                        for role in ("lhs", "rhs", "output")
                    },
                    "batch_tile": list(output_operand.tile_shape[:-2]),
                    "m_tile": output_operand.tile_shape[-2],
                    "n_tile": output_operand.tile_shape[-1],
                    "k_tile": lhs_operand.tile_shape[-1],
                },
            ),
            expansion_kind="compute",
            roles=("lhs", "rhs", "output"),
            route_hop=None,
            internal_sources=(),
            attributes={"compute_unit": placement.unit},
        )


def default_target_lowerer() -> TargetLowerer:
    return TargetLowerer()


def declare_internal_resources(
    target_plan: TargetPlan,
    graph: OperatorGraph,
    execution_graph: ExecutionGraph,
    payloads: Mapping[str, tuple[str, ...]],
) -> TargetPlan:
    """Add backend-private scratch declarations without redefining externals.

    Target lowering owns all externally visible operands.  A backend may have
    implementation-private state (for example Softmax max/sum workspaces);
    this hook promotes those payload declarations into the TargetPlan before
    memory allocation and payload validation.
    """

    graph_tensors = {tensor.name for tensor in graph.tensors}
    tasks = {task.task_id: task for task in execution_graph.tasks}
    updated: list[TargetInstructionPlan] = []
    symbolic_map = {
        key: set(value) for key, value in target_plan.symbolic_buffer_map.items()
    }
    for item in target_plan.instructions:
        instruction = item.instruction
        existing = list(instruction.operands)
        scratch_records: dict[tuple[Any, ...], dict[str, Any]] = {}
        for task_id in payloads.get(instruction.tisa_id, ()):
            task = tasks[task_id]
            for regions, access in (
                (task.reads, AccessType.READ.value),
                (task.writes, AccessType.WRITE.value),
            ):
                for region in regions:
                    if region.tensor in graph_tensors:
                        continue
                    valid_bytes = region.valid_bytes or (
                        math.prod(region.shape) * dtype_bytes(region.dtype, default=2)
                    )
                    key = (
                        region.tensor,
                        region.memory,
                        region.starts,
                        region.shape,
                        valid_bytes,
                    )
                    record = scratch_records.setdefault(
                        key,
                        {"region": region, "accesses": set(), "valid_bytes": valid_bytes},
                    )
                    record["accesses"].add(access)
        scratch_ids: list[str] = []
        for index, record in enumerate(scratch_records.values()):
            region = record["region"]
            accesses = record["accesses"]
            access = (
                AccessType.READ_WRITE
                if accesses == {AccessType.READ.value, AccessType.WRITE.value}
                else AccessType(next(iter(accesses)))
            )
            symbolic_id = (
                f"symbolic.scratch.{item.abstract_tisa_id}.{region.tensor}.{index}"
            )
            buffer_id = (
                f"scratch.{instruction.tisa_id}.{region.tensor}.{index}@{region.memory}"
            )
            valid_bytes = int(record["valid_bytes"])
            strides = _packed_strides(
                region.shape,
                max(1, valid_bytes // max(1, math.prod(region.shape))),
            )
            operand = TISAOperand(
                name=f"scratch.{region.tensor}.{index}",
                tile_shape=region.shape,
                tile_mem=TileMem(
                    base=buffer_id,
                    scope=f"target-instruction:{instruction.tisa_id}",
                    tensor=region.tensor,
                    dtype=region.dtype,
                    offset_bytes=0,
                    size_bytes=valid_bytes,
                    strides_bytes=strides,
                    stride_expr=(
                        " + ".join(
                            f"i{axis}*{stride}"
                            for axis, stride in enumerate(strides)
                        )
                        or "0"
                    ),
                    layout="packed",
                    logical_starts=region.starts,
                    logical_shape=region.shape,
                    memory_space=region.memory,
                    visibility="Private",
                    role="scratch",
                    owner=instruction.tisa_id,
                    domain=instruction.unit_map.unit,
                    symbolic_buffer_id=symbolic_id,
                    buffer_id=buffer_id,
                    valid_bytes=valid_bytes,
                ),
                access_type=access,
            )
            existing.append(operand)
            scratch_ids.append(buffer_id)
            symbolic_map.setdefault(symbolic_id, set()).add(buffer_id)
        updated_instruction = replace(
            instruction,
            operands=tuple(existing),
            attributes={
                **dict(instruction.attributes),
                "internal_resource_contract": (
                    "explicit_target_plan" if scratch_ids else None
                ),
                "internal_scratch_buffers": scratch_ids,
            },
        )
        updated.append(replace(item, instruction=updated_instruction))
    result = replace(
        target_plan,
        instructions=tuple(updated),
        symbolic_buffer_map={
            key: tuple(sorted(value)) for key, value in symbolic_map.items()
        },
        attributes={
            **dict(target_plan.attributes),
            "internal_resource_policy": "backend_declared_before_allocation",
        },
    )
    issues = result.validate()
    if issues:
        raise ValueError("target internal resource declaration is invalid: " + "; ".join(issues))
    return result


__all__ = [
    "TargetLowerer",
    "declare_internal_resources",
    "default_target_lowerer",
]
