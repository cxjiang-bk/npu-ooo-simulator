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
                        abstract, machine, root=root
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
                drafts.append(draft)

        (
            drafts,
            onchip_handoffs,
            elided_replacements,
            onchip_rejections,
        ) = self._apply_onchip_handoffs(graph, machine, drafts, root=root)
        for draft in drafts:
            for operand in draft.instruction.operands:
                record_operand(operand)

        mapped: dict[str, list[str]] = {}
        for draft in drafts:
            mapped.setdefault(draft.abstract.tisa_id, []).append(
                draft.instruction.tisa_id
            )
        terminals: dict[str, tuple[str, ...]] = {}
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
        unresolved = dict(elided_replacements)
        while unresolved:
            progressed = False
            for abstract_id, replacement in tuple(unresolved.items()):
                target_id = replacement.get("target_tisa_id")
                source_abstract = replacement.get("source_abstract_tisa_id")
                if target_id:
                    terminals[abstract_id] = (str(target_id),)
                elif source_abstract in terminals:
                    terminals[abstract_id] = terminals[str(source_abstract)]
                else:
                    continue
                del unresolved[abstract_id]
                progressed = True
            if not progressed:
                raise ValueError(
                    "cannot resolve elided target dependencies: "
                    + ", ".join(sorted(unresolved))
                )
        resolved_handoffs = []
        for record in onchip_handoffs:
            producer_terminal = terminals[str(record["producer_store_abstract_id"])]
            if len(producer_terminal) != 1:
                raise ValueError("on-chip handoff requires one producer terminal")
            resolved_handoffs.append(
                {**record, "producer_target_tisa_id": producer_terminal[0]}
            )
        handoff_target_ids = {
            str(item["producer_target_tisa_id"]) for item in resolved_handoffs
        }

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
                    **(
                        {
                            "onchip_handoff_producer": True,
                            "handoff_tensor": next(
                                item["tensor"]
                                for item in resolved_handoffs
                                if item["producer_target_tisa_id"]
                                == draft.instruction.tisa_id
                            ),
                        }
                        if draft.instruction.tisa_id in handoff_target_ids
                        else {}
                    ),
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
                "onchip_handoff_policy": machine.attributes.get(
                    "onchip_handoff_policy", "root_memory"
                ),
                "onchip_handoffs": resolved_handoffs,
                "onchip_handoff_rejections": onchip_rejections,
                "elided_abstract_instructions": {
                    key: {
                        **dict(value),
                        "resolved_target_tisa_ids": list(terminals[key]),
                    }
                    for key, value in elided_replacements.items()
                },
            },
        )
        issues = target_plan.validate(abstract_program)
        if issues:
            raise ValueError("target plan is invalid: " + "; ".join(issues))
        return target_plan

    @staticmethod
    def _apply_onchip_handoffs(
        graph: OperatorGraph,
        machine: MachineConfig,
        drafts: list[_Draft],
        *,
        root: str,
    ) -> tuple[
        list[_Draft],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Elide one proven root round-trip for a compatible Attention edge."""

        policy = str(machine.attributes.get("onchip_handoff_policy", "root_memory"))
        if policy == "root_memory":
            return drafts, [], {}, []
        if policy != "attention_single_consumer":
            raise ValueError(f"unknown on-chip handoff policy '{policy}'")
        operators = {item.op_id: item for item in graph.operators}
        fanout: dict[str, int] = {}
        for edge in graph.edges:
            fanout[edge.tensor] = fanout.get(edge.tensor, 0) + 1
        retained = list(drafts)
        handoffs: list[dict[str, Any]] = []
        elided: dict[str, dict[str, Any]] = {}
        rejections: list[dict[str, Any]] = []

        def reject(edge: Any, reason: str) -> None:
            rejections.append(
                {
                    "producer": edge.producer,
                    "consumer": edge.consumer,
                    "tensor": edge.tensor,
                    "reason": reason,
                }
            )

        for edge in graph.edges:
            producer = operators[edge.producer]
            consumer = operators[edge.consumer]
            if producer.normalized_type not in {"matmul", "batched_matmul", "gemv"}:
                continue
            if consumer.normalized_type != "softmax":
                continue
            if fanout.get(edge.tensor, 0) != 1:
                reject(edge, "fanout_is_not_one")
                continue
            producer_stores = [
                item for item in retained
                if item.abstract.operator_id == edge.producer
                and item.abstract.attributes.get("tisa_stage") == "store"
            ]
            consumer_loads = [
                item for item in retained
                if item.abstract.operator_id == edge.consumer
                and item.abstract.attributes.get("tisa_stage") == "load"
            ]
            consumer_computes = [
                item for item in retained
                if item.abstract.operator_id == edge.consumer
                and item.abstract.attributes.get("tisa_stage") == "compute"
            ]
            root_stores = [
                item for item in producer_stores
                if any(
                    operand.tile_mem.tensor == edge.tensor
                    and operand.tile_mem.physical_space == root
                    and operand.normalized_access in {"write", "read_write"}
                    for operand in item.instruction.operands
                )
            ]
            if len(root_stores) != 1 or len(consumer_loads) != 1 or len(consumer_computes) != 1:
                reject(edge, "requires_single_store_load_compute_tile")
                continue
            root_store, consumer_load, consumer_compute = (
                root_stores[0], consumer_loads[0], consumer_computes[0]
            )
            producer_sources = [
                operand for operand in root_store.instruction.operands
                if operand.tile_mem.tensor == edge.tensor
                and operand.tile_mem.physical_space != root
                and operand.normalized_access in {"read", "read_write"}
            ]
            consumer_destinations = [
                operand for operand in consumer_load.instruction.operands
                if operand.tile_mem.tensor == edge.tensor
                and operand.tile_mem.physical_space != root
                and operand.normalized_access in {"write", "read_write"}
            ]
            compute_inputs = [
                operand for operand in consumer_compute.instruction.operands
                if operand.tile_mem.tensor == edge.tensor
                and operand.normalized_access in {"read", "read_write"}
            ]
            if not (
                len(producer_sources) == len(consumer_destinations) == len(compute_inputs) == 1
            ):
                reject(edge, "operand_roles_are_ambiguous")
                continue
            source = producer_sources[0]
            destination = consumer_destinations[0]
            compute_input = compute_inputs[0]
            if source.tile_mem.role != "output" or destination.tile_mem.role != "input":
                reject(edge, "operand_role_mismatch")
                continue
            if source.tile_mem.physical_space != destination.tile_mem.physical_space:
                reject(edge, "producer_consumer_memory_mismatch")
                continue
            try:
                consumer_rule = machine.operation_class(consumer.normalized_type)
            except KeyError:
                reject(edge, "consumer_target_class_missing")
                continue
            if consumer_rule.local_memory != source.tile_mem.physical_space:
                reject(edge, "consumer_eu_cannot_access_producer_memory")
                continue
            if (
                source.tile_shape != destination.tile_shape
                or source.tile_shape != compute_input.tile_shape
            ):
                reject(edge, "tile_geometry_mismatch")
                continue
            if source.tile_mem.dtype != destination.tile_mem.dtype:
                reject(edge, "dtype_mismatch")
                continue
            if source.tile_mem.layout != destination.tile_mem.layout:
                reject(edge, "layout_mismatch")
                continue
            capacity = machine.memory(source.tile_mem.physical_space).capacity_bytes
            required = int(source.tile_mem.size_bytes or source.tile_mem.valid_bytes or 0)
            if capacity is not None and required > capacity:
                reject(edge, "producer_tile_exceeds_local_capacity")
                continue
            if not root_store.internal_sources and not root_store.abstract.dependencies:
                reject(edge, "producer_store_has_no_replacement_token")
                continue
            producer_abstract_id = root_store.abstract.tisa_id
            consumer_abstract_id = consumer_load.abstract.tisa_id
            source_buffer = str(source.tile_mem.buffer_id)
            rewritten_operands = tuple(
                replace(
                    operand,
                    tile_mem=replace(
                        operand.tile_mem,
                        base=source_buffer,
                        buffer_id=source_buffer,
                        symbolic_buffer_id=source.tile_mem.symbolic_buffer_id,
                        memory_space=source.tile_mem.physical_space,
                        layout=source.tile_mem.layout,
                        strides_bytes=source.tile_mem.strides_bytes,
                        size_bytes=source.tile_mem.size_bytes,
                        valid_bytes=source.tile_mem.valid_bytes,
                    ),
                )
                if operand is compute_input
                else operand
                for operand in consumer_compute.instruction.operands
            )
            replacement_compute = replace(
                consumer_compute,
                instruction=replace(
                    consumer_compute.instruction,
                    operands=rewritten_operands,
                    attributes={
                        **dict(consumer_compute.instruction.attributes),
                        "onchip_handoff_consumer": True,
                        "handoff_tensor": edge.tensor,
                        "handoff_buffer_id": source_buffer,
                    },
                ),
                attributes={
                    **dict(consumer_compute.attributes),
                    "onchip_handoff_consumer": True,
                },
            )
            retained = [
                replacement_compute if item is consumer_compute else item
                for item in retained
                if item is not root_store and item is not consumer_load
            ]
            store_replacement: dict[str, Any]
            if root_store.internal_sources:
                store_replacement = {
                    "target_tisa_id": root_store.internal_sources[-1]
                }
            else:
                source_abstract = next(
                    dependency.source
                    for dependency in root_store.abstract.dependencies
                )
                store_replacement = {"source_abstract_tisa_id": source_abstract}
            elided[producer_abstract_id] = {
                **store_replacement,
                "reason": "onchip_handoff_root_store_elided",
                "tensor": edge.tensor,
            }
            elided[consumer_abstract_id] = {
                "source_abstract_tisa_id": producer_abstract_id,
                "reason": "onchip_handoff_root_load_elided",
                "tensor": edge.tensor,
            }
            handoffs.append(
                {
                    "producer": edge.producer,
                    "consumer": edge.consumer,
                    "tensor": edge.tensor,
                    "memory": source.tile_mem.physical_space,
                    "root_memory": root,
                    "buffer_id": source_buffer,
                    "dtype": source.tile_mem.dtype,
                    "layout": source.tile_mem.layout,
                    "tile_shape": list(source.tile_shape),
                    "valid_bytes": int(source.tile_mem.valid_bytes or required),
                    "producer_store_abstract_id": producer_abstract_id,
                    "consumer_load_abstract_id": consumer_abstract_id,
                    "removed_target_tisa_ids": [
                        root_store.instruction.tisa_id,
                        consumer_load.instruction.tisa_id,
                    ],
                    "checks": {
                        "fanout": 1,
                        "same_memory": True,
                        "eu_reachable": True,
                        "role_compatible": True,
                        "layout_compatible": True,
                        "dtype_compatible": True,
                        "tile_geometry_compatible": True,
                        "capacity_checked": True,
                        "slot_reuse_guarded_by_memory_plan": True,
                    },
                }
            )
        return retained, handoffs, elided, rejections

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
    ) -> _Draft:
        semantic_type = str(abstract.attributes.get("semantic_op_type", ""))
        try:
            rule = machine.operation_class(semantic_type)
        except KeyError as exc:
            raise ValueError(
                f"machine '{machine.config_id}' has no explicit target class for "
                f"operation '{semantic_type}'"
            ) from exc
        local = rule.local_memory
        if not rule.direct and (
            len(rule.input_route) != 2 or len(rule.output_route) != 2
        ):
            raise ValueError(
                f"generic target class '{rule.class_id}' currently requires one-hop routes; "
                "use an operation-specific target lowerer for multi-hop expansion"
            )
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
        if rule.direct:
            unit = rule.unit
        elif stage == "load":
            path = _path(machine, rule.input_route[0], rule.input_route[1])
            unit = path.engine
        elif stage == "store":
            path = _path(machine, rule.output_route[0], rule.output_route[1])
            unit = path.engine
        else:
            unit = rule.unit
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
                    "target_class": rule.class_id,
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
