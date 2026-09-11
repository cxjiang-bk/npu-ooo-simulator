from __future__ import annotations

from dataclasses import replace
from typing import Callable

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    BufferRegion,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    ScheduleSpec,
    TileGraph,
    TargetPlan,
    build_tile_graph,
)

from .elementwise import lower_elementwise_graph
from .layernorm import lower_layernorm_graph
from .matmul import LoweringResult, _regions_overlap, _root_memory, lower_matmul_graph
from .norm import lower_rmsnorm_graph
from .reduce import lower_reduce_graph
from .softmax import lower_softmax_graph
from .swiglu import lower_swiglu_graph
from .kv_cache import lower_kv_cache_graph
from .conv2d import lower_conv2d_graph
from .batch_norm import lower_batch_norm_graph
from .pool import lower_pool_graph
from .transform import lower_transform_graph
from .embedding import lower_embedding_graph


GraphLowerer = Callable[[OperatorGraph, ScheduleSpec, MachineConfig], LoweringResult]


def _apply_onchip_payload_handoffs(
    tasks_by_operator: dict[str, tuple[ExecutionTask, ...]],
    target_plan: TargetPlan,
) -> dict[str, tuple[ExecutionTask, ...]]:
    """Remove legacy root-load payloads replaced by a planned local handoff."""

    handoffs = target_plan.attributes.get("onchip_handoffs", ())
    if not handoffs:
        return tasks_by_operator
    all_tasks = tuple(
        task for tasks in tasks_by_operator.values() for task in tasks
    )
    replacement_by_dropped: dict[str, tuple[str, ...]] = {}
    dropped: set[str] = set()
    for record in handoffs:
        producer_target = str(record["producer_target_tisa_id"])
        replacements = tuple(
            task.task_id
            for task in all_tasks
            if task.attributes.get("target_tisa_id") == producer_target
        )
        if not replacements:
            raise ValueError(
                f"on-chip handoff producer target '{producer_target}' has no payload"
            )
        consumer = str(record["consumer"])
        tensor = str(record["tensor"])
        memory = str(record["memory"])
        root_memory = str(record["root_memory"])
        candidates = tuple(
            task
            for task in tasks_by_operator[consumer]
            if task.primitive in {"load", "load_transpose"}
            and any(
                region.tensor == tensor and region.memory == root_memory
                for region in task.reads
            )
            and any(
                region.tensor == tensor and region.memory == memory
                for region in task.writes
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                f"on-chip handoff {record['producer']}->{consumer} resolves to "
                f"{len(candidates)} consumer root-load payloads"
            )
        dropped_task = candidates[0]
        dropped.add(dropped_task.task_id)
        replacement_by_dropped[dropped_task.task_id] = replacements

    def expand(predecessor: str) -> tuple[str, ...]:
        return replacement_by_dropped.get(predecessor, (predecessor,))

    return {
        operator_id: tuple(
            replace(
                task,
                predecessors=tuple(
                    sorted(
                        {
                            replacement
                            for predecessor in task.predecessors
                            for replacement in expand(predecessor)
                        }
                    )
                ),
                attributes={
                    **dict(task.attributes),
                    **(
                        {"onchip_handoff_payload": True}
                        if any(
                            predecessor in replacement_by_dropped
                            for predecessor in task.predecessors
                        )
                        else {}
                    ),
                },
            )
            for task in tasks
            if task.task_id not in dropped
        )
        for operator_id, tasks in tasks_by_operator.items()
    }


def _bind_operator_payloads(
    operator: object,
    tasks: tuple[ExecutionTask, ...],
    target_plan: TargetPlan,
) -> tuple[tuple[ExecutionTask, ...], dict[str, tuple[str, ...]]]:
    """Assign every generated task to one planned target instruction."""

    operator_id = str(getattr(operator, "op_id"))
    target_items = tuple(
        item for item in target_plan.instructions
        if item.instruction.operator_id == operator_id
    )
    consumed: set[str] = set()
    payloads: dict[str, tuple[str, ...]] = {}
    ownership_sources: dict[str, str] = {}
    composite_ops = {"softmax", "rmsnorm", "layernorm", "swiglu", "kv_cache_update"}
    for item in target_items:
        instruction = item.instruction
        exact = tuple(
            task for task in tasks
            if task.attributes.get("target_tisa_id") == instruction.tisa_id
            and task.task_id not in consumed
        )
        if exact:
            selected = exact
            ownership_source = "target_instruction_plan"
        else:
            stage = str(instruction.attributes.get("tisa_stage", ""))
            semantic_op = str(instruction.attributes.get("semantic_op_type", ""))
            if semantic_op in composite_ops:
                if stage == "load":
                    primitives = {"load", "load_transpose", "copy", "transpose"}
                elif stage == "store":
                    primitives = {"store"}
                else:
                    primitives = set(instruction.attributes.get("payload_primitives", ()))
            else:
                primitives = {str(instruction.attributes.get("primitive", instruction.op_type))}
            selected = tuple(
                task for task in tasks
                if task.tile_id == instruction.tile_id
                and task.primitive in primitives
                and task.task_id not in consumed
            )
            ownership_source = "target_plan_recipe"
        if not selected:
            raise ValueError(
                f"target instruction '{instruction.tisa_id}' has no generated payload"
            )
        resources = {task.resource for task in selected}
        if resources != {instruction.unit_map.unit}:
            raise ValueError(
                f"target instruction '{instruction.tisa_id}' expects resource "
                f"'{instruction.unit_map.unit}', payload uses {sorted(resources)}"
            )
        payloads[instruction.tisa_id] = tuple(task.task_id for task in selected)
        ownership_sources[instruction.tisa_id] = ownership_source
        consumed.update(payloads[instruction.tisa_id])
        for task in selected:
            if task.attributes.get("target_tisa_id") not in {None, instruction.tisa_id}:
                raise ValueError(f"task '{task.task_id}' has conflicting target owner")
    unowned = sorted({task.task_id for task in tasks} - consumed)
    if unowned:
        raise ValueError(
            f"operator '{operator_id}' generated unowned payload tasks: "
            + ", ".join(unowned[:8])
        )
    owner = {
        task_id: tisa_id
        for tisa_id, task_ids in payloads.items()
        for task_id in task_ids
    }
    instructions = {
        item.instruction.tisa_id: item for item in target_items
    }
    updated = tuple(
        replace(
            task,
            attributes={
                **dict(task.attributes),
                "target_tisa_id": owner[task.task_id],
                "abstract_tisa_id": instructions[owner[task.task_id]].abstract_tisa_id,
                "payload_ownership": "target_plan",
                "payload_generation": ownership_sources[owner[task.task_id]],
            },
        )
        for task in tasks
    )
    return updated, payloads


class LoweringRegistry:
    """Map semantic operator types to independently testable graph lowerers."""

    def __init__(self) -> None:
        self._lowerers: dict[str, GraphLowerer] = {}

    def register(self, operator_types: tuple[str, ...], lowerer: GraphLowerer) -> None:
        for operator_type in operator_types:
            if not operator_type:
                raise ValueError("registered operator type must not be empty")
            if operator_type in self._lowerers:
                raise ValueError(f"operator type '{operator_type}' already has a lowerer")
            self._lowerers[operator_type] = lowerer

    def lowerer_for(self, operator_type: str) -> GraphLowerer:
        try:
            return self._lowerers[operator_type]
        except KeyError as exc:
            raise NotImplementedError(
                f"no registered lowering for operator type '{operator_type}'"
            ) from exc

    @property
    def supported_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._lowerers))


def default_lowering_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    registry.register(("matmul", "batched_matmul", "gemv"), lower_matmul_graph)
    registry.register(("elementwise", "residual_add"), lower_elementwise_graph)
    registry.register(("reduce",), lower_reduce_graph)
    registry.register(("softmax",), lower_softmax_graph)
    registry.register(("rmsnorm",), lower_rmsnorm_graph)
    registry.register(("layernorm",), lower_layernorm_graph)
    registry.register(("swiglu",), lower_swiglu_graph)
    registry.register(("kv_cache_update",), lower_kv_cache_graph)
    registry.register(("conv2d",), lower_conv2d_graph)
    registry.register(("batch_norm",), lower_batch_norm_graph)
    registry.register(("pool",), lower_pool_graph)
    registry.register(("reshape", "transpose", "slice", "concatenate"), lower_transform_graph)
    registry.register(("embedding",), lower_embedding_graph)
    return registry


def _single_operator_graph(graph: OperatorGraph, operator_id: str) -> OperatorGraph:
    operator = next(item for item in graph.operators if item.op_id == operator_id)
    return OperatorGraph(
        graph_id=f"{graph.graph_id}.{operator_id}",
        tensors=graph.tensors,
        operators=(operator,),
        attributes={**graph.attributes, "parent_graph": graph.graph_id},
    )


def _single_operator_schedule(schedule: ScheduleSpec, operator_id: str) -> ScheduleSpec:
    return ScheduleSpec(
        schedule_id=f"{schedule.schedule_id}.{operator_id}",
        operator_schedules=(schedule.for_operator(operator_id),),
        attributes={**schedule.attributes, "parent_schedule": schedule.schedule_id},
    )


def _root_regions(
    tasks_by_operator: dict[str, list],
    operator_id: str,
    tensor: str,
    root_memory: str,
    *,
    writes: bool,
) -> list[tuple[BufferRegion, str]]:
    matches: list[tuple[BufferRegion, str]] = []
    for task in tasks_by_operator[operator_id]:
        regions = task.writes if writes else task.reads
        matches.extend(
            (region, task.task_id)
            for region in regions
            if region.tensor == tensor and region.memory == root_memory
        )
    return matches


def lower_mixed_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
    *,
    registry: LoweringRegistry | None = None,
    tile_graph: TileGraph | None = None,
    target_plan: object | None = None,
) -> LoweringResult:
    """Lower a heterogeneous graph and connect explicit root-memory handoffs."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))

    active_registry = registry or default_lowering_registry()
    if not isinstance(target_plan, TargetPlan):
        raise ValueError("mixed target payload lowering requires a TargetPlan")
    operators = {operator.op_id: operator for operator in graph.operators}
    tasks = []
    payloads: dict[str, tuple[str, ...]] = {}
    raw_tasks_by_operator: dict[str, tuple[ExecutionTask, ...]] = {}
    statistics: dict[str, int | float] = {}
    for operator_id in graph.topological_order():
        operator = operators[operator_id]
        lowerer = active_registry.lowerer_for(operator.normalized_type)
        lowering_args = (
            _single_operator_graph(graph, operator_id),
            _single_operator_schedule(schedule, operator_id),
            machine,
        )
        if operator.normalized_type in {"matmul", "batched_matmul", "gemv"}:
            if target_plan is None:
                raise ValueError("Matmul payload lowering requires a TargetPlan")
            lowered = lowerer(*lowering_args, target_plan=target_plan)
        else:
            lowered = lowerer(*lowering_args)
        raw_tasks_by_operator[operator_id] = lowered.execution_graph.tasks
        for name, value in lowered.statistics.items():
            if name not in {"tile_count", "task_count"}:
                statistics[f"{operator_id}.{name}"] = value

    raw_tasks_by_operator = _apply_onchip_payload_handoffs(
        raw_tasks_by_operator, target_plan
    )
    for operator_id in graph.topological_order():
        operator = operators[operator_id]
        owned_tasks, operator_payloads = _bind_operator_payloads(
            operator, raw_tasks_by_operator[operator_id], target_plan
        )
        tasks.extend(owned_tasks)
        overlap = set(payloads) & set(operator_payloads)
        if overlap:
            raise ValueError(
                "duplicate target payload ownership: " + ", ".join(sorted(overlap))
            )
        payloads.update(operator_payloads)

    # GC owns the semantic tile graph.  Resolve it before backend handoffs so
    # every execution predecessor can retain the originating typed edge.
    if tile_graph is None:
        tile_graph = build_tile_graph(graph, schedule)

    tasks_by_operator = {
        operator_id: [task for task in tasks if task.operator_id == operator_id]
        for operator_id in operators
    }
    predecessor_sets = {
        task.task_id: set(task.predecessors)
        for task in tasks
    }
    root_memory = _root_memory(machine)
    cross_operator_dependencies: set[tuple[str, str, str]] = set()
    onchip_edges = {
        (str(item["producer"]), str(item["consumer"]), str(item["tensor"])): item
        for item in target_plan.attributes.get("onchip_handoffs", ())
    }
    for edge in graph.edges:
        onchip = onchip_edges.get((edge.producer, edge.consumer, edge.tensor))
        if onchip is not None:
            memory = str(onchip["memory"])
            producer_regions = [
                (region, task.task_id)
                for task in tasks_by_operator[edge.producer]
                for region in task.writes
                if region.tensor == edge.tensor and region.memory == memory
            ]
            consumer_regions = [
                (region, task.task_id)
                for task in tasks_by_operator[edge.consumer]
                for region in task.reads
                if region.tensor == edge.tensor and region.memory == memory
            ]
            if not producer_regions or not consumer_regions:
                raise ValueError(
                    f"on-chip edge {edge.producer}->{edge.consumer} for tensor "
                    f"'{edge.tensor}' has no local producer/consumer payload"
                )
            for consumer_region, consumer_task_id in consumer_regions:
                overlapping = [
                    producer_task_id
                    for producer_region, producer_task_id in producer_regions
                    if _regions_overlap(producer_region, consumer_region)
                ]
                if not overlapping:
                    raise ValueError(
                        f"on-chip consumer task '{consumer_task_id}' has no overlapping "
                        f"producer for tensor '{edge.tensor}'"
                    )
                predecessor_sets[consumer_task_id].update(overlapping)
                cross_operator_dependencies.update(
                    (producer_task_id, consumer_task_id, edge.tensor)
                    for producer_task_id in overlapping
                )
            continue
        producer_regions = _root_regions(
            tasks_by_operator,
            edge.producer,
            edge.tensor,
            root_memory,
            writes=True,
        )
        consumer_regions = _root_regions(
            tasks_by_operator,
            edge.consumer,
            edge.tensor,
            root_memory,
            writes=False,
        )
        if not producer_regions or not consumer_regions:
            raise ValueError(
                f"edge {edge.producer}->{edge.consumer} for tensor '{edge.tensor}' "
                "has no root-memory producer/consumer task"
            )
        for consumer_region, consumer_task_id in consumer_regions:
            overlapping = [
                producer_task_id
                for producer_region, producer_task_id in producer_regions
                if _regions_overlap(producer_region, consumer_region)
            ]
            if not overlapping:
                raise ValueError(
                    f"consumer task '{consumer_task_id}' has no overlapping producer region "
                    f"for tensor '{edge.tensor}'"
                )
            predecessor_sets[consumer_task_id].update(overlapping)
            cross_operator_dependencies.update(
                (producer_task_id, consumer_task_id, edge.tensor)
                for producer_task_id in overlapping
            )

    tile_dependencies = tuple(tile_graph.dependencies)
    task_by_id = {task.task_id: task for task in tasks}

    def dependency_metadata(task: object, predecessor_id: str) -> dict[str, object]:
        predecessor = task_by_id[predecessor_id]
        matches = [
            dependency
            for dependency in tile_dependencies
            if dependency.producer == predecessor.tile_id
            and dependency.consumer == task.tile_id
        ]
        if matches:
            # A pair of tiles can carry more than one semantic reason.  Keep
            # each reason in deterministic order rather than collapsing it.
            return {
                "source": "gc_tile_dependency",
                "edges": [
                    {
                        "tensor": dependency.tensor,
                        "kind": dependency.kind,
                        "hazard_kind": dependency.hazard_kind,
                        "condition": dependency.condition,
                        "producer_tile": dependency.producer,
                        "consumer_tile": dependency.consumer,
                        "producer_region": dependency._region_to_dict(dependency.producer_region),
                        "consumer_region": dependency._region_to_dict(dependency.consumer_region),
                        "provenance": dict(dependency.provenance),
                    }
                    for dependency in sorted(
                        matches,
                        key=lambda item: (
                            item.tensor or "",
                            item.hazard_kind,
                            item.condition,
                        ),
                    )
                ],
            }
        if predecessor.tile_id == task.tile_id:
            return {
                "source": "backend_payload_order",
                "edges": [
                    {
                        "kind": "control",
                        "hazard_kind": "CONTROL",
                        "condition": "payload_stage_complete",
                    }
                ],
            }
        return {
            "source": "execution_graph_edge",
            "edges": [
                {
                    "kind": "control",
                    "hazard_kind": "CONTROL",
                    "condition": "task_complete",
                }
            ],
        }

    ordered_tasks = tuple(
        replace(
            task,
            predecessors=tuple(sorted(predecessor_sets[task.task_id])),
            program_order=program_order,
            attributes={
                **dict(task.attributes),
                "dependency_provenance": {
                    predecessor_id: dependency_metadata(task, predecessor_id)
                    for predecessor_id in sorted(predecessor_sets[task.task_id])
                },
            },
        )
        for program_order, task in enumerate(tasks)
    )
    execution_graph = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=ordered_tasks,
        attributes={
            "source": "mixed-lowering-registry",
            "root_memory": root_memory,
            "handoff": (
                "target_plan_onchip_and_root"
                if onchip_edges
                else "root_memory"
            ),
            "onchip_handoff_count": len(onchip_edges),
            "cross_operator_dependency_count": len(cross_operator_dependencies),
            "registered_operator_types": list(active_registry.supported_types),
        },
    )
    issues = execution_graph.validate()
    if issues:
        raise ValueError("; ".join(issues))
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution_graph,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(execution_graph.tasks),
            "cross_operator_dependency_count": len(cross_operator_dependencies),
            **statistics,
        },
        payloads=payloads,
    )
