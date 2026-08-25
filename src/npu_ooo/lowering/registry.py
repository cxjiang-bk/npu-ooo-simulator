from __future__ import annotations

from dataclasses import replace
from typing import Callable

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    BufferRegion,
    ExecutionGraph,
    OperatorGraph,
    ScheduleSpec,
    TileGraph,
    build_tile_graph,
)

from .elementwise import lower_elementwise_graph
from .layernorm import lower_layernorm_graph
from .matmul import LoweringResult, _regions_overlap, _root_memory, lower_matmul_graph
from .norm import lower_rmsnorm_graph
from .reduce import lower_reduce_graph
from .softmax import lower_softmax_graph


GraphLowerer = Callable[[OperatorGraph, ScheduleSpec, MachineConfig], LoweringResult]


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
) -> LoweringResult:
    """Lower a heterogeneous graph and connect explicit root-memory handoffs."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))

    active_registry = registry or default_lowering_registry()
    operators = {operator.op_id: operator for operator in graph.operators}
    tasks = []
    statistics: dict[str, int | float] = {}
    for operator_id in graph.topological_order():
        operator = operators[operator_id]
        lowerer = active_registry.lowerer_for(operator.normalized_type)
        lowered = lowerer(
            _single_operator_graph(graph, operator_id),
            _single_operator_schedule(schedule, operator_id),
            machine,
        )
        tasks.extend(lowered.execution_graph.tasks)
        for name, value in lowered.statistics.items():
            if name not in {"tile_count", "task_count"}:
                statistics[f"{operator_id}.{name}"] = value

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
    for edge in graph.edges:
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

    ordered_tasks = tuple(
        replace(
            task,
            predecessors=tuple(sorted(predecessor_sets[task.task_id])),
            program_order=program_order,
        )
        for program_order, task in enumerate(tasks)
    )
    execution_graph = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=ordered_tasks,
        attributes={
            "source": "mixed-lowering-registry",
            "root_memory": root_memory,
            "handoff": "root_memory",
            "cross_operator_dependency_count": len(cross_operator_dependencies),
            "registered_operator_types": list(active_registry.supported_types),
        },
    )
    issues = execution_graph.validate()
    if issues:
        raise ValueError("; ".join(issues))
    # TISA-first codegen may already own the semantic TileGraph.  Reusing it
    # keeps backend payload lowering from silently choosing a second tiling
    # result; the optional argument preserves the legacy lowering API.
    if tile_graph is None:
        tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution_graph,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(execution_graph.tasks),
            "cross_operator_dependency_count": len(cross_operator_dependencies),
            **statistics,
        },
    )
