from __future__ import annotations

"""TISA-first compiler and analytical payload backend.

The semantic builder in this module deliberately does not inspect
``ExecutionTask``.  It derives scheduler-visible stages from the canonical
operator, schedule and tile graph.  A backend is then responsible for
materializing those stages as a payload and proving that every generated
primitive belongs to exactly one stage.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    OperatorGraph,
    ScheduleSpec,
    TISADependency,
    TISAInstruction,
    TISAProgram,
    TISAOperand,
    TileGraph,
    TileInstance,
    TileMem,
    UnitMap,
)
from npu_ooo.lowering import LoweringRegistry, default_lowering_registry, lower_mixed_graph


@dataclass(frozen=True)
class TISAStage:
    """Backend-independent stage contract for one semantic tile."""

    key: str
    primitive: str
    unit_map: UnitMap
    ordinal: int
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class TISAFirstResult:
    program: TISAProgram
    artifact: BackendArtifact
    statistics: Mapping[str, int | float]


def _unit_map(primitive: str) -> UnitMap:
    if primitive in {"load", "load_transpose", "store"}:
        return UnitMap("dma", affinity="data")
    if primitive in {"matmul"}:
        return UnitMap("tensor", affinity="matrix")
    return UnitMap("vector", affinity="vector")


def _operator_tiles(tile_graph: TileGraph, operator_id: str) -> tuple[TileInstance, ...]:
    return tuple(
        sorted(
            (tile for tile in tile_graph.tiles if tile.operator_id == operator_id),
            key=lambda tile: (tile.ordinal, tile.tile_id),
        )
    )


def _reduction_name(operator: Any) -> str | None:
    return operator.reduction_dims[0][0] if operator.reduction_dims else None


def _row_key(tile: TileInstance, operator: Any) -> tuple[int, ...]:
    reduction = _reduction_name(operator)
    return tuple(
        tile.bound_map[name][0]
        for name, _ in operator.iteration_dims
        if name != reduction
    )


def _is_first_reduction_tile(tile: TileInstance, operator: Any) -> bool:
    reduction = _reduction_name(operator)
    if reduction is None:
        return True
    return tile.bound_map[reduction][0] == 0


def _is_last_reduction_tile(tile: TileInstance, operator: Any) -> bool:
    reduction = _reduction_name(operator)
    if reduction is None:
        return True
    extent = dict(operator.reduction_dims)[reduction]
    return tile.bound_map[reduction][1] == extent


def _stages_for_tile(operator: Any, tile: TileInstance) -> tuple[TISAStage, ...]:
    op_type = operator.normalized_type
    stages: list[tuple[str, str]] = []
    if op_type in {"matmul", "batched_matmul", "gemv"}:
        stages.append(("load", "load"))
        if operator.attributes.get("rhs_transposed"):
            stages.append(("load_transpose", "load_transpose"))
        stages.append(("compute", "matmul"))
        if _is_last_reduction_tile(tile, operator):
            stages.append(("store", "store"))
    elif op_type in {"elementwise", "residual_add"}:
        stages = [("load", "load"), ("compute", "elementwise"), ("store", "store")]
    elif op_type == "reduce":
        stages = [("load", "load"), ("compute", "reduce")]
        if _is_last_reduction_tile(tile, operator):
            stages.append(("store", "store"))
    elif op_type == "softmax":
        stages = [
            ("load", "load"),
            ("max", "reduce_max"),
            ("exp", "exp"),
            ("sum", "reduce_sum"),
            ("normalize", "normalize"),
            ("store", "store"),
        ]
    elif op_type == "rmsnorm":
        stages = [
            ("load", "load"),
            ("square", "square"),
            ("sum_square", "reduce_sum_square"),
            ("normalize", "rmsnorm"),
            ("store", "store"),
        ]
    elif op_type == "layernorm":
        stages = [("load", "load"), ("sum", "reduce_sum")]
        if _is_first_reduction_tile(tile, operator):
            stages.append(("mean", "layernorm_mean"))
        stages.extend(
            [
                ("center", "center"),
                ("variance", "reduce_sum_square"),
                ("normalize", "layernorm"),
                ("store", "store"),
            ]
        )
    else:
        raise NotImplementedError(
            f"TISA-first semantic builder does not support operator '{op_type}'"
        )
    return tuple(
        TISAStage(
            key=key,
            primitive=primitive,
            unit_map=_unit_map(primitive),
            ordinal=index,
            attributes={"tisa_stage": key, "primitive": primitive},
        )
        for index, (key, primitive) in enumerate(stages)
    )


def _operand_shape(operator: Any, tile: TileInstance) -> tuple[int, ...]:
    shape = tuple(stop - start for _name, start, stop in tile.bounds)
    return shape or (1,)


def _stage_operands(operator: Any, tile: TileInstance, stage: TISAStage) -> tuple[TISAOperand, ...]:
    shape = _operand_shape(operator, tile)
    is_load = stage.primitive in {"load", "load_transpose"}
    is_store = stage.primitive == "store"
    names: list[tuple[str, AccessType]] = []
    if not is_store:
        names.extend((name, AccessType.READ) for name in operator.inputs)
    if not is_load:
        names.extend((name, AccessType.WRITE) for name in operator.outputs)
    if not names:
        names.append((operator.outputs[0], AccessType.WRITE))
    operands: list[TISAOperand] = []
    seen: set[tuple[str, str]] = set()
    for index, (name, access) in enumerate(names):
        key = (name, access.value)
        if key in seen:
            continue
        seen.add(key)
        operands.append(
            TISAOperand(
                name=f"{name}:{access.value}:{index}",
                tile_shape=shape,
                tile_mem=TileMem(
                    base=name,
                    scope="logical",
                    tensor=name,
                    offset_bytes=None,
                    size_bytes=None,
                ),
                access_type=access,
            )
        )
    return tuple(operands)


class TISASemanticBuilder:
    """Build TISA descriptors from semantic graph and tile information only."""

    def build(
        self,
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        tile_graph: TileGraph,
        machine: MachineConfig,
        *,
        program_id: str,
    ) -> TISAProgram:
        graph_issues = graph.validate()
        schedule_issues = schedule.validate(graph)
        tile_issues = tile_graph.validate()
        machine_issues = machine.validate()
        if graph_issues or schedule_issues or tile_issues or machine_issues:
            raise ValueError(
                "; ".join((*graph_issues, *schedule_issues, *tile_issues, *machine_issues))
            )
        operators = {operator.op_id: operator for operator in graph.operators}
        tiles = {tile.tile_id: tile for tile in tile_graph.tiles}
        stage_map: dict[tuple[str, str], TISAStage] = {}
        instructions: list[TISAInstruction] = []
        source_order: dict[str, int] = {}
        for order, tile_id in enumerate(tile_graph.topological_order()):
            tile = tiles[tile_id]
            operator = operators[tile.operator_id]
            for stage in _stages_for_tile(operator, tile):
                tisa_id = f"tisa.{tile_id}.s{stage.ordinal:02d}"
                stage_map[(tile_id, stage.key)] = stage
                source_order[tisa_id] = order * 100 + stage.ordinal
                instructions.append(
                    TISAInstruction(
                        tisa_id=tisa_id,
                        tile_id=tile_id,
                        operator_id=operator.op_id,
                        # ``op_type`` names the scheduler-visible stage.  The
                        # semantic operator family remains explicit in the
                        # attributes so backend-independent analyses retain
                        # the composite identity (e.g. softmax/rmsnorm).
                        op_type=stage.primitive,
                        operands=_stage_operands(operator, tile, stage),
                        unit_map=stage.unit_map,
                        attributes={
                            **dict(tile.attributes),
                            **dict(stage.attributes),
                            "semantic_boundary": "tile",
                            "semantic_tile_id": tile_id,
                            "semantic_op_type": operator.normalized_type,
                            "source_program_order": source_order[tisa_id],
                        },
                        payload_ref=f"payload:{tisa_id}",
                    )
                )

        by_id = {instruction.tisa_id: instruction for instruction in instructions}
        dependencies: dict[str, dict[str, str]] = {tisa_id: {} for tisa_id in by_id}

        def add_dependency(target: str, source: str, kind: str = "RAW") -> None:
            if target == source:
                return
            current = dependencies[target].get(source)
            if current is None or (current != "RAW" and kind == "RAW"):
                dependencies[target][source] = kind

        def instruction_id(tile_id: str, key: str) -> str:
            try:
                stage = stage_map[(tile_id, key)]
            except KeyError as exc:
                raise ValueError(f"missing TISA stage '{key}' for tile '{tile_id}'") from exc
            return f"tisa.{tile_id}.s{stage.ordinal:02d}"

        for tile in tile_graph.tiles:
            stages = _stages_for_tile(operators[tile.operator_id], tile)
            for previous, current in zip(stages, stages[1:]):
                add_dependency(
                    instruction_id(tile.tile_id, current.key),
                    instruction_id(tile.tile_id, previous.key),
                )

        # Graph edges are represented at the semantic tile boundary.  The
        # producer's terminal stage is sufficient because backend payloads are
        # bound only after the descriptor program has been built.
        for dependency in tile_graph.dependencies:
            producer = tiles[dependency.producer]
            consumer = tiles[dependency.consumer]
            producer_stages = _stages_for_tile(operators[producer.operator_id], producer)
            consumer_stages = _stages_for_tile(operators[consumer.operator_id], consumer)
            add_dependency(
                instruction_id(consumer.tile_id, consumer_stages[0].key),
                instruction_id(producer.tile_id, producer_stages[-1].key),
            )

        # Reduction barriers and matrix partial accumulation are semantic
        # dependencies, not artifacts of a particular primitive graph.
        by_operator: dict[str, list[TileInstance]] = {}
        for tile in tile_graph.tiles:
            by_operator.setdefault(tile.operator_id, []).append(tile)
        for operator_id, op_tiles in by_operator.items():
            operator = operators[operator_id]
            op_type = operator.normalized_type
            rows: dict[tuple[int, ...], list[TileInstance]] = {}
            for tile in op_tiles:
                rows.setdefault(_row_key(tile, operator), []).append(tile)
            reduction = _reduction_name(operator)
            for row_tiles in rows.values():
                ordered = sorted(
                    row_tiles,
                    key=lambda tile: tile.bound_map[reduction][0] if reduction else tile.ordinal,
                )
                if op_type in {"matmul", "batched_matmul", "gemv"} and reduction:
                    output_dims = tuple(name for name, _ in operator.iteration_dims if name != reduction)
                    groups: dict[tuple[int, ...], list[TileInstance]] = {}
                    for tile in ordered:
                        groups.setdefault(
                            tuple(tile.bound_map[name][0] for name in output_dims), []
                        ).append(tile)
                    for output_tiles in groups.values():
                        output_tiles.sort(key=lambda tile: tile.bound_map[reduction][0])
                        for previous, current in zip(output_tiles, output_tiles[1:]):
                            add_dependency(
                                instruction_id(current.tile_id, "compute"),
                                instruction_id(previous.tile_id, "compute"),
                            )
                barrier_stage = {
                    "reduce": "compute",
                    "softmax": "max",
                    "rmsnorm": "sum_square",
                    "layernorm": "sum",
                }.get(op_type)
                if barrier_stage and reduction:
                    for previous, current in zip(ordered, ordered[1:]):
                        add_dependency(
                            instruction_id(current.tile_id, barrier_stage),
                            instruction_id(previous.tile_id, barrier_stage),
                        )
                if op_type == "softmax" and reduction:
                    for previous, current in zip(ordered, ordered[1:]):
                        add_dependency(
                            instruction_id(current.tile_id, "sum"),
                            instruction_id(previous.tile_id, "sum"),
                        )
                # LayerNorm has two row-wise reductions.  The variance pass
                # must observe the same tile ordering as the mean pass so the
                # backend's reduce_sum_square edge has a semantic owner edge.
                if op_type == "layernorm" and reduction:
                    for previous, current in zip(ordered, ordered[1:]):
                        add_dependency(
                            instruction_id(current.tile_id, "variance"),
                            instruction_id(previous.tile_id, "variance"),
                        )

                if op_type == "softmax":
                    final = ordered[-1]
                    for tile in ordered:
                        add_dependency(
                            instruction_id(tile.tile_id, "exp"),
                            instruction_id(final.tile_id, "max"),
                        )
                        add_dependency(
                            instruction_id(tile.tile_id, "normalize"),
                            instruction_id(final.tile_id, "sum"),
                        )
                elif op_type == "rmsnorm":
                    final = ordered[-1]
                    for tile in ordered:
                        add_dependency(
                            instruction_id(tile.tile_id, "normalize"),
                            instruction_id(final.tile_id, "sum_square"),
                        )
                elif op_type == "layernorm":
                    first = ordered[0]
                    final = ordered[-1]
                    mean_id = instruction_id(first.tile_id, "mean")
                    for tile in ordered:
                        add_dependency(instruction_id(tile.tile_id, "center"), mean_id)
                        add_dependency(
                            instruction_id(tile.tile_id, "normalize"),
                            instruction_id(final.tile_id, "variance"),
                        )
                    add_dependency(mean_id, instruction_id(final.tile_id, "sum"))

        # Stable topological order is part of the descriptor contract.
        successors = {tisa_id: [] for tisa_id in by_id}
        indegree = {tisa_id: 0 for tisa_id in by_id}
        for target, sources in dependencies.items():
            for source in sources:
                successors[source].append(target)
                indegree[target] += 1
        ready = sorted(
            (tisa_id for tisa_id, degree in indegree.items() if degree == 0),
            key=lambda tisa_id: (source_order[tisa_id], tisa_id),
        )
        ordered: list[TISAInstruction] = []
        while ready:
            current = ready.pop(0)
            instruction = by_id[current]
            ordered.append(
                replace(
                    instruction,
                    dependencies=tuple(
                        TISADependency(source=source, kind=kind)
                        for source, kind in sorted(dependencies[current].items())
                    ),
                    attributes={
                        **dict(instruction.attributes),
                        "program_order": len(ordered),
                    },
                )
            )
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=lambda tisa_id: (source_order[tisa_id], tisa_id))
        if len(ordered) != len(instructions):
            raise ValueError("TISA-first semantic dependency graph contains a cycle")
        program = TISAProgram(
            program_id=program_id,
            instructions=tuple(ordered),
            attributes={
                "source": "tile-graph-semantic-builder",
                "scheduler_granularity": "tisa_tile",
                "codegen_direction": "tilegraph->tisa->backend-payload",
            },
        )
        issues = program.validate()
        if issues:
            raise ValueError("TISA-first program construction failed: " + "; ".join(issues))
        return program


class AnalyticalBackendCodegen:
    """Materialize an already-built TISA program as analytical payloads."""

    def lower(
        self,
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        tile_graph: TileGraph,
        machine: MachineConfig,
        *,
        program: TISAProgram,
        registry: LoweringRegistry | None = None,
    ) -> TISAFirstResult:
        lowering = lower_mixed_graph(
            graph,
            schedule,
            machine,
            registry=registry or default_lowering_registry(),
            tile_graph=tile_graph,
        )
        tasks_by_stage: dict[tuple[str, str], list[str]] = {}
        for task in lowering.execution_graph.tasks:
            tasks_by_stage.setdefault((task.tile_id, task.primitive), []).append(task.task_id)
        payloads: dict[str, tuple[str, ...]] = {}
        consumed: set[str] = set()
        for instruction in program.instructions:
            stage = str(instruction.attributes.get("primitive", ""))
            task_ids = tuple(tasks_by_stage.get((instruction.tile_id, stage), ()))
            if not task_ids:
                raise ValueError(
                    f"analytical backend has no payload for TISA instruction '{instruction.tisa_id}' "
                    f"stage '{stage}' on tile '{instruction.tile_id}'"
                )
            payloads[instruction.tisa_id] = task_ids
            consumed.update(task_ids)
        all_tasks = {task.task_id for task in lowering.execution_graph.tasks}
        unbound = sorted(all_tasks - consumed)
        if unbound:
            raise ValueError(
                "analytical backend generated primitive tasks without TISA ownership: "
                + ", ".join(unbound[:8])
            )
        artifact = BackendArtifact(
            artifact_id=f"{graph.graph_id}.analytical-tisa-first",
            program=program,
            execution_graph=lowering.execution_graph,
            payloads=payloads,
            backend="analytical",
            attributes={
                "calibration_status": "analytical",
                "codegen_direction": "tilegraph->tisa->analytical-payload",
                "lowering_statistics": lowering.statistics,
            },
        )
        issues = artifact.validate()
        if issues:
            raise ValueError("TISA-first backend artifact is invalid: " + "; ".join(issues))
        return TISAFirstResult(
            program=program,
            artifact=artifact,
            statistics=lowering.statistics,
        )


def compile_tisa_first(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    tile_graph: TileGraph,
    machine: MachineConfig,
    *,
    program_id: str,
    registry: LoweringRegistry | None = None,
) -> TISAFirstResult:
    program = TISASemanticBuilder().build(
        graph,
        schedule,
        tile_graph,
        machine,
        program_id=program_id,
    )
    return AnalyticalBackendCodegen().lower(
        graph,
        schedule,
        tile_graph,
        machine,
        program=program,
        registry=registry,
    )
