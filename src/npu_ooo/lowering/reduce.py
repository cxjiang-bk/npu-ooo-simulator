from __future__ import annotations

import math
from typing import Any

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BufferRegion,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    ScheduleSpec,
    TileInstance,
    build_tile_graph,
)
from npu_ooo.ir.model import ModelInstance

from .matmul import (
    LoweringResult,
    _local_memory,
    _region,
    _regions_overlap,
    _root_memory,
    _transfer_timing,
    _unit_for,
)


def _reduce_timing(machine: MachineConfig, elements: int) -> tuple[float, float, str]:
    unit = _unit_for(machine, "reduce")
    configured_rate = unit.attributes.get("elements_per_cycle", unit.attributes.get("lanes", 1))
    rate = float(configured_rate) if isinstance(configured_rate, (int, float)) and configured_rate > 0 else 1.0
    duration = unit.latency_cycles + math.ceil(elements / rate)
    return float(duration), float(unit.initiation_interval_cycles), unit.name


def _reduce_regions(operator, tensors: dict[str, Any], tile: TileInstance, memory: str) -> tuple[BufferRegion, BufferRegion]:
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise ValueError(f"reduce operator '{operator.op_id}' requires one input and one output")
    iteration = tuple(name for name, _ in operator.iteration_dims)
    reduction = tuple(name for name, _ in operator.reduction_dims)
    if len(iteration) != 1 or len(reduction) != 1:
        raise ValueError(f"reduce operator '{operator.op_id}' currently requires one iteration and one reduction dimension")
    input_tensor = tensors[operator.inputs[0]]
    output_tensor = tensors[operator.outputs[0]]
    bounds = tile.bound_map
    input_starts = (bounds[iteration[0]][0], bounds[reduction[0]][0])
    input_shape = (
        bounds[iteration[0]][1] - bounds[iteration[0]][0],
        bounds[reduction[0]][1] - bounds[reduction[0]][0],
    )
    output_starts = (bounds[iteration[0]][0],)
    output_shape = (bounds[iteration[0]][1] - bounds[iteration[0]][0],)
    return (
        _region(input_tensor, memory, input_starts, input_shape, AccessType.READ),
        _region(output_tensor, memory, output_starts, output_shape, AccessType.WRITE),
    )


def lower_reduce_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower one-dimensional reductions into load -> ARU reduce -> store tasks."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    local = _local_memory(machine, root)
    tasks: list[ExecutionTask] = []
    producer_stores: dict[str, list[tuple[BufferRegion, str]]] = {}
    task_order = 0
    input_elements = 0
    transfer_bytes = 0

    for operator_id in graph.topological_order():
        operator = next(operator for operator in graph.operators if operator.op_id == operator_id)
        if operator.normalized_type != "reduce":
            raise NotImplementedError(f"reduce lowering does not support '{operator.normalized_type}'")
        op_schedule = schedule.for_operator(operator_id)
        from npu_ooo.ir.tile import enumerate_operator_tiles

        tiles = enumerate_operator_tiles(operator, op_schedule)
        reduction_name = operator.reduction_dims[0][0]
        reduction_extent = dict(operator.reduction_dims)[reduction_name]
        partials: dict[tuple[int, ...], str] = {}
        for tile in tiles:
            input_global, output_global = _reduce_regions(operator, tensors, tile, root)
            input_local, output_local_write = _reduce_regions(operator, tensors, tile, local)
            input_local = BufferRegion(**{**input_local.__dict__, "access": AccessType.READ})
            output_local_read = BufferRegion(**{**output_local_write.__dict__, "access": AccessType.READ})
            output_local_access = AccessType.READ_WRITE if tile.bound_map[reduction_name][0] > 0 else AccessType.WRITE
            output_local_write = BufferRegion(**{**output_local_write.__dict__, "access": output_local_access})
            load_id = f"{tile.tile_id}.load"
            predecessors = {
                store_id
                for region, store_id in producer_stores.get(operator.inputs[0], [])
                if _regions_overlap(region, input_global)
            }
            load_duration, load_ii, load_unit = _transfer_timing(machine, root, local, input_global.size_bytes)
            tasks.append(
                ExecutionTask(
                    task_id=load_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="load",
                    resource=load_unit,
                    reads=(input_global,),
                    writes=(BufferRegion(**{**input_local.__dict__, "access": AccessType.WRITE}),),
                    predecessors=tuple(sorted(predecessors)),
                    duration_cycles=load_duration,
                    initiation_interval_cycles=load_ii,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={"iteration": tile.ordinal},
                )
            )
            task_order += 1
            output_key = tuple(tile.bound_map[name][0] for name, _ in operator.iteration_dims)
            reduce_predecessors = {load_id}
            partial_predecessor = partials.get(output_key)
            reduce_reads = [input_local]
            if partial_predecessor is not None:
                reduce_predecessors.add(partial_predecessor)
                reduce_reads.append(output_local_read)
            elements = math.prod(input_local.shape)
            reduce_duration, reduce_ii, reduce_unit = _reduce_timing(machine, elements)
            reduce_id = f"{tile.tile_id}.reduce"
            tasks.append(
                ExecutionTask(
                    task_id=reduce_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="reduce",
                    resource=reduce_unit,
                    reads=tuple(reduce_reads),
                    writes=(output_local_write,),
                    predecessors=tuple(sorted(reduce_predecessors)),
                    duration_cycles=reduce_duration,
                    initiation_interval_cycles=reduce_ii,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={
                        "elements": elements,
                        "reduction_start": tile.bound_map[reduction_name][0],
                        "reduction_stop": tile.bound_map[reduction_name][1],
                        "iteration": tile.ordinal,
                    },
                )
            )
            task_order += 1
            partials[output_key] = reduce_id
            input_elements += elements
            transfer_bytes += input_global.size_bytes
            if tile.bound_map[reduction_name][1] == reduction_extent:
                store_id = f"{tile.tile_id}.store"
                store_duration, store_ii, store_unit = _transfer_timing(machine, local, root, output_global.size_bytes)
                tasks.append(
                    ExecutionTask(
                        task_id=store_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="store",
                        resource=store_unit,
                        reads=(output_local_read,),
                        writes=(BufferRegion(**{**output_global.__dict__, "access": AccessType.WRITE}),),
                        predecessors=(reduce_id,),
                        duration_cycles=store_duration,
                        initiation_interval_cycles=store_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"final_reduction_tile": True, "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                producer_stores.setdefault(operator.outputs[0], []).append((output_global, store_id))
                transfer_bytes += output_global.size_bytes

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={"source": "reduce-lowering", "root_memory": root, "local_memory": local},
    )
    issues = execution.validate()
    if issues:
        raise ValueError("; ".join(issues))
    tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(execution.tasks),
            "input_elements": input_elements,
            "transfer_bytes": transfer_bytes,
        },
    )


def lower_reduce(model: ModelInstance, machine: MachineConfig, schedule: ScheduleSpec) -> LoweringResult:
    return lower_reduce_graph(model.graph, schedule, machine)
