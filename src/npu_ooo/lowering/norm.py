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
    build_tile_graph,
)
from npu_ooo.ir.model import ModelInstance

from .elementwise import _elementwise_timing
from .matmul import LoweringResult, _local_memory, _region, _regions_overlap, _root_memory, _transfer_timing
from .reduce import _reduce_timing


def _virtual_region(tensor: str, memory: str, shape: tuple[int, ...], starts: tuple[int, ...], access: AccessType) -> BufferRegion:
    elements = math.prod(shape)
    return BufferRegion(
        tensor=tensor,
        memory=memory,
        shape=shape,
        starts=starts,
        access=access,
        offset_bytes=0,
        size_bytes=elements * 2,
    )


def lower_rmsnorm_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower row-wise RMSNorm into explicit square/sum/normalize stages."""

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
        if operator.normalized_type != "rmsnorm":
            raise NotImplementedError(f"RMSNorm lowering does not support '{operator.normalized_type}'")
        if len(operator.inputs) != 1 or len(operator.outputs) != 1:
            raise ValueError(f"RMSNorm operator '{operator.op_id}' requires one input and one output")
        iteration = tuple(name for name, _ in operator.iteration_dims)
        reduction = tuple(name for name, _ in operator.reduction_dims)
        if not iteration or len(reduction) != 1:
            raise ValueError(
                f"RMSNorm operator '{operator.op_id}' requires iteration dimensions and one reduction dimension"
            )
        input_tensor = tensors[operator.inputs[0]]
        output_tensor = tensors[operator.outputs[0]]
        if tuple(input_tensor.shape) != tuple(output_tensor.shape):
            raise ValueError(f"RMSNorm operator '{operator.op_id}' input/output shapes must match")
        from npu_ooo.ir.tile import enumerate_operator_tiles

        tiles = enumerate_operator_tiles(operator, schedule.for_operator(operator_id))
        reduction_name = reduction[0]
        rows: dict[tuple[int, ...], list[Any]] = {}
        for tile in tiles:
            rows.setdefault(
                tuple(tile.bound_map[name][0] for name in iteration),
                [],
            ).append(tile)

        for row_key, row_tiles in rows.items():
            row_tiles.sort(key=lambda tile: tile.bound_map[reduction_name][0])
            sum_final: str | None = None
            iteration_shape = tuple(row_tiles[0].extent(name) for name in iteration)
            sum_region = _virtual_region(
                f"{operator_id}.sumsq",
                local,
                iteration_shape,
                row_key,
                AccessType.READ,
            )
            square_ids: dict[str, str] = {}
            load_ids: dict[str, str] = {}
            for tile in row_tiles:
                bounds = tile.bound_map
                starts = (
                    *(bounds[name][0] for name in iteration),
                    bounds[reduction_name][0],
                )
                shape = (
                    *(bounds[name][1] - bounds[name][0] for name in iteration),
                    bounds[reduction_name][1] - bounds[reduction_name][0],
                )
                input_global = _region(input_tensor, root, starts, shape, AccessType.READ)
                input_local = _region(input_tensor, local, starts, shape, AccessType.READ)
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
                load_ids[tile.tile_id] = load_id
                square_region = _virtual_region(f"{operator_id}.square", local, shape, starts, AccessType.WRITE)
                square_id = f"{tile.tile_id}.square"
                square_duration, square_ii, square_unit = _elementwise_timing(machine, math.prod(shape))
                tasks.append(
                    ExecutionTask(
                        task_id=square_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="square",
                        resource=square_unit,
                        reads=(input_local,),
                        writes=(square_region,),
                        predecessors=(load_id,),
                        duration_cycles=square_duration,
                        initiation_interval_cycles=square_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"iteration": tile.ordinal},
                    )
                )
                task_order += 1
                square_ids[tile.tile_id] = square_id
                sum_access = AccessType.READ_WRITE if sum_final is not None else AccessType.WRITE
                sum_id = f"{tile.tile_id}.reduce_sum_square"
                sum_duration, sum_ii, sum_unit = _reduce_timing(machine, math.prod(shape))
                sum_preds = {square_id}
                sum_reads = [_virtual_region(f"{operator_id}.square", local, shape, starts, AccessType.READ)]
                if sum_final is not None:
                    sum_preds.add(sum_final)
                    sum_reads.append(sum_region)
                tasks.append(
                    ExecutionTask(
                        task_id=sum_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="reduce_sum_square",
                        resource=sum_unit,
                        reads=tuple(sum_reads),
                        writes=(BufferRegion(**{**sum_region.__dict__, "access": sum_access}),),
                        predecessors=tuple(sorted(sum_preds)),
                        duration_cycles=sum_duration,
                        initiation_interval_cycles=sum_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"reduction": "sum_square", "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                sum_final = sum_id
                input_elements += math.prod(shape)
                transfer_bytes += input_global.size_bytes
            assert sum_final is not None

            for tile in row_tiles:
                bounds = tile.bound_map
                starts = (
                    *(bounds[name][0] for name in iteration),
                    bounds[reduction_name][0],
                )
                shape = (
                    *(bounds[name][1] - bounds[name][0] for name in iteration),
                    bounds[reduction_name][1] - bounds[reduction_name][0],
                )
                input_local = _region(input_tensor, local, starts, shape, AccessType.READ)
                output_local = _region(output_tensor, local, starts, shape, AccessType.WRITE)
                normalize_id = f"{tile.tile_id}.rmsnorm"
                normalize_duration, normalize_ii, normalize_unit = _elementwise_timing(machine, math.prod(shape))
                tasks.append(
                    ExecutionTask(
                        task_id=normalize_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="rmsnorm",
                        resource=normalize_unit,
                        reads=(input_local, sum_region),
                        writes=(output_local,),
                        predecessors=(load_ids[tile.tile_id], sum_final),
                        duration_cycles=normalize_duration,
                        initiation_interval_cycles=normalize_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"iteration": tile.ordinal},
                    )
                )
                task_order += 1
                output_global = _region(output_tensor, root, starts, shape, AccessType.WRITE)
                store_id = f"{tile.tile_id}.store"
                store_duration, store_ii, store_unit = _transfer_timing(machine, local, root, output_global.size_bytes)
                tasks.append(
                    ExecutionTask(
                        task_id=store_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="store",
                        resource=store_unit,
                        reads=(BufferRegion(**{**output_local.__dict__, "access": AccessType.READ}),),
                        writes=(output_global,),
                        predecessors=(normalize_id,),
                        duration_cycles=store_duration,
                        initiation_interval_cycles=store_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"iteration": tile.ordinal},
                    )
                )
                task_order += 1
                producer_stores.setdefault(output_tensor.name, []).append((output_global, store_id))
                transfer_bytes += output_global.size_bytes

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={"source": "rmsnorm-lowering", "root_memory": root, "local_memory": local},
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
            "composite_stage_count": 3,
        },
    )


def lower_rmsnorm(model: ModelInstance, machine: MachineConfig, schedule: ScheduleSpec) -> LoweringResult:
    return lower_rmsnorm_graph(model.graph, schedule, machine)
