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

from .elementwise import _elementwise_timing
from .matmul import (
    LoweringResult,
    _local_memory,
    _region,
    _regions_overlap,
    _root_memory,
    _transfer_timing,
)
from .reduce import _reduce_timing


def _virtual_region(
    tensor: str,
    memory: str,
    shape: tuple[int, ...],
    starts: tuple[int, ...],
    access: AccessType,
    *,
    dtype: str = "fp16",
) -> BufferRegion:
    elements = math.prod(shape)
    dtype_size = 2 if dtype in {"fp16", "bf16"} else 4
    return BufferRegion(
        tensor=tensor,
        memory=memory,
        shape=shape,
        starts=starts,
        dtype=dtype,
        access=access,
        offset_bytes=math.prod(starts) * dtype_size if starts else 0,
        size_bytes=elements * dtype_size,
    )


def lower_softmax_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower row-wise softmax into explicit max/exp/sum/normalize primitives."""

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
        if operator.normalized_type != "softmax":
            raise NotImplementedError(f"softmax lowering does not support '{operator.normalized_type}'")
        if len(operator.inputs) != 1 or len(operator.outputs) != 1:
            raise ValueError(f"softmax operator '{operator.op_id}' requires one input and one output")
        iteration = tuple(name for name, _ in operator.iteration_dims)
        reduction = tuple(name for name, _ in operator.reduction_dims)
        if len(iteration) != 1 or len(reduction) != 1:
            raise ValueError(f"softmax operator '{operator.op_id}' requires one iteration and one reduction dimension")
        input_tensor = tensors[operator.inputs[0]]
        output_tensor = tensors[operator.outputs[0]]
        if tuple(input_tensor.shape) != tuple(output_tensor.shape):
            raise ValueError(f"softmax operator '{operator.op_id}' input/output shapes must match")
        from npu_ooo.ir.tile import enumerate_operator_tiles

        tiles = enumerate_operator_tiles(operator, schedule.for_operator(operator_id))
        reduction_name = reduction[0]
        rows: dict[tuple[int, ...], list[TileInstance]] = {}
        for tile in tiles:
            row_key = (tile.bound_map[iteration[0]][0],)
            rows.setdefault(row_key, []).append(tile)

        for row_key, row_tiles in rows.items():
            row_tiles.sort(key=lambda tile: tile.bound_map[reduction_name][0])
            max_final: str | None = None
            sum_final: str | None = None
            load_ids: dict[str, str] = {}
            exp_ids: dict[str, str] = {}
            max_region = _virtual_region(f"{operator_id}.max", local, (row_tiles[0].extent(iteration[0]),), row_key, AccessType.READ)
            sum_region = _virtual_region(f"{operator_id}.sum", local, (row_tiles[0].extent(iteration[0]),), row_key, AccessType.READ)

            # Stage 1: load each reduction tile and accumulate row-wise max.
            for tile in row_tiles:
                bounds = tile.bound_map
                input_starts = (bounds[iteration[0]][0], bounds[reduction_name][0])
                input_shape = (
                    bounds[iteration[0]][1] - bounds[iteration[0]][0],
                    bounds[reduction_name][1] - bounds[reduction_name][0],
                )
                input_global = _region(input_tensor, root, input_starts, input_shape, AccessType.READ)
                input_local = _region(input_tensor, local, input_starts, input_shape, AccessType.READ)
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
                input_elements += math.prod(input_shape)
                transfer_bytes += input_global.size_bytes
                max_access = AccessType.READ_WRITE if max_final is not None else AccessType.WRITE
                max_id = f"{tile.tile_id}.reduce_max"
                max_duration, max_ii, max_unit = _reduce_timing(machine, math.prod(input_shape))
                max_preds = {load_id}
                max_reads = [input_local]
                if max_final is not None:
                    max_preds.add(max_final)
                    max_reads.append(max_region)
                tasks.append(
                    ExecutionTask(
                        task_id=max_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="reduce_max",
                        resource=max_unit,
                        reads=tuple(max_reads),
                        writes=(BufferRegion(**{**max_region.__dict__, "access": max_access}),),
                        predecessors=tuple(sorted(max_preds)),
                        duration_cycles=max_duration,
                        initiation_interval_cycles=max_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"reduction": "max", "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                max_final = max_id

            assert max_final is not None

            # Stage 2: exponentiation, followed by sum reduction over exp tiles.
            for tile in row_tiles:
                bounds = tile.bound_map
                shape = (
                    bounds[iteration[0]][1] - bounds[iteration[0]][0],
                    bounds[reduction_name][1] - bounds[reduction_name][0],
                )
                starts = (bounds[iteration[0]][0], bounds[reduction_name][0])
                input_local = _region(input_tensor, local, starts, shape, AccessType.READ)
                exp_region = _virtual_region(f"{operator_id}.exp", local, shape, starts, AccessType.WRITE)
                exp_id = f"{tile.tile_id}.exp"
                exp_duration, exp_ii, exp_unit = _elementwise_timing(machine, math.prod(shape))
                tasks.append(
                    ExecutionTask(
                        task_id=exp_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="exp",
                        resource=exp_unit,
                        reads=(input_local, max_region),
                        writes=(exp_region,),
                        predecessors=(load_ids[tile.tile_id], max_final),
                        duration_cycles=exp_duration,
                        initiation_interval_cycles=exp_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"iteration": tile.ordinal},
                    )
                )
                task_order += 1
                exp_ids[tile.tile_id] = exp_id
                sum_access = AccessType.READ_WRITE if sum_final is not None else AccessType.WRITE
                sum_id = f"{tile.tile_id}.reduce_sum"
                sum_duration, sum_ii, sum_unit = _reduce_timing(machine, math.prod(shape))
                sum_preds = {exp_id}
                sum_reads = [_virtual_region(f"{operator_id}.exp", local, shape, starts, AccessType.READ)]
                if sum_final is not None:
                    sum_preds.add(sum_final)
                    sum_reads.append(sum_region)
                tasks.append(
                    ExecutionTask(
                        task_id=sum_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="reduce_sum",
                        resource=sum_unit,
                        reads=tuple(sum_reads),
                        writes=(BufferRegion(**{**sum_region.__dict__, "access": sum_access}),),
                        predecessors=tuple(sorted(sum_preds)),
                        duration_cycles=sum_duration,
                        initiation_interval_cycles=sum_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"reduction": "sum", "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                sum_final = sum_id

            assert sum_final is not None

            # Stage 3: normalize every tile and write the output.
            for tile in row_tiles:
                bounds = tile.bound_map
                shape = (
                    bounds[iteration[0]][1] - bounds[iteration[0]][0],
                    bounds[reduction_name][1] - bounds[reduction_name][0],
                )
                starts = (bounds[iteration[0]][0], bounds[reduction_name][0])
                exp_region = _virtual_region(f"{operator_id}.exp", local, shape, starts, AccessType.READ)
                output_local = _region(output_tensor, local, starts, shape, AccessType.WRITE)
                output_global = _region(output_tensor, root, starts, shape, AccessType.WRITE)
                normalize_id = f"{tile.tile_id}.normalize"
                normalize_duration, normalize_ii, normalize_unit = _elementwise_timing(machine, math.prod(shape))
                tasks.append(
                    ExecutionTask(
                        task_id=normalize_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="normalize",
                        resource=normalize_unit,
                        reads=(exp_region, sum_region),
                        writes=(output_local,),
                        predecessors=(exp_ids[tile.tile_id], sum_final),
                        duration_cycles=normalize_duration,
                        initiation_interval_cycles=normalize_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"iteration": tile.ordinal},
                    )
                )
                task_order += 1
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
        attributes={"source": "softmax-lowering", "root_memory": root, "local_memory": local},
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
            "composite_stage_count": 4,
        },
    )


def lower_softmax(model: ModelInstance, machine: MachineConfig, schedule: ScheduleSpec) -> LoweringResult:
    return lower_softmax_graph(model.graph, schedule, machine)
