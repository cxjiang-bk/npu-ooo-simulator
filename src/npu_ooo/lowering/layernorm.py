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
    TileGraph,
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
)
from .norm import _elementwise_timing
from .reduce import _reduce_timing


def _virtual_region(
    tensor: str,
    memory: str,
    shape: tuple[int, ...],
    starts: tuple[int, ...],
    access: AccessType,
) -> BufferRegion:
    return BufferRegion(
        tensor=tensor,
        memory=memory,
        shape=shape,
        starts=starts,
        dtype="fp32",
        access=access,
        offset_bytes=0,
        size_bytes=math.prod(shape) * 4,
        layout="row_scalar",
    )


def lower_layernorm_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower LayerNorm with explicit mean and variance reduction barriers."""

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
        if operator.normalized_type != "layernorm":
            raise NotImplementedError(f"LayerNorm lowering does not support operator type '{operator.normalized_type}'")
        if len(operator.inputs) not in {1, 3} or len(operator.outputs) != 1:
            raise ValueError(
                f"LayerNorm operator '{operator.op_id}' requires x, optional weight/bias, and one output"
            )
        iteration = tuple(name for name, _ in operator.iteration_dims)
        reduction = tuple(name for name, _ in operator.reduction_dims)
        if not iteration or len(reduction) != 1:
            raise ValueError(
                f"LayerNorm operator '{operator.op_id}' requires iteration dimensions and one reduction dimension"
            )
        input_tensor = tensors[operator.inputs[0]]
        output_tensor = tensors[operator.outputs[0]]
        if tuple(input_tensor.shape) != tuple(output_tensor.shape):
            raise ValueError(f"LayerNorm operator '{operator.op_id}' input/output shapes must match")
        affine = len(operator.inputs) == 3
        weight_tensor = tensors[operator.inputs[1]] if affine else None
        bias_tensor = tensors[operator.inputs[2]] if affine else None
        reduction_extent = dict(operator.reduction_dims)[reduction[0]]
        if affine and (
            tuple(weight_tensor.shape) != (reduction_extent,)
            or tuple(bias_tensor.shape) != (reduction_extent,)
        ):
            raise ValueError(
                f"LayerNorm operator '{operator.op_id}' weight/bias must match reduction extent {reduction_extent}"
            )
        from npu_ooo.ir.tile import enumerate_operator_tiles

        tiles = enumerate_operator_tiles(operator, schedule.for_operator(operator_id))
        reduction_name = reduction[0]
        rows: dict[tuple[int, ...], list[TileInstance]] = {}
        for tile in tiles:
            rows.setdefault(
                tuple(tile.bound_map[name][0] for name in iteration),
                [],
            ).append(tile)

        for row_key, row_tiles in rows.items():
            row_tiles.sort(key=lambda tile: tile.bound_map[reduction_name][0])
            iteration_shape = tuple(row_tiles[0].extent(name) for name in iteration)
            row_extent = math.prod(iteration_shape)
            sum_region = _virtual_region(
                f"{operator_id}.sum", local, iteration_shape, row_key, AccessType.READ
            )
            mean_region = _virtual_region(
                f"{operator_id}.mean", local, iteration_shape, row_key, AccessType.READ
            )
            variance_region = _virtual_region(
                f"{operator_id}.variance", local, iteration_shape, row_key, AccessType.READ
            )
            sum_final: str | None = None
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
                load_preds = {
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
                        predecessors=tuple(sorted(load_preds)),
                        duration_cycles=load_duration,
                        initiation_interval_cycles=load_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"iteration": tile.ordinal},
                    )
                )
                task_order += 1
                sum_id = f"{tile.tile_id}.reduce_sum"
                sum_duration, sum_ii, sum_unit = _reduce_timing(machine, math.prod(shape))
                sum_preds = {load_id}
                sum_reads = [input_local]
                sum_access = AccessType.WRITE
                if sum_final is not None:
                    sum_preds.add(sum_final)
                    sum_reads.append(sum_region)
                    sum_access = AccessType.READ_WRITE
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
                        attributes={"reduction": "mean", "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                sum_final = sum_id
                input_elements += math.prod(shape)
                transfer_bytes += input_global.size_bytes

            row_label = "_".join(f"{value:04d}" for value in row_key)
            mean_id = f"{operator_id}.r{row_label}.mean"
            mean_duration, mean_ii, mean_unit = _elementwise_timing(machine, row_extent)
            tasks.append(
                ExecutionTask(
                    task_id=mean_id,
                    tile_id=row_tiles[0].tile_id,
                    operator_id=operator_id,
                    primitive="layernorm_mean",
                    resource=mean_unit,
                    reads=(sum_region,),
                    writes=(BufferRegion(**{**mean_region.__dict__, "access": AccessType.WRITE}),),
                    predecessors=(sum_final,),
                    duration_cycles=mean_duration,
                    initiation_interval_cycles=mean_ii,
                    stage_id=schedule.for_operator(operator_id).stage_id,
                    program_order=task_order,
                    attributes={"reduction_extent": dict(operator.reduction_dims)[reduction_name]},
                )
            )
            task_order += 1

            variance_final: str | None = None
            centered_by_tile: dict[str, BufferRegion] = {}
            input_local_by_tile: dict[str, BufferRegion] = {}
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
                centered = _virtual_region(
                    f"{operator_id}.centered", local, shape, starts, AccessType.WRITE
                )
                center_id = f"{tile.tile_id}.center"
                center_duration, center_ii, center_unit = _elementwise_timing(machine, math.prod(shape))
                tasks.append(
                    ExecutionTask(
                        task_id=center_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="center",
                        resource=center_unit,
                        reads=(input_local, mean_region),
                        writes=(centered,),
                        predecessors=(f"{tile.tile_id}.load", mean_id),
                        duration_cycles=center_duration,
                        initiation_interval_cycles=center_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"iteration": tile.ordinal},
                    )
                )
                task_order += 1
                variance_id = f"{tile.tile_id}.reduce_sum_square"
                variance_duration, variance_ii, variance_unit = _reduce_timing(machine, math.prod(shape))
                variance_preds = {center_id}
                variance_reads = [centered]
                variance_access = AccessType.WRITE
                if variance_final is not None:
                    variance_preds.add(variance_final)
                    variance_reads.append(variance_region)
                    variance_access = AccessType.READ_WRITE
                tasks.append(
                    ExecutionTask(
                        task_id=variance_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="reduce_sum_square",
                        resource=variance_unit,
                        reads=tuple(variance_reads),
                        writes=(BufferRegion(**{**variance_region.__dict__, "access": variance_access}),),
                        predecessors=tuple(sorted(variance_preds)),
                        duration_cycles=variance_duration,
                        initiation_interval_cycles=variance_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"reduction": "variance", "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                variance_final = variance_id
                centered_by_tile[tile.tile_id] = centered
                input_local_by_tile[tile.tile_id] = input_local

            assert variance_final is not None
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
                output_local = _region(output_tensor, local, starts, shape, AccessType.WRITE)
                normalize_id = f"{tile.tile_id}.layernorm"
                normalize_duration, normalize_ii, normalize_unit = _elementwise_timing(machine, math.prod(shape))
                center_id = f"{tile.tile_id}.center"
                normalize_reads = [
                    input_local_by_tile[tile.tile_id],
                    mean_region,
                    variance_region,
                ]
                normalize_predecessors = {center_id, mean_id, variance_final}
                if affine:
                    parameter_start = (bounds[reduction_name][0],)
                    parameter_shape = (bounds[reduction_name][1] - bounds[reduction_name][0],)
                    for parameter_name, parameter_tensor in (
                        ("weight", weight_tensor),
                        ("bias", bias_tensor),
                    ):
                        parameter_global = _region(
                            parameter_tensor,
                            root,
                            parameter_start,
                            parameter_shape,
                            AccessType.READ,
                        )
                        parameter_local = _region(
                            parameter_tensor,
                            local,
                            parameter_start,
                            parameter_shape,
                            AccessType.READ,
                        )
                        parameter_load_id = f"{tile.tile_id}.load_{parameter_name}"
                        parameter_duration, parameter_ii, parameter_unit = _transfer_timing(
                            machine,
                            root,
                            local,
                            parameter_global.size_bytes,
                        )
                        tasks.append(
                            ExecutionTask(
                                task_id=parameter_load_id,
                                tile_id=tile.tile_id,
                                operator_id=operator_id,
                                primitive="load",
                                resource=parameter_unit,
                                reads=(parameter_global,),
                                writes=(
                                    BufferRegion(
                                        **{**parameter_local.__dict__, "access": AccessType.WRITE}
                                    ),
                                ),
                                duration_cycles=parameter_duration,
                                initiation_interval_cycles=parameter_ii,
                                stage_id=tile.stage_id,
                                program_order=task_order,
                                attributes={
                                    "operand": parameter_name,
                                    "affine": True,
                                    "iteration": tile.ordinal,
                                },
                            )
                        )
                        task_order += 1
                        normalize_reads.append(parameter_local)
                        normalize_predecessors.add(parameter_load_id)
                        transfer_bytes += parameter_global.size_bytes
                tasks.append(
                    ExecutionTask(
                        task_id=normalize_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="layernorm",
                        resource=normalize_unit,
                        reads=tuple(normalize_reads),
                        writes=(output_local,),
                        predecessors=tuple(sorted(normalize_predecessors)),
                        duration_cycles=normalize_duration,
                        initiation_interval_cycles=normalize_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={
                            "epsilon": operator.attributes.get("epsilon", 1e-5),
                            "affine": affine,
                            "iteration": tile.ordinal,
                        },
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
        attributes={"source": "layernorm-lowering", "root_memory": root, "local_memory": local},
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
            "composite_stage_count": 6,
        },
    )


def lower_layernorm(
    model: ModelInstance,
    machine: MachineConfig,
    schedule: ScheduleSpec,
) -> LoweringResult:
    return lower_layernorm_graph(model.graph, schedule, machine)
