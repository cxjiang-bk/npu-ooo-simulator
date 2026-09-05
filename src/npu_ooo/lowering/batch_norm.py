from __future__ import annotations

"""Analytical lowering for inference-only NCHW batch normalization."""

import math

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

from .elementwise import _elementwise_timing
from .matmul import LoweringResult, _local_memory, _region, _root_memory, _transfer_timing, _unit_for


def _broadcast_region(
    tensor,
    output_shape: tuple[int, ...],
    starts: tuple[int, ...],
    memory: str,
    access: AccessType,
    feature_index: int,
) -> BufferRegion:
    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) > len(output_shape):
        raise ValueError(f"batch norm operand '{tensor.name}' rank exceeds activation rank")
    if len(shape) == 1:
        if shape[0] != output_shape[feature_index]:
            raise ValueError(
                f"batch norm operand '{tensor.name}' shape {shape} cannot match feature extent"
            )
        return _region(
            tensor,
            memory,
            (0,),
            shape,
            access,
        )
    padded = (1,) * (len(output_shape) - len(shape)) + shape
    region_starts: list[int] = []
    region_shape: list[int] = []
    for axis, extent in enumerate(padded):
        output_extent = output_shape[axis]
        if extent not in {1, output_extent}:
            raise ValueError(
                f"batch norm operand '{tensor.name}' shape {shape} cannot broadcast to {output_shape}"
            )
        if axis < len(output_shape) - len(shape):
            continue
        if len(shape) == 1 and axis != feature_index:
            continue
        if extent == 1:
            region_starts.append(0)
            region_shape.append(1)
        else:
            region_starts.append(starts[axis])
            region_shape.append(output_extent if len(shape) == len(output_shape) else extent)
    return _region(tensor, memory, tuple(region_starts), tuple(region_shape), access)


def lower_batch_norm_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower inference BatchNorm to load -> vector batch_norm -> store."""

    issues = (*graph.validate(), *schedule.validate(graph), *machine.validate())
    if issues:
        raise ValueError("; ".join(issues))
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    local = _local_memory(machine, root)
    tasks: list[ExecutionTask] = []
    transfer_bytes = 0
    elements = 0

    from npu_ooo.ir.tile import enumerate_operator_tiles

    for operator_id in graph.topological_order():
        operator = next(item for item in graph.operators if item.op_id == operator_id)
        if operator.normalized_type != "batch_norm":
            raise NotImplementedError(
                f"batch norm lowering does not support '{operator.normalized_type}'"
            )
        if len(operator.inputs) != 5 or len(operator.outputs) != 1:
            raise ValueError(f"batch norm operator '{operator.op_id}' requires five inputs and one output")
        input_tensor = tensors[operator.inputs[0]]
        output_tensor = tensors[operator.outputs[0]]
        if tuple(input_tensor.shape) != tuple(output_tensor.shape) or len(input_tensor.shape) != 4:
            raise ValueError(f"batch norm operator '{operator.op_id}' requires matching rank-4 activation shapes")
        feature_index = int(operator.attributes.get("feature_index", 1))
        if feature_index < 0 or feature_index >= len(input_tensor.shape):
            raise ValueError(f"batch norm operator '{operator.op_id}' has invalid feature_index")
        channel_extent = input_tensor.shape[feature_index]
        for name in operator.inputs[1:]:
            if tuple(tensors[name].shape) != (channel_extent,):
                raise ValueError(f"batch norm statistics '{name}' must match feature extent {channel_extent}")
        output_shape = tuple(int(value) for value in output_tensor.shape)
        dimensions = tuple(name for name, _ in operator.iteration_dims)
        if dimensions != tuple(f"d{axis}" for axis in range(4)):
            raise ValueError(f"batch norm operator '{operator.op_id}' requires canonical rank-4 dimensions")
        for tile in enumerate_operator_tiles(operator, schedule.for_operator(operator_id)):
            starts = tuple(tile.bound_map[name][0] for name in dimensions)
            tile_shape = tuple(tile.bound_map[name][1] - tile.bound_map[name][0] for name in dimensions)
            input_regions = []
            load_ids = []
            for input_index, input_name in enumerate(operator.inputs):
                tensor = tensors[input_name]
                if input_index == 0:
                    region = _region(tensor, root, starts, tile_shape, AccessType.READ)
                    local_region = _region(tensor, local, starts, tile_shape, AccessType.WRITE)
                else:
                    region = _broadcast_region(tensor, output_shape, starts, root, AccessType.READ, feature_index)
                    local_region = _broadcast_region(tensor, output_shape, starts, local, AccessType.WRITE, feature_index)
                load_id = f"{tile.tile_id}.load_{input_index}"
                duration, ii, unit = _transfer_timing(machine, root, local, region.size_bytes)
                tasks.append(
                    ExecutionTask(
                        task_id=load_id,
                        tile_id=tile.tile_id,
                        operator_id=operator.op_id,
                        primitive="load",
                        resource=unit,
                        reads=(region,),
                        writes=(local_region,),
                        duration_cycles=duration,
                        initiation_interval_cycles=ii,
                        stage_id=tile.stage_id,
                        program_order=len(tasks),
                        attributes={"operand": input_index, "iteration": tile.ordinal},
                    )
                )
                input_regions.append(local_region)
                load_ids.append(load_id)
                transfer_bytes += region.size_bytes
            output_local = _region(output_tensor, local, starts, tile_shape, AccessType.WRITE)
            compute_duration, compute_ii, compute_unit = _elementwise_timing(machine, math.prod(tile_shape))
            compute_id = f"{tile.tile_id}.batch_norm"
            tasks.append(
                ExecutionTask(
                    task_id=compute_id,
                    tile_id=tile.tile_id,
                    operator_id=operator.op_id,
                    primitive="batch_norm",
                    resource=_unit_for(machine, "batch_norm").name,
                    reads=tuple(input_regions),
                    writes=(output_local,),
                    predecessors=tuple(load_ids),
                    duration_cycles=compute_duration,
                    initiation_interval_cycles=compute_ii,
                    stage_id=tile.stage_id,
                    program_order=len(tasks),
                    attributes={
                        "feature_index": feature_index,
                        "epsilon": operator.attributes.get("epsilon", 1e-5),
                        "inference": True,
                        "elements": math.prod(tile_shape),
                    },
                )
            )
            output_root = _region(output_tensor, root, starts, tile_shape, AccessType.WRITE)
            store_id = f"{tile.tile_id}.store"
            store_duration, store_ii, store_unit = _transfer_timing(machine, local, root, output_root.size_bytes)
            tasks.append(
                ExecutionTask(
                    task_id=store_id,
                    tile_id=tile.tile_id,
                    operator_id=operator.op_id,
                    primitive="store",
                    resource=store_unit,
                    reads=(BufferRegion(**{**output_local.__dict__, "access": AccessType.READ}),),
                    writes=(output_root,),
                    predecessors=(compute_id,),
                    duration_cycles=store_duration,
                    initiation_interval_cycles=store_ii,
                    stage_id=tile.stage_id,
                    program_order=len(tasks),
                )
            )
            elements += math.prod(tile_shape)
            transfer_bytes += output_root.size_bytes

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={"source": "batch-norm-lowering", "root_memory": root, "local_memory": local},
    )
    issues = execution.validate()
    if issues:
        raise ValueError("batch norm execution graph is invalid: " + "; ".join(issues))
    tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={"tile_count": len(tile_graph.tiles), "task_count": len(tasks), "elements": elements, "transfer_bytes": transfer_bytes},
    )


__all__ = ["lower_batch_norm_graph"]
