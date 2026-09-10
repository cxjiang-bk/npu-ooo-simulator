from __future__ import annotations

"""Analytical lowering for static NCHW/OIHW 2-D convolution tiles."""

from dataclasses import dataclass
import math

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BufferRegion,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    ScheduleSpec,
    dtype_bytes,
)

from .matmul import LoweringResult, _local_memory, _path, _root_memory, _transfer_timing, _unit_for


def _region(
    tensor,
    memory: str,
    starts: tuple[int, ...],
    shape: tuple[int, ...],
    access: AccessType,
) -> BufferRegion:
    strides: list[int] = []
    stride = 1
    for extent in reversed(tensor.shape):
        strides.append(stride)
        stride *= int(extent)
    strides.reverse()
    offset = sum(start * stride_value for start, stride_value in zip(starts, strides))
    return BufferRegion(
        tensor=tensor.name,
        memory=memory,
        shape=shape,
        starts=starts,
        dtype=tensor.dtype,
        access=access,
        offset_bytes=offset * dtype_bytes(tensor.dtype, default=2),
        size_bytes=math.prod(shape) * dtype_bytes(tensor.dtype, default=2),
        layout=tensor.layout,
    )


def _compute_timing(machine: MachineConfig, output_shape: tuple[int, ...], reduction: int) -> tuple[float, float, str]:
    unit = _unit_for(machine, "conv2d")
    macs = math.prod(output_shape) * reduction
    configured_rate = unit.attributes.get("macs_per_cycle")
    if isinstance(configured_rate, (int, float)) and configured_rate > 0:
        rate = float(configured_rate)
    else:
        rate = float(
            unit.attributes.get("rows", 16)
            * unit.attributes.get("cols", 16)
            * max(1, unit.attributes.get("k", 1))
            * max(1, unit.issue_width)
        )
    return float(unit.latency_cycles + math.ceil(macs / rate)), float(unit.initiation_interval_cycles), unit.name


def lower_conv2d_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower one or more static NCHW convolution operators to DMA/MXU tasks."""

    issues = (*graph.validate(), *schedule.validate(graph), *machine.validate())
    if issues:
        raise ValueError("; ".join(issues))
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    local = _local_memory(machine, root)
    tasks: list[ExecutionTask] = []
    statistics: dict[str, int | float] = {"conv2d_count": 0, "macs": 0, "tile_count": 0}

    from npu_ooo.ir.tile import enumerate_operator_tiles

    for operator_id in graph.topological_order():
        operator = next(item for item in graph.operators if item.op_id == operator_id)
        if operator.normalized_type != "conv2d":
            raise NotImplementedError(
                f"convolution lowering does not support operator type '{operator.normalized_type}'"
            )
        if len(operator.inputs) != 2 or len(operator.outputs) != 1:
            raise ValueError(f"convolution operator '{operator.op_id}' requires input, weight and output")
        input_tensor = tensors[operator.inputs[0]]
        weight_tensor = tensors[operator.inputs[1]]
        output_tensor = tensors[operator.outputs[0]]
        if tuple(map(int, input_tensor.shape)) != tuple(input_tensor.shape) or len(input_tensor.shape) != 4:
            raise ValueError(f"convolution operator '{operator.op_id}' requires static rank-4 input")
        if len(weight_tensor.shape) != 4 or len(output_tensor.shape) != 4:
            raise ValueError(f"convolution operator '{operator.op_id}' requires static rank-4 tensors")
        schedule_spec = schedule.for_operator(operator.op_id)
        tiles = enumerate_operator_tiles(operator, schedule_spec)
        kernel = tuple(int(value) for value in operator.attributes.get("kernel_shape", weight_tensor.shape[2:]))
        stride = tuple(int(value) for value in operator.attributes.get("window_strides", (1, 1)))
        padding = tuple(int(value) for value in operator.attributes.get("padding", (0, 0, 0, 0)))
        stats_macs = 0
        for tile in tiles:
            bounds = tile.bound_map
            n_start, n_stop = bounds["N"]
            o_start, o_stop = bounds["O"]
            oh_start, oh_stop = bounds["OH"]
            ow_start, ow_stop = bounds["OW"]
            k_start, k_stop = bounds["K"]
            output_shape = (n_stop - n_start, o_stop - o_start, oh_stop - oh_start, ow_stop - ow_start)
            input_h, input_w = int(input_tensor.shape[2]), int(input_tensor.shape[3])
            input_start_h = max(0, oh_start * stride[0] - padding[0])
            input_start_w = max(0, ow_start * stride[1] - padding[2])
            input_shape = (
                n_stop - n_start,
                int(input_tensor.shape[1]),
                max(1, min(input_h - input_start_h, (oh_stop - oh_start - 1) * stride[0] + kernel[0])),
                max(1, min(input_w - input_start_w, (ow_stop - ow_start - 1) * stride[1] + kernel[1])),
            )
            weight_shape = tuple(int(value) for value in weight_tensor.shape)
            output_root = _region(output_tensor, root, (n_start, o_start, oh_start, ow_start), output_shape, AccessType.WRITE)
            output_local = _region(output_tensor, local, (n_start, o_start, oh_start, ow_start), output_shape, AccessType.WRITE)
            input_root = _region(input_tensor, root, (n_start, 0, input_start_h, input_start_w), input_shape, AccessType.READ)
            input_local = _region(input_tensor, local, (n_start, 0, input_start_h, input_start_w), input_shape, AccessType.WRITE)
            weight_root = _region(weight_tensor, root, (0, 0, 0, 0), weight_shape, AccessType.READ)
            weight_local = _region(weight_tensor, local, (0, 0, 0, 0), weight_shape, AccessType.WRITE)
            tile_id = tile.tile_id
            input_duration, input_ii, input_unit = _transfer_timing(machine, root, local, input_root.size_bytes)
            weight_duration, weight_ii, weight_unit = _transfer_timing(machine, root, local, weight_root.size_bytes)
            input_load = ExecutionTask(
                task_id=f"{tile_id}.load_input",
                tile_id=tile_id,
                operator_id=operator.op_id,
                primitive="load",
                resource=input_unit,
                reads=(input_root,),
                writes=(input_local,),
                duration_cycles=input_duration,
                initiation_interval_cycles=input_ii,
                stage_id=tile.stage_id,
            )
            weight_load = ExecutionTask(
                task_id=f"{tile_id}.load_weight",
                tile_id=tile_id,
                operator_id=operator.op_id,
                primitive="load",
                resource=weight_unit,
                reads=(weight_root,),
                writes=(weight_local,),
                duration_cycles=weight_duration,
                initiation_interval_cycles=weight_ii,
                stage_id=tile.stage_id,
            )
            compute_duration, compute_ii, compute_unit = _compute_timing(
                machine, output_shape, k_stop - k_start
            )
            compute = ExecutionTask(
                task_id=f"{tile_id}.conv2d",
                tile_id=tile_id,
                operator_id=operator.op_id,
                primitive="conv2d",
                resource=compute_unit,
                reads=(input_local, weight_local),
                writes=(output_local,),
                predecessors=(input_load.task_id, weight_load.task_id),
                duration_cycles=compute_duration,
                initiation_interval_cycles=compute_ii,
                stage_id=tile.stage_id,
                attributes={
                    "kernel_shape": list(kernel),
                    "window_strides": list(stride),
                    "padding": list(padding),
                    "reduction_start": k_start,
                    "reduction_stop": k_stop,
                },
            )
            tasks.extend((input_load, weight_load, compute))
            if k_stop == int(dict(operator.reduction_dims)["K"]):
                store_duration, store_ii, store_unit = _transfer_timing(machine, local, root, output_root.size_bytes)
                tasks.append(
                    ExecutionTask(
                        task_id=f"{tile_id}.store",
                        tile_id=tile_id,
                        operator_id=operator.op_id,
                        primitive="store",
                        resource=store_unit,
                        reads=(output_local,),
                        writes=(output_root,),
                        predecessors=(compute.task_id,),
                        duration_cycles=store_duration,
                        initiation_interval_cycles=store_ii,
                        stage_id=tile.stage_id,
                    )
                )
            stats_macs += math.prod(output_shape) * (k_stop - k_start)
        statistics["conv2d_count"] = int(statistics["conv2d_count"]) + 1
        statistics["tile_count"] = int(statistics["tile_count"]) + len(tiles)
        statistics["macs"] = int(statistics["macs"]) + stats_macs

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={"source": "conv2d-lowering", "layout": "nchw_oihw_nchw"},
    )
    issues = execution.validate()
    if issues:
        raise ValueError("convolution execution graph is invalid: " + "; ".join(issues))
    from npu_ooo.ir import build_tile_graph

    tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={**statistics, "task_count": len(tasks)},
    )


__all__ = ["lower_conv2d_graph"]
