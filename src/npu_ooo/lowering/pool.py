from __future__ import annotations

"""Analytical lowering for static NCHW max/sum pooling windows."""

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
from .matmul import (
    LoweringResult,
    _local_memory,
    _region,
    _root_memory,
    _transfer_timing,
    _unit_for,
)


def lower_pool_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower NCHW reduce_window pooling into load -> vector pool -> store."""

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
        if operator.normalized_type != "pool":
            raise NotImplementedError(
                f"pool lowering does not support '{operator.normalized_type}'"
            )
        if len(operator.inputs) != 1 or len(operator.outputs) != 1:
            raise ValueError(
                f"pool operator '{operator.op_id}' requires one input and one output"
            )
        input_tensor = tensors[operator.inputs[0]]
        output_tensor = tensors[operator.outputs[0]]
        if len(input_tensor.shape) != 4 or len(output_tensor.shape) != 4:
            raise ValueError(
                f"pool operator '{operator.op_id}' requires rank-4 NCHW tensors"
            )

        window = tuple(
            int(value)
            for value in operator.attributes.get("window_dimensions", (1, 1, 1, 1))
        )
        stride = tuple(
            int(value)
            for value in operator.attributes.get("window_strides", (1, 1, 1, 1))
        )
        padding = tuple(
            int(value) for value in operator.attributes.get("padding", (0,) * 8)
        )
        if (
            len(window) != 4
            or len(stride) != 4
            or len(padding) != 8
            or window[:2] != (1, 1)
            or stride[:2] != (1, 1)
        ):
            raise ValueError(
                f"pool operator '{operator.op_id}' requires N/C-preserving "
                "rank-4 window metadata"
            )
        kernel_h, kernel_w = window[2:]
        stride_h, stride_w = stride[2:]
        pad_top, pad_left = padding[4], padding[6]
        output_dims = tuple(name for name, _ in operator.iteration_dims)
        if output_dims != tuple(f"d{axis}" for axis in range(4)):
            raise ValueError(
                f"pool operator '{operator.op_id}' requires canonical rank-4 dimensions"
            )

        for tile in enumerate_operator_tiles(operator, schedule.for_operator(operator_id)):
            bounds = tile.bound_map
            n_start, n_stop = bounds["d0"]
            c_start, c_stop = bounds["d1"]
            oh_start, oh_stop = bounds["d2"]
            ow_start, ow_stop = bounds["d3"]
            input_start_h = max(0, oh_start * stride_h - pad_top)
            input_start_w = max(0, ow_start * stride_w - pad_left)
            input_h_extent = min(
                int(input_tensor.shape[2]) - input_start_h,
                (oh_stop - oh_start - 1) * stride_h + kernel_h,
            )
            input_w_extent = min(
                int(input_tensor.shape[3]) - input_start_w,
                (ow_stop - ow_start - 1) * stride_w + kernel_w,
            )
            input_shape = (
                n_stop - n_start,
                c_stop - c_start,
                max(1, input_h_extent),
                max(1, input_w_extent),
            )
            output_shape = (
                n_stop - n_start,
                c_stop - c_start,
                oh_stop - oh_start,
                ow_stop - ow_start,
            )
            output_starts = (n_start, c_start, oh_start, ow_start)
            input_starts = (n_start, c_start, input_start_h, input_start_w)
            input_root = _region(
                input_tensor, root, input_starts, input_shape, AccessType.READ
            )
            input_local = _region(
                input_tensor, local, input_starts, input_shape, AccessType.WRITE
            )
            output_root = _region(
                output_tensor, root, output_starts, output_shape, AccessType.WRITE
            )
            output_local = _region(
                output_tensor, local, output_starts, output_shape, AccessType.WRITE
            )

            load_id = f"{tile.tile_id}.load"
            load_duration, load_ii, load_unit = _transfer_timing(
                machine, root, local, input_root.size_bytes
            )
            tasks.append(
                ExecutionTask(
                    task_id=load_id,
                    tile_id=tile.tile_id,
                    operator_id=operator.op_id,
                    primitive="load",
                    resource=load_unit,
                    reads=(input_root,),
                    writes=(input_local,),
                    duration_cycles=load_duration,
                    initiation_interval_cycles=load_ii,
                    stage_id=tile.stage_id,
                    program_order=len(tasks),
                )
            )

            compute_id = f"{tile.tile_id}.pool"
            compute_duration, compute_ii, _ = _elementwise_timing(
                machine, math.prod(output_shape) * kernel_h * kernel_w
            )
            tasks.append(
                ExecutionTask(
                    task_id=compute_id,
                    tile_id=tile.tile_id,
                    operator_id=operator.op_id,
                    primitive="pool",
                    resource=_unit_for(machine, "pool").name,
                    reads=(input_local,),
                    writes=(output_local,),
                    predecessors=(load_id,),
                    duration_cycles=compute_duration,
                    initiation_interval_cycles=compute_ii,
                    stage_id=tile.stage_id,
                    program_order=len(tasks),
                    attributes={
                        "window_dimensions": list(window),
                        "window_strides": list(stride),
                        "padding": list(padding),
                        "reducer": operator.attributes.get("pool_reducer", "add"),
                        "elements": math.prod(output_shape),
                    },
                )
            )

            store_id = f"{tile.tile_id}.store"
            store_duration, store_ii, store_unit = _transfer_timing(
                machine, local, root, output_root.size_bytes
            )
            tasks.append(
                ExecutionTask(
                    task_id=store_id,
                    tile_id=tile.tile_id,
                    operator_id=operator.op_id,
                    primitive="store",
                    resource=store_unit,
                    reads=(
                        BufferRegion(
                            **{
                                **output_local.__dict__,
                                "access": AccessType.READ,
                            }
                        ),
                    ),
                    writes=(output_root,),
                    predecessors=(compute_id,),
                    duration_cycles=store_duration,
                    initiation_interval_cycles=store_ii,
                    stage_id=tile.stage_id,
                    program_order=len(tasks),
                )
            )
            transfer_bytes += input_root.size_bytes + output_root.size_bytes
            elements += math.prod(output_shape)

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={
            "source": "pool-lowering",
            "root_memory": root,
            "local_memory": local,
        },
    )
    issues = execution.validate()
    if issues:
        raise ValueError("pool execution graph is invalid: " + "; ".join(issues))
    tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(tasks),
            "elements": elements,
            "transfer_bytes": transfer_bytes,
        },
    )


__all__ = ["lower_pool_graph"]
