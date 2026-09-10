from __future__ import annotations

"""Analytical payload lowering for reshape and transpose operations."""

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
    tensor_layout,
)

from .matmul import LoweringResult, _region, _root_memory, _unit_for


def _transpose_geometry(
    tile: Any,
    dimensions: tuple[str, ...],
    permutation: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map an output tile to its source coordinates.

    StableHLO uses ``output[d] = input[permutation[d]]``.  The returned source
    starts and extents therefore index the source tensor in source-axis order.
    """

    output_starts = tuple(tile.bound_map[name][0] for name in dimensions)
    output_shape = tuple(
        tile.bound_map[name][1] - tile.bound_map[name][0] for name in dimensions
    )
    source_starts = [0] * len(permutation)
    source_shape = [0] * len(permutation)
    for output_axis, source_axis in enumerate(permutation):
        source_starts[source_axis] = output_starts[output_axis]
        source_shape[source_axis] = output_shape[output_axis]
    return tuple(source_starts), tuple(source_shape)


def _slice_region(
    tensor: Any,
    memory: str,
    starts: tuple[int, ...],
    shape: tuple[int, ...],
    slice_strides: tuple[int, ...],
    access: AccessType,
) -> BufferRegion:
    """Build a source region for a possibly non-unit-stride slice."""

    layout = tensor_layout(tensor)
    if len(slice_strides) != len(starts) or any(value <= 0 for value in slice_strides):
        raise ValueError(f"tensor '{tensor.name}' slice strides must be positive and rank-matched")
    source_strides = layout.strides_bytes
    effective_strides = (
        tuple(stride * step for stride, step in zip(source_strides, slice_strides))
        if source_strides is not None
        else None
    )
    interval = layout.interval(starts, shape, strides_bytes=effective_strides)
    if interval is None:
        offset_bytes = 0
        size_bytes = layout.allocation_size_bytes
    else:
        offset_bytes, size_bytes = interval
    return BufferRegion(
        tensor=tensor.name,
        memory=memory,
        shape=shape,
        starts=starts,
        dtype=tensor.dtype,
        access=access,
        offset_bytes=offset_bytes,
        size_bytes=size_bytes,
        layout=tensor.layout,
        strides_bytes=effective_strides,
    )


def lower_transform_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower views/layout transforms and output-tiled static broadcasts."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))

    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    memory = machine.memory(root)
    tile_graph = build_tile_graph(graph, schedule)
    tasks: list[ExecutionTask] = []
    bytes_moved = 0

    for operator_id in graph.topological_order():
        operator = next(item for item in graph.operators if item.op_id == operator_id)
        if operator.normalized_type not in {"reshape", "transpose", "slice"}:
            raise NotImplementedError(
                f"transform lowering does not support '{operator.normalized_type}'"
            )
        if len(operator.inputs) != 1 or len(operator.outputs) != 1:
            raise ValueError(
                f"transform operator '{operator.op_id}' requires one input and one output"
            )
        input_tensor = tensors[operator.inputs[0]]
        output_tensor = tensors[operator.outputs[0]]
        is_broadcast = bool(
            operator.attributes.get("broadcast")
            or operator.attributes.get("stablehlo_op") == "stablehlo.broadcast_in_dim"
        )
        input_elements = math.prod(input_tensor.shape)
        output_elements = math.prod(output_tensor.shape)
        is_slice = operator.normalized_type == "slice"
        dynamic_index = operator.attributes.get("dynamic_index")
        is_dynamic_slice = is_slice and isinstance(dynamic_index, dict)
        permutation = operator.attributes.get("transpose_dims")
        if operator.normalized_type == "transpose":
            if not isinstance(permutation, (tuple, list)) or len(permutation) != len(input_tensor.shape):
                raise ValueError(
                    f"transpose operator '{operator.op_id}' requires a complete permutation"
                )
            permutation = tuple(int(value) for value in permutation)
            if sorted(permutation) != list(range(len(input_tensor.shape))):
                raise ValueError(
                    f"transpose operator '{operator.op_id}' has invalid permutation {permutation}"
                )
            expected_output = tuple(input_tensor.shape[index] for index in permutation)
            if tuple(output_tensor.shape) != expected_output:
                raise ValueError(
                    f"transpose operator '{operator.op_id}' output shape {output_tensor.shape} "
                    f"does not match permutation {permutation} of input shape {input_tensor.shape}"
                )
        if not is_broadcast and not is_slice and input_elements != output_elements:
            raise ValueError(
                f"transform operator '{operator.op_id}' changes element count "
                f"from {input_elements} to {output_elements}"
            )
        if is_broadcast:
            dimensions = operator.attributes.get("broadcast_dimensions")
            if not isinstance(dimensions, (tuple, list)):
                raise ValueError(
                    f"broadcast operator '{operator.op_id}' is missing broadcast_dimensions"
                )
            if len(dimensions) != len(input_tensor.shape) or len(set(dimensions)) != len(dimensions):
                raise ValueError(
                    f"broadcast operator '{operator.op_id}' has invalid broadcast_dimensions "
                    f"{dimensions}"
                )
            for source_axis, result_axis in enumerate(dimensions):
                if not isinstance(result_axis, int) or result_axis < 0 or result_axis >= len(output_tensor.shape):
                    raise ValueError(
                        f"broadcast operator '{operator.op_id}' has an out-of-range result dimension"
                    )
        input_layout = tensor_layout(input_tensor)
        output_layout = tensor_layout(output_tensor)
        reshape_materialization = "none"
        if operator.normalized_type == "reshape":
            if input_layout.concrete and output_layout.concrete and input_layout.contiguous and output_layout.contiguous:
                reshape_materialization = "contiguous_view_compatible"
            else:
                reshape_materialization = "strided_materialize_copy"
                source_extent = input_tensor.shape[source_axis]
                result_extent = output_tensor.shape[result_axis]
                if source_extent != 1 and source_extent != result_extent:
                    raise ValueError(
                        f"broadcast operator '{operator.op_id}' cannot map input dimension "
                        f"{source_axis}={source_extent} to output dimension {result_axis}={result_extent}"
                    )

        operator_tiles = tuple(
            tile for tile in tile_graph.tiles if tile.operator_id == operator_id
        )
        if (
            not is_broadcast
            and not is_dynamic_slice
            and operator.normalized_type not in {"transpose"}
            and len(operator_tiles) != 1
        ):
            raise ValueError(
                f"transform operator '{operator.op_id}' requires a full-tensor schedule"
            )
        primitive = "copy" if operator.normalized_type in {"reshape", "slice"} else "transpose"
        unit = _unit_for(machine, primitive)
        dimensions = tuple(name for name, _ in operator.iteration_dims)
        for tile in operator_tiles:
            if is_broadcast:
                broadcast_dimensions = operator.attributes.get("broadcast_dimensions")
                if not isinstance(broadcast_dimensions, (tuple, list)):
                    raise ValueError(
                        f"broadcast operator '{operator.op_id}' is missing broadcast_dimensions"
                    )
                source_starts: list[int] = []
                source_shape: list[int] = []
                for source_axis, output_axis in enumerate(broadcast_dimensions):
                    output_start, output_stop = tile.bound_map[dimensions[output_axis]]
                    source_extent = int(input_tensor.shape[source_axis])
                    output_extent = output_stop - output_start
                    if source_extent == 1:
                        source_starts.append(0)
                        source_shape.append(1)
                    elif source_extent == output_extent or output_stop <= source_extent:
                        source_starts.append(output_start)
                        source_shape.append(output_extent)
                    else:
                        raise ValueError(
                            f"broadcast operator '{operator.op_id}' tile exceeds source extent"
                        )
                output_starts = tuple(tile.bound_map[name][0] for name in dimensions)
                output_shape = tuple(tile.bound_map[name][1] - tile.bound_map[name][0] for name in dimensions)
                input_region = _region(
                    input_tensor, root, tuple(source_starts), tuple(source_shape), AccessType.READ
                )
                output_region = _region(
                    output_tensor, root, output_starts, output_shape, AccessType.WRITE
                )
            elif is_dynamic_slice:
                output_starts = tuple(tile.bound_map[name][0] for name in dimensions)
                output_shape = tuple(
                    tile.bound_map[name][1] - tile.bound_map[name][0] for name in dimensions
                )
                input_region = _region(
                    input_tensor,
                    root,
                    (0,) * len(input_tensor.shape),
                    tuple(int(value) for value in operator.attributes.get("slice_sizes", input_tensor.shape)),
                    AccessType.READ,
                )
                output_region = _region(
                    output_tensor,
                    root,
                    output_starts,
                    output_shape,
                    AccessType.WRITE,
                )
            elif is_slice:
                starts = operator.attributes.get("slice_starts")
                limits = operator.attributes.get("slice_limits")
                strides = operator.attributes.get("slice_strides")
                if not (
                    isinstance(starts, (tuple, list))
                    and isinstance(limits, (tuple, list))
                    and isinstance(strides, (tuple, list))
                    and len(starts) == len(input_tensor.shape)
                    and len(limits) == len(input_tensor.shape)
                    and len(strides) == len(input_tensor.shape)
                ):
                    raise ValueError(
                        f"slice operator '{operator.op_id}' is missing complete static bounds"
                    )
                source_shape = tuple(
                    (int(limit) - int(start) + int(stride) - 1) // int(stride)
                    for start, limit, stride in zip(starts, limits, strides)
                )
                input_region = _slice_region(
                    input_tensor,
                    root,
                    tuple(int(value) for value in starts),
                    source_shape,
                    tuple(int(value) for value in strides),
                    AccessType.READ,
                )
                output_region = _region(
                    output_tensor,
                    root,
                    (0,) * len(output_tensor.shape),
                    tuple(int(value) for value in output_tensor.shape),
                    AccessType.WRITE,
                )
            elif operator.normalized_type == "transpose":
                source_starts, source_shape = _transpose_geometry(
                    tile, dimensions, tuple(permutation)
                )
                input_region = _region(
                    input_tensor,
                    root,
                    source_starts,
                    source_shape,
                    AccessType.READ,
                )
                output_starts = tuple(tile.bound_map[name][0] for name in dimensions)
                output_shape = tuple(
                    tile.bound_map[name][1] - tile.bound_map[name][0] for name in dimensions
                )
                output_region = _region(
                    output_tensor,
                    root,
                    output_starts,
                    output_shape,
                    AccessType.WRITE,
                )
            else:
                # A reshape is a view only when both tensors have a concrete
                # contiguous layout.  Non-contiguous layouts are materialized
                # through the same copy primitive with conservative regions.
                input_region = _region(
                    input_tensor,
                    root,
                    (0,) * len(input_tensor.shape),
                    tuple(input_tensor.shape),
                    AccessType.READ,
                )
                output_region = _region(
                    output_tensor,
                    root,
                    (0,) * len(output_tensor.shape),
                    tuple(output_tensor.shape),
                    AccessType.WRITE,
                )
            transfer_cycles = math.ceil(
                input_region.size_bytes / memory.read_bandwidth_bytes_per_cycle
            ) + math.ceil(
                output_region.size_bytes / memory.write_bandwidth_bytes_per_cycle
            )
            duration = (
                unit.latency_cycles
                + memory.read_latency_cycles
                + memory.write_latency_cycles
                + transfer_cycles
            )
            tasks.append(
                ExecutionTask(
                    task_id=f"{tile.tile_id}.{primitive}",
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive=primitive,
                    resource=unit.name,
                    reads=(input_region,),
                    writes=(output_region,),
                    duration_cycles=float(duration),
                    initiation_interval_cycles=float(unit.initiation_interval_cycles),
                    stage_id=tile.stage_id,
                    program_order=len(tasks),
                    attributes={
                        "semantic_family": operator.normalized_type,
                        "frontend_target": operator.attributes.get("frontend_target", ""),
                        "transpose_dims": operator.attributes.get("transpose_dims"),
                        "input_strides_bytes": list(input_region.strides_bytes)
                        if input_region.strides_bytes is not None
                        else None,
                        "output_strides_bytes": list(output_region.strides_bytes)
                        if output_region.strides_bytes is not None
                        else None,
                        "stride_aware": bool(
                            input_region.strides_bytes is not None
                            or output_region.strides_bytes is not None
                        ),
                        "reshape_materialization": reshape_materialization,
                        "broadcast": is_broadcast,
                        "broadcast_dimensions": operator.attributes.get("broadcast_dimensions"),
                        "full_tensor_transform": not is_broadcast,
                        "transform_granularity": "output_tile" if (is_broadcast or is_dynamic_slice) else "full_tensor",
                        "dynamic_index": operator.attributes.get("dynamic_index"),
                        "dynamic_address": is_dynamic_slice,
                    },
                )
            )
            bytes_moved += input_region.size_bytes + output_region.size_bytes

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={
            "source": "transform-lowering",
            "root_memory": root,
            "granularity": "output_tile" if any(
                bool(operator.attributes.get("broadcast"))
                or isinstance(operator.attributes.get("dynamic_index"), dict)
                for operator in graph.operators
            ) else "full_tensor",
        },
    )
    issues = execution.validate()
    if issues:
        raise ValueError("; ".join(issues))
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(execution.tasks),
            "transfer_bytes": bytes_moved,
        },
    )


__all__ = ["lower_transform_graph"]
