from __future__ import annotations

"""Analytical payload lowering for reshape and transpose operations."""

import math

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    ScheduleSpec,
    build_tile_graph,
)

from .matmul import LoweringResult, _region, _root_memory, _unit_for


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
        if operator.normalized_type not in {"reshape", "transpose"}:
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
        if not is_broadcast and input_elements != output_elements:
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
        if not is_broadcast and len(operator_tiles) != 1:
            raise ValueError(
                f"transform operator '{operator.op_id}' requires a full-tensor schedule"
            )
        primitive = (
            "copy"
            if operator.normalized_type == "reshape"
            else "transpose"
        )
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
            else:
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
                        "broadcast": is_broadcast,
                        "broadcast_dimensions": operator.attributes.get("broadcast_dimensions"),
                        "full_tensor_transform": not is_broadcast,
                        "transform_granularity": "output_tile" if is_broadcast else "full_tensor",
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
