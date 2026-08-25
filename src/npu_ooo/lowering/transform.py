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
    """Lower a full-tensor view/layout transform to one DMA payload."""

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
        input_elements = math.prod(input_tensor.shape)
        output_elements = math.prod(output_tensor.shape)
        if input_elements != output_elements:
            raise ValueError(
                f"transform operator '{operator.op_id}' changes element count "
                f"from {input_elements} to {output_elements}"
            )

        operator_tiles = tuple(
            tile for tile in tile_graph.tiles if tile.operator_id == operator_id
        )
        if len(operator_tiles) != 1:
            raise ValueError(
                f"transform operator '{operator.op_id}' requires a full-tensor schedule"
            )
        tile = operator_tiles[0]
        primitive = "copy" if operator.normalized_type == "reshape" else "transpose"
        unit = _unit_for(machine, primitive)
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
                    "full_tensor_transform": True,
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
            "granularity": "full_tensor",
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
