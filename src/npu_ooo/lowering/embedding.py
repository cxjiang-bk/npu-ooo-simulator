from __future__ import annotations

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
    enumerate_operator_tiles,
    tensor_layout,
)

from .matmul import LoweringResult, _root_memory


def _region(tensor, memory: str, starts: tuple[int, ...], shape: tuple[int, ...], access: AccessType) -> BufferRegion:
    offset, span = tensor_layout(tensor).interval(starts, shape)
    return BufferRegion(
        tensor=tensor.name,
        memory=memory,
        shape=shape,
        starts=starts,
        dtype=tensor.dtype,
        access=access,
        offset_bytes=offset,
        size_bytes=span,
        layout=tensor_layout(tensor).layout,
        strides_bytes=tensor_layout(tensor).strides_bytes,
    )


def _gather_unit(machine: MachineConfig) -> str:
    for unit in machine.execution_units:
        if "gather" in unit.supported_ops:
            return unit.name
    raise ValueError(
        f"machine '{machine.config_id}' has no execution unit with gather capability"
    )


def lower_embedding_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower the proven two-dimensional embedding-gather contract.

    Indices are runtime values, so the table operand keeps a conservative full
    read region.  Timing scales with the selected output bytes rather than the
    full table allocation; the distinction is retained in task attributes.
    """

    issues = (*graph.validate(), *schedule.validate(graph), *machine.validate())
    if issues:
        raise ValueError("; ".join(issues))
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    root_memory = machine.memory(root)
    unit = _gather_unit(machine)
    tasks: list[ExecutionTask] = []
    total_output_bytes = 0
    total_index_bytes = 0

    for operator_id in graph.topological_order():
        operator = next(item for item in graph.operators if item.op_id == operator_id)
        if operator.normalized_type != "embedding":
            raise NotImplementedError(
                f"embedding lowering does not support '{operator.normalized_type}'"
            )
        if len(operator.inputs) != 2 or len(operator.outputs) != 1:
            raise ValueError(
                f"embedding operator '{operator_id}' requires table, indices and one output"
            )
        table = tensors[operator.inputs[0]]
        indices = tensors[operator.inputs[1]]
        output = tensors[operator.outputs[0]]
        table_shape = tuple(int(value) for value in table.shape)
        index_shape = tuple(int(value) for value in indices.shape)
        output_shape = tuple(int(value) for value in output.shape)
        if len(table_shape) != 2 or output_shape != (*index_shape, table_shape[1]):
            raise ValueError(
                f"embedding operator '{operator_id}' does not satisfy indices_shape + embedding_dim"
            )
        table_region = _region(
            table,
            root,
            (0, 0),
            table_shape,
            AccessType.READ,
        )
        dimension_names = tuple(name for name, _ in operator.iteration_dims)
        for tile in enumerate_operator_tiles(operator, schedule.for_operator(operator_id)):
            output_starts = tuple(tile.bound_map[name][0] for name in dimension_names)
            output_tile_shape = tuple(tile.extent(name) for name in dimension_names)
            index_starts = output_starts[: len(index_shape)]
            index_tile_shape = output_tile_shape[: len(index_shape)]
            index_region = _region(
                indices,
                root,
                index_starts,
                index_tile_shape,
                AccessType.READ,
            )
            output_region = _region(
                output,
                root,
                output_starts,
                output_tile_shape,
                AccessType.WRITE,
            )
            selected_bytes = output_region.size_bytes
            transfer_bytes = index_region.size_bytes + selected_bytes
            duration = (
                root_memory.read_latency_cycles
                + root_memory.write_latency_cycles
                + math.ceil(
                    transfer_bytes
                    / min(
                        root_memory.read_bandwidth_bytes_per_cycle,
                        root_memory.write_bandwidth_bytes_per_cycle,
                    )
                )
            )
            tasks.append(
                ExecutionTask(
                    task_id=f"{tile.tile_id}.gather",
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="gather",
                    resource=unit,
                    reads=(table_region, index_region),
                    writes=(output_region,),
                    duration_cycles=max(1.0, float(duration)),
                    initiation_interval_cycles=machine.unit(unit).initiation_interval_cycles,
                    stage_id=tile.stage_id,
                    program_order=len(tasks),
                    attributes={
                        "semantic_family": "embedding",
                        "table_tensor": table.name,
                        "indices_tensor": indices.name,
                        "vocabulary_size": table_shape[0],
                        "embedding_dim": table_shape[1],
                        "table_region_policy": "conservative_full_table",
                        "selected_output_bytes": selected_bytes,
                        "index_bytes": index_region.size_bytes,
                    },
                )
            )
            total_output_bytes += selected_bytes
            total_index_bytes += index_region.size_bytes

    execution_graph = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={
            "source": "embedding-lowering",
            "root_memory": root,
            "table_region_policy": "conservative_full_table",
        },
    )
    execution_issues = execution_graph.validate()
    if execution_issues:
        raise ValueError("; ".join(execution_issues))
    tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution_graph,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(tasks),
            "embedding_output_bytes": total_output_bytes,
            "embedding_index_bytes": total_index_bytes,
        },
    )
