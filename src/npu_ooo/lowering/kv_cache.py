from __future__ import annotations

"""Analytical lowering for the fixed-window KV-cache state update."""

import math
from typing import Mapping

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

from .matmul import LoweringResult, _local_memory, _root_memory, _transfer_timing, _unit_for


def _region(tensor, memory: str, access: AccessType) -> BufferRegion:
    element_bytes = dtype_bytes(tensor.dtype, default=2)
    return BufferRegion(
        tensor=tensor.name,
        memory=memory,
        shape=tuple(int(value) for value in tensor.shape),
        starts=(0,) * len(tensor.shape),
        dtype=tensor.dtype,
        access=access,
        size_bytes=math.prod(tensor.shape) * element_bytes,
        layout=tensor.layout,
    )


def lower_kv_cache_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower one stateful cache append into two loads, update, and store."""

    issues = (*graph.validate(), *schedule.validate(graph), *machine.validate())
    if issues:
        raise ValueError("; ".join(issues))
    if len(graph.operators) != 1:
        raise ValueError("KV-cache lowering expects one canonical operator")
    operator = graph.operators[0]
    if operator.normalized_type != "kv_cache_update":
        raise NotImplementedError(
            f"KV-cache lowering does not support '{operator.normalized_type}'"
        )
    if len(operator.inputs) != 2 or len(operator.outputs) != 1:
        raise ValueError("KV-cache update requires cache input, update input, and one output")
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    cache = tensors[operator.inputs[0]]
    update = tensors[operator.inputs[1]]
    output = tensors[operator.outputs[0]]
    root = _root_memory(machine)
    local = _local_memory(machine, root)
    cache_root = _region(cache, root, AccessType.READ)
    update_root = _region(update, root, AccessType.READ)
    cache_local = _region(cache, local, AccessType.READ)
    update_local = _region(update, local, AccessType.READ)
    output_local = _region(output, local, AccessType.WRITE)
    output_root = _region(output, root, AccessType.WRITE)
    cache_load_duration, cache_load_ii, cache_load_unit = _transfer_timing(
        machine, root, local, cache_root.size_bytes
    )
    update_load_duration, update_load_ii, update_load_unit = _transfer_timing(
        machine, root, local, update_root.size_bytes
    )
    update_unit = _unit_for(machine, "kv_cache_update")
    rate = max(1, int(update_unit.attributes.get("elements_per_cycle", 1)))
    update_duration = float(update_unit.latency_cycles + math.ceil(math.prod(output.shape) / rate))
    store_duration, store_ii, store_unit = _transfer_timing(
        machine, local, root, output_root.size_bytes
    )
    tile_id = f"{operator.op_id}.t0000"
    state_attributes = {
        "state_id": operator.attributes.get("state_id", cache.name),
        "state_buffer": operator.attributes.get("state_buffer", cache.name),
        "stateful": True,
        "state_update": operator.attributes.get("state_update", False),
        "dynamic_index": operator.attributes.get("dynamic_index"),
    }
    dynamic_index = operator.attributes.get("dynamic_index")
    dynamic_index_attributes = (
        dynamic_index.get("attributes", {})
        if isinstance(dynamic_index, Mapping)
        else {}
    )
    state_region = (
        {
            "tensor": operator.attributes.get("state_buffer", cache.name),
            "dynamic_index": dynamic_index,
            "window_shape": list(
                dynamic_index_attributes.get("update_shape", ())
                if isinstance(dynamic_index_attributes, Mapping)
                else ()
            ),
            "address_semantics": "dynamic_index_window",
        }
        if operator.attributes.get("state_update") and isinstance(dynamic_index, Mapping)
        else None
    )
    cache_load = ExecutionTask(
        task_id=f"{tile_id}.load_cache",
        tile_id=tile_id,
        operator_id=operator.op_id,
        primitive="load",
        resource=cache_load_unit,
        reads=(cache_root,),
        writes=(cache_local,),
        duration_cycles=cache_load_duration,
        initiation_interval_cycles=cache_load_ii,
        attributes={**state_attributes, "operand": "cache"},
    )
    update_load = ExecutionTask(
        task_id=f"{tile_id}.load_update",
        tile_id=tile_id,
        operator_id=operator.op_id,
        primitive="load",
        resource=update_load_unit,
        reads=(update_root,),
        writes=(update_local,),
        duration_cycles=update_load_duration,
        initiation_interval_cycles=update_load_ii,
        attributes={**state_attributes, "operand": "update"},
    )
    compute = ExecutionTask(
        task_id=f"{tile_id}.kv_cache_update",
        tile_id=tile_id,
        operator_id=operator.op_id,
        primitive="kv_cache_update",
        resource=update_unit.name,
        reads=(cache_local, update_local),
        writes=(output_local,),
        predecessors=(cache_load.task_id, update_load.task_id),
        duration_cycles=update_duration,
        initiation_interval_cycles=float(update_unit.initiation_interval_cycles),
        attributes={
            **state_attributes,
            "semantic_family": "kv_cache",
            "cache_axis": operator.attributes.get("cache_axis"),
            "state_transition": operator.attributes.get("state_transition"),
        },
    )
    store = ExecutionTask(
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
        attributes={**state_attributes},
    )
    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=(cache_load, update_load, compute, store),
        attributes={
            "source": "kv-cache-lowering",
            "stateful": True,
            "state_id": operator.attributes.get("state_id", cache.name),
            "persistent_buffers": [operator.attributes.get("state_buffer", cache.name)],
            "dynamic_index": dynamic_index,
            "state_region": state_region,
        },
    )
    issues = execution.validate()
    if issues:
        raise ValueError("KV-cache execution graph is invalid: " + "; ".join(issues))
    from npu_ooo.ir import build_tile_graph

    tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(execution.tasks),
            "stateful_operator_count": 1,
            "state_id": operator.attributes.get("state_id", cache.name),
        },
    )


__all__ = ["lower_kv_cache_graph"]
