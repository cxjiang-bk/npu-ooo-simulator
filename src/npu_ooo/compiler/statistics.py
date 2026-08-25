from __future__ import annotations

"""Inspectable statistics for one compiled graph/TISA/backend artifact."""

import math
from typing import Any

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import BackendArtifact, OperatorGraph, TISAProgram, TileGraph


def build_compile_statistics(
    graph: OperatorGraph,
    tile_graph: TileGraph,
    program: TISAProgram,
    backend_artifact: BackendArtifact,
    machine: MachineConfig,
) -> dict[str, Any]:
    """Summarize tiling, dependency, compute and root-memory behavior."""

    root_memories = {
        memory.name for memory in machine.memory_levels if memory.parent is None
    }
    tiles_by_operator: dict[str, list[Any]] = {}
    for tile in tile_graph.tiles:
        tiles_by_operator.setdefault(tile.operator_id, []).append(tile)
    instructions_by_operator: dict[str, list[Any]] = {}
    for instruction in program.instructions:
        instructions_by_operator.setdefault(instruction.operator_id, []).append(instruction)
    tasks_by_operator: dict[str, list[Any]] = {}
    for task in backend_artifact.execution_graph.tasks:
        tasks_by_operator.setdefault(task.operator_id, []).append(task)

    tile_incoming = {operator.op_id: 0 for operator in graph.operators}
    tile_outgoing = {operator.op_id: 0 for operator in graph.operators}
    tile_owner = {tile.tile_id: tile.operator_id for tile in tile_graph.tiles}
    for dependency in tile_graph.dependencies:
        tile_outgoing[tile_owner[dependency.producer]] += 1
        tile_incoming[tile_owner[dependency.consumer]] += 1

    operator_statistics: list[dict[str, Any]] = []
    total_macs = 0
    total_root_read_bytes = 0
    total_root_write_bytes = 0
    for operator in graph.operators:
        operator_tiles = tiles_by_operator.get(operator.op_id, ())
        operator_tasks = tasks_by_operator.get(operator.op_id, ())
        macs = 0
        if operator.normalized_type in {"matmul", "batched_matmul", "gemv"}:
            macs = sum(
                math.prod(tile.extent(name) for name, _extent in operator.iteration_dims)
                * math.prod(tile.extent(name) for name, _extent in operator.reduction_dims)
                for tile in operator_tiles
            )
        root_read_bytes = sum(
            region.size_bytes
            for task in operator_tasks
            for region in task.reads
            if region.memory in root_memories
        )
        root_write_bytes = sum(
            region.size_bytes
            for task in operator_tasks
            for region in task.writes
            if region.memory in root_memories
        )
        total_macs += macs
        total_root_read_bytes += root_read_bytes
        total_root_write_bytes += root_write_bytes
        operator_statistics.append(
            {
                "operator_id": operator.op_id,
                "op_type": operator.normalized_type,
                "tile_count": len(operator_tiles),
                "tisa_instruction_count": len(
                    instructions_by_operator.get(operator.op_id, ())
                ),
                "payload_task_count": len(operator_tasks),
                "incoming_tile_dependency_count": tile_incoming[operator.op_id],
                "outgoing_tile_dependency_count": tile_outgoing[operator.op_id],
                "macs": macs,
                "root_read_bytes": root_read_bytes,
                "root_write_bytes": root_write_bytes,
            }
        )

    tisa_dependency_count = sum(
        len(instruction.dependencies) for instruction in program.instructions
    )
    return {
        "schema_version": 1,
        "summary": {
            "operator_count": len(graph.operators),
            "tensor_count": len(graph.tensors),
            "tile_count": len(tile_graph.tiles),
            "tile_dependency_count": len(tile_graph.dependencies),
            "tisa_instruction_count": len(program.instructions),
            "tisa_dependency_count": tisa_dependency_count,
            "payload_task_count": len(backend_artifact.execution_graph.tasks),
            "macs": total_macs,
            "root_read_bytes": total_root_read_bytes,
            "root_write_bytes": total_root_write_bytes,
        },
        "dependencies": {
            "model": tile_graph.attributes.get("dependency_model", "unspecified"),
            "graph_edge_count": len(graph.edges),
            "conservative_edge_count": tile_graph.attributes.get(
                "conservative_edge_count", 0
            ),
            "avoided_all_to_all_dependencies": tile_graph.attributes.get(
                "avoided_all_to_all_dependencies", 0
            ),
        },
        "operators": operator_statistics,
    }


__all__ = ["build_compile_statistics"]
