from __future__ import annotations

import html
from pathlib import Path

from npu_ooo.ir import BackendArtifact, ExecutionGraph, OperatorGraph, TileGraph

from .layout import artifact_path, finalize_artifact


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def write_operator_graph_dot(graph: OperatorGraph, path: str | Path) -> None:
    lines = ["digraph operator_graph {", "  rankdir=LR;", "  node [fontname=Helvetica];"]
    for tensor in graph.tensors:
        shape = "x".join(str(value) for value in tensor.shape)
        label = f"{tensor.name}\n{shape} {tensor.dtype}"
        lines.append(f"  {_quote('tensor:' + tensor.name)} [shape=ellipse,label={_quote(label)}];")
    for operator in graph.operators:
        label = f"{operator.op_id}\n{operator.normalized_type}"
        lines.append(
            f"  {_quote('op:' + operator.op_id)} [shape=box,style=filled,fillcolor=lightgoldenrod1,label={_quote(label)}];"
        )
        for tensor in operator.inputs:
            lines.append(f"  {_quote('tensor:' + tensor)} -> {_quote('op:' + operator.op_id)};")
        for tensor in operator.outputs:
            lines.append(f"  {_quote('op:' + operator.op_id)} -> {_quote('tensor:' + tensor)};")
    lines.append("}")
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finalize_artifact(target, compatibility)


def write_tile_graph_dot(graph: TileGraph, path: str | Path) -> None:
    lines = ["digraph tile_graph {", "  rankdir=LR;", "  compound=true;", "  node [shape=box,fontname=Helvetica,fontsize=9];"]
    by_operator: dict[str, list] = {}
    for tile in graph.tiles:
        by_operator.setdefault(tile.operator_id, []).append(tile)
    for operator_id, tiles in by_operator.items():
        lines.append(f"  subgraph {_quote('cluster_' + operator_id)} {{")
        lines.append(f"    label={_quote(operator_id)};")
        for tile in tiles:
            bounds = ", ".join(
                f"{name}=[{start},{stop})" for name, start, stop in tile.bounds
            )
            coordinates = ", ".join(
                f"{name}={value}" for name, value in tile.coordinates
            )
            label = (
                f"{tile.tile_id}\niter: {coordinates}\nrange: {bounds}"
                f"\nstage={tile.stage_id}"
            )
            lines.append(f"    {_quote(tile.tile_id)} [label={_quote(label)}];")
        lines.append("  }")
    colors = {
        "region_data": "#2563eb",
        "data": "#2563eb",
        "state": "#7c3aed",
        "accumulate": "#ea580c",
        "buffer_reuse": "#dc2626",
        "control": "#475569",
    }
    for dependency in graph.dependencies:
        label = (
            f"{dependency.kind}/{dependency.hazard_kind}"
            f"\n{dependency.tensor or ''}\n{dependency.condition}"
        )
        lines.append(
            f"  {_quote(dependency.producer)} -> {_quote(dependency.consumer)} "
            f"[label={_quote(label)},fontsize=8,color={_quote(colors.get(dependency.kind, '#64748b'))}];"
        )
    lines.append("}")
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finalize_artifact(target, compatibility)


def program_hierarchy(artifact: BackendArtifact) -> dict:
    """Return bounded model/operator/tile/TISA/payload navigation data."""

    tasks = {item.task_id: item for item in artifact.execution_graph.tasks}
    operators: dict[str, dict] = {}
    for instruction in artifact.program.instructions:
        operator = operators.setdefault(
            instruction.operator_id,
            {"operator_id": instruction.operator_id, "tiles": {}},
        )
        tile = operator["tiles"].setdefault(
            instruction.tile_id,
            {"tile_id": instruction.tile_id, "instructions": []},
        )
        tile["instructions"].append(
            {
                "tisa_id": instruction.tisa_id,
                "op_type": instruction.op_type,
                "unit": instruction.unit_map.unit,
                "abstract_tisa_id": instruction.attributes.get("abstract_tisa_id"),
                "target_stage_kind": instruction.attributes.get("target_stage_kind"),
                "operands": [
                    {
                        "name": operand.name,
                        "access": operand.normalized_access,
                        "memory": operand.tile_mem.physical_space,
                        "buffer_id": operand.tile_mem.buffer_id,
                        "allocation_id": operand.tile_mem.allocation_id,
                    }
                    for operand in instruction.operands
                ],
                "dependencies": [item.to_dict() for item in instruction.dependencies],
                "payload": [
                    {
                        "task_id": task_id,
                        "primitive": tasks[task_id].primitive,
                        "resource": tasks[task_id].resource,
                    }
                    for task_id in artifact.payloads[instruction.tisa_id]
                ],
            }
        )
    return {
        "schema_version": 1,
        "artifact_id": artifact.artifact_id,
        "program_id": artifact.program.program_id,
        "shared_workload_hash": artifact.attributes.get("shared_workload_hash"),
        "operators": [
            {
                "operator_id": item["operator_id"],
                "tiles": list(item["tiles"].values()),
            }
            for item in operators.values()
        ],
        "static_control": (
            artifact.static_control.to_dict()
            if artifact.static_control is not None
            else None
        ),
    }


def write_tisa_graph_dot(
    artifact: BackendArtifact,
    path: str | Path,
    *,
    max_nodes: int = 256,
    include_static_control: bool = True,
) -> None:
    instructions = artifact.program.instructions[:max_nodes]
    visible = {item.tisa_id for item in instructions}
    lines = [
        "digraph tisa_program {",
        "  rankdir=LR;",
        "  compound=true;",
        "  node [shape=box,fontname=Helvetica,fontsize=8];",
    ]
    by_operator: dict[str, list] = {}
    for instruction in instructions:
        by_operator.setdefault(instruction.operator_id, []).append(instruction)
    for operator_id, rows in by_operator.items():
        lines.append(f"  subgraph {_quote('cluster_' + operator_id)} {{")
        lines.append(f"    label={_quote(operator_id)};")
        for instruction in rows:
            memories = ",".join(
                sorted({item.tile_mem.physical_space for item in instruction.operands})
            )
            abstract = instruction.attributes.get("abstract_tisa_id", "")
            label = (
                f"{instruction.tisa_id}\n{instruction.op_type} @ {instruction.unit_map.unit}"
                f"\nmem={memories}\nabstract={abstract}"
            )
            lines.append(
                f"    {_quote(instruction.tisa_id)} [label={_quote(label)}];"
            )
        lines.append("  }")
    edge_colors = {"RAW": "#2563eb", "WAR": "#ea580c", "WAW": "#dc2626", "BUFFER_REUSE": "#dc2626"}
    for instruction in instructions:
        for dependency in instruction.dependencies:
            if dependency.source not in visible:
                continue
            label = f"{dependency.kind}\n{dependency.condition}"
            lines.append(
                f"  {_quote(dependency.source)} -> {_quote(instruction.tisa_id)} "
                f"[label={_quote(label)},color={_quote(edge_colors.get(dependency.kind, '#64748b'))}];"
            )
    if include_static_control and artifact.static_control is not None:
        control_colors = {"set": "#16a34a", "wait": "#ca8a04", "fence": "#dc2626"}
        for stream in artifact.static_control.streams:
            lines.append(f"  subgraph {_quote('cluster_control_' + stream.stream_id)} {{")
            lines.append(f"    label={_quote('static stream ' + stream.stream_id)};")
            previous = None
            for command in stream.commands:
                if command.tisa_id not in visible:
                    continue
                if command.kind == "issue":
                    current = command.tisa_id if command.tisa_id in visible else None
                else:
                    current = command.command_id
                    lines.append(
                        f"    {_quote(current)} [shape=diamond,label={_quote(command.kind + ':' + ','.join(command.event_ids))},"
                        f"color={_quote(control_colors[command.kind])}];"
                    )
                if previous is not None and current is not None:
                    lines.append(
                        f"    {_quote(previous)} -> {_quote(current)} "
                        "[style=dashed,color=\"#64748b\",label=\"static order\"];"
                    )
                if current is not None:
                    previous = current
            lines.append("  }")
    if len(artifact.program.instructions) > max_nodes:
        lines.append(
            f"  truncated [shape=note,label={_quote(f'truncated: showing {max_nodes}/{len(artifact.program.instructions)} TISA')}];"
        )
    lines.append("}")
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finalize_artifact(target, compatibility)


def write_execution_graph_dot(graph: ExecutionGraph, path: str | Path) -> None:
    lines = ["digraph execution_graph {", "  rankdir=LR;", "  compound=true;", "  node [shape=box,fontname=Helvetica,fontsize=8];"]
    by_operator: dict[str, list] = {}
    for task in graph.tasks:
        by_operator.setdefault(task.operator_id, []).append(task)
    for operator_id, tasks in by_operator.items():
        lines.append(f"  subgraph {_quote('cluster_' + operator_id)} {{")
        lines.append(f"    label={_quote(operator_id)};")
        for task in tasks:
            label = f"{task.task_id}\n{task.primitive} @ {task.resource}"
            lines.append(f"    {_quote(task.task_id)} [label={_quote(label)}];")
        lines.append("  }")
    for task in graph.tasks:
        for predecessor in task.predecessors:
            lines.append(f"  {_quote(predecessor)} -> {_quote(task.task_id)};")
    lines.append("}")
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finalize_artifact(target, compatibility)


def write_operator_graph_svg(graph: OperatorGraph, path: str | Path) -> None:
    """Render a small dependency-free SVG for the semantic operator graph."""

    operators = {operator.op_id: operator for operator in graph.operators}
    levels: dict[str, int] = {}
    for operator_id in graph.topological_order():
        predecessors = [edge.producer for edge in graph.edges if edge.consumer == operator_id]
        levels[operator_id] = max((levels[predecessor] + 1 for predecessor in predecessors), default=0)
    consumers: dict[str, list[str]] = {}
    producers: dict[str, str] = {}
    for operator in graph.operators:
        for tensor in operator.inputs:
            consumers.setdefault(tensor, []).append(operator.op_id)
        for tensor in operator.outputs:
            producers[tensor] = operator.op_id

    columns: dict[int, list[tuple[str, str]]] = {}
    for operator in graph.operators:
        columns.setdefault(levels[operator.op_id] * 2 + 1, []).append(("operator", operator.op_id))
    for tensor in graph.tensors:
        producer = producers.get(tensor.name)
        if producer is not None:
            column = levels[producer] * 2 + 2
        elif tensor.name in consumers:
            column = min(levels[consumer] * 2 for consumer in consumers[tensor.name])
        else:
            column = 0
        columns.setdefault(column, []).append(("tensor", tensor.name))

    node_width = 154
    node_height = 54
    x_step = 230
    y_step = 86
    margin_x = 40
    margin_y = 58
    max_column = max(columns, default=0)
    max_rows = max((len(nodes) for nodes in columns.values()), default=1)
    width = margin_x * 2 + max_column * x_step + node_width
    height = margin_y * 2 + max_rows * y_step
    positions: dict[tuple[str, str], tuple[float, float]] = {}
    for column, nodes in columns.items():
        block_height = (len(nodes) - 1) * y_step
        first_y = margin_y + (max_rows - 1) * y_step / 2 - block_height / 2
        for index, node in enumerate(nodes):
            positions[node] = (margin_x + column * x_step, first_y + index * y_step)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="#555"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{margin_x}" y="28" font-family="sans-serif" font-size="17" font-weight="bold">{html.escape(graph.graph_id)}</text>',
    ]

    def connect(source: tuple[str, str], target: tuple[str, str]) -> None:
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        elements.append(
            f'<line x1="{source_x + node_width}" y1="{source_y + node_height / 2}" x2="{target_x}" y2="{target_y + node_height / 2}" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )

    for operator in graph.operators:
        for tensor in operator.inputs:
            connect(("tensor", tensor), ("operator", operator.op_id))
        for tensor in operator.outputs:
            connect(("operator", operator.op_id), ("tensor", tensor))

    tensors = {tensor.name: tensor for tensor in graph.tensors}
    for node, (x, y) in positions.items():
        kind, name = node
        if kind == "operator":
            operator = operators[name]
            elements.append(f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" rx="4" fill="#f4d88a" stroke="#7a6120"/>')
            elements.append(f'<text x="{x + node_width / 2}" y="{y + 22}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">{html.escape(name)}</text>')
            elements.append(f'<text x="{x + node_width / 2}" y="{y + 41}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(operator.normalized_type)}</text>')
        else:
            tensor = tensors[name]
            shape = " x ".join(str(value) for value in tensor.shape)
            elements.append(f'<ellipse cx="{x + node_width / 2}" cy="{y + node_height / 2}" rx="{node_width / 2}" ry="{node_height / 2}" fill="#dcecf5" stroke="#35657d"/>')
            elements.append(f'<text x="{x + node_width / 2}" y="{y + 22}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">{html.escape(name)}</text>')
            elements.append(f'<text x="{x + node_width / 2}" y="{y + 41}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(shape + ' ' + tensor.dtype)}</text>')
    elements.append("</svg>")
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(elements), encoding="utf-8")
    finalize_artifact(target, compatibility)
