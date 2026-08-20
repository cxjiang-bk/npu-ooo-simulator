from __future__ import annotations

import html
from pathlib import Path

from npu_ooo.ir import ExecutionGraph, OperatorGraph, TileGraph


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
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tile_graph_dot(graph: TileGraph, path: str | Path) -> None:
    lines = ["digraph tile_graph {", "  rankdir=LR;", "  node [shape=box,fontname=Helvetica,fontsize=9];"]
    for tile in graph.tiles:
        bounds = ", ".join(f"{name}=[{start},{stop})" for name, start, stop in tile.bounds)
        label = f"{tile.tile_id}\n{bounds}"
        lines.append(f"  {_quote(tile.tile_id)} [label={_quote(label)}];")
    for dependency in graph.dependencies:
        label = dependency.tensor or dependency.kind
        lines.append(
            f"  {_quote(dependency.producer)} -> {_quote(dependency.consumer)} [label={_quote(label)},fontsize=8];"
        )
    lines.append("}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    Path(path).write_text("\n".join(elements), encoding="utf-8")
