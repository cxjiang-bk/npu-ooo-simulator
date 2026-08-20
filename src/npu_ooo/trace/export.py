from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from npu_ooo.scheduler import ScheduleResult


def write_artifact_json(artifact: Any, path: str | Path) -> None:
    payload = artifact.to_dict() if hasattr(artifact, "to_dict") else artifact
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_json(result: ScheduleResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(result: ScheduleResult, path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "resource",
                "instance",
                "issue",
                "start",
                "finish",
                "duration",
                "queue_wait",
                "ready_wait",
                "dependency_ready",
                "resource_ready",
            ),
        )
        writer.writeheader()
        for timing in result.timings:
            writer.writerow(timing.to_dict())


def write_svg(result: ScheduleResult, path: str | Path, *, width: int = 1600, row_height: int = 28) -> None:
    """Write a dependency-free SVG swimlane for quick local inspection."""

    lanes = sorted({(timing.resource, timing.instance) for timing in result.timings})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    label_width = 180
    chart_width = max(1, width - label_width - 20)
    scale = chart_width / max(result.total_cycles, 1.0)
    height = 48 + row_height * len(lanes)
    palette = {"load": "#2f6f9f", "matmul": "#b45f06", "store": "#3b7d3b"}
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="12" y="22" font-family="sans-serif" font-size="16">policy={html.escape(result.policy)} total={result.total_cycles:g} cycles</text>',
    ]
    for index, (resource, instance) in enumerate(lanes):
        y = 42 + index * row_height
        elements.append(f'<line x1="{label_width}" y1="{y + row_height - 4}" x2="{width}" y2="{y + row_height - 4}" stroke="#dddddd"/>')
        elements.append(f'<text x="12" y="{y + 16}" font-family="sans-serif" font-size="12">{html.escape(resource)}[{instance}]</text>')
    for task in result.timings:
        lane = (task.resource, task.instance)
        y = 42 + lane_index[lane] * row_height + 3
        x = label_width + task.start * scale
        rect_width = max(1.0, task.duration * scale)
        primitive = task.task_id.rsplit(".", 1)[-1]
        color = palette.get("matmul" if primitive == "mxu" else primitive, "#777777")
        elements.append(
            f'<rect x="{x:.2f}" y="{y}" width="{rect_width:.2f}" height="{row_height - 8}" fill="{color}" rx="2"><title>{html.escape(task.task_id)} [{task.start:g}, {task.finish:g}]</title></rect>'
        )
    elements.append("</svg>")
    Path(path).write_text("\n".join(elements), encoding="utf-8")
