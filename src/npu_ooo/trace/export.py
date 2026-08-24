from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from npu_ooo.scheduler import ScheduleResult

from .layout import artifact_path, finalize_artifact


_PRIMITIVE_PALETTE = {
    "runtime_submit": "#546e7a",
    "tisa_instruction": "#263238",
    "load": "#377eb8",
    "load_transpose": "#00a6a6",
    "store": "#4daf4a",
    "matmul": "#ff7f00",
    "elementwise": "#984ea3",
    "square": "#6a5acd",
    "center": "#8c6d31",
    "reduce": "#e41a1c",
    "reduce_max": "#c0392b",
    "reduce_sum": "#e6550d",
    "reduce_sum_square": "#d95f0e",
    "layernorm_mean": "#a65628",
    "exp": "#f781bf",
    "normalize": "#1b9e77",
    "rmsnorm": "#00876c",
    "layernorm": "#2a9d8f",
}

_FALLBACK_PALETTE = (
    "#5c6bc0",
    "#bc5090",
    "#7a5195",
    "#ef5675",
    "#ffa600",
    "#3d9970",
    "#8c564b",
    "#607d8b",
)

_PRIMITIVE_ORDER = tuple(_PRIMITIVE_PALETTE)


def write_artifact_json(artifact: Any, path: str | Path) -> None:
    payload = artifact.to_dict() if hasattr(artifact, "to_dict") else artifact
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    finalize_artifact(target, compatibility)


def write_json(result: ScheduleResult, path: str | Path) -> None:
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    finalize_artifact(target, compatibility)


def write_csv(result: ScheduleResult, path: str | Path) -> None:
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "tile_id",
                "operator_id",
                "primitive",
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
        issue_details = {
            event.task_id: dict(event.details)
            for event in result.events
            if event.event == "ISSUE"
        }
        for timing in result.timings:
            task_id = timing.task_id
            details = issue_details.get(task_id, {})
            writer.writerow(
                {
                    **timing.to_dict(),
                    "tile_id": details.get("tile_id", ""),
                    "operator_id": details.get("operator_id", ""),
                    "primitive": details.get("primitive", ""),
                }
            )
    finalize_artifact(target, compatibility)


def write_instruction_csv(result: ScheduleResult, path: str | Path) -> None:
    """Write scheduler-visible TISA timings separately from payload tasks."""

    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    issue_details = {
        event.task_id: dict(event.details)
        for event in result.events
        if event.event == "TISA_ISSUE"
    }
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "tisa_id",
                "tile_id",
                "operator_id",
                "op_type",
                "unit_map",
                "payload_task_count",
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
        for timing in result.instruction_timings:
            details = issue_details.get(timing.task_id, {})
            writer.writerow(
                {
                    "tisa_id": timing.task_id,
                    "tile_id": details.get("tile_id", ""),
                    "operator_id": details.get("operator_id", ""),
                    "op_type": details.get("op_type", ""),
                    "unit_map": details.get("unit_map", ""),
                    "payload_task_count": details.get("payload_task_count", ""),
                    "resource": timing.resource,
                    "instance": timing.instance,
                    "issue": timing.issue,
                    "start": timing.start,
                    "finish": timing.finish,
                    "duration": timing.duration,
                    "queue_wait": timing.queue_wait,
                    "ready_wait": timing.ready_wait,
                    "dependency_ready": timing.dependency_ready,
                    "resource_ready": timing.resource_ready,
                }
            )
    finalize_artifact(target, compatibility)


def _primitive_color(primitive: str) -> str:
    if primitive in _PRIMITIVE_PALETTE:
        return _PRIMITIVE_PALETTE[primitive]
    stable_index = sum((index + 1) * ord(character) for index, character in enumerate(primitive))
    return _FALLBACK_PALETTE[stable_index % len(_FALLBACK_PALETTE)]


def _nice_tick_step(total_cycles: float, chart_width: float) -> float:
    """Choose a readable 1/2/5-based major tick interval."""

    if total_cycles <= 0:
        return 1.0
    desired_ticks = max(2, int(chart_width // 110))
    raw_step = total_cycles / desired_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        multiplier = 1
    elif normalized <= 2:
        multiplier = 2
    elif normalized <= 5:
        multiplier = 5
    else:
        multiplier = 10
    return float(multiplier * magnitude)


def _tick_label(value: float) -> str:
    return f"{value:g}"


def _legend_layout(primitives: tuple[str, ...], width: int) -> tuple[list[tuple[str, int, int]], int]:
    """Lay out legend entries without letting them overflow the SVG width."""

    if not primitives:
        return [], 0
    left = 92
    right = max(left + 1, width - 20)
    x = left
    row = 0
    positions: list[tuple[str, int, int]] = []
    for primitive in primitives:
        item_width = max(78, 36 + len(primitive) * 7)
        if x + item_width > right and x > left:
            row += 1
            x = left
        positions.append((primitive, x, row))
        x += item_width
    return positions, row + 1


def write_svg(result: ScheduleResult, path: str | Path, *, width: int = 1600, row_height: int = 28) -> None:
    """Write a dependency-free SVG swimlane for quick local inspection."""

    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    all_timings = (*result.runtime_timings, *result.instruction_timings, *result.timings)
    lanes = sorted({(timing.resource, timing.instance) for timing in all_timings})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    primitive_issue_details = {
        event.task_id: dict(event.details)
        for event in result.events
        if event.event == "ISSUE"
    }
    tisa_issue_details = {
        event.task_id: dict(event.details)
        for event in result.events
        if event.event == "TISA_ISSUE"
    }
    runtime_issue_details = {
        event.task_id: dict(event.details)
        for event in result.events
        if event.event == "RUNTIME_SUBMIT_START"
    }
    primitive_by_task = {
        timing.task_id: str(
            primitive_issue_details.get(timing.task_id, {}).get("primitive", "unknown")
        )
        for timing in result.timings
    }
    primitive_by_task.update(
        {timing.task_id: "tisa_instruction" for timing in result.instruction_timings}
    )
    primitive_by_task.update(
        {timing.task_id: "runtime_submit" for timing in result.runtime_timings}
    )
    present_primitives = set(primitive_by_task.values())
    primitive_rank = {primitive: index for index, primitive in enumerate(_PRIMITIVE_ORDER)}
    primitives = tuple(
        sorted(
            present_primitives,
            key=lambda primitive: (primitive_rank.get(primitive, len(primitive_rank)), primitive),
        )
    )
    label_width = 180
    chart_width = max(1, width - label_width - 20)
    scale = chart_width / max(result.total_cycles, 1.0)
    legend_positions, legend_rows = _legend_layout(primitives, width)
    legend_top = 34
    legend_row_height = 22
    axis_top = legend_top + legend_rows * legend_row_height + 10
    lane_top = axis_top + 30
    chart_bottom = lane_top + row_height * len(lanes)
    height = chart_bottom + 16
    major_step = _nice_tick_step(result.total_cycles, chart_width)
    minor_step = major_step / 5
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<title>{html.escape(result.policy)} scheduling swimlane</title>',
        f'<desc>{len(result.runtime_timings)} runtime chunks, {len(result.instruction_timings)} TISA instructions and {len(result.timings)} payload tasks over {result.total_cycles:g} cycles.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="12" y="22" font-family="sans-serif" font-size="16" fill="#1f2933">policy={html.escape(result.policy)}  total={result.total_cycles:g} cycles  runtime={len(result.runtime_timings)}  TISA={len(result.instruction_timings)}  payload={len(result.timings)}</text>',
        '<g id="legend" font-family="sans-serif" font-size="11" fill="#27313a">',
        f'<text x="12" y="{legend_top + 13}" font-weight="600">Primitive</text>',
    ]
    for primitive, x, row in legend_positions:
        y = legend_top + row * legend_row_height
        elements.append(
            f'<rect x="{x}" y="{y + 2}" width="14" height="14" fill="{_primitive_color(primitive)}" stroke="#334155" stroke-width="0.5"/>'
        )
        elements.append(
            f'<text x="{x + 20}" y="{y + 13}">{html.escape("TISA instruction" if primitive == "tisa_instruction" else "Runtime submit" if primitive == "runtime_submit" else primitive)}</text>'
        )
    elements.append("</g>")

    elements.append('<g id="lane-backgrounds">')
    for index, _lane in enumerate(lanes):
        if index % 2:
            y = lane_top + index * row_height
            elements.append(
                f'<rect x="{label_width}" y="{y}" width="{chart_width}" height="{row_height}" fill="#f8fafc"/>'
            )
    elements.append("</g>")

    elements.extend(
        (
            '<g id="cycle-axis" font-family="sans-serif" font-size="10" fill="#4b5563">',
            f'<text x="{label_width - 10}" y="{axis_top + 5}" text-anchor="end" font-size="11" font-weight="600">Cycle</text>',
            f'<line x1="{label_width}" y1="{axis_top + 10}" x2="{label_width + chart_width}" y2="{axis_top + 10}" stroke="#64748b" stroke-width="1"/>',
        )
    )
    minor_tick = 0.0
    while minor_tick <= result.total_cycles + 1e-9:
        x = label_width + minor_tick * scale
        is_major = abs((minor_tick / major_step) - round(minor_tick / major_step)) < 1e-8
        if not is_major:
            elements.append(
                f'<line x1="{x:.2f}" y1="{axis_top + 7}" x2="{x:.2f}" y2="{chart_bottom}" stroke="#edf1f5" stroke-width="1"/>'
            )
        minor_tick += minor_step
    major_tick = 0.0
    while major_tick <= result.total_cycles + 1e-9:
        x = label_width + major_tick * scale
        elements.append(
            f'<line x1="{x:.2f}" y1="{axis_top + 6}" x2="{x:.2f}" y2="{chart_bottom}" stroke="#d5dce3" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{axis_top}" text-anchor="middle">{_tick_label(major_tick)}</text>'
        )
        major_tick += major_step
    elements.append("</g>")

    for index, (resource, instance) in enumerate(lanes):
        y = lane_top + index * row_height
        elements.append(f'<line x1="{label_width}" y1="{y + row_height}" x2="{label_width + chart_width}" y2="{y + row_height}" stroke="#d9e0e6"/>')
        elements.append(f'<text x="12" y="{y + 18}" font-family="sans-serif" font-size="12" fill="#27313a">{html.escape(resource)}[{instance}]</text>')
    for task in all_timings:
        lane = (task.resource, task.instance)
        y = lane_top + lane_index[lane] * row_height + 4
        x = label_width + task.start * scale
        rect_width = max(1.0, task.duration * scale)
        primitive = primitive_by_task[task.task_id]
        color = _primitive_color(primitive)
        details = (
            runtime_issue_details.get(task.task_id, {})
            if primitive == "runtime_submit"
            else tisa_issue_details.get(task.task_id, {})
            if primitive == "tisa_instruction"
            else primitive_issue_details.get(task.task_id, {})
        )
        extra = (
            f" | op={details.get('op_type', '')} | tile={details.get('tile_id', '')}"
            f" | payload={details.get('payload_task_count', '')}"
            if primitive == "tisa_instruction"
            else ""
        )
        elements.append(
            f'<rect x="{x:.2f}" y="{y}" width="{rect_width:.2f}" height="{row_height - 8}" fill="{color}" stroke="#ffffff" stroke-width="0.6" rx="2"><title>{html.escape(task.task_id)} | {html.escape(primitive)}{html.escape(extra)} | issue={task.issue:g} start={task.start:g} finish={task.finish:g} duration={task.duration:g}</title></rect>'
        )
    elements.append("</svg>")
    target.write_text("\n".join(elements), encoding="utf-8")
    finalize_artifact(target, compatibility)


def write_png(result: ScheduleResult, path: str | Path, *, width: int = 1600, row_height: int = 28) -> None:
    """Rasterize the swimlane SVG using an installed ImageMagick or librsvg binary."""

    converter = shutil.which("convert") or shutil.which("magick") or shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("PNG swimlane export requires ImageMagick ('convert') or 'rsvg-convert'")
    target, compatibility = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as handle:
        temporary_svg = Path(handle.name)
    try:
        write_svg(result, temporary_svg, width=width, row_height=row_height)
        if Path(converter).name == "rsvg-convert":
            command = [converter, str(temporary_svg), "-o", str(target)]
        else:
            command = [converter, str(temporary_svg), str(target)]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        finalize_artifact(target, compatibility)
    finally:
        temporary_svg.unlink(missing_ok=True)
