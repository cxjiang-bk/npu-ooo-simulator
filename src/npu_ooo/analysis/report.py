"""Offline bubble, wait-chain and static/dynamic comparison analysis."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RunAnalysis:
    run_dir: str
    policy: str
    total_cycles: float
    identities: Mapping[str, Any]
    resources: tuple[Mapping[str, Any], ...]
    bubbles: tuple[Mapping[str, Any], ...]
    waits: tuple[Mapping[str, Any], ...]
    critical_wait_chain: tuple[Mapping[str, Any], ...]
    bottleneck: Mapping[str, Any]
    buffer_summary: Mapping[str, Any]
    timeline: Mapping[str, Any]
    assumptions: Mapping[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_dir": self.run_dir,
            "policy": self.policy,
            "total_cycles": self.total_cycles,
            "identities": dict(self.identities),
            "resources": [dict(item) for item in self.resources],
            "bubbles": [dict(item) for item in self.bubbles],
            "waits": [dict(item) for item in self.waits],
            "critical_wait_chain": [dict(item) for item in self.critical_wait_chain],
            "bottleneck": dict(self.bottleneck),
            "buffer_summary": dict(self.buffer_summary),
            "timeline": dict(self.timeline),
            "assumptions": dict(self.assumptions),
        }


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ValueError(f"analysis input is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"analysis input is invalid JSON '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"analysis input must be a JSON object: {path}")
    return value


def _physical_timings(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in summary.get("timings", ())
        if isinstance(item, Mapping)
    ]


def _merge_intervals(rows: list[Mapping[str, Any]]) -> list[tuple[float, float]]:
    intervals: list[list[float]] = []
    for row in sorted(rows, key=lambda item: (float(item["start"]), float(item["finish"]))):
        start, finish = float(row["start"]), float(row["finish"])
        if not intervals or start > intervals[-1][1]:
            intervals.append([start, finish])
        else:
            intervals[-1][1] = max(intervals[-1][1], finish)
    return [(item[0], item[1]) for item in intervals]


def _resource_analysis(
    timings: list[dict[str, Any]], total_cycles: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in timings:
        grouped.setdefault((str(item["resource"]), int(item["instance"])), []).append(item)
    global_start = min((float(item["start"]) for item in timings), default=0.0)
    global_end = max((float(item["finish"]) for item in timings), default=total_cycles)
    resources: list[dict[str, Any]] = []
    bubbles: list[dict[str, Any]] = []
    for (resource, instance), rows in sorted(grouped.items()):
        intervals = _merge_intervals(rows)
        busy = sum(end - start for start, end in intervals)
        cursor = global_start
        for index, (start, end) in enumerate(intervals):
            if start > cursor:
                phase = "startup" if index == 0 else "steady"
                next_task = min(
                    (
                        row
                        for row in rows
                        if abs(float(row["start"]) - start) < 1e-9
                    ),
                    key=lambda row: str(row["task_id"]),
                    default=None,
                )
                bubbles.append(
                    {
                        "bubble_id": f"{resource}.{instance}.{len(bubbles)}",
                        "resource": resource,
                        "instance": instance,
                        "start": cursor,
                        "end": start,
                        "duration": start - cursor,
                        "phase": phase,
                        "actionable": phase == "steady",
                        "attribution": "pending",
                        "next_task_id": (
                            str(next_task["task_id"])
                            if next_task is not None
                            else None
                        ),
                    }
                )
            cursor = max(cursor, end)
        if cursor < global_end:
            bubbles.append(
                {
                    "bubble_id": f"{resource}.{instance}.{len(bubbles)}",
                    "resource": resource,
                    "instance": instance,
                    "start": cursor,
                    "end": global_end,
                    "duration": global_end - cursor,
                    "phase": "drain",
                    "actionable": False,
                    "attribution": "no_later_physical_task_observed",
                }
            )
        resources.append(
            {
                "resource": resource,
                "instance": instance,
                "busy_cycles": busy,
                "observation_start": global_start,
                "observation_end": global_end,
                "utilization": busy / max(1.0, global_end - global_start),
                "task_count": len(rows),
            }
        )
    return resources, bubbles


def _bound_dependencies(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    loaded = _read_json(
        run_dir / "05_runtime" / "bound_device_program.json", required=False
    )
    return {
        str(item["instruction"]["tisa_id"]): [
            dict(dependency) for dependency in item.get("dependencies", ())
        ]
        for item in loaded.get("descriptors", ())
        if isinstance(item, Mapping) and isinstance(item.get("instruction"), Mapping)
    }


def _static_waits(run_dir: Path) -> list[dict[str, Any]]:
    perfetto = _read_json(run_dir / "07_trace" / "perfetto.json", required=False)
    waits: list[dict[str, Any]] = []
    for item in perfetto.get("traceEvents", ()):
        if not isinstance(item, Mapping):
            continue
        args = item.get("args", {})
        if not isinstance(args, Mapping) or args.get("event") not in {
            "STATIC_WAIT_INTERVAL",
            "STATIC_SET_WAIT",
            "STATIC_RESOURCE_WAIT",
            "STATIC_DESCRIPTOR_WAIT",
        }:
            continue
        waits.append(
            {
                "wait_id": str(item.get("name")),
                "reason": args.get("reason", "unknown"),
                "start": float(item.get("ts", 0.0)),
                "end": float(item.get("ts", 0.0)) + float(item.get("dur", 0.0)),
                "duration": float(item.get("dur", 0.0)),
                "blockers": list(args.get("blockers", ())),
                "buffer_slots": list(args.get("buffer_slots", ())),
                "stream_id": args.get("stream_id"),
                "source": "observed_static_control_event",
                "tisa_id": args.get("tisa_id"),
            }
        )
    return waits


def _instruction_waits(
    summary: Mapping[str, Any], dependencies: Mapping[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    pipeline = summary.get("metrics", {}).get("instruction_pipeline", {})
    waits: list[dict[str, Any]] = []
    for tisa_id, stages in pipeline.items():
        if not isinstance(stages, Mapping):
            continue
        received = stages.get("received")
        issued = stages.get("issued")
        if received is None or issued is None or float(issued) <= float(received):
            continue
        blockers: list[dict[str, Any]] = []
        for dependency in dependencies.get(str(tisa_id), ()):
            source = dependency.get("source", {})
            source_id = source.get("tisa_id") if isinstance(source, Mapping) else None
            source_stages = pipeline.get(source_id, {})
            completion = (
                source_stages.get("completed")
                if isinstance(source_stages, Mapping)
                else None
            )
            if completion is not None and float(completion) > float(received):
                blockers.append(
                    {
                        "tisa_id": source_id,
                        "condition": dependency.get("condition"),
                        "kind": dependency.get("kind"),
                        "completion": completion,
                        "provenance": dependency.get("provenance", {}),
                    }
                )
        reason = "data_dependency" if blockers else "queue_resource_or_control"
        waits.append(
            {
                "wait_id": f"{tisa_id}.receive_to_issue",
                "tisa_id": tisa_id,
                "reason": reason,
                "start": float(received),
                "end": float(issued),
                "duration": float(issued) - float(received),
                "blockers": blockers,
                "source": "inferred_from_saved_lifecycle",
            }
        )
    return waits


def _attribute_bubbles(
    bubbles: list[dict[str, Any]],
    waits: list[dict[str, Any]],
    task_parents: Mapping[str, str],
) -> None:
    for bubble in bubbles:
        if bubble["phase"] != "steady":
            continue
        next_tisa = task_parents.get(str(bubble.get("next_task_id")))
        overlapping = [
            wait
            for wait in waits
            if wait["start"] < bubble["end"] and wait["end"] > bubble["start"]
            and (next_tisa is None or wait.get("tisa_id") == next_tisa)
        ]
        if not overlapping:
            bubble["attribution"] = "unknown_from_saved_trace"
            bubble["blockers"] = []
            continue
        selected = max(overlapping, key=lambda item: item["duration"])
        bubble["attribution"] = selected["reason"]
        bubble["blockers"] = selected.get("blockers", [])
        bubble["wait_id"] = selected["wait_id"]


def _task_parents(run_dir: Path) -> dict[str, str]:
    perfetto = _read_json(run_dir / "07_trace" / "perfetto.json", required=False)
    result: dict[str, str] = {}
    for item in perfetto.get("traceEvents", ()):
        if not isinstance(item, Mapping):
            continue
        args = item.get("args", {})
        if not isinstance(args, Mapping):
            continue
        parent = args.get("parent_tisa_id")
        if parent:
            result[str(item.get("name"))] = str(parent)
    return result


def _queue_transitions(run_dir: Path) -> list[dict[str, Any]]:
    perfetto = _read_json(run_dir / "07_trace" / "perfetto.json", required=False)
    state = {
        "reception": 0,
        "wq": 0,
        "iq": 0,
        "rob": 0,
        "completion_pending": 0,
        "wq_by_unit": {},
        "iq_by_unit": {},
    }
    transitions: list[dict[str, Any]] = []
    relevant = (
        "TISA_RECEIVE",
        "TISA_DISPATCH",
        "TISA_SELECT",
        "TISA_ISSUE",
        "TISA_EXECUTION_DONE",
        "TISA_COMPLETE",
        "TISA_RETIRE",
    )
    events = []
    for item in perfetto.get("traceEvents", ()):
        if not isinstance(item, Mapping):
            continue
        args = item.get("args", {})
        event = args.get("event") if isinstance(args, Mapping) else None
        if event in relevant:
            resource = args.get("unit_map") if isinstance(args, Mapping) else None
            if isinstance(resource, Mapping):
                resource = resource.get("unit")
            if not resource and isinstance(args, Mapping):
                resource = args.get("resource")
            if not resource:
                category = str(item.get("cat", ""))
                if "/" in category:
                    resource = category.split("/", 1)[1].split("[", 1)[0]
            events.append(
                (
                    float(item.get("ts", 0.0)),
                    str(event),
                    str(item.get("name")),
                    str(resource) if resource else None,
                )
            )
    order = {name: index for index, name in enumerate(relevant)}
    has_dispatch = any(event == "TISA_DISPATCH" for _cycle, event, _tid, _resource in events)
    for cycle, event, tisa_id, resource in sorted(
        events, key=lambda item: (item[0], order.get(item[1], 99))
    ):
        if event == "TISA_RECEIVE":
            state["reception"] += 1
        elif event == "TISA_DISPATCH":
            state["reception"] = max(0, state["reception"] - 1)
            state["wq"] += 1
            state["rob"] += 1
            if resource:
                state["wq_by_unit"][resource] = state["wq_by_unit"].get(resource, 0) + 1
        elif event == "TISA_SELECT":
            state["wq"] = max(0, state["wq"] - 1)
            state["iq"] += 1
            if resource:
                state["wq_by_unit"][resource] = max(
                    0, state["wq_by_unit"].get(resource, 0) - 1
                )
                state["iq_by_unit"][resource] = state["iq_by_unit"].get(resource, 0) + 1
        elif event == "TISA_ISSUE":
            if has_dispatch:
                state["iq"] = max(0, state["iq"] - 1)
                if resource:
                    state["iq_by_unit"][resource] = max(
                        0, state["iq_by_unit"].get(resource, 0) - 1
                    )
            else:
                state["reception"] = max(0, state["reception"] - 1)
                state["rob"] += 1
        elif event == "TISA_EXECUTION_DONE":
            state["completion_pending"] += 1
        elif event == "TISA_COMPLETE":
            state["completion_pending"] = max(0, state["completion_pending"] - 1)
        elif event == "TISA_RETIRE":
            state["rob"] = max(0, state["rob"] - 1)
        snapshot = {
            "cycle": cycle,
            "event": event,
            "tisa_id": tisa_id,
            "wq_by_unit": dict(state["wq_by_unit"]),
            "iq_by_unit": dict(state["iq_by_unit"]),
        }
        snapshot.update(
            {
                key: value
                for key, value in state.items()
                if key not in {"wq_by_unit", "iq_by_unit"}
            }
        )
        transitions.append(
            snapshot
        )
    return transitions


def _critical_chain(waits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not waits:
        return []
    current = max(waits, key=lambda item: item["duration"])
    chain = [current]
    seen = {current.get("tisa_id")}
    by_tisa = {item.get("tisa_id"): item for item in waits if item.get("tisa_id")}
    while current.get("blockers"):
        blocker = max(
            (item for item in current["blockers"] if isinstance(item, Mapping)),
            key=lambda item: float(item.get("completion", 0.0)),
            default=None,
        )
        if blocker is None or blocker.get("tisa_id") in seen:
            break
        predecessor = by_tisa.get(blocker.get("tisa_id"))
        if predecessor is None:
            chain.append({"blocker": dict(blocker), "source": "dependency_terminal"})
            break
        chain.append(predecessor)
        seen.add(predecessor.get("tisa_id"))
        current = predecessor
    return chain


def analyze_run(run_dir: str | Path) -> RunAnalysis:
    root = Path(run_dir).expanduser().resolve()
    summary = _read_json(root / "06_simulation" / "summary.json")
    manifest = _read_json(root / "manifest.json", required=False)
    timings = _physical_timings(summary)
    task_parents = _task_parents(root)
    for item in timings:
        item["parent_tisa_id"] = task_parents.get(str(item.get("task_id")))
    resources, bubbles = _resource_analysis(timings, float(summary["total_cycles"]))
    dependencies = _bound_dependencies(root)
    observed_static_waits = _static_waits(root)
    for wait in observed_static_waits:
        wait["event_blockers"] = list(wait.get("blockers", ()))
        target = wait.get("tisa_id")
        if target in dependencies:
            wait["blockers"] = [
                {
                    "tisa_id": (
                        item.get("source", {}).get("tisa_id")
                        if isinstance(item.get("source"), Mapping)
                        else None
                    ),
                    "condition": item.get("condition"),
                    "kind": item.get("kind"),
                    "provenance": item.get("provenance", {}),
                }
                for item in dependencies[target]
            ]
    waits = [*observed_static_waits, *_instruction_waits(summary, dependencies)]
    _attribute_bubbles(bubbles, waits, task_parents)
    actionable = [item for item in bubbles if item["actionable"]]
    longest = max(actionable, key=lambda item: item["duration"], default=None)
    busiest = max(resources, key=lambda item: item["utilization"], default=None)
    buffer = _read_json(root / "05_runtime" / "buffer_lifecycle.json", required=False)
    buffer_summary = {
        memory: {
            key: value
            for key, value in item.items()
            if key
            in {
                "allocated_bytes",
                "valid_bytes",
                "capacity_bytes",
                "protected_peak_bytes",
                "retained_peak_bytes",
            }
        }
        for memory, item in buffer.get("memories", {}).items()
    }
    metrics = summary.get("metrics", {})
    return RunAnalysis(
        run_dir=str(root),
        policy=str(summary.get("policy", manifest.get("policy", "unknown"))),
        total_cycles=float(summary["total_cycles"]),
        identities={
            "compile_artifact_id": manifest.get("compile_artifact_id"),
            "shared_workload_hash": metrics.get("shared_workload_hash"),
            "static_control_hash": metrics.get("static_control_hash"),
            "dynamic_control_hash": metrics.get("dynamic_control_hash"),
            "timing_provider": manifest.get("timing_provider"),
            "event_backend": manifest.get("event_backend"),
            "machine_hash": manifest.get("machine_hash"),
        },
        resources=tuple(resources),
        bubbles=tuple(bubbles),
        waits=tuple(sorted(waits, key=lambda item: item["duration"], reverse=True)),
        critical_wait_chain=tuple(_critical_chain(waits)),
        bottleneck={
            "longest_actionable_bubble": longest,
            "highest_observed_utilization": busiest,
            "interpretation": (
                "candidate_only; validate with a controlled parameter change"
                if longest is not None
                else "no steady physical-EU bubble observed"
            ),
        },
        buffer_summary=buffer_summary,
        timeline={
            "physical_intervals": timings,
            "instruction_pipeline": summary.get("metrics", {}).get(
                "instruction_pipeline", {}
            ),
            "queue_transitions": _queue_transitions(root),
            "buffer_occupancy": {
                memory: {
                    "protected": item.get("protected_occupancy", []),
                    "retained": item.get("retained_occupancy", []),
                    "capacity_bytes": item.get("capacity_bytes"),
                    "allocated_bytes": item.get("allocated_bytes"),
                }
                for memory, item in buffer.get("memories", {}).items()
            },
        },
        assumptions={
            "busy_time": "physical payload task intervals, not TISA issue-complete",
            "bubble_phases": "global physical-work window; edge gaps are startup/drain",
            "wait_attribution": "observed static intervals, otherwise lifecycle inference",
            "unknown_policy": "unrecoverable blockers remain explicitly unknown",
            "overlapping_stalls": "raw reasons are not summed into exclusive total time",
            "calibration": metrics.get("calibration_status", "unknown"),
        },
    )


def compare_runs(first: RunAnalysis, second: RunAnalysis) -> dict[str, Any]:
    first_hash = first.identities.get("shared_workload_hash")
    second_hash = second.identities.get("shared_workload_hash")
    same_workload = first_hash is not None and first_hash == second_hash
    return {
        "schema_version": 1,
        "first": {"run_dir": first.run_dir, "policy": first.policy, "cycles": first.total_cycles},
        "second": {"run_dir": second.run_dir, "policy": second.policy, "cycles": second.total_cycles},
        "same_shared_workload": same_workload,
        "shared_workload_hash": first_hash if same_workload else None,
        "cycle_delta_second_minus_first": second.total_cycles - first.total_cycles,
        "speedup_first_over_second": (
            first.total_cycles / second.total_cycles if second.total_cycles else None
        ),
        "fairness": {
            "machine_hash_equal": first.identities.get("machine_hash")
            == second.identities.get("machine_hash"),
            "timing_provider_equal": first.identities.get("timing_provider")
            == second.identities.get("timing_provider"),
            "control_programs": {
                "first_static": first.identities.get("static_control_hash"),
                "second_static": second.identities.get("static_control_hash"),
                "first_dynamic": first.identities.get("dynamic_control_hash"),
                "second_dynamic": second.identities.get("dynamic_control_hash"),
            },
        },
        "warning": None
        if same_workload
        else "shared workload hashes differ; cycle delta is not a fair scheduler comparison",
    }


def _nice_cycle_step(total_cycles: float, target_ticks: int = 10) -> float:
    if total_cycles <= 0:
        return 1.0
    raw = total_cycles / max(2, target_ticks)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    multiplier = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    return float(multiplier * magnitude)


def _timeline_curve(
    values: Any,
    *,
    total_cycles: float,
    value_key: str,
) -> list[tuple[float, float]]:
    raw_rows = sorted(
        (
            (float(item.get("cycle", 0.0)), float(item.get(value_key, 0.0)))
            for item in values
            if isinstance(item, Mapping)
        ),
        key=lambda item: item[0],
    )
    rows: list[tuple[float, float]] = []
    for cycle, value in raw_rows:
        cycle = min(total_cycles, max(0.0, cycle))
        if rows and rows[-1][0] == cycle:
            rows[-1] = (cycle, value)
        else:
            rows.append((cycle, value))
    if not rows:
        return [(0.0, 0.0), (total_cycles, 0.0)]
    if rows[0][0] > 0:
        rows.insert(0, (0.0, 0.0))
    if rows[-1][0] < total_cycles:
        rows.append((total_cycles, rows[-1][1]))
    return rows


def _step_path(
    rows: list[tuple[float, float]],
    *,
    x_scale,
    y_scale,
) -> str:
    if not rows:
        return ""
    commands = [f"M{x_scale(rows[0][0]):.2f},{y_scale(rows[0][1]):.2f}"]
    previous = rows[0][1]
    for cycle, value in rows[1:]:
        x = x_scale(cycle)
        commands.append(f"H{x:.2f}")
        if value != previous:
            commands.append(f"V{y_scale(value):.2f}")
        previous = value
    return " ".join(commands)


def _byte_label(value: float | int | None) -> str:
    if value is None:
        return "unbounded"
    number = float(value)
    for scale, suffix in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if abs(number) >= scale:
            return f"{number / scale:.2f} {suffix}"
    return f"{number:g} B"


def _joint_timeline_svg(analysis: RunAnalysis) -> str:
    width = 1440
    label_width = 250
    right_margin = 28
    chart_width = width - label_width - right_margin
    total_cycles = max(1.0, analysis.total_cycles)
    physical = analysis.timeline.get("physical_intervals", [])
    lanes = sorted(
        {(str(item["resource"]), int(item["instance"])) for item in physical}
    )
    queue = analysis.timeline.get("queue_transitions", [])
    buffers = analysis.timeline.get("buffer_occupancy", {})
    physical_height = 30
    queue_height = 58
    buffer_height = 86
    section_gap = 30
    axis_top = 52
    physical_top = axis_top + 24
    queue_top = physical_top + len(lanes) * physical_height + section_gap
    queue_units = sorted(
        {
            str(item.get("resource"))
            for item in physical
            if isinstance(item, Mapping) and item.get("resource")
        }
        | {
            unit
            for row in queue
            if isinstance(row, Mapping)
            for field in ("wq_by_unit", "iq_by_unit")
            for unit in (row.get(field, {}) or {})
        }
    )
    queue_fields = [
        ("wq_by_unit", unit, f"WQ[{unit}]", "#2563eb")
        for unit in queue_units
    ] + [
        ("iq_by_unit", unit, f"IQ[{unit}]", "#16a34a")
        for unit in queue_units
    ] + [
        ("rob", None, "ROB occupancy", "#7c3aed"),
        ("completion_pending", None, "Completion pending", "#ea580c"),
    ]
    queue_visible = bool(queue_units or queue)
    buffer_top = queue_top + (len(queue_fields) * queue_height if queue_visible else 0) + section_gap
    chart_bottom = buffer_top + len(buffers) * buffer_height
    height = int(chart_bottom + 46)

    def x_scale(cycle: float) -> float:
        return label_width + cycle / total_cycles * chart_width

    parts = [
        f'<svg id="joint-timeline" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Physical execution, queue occupancy, and buffer occupancy over cycles">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{label_width}" y="18" font-size="12" font-weight="600">Cycle</text>',
    ]

    legend_x = label_width + 70
    for label, color, dashed in (
        ("WQ", "#2563eb", False),
        ("IQ", "#16a34a", False),
        ("ROB", "#7c3aed", False),
        ("Completion pending", "#ea580c", False),
        ("Protected bytes", "#dc2626", False),
        ("Retained bytes", "#16a34a", True),
    ):
        dash = ' stroke-dasharray="5 3"' if dashed else ""
        parts.append(
            f'<line x1="{legend_x}" y1="14" x2="{legend_x + 22}" y2="14" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
            f'<text x="{legend_x + 27}" y="18" font-size="11">{html.escape(label)}</text>'
        )
        legend_x += 50 + len(label) * 6.2

    tick_step = _nice_cycle_step(total_cycles)
    minor_step = tick_step / 2
    tick = 0.0
    while tick <= total_cycles + 1e-9:
        x = x_scale(tick)
        major = abs((tick / tick_step) - round(tick / tick_step)) < 1e-8
        grid_class = "cycle-grid-major" if major else "cycle-grid-minor"
        grid_color = "#cbd5e1" if major else "#edf2f7"
        grid_width = "1" if major else "0.7"
        parts.append(
            f'<line class="{grid_class}" x1="{x:.2f}" y1="{axis_top}" '
            f'x2="{x:.2f}" y2="{chart_bottom}" stroke="{grid_color}" '
            f'stroke-width="{grid_width}"/>'
        )
        endpoint_gap = x_scale(total_cycles) - x
        if major and (tick <= 0 or endpoint_gap <= 1e-8 or endpoint_gap >= 48):
            parts.append(
                f'<text x="{x:.2f}" y="{axis_top - 8}" text-anchor="middle" '
                f'font-size="11">{tick:g}</text>'
            )
        tick += minor_step
    if total_cycles % tick_step:
        x = x_scale(total_cycles)
        parts.append(
            f'<line x1="{x:.2f}" y1="{axis_top}" x2="{x:.2f}" y2="{chart_bottom}" '
            'stroke="#94a3b8" stroke-width="1"/>'
            f'<text x="{x:.2f}" y="{axis_top - 8}" text-anchor="end" '
            f'font-size="11">{analysis.total_cycles:g}</text>'
        )

    parts.append(
        f'<text x="5" y="{physical_top - 8}" font-size="12" font-weight="600">Physical EU execution</text>'
    )
    for row, (resource, instance) in enumerate(lanes):
        y = physical_top + row * physical_height
        parts.append(
            f'<rect data-chart-frame x="{label_width}" y="{y}" width="{chart_width}" '
            f'height="{physical_height - 4}" fill="none" stroke="#d8dee9"/>'
            f'<text x="8" y="{y + 18}" font-size="12">{html.escape(resource)}[{instance}]</text>'
        )
        for item in physical:
            if str(item["resource"]) != resource or int(item["instance"]) != instance:
                continue
            x = x_scale(float(item["start"]))
            rect_width = max(
                1.0,
                x_scale(float(item["finish"])) - x_scale(float(item["start"])),
            )
            parent = str(item.get("parent_tisa_id") or "")
            title = f"{item['task_id']} parent={parent} [{item['start']},{item['finish']}]"
            parts.append(
                f'<rect class="timeline-item" data-tisa="{html.escape(parent)}" '
                f'x="{x:.2f}" y="{y + 4}" width="{rect_width:.2f}" height="18" '
                f'fill="#3b82f6"><title>{html.escape(title)}</title></rect>'
            )

    if queue_visible:
        parts.append(
            f'<text x="5" y="{queue_top - 8}" font-size="12" font-weight="600">'
            'Scheduler state (WQ occupancy / IQ occupancy by EU)</text>'
        )
        for row, (field, unit, label, color) in enumerate(queue_fields):
            y = queue_top + row * queue_height
            if unit is None:
                curve_values = queue
                value_key = field
            else:
                curve_values = [
                    {
                        "cycle": item.get("cycle", 0.0),
                        "value": (item.get(field, {}) or {}).get(unit, 0),
                    }
                    for item in queue
                    if isinstance(item, Mapping)
                ]
                value_key = "value"
            if not curve_values:
                curve_values = [{"cycle": 0.0, value_key: 0.0}]
            curve = _timeline_curve(
                curve_values, total_cycles=total_cycles, value_key=value_key
            )
            peak = max((value for _cycle, value in curve), default=0.0)

            def queue_y(value: float, *, top=y, maximum=max(1.0, peak)) -> float:
                return top + queue_height - 10 - value / maximum * (queue_height - 20)

            path = _step_path(curve, x_scale=x_scale, y_scale=queue_y)
            parts.append(
                f'<rect class="queue-panel" data-chart-frame x="{label_width}" '
                f'y="{y}" width="{chart_width}" '
                f'height="{queue_height - 4}" fill="none" stroke="#d8dee9"/>'
                f'<text x="8" y="{y + 19}" font-size="12" font-weight="600">{label}</text>'
                f'<text x="8" y="{y + 38}" font-size="11" fill="#64748b">0 … {peak:g} entries</text>'
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2">'
                f'<title>{label}, peak={peak:g}</title></path>'
            )

    parts.append(
        f'<text x="5" y="{buffer_top - 8}" font-size="12" font-weight="600">Buffer occupancy (independent y-scale per memory)</text>'
    )
    for row, (memory, item) in enumerate(sorted(buffers.items())):
        y = buffer_top + row * buffer_height
        protected = _timeline_curve(
            item.get("protected", []),
            total_cycles=total_cycles,
            value_key="bytes",
        )
        retained = _timeline_curve(
            item.get("retained", []),
            total_cycles=total_cycles,
            value_key="bytes",
        )
        observed_peak = max(
            [value for _cycle, value in protected]
            + [value for _cycle, value in retained]
            + [1.0]
        )
        y_max = max(1.0, observed_peak)

        def buffer_y(value: float, *, top=y, maximum=y_max) -> float:
            return top + buffer_height - 12 - value / maximum * (buffer_height - 26)

        protected_path = _step_path(protected, x_scale=x_scale, y_scale=buffer_y)
        retained_path = _step_path(retained, x_scale=x_scale, y_scale=buffer_y)
        parts.append(
            f'<rect class="buffer-panel" data-chart-frame x="{label_width}" '
            f'y="{y}" width="{chart_width}" '
            f'height="{buffer_height - 4}" fill="none" stroke="#d8dee9"/>'
            f'<text x="8" y="{y + 19}" font-size="12" font-weight="600">{html.escape(memory)}</text>'
            f'<text x="8" y="{y + 37}" font-size="11" fill="#64748b">peak {_byte_label(observed_peak)}</text>'
            f'<text x="8" y="{y + 54}" font-size="11" fill="#64748b">allocated {_byte_label(item.get("allocated_bytes"))}</text>'
            f'<text x="8" y="{y + 70}" font-size="11" fill="#64748b">capacity {_byte_label(item.get("capacity_bytes"))}</text>'
            f'<text x="{label_width - 8}" y="{y + 13}" text-anchor="end" font-size="10">{_byte_label(observed_peak)}</text>'
            f'<text x="{label_width - 8}" y="{y + buffer_height - 10}" text-anchor="end" font-size="10">0</text>'
            f'<path d="{protected_path}" fill="none" stroke="#dc2626" stroke-width="2">'
            f'<title>{html.escape(memory)} protected, peak={_byte_label(max(value for _cycle, value in protected))}</title></path>'
            f'<path d="{retained_path}" fill="none" stroke="#16a34a" stroke-width="2" '
            f'stroke-dasharray="5 3"><title>{html.escape(memory)} retained, '
            f'peak={_byte_label(max(value for _cycle, value in retained))}</title></path>'
        )
    parts.append(
        f'<text x="{label_width + chart_width / 2}" y="{height - 12}" '
        'text-anchor="middle" font-size="12" font-weight="600">Cycle</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def write_analysis_report(
    analysis: RunAnalysis,
    output_dir: str | Path,
    *,
    comparison: Mapping[str, Any] | None = None,
    hierarchy: Mapping[str, Any] | None = None,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "analysis.json").write_text(
        json.dumps(analysis.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    if comparison is not None:
        (root / "comparison.json").write_text(
            json.dumps(dict(comparison), indent=2, sort_keys=True), encoding="utf-8"
        )
    lines = [
        "# Scheduling analysis",
        "",
        f"- run: `{analysis.run_dir}`",
        f"- policy: `{analysis.policy}`",
        f"- total cycles: `{analysis.total_cycles:g}`",
        f"- shared workload: `{analysis.identities.get('shared_workload_hash')}`",
        "",
        "## Physical EU utilization",
        "",
        "| EU | busy cycles | utilization | tasks |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {item['resource']}[{item['instance']}] | {item['busy_cycles']:g} | "
        f"{item['utilization']:.3f} | {item['task_count']} |"
        for item in analysis.resources
    )
    lines.extend(["", "## Longest waits", ""])
    for item in analysis.waits[:10]:
        lines.append(
            f"- `{item['wait_id']}`: {item['duration']:g} cycles, "
            f"reason={item['reason']}, blockers={item.get('blockers', [])}"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report separates observation from inference. A long wait is a candidate "
            "bottleneck only; confirm it with a controlled parameter change.",
        ]
    )
    if comparison is not None:
        lines.extend(
            [
                "",
                "## Comparison",
                "",
                f"- same shared workload: `{comparison.get('same_shared_workload')}`",
                f"- cycle delta (second-first): `{comparison.get('cycle_delta_second_minus_first')}`",
                f"- speedup first/second: `{comparison.get('speedup_first_over_second')}`",
                f"- warning: `{comparison.get('warning')}`",
            ]
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = analysis.to_dict()
    encoded = html.escape(json.dumps(payload, ensure_ascii=False))
    resource_rows = "".join(
        "<tr><td>{}[{}]</td><td>{:g}</td><td>{:.1%}</td><td>{}</td></tr>".format(
            html.escape(str(item["resource"])),
            item["instance"],
            item["busy_cycles"],
            item["utilization"],
            item["task_count"],
        )
        for item in analysis.resources
    )
    bubble_rows_parts = []
    for item in sorted(
        analysis.bubbles, key=lambda row: row["duration"], reverse=True
    )[:100]:
        blocker = next(
            (
                value.get("tisa_id")
                for value in item.get("blockers", ())
                if isinstance(value, Mapping) and value.get("tisa_id")
            ),
            "",
        )
        bubble_rows_parts.append(
            "<tr data-id='{}' data-tisa='{}'><td>{}[{}]</td><td>{:g}–{:g}</td>"
            "<td>{:g}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(item["bubble_id"])),
                html.escape(str(blocker)),
                html.escape(str(item["resource"])),
                item["instance"],
                item["start"],
                item["end"],
                item["duration"],
                html.escape(str(item["phase"])),
                html.escape(str(item["attribution"])),
            )
        )
    bubble_rows = "".join(bubble_rows_parts)
    buffer_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(memory),
            item.get("allocated_bytes"),
            item.get("protected_peak_bytes"),
            item.get("retained_peak_bytes"),
            item.get("capacity_bytes"),
        )
        for memory, item in analysis.buffer_summary.items()
    )
    hierarchy_rows: list[str] = []
    if hierarchy is not None:
        count = 0
        for operator in hierarchy.get("operators", ()):
            if count >= 256:
                break
            tiles = []
            for tile in operator.get("tiles", ()):
                instructions = []
                for instruction in tile.get("instructions", ()):
                    if count >= 256:
                        break
                    tisa_id = str(instruction.get("tisa_id", ""))
                    instructions.append(
                        f'<button class="graph-node" data-tisa="{html.escape(tisa_id)}">'
                        f'{html.escape(tisa_id)} · {html.escape(str(instruction.get("op_type")))} '
                        f'@ {html.escape(str(instruction.get("unit")))}</button>'
                    )
                    count += 1
                tiles.append(
                    f'<details><summary>{html.escape(str(tile.get("tile_id")))}</summary>'
                    + "".join(instructions)
                    + "</details>"
                )
            hierarchy_rows.append(
                f'<details><summary>{html.escape(str(operator.get("operator_id")))}</summary>'
                + "".join(tiles)
                + "</details>"
            )
    hierarchy_html = "".join(hierarchy_rows) or "<p>No compile hierarchy available.</p>"
    buffer_detail = _read_json(
        Path(analysis.run_dir) / "05_runtime" / "buffer_lifecycle.json",
        required=False,
    )
    version_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td><button class='graph-node' data-tisa='{}'>{}</button>"
        "</td><td>{}</td><td>{:g}–{:g}</td></tr>".format(
            html.escape(str(item.get("version_id"))),
            html.escape(str(item.get("memory"))),
            html.escape(str(item.get("producer"))),
            html.escape(str(item.get("producer"))),
            html.escape(", ".join(str(value) for value in item.get("consumers", ()))),
            float(item.get("valid_cycle", 0)),
            float(item.get("release_cycle", 0)),
        )
        for item in buffer_detail.get("versions", ())[:256]
    )
    document = f"""<!doctype html>
<meta charset="utf-8"><title>Scheduling analysis</title>
<style>
body{{font:14px system-ui;margin:24px;color:#172033}} table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}
th,td{{border:1px solid #d8dee9;padding:6px;text-align:left}} th{{background:#eef2f7}}
tr:hover,tr.selected{{background:#fff3bf}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{border:1px solid #d8dee9;border-radius:6px;padding:12px;min-width:180px}}
code{{word-break:break-all}} details{{margin:10px 0}} .note{{color:#5b6472}}
.graph-node{{display:block;border:0;background:#eef2ff;margin:3px;padding:4px;text-align:left;cursor:pointer}}
.timeline-item.selected{{stroke:#111;stroke-width:2;opacity:.55}}
</style>
<h1>Scheduling analysis</h1>
<div class="cards"><div class="card"><b>Policy</b><br>{html.escape(analysis.policy)}</div>
<div class="card"><b>Total cycles</b><br>{analysis.total_cycles:g}</div>
<div class="card"><b>Workload hash</b><br><code>{html.escape(str(analysis.identities.get('shared_workload_hash')))}</code></div></div>
<h2>Joint timeline</h2><p class="note">Physical EU intervals, separate reconstructed scheduler-state panels, and protected/retained buffer occupancy share one cycle axis.</p>
{_joint_timeline_svg(analysis)}<pre id="selection">Click a physical interval to inspect its parent TISA.</pre>
<h2>Physical EU utilization and bubbles</h2><p class="note">Busy intervals come from payload tasks, not TISA issue→complete spans.</p>
<table><tr><th>EU</th><th>Busy</th><th>Utilization</th><th>Tasks</th></tr>{resource_rows}</table>
<table id="bubbles"><tr><th>EU</th><th>Interval</th><th>Cycles</th><th>Phase</th><th>Attribution</th></tr>{bubble_rows}</table>
<h2>Buffer occupancy</h2><table><tr><th>Memory</th><th>Allocated</th><th>Protected peak</th><th>Retained peak</th><th>Capacity</th></tr>{buffer_rows}</table>
<details><summary>Buffer versions and producer/consumers</summary><table><tr><th>Version</th><th>Memory</th><th>Producer</th><th>Consumers</th><th>Valid–release</th></tr>{version_rows}</table></details>
<h2>Operator → tile → TISA navigator</h2>{hierarchy_html}
<details><summary>Critical wait chain</summary><pre>{html.escape(json.dumps(payload['critical_wait_chain'], indent=2, ensure_ascii=False))}</pre></details>
<details><summary>Full auditable analysis JSON</summary><pre id="data">{encoded}</pre></details>
<p>Program navigation: <a href="program_hierarchy.json">hierarchy JSON</a> · <a href="tisa_graph.dot">TISA DOT</a> · <a href="../07_trace/perfetto.json">Perfetto</a></p>
<script>
function selectTisa(id){{document.querySelectorAll('.timeline-item').forEach(x=>x.classList.toggle('selected',id && x.dataset.tisa===id));document.getElementById('selection').textContent=id?'selected TISA: '+id:'No associated TISA in saved trace.';}}
document.querySelectorAll('#bubbles tr[data-id]').forEach(r=>r.onclick=()=>{{document.querySelectorAll('#bubbles .selected').forEach(x=>x.classList.remove('selected'));r.classList.add('selected');selectTisa(r.dataset.tisa)}});
document.querySelectorAll('.timeline-item,.graph-node').forEach(r=>r.onclick=()=>selectTisa(r.dataset.tisa));
</script>
"""
    (root / "report.html").write_text(document, encoding="utf-8")


__all__ = ["RunAnalysis", "analyze_run", "compare_runs", "write_analysis_report"]
