"""Import versioned RTL completion traces into MXU timing profiles.

The importer is intentionally offline.  It turns a small, stable trace
contract into ``npu_ooo.systolic_mxu_profile.v1`` and never runs a simulator
or infers timing from a waveform implicitly.  This keeps the interval being
calibrated visible in the generated profile metadata.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


TRACE_FORMAT = "npu_ooo.rtl_completion_trace.v1"
PROFILE_FORMAT = "npu_ooo.systolic_mxu_profile.v1"
INTERVALS = ("compute_start_to_compute_done", "descriptor_issue_to_done")
AGGREGATIONS = ("max", "median", "p95")


def _number(value: object, *, field_name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive finite" if positive else "finite non-negative"
        raise ValueError(f"{field_name} must be a {qualifier} number")
    return result


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True)
class RTLCompletionRecord:
    """One observed MXU instruction and its completion markers."""

    instruction_id: str
    batch: int
    m: int
    n: int
    k: int
    descriptor_issue_cycle: float | None = None
    compute_start_cycle: float | None = None
    compute_done_cycle: float | None = None
    psb_write_done_cycle: float | None = None
    initiation_interval_cycles: float | None = None
    clock_period_ns: float | None = None
    source: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, index: int = 0) -> "RTLCompletionRecord":
        if not isinstance(payload, Mapping):
            raise ValueError(f"records[{index}] must be an object")
        instruction_id = payload.get("instruction_id", f"record-{index}")
        if not isinstance(instruction_id, str) or not instruction_id:
            raise ValueError(f"records[{index}].instruction_id must be a non-empty string")
        shape = payload.get("shape", payload)
        if not isinstance(shape, Mapping):
            raise ValueError(f"records[{index}].shape must be an object")

        def optional_number(name: str, *, positive: bool = False) -> float | None:
            value = payload.get(name)
            if value in (None, ""):
                return None
            return _number(value, field_name=f"records[{index}].{name}", positive=positive)

        return cls(
            instruction_id=instruction_id,
            batch=_positive_int(shape.get("batch", 1), field_name=f"records[{index}].batch"),
            m=_positive_int(shape.get("m"), field_name=f"records[{index}].m"),
            n=_positive_int(shape.get("n"), field_name=f"records[{index}].n"),
            k=_positive_int(shape.get("k"), field_name=f"records[{index}].k"),
            descriptor_issue_cycle=optional_number("descriptor_issue_cycle"),
            compute_start_cycle=optional_number("compute_start_cycle"),
            compute_done_cycle=optional_number("compute_done_cycle"),
            psb_write_done_cycle=optional_number("psb_write_done_cycle"),
            initiation_interval_cycles=optional_number(
                "initiation_interval_cycles", positive=True
            ),
            clock_period_ns=optional_number("clock_period_ns", positive=True),
            source=(str(payload["source"]) if payload.get("source") not in (None, "") else None),
        )

    @property
    def shape_key(self) -> tuple[int, int, int, int]:
        return (self.batch, self.m, self.n, self.k)

    def interval(self, selection: str) -> tuple[float, str, str]:
        if selection not in INTERVALS:
            raise ValueError(f"interval must be one of {', '.join(INTERVALS)}")
        if selection == "compute_start_to_compute_done":
            start_name, end_name = "compute_start_cycle", "compute_done_cycle"
            start, end = self.compute_start_cycle, self.compute_done_cycle
        else:
            start_name = "descriptor_issue_cycle"
            end_name = "psb_write_done_cycle" if self.psb_write_done_cycle is not None else "compute_done_cycle"
            start, end = self.descriptor_issue_cycle, getattr(self, end_name)
        if start is None or end is None:
            raise ValueError(
                f"record '{self.instruction_id}' lacks {start_name} or {end_name} for {selection}"
            )
        duration = end - start
        if duration <= 0:
            raise ValueError(
                f"record '{self.instruction_id}' has non-positive {selection} interval: {duration:g}"
            )
        return duration, start_name, end_name


def _records_from_payload(payload: Any) -> tuple[RTLCompletionRecord, Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if payload.get("format") != TRACE_FORMAT:
            raise ValueError(f"RTL trace format must be '{TRACE_FORMAT}'")
        raw_records = payload.get("records")
        metadata = payload.get("metadata", {})
    else:
        raise ValueError("RTL trace payload must be an object")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("RTL trace records must be a non-empty list")
    if not isinstance(metadata, Mapping):
        raise ValueError("RTL trace metadata must be an object")
    records = tuple(RTLCompletionRecord.from_mapping(item, index=i) for i, item in enumerate(raw_records))
    return records, metadata


def load_rtl_completion_trace(path: str | Path) -> tuple[tuple[RTLCompletionRecord, ...], Mapping[str, Any]]:
    """Load the versioned JSON trace or a flat CSV with the same field names."""

    trace_path = Path(path)
    try:
        if trace_path.suffix.lower() == ".csv":
            with trace_path.open(newline="", encoding="utf-8") as handle:
                rows = []
                for row in csv.DictReader(handle):
                    normalized = dict(row)
                    for field in ("batch", "m", "n", "k"):
                        if normalized.get(field) not in (None, ""):
                            try:
                                normalized[field] = int(normalized[field])
                            except (TypeError, ValueError) as exc:
                                raise ValueError(f"CSV field '{field}' must be an integer") from exc
                    for field in (
                        "descriptor_issue_cycle",
                        "compute_start_cycle",
                        "compute_done_cycle",
                        "psb_write_done_cycle",
                        "initiation_interval_cycles",
                        "clock_period_ns",
                    ):
                        if normalized.get(field) not in (None, ""):
                            try:
                                normalized[field] = float(normalized[field])
                            except (TypeError, ValueError) as exc:
                                raise ValueError(f"CSV field '{field}' must be numeric") from exc
                    rows.append(normalized)
            return _records_from_payload({"format": TRACE_FORMAT, "records": rows, "metadata": {}})
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot load RTL completion trace '{trace_path}': {exc}") from exc
    return _records_from_payload(payload)


def _aggregate(values: Iterable[float], method: str) -> float:
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("cannot aggregate an empty sample set")
    if method not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {', '.join(AGGREGATIONS)}")
    if method == "max":
        return samples[-1]
    if method == "median":
        return float(median(samples))
    rank = max(0, math.ceil(0.95 * len(samples)) - 1)
    return samples[rank]


def build_systolic_mxu_profile(
    records: Iterable[RTLCompletionRecord],
    *,
    metadata: Mapping[str, Any] | None = None,
    interval: str = "compute_start_to_compute_done",
    aggregation: str = "median",
    unmatched_matmul: str = "error",
    name: str = "systolic_mxu_profile",
) -> dict[str, Any]:
    """Aggregate RTL observations into the provider's stable profile schema."""

    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {', '.join(INTERVALS)}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {', '.join(AGGREGATIONS)}")
    if unmatched_matmul not in {"analytical", "error"}:
        raise ValueError("unmatched_matmul must be analytical or error")
    if not isinstance(name, str) or not name:
        raise ValueError("profile name must be a non-empty string")

    grouped: dict[tuple[int, int, int, int], list[RTLCompletionRecord]] = {}
    for record in records:
        grouped.setdefault(record.shape_key, []).append(record)
    if not grouped:
        raise ValueError("at least one RTL completion record is required")

    profiles: list[dict[str, Any]] = []
    interval_end_fields: set[str] = set()
    for shape in sorted(grouped):
        samples: list[float] = []
        starts: list[float] = []
        explicit_iis: list[float] = []
        for record in grouped[shape]:
            duration, start_name, end_name = record.interval(interval)
            interval_end_fields.add(end_name)
            samples.append(duration)
            starts.append(getattr(record, start_name))  # type: ignore[arg-type]
            if record.initiation_interval_cycles is not None:
                explicit_iis.append(record.initiation_interval_cycles)
        if explicit_iis:
            initiation_interval = _aggregate(explicit_iis, aggregation)
        else:
            deltas = [right - left for left, right in zip(sorted(starts), sorted(starts)[1:]) if right > left]
            initiation_interval = _aggregate(deltas, aggregation) if deltas else _aggregate(samples, aggregation)
        profiles.append(
            {
                "shape": {"batch": shape[0], "m": shape[1], "n": shape[2], "k": shape[3]},
                "duration_cycles": _aggregate(samples, aggregation),
                "initiation_interval_cycles": initiation_interval,
                "attributes": {"sample_count": len(samples)},
            }
        )

    source = dict(metadata or {})
    source.update(
        {
            "trace_format": TRACE_FORMAT,
            "interval": interval,
            "interval_end_fields": sorted(interval_end_fields),
            "aggregation": aggregation,
            "record_count": sum(len(items) for items in grouped.values()),
            "shape_count": len(grouped),
            "calibration_status": source.get("calibration_status", "rtl-observed"),
        }
    )
    return {
        "format": PROFILE_FORMAT,
        "name": name,
        "metadata": source,
        "unmatched_matmul": unmatched_matmul,
        "matmul_profiles": profiles,
    }


def import_rtl_completion_trace(
    path: str | Path,
    *,
    interval: str = "compute_start_to_compute_done",
    aggregation: str = "median",
    unmatched_matmul: str = "error",
    name: str = "systolic_mxu_profile",
) -> dict[str, Any]:
    records, metadata = load_rtl_completion_trace(path)
    return build_systolic_mxu_profile(
        records,
        metadata=metadata,
        interval=interval,
        aggregation=aggregation,
        unmatched_matmul=unmatched_matmul,
        name=name,
    )


__all__ = [
    "AGGREGATIONS",
    "INTERVALS",
    "PROFILE_FORMAT",
    "TRACE_FORMAT",
    "RTLCompletionRecord",
    "build_systolic_mxu_profile",
    "import_rtl_completion_trace",
    "load_rtl_completion_trace",
]
