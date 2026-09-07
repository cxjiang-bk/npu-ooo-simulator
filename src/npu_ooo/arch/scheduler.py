"""Explicit control-pipeline parameters for tile scheduling experiments."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SchedulerPipelineConfig:
    receive_width: int = 1
    dispatch_width: int = 1
    select_width: int = 1
    issue_width: int = 1
    completion_width: int = 1
    retire_width: int = 1
    dispatch_latency: int = 1
    select_latency: int = 1
    wakeup_latency: int = 1
    completion_latency: int = 0
    retire_latency: int = 1
    iq_entries: int = 8
    inflight_entries: int = 16
    max_cycles: int = 1_000_000

    def validate(self) -> tuple[str, ...]:
        zero_allowed = {"wakeup_latency", "completion_latency"}
        return tuple(
            f"scheduler pipeline {name} must be an integer >= {minimum}"
            for name, value in asdict(self).items()
            for minimum in (0 if name in zero_allowed else 1,)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchedulerPipelineConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("scheduler pipeline must be a JSON object")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "unknown scheduler pipeline fields: " + ", ".join(sorted(unknown))
            )
        result = cls(**payload)
        issues = result.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return result
