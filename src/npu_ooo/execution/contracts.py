"""Issue/feedback protocol between scheduler and execution backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from npu_ooo.ir import BoundTISADescriptor, CompletionToken


@dataclass(frozen=True)
class PayloadRegistration:
    handle: str
    tisa_id: str
    resource: str
    task_count: int
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.handle or not self.tisa_id or not self.resource:
            issues.append("payload registration identities must not be empty")
        if self.task_count <= 0:
            issues.append("payload registration task_count must be positive")
        return tuple(issues)


@dataclass(frozen=True)
class PayloadStep:
    task_id: str
    primitive: str
    start_offset: float
    finish_offset: float


@dataclass(frozen=True)
class PayloadEstimate:
    handle: str
    resource: str
    duration_cycles: float
    initiation_interval_cycles: float
    steps: tuple[PayloadStep, ...]

    @property
    def duration(self) -> float:
        return self.duration_cycles


@dataclass(frozen=True)
class IssueRequest:
    descriptor: BoundTISADescriptor
    cycle: float

    def validate(self) -> tuple[str, ...]:
        issues = list(self.descriptor.validate())
        if self.cycle < 0:
            issues.append("execution issue cycle must be non-negative")
        return tuple(issues)


@dataclass(frozen=True)
class IssueReceipt:
    accepted: bool
    descriptor_id: str
    payload_handle: str
    resource: str | None = None
    instance: int | None = None
    issue_cycle: float | None = None
    expected_done_cycle: float | None = None
    rejection_reason: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.descriptor_id or not self.payload_handle:
            issues.append("issue receipt identities must not be empty")
        if self.accepted:
            if self.resource is None or self.instance is None:
                issues.append("accepted issue must name resource and instance")
            if self.issue_cycle is None or self.expected_done_cycle is None:
                issues.append("accepted issue must name issue/done cycles")
            elif self.expected_done_cycle <= self.issue_cycle:
                issues.append("accepted issue completion must follow issue")
            if self.rejection_reason is not None:
                issues.append("accepted issue cannot carry a rejection reason")
        elif not self.rejection_reason:
            issues.append("rejected issue must carry a reason")
        return tuple(issues)


@dataclass(frozen=True)
class ExecutionFeedback:
    descriptor_id: str
    token: CompletionToken
    kind: str
    cycle: float
    resource: str
    instance: int
    condition: str = "complete"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.token.validate())
        if not self.descriptor_id or not self.kind or not self.resource or not self.condition:
            issues.append("execution feedback fields must not be empty")
        if self.cycle < 0 or self.instance < 0:
            issues.append("execution feedback cycle/instance must be non-negative")
        if self.kind not in {"execution_done", "partial_ready"}:
            issues.append(f"execution feedback kind '{self.kind}' is unsupported")
        return tuple(issues)


class ExecutionBackend(Protocol):
    """Resource owner and payload executor used by any scheduler."""

    name: str

    def registrations(self) -> tuple[PayloadRegistration, ...]: ...

    def estimate(self, payload_handle: str) -> PayloadEstimate: ...

    def can_accept(
        self, descriptor: BoundTISADescriptor, cycle: float
    ) -> tuple[bool, str | None]: ...

    def next_accept_cycle(self, descriptor: BoundTISADescriptor) -> float | None: ...

    def issue(self, request: IssueRequest) -> IssueReceipt: ...

    def advance(self, cycle: float) -> tuple[ExecutionFeedback, ...]: ...

    def next_feedback_cycle(self) -> float | None: ...

    def task_timings(self) -> tuple[Any, ...]: ...

    def trace_events(self) -> tuple[Any, ...]: ...

    def metrics(self) -> Mapping[str, Any]: ...


__all__ = [
    "ExecutionBackend",
    "ExecutionFeedback",
    "IssueReceipt",
    "IssueRequest",
    "PayloadRegistration",
    "PayloadEstimate",
    "PayloadStep",
]
