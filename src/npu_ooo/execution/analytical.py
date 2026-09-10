"""Analytical payload executor implementing the device execution contract."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import heapq
import math

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import BackendArtifact, BoundTISADescriptor, CompletionToken
from npu_ooo.simulator.core import (
    AnalyticalTimingModel,
    TaskTiming,
    TimingModel,
    TraceEvent,
)

from .contracts import (
    ExecutionFeedback,
    IssueReceipt,
    IssueRequest,
    PayloadEstimate,
    PayloadRegistration,
    PayloadStep,
)


@dataclass
class _Instance:
    busy_until: float = 0.0
    next_issue: float = 0.0


class AnalyticalExecutionBackend:
    """Own EU availability, payload timing and physical-done events."""

    name = "analytical_execution"

    @staticmethod
    def _unit_matches(requested: str, physical: str) -> bool:
        requested = requested.lower()
        physical = physical.lower()
        aliases = {
            "dma": {"dma", "gdma", "ldma", "de", "copy"},
            "tensor": {"mxu", "me", "tensor", "matrix"},
            "vector": {"aru", "ve", "vector", "vu"},
            "scalar": {"scalar", "cpu"},
        }
        return requested == physical or physical in aliases.get(requested, set())

    def __init__(
        self,
        artifact: BackendArtifact,
        machine: MachineConfig,
        timing_model: TimingModel | None = None,
    ) -> None:
        self._artifact = artifact
        self._machine = machine
        self._timing = timing_model or AnalyticalTimingModel()
        self._plans = {
            f"{artifact.artifact_id}::{instruction.tisa_id}": self._build_plan(
                instruction.tisa_id
            )
            for instruction in artifact.program.instructions
        }
        self._instances = {
            unit.name: [_Instance() for _index in range(unit.count)]
            for unit in machine.execution_units
        }
        self._feedback: list[tuple[float, int, ExecutionFeedback]] = []
        self._serial = 0
        self._issued: set[str] = set()
        self._timings: dict[str, TaskTiming] = {}
        self._events: list[TraceEvent] = []
        root_memories = {
            level.name for level in machine.memory_levels if level.parent is None
        }
        traffic = Counter()
        for task in artifact.execution_graph.tasks:
            for direction, regions in (("read", task.reads), ("write", task.writes)):
                for region in regions:
                    if region.memory not in root_memories:
                        continue
                    size = int(region.valid_bytes or region.size_bytes)
                    traffic[f"{region.memory}.{direction}_bytes"] += size
                    traffic[f"{region.memory}.total_bytes"] += size
                    traffic[f"offchip_{direction}_bytes"] += size
                    traffic["offchip_total_bytes"] += size
        self._metrics = {
            "memory_traffic_bytes": dict(traffic),
            "offchip_read_bytes": traffic["offchip_read_bytes"],
            "offchip_write_bytes": traffic["offchip_write_bytes"],
            "offchip_total_bytes": traffic["offchip_total_bytes"],
            "payload_internal_task_count": len(artifact.execution_graph.tasks),
        }

    def _build_plan(self, tisa_id: str) -> PayloadEstimate:
        handle = f"{self._artifact.artifact_id}::{tisa_id}"
        task_ids = self._artifact.payloads[tisa_id]
        tasks = {
            task_id: self._artifact.execution_graph.task(task_id)
            for task_id in task_ids
        }
        resources = {task.resource for task in tasks.values()}
        if len(resources) != 1:
            raise ValueError(f"payload '{handle}' must use exactly one resource")
        resource = next(iter(resources))
        remaining = {
            task_id: {
                predecessor
                for predecessor in task.predecessors
                if predecessor in tasks
            }
            for task_id, task in tasks.items()
        }
        ordered: list[str] = []
        while len(ordered) < len(tasks):
            ready = [
                task_id
                for task_id, predecessors in remaining.items()
                if task_id not in ordered and predecessors.issubset(ordered)
            ]
            if not ready:
                raise ValueError(f"payload '{handle}' contains an internal cycle")
            ordered.append(
                min(ready, key=lambda task_id: (tasks[task_id].program_order, task_id))
            )
        offset = 0.0
        steps: list[PayloadStep] = []
        intervals: list[float] = []
        for task_id in ordered:
            task = tasks[task_id]
            timing = self._timing.timing(task, self._machine)
            steps.append(
                PayloadStep(
                    task_id=task_id,
                    primitive=task.primitive,
                    start_offset=offset,
                    finish_offset=offset + timing.duration_cycles,
                )
            )
            offset += timing.duration_cycles
            intervals.append(timing.initiation_interval_cycles)
        return PayloadEstimate(
            handle=handle,
            resource=resource,
            duration_cycles=offset,
            initiation_interval_cycles=max(
                self._machine.unit(resource).initiation_interval_cycles,
                *intervals,
            ),
            steps=tuple(steps),
        )

    def registrations(self) -> tuple[PayloadRegistration, ...]:
        return tuple(
            PayloadRegistration(
                handle=handle,
                tisa_id=handle.rsplit("::", 1)[1],
                resource=plan.resource,
                task_count=len(plan.steps),
            )
            for handle, plan in self._plans.items()
        )

    def estimate(self, payload_handle: str) -> PayloadEstimate:
        try:
            return self._plans[payload_handle]
        except KeyError as exc:
            raise KeyError(f"unregistered payload '{payload_handle}'") from exc

    def _available_instance(
        self, descriptor: BoundTISADescriptor, cycle: float
    ) -> int | None:
        plan = self.estimate(descriptor.payload_handle)
        if not self._unit_matches(descriptor.instruction.unit_map.unit, plan.resource):
            return None
        return next(
            (
                index
                for index, state in enumerate(self._instances[plan.resource])
                if state.busy_until <= cycle and state.next_issue <= cycle
            ),
            None,
        )

    def can_accept(
        self, descriptor: BoundTISADescriptor, cycle: float
    ) -> tuple[bool, str | None]:
        if descriptor.payload_handle not in self._plans:
            return False, "payload_unregistered"
        if descriptor.descriptor_id in self._issued:
            return False, "duplicate_issue"
        plan = self._plans[descriptor.payload_handle]
        if not self._unit_matches(descriptor.instruction.unit_map.unit, plan.resource):
            return False, "unit_map_mismatch"
        if self._available_instance(descriptor, cycle) is None:
            return False, "execution_resource_busy"
        return True, None

    def next_accept_cycle(self, descriptor: BoundTISADescriptor) -> float | None:
        if descriptor.payload_handle not in self._plans:
            return None
        plan = self._plans[descriptor.payload_handle]
        if plan.resource not in self._instances:
            return None
        return min(
            max(state.busy_until, state.next_issue)
            for state in self._instances[plan.resource]
        )

    def issue(self, request: IssueRequest) -> IssueReceipt:
        request_issues = request.validate()
        if request_issues:
            raise ValueError("invalid execution issue: " + "; ".join(request_issues))
        descriptor = request.descriptor
        accepted, reason = self.can_accept(descriptor, request.cycle)
        if not accepted:
            return IssueReceipt(
                accepted=False,
                descriptor_id=descriptor.descriptor_id,
                payload_handle=descriptor.payload_handle,
                rejection_reason=reason,
            )
        plan = self._plans[descriptor.payload_handle]
        instance = self._available_instance(descriptor, request.cycle)
        assert instance is not None
        state = self._instances[plan.resource][instance]
        done = math.ceil(request.cycle + plan.duration_cycles)
        state.busy_until = done
        state.next_issue = request.cycle + plan.initiation_interval_cycles
        self._issued.add(descriptor.descriptor_id)
        for step in plan.steps:
            start = request.cycle + step.start_offset
            finish = request.cycle + step.finish_offset
            self._timings[step.task_id] = TaskTiming(
                task_id=step.task_id,
                resource=plan.resource,
                instance=instance,
                issue=start,
                start=start,
                finish=finish,
                dependency_ready=start,
                resource_ready=start,
            )
            details = {
                "primitive": step.primitive,
                "parent_descriptor_id": descriptor.descriptor_id,
                "parent_tisa_id": descriptor.instruction.tisa_id,
            }
            self._events.extend(
                (
                    TraceEvent(start, "ISSUE", step.task_id, plan.resource, instance, details),
                    TraceEvent(start, "START", step.task_id, plan.resource, instance, details),
                    TraceEvent(finish, "COMPLETE", step.task_id, plan.resource, instance, details),
                )
            )
        for condition in descriptor.attributes.get("feedback_conditions", ()):
            task_id = str(condition).removeprefix("payload_ready:")
            step = next((item for item in plan.steps if item.task_id == task_id), None)
            if step is None:
                raise ValueError(
                    f"partial-ready task '{task_id}' is outside the source payload for "
                    f"descriptor '{descriptor.descriptor_id}'"
                )
            self._push_feedback(
                ExecutionFeedback(
                    descriptor_id=descriptor.descriptor_id,
                    token=CompletionToken(
                        descriptor.invocation_id,
                        descriptor.instruction.tisa_id,
                        kind=str(condition),
                    ),
                    kind="partial_ready",
                    cycle=request.cycle + step.finish_offset,
                    resource=plan.resource,
                    instance=instance,
                    condition=str(condition),
                )
            )
        self._push_feedback(
            ExecutionFeedback(
                descriptor_id=descriptor.descriptor_id,
                token=descriptor.completion_token,
                kind="execution_done",
                cycle=done,
                resource=plan.resource,
                instance=instance,
            )
        )
        receipt = IssueReceipt(
            accepted=True,
            descriptor_id=descriptor.descriptor_id,
            payload_handle=descriptor.payload_handle,
            resource=plan.resource,
            instance=instance,
            issue_cycle=request.cycle,
            expected_done_cycle=done,
            attributes={"payload_task_count": len(plan.steps)},
        )
        issues = receipt.validate()
        if issues:
            raise ValueError("invalid issue receipt: " + "; ".join(issues))
        return receipt

    def _push_feedback(self, feedback: ExecutionFeedback) -> None:
        issues = feedback.validate()
        if issues:
            raise ValueError("invalid execution feedback: " + "; ".join(issues))
        self._serial += 1
        heapq.heappush(self._feedback, (feedback.cycle, self._serial, feedback))

    def advance(self, cycle: float) -> tuple[ExecutionFeedback, ...]:
        result: list[ExecutionFeedback] = []
        while self._feedback and self._feedback[0][0] <= cycle:
            _ready, _serial, feedback = heapq.heappop(self._feedback)
            result.append(feedback)
        return tuple(result)

    def next_feedback_cycle(self) -> float | None:
        return self._feedback[0][0] if self._feedback else None

    def task_timings(self) -> tuple[TaskTiming, ...]:
        return tuple(self._timings.values())

    def trace_events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def metrics(self):
        return dict(self._metrics)


__all__ = ["AnalyticalExecutionBackend"]
