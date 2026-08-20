from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import ExecutionGraph, ExecutionTask


class SchedulerPolicy(str, Enum):
    SEQUENTIAL = "sequential"
    STATIC_PIPELINE = "static_pipeline"
    DYNAMIC_READY_QUEUE = "dynamic_ready_queue"


@dataclass(frozen=True)
class TaskTiming:
    task_id: str
    resource: str
    instance: int
    issue: float
    start: float
    finish: float
    dependency_ready: float
    resource_ready: float

    @property
    def duration(self) -> float:
        return self.finish - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource": self.resource,
            "instance": self.instance,
            "issue": self.issue,
            "start": self.start,
            "finish": self.finish,
            "duration": self.duration,
            "dependency_ready": self.dependency_ready,
            "resource_ready": self.resource_ready,
        }


@dataclass(frozen=True)
class TraceEvent:
    timestamp: float
    event: str
    task_id: str
    resource: str
    instance: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "task_id": self.task_id,
            "resource": self.resource,
            "instance": self.instance,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ScheduleResult:
    policy: str
    graph_id: str
    total_cycles: float
    timings: tuple[TaskTiming, ...]
    events: tuple[TraceEvent, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def timing(self, task_id: str) -> TaskTiming:
        for timing in self.timings:
            if timing.task_id == task_id:
                return timing
        raise KeyError(task_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "graph_id": self.graph_id,
            "total_cycles": self.total_cycles,
            "timings": [timing.to_dict() for timing in self.timings],
            "events": [event.to_dict() for event in self.events],
            "metrics": dict(self.metrics),
        }

    def perfetto_trace(self) -> dict[str, Any]:
        """Return a Chrome/Perfetto trace-event JSON object."""

        trace_events: list[dict[str, Any]] = []
        for event in self.events:
            if event.event not in {"START", "COMPLETE"}:
                continue
            phase = "B" if event.event == "START" else "E"
            trace_events.append(
                {
                    "name": event.task_id,
                    "cat": event.resource,
                    "ph": phase,
                    "ts": event.timestamp,
                    "pid": 1,
                    "tid": f"{event.resource}[{event.instance}]",
                    "args": dict(event.details),
                }
            )
        return {"traceEvents": trace_events, "displayTimeUnit": "ns"}


@dataclass
class _ResourceInstance:
    unit_name: str
    instance: int
    pipeline_depth: int
    queue_depth: int
    initiation_interval: float
    issue_cursor: float = 0.0
    available_cursor: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)

    def earliest(self, dependency_ready: float, duration: float, serial_cursor: float | None) -> tuple[float, float]:
        resource_ready = self.issue_cursor if self.pipeline_depth > 1 else self.available_cursor
        candidate = max(dependency_ready, resource_ready)
        if serial_cursor is not None:
            candidate = max(candidate, serial_cursor)
        if self.pipeline_depth > 1:
            # A pipelined unit may overlap executions, but its configured queue
            # still bounds the number of operations whose completion is pending.
            while True:
                active = [finish for issue, finish in self.history if issue <= candidate < finish]
                if len(active) < self.queue_depth:
                    break
                candidate = min(active)
        return candidate, resource_ready

    def reserve(self, start: float, finish: float) -> None:
        self.issue_cursor = start + self.initiation_interval
        if self.pipeline_depth <= 1:
            self.available_cursor = finish
        self.history.append((start, finish))


def _critical_path_lengths(graph: ExecutionGraph, machine: MachineConfig) -> dict[str, float]:
    tasks = {task.task_id: task for task in graph.tasks}
    lengths: dict[str, float] = {}
    for task_id in reversed(graph.topological_order()):
        task = tasks[task_id]
        unit = machine.unit(task.resource)
        duration = float(task.duration_cycles if task.duration_cycles is not None else unit.latency_cycles)
        successors = [candidate for candidate in graph.tasks if task_id in candidate.predecessors]
        lengths[task_id] = duration + max((lengths[successor.task_id] for successor in successors), default=0.0)
    return lengths


def schedule_execution_graph(
    graph: ExecutionGraph,
    machine: MachineConfig,
    policy: SchedulerPolicy | str = SchedulerPolicy.STATIC_PIPELINE,
) -> ScheduleResult:
    """Schedule an ExecutionGraph with deterministic resource and dependency semantics."""

    normalized_policy = policy.value if isinstance(policy, SchedulerPolicy) else str(policy)
    try:
        selected_policy = SchedulerPolicy(normalized_policy)
    except ValueError as exc:
        raise ValueError(f"unsupported scheduler policy '{normalized_policy}'") from exc
    graph_issues = graph.validate()
    machine_issues = machine.validate()
    if graph_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *machine_issues)))
    tasks = {task.task_id: task for task in graph.tasks}
    order_index = {task_id: index for index, task_id in enumerate(graph.topological_order())}
    critical_path = _critical_path_lengths(graph, machine)
    resource_instances: dict[str, list[_ResourceInstance]] = {}
    for unit in machine.execution_units:
        resource_instances[unit.name] = [
            _ResourceInstance(
                unit_name=unit.name,
                instance=index,
                pipeline_depth=unit.pipeline_depth,
                queue_depth=unit.queue_depth,
                initiation_interval=unit.initiation_interval_cycles,
            )
            for index in range(unit.count)
        ]

    finish_by_task: dict[str, float] = {}
    timing_by_task: dict[str, TaskTiming] = {}
    unscheduled = set(tasks)
    serial_cursor: float | None = 0.0 if selected_policy == SchedulerPolicy.SEQUENTIAL else None
    dependency_wait = 0.0
    resource_wait = 0.0
    while unscheduled:
        ready = [
            tasks[task_id]
            for task_id in unscheduled
            if all(predecessor in finish_by_task for predecessor in tasks[task_id].predecessors)
        ]
        if not ready:
            raise ValueError("execution graph became unschedulable; dependency cycle or missing predecessor")

        candidates: list[tuple[float, int, ExecutionTask, _ResourceInstance, float, float]] = []
        for task in ready:
            instances = resource_instances.get(task.resource)
            if not instances:
                raise ValueError(f"task '{task.task_id}' references unknown resource '{task.resource}'")
            unit = machine.unit(task.resource)
            duration = float(task.duration_cycles if task.duration_cycles is not None else unit.latency_cycles)
            dependency_ready = max((finish_by_task[p] for p in task.predecessors), default=0.0)
            for instance in instances:
                start, resource_ready = instance.earliest(dependency_ready, duration, serial_cursor)
                if selected_policy == SchedulerPolicy.DYNAMIC_READY_QUEUE:
                    key = (start, -critical_path[task.task_id], task.program_order, task.task_id)
                else:
                    key = (task.program_order, order_index[task.task_id], task.task_id)
                candidates.append((start, key[1] if isinstance(key[1], int) else 0, task, instance, dependency_ready, resource_ready))
        if selected_policy == SchedulerPolicy.DYNAMIC_READY_QUEUE:
            # Dynamic policy prefers the task that can run now, then the longest
            # remaining critical path; ties are stable by program order.
            selected = min(candidates, key=lambda item: (item[0], -critical_path[item[2].task_id], item[2].program_order, item[2].task_id, item[3].instance))
        else:
            selected = min(candidates, key=lambda item: (item[2].program_order, item[2].task_id, item[3].instance))
        start, _unused, task, instance, dependency_ready, resource_ready = selected
        unit = machine.unit(task.resource)
        duration = float(task.duration_cycles if task.duration_cycles is not None else unit.latency_cycles)
        finish = start + duration
        instance.reserve(start, finish)
        timing_by_task[task.task_id] = TaskTiming(
            task_id=task.task_id,
            resource=task.resource,
            instance=instance.instance,
            issue=start,
            start=start,
            finish=finish,
            dependency_ready=dependency_ready,
            resource_ready=resource_ready,
        )
        finish_by_task[task.task_id] = finish
        unscheduled.remove(task.task_id)
        dependency_wait += max(0.0, start - dependency_ready)
        resource_wait += max(0.0, start - max(dependency_ready, resource_ready))
        if serial_cursor is not None:
            serial_cursor = finish

    events: list[TraceEvent] = []
    event_order = {"ISSUE": 0, "START": 1, "COMPLETE": 2, "WAKE_UP": 3}
    for task in graph.tasks:
        timing = timing_by_task[task.task_id]
        details = {"primitive": task.primitive, "operator_id": task.operator_id, "tile_id": task.tile_id}
        events.extend(
            (
                TraceEvent(timing.issue, "ISSUE", task.task_id, task.resource, timing.instance, details),
                TraceEvent(timing.start, "START", task.task_id, task.resource, timing.instance, details),
                TraceEvent(timing.finish, "COMPLETE", task.task_id, task.resource, timing.instance, details),
            )
        )
        for successor in graph.tasks:
            if task.task_id in successor.predecessors:
                events.append(
                    TraceEvent(
                        timing.finish,
                        "WAKE_UP",
                        successor.task_id,
                        successor.resource,
                        timing.instance,
                        {"predecessor": task.task_id},
                    )
                )
    events.sort(key=lambda event: (event.timestamp, event_order[event.event], event.task_id, event.instance))
    total_cycles = max(finish_by_task.values(), default=0.0)
    busy_by_resource: dict[str, float] = {}
    for timing in timing_by_task.values():
        busy_by_resource[timing.resource] = busy_by_resource.get(timing.resource, 0.0) + timing.duration
    metrics = {
        "task_count": len(tasks),
        "resource_busy_cycles": busy_by_resource,
        "dependency_wait_cycles": dependency_wait,
        "resource_wait_cycles": resource_wait,
        "calibration_status": machine.attributes.get("calibration_status", "unspecified"),
    }
    return ScheduleResult(
        policy=selected_policy.value,
        graph_id=graph.graph_id,
        total_cycles=total_cycles,
        timings=tuple(timing_by_task[task.task_id] for task in graph.tasks),
        events=tuple(events),
        metrics=metrics,
    )
