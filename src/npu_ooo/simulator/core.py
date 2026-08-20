from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
from typing import Any, Mapping, Protocol

from npu_ooo.arch import ExecutionUnitConfig, MachineConfig
from npu_ooo.ir import ExecutionGraph, ExecutionTask
from .address import add_address_dependencies


class TimingModel(Protocol):
    """Backend contract for task latency and initiation interval."""

    name: str

    def timing(self, task: ExecutionTask, machine: MachineConfig) -> "TaskTimingSpec":
        ...


@dataclass(frozen=True)
class TaskTimingSpec:
    duration_cycles: float
    initiation_interval_cycles: float

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.duration_cycles <= 0:
            issues.append("task timing duration must be positive")
        if self.initiation_interval_cycles <= 0:
            issues.append("task timing initiation interval must be positive")
        return tuple(issues)


@dataclass(frozen=True)
class AnalyticalTimingModel:
    """Use lowered task timing, falling back to MachineConfig unit defaults."""

    name: str = "analytical"

    def timing(self, task: ExecutionTask, machine: MachineConfig) -> TaskTimingSpec:
        unit = machine.unit(task.resource)
        spec = TaskTimingSpec(
            duration_cycles=float(
                task.duration_cycles if task.duration_cycles is not None else unit.latency_cycles
            ),
            initiation_interval_cycles=float(
                task.initiation_interval_cycles
                if task.initiation_interval_cycles is not None
                else unit.initiation_interval_cycles
            ),
        )
        issues = spec.validate()
        if issues:
            raise ValueError(f"task '{task.task_id}' has invalid timing: {'; '.join(issues)}")
        return spec


@dataclass(frozen=True)
class SimulatorConfig:
    """Runtime capacities applied equally to every scheduling policy."""

    instruction_queue_depth: int | None = None
    rob_entries: int | None = None
    max_inflight_tiles: int | None = None
    dependency_window: int | None = None
    ready_queue_depth: int | None = None
    address_scoreboard: bool = False

    def resolved(self, machine: MachineConfig) -> "SimulatorConfig":
        scheduler = machine.scheduler
        result = SimulatorConfig(
            instruction_queue_depth=(
                self.instruction_queue_depth
                if self.instruction_queue_depth is not None
                else scheduler.instruction_queue_depth
            ),
            rob_entries=self.rob_entries if self.rob_entries is not None else scheduler.rob_entries,
            max_inflight_tiles=(
                self.max_inflight_tiles
                if self.max_inflight_tiles is not None
                else scheduler.max_inflight_tiles
            ),
            dependency_window=(
                self.dependency_window
                if self.dependency_window is not None
                else scheduler.dependency_window
            ),
            ready_queue_depth=(
                self.ready_queue_depth
                if self.ready_queue_depth is not None
                else scheduler.instruction_queue_depth
            ),
            address_scoreboard=self.address_scoreboard,
        )
        issues = result.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return result

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for name, value in (
            ("instruction_queue_depth", self.instruction_queue_depth),
            ("rob_entries", self.rob_entries),
            ("max_inflight_tiles", self.max_inflight_tiles),
            ("dependency_window", self.dependency_window),
            ("ready_queue_depth", self.ready_queue_depth),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                issues.append(f"simulator {name} must be positive when specified")
        return tuple(issues)

    def to_dict(self) -> dict[str, int | None]:
        return {
            "instruction_queue_depth": self.instruction_queue_depth,
            "rob_entries": self.rob_entries,
            "max_inflight_tiles": self.max_inflight_tiles,
            "dependency_window": self.dependency_window,
            "ready_queue_depth": self.ready_queue_depth,
            "address_scoreboard": self.address_scoreboard,
        }


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

    @property
    def queue_wait(self) -> float:
        return max(0.0, self.start - self.issue)

    @property
    def ready_wait(self) -> float:
        return max(0.0, self.issue - self.dependency_ready)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource": self.resource,
            "instance": self.instance,
            "issue": self.issue,
            "start": self.start,
            "finish": self.finish,
            "duration": self.duration,
            "queue_wait": self.queue_wait,
            "ready_wait": self.ready_wait,
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
class SimulationResult:
    backend: str
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
            "backend": self.backend,
            "policy": self.policy,
            "graph_id": self.graph_id,
            "total_cycles": self.total_cycles,
            "timings": [timing.to_dict() for timing in self.timings],
            "events": [event.to_dict() for event in self.events],
            "metrics": dict(self.metrics),
        }

    def perfetto_trace(self) -> dict[str, Any]:
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
        return {"traceEvents": trace_events, "displayTimeUnit": "cycle"}


@dataclass
class _ResourceState:
    unit: ExecutionUnitConfig
    instance: int
    next_issue: float = 0.0
    busy_until: float = 0.0
    in_flight: int = 0
    issued_timestamp: float | None = None
    issued_this_timestamp: int = 0

    def _reset_issue_counter(self, now: float) -> None:
        if self.issued_timestamp != now:
            self.issued_timestamp = now
            self.issued_this_timestamp = 0

    def can_issue(self, now: float) -> bool:
        self._reset_issue_counter(now)
        return (
            self.in_flight < self.unit.queue_depth
            and now + 1e-9 >= self.next_issue
            and self.issued_this_timestamp < self.unit.issue_width
        )

    def estimate(self, now: float, duration: float) -> tuple[float, float]:
        resource_ready = max(now, self.next_issue)
        if self.unit.pipeline_depth <= 1:
            start = max(now, self.busy_until)
        else:
            start = now
        return start, resource_ready

    def reserve(self, now: float, duration: float) -> tuple[float, float]:
        self._reset_issue_counter(now)
        start, resource_ready = self.estimate(now, duration)
        finish = start + duration
        self.issued_this_timestamp += 1
        if self.issued_this_timestamp >= self.unit.issue_width:
            self.next_issue = now + self.unit.initiation_interval_cycles
        else:
            self.next_issue = now
        if self.unit.pipeline_depth <= 1:
            self.busy_until = finish
        self.in_flight += 1
        return start, resource_ready

    def complete(self) -> None:
        self.in_flight -= 1
        if self.in_flight < 0:
            raise RuntimeError(f"resource '{self.unit.name}[{self.instance}]' completed without an in-flight task")


class _EventKind(str, Enum):
    COMPLETE = "COMPLETE"


def _critical_path_lengths(graph: ExecutionGraph, machine: MachineConfig, timing_model: TimingModel) -> dict[str, float]:
    tasks = {task.task_id: task for task in graph.tasks}
    lengths: dict[str, float] = {}
    for task_id in reversed(graph.topological_order()):
        task = tasks[task_id]
        duration = timing_model.timing(task, machine).duration_cycles
        successors = [candidate for candidate in graph.tasks if task_id in candidate.predecessors]
        lengths[task_id] = duration + max((lengths[successor.task_id] for successor in successors), default=0.0)
    return lengths


def _validate_policy(policy: str) -> str:
    supported = {"sequential", "static_pipeline", "dynamic_ready_queue"}
    if policy not in supported:
        raise ValueError(f"unsupported scheduler policy '{policy}'")
    return policy


def simulate_execution_graph(
    graph: ExecutionGraph,
    machine: MachineConfig,
    policy: str = "static_pipeline",
    *,
    timing_model: TimingModel | None = None,
    config: SimulatorConfig | None = None,
) -> SimulationResult:
    """Run a deterministic event simulation over one shared ExecutionGraph."""

    policy = _validate_policy(policy)
    graph_issues = graph.validate()
    machine_issues = machine.validate()
    if graph_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *machine_issues)))
    timing_model = timing_model or AnalyticalTimingModel()
    config = (config or SimulatorConfig()).resolved(machine)
    address_dependencies = ()
    if config.address_scoreboard:
        graph, address_dependencies = add_address_dependencies(graph)
    tasks = {task.task_id: task for task in graph.tasks}
    graph_order = graph.topological_order()
    order_index = {task_id: index for index, task_id in enumerate(graph_order)}
    program_order = {task.task_id: task.program_order for task in graph.tasks}
    critical_path = _critical_path_lengths(graph, machine, timing_model)
    successors: dict[str, list[str]] = {task_id: [] for task_id in tasks}
    remaining_predecessors = {task.task_id: len(task.predecessors) for task in graph.tasks}
    dependency_ready = {task.task_id: 0.0 for task in graph.tasks}
    for task in graph.tasks:
        for predecessor in task.predecessors:
            successors[predecessor].append(task.task_id)

    resources: dict[str, list[_ResourceState]] = {
        unit.name: [_ResourceState(unit, index) for index in range(unit.count)]
        for unit in machine.execution_units
    }
    ready: set[str] = {task_id for task_id, count in remaining_predecessors.items() if count == 0}
    issued: set[str] = set()
    completed: set[str] = set()
    finish_by_task: dict[str, float] = {}
    timings: dict[str, TaskTiming] = {}
    events: list[TraceEvent] = []
    event_queue: list[tuple[float, int, _EventKind, str, str, int]] = []
    event_serial = 0
    now = 0.0
    rob_occupancy = 0
    inflight_tiles: set[str] = set()
    tile_task_count: dict[str, int] = {}
    tile_completed_count: dict[str, int] = {}
    for task in graph.tasks:
        tile_task_count[task.tile_id] = tile_task_count.get(task.tile_id, 0) + 1
        tile_completed_count[task.tile_id] = 0

    metrics: dict[str, Any] = {
        "backend": timing_model.name,
        "policy": policy,
        "simulator_config": config.to_dict(),
        "address_dependency_count": len(address_dependencies),
        "resource_busy_cycles": {},
        "ready_set_peak": len(ready),
        "visible_ready_peak": 0,
        "rob_peak": 0,
        "inflight_tile_peak": 0,
        "issued_task_count": 0,
        "completed_task_count": 0,
        "queue_wait_cycles": 0.0,
        "ready_wait_cycles": 0.0,
        "rob_block_events": 0,
        "window_block_events": 0,
        "resource_block_events": 0,
        "tile_window_block_events": 0,
    }

    def push_complete(task: ExecutionTask, resource: _ResourceState, finish: float) -> None:
        nonlocal event_serial
        event_serial += 1
        heapq.heappush(
            event_queue,
            (finish, event_serial, _EventKind.COMPLETE, task.task_id, task.resource, resource.instance),
        )

    def visible_ready_tasks() -> list[ExecutionTask]:
        ordered = sorted(
            (tasks[task_id] for task_id in ready),
            key=lambda task: (program_order[task.task_id], order_index[task.task_id], task.task_id),
        )
        queue_limit = config.ready_queue_depth or len(ordered)
        window_budget = max(0, (config.dependency_window or len(ordered)) - rob_occupancy)
        return ordered[: min(queue_limit, window_budget)]

    def candidate_resources(task: ExecutionTask) -> list[tuple[_ResourceState, float, float]]:
        candidates: list[tuple[_ResourceState, float, float]] = []
        for resource in resources.get(task.resource, ()):
            if resource.can_issue(now):
                timing = timing_model.timing(task, machine)
                start, resource_ready = resource.estimate(now, timing.duration_cycles)
                candidates.append((resource, start, resource_ready))
        return candidates

    while len(completed) < len(tasks):
        if event_queue and event_queue[0][0] < now - 1e-9:
            raise RuntimeError("event queue moved backwards in time")

        while event_queue and event_queue[0][0] <= now + 1e-9:
            finish, _serial, _kind, task_id, resource_name, resource_instance = heapq.heappop(event_queue)
            resource = resources[resource_name][resource_instance]
            resource.complete()
            completed.add(task_id)
            finish_by_task[task_id] = finish
            rob_occupancy -= 1
            task = tasks[task_id]
            tile_completed_count[task.tile_id] += 1
            if tile_completed_count[task.tile_id] == tile_task_count[task.tile_id]:
                inflight_tiles.discard(task.tile_id)
            details = {"primitive": task.primitive, "operator_id": task.operator_id, "tile_id": task.tile_id}
            events.append(TraceEvent(finish, "COMPLETE", task_id, resource_name, resource_instance, details))
            for successor in successors[task_id]:
                remaining_predecessors[successor] -= 1
                dependency_ready[successor] = max(dependency_ready[successor], finish)
                events.append(
                    TraceEvent(
                        finish,
                        "WAKE_UP",
                        successor,
                        tasks[successor].resource,
                        resource_instance,
                        {"predecessor": task_id},
                    )
                )
                if remaining_predecessors[successor] == 0:
                    ready.add(successor)
            metrics["completed_task_count"] += 1

        metrics["ready_set_peak"] = max(metrics["ready_set_peak"], len(ready))
        metrics["rob_peak"] = max(metrics["rob_peak"], rob_occupancy)
        metrics["inflight_tile_peak"] = max(metrics["inflight_tile_peak"], len(inflight_tiles))

        issued_at_now = 0
        while True:
            if policy == "sequential" and rob_occupancy:
                break
            if rob_occupancy >= (config.rob_entries or math.inf):
                metrics["rob_block_events"] += 1
                break
            visible = visible_ready_tasks()
            metrics["visible_ready_peak"] = max(metrics["visible_ready_peak"], len(visible))
            if not visible:
                if ready and rob_occupancy >= (config.dependency_window or math.inf):
                    metrics["window_block_events"] += 1
                break
            candidates: list[tuple[ExecutionTask, _ResourceState, float, float]] = []
            for task in visible:
                if task.tile_id not in inflight_tiles and len(inflight_tiles) >= (config.max_inflight_tiles or math.inf):
                    continue
                for resource, start, resource_ready in candidate_resources(task):
                    candidates.append((task, resource, start, resource_ready))
            if not candidates:
                if ready:
                    metrics["resource_block_events"] += 1
                blocked_by_tiles = all(
                    task.tile_id not in inflight_tiles
                    and len(inflight_tiles) >= (config.max_inflight_tiles or math.inf)
                    for task in visible
                )
                if blocked_by_tiles:
                    metrics["tile_window_block_events"] += 1
                break
            if policy == "dynamic_ready_queue":
                selected = min(
                    candidates,
                    key=lambda item: (
                        item[2],
                        -critical_path[item[0].task_id],
                        item[0].program_order,
                        item[0].task_id,
                        item[1].instance,
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda item: (item[0].program_order, item[0].task_id, item[1].instance),
                )
            task, resource, _estimated_start, _estimated_resource_ready = selected
            spec = timing_model.timing(task, machine)
            start, resource_ready = resource.reserve(now, spec.duration_cycles)
            finish = start + spec.duration_cycles
            ready.remove(task.task_id)
            issued.add(task.task_id)
            rob_occupancy += 1
            inflight_tiles.add(task.tile_id)
            metrics["rob_peak"] = max(metrics["rob_peak"], rob_occupancy)
            metrics["inflight_tile_peak"] = max(metrics["inflight_tile_peak"], len(inflight_tiles))
            timings[task.task_id] = TaskTiming(
                task_id=task.task_id,
                resource=task.resource,
                instance=resource.instance,
                issue=now,
                start=start,
                finish=finish,
                dependency_ready=dependency_ready[task.task_id],
                resource_ready=resource_ready,
            )
            details = {
                "primitive": task.primitive,
                "operator_id": task.operator_id,
                "tile_id": task.tile_id,
                "backend": timing_model.name,
            }
            events.extend(
                (
                    TraceEvent(now, "ISSUE", task.task_id, task.resource, resource.instance, details),
                    TraceEvent(start, "START", task.task_id, task.resource, resource.instance, details),
                )
            )
            push_complete(task, resource, finish)
            metrics["issued_task_count"] += 1
            metrics["queue_wait_cycles"] += max(0.0, start - now)
            metrics["ready_wait_cycles"] += max(0.0, now - dependency_ready[task.task_id])
            metrics["resource_busy_cycles"][task.resource] = metrics["resource_busy_cycles"].get(task.resource, 0.0) + spec.duration_cycles
            issued_at_now += 1
            if issued_at_now > sum(resource.unit.issue_width for resources_for_unit in resources.values() for resource in resources_for_unit):
                break

        if len(completed) >= len(tasks):
            break
        next_times: list[float] = []
        if event_queue:
            next_times.append(event_queue[0][0])
        if ready:
            for task in visible_ready_tasks():
                for resource in resources.get(task.resource, ()):
                    if resource.in_flight < resource.unit.queue_depth and resource.next_issue > now + 1e-9:
                        next_times.append(resource.next_issue)
        if not next_times:
            raise RuntimeError(
                f"execution graph deadlocked at cycle {now}; ready={sorted(ready)}, rob={rob_occupancy}, inflight_tiles={sorted(inflight_tiles)}"
            )
        next_now = min(time for time in next_times if time > now + 1e-9) if any(time > now + 1e-9 for time in next_times) else None
        if next_now is None:
            raise RuntimeError("simulator made no time progress")
        now = next_now

    events.sort(key=lambda event: (event.timestamp, {"ISSUE": 0, "START": 1, "COMPLETE": 2, "WAKE_UP": 3}[event.event], event.task_id, event.instance))
    metrics["calibration_status"] = machine.attributes.get("calibration_status", "unspecified")
    metrics["task_count"] = len(tasks)
    return SimulationResult(
        backend=timing_model.name,
        policy=policy,
        graph_id=graph.graph_id,
        total_cycles=max((timing.finish for timing in timings.values()), default=0.0),
        timings=tuple(timings[task.task_id] for task in graph.tasks),
        events=tuple(events),
        metrics=metrics,
    )
