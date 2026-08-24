from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

from npu_ooo.arch import ExecutionUnitConfig, MachineConfig
from npu_ooo.ir import ExecutionGraph, ExecutionTask
from .address import AddressConflict, AddressScoreboard


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
class TimingTableModel:
    """Override analytical task timing from a small canonical JSON table."""

    entries: Mapping[str, TaskTimingSpec]
    name: str = "timing_table"
    fallback: TimingModel = field(default_factory=AnalyticalTimingModel)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimingTableModel":
        if not isinstance(payload, Mapping):
            raise ValueError("timing table payload must be an object")
        raw_entries = payload.get("entries", payload)
        if not isinstance(raw_entries, Mapping):
            raise ValueError("timing table entries must be an object")
        entries: dict[str, TaskTimingSpec] = {}
        for key, value in raw_entries.items():
            if key in {"name", "fallback", "metadata"}:
                continue
            if not isinstance(key, str) or not key:
                raise ValueError("timing table keys must be non-empty strings")
            if not isinstance(value, Mapping):
                raise ValueError(f"timing table entry '{key}' must be an object")
            spec = TaskTimingSpec(
                duration_cycles=float(value["duration_cycles"]),
                initiation_interval_cycles=float(value["initiation_interval_cycles"]),
            )
            issues = spec.validate()
            if issues:
                raise ValueError(f"timing table entry '{key}': {'; '.join(issues)}")
            entries[key] = spec
        return cls(entries=entries, name=str(payload.get("name", "timing_table")))

    @classmethod
    def from_path(cls, path: str | Path) -> "TimingTableModel":
        timing_path = Path(path)
        try:
            payload = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load timing table '{timing_path}': {exc}") from exc
        return cls.from_dict(payload)

    def timing(self, task: ExecutionTask, machine: MachineConfig) -> TaskTimingSpec:
        keys = (
            str(task.attributes.get("timing_key", "")),
            task.task_id,
            f"{task.resource}:{task.primitive}",
            task.primitive,
            task.resource,
            "default",
        )
        for key in keys:
            if key and key in self.entries:
                return self.entries[key]
        return self.fallback.timing(task, machine)


@dataclass(frozen=True)
class StaticPipelineConfig:
    """Explicit compile-time reservations for a static pipeline.

    A reservation may be supplied directly per task, or derived from a task's
    numeric ``attributes[iteration_attribute]`` together with ``stage_id``.
    The latter is useful for hand-written dual/triple-stage golden cases while
    the former preserves exact compiler-emitted reservations.
    """

    stage_count: int = 2
    stage_offsets: tuple[float, ...] = ()
    initiation_interval_cycles: float = 1.0
    iteration_attribute: str = "iteration"
    task_issue_cycles: tuple[tuple[str, float], ...] = ()

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if isinstance(self.stage_count, bool) or not isinstance(self.stage_count, int) or self.stage_count <= 0:
            issues.append("static pipeline stage_count must be positive")
        if not self.iteration_attribute:
            issues.append("static pipeline iteration_attribute must not be empty")
        if (
            isinstance(self.initiation_interval_cycles, bool)
            or not isinstance(self.initiation_interval_cycles, (int, float))
            or self.initiation_interval_cycles <= 0
        ):
            issues.append("static pipeline initiation_interval_cycles must be positive")
        if self.stage_offsets and len(self.stage_offsets) != self.stage_count:
            issues.append("static pipeline stage_offsets must match stage_count")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            for value in self.stage_offsets
        ):
            issues.append("static pipeline stage_offsets must be non-negative numbers")
        task_ids = [task_id for task_id, _cycle in self.task_issue_cycles]
        if len(set(task_ids)) != len(task_ids):
            issues.append("static pipeline task_issue_cycles must contain unique task ids")
        if any(not task_id for task_id in task_ids):
            issues.append("static pipeline task_issue_cycles task ids must not be empty")
        if any(
            isinstance(cycle, bool) or not isinstance(cycle, (int, float)) or cycle < 0
            for _task_id, cycle in self.task_issue_cycles
        ):
            issues.append("static pipeline task issue cycles must be non-negative numbers")
        return tuple(issues)

    def issue_cycle(self, task: ExecutionTask) -> float | None:
        explicit = dict(self.task_issue_cycles)
        if task.task_id in explicit:
            return float(explicit[task.task_id])
        if not self.stage_offsets:
            return None
        iteration = task.attributes.get(self.iteration_attribute)
        if isinstance(iteration, bool) or not isinstance(iteration, (int, float)) or iteration < 0:
            return None
        if task.stage_id >= self.stage_count:
            return None
        return float(self.stage_offsets[task.stage_id]) + float(iteration) * float(self.initiation_interval_cycles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_count": self.stage_count,
            "stage_offsets": list(self.stage_offsets),
            "initiation_interval_cycles": self.initiation_interval_cycles,
            "iteration_attribute": self.iteration_attribute,
            "task_issue_cycles": {task_id: cycle for task_id, cycle in self.task_issue_cycles},
        }


@dataclass(frozen=True)
class SimulatorConfig:
    """Runtime capacities applied equally to every scheduling policy."""

    instruction_queue_depth: int | None = None
    rob_entries: int | None = None
    max_inflight_tiles: int | None = None
    dependency_window: int | None = None
    ready_queue_depth: int | None = None
    address_scoreboard: bool = False
    static_pipeline: StaticPipelineConfig | None = None
    dynamic_priority: str = "critical_path"

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
                else (
                    self.instruction_queue_depth
                    if self.instruction_queue_depth is not None
                    else scheduler.instruction_queue_depth
                )
            ),
            address_scoreboard=self.address_scoreboard,
            static_pipeline=self.static_pipeline,
            dynamic_priority=self.dynamic_priority,
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
        if self.static_pipeline is not None:
            issues.extend(self.static_pipeline.validate())
        if self.dynamic_priority not in {"critical_path", "oldest_first"}:
            issues.append(
                "simulator dynamic_priority must be 'critical_path' or 'oldest_first'"
            )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_queue_depth": self.instruction_queue_depth,
            "rob_entries": self.rob_entries,
            "max_inflight_tiles": self.max_inflight_tiles,
            "dependency_window": self.dependency_window,
            "ready_queue_depth": self.ready_queue_depth,
            "address_scoreboard": self.address_scoreboard,
            "static_pipeline": (
                self.static_pipeline.to_dict() if self.static_pipeline is not None else None
            ),
            "dynamic_priority": self.dynamic_priority,
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
    instruction_timings: tuple[TaskTiming, ...] = ()
    runtime_timings: tuple[TaskTiming, ...] = ()

    def timing(self, task_id: str) -> TaskTiming:
        for timing in self.timings:
            if timing.task_id == task_id:
                return timing
        raise KeyError(task_id)

    def instruction_timing(self, tisa_id: str) -> TaskTiming:
        for timing in self.instruction_timings:
            if timing.task_id == tisa_id:
                return timing
        raise KeyError(tisa_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "policy": self.policy,
            "graph_id": self.graph_id,
            "total_cycles": self.total_cycles,
            "timings": [timing.to_dict() for timing in self.timings],
            "instruction_timings": [
                timing.to_dict() for timing in self.instruction_timings
            ],
            "runtime_timings": [timing.to_dict() for timing in self.runtime_timings],
            "events": [event.to_dict() for event in self.events],
            "metrics": dict(self.metrics),
        }

    def perfetto_trace(self) -> dict[str, Any]:
        trace_events: list[dict[str, Any]] = []
        for event in self.events:
            if event.event in {"START", "COMPLETE"}:
                phase = "B" if event.event == "START" else "E"
                pid = 2 if self.instruction_timings else 1
            elif event.event in {"TISA_ISSUE", "TISA_COMPLETE"}:
                phase = "B" if event.event == "TISA_ISSUE" else "E"
                pid = 1
            elif event.event in {"RUNTIME_SUBMIT_START", "RUNTIME_SUBMIT_COMPLETE"}:
                phase = "B" if event.event == "RUNTIME_SUBMIT_START" else "E"
                pid = 0
            else:
                continue
            trace_events.append(
                {
                    "name": event.task_id,
                    "cat": event.resource,
                    "ph": phase,
                    "ts": event.timestamp,
                    "pid": pid,
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
    if config.static_pipeline is not None and policy != "static_pipeline":
        raise ValueError("static_pipeline configuration is only valid with the static_pipeline policy")
    static_pipeline = config.static_pipeline if policy == "static_pipeline" else None
    if static_pipeline is not None and static_pipeline.stage_offsets:
        invalid_stages = sorted(
            {task.stage_id for task in graph.tasks if task.stage_id >= static_pipeline.stage_count}
        )
        if invalid_stages:
            raise ValueError(
                "static pipeline stage_count does not cover task stage ids: "
                + ", ".join(str(stage) for stage in invalid_stages)
            )
    address_scoreboard = AddressScoreboard() if config.address_scoreboard else None
    observed_address_hazards: dict[tuple[str, str, str, str, str], AddressConflict] = {}
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
    stall_started: dict[tuple[str, str], float] = {}
    stall_cycles_by_reason: dict[str, float] = {}
    tile_task_count: dict[str, int] = {}
    tile_completed_count: dict[str, int] = {}
    for task in graph.tasks:
        tile_task_count[task.tile_id] = tile_task_count.get(task.tile_id, 0) + 1
        tile_completed_count[task.tile_id] = 0

    metrics: dict[str, Any] = {
        "backend": timing_model.name,
        "policy": policy,
        "simulator_config": config.to_dict(),
        "address_dependency_count": 0,
        "address_hazard_count": 0,
        "address_hazards": [],
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
        "address_scoreboard_block_events": 0,
        "static_reservation_count": 0,
        "static_block_events": 0,
        "stall_cycles_by_reason": stall_cycles_by_reason,
        "stall_by_reason": stall_cycles_by_reason,
        "pipeline_drain_cycles": 0.0,
        "queue_occupancy_timeline": [],
    }

    def record_occupancy(timestamp: float, event: str) -> None:
        metrics["queue_occupancy_timeline"].append(
            {
                "timestamp": timestamp,
                "event": event,
                "rob": rob_occupancy,
                "ready": len(ready),
                "inflight_tiles": len(inflight_tiles),
                "resources": {
                    name: sum(resource.in_flight for resource in states)
                    for name, states in resources.items()
                },
            }
        )

    record_occupancy(0.0, "INIT")

    static_reservations = {
        task.task_id: static_pipeline.issue_cycle(task)
        for task in graph.tasks
        if static_pipeline is not None and static_pipeline.issue_cycle(task) is not None
    }
    metrics["static_reservation_count"] = len(static_reservations)

    def begin_stall(task: ExecutionTask, reason: str, details: Mapping[str, Any] | None = None) -> None:
        key = (task.task_id, reason)
        if key in stall_started:
            return
        stall_started[key] = now
        events.append(
            TraceEvent(
                now,
                "STALL_BEGIN",
                task.task_id,
                task.resource,
                details={"reason": reason, **dict(details or {})},
            )
        )

    def end_stalls(task: ExecutionTask, timestamp: float) -> None:
        for key in tuple(stall_started):
            task_id, reason = key
            if task_id != task.task_id:
                continue
            started = stall_started.pop(key)
            elapsed = max(0.0, timestamp - started)
            stall_cycles_by_reason[reason] = stall_cycles_by_reason.get(reason, 0.0) + elapsed
            events.append(
                TraceEvent(
                    timestamp,
                    "STALL_END",
                    task.task_id,
                    task.resource,
                    details={"reason": reason, "duration": elapsed},
                )
            )

    def push_complete(task: ExecutionTask, resource: _ResourceState, finish: float) -> None:
        nonlocal event_serial
        event_serial += 1
        heapq.heappush(
            event_queue,
            (finish, event_serial, _EventKind.COMPLETE, task.task_id, task.resource, resource.instance),
        )

    def visible_ready_tasks() -> list[ExecutionTask]:
        if static_pipeline is not None:
            ordered = sorted(
                (tasks[task_id] for task_id in ready),
                key=lambda task: (
                    static_reservations.get(task.task_id, math.inf),
                    program_order[task.task_id],
                    order_index[task.task_id],
                    task.task_id,
                ),
            )
        else:
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

    def static_reservation(task: ExecutionTask) -> float | None:
        return static_reservations.get(task.task_id)

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
            if address_scoreboard is not None:
                address_scoreboard.release(task_id)
            tile_completed_count[task.tile_id] += 1
            if tile_completed_count[task.tile_id] == tile_task_count[task.tile_id]:
                inflight_tiles.discard(task.tile_id)
            record_occupancy(finish, "COMPLETE")
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
            blocked_by_address: dict[str, AddressConflict] = {}
            blocked_by_static: set[str] = set()
            for task in visible:
                if task.tile_id not in inflight_tiles and len(inflight_tiles) >= (config.max_inflight_tiles or math.inf):
                    continue
                reservation = static_reservation(task)
                if reservation is not None and now + 1e-9 < reservation:
                    blocked_by_static.add(task.task_id)
                    begin_stall(task, "static_reservation", {"reservation_cycle": reservation})
                    continue
                if address_scoreboard is not None:
                    conflict = address_scoreboard.conflict(task)
                    if conflict is not None:
                        blocked_by_address[task.task_id] = conflict
                        observed_address_hazards.setdefault(
                            (
                                conflict.predecessor,
                                conflict.successor,
                                conflict.kind.value,
                                conflict.tensor,
                                conflict.memory,
                            ),
                            conflict,
                        )
                        begin_stall(
                            task,
                            f"address_{conflict.kind.value.lower()}",
                            {
                                "predecessor": conflict.predecessor,
                                "tensor": conflict.tensor,
                                "memory": conflict.memory,
                            },
                        )
                        continue
                for resource, start, resource_ready in candidate_resources(task):
                    candidates.append((task, resource, start, resource_ready))
            if not candidates:
                if ready:
                    if blocked_by_address:
                        metrics["address_scoreboard_block_events"] = metrics.get("address_scoreboard_block_events", 0) + 1
                    elif blocked_by_static:
                        metrics["static_block_events"] += 1
                    else:
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
                if config.dynamic_priority == "critical_path":
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
                        key=lambda item: (
                            item[0].program_order,
                            item[2],
                            item[0].task_id,
                            item[1].instance,
                        ),
                    )
            else:
                selected = min(
                    candidates,
                    key=lambda item: (
                        static_reservation(item[0])
                        if static_reservation(item[0]) is not None
                        else math.inf,
                        item[0].program_order,
                        item[0].task_id,
                        item[1].instance,
                    ),
                )
            task, resource, _estimated_start, _estimated_resource_ready = selected
            spec = timing_model.timing(task, machine)
            start, resource_ready = resource.reserve(now, spec.duration_cycles)
            finish = start + spec.duration_cycles
            end_stalls(task, now)
            ready.remove(task.task_id)
            issued.add(task.task_id)
            rob_occupancy += 1
            inflight_tiles.add(task.tile_id)
            if address_scoreboard is not None:
                address_scoreboard.reserve(task)
            record_occupancy(now, "ISSUE")
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
            if start > now:
                stall_cycles_by_reason["resource_queue"] = stall_cycles_by_reason.get("resource_queue", 0.0) + (start - now)
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
                reservation = static_reservation(task)
                if reservation is not None and reservation > now + 1e-9:
                    next_times.append(reservation)
                    continue
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

    for (task_id, reason), started in tuple(stall_started.items()):
        end_stalls(tasks[task_id], max((timing.finish for timing in timings.values()), default=now))
    metrics["address_dependency_count"] = len(observed_address_hazards)
    metrics["address_hazard_count"] = len(observed_address_hazards)
    metrics["address_hazards"] = [
        conflict.to_dependency().to_dict()
        for conflict in sorted(
            observed_address_hazards.values(),
            key=lambda item: (item.predecessor, item.successor, item.kind.value, item.tensor, item.memory),
        )
    ]
    if static_reservations:
        last_reserved_issue = max(static_reservations.values())
        metrics["pipeline_drain_cycles"] = max(
            0.0,
            max((timing.finish for timing in timings.values()), default=0.0) - last_reserved_issue,
        )
    events.sort(
        key=lambda event: (
            event.timestamp,
            {"STALL_BEGIN": 0, "ISSUE": 1, "START": 2, "COMPLETE": 3, "WAKE_UP": 4, "STALL_END": 5}[event.event],
            event.task_id,
            event.instance,
        )
    )
    total_cycles = max((timing.finish for timing in timings.values()), default=0.0)
    metrics["resource_utilization"] = {
        unit.name: (
            metrics["resource_busy_cycles"].get(unit.name, 0.0)
            / (total_cycles * unit.count)
            if total_cycles > 0
            else 0.0
        )
        for unit in machine.execution_units
    }
    metrics["queue_peak_occupancy"] = {
        "rob": metrics["rob_peak"],
        "ready": metrics["ready_set_peak"],
        "visible_ready": metrics["visible_ready_peak"],
        "inflight_tiles": metrics["inflight_tile_peak"],
    }
    metrics["completed_tile_count"] = len(tile_task_count)
    metrics["calibration_status"] = machine.attributes.get("calibration_status", "unspecified")
    metrics["task_count"] = len(tasks)
    return SimulationResult(
        backend=timing_model.name,
        policy=policy,
        graph_id=graph.graph_id,
        total_cycles=total_cycles,
        timings=tuple(timings[task.task_id] for task in graph.tasks),
        events=tuple(events),
        metrics=metrics,
    )
