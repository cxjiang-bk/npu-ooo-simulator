"""Execution of compiler-generated fixed streams and explicit synchronization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from npu_ooo.execution import ExecutionBackend, IssueRequest
from npu_ooo.ir import LoadedDeviceProgram, StaticControlCommand
from npu_ooo.simulator.core import SimulationResult, SimulatorConfig, TaskTiming, TraceEvent


@dataclass
class _OpenInterval:
    start: float
    event: str
    task_id: str
    resource: str
    details: dict[str, Any]


class _StaticStreamExecutor:
    def __init__(
        self,
        loaded: LoadedDeviceProgram,
        execution: ExecutionBackend,
        config: SimulatorConfig,
        machine,
        audit: Mapping[str, Any],
    ) -> None:
        if loaded.static_control is None:
            raise ValueError(
                "static_streams requires a compiler-generated static control program; "
                "recompile this package"
            )
        issues = loaded.static_control.validate(
            {item.instruction.tisa_id for item in loaded.descriptors}
        )
        if issues:
            raise ValueError("invalid static control: " + "; ".join(issues))
        self.loaded = loaded
        self.control = loaded.static_control
        self.execution = execution
        self.config = config.resolved(machine)
        self.pipe = self.config.pipeline
        self.machine = machine
        self.audit = dict(audit)
        self.descriptors = {
            item.instruction.tisa_id: item for item in loaded.descriptors
        }
        self.arrivals = {
            loaded.descriptor(item.descriptor_id).instruction.tisa_id: item.arrival_cycle
            for item in loaded.envelopes
        }
        self.streams = {item.stream_id: item for item in self.control.streams}
        self.pointers = {stream_id: 0 for stream_id in self.streams}
        self.pending_control: dict[str, tuple[StaticControlCommand, float]] = {}
        self.feedback: set[tuple[str, str]] = set()
        self.signaled: set[str] = set()
        self.issued: dict[str, float] = {}
        self.physical_done: dict[str, float] = {}
        self.completed: dict[str, float] = {}
        self.pending_done: dict[str, float] = {}
        self.instruction_timings: dict[str, TaskTiming] = {}
        self.events: list[TraceEvent] = []
        self.runtime_timings: list[TaskTiming] = []
        self.control_records: list[dict[str, Any]] = []
        self.open_intervals: dict[tuple[str, str], _OpenInterval] = {}
        self.control_counts = Counter()
        self.control_busy_cycles = 0.0
        self.now = 0
        self._validate_runtime_aliases()
        chunks: dict[str, list[Any]] = {}
        for envelope in loaded.envelopes:
            chunks.setdefault(envelope.chunk_id, []).append(envelope)
        for chunk_id, envelopes in sorted(
            chunks.items(), key=lambda item: min(row.chunk_order for row in item[1])
        ):
            finish = max(item.arrival_cycle for item in envelopes)
            start = finish - loaded.launch_latency_cycles
            self.runtime_timings.append(
                TaskTiming(
                    chunk_id,
                    "Runtime/Submit",
                    0,
                    start,
                    start,
                    finish,
                    start,
                    start,
                )
            )
            details = {
                "descriptor_count": len(envelopes),
                "runtime_policy": loaded.attributes.get("runtime_policy"),
            }
            self.events.extend(
                (
                    TraceEvent(
                        start,
                        "RUNTIME_SUBMIT_START",
                        chunk_id,
                        "Runtime/Submit",
                        details=details,
                    ),
                    TraceEvent(
                        finish,
                        "RUNTIME_SUBMIT_COMPLETE",
                        chunk_id,
                        "Runtime/Submit",
                        details=details,
                    ),
                )
            )
        for tisa_id, arrival in self.arrivals.items():
            descriptor = self.descriptors[tisa_id]
            self.events.append(
                TraceEvent(
                    arrival,
                    "TISA_RECEIVE",
                    tisa_id,
                    f"Static/{descriptor.instruction.unit_map.unit}",
                    details={
                        "operator_id": descriptor.instruction.operator_id,
                        "tile_id": descriptor.instruction.tile_id,
                        "static_streams": True,
                    },
                )
            )

    def _validate_runtime_aliases(self) -> None:
        """Reject bindings whose new alias edges are not compile-time ordered."""

        successors: dict[str, set[str]] = {item: set() for item in self.descriptors}
        for target, descriptor in self.descriptors.items():
            for dependency in descriptor.instruction.dependencies:
                successors[dependency.source].add(target)
        descendants: dict[str, set[str]] = {}

        def visit(source: str) -> set[str]:
            if source in descendants:
                return descendants[source]
            result: set[str] = set()
            for target in successors[source]:
                result.add(target)
                result.update(visit(target))
            descendants[source] = result
            return result

        uncovered: list[str] = []
        for target, descriptor in self.descriptors.items():
            for dependency in descriptor.dependencies:
                if dependency.provenance.get("source") != "runtime_binding_alias":
                    continue
                if target not in visit(dependency.source.tisa_id):
                    uncovered.append(f"{dependency.source.tisa_id}->{target}")
        if uncovered:
            raise ValueError(
                "runtime binding introduces alias ordering absent from compiler static "
                "control; recompile for this binding or use dynamic_ready_queue: "
                + ", ".join(uncovered[:8])
            )

    def _head(self, stream_id: str) -> StaticControlCommand | None:
        stream = self.streams[stream_id]
        pointer = self.pointers[stream_id]
        return stream.commands[pointer] if pointer < len(stream.commands) else None

    def _event_runtime_id(self, event_id: str) -> str:
        event = next(item for item in self.control.events if item.event_id == event_id)
        return f"{self.loaded.invocation_id}::{event_id}::g{event.generation}"

    def _event_ready(self, event_id: str) -> bool:
        return event_id in self.signaled

    def _set_source_ready(self, command: StaticControlCommand) -> bool:
        event = next(item for item in self.control.events if item.event_id == command.event_ids[0])
        if event.condition.startswith("payload_ready:"):
            return (event.source_tisa_id, event.condition) in self.feedback
        return event.source_tisa_id in self.completed

    def _blockers(self, command: StaticControlCommand) -> list[str]:
        return [
            self._event_runtime_id(event_id)
            for event_id in command.event_ids
            if not self._event_ready(event_id)
        ]

    def _open_wait(
        self,
        stream_id: str,
        command: StaticControlCommand,
        event: str,
        reason: str,
        details: Mapping[str, Any],
    ) -> None:
        key = (stream_id, command.command_id)
        if key not in self.open_intervals:
            self.open_intervals[key] = _OpenInterval(
                self.now,
                event,
                command.command_id,
                f"Control/{stream_id}",
                {
                    "reason": reason,
                    "stream_id": stream_id,
                    "tisa_id": command.tisa_id,
                    **dict(details),
                },
            )

    def _close_wait(self, stream_id: str, command: StaticControlCommand) -> None:
        current = self.open_intervals.pop((stream_id, command.command_id), None)
        if current is None:
            return
        self.events.append(
            TraceEvent(
                current.start,
                current.event,
                current.task_id,
                current.resource,
                details={
                    **current.details,
                    "start": current.start,
                    "end": self.now,
                    "duration": self.now - current.start,
                },
            )
        )

    def _control_latency(self, command: StaticControlCommand) -> int:
        if command.kind == "wait":
            return self.pipe.wait_latency
        if command.kind == "fence":
            return self.pipe.fence_latency
        return self.pipe.control_latency

    def _start_control(self, stream_id: str, command: StaticControlCommand) -> None:
        self._close_wait(stream_id, command)
        latency = self._control_latency(command)
        finish = self.now + latency
        self.control_counts[command.kind] += 1
        self.control_busy_cycles += latency
        self.control_records.append(
            {
                "command_id": command.command_id,
                "kind": command.kind,
                "stream_id": stream_id,
                "start": self.now,
                "finish": finish,
                "duration": latency,
                "event_ids": [self._event_runtime_id(item) for item in command.event_ids],
                "tisa_id": command.tisa_id,
            }
        )
        self.events.append(
            TraceEvent(
                self.now,
                "STATIC_CONTROL_START",
                command.command_id,
                f"Control/{stream_id}",
                details={
                    "kind": command.kind,
                    "tisa_id": command.tisa_id,
                    "event_ids": [
                        self._event_runtime_id(item) for item in command.event_ids
                    ],
                    "scope": command.scope,
                    "source_kind": command.source_kind,
                },
            )
        )
        if latency == 0:
            self._finish_control(stream_id, command)
        else:
            self.pending_control[stream_id] = (command, finish)

    def _finish_control(self, stream_id: str, command: StaticControlCommand) -> None:
        if command.kind == "set":
            self.signaled.add(command.event_ids[0])
            event_name = "STATIC_SET"
        elif command.kind == "wait":
            event_name = "STATIC_WAIT_SATISFIED"
        else:
            event_name = "STATIC_FENCE_SATISFIED"
        self.events.append(
            TraceEvent(
                self.now,
                event_name,
                command.command_id,
                f"Control/{stream_id}",
                details={
                    "kind": command.kind,
                    "tisa_id": command.tisa_id,
                    "event_ids": [
                        self._event_runtime_id(item) for item in command.event_ids
                    ],
                    "scope": command.scope,
                    "source_kind": command.source_kind,
                    "consuming": False,
                },
            )
        )
        self.pointers[stream_id] += 1
        self.pending_control.pop(stream_id, None)

    def _accept_feedback(self) -> None:
        for feedback in self.execution.advance(self.now):
            tisa_id = self.loaded.descriptor(feedback.descriptor_id).instruction.tisa_id
            if feedback.kind == "partial_ready":
                self.feedback.add((tisa_id, feedback.condition))
                self.events.append(
                    TraceEvent(
                        feedback.cycle,
                        "TISA_PARTIAL_READY",
                        tisa_id,
                        f"TISA/{feedback.resource}",
                        feedback.instance,
                        {"condition": feedback.condition},
                    )
                )
            else:
                self.physical_done[tisa_id] = feedback.cycle
                self.pending_done[tisa_id] = feedback.cycle
                self.events.append(
                    TraceEvent(
                        feedback.cycle,
                        "TISA_EXECUTION_DONE",
                        tisa_id,
                        f"TISA/{feedback.resource}",
                        feedback.instance,
                        {"static_streams": True},
                    )
                )
        due = [
            tisa_id
            for tisa_id, cycle in self.pending_done.items()
            if tisa_id not in self.completed
            and cycle + self.pipe.completion_latency <= self.now
        ]
        due.sort(key=lambda item: (self.pending_done[item], item))
        for tisa_id in due[: self.pipe.completion_width]:
            self.completed[tisa_id] = self.now
            self.feedback.add((tisa_id, "complete"))
            descriptor = self.descriptors[tisa_id]
            self.events.append(
                TraceEvent(
                    self.now,
                    "TISA_COMPLETE",
                    tisa_id,
                    f"TISA/{descriptor.instruction.unit_map.unit}",
                    details={"static_streams": True},
                )
            )
            self.events.append(
                TraceEvent(
                    self.now,
                    "TISA_RETIRE",
                    tisa_id,
                    f"TISA/{descriptor.instruction.unit_map.unit}",
                    details={"static_streams": True, "retire_model": "completion"},
                )
            )
            issue = self.issued[tisa_id]
            self.instruction_timings[tisa_id] = TaskTiming(
                tisa_id,
                f"TISA/{descriptor.instruction.unit_map.unit}",
                0,
                issue,
                issue,
                self.now,
                issue,
                issue,
            )

    def _complete_controls(self) -> None:
        for stream_id, (command, finish) in tuple(self.pending_control.items()):
            if finish <= self.now:
                self._finish_control(stream_id, command)

    def _issue(self, stream_id: str, command: StaticControlCommand) -> bool:
        assert command.tisa_id is not None
        descriptor = self.descriptors[command.tisa_id]
        arrival = self.arrivals.get(command.tisa_id, 0.0)
        if arrival > self.now:
            self._open_wait(
                stream_id,
                command,
                "STATIC_DESCRIPTOR_WAIT",
                "descriptor_not_arrived",
                {"arrival_cycle": arrival},
            )
            return False
        accepted, reason = self.execution.can_accept(descriptor, self.now)
        if not accepted:
            self._open_wait(
                stream_id,
                command,
                "STATIC_RESOURCE_WAIT",
                reason or "execution_rejected",
                {"resource": self.streams[stream_id].resource},
            )
            return False
        self._close_wait(stream_id, command)
        receipt = self.execution.issue(IssueRequest(descriptor, self.now))
        if not receipt.accepted:
            self._open_wait(
                stream_id,
                command,
                "STATIC_RESOURCE_WAIT",
                receipt.rejection_reason or "execution_rejected",
                {"resource": self.streams[stream_id].resource},
            )
            return False
        self.issued[command.tisa_id] = self.now
        self.pointers[stream_id] += 1
        self.events.append(
            TraceEvent(
                self.now,
                "TISA_ISSUE",
                command.tisa_id,
                f"TISA/{receipt.resource}",
                int(receipt.instance),
                {
                    "static_stream_id": stream_id,
                    "static_command_id": command.command_id,
                    "operator_id": descriptor.instruction.operator_id,
                    "tile_id": descriptor.instruction.tile_id,
                },
            )
        )
        return True

    def run(self) -> SimulationResult:
        while True:
            if self.now >= self.pipe.max_cycles:
                raise RuntimeError(
                    f"static stream executor exceeded max_cycles={self.pipe.max_cycles}"
                )
            self._accept_feedback()
            self._complete_controls()
            control_started = 0
            issue_started = 0
            resource_issued = Counter()
            made_progress = True
            while made_progress:
                made_progress = False
                for stream_id in sorted(self.streams):
                    if stream_id in self.pending_control:
                        continue
                    command = self._head(stream_id)
                    if command is None:
                        continue
                    if command.kind == "issue":
                        resource = self.streams[stream_id].resource
                        if (
                            issue_started >= self.pipe.issue_width
                            or resource_issued[resource]
                            >= self.machine.unit(resource).issue_width
                        ):
                            continue
                        if self._issue(stream_id, command):
                            issue_started += 1
                            resource_issued[resource] += 1
                            made_progress = True
                        continue
                    if control_started >= self.pipe.control_width:
                        continue
                    if command.kind == "set" and not self._set_source_ready(command):
                        self._open_wait(
                            stream_id,
                            command,
                            "STATIC_SET_WAIT",
                            "producer_feedback_pending",
                            {"tisa_id": command.tisa_id},
                        )
                        continue
                    if command.kind in {"wait", "fence"}:
                        blockers = self._blockers(command)
                        if blockers:
                            self._open_wait(
                                stream_id,
                                command,
                                "STATIC_WAIT_INTERVAL",
                                "event_pending",
                                {
                                    "kind": command.kind,
                                    "blockers": blockers,
                                    "buffer_slots": list(command.buffer_slots),
                                },
                            )
                            continue
                    self._start_control(stream_id, command)
                    control_started += 1
                    made_progress = True
            done_commands = all(
                self.pointers[stream_id] == len(stream.commands)
                for stream_id, stream in self.streams.items()
            )
            if done_commands and len(self.completed) == len(self.descriptors):
                break
            self.now += 1

        for stream_id, command in (
            (stream_id, self._head(stream_id)) for stream_id in self.streams
        ):
            if command is not None:
                self._close_wait(stream_id, command)
        execution_timings = tuple(self.execution.task_timings())
        self.events.extend(self.execution.trace_events())
        self.events.sort(key=lambda item: item.timestamp)
        busy = Counter()
        for timing in execution_timings:
            busy[timing.resource] += timing.duration
        metrics = {
            "scheduler_target": "compiler_static_control",
            "scheduler_model": "static-streams-v1",
            "event_backend": self.audit.get("event_backend", "cycle_event"),
            "policy": "static_streams",
            "static_control_semantics": "fixed_per_eu_streams+set_wait_fence",
            "static_control_assumption": "project_model_uncalibrated",
            "static_control_counts": dict(self.control_counts),
            "static_control_busy_cycles": self.control_busy_cycles,
            "static_control_records": self.control_records,
            "static_wait_interval_count": sum(
                event.event
                in {"STATIC_WAIT_INTERVAL", "STATIC_SET_WAIT", "STATIC_RESOURCE_WAIT"}
                for event in self.events
            ),
            "shared_workload_hash": self.control.workload_hash,
            "static_control_hash": self.control.control_hash,
            "dynamic_control_hash": self.control.attributes.get("dynamic_control_hash"),
            "static_controls_consumed": True,
            "tisa_instruction_count": len(self.descriptors),
            "issued_instruction_count": len(self.issued),
            "completed_instruction_count": len(self.completed),
            "payload_task_count": len(execution_timings),
            "resource_busy_cycles": dict(busy),
            "instruction_pipeline": {
                tisa_id: {
                    "received": self.arrivals.get(tisa_id, 0.0),
                    "issued": self.issued[tisa_id],
                    "done": self.physical_done[tisa_id],
                    "completed": self.completed[tisa_id],
                    "retired": self.completed[tisa_id],
                }
                for tisa_id in self.descriptors
            },
            "runtime_synchronization_cycles": self.loaded.synchronization_cycles,
            "runtime_launch_count": len(self.runtime_timings),
            "runtime_submit_cycles": float(
                self.loaded.attributes.get("runtime_submit_cycles", 0.0)
            ),
            "device_finish_cycle": self.now,
            "completion_finish_cycle": max(self.completed.values(), default=0),
            "total_cycles_including_runtime": self.now
            + self.loaded.synchronization_cycles,
            "compile_schedule_estimate_cycles": self.control.attributes.get(
                "schedule_estimate_cycles"
            ),
            "timing_calibration_status": self.audit.get(
                "timing_calibration_status", "analytical"
            ),
            "scheduler_calibration_status": "uncalibrated",
            "calibration_status": "analytical",
        }
        backend_metrics = getattr(self.execution, "metrics", None)
        if callable(backend_metrics):
            metrics.update(backend_metrics())
        return SimulationResult(
            backend=str(self.audit.get("timing_provider_name", self.execution.name)),
            policy="static_streams",
            graph_id=self.loaded.program_id,
            total_cycles=self.now + self.loaded.synchronization_cycles,
            timings=execution_timings,
            instruction_timings=tuple(
                self.instruction_timings[item]
                for item in self.descriptors
            ),
            runtime_timings=tuple(self.runtime_timings),
            events=tuple(self.events),
            metrics=metrics,
        )


def schedule_loaded_static_program(
    loaded: LoadedDeviceProgram,
    machine,
    execution: ExecutionBackend,
    *,
    config: SimulatorConfig | None = None,
    audit: Mapping[str, Any] | None = None,
) -> SimulationResult:
    return _StaticStreamExecutor(
        loaded,
        execution,
        config or SimulatorConfig(dynamic_priority="oldest_first"),
        machine,
        audit or {},
    ).run()


__all__ = ["schedule_loaded_static_program"]
