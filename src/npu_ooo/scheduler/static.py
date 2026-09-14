"""Execution of compiler-generated fixed streams and explicit synchronization."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping

from npu_ooo.execution import ExecutionBackend, IssueRequest
from npu_ooo.ir import LoadedDeviceProgram, StaticControlCommand
from npu_ooo.simulator.core import SimulationResult, SimulatorConfig, TaskTiming, TraceEvent
from npu_ooo.scheduler.semantics import dependency_details


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
        self.streams = {item.stream_id: item for item in self.control.streams}
        self.commands = {
            command.command_id: command
            for stream in self.control.streams
            for command in stream.commands
        }
        if {
            envelope.command_id for envelope in loaded.static_command_envelopes
        } != set(self.commands):
            raise ValueError(
                "static_streams requires one runtime-submitted TISA/control program; "
                "regenerate the runtime submission"
            )
        self.wq = {stream_id: deque() for stream_id in self.streams}
        self.reception = deque()
        self.next_receive = 0
        self.received = set()
        self.dispatched = set()
        self.receive_cycles: dict[str, float] = {}
        self.dispatch_cycles: dict[str, float] = {}
        self.dispatch_counts = Counter()
        self.stall_cycles = Counter()
        self.snapshots: list[dict[str, Any]] = []
        self.program_input = []
        for envelope in sorted(
            loaded.static_command_envelopes, key=lambda item: item.command_order
        ):
            command_id = envelope.command_id
            self.program_input.append(
                (envelope.arrival_cycle, command_id, envelope.chunk_id)
            )
        for stream in self.control.streams:
            positions = [
                next(
                    envelope.command_order
                    for envelope in loaded.static_command_envelopes
                    if envelope.command_id == command.command_id
                )
                for command in stream.commands
            ]
            if positions != sorted(positions):
                raise ValueError(
                    f"runtime static program changes command order for '{stream.stream_id}'"
                )
        self.pending_control: dict[str, tuple[StaticControlCommand, float]] = {}
        self.feedback: set[tuple[str, str]] = set()
        self.condition_ready: dict[tuple[str, str], float] = {}
        self.set_wait_table: Counter[str] = Counter()
        self.issued: dict[str, float] = {}
        self.instances: dict[str, int] = {}
        self.physical_done: dict[str, float] = {}
        self.completed: dict[str, float] = {}
        self.pending_done: dict[str, float] = {}
        self.instruction_timings: dict[str, TaskTiming] = {}
        self.events: list[TraceEvent] = []
        self.runtime_timings: list[TaskTiming] = []
        self.control_records: list[dict[str, Any]] = []
        self.open_intervals: dict[tuple[str, str], _OpenInterval] = {}
        self.control_counts = Counter()
        self.completed_commands: set[str] = set()
        self.control_busy_cycles = 0.0
        self.now = 0
        self._validate_runtime_aliases()
        chunks: dict[str, list[Any]] = {}
        for envelope in loaded.envelopes:
            chunks.setdefault(envelope.chunk_id, []).append(envelope)
        for envelope in loaded.static_command_envelopes:
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
                "descriptor_count": sum(
                    hasattr(item, "descriptor_id") for item in envelopes
                ),
                "command_count": sum(
                    hasattr(item, "command_id") for item in envelopes
                ),
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
    def _validate_runtime_aliases(self) -> None:
        """Prove every runtime alias edge from typed static-control causality."""

        Node = tuple[str, ...]
        successors: dict[Node, set[Node]] = {}

        def connect(source: Node, target: Node) -> None:
            successors.setdefault(source, set()).add(target)
            successors.setdefault(target, set())

        def condition_node(source: str, condition: str) -> Node:
            if condition.startswith("payload_ready:"):
                return ("condition", source, condition)
            return ("complete", source)

        command_nodes: dict[str, Node] = {}
        issue_nodes: dict[str, Node] = {}
        set_nodes: dict[str, list[Node]] = {}
        event_by_id = {event.event_id: event for event in self.control.events}

        for stream in self.control.streams:
            previous: Node | None = None
            for command in stream.commands:
                node = ("command", command.command_id)
                command_nodes[command.command_id] = node
                successors.setdefault(node, set())
                if previous is not None:
                    connect(previous, node)
                previous = node
                if command.kind == "issue":
                    assert command.tisa_id is not None
                    issue_nodes[command.tisa_id] = node
                    connect(node, ("complete", command.tisa_id))
                elif command.kind == "set":
                    for event_id in command.event_ids:
                        set_nodes.setdefault(event_id, []).append(node)

        for event_id, event in event_by_id.items():
            source_issue = issue_nodes[event.source_tisa_id]
            ready = condition_node(event.source_tisa_id, event.condition)
            connect(source_issue, ready)
            for node in set_nodes.get(event_id, ()):
                connect(ready, node)

        for stream in self.control.streams:
            for command in stream.commands:
                if command.kind not in {"wait", "fence"}:
                    continue
                wait_node = command_nodes[command.command_id]
                for event_id in command.event_ids:
                    for set_node in set_nodes.get(event_id, ()):
                        connect(set_node, wait_node)

        def happens_before(source: Node, target: Node) -> bool:
            pending = [source]
            visited = {source}
            while pending:
                current = pending.pop()
                if current == target:
                    return True
                for successor in successors.get(current, ()):
                    if successor not in visited:
                        visited.add(successor)
                        pending.append(successor)
            return False

        uncovered: list[str] = []
        for target, descriptor in self.descriptors.items():
            for dependency in descriptor.dependencies:
                if dependency.provenance.get("source") != "runtime_binding_alias":
                    continue
                source = dependency.source.tisa_id
                required = condition_node(source, dependency.condition)
                if not happens_before(required, issue_nodes[target]):
                    uncovered.append(
                        f"{source}-[{dependency.condition}]->{target}"
                    )
        if uncovered:
            raise ValueError(
                "runtime binding alias condition lacks compiler static-control "
                "happens-before proof; recompile for this binding or use "
                "dynamic_ready_queue: "
                + ", ".join(uncovered[:8])
            )

    def _head(self, stream_id: str) -> StaticControlCommand | None:
        queue = self.wq[stream_id]
        return self.commands[queue[0]] if queue else None

    def _event_runtime_id(self, event_id: str) -> str:
        event = next(item for item in self.control.events if item.event_id == event_id)
        return f"{self.loaded.invocation_id}::{event_id}::g{event.generation}"

    def _event_ready(self, event_id: str) -> bool:
        return self.set_wait_table[event_id] > 0

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
            event_id = command.event_ids[0]
            event = next(item for item in self.control.events if item.event_id == event_id)
            # The RTL table keeps one token for each destination wait.  The
            # StaticEvent consumer list provides the equivalent fan-out count.
            self.set_wait_table[event_id] += max(1, len(event.consumers))
            event_name = "STATIC_SET"
        elif command.kind in {"wait", "fence"}:
            for event_id in command.event_ids:
                if self.set_wait_table[event_id] <= 0:
                    raise RuntimeError(
                        f"static {command.kind} consumed an unavailable event token "
                        f"'{event_id}'"
                    )
                self.set_wait_table[event_id] -= 1
            event_name = (
                "STATIC_WAIT_SATISFIED"
                if command.kind == "wait"
                else "STATIC_FENCE_SATISFIED"
            )
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
                    "consuming": command.kind in {"wait", "fence"},
                    "set_wait_tokens": {
                        event_id: self.set_wait_table[event_id]
                        for event_id in command.event_ids
                    },
                },
            )
        )
        queue = self.wq[stream_id]
        if not queue or queue[0] != command.command_id:
            raise RuntimeError(
                f"static control '{command.command_id}' is not the WQ head"
            )
        queue.popleft()
        self.completed_commands.add(command.command_id)
        self.pending_control.pop(stream_id, None)

    def _accept_feedback(self) -> None:
        for feedback in self.execution.advance(self.now):
            tisa_id = self.loaded.descriptor(feedback.descriptor_id).instruction.tisa_id
            if feedback.kind == "partial_ready":
                self.feedback.add((tisa_id, feedback.condition))
                self.condition_ready[(tisa_id, feedback.condition)] = feedback.cycle
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
                self.instances[tisa_id],
                issue,
                issue,
                self.now,
                issue,
                issue,
            )

    def _validate_dependency_timing(self) -> int:
        """Verify the executed schedule against every bound dependency condition."""

        violations: list[str] = []
        validated = 0
        for target, descriptor in self.descriptors.items():
            target_issue = self.issued[target]
            for dependency in descriptor.dependencies:
                source = dependency.source.tisa_id
                if dependency.condition.startswith("payload_ready:"):
                    ready = self.condition_ready.get((source, dependency.condition))
                else:
                    ready = self.completed.get(source)
                validated += 1
                if ready is None or ready > target_issue:
                    violations.append(
                        f"{source}-[{dependency.condition}]->{target} "
                        f"ready={ready} issue={target_issue}"
                    )
        if violations:
            raise RuntimeError(
                "static stream dependency timing invariant failed: "
                + ", ".join(violations[:8])
            )
        return validated

    def _complete_controls(self) -> None:
        for stream_id, (command, finish) in tuple(self.pending_control.items()):
            if finish <= self.now:
                self._finish_control(stream_id, command)

    def _receive(self) -> None:
        """Admit the unified TISA/control program into one Reception FIFO."""

        received = 0
        while self.next_receive < len(self.program_input):
            ready, command_id, chunk_id = self.program_input[self.next_receive]
            if ready > self.now:
                break
            if len(self.reception) >= self.config.instruction_queue_depth:
                self.stall_cycles["reception_full"] += 1
                break
            if received >= self.pipe.receive_width:
                self.stall_cycles["receive_bandwidth"] += 1
                break
            self.reception.append(command_id)
            command = self.commands[command_id]
            if command.kind == "issue":
                assert command.tisa_id is not None
                self.received.add(command.tisa_id)
                self.receive_cycles[command.tisa_id] = self.now
            self.next_receive += 1
            received += 1
            descriptor = (
                self.descriptors[command.tisa_id]
                if command.tisa_id is not None
                else None
            )
            self.events.append(
                TraceEvent(
                    self.now,
                    "TISA_RECEIVE",
                    command_id,
                    (
                        f"Static/{descriptor.instruction.unit_map.unit}"
                        if descriptor is not None
                        else f"Static/{command.stream_id}"
                    ),
                    details={
                        "runtime_ready_cycle": ready,
                        "runtime_chunk_id": chunk_id,
                        "program_command_id": command_id,
                        "kind": command.kind,
                        "tisa_id": command.tisa_id,
                        **(
                            {
                                "operator_id": descriptor.instruction.operator_id,
                                "tile_id": descriptor.instruction.tile_id,
                                "dependencies": dependency_details(descriptor),
                                "runtime_operands": [
                                    item.to_dict() for item in descriptor.operands
                                ],
                            }
                            if descriptor is not None
                            else {}
                        ),
                        "static_streams": True,
                    },
                )
            )

    def _dispatch(self) -> None:
        """Route the Reception FIFO head to its owning EU WQ."""

        dispatched_streams: set[str] = set()
        for _ in range(self.pipe.dispatch_width):
            if not self.reception:
                break
            command_id = self.reception[0]
            command = self.commands[command_id]
            stream = self.streams[command.stream_id]
            if stream.stream_id in dispatched_streams:
                self.stall_cycles["dispatch_per_eu"] += 1
                break
            capacity = self.machine.unit(stream.resource).queue_depth
            if len(self.wq[stream.stream_id]) >= capacity:
                self.stall_cycles["wq_full"] += 1
                break
            self.reception.popleft()
            self.wq[stream.stream_id].append(command_id)
            dispatched_streams.add(stream.stream_id)
            self.dispatch_counts[stream.stream_id] += 1
            if command.kind == "issue":
                assert command.tisa_id is not None
                self.dispatched.add(command.tisa_id)
                self.dispatch_cycles[command.tisa_id] = self.now
            self.events.append(
                TraceEvent(
                    self.now,
                    "TISA_DISPATCH",
                    command_id,
                    f"Static/{stream.resource}",
                    stream.instance,
                    {
                        "static_stream_id": stream.stream_id,
                        "static_dispatch": "reception_fifo_head",
                        "program_command_id": command_id,
                        "command_kind": command.kind,
                        "tisa_id": command.tisa_id,
                        "queue_depth": capacity,
                    },
                )
            )

    def _issue(self, stream_id: str, command: StaticControlCommand) -> bool:
        assert command.tisa_id is not None
        descriptor = self.descriptors[command.tisa_id]
        queue = self.wq[stream_id]
        if not queue or queue[0] != command.command_id:
            self._open_wait(
                stream_id,
                command,
                "STATIC_WQ_WAIT",
                "descriptor_not_dispatched",
                {"queue_depth": len(queue)},
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
        receipt = self.execution.issue(
            IssueRequest(
                descriptor,
                self.now,
                instance=self.streams[stream_id].instance,
            )
        )
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
        assert receipt.instance is not None
        self.instances[command.tisa_id] = receipt.instance
        queue.popleft()
        self.completed_commands.add(command.command_id)
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
            self._dispatch()
            self._receive()
            control_started = 0
            issue_started = 0
            resource_issued = Counter()
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
                        self.stall_cycles["issue_bandwidth"] += 1
                        continue
                    if self._issue(stream_id, command):
                        issue_started += 1
                        resource_issued[resource] += 1
                    continue
                if control_started >= self.pipe.control_width:
                    self.stall_cycles["dispatch_bandwidth"] += 1
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
            self.snapshots.append(
                {
                    "timestamp": self.now,
                    "event": "CYCLE_END",
                    "reception_queue": len(self.reception),
                    "wq": {stream_id: len(queue) for stream_id, queue in self.wq.items()},
                    "pending_control": len(self.pending_control),
                }
            )
            done_commands = len(self.completed_commands) == len(self.commands)
            if done_commands and len(self.completed) == len(self.descriptors):
                break
            self.now += 1

        dependency_timing_validated_count = self._validate_dependency_timing()
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
            "scheduler_model": "static-streams-v2",
            "event_backend": self.audit.get("event_backend", "cycle_event"),
            "policy": "static_streams",
            "static_control_semantics": "shared_reception+per_eu_wq+set_wait_fence",
            "static_control_assumption": "project_model_uncalibrated",
            "static_control_counts": dict(self.control_counts),
            "static_control_busy_cycles": self.control_busy_cycles,
            "static_control_records": self.control_records,
            "set_wait_table_final": dict(self.set_wait_table),
            "static_wait_interval_count": sum(
                event.event
                in {"STATIC_WAIT_INTERVAL", "STATIC_SET_WAIT", "STATIC_RESOURCE_WAIT"}
                for event in self.events
            ),
            "stall_cycles": dict(self.stall_cycles),
            "queue_occupancy_timeline": self.snapshots,
            "reception_queue_peak": max(
                (item["reception_queue"] for item in self.snapshots), default=0
            ),
            "wq_peak": {
                stream_id: max(
                    (item["wq"][stream_id] for item in self.snapshots), default=0
                )
                for stream_id in self.streams
            },
            "wq_peak_by_resource": {
                resource: max(
                    (
                        sum(
                            item["wq"][stream_id]
                            for stream_id, stream in self.streams.items()
                            if stream.resource == resource
                        )
                        for item in self.snapshots
                    ),
                    default=0,
                )
                for resource in {stream.resource for stream in self.streams.values()}
            },
            "static_dispatch_semantics": "one_head_per_eu_per_cycle",
            "static_dispatch_counts": dict(self.dispatch_counts),
            "shared_workload_hash": self.control.workload_hash,
            "static_control_hash": self.control.control_hash,
            "dynamic_control_hash": self.control.attributes.get("dynamic_control_hash"),
            "static_controls_consumed": True,
            "tisa_instruction_count": len(self.descriptors),
            "issued_instruction_count": len(self.issued),
            "retired_instruction_count": len(self.completed),
            "retirement_order": "completion_ready_order",
            "dependency_timing_validated_count": dependency_timing_validated_count,
            "completed_instruction_count": len(self.completed),
            "payload_task_count": len(execution_timings),
            "resource_busy_cycles": dict(busy),
            "instruction_pipeline": {
                tisa_id: {
                    "received": self.receive_cycles[tisa_id],
                    "dispatched": self.dispatch_cycles[tisa_id],
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
            "timing_provider_coverage": self.audit.get("timing_provider_coverage"),
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
