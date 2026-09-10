"""Clock-edge TISA scheduler. See docs/device-scheduler.md for edge semantics.

The payload recipe and timing provider are shared with the event baseline.
Reception, WQ, IQ, operand tracking, completion and retirement own separate
state. Reverse pipeline evaluation guarantees a registered boundary between
receive/dispatch/select/issue, even when several widths exceed one.
"""

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math

from npu_ooo.backend.contracts import validate_backend_capability
from npu_ooo.execution import AnalyticalExecutionBackend, IssueRequest
from npu_ooo.runtime import load_device_program, load_implicit_device_program
from npu_ooo.scheduler.semantics import (
    address_conflict,
    address_observation,
    critical_path_lengths,
    dependency_details,
    memory_accesses,
    memory_port_conflict,
    unit_map_matches,
)

from .core import (
    AnalyticalTimingModel,
    SimulationResult,
    SimulatorConfig,
    TaskTiming,
    TraceEvent,
)


STALL_REASONS = (
    "reception_full",
    "wq_full",
    "iq_full",
    "rob_full",
    "dependency_wait",
    "fu_busy",
    "fu_table_full",
    "tile_window_full",
    "address_hazard",
    "memory_bank_port_conflict",
    "completion_bandwidth",
    "retire_backpressure",
    "receive_bandwidth",
    "dispatch_bandwidth",
    "select_bandwidth",
    "issue_bandwidth",
    "static_order",
)


@dataclass
class _Entry:
    tisa_id: str
    resource: str
    received: int | None = None
    dispatched: int | None = None
    wakeup: int | None = None
    selected: int | None = None
    issued: int | None = None
    done: int | None = None
    completed: int | None = None
    retired: int | None = None
    instance: int = 0


class _CycleScheduler:
    def __init__(self, loaded, machine, policy, execution, config, audit):
        if policy not in {"sequential", "static_pipeline", "dynamic_ready_queue"}:
            raise ValueError(f"unsupported scheduler policy '{policy}'")
        issues = (*loaded.validate(), *machine.validate())
        if issues:
            raise ValueError("; ".join(issues))
        self.config = config.resolved(machine)
        if self.config.static_pipeline is not None:
            raise ValueError(
                "primitive reservations are not supported by the TISA cycle scheduler"
            )
        self.pipe = self.config.pipeline
        self.loaded, self.machine, self.policy = loaded, machine, policy
        self.execution, self.audit = execution, audit
        self.descriptors = {
            item.instruction.tisa_id: item for item in loaded.descriptors
        }
        self.instructions = {
            tisa_id: descriptor.instruction
            for tisa_id, descriptor in self.descriptors.items()
        }
        self.program_order = {
            tid: descriptor.program_order for tid, descriptor in self.descriptors.items()
        }
        self.plans = {
            tid: execution.estimate(descriptor.payload_handle)
            for tid, descriptor in self.descriptors.items()
        }
        self.entries = {
            tid: _Entry(tid, self.plans[tid].resource) for tid in self.instructions
        }
        self.units = {unit.name: unit for unit in machine.execution_units}
        self.runtime_operands = {
            tid: descriptor.operands for tid, descriptor in self.descriptors.items()
        }
        self.accesses = {
            tid: memory_accesses(self.descriptors[tid], machine)
            for tid in self.instructions
        }
        for tid, instruction in self.instructions.items():
            resource = self.entries[tid].resource
            if instruction.unit_map.quantity != 1 or not unit_map_matches(
                self.descriptors[tid], resource
            ):
                raise ValueError(
                    f"'{tid}' must request one unit matching its payload resource"
                )
            if len(instruction.operands) > self.pipe.inflight_entries:
                raise ValueError(
                    f"'{tid}' exceeds per-unit inflight_entries operand capacity"
                )
            durations = [
                step.finish_offset - step.start_offset for step in self.plans[tid].steps
            ]
            if any(not math.isfinite(value) or value <= 0 for value in durations):
                raise ValueError(
                    f"'{tid}' payload durations must be finite and positive"
                )
        oracle_priority = self.config.dynamic_priority in {
            "critical_path",
            "oracle_critical_path",
        }
        self.priorities = (
            critical_path_lengths(tuple(self.descriptors.values()), execution)
            if oracle_priority
            else {tid: 0.0 for tid in self.descriptors}
        )
        self.events, self.runtime_timings, self.snapshots = [], [], []
        self.timings, self.instruction_timings = {}, {}
        self.stall_cycles = Counter({reason: 0 for reason in STALL_REASONS})
        self.stall_instruction_cycles = Counter({reason: 0 for reason in STALL_REASONS})
        self.stall_seen, self.cycle_reasons = set(), set()
        self.active_stalls: dict[tuple[str, str, str], float] = {}
        self.hazards = {}
        self.stream = self._descriptor_stream()
        self.submission_order = {tid: i for i, (_, tid, _) in enumerate(self.stream)}
        self.static_order = tuple(
            item.tisa_id
            for item in sorted(
                loaded.static_schedule.entries, key=lambda row: row.order
            )
        ) if loaded.static_schedule is not None else tuple(
            sorted(self.instructions, key=self.program_order.__getitem__)
        )
        self.next_receive = 0
        self.reception = deque()
        self.wq = {name: [] for name in self.units}
        self.iq = {name: [] for name in self.units}
        self.fu = {name: set() for name in self.units}
        self.running = {}
        self.rob = deque()
        self.completed, self.retired, self.issued = set(), set(), set()
        self.tile_remaining = Counter(
            item.tile_id for item in self.instructions.values()
        )
        self.active_tiles = set()
        self.feedback_ready = {}
        self.pending_done = {}
        self.pending_partial = {}
        self.now = 0
        self.progress = 0

    def _descriptor_stream(self):
        self.submit_cycles = float(
            self.loaded.attributes.get("runtime_submit_cycles", 0.0)
        )
        self.submit_busy = (
            len({item.chunk_id for item in self.loaded.envelopes})
            * self.loaded.launch_latency_cycles
        )
        self.request_wait = max(0.0, self.submit_cycles - self.submit_busy)
        self.sync_cycles = self.loaded.synchronization_cycles
        stream = [
            (
                envelope.arrival_cycle,
                self.loaded.descriptor(envelope.descriptor_id).instruction.tisa_id,
                envelope.chunk_id,
            )
            for envelope in sorted(
                self.loaded.envelopes, key=lambda item: item.descriptor_order
            )
        ]
        chunks: dict[str, list] = {}
        for envelope in self.loaded.envelopes:
            chunks.setdefault(envelope.chunk_id, []).append(envelope)
        for chunk_id, envelopes in sorted(
            chunks.items(), key=lambda item: min(row.chunk_order for row in item[1])
        ):
            finish = max(item.arrival_cycle for item in envelopes)
            start = finish - self.loaded.launch_latency_cycles
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
                "runtime_policy": self.loaded.attributes.get("runtime_policy"),
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
        return stream

    def event(self, event, tid, **details):
        if event != "TISA_STALL":
            self.progress += 1
        entry, instruction = self.entries[tid], self.instructions[tid]
        self.events.append(
            TraceEvent(
                self.now,
                event,
                tid,
                f"TISA/{entry.resource}",
                entry.instance,
                {
                    "op_type": instruction.op_type,
                    "operator_id": instruction.operator_id,
                    "tile_id": instruction.tile_id,
                    "unit_map": instruction.unit_map.unit,
                    "payload_task_count": len(self.plans[tid].steps),
                    "stage_event": event,
                    "dependencies": dependency_details(self.descriptors[tid]),
                    **details,
                },
            )
        )

    def stall(self, tid, reason, stage, **_details):
        key = (tid, reason, stage)
        if key in self.stall_seen:
            return
        self.stall_seen.add(key)
        self.cycle_reasons.add(reason)
        self.stall_instruction_cycles[reason] += 1
        self.active_stalls.setdefault(key, self.now)

    def _close_stall(self, key: tuple[str, str, str], end: float) -> None:
        start = self.active_stalls.pop(key)
        tid, reason, stage = key
        entry = self.entries[tid]
        self.events.append(
            TraceEvent(
                start,
                "TISA_STALL",
                tid,
                f"TISA/{entry.resource}",
                entry.instance,
                {
                    "reason": reason,
                    "stage": stage,
                    "dependency_count": len(self.descriptors[tid].dependencies),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                },
            )
        )

    def _close_inactive_stalls(self, end: float) -> None:
        for key in tuple(self.active_stalls):
            if key not in self.stall_seen:
                self._close_stall(key, end)

    def retire(self):
        for _ in range(self.pipe.retire_width):
            if not self.rob:
                break
            tid = self.rob[0]
            entry = self.entries[tid]
            if (
                entry.completed is None
                or entry.completed + self.pipe.retire_latency > self.now
            ):
                break
            self.rob.popleft()
            entry.retired = self.now
            self.retired.add(tid)
            self.event("TISA_RETIRE", tid)
        for tid in self.rob:
            entry = self.entries[tid]
            if (
                entry.completed is not None
                and entry.completed + self.pipe.retire_latency <= self.now
            ):
                self.stall(tid, "retire_backpressure", "retire", head=self.rob[0])

    def complete(self):
        # The execution backend owns physical completion.  Scheduler feedback
        # acceptance remains subject to completion latency and bandwidth.
        for feedback in self.execution.advance(self.now):
            tid = self.loaded.descriptor(feedback.descriptor_id).instruction.tisa_id
            if feedback.kind == "partial_ready":
                self.pending_partial[(tid, feedback.condition)] = feedback.cycle
                continue
            self.pending_done[tid] = feedback.cycle
            self.running.pop(tid, None)
            self.event("TISA_EXECUTION_DONE", tid)
        due = [
            tid for tid, done in self.pending_done.items()
            if tid not in self.completed
            and done + self.pipe.completion_latency <= self.now
        ]
        due.sort(key=lambda tid: (self.pending_done[tid], self.submission_order[tid]))
        for tid in due[: self.pipe.completion_width]:
            entry = self.entries[tid]
            entry.completed = self.now
            self.completed.add(tid)
            self.fu[entry.resource].remove(tid)
            self.tile_remaining[self.instructions[tid].tile_id] -= 1
            if not self.tile_remaining[self.instructions[tid].tile_id]:
                self.active_tiles.discard(self.instructions[tid].tile_id)
            self.event("TISA_COMPLETE", tid)
            self.instruction_timings[tid] = TaskTiming(
                tid,
                f"TISA/{entry.resource}",
                entry.instance,
                entry.issued,
                entry.issued,
                self.now,
                entry.wakeup,
                entry.issued,
            )
        for tid in due[self.pipe.completion_width :]:
            self.stall(tid, "completion_bandwidth", "complete")
        # Partial readiness is a sideband path but still observes modeled feedback latency.
        for key, physical_ready in self.pending_partial.items():
            source, condition = key
            if key not in self.feedback_ready and physical_ready + self.pipe.completion_latency <= self.now:
                self.feedback_ready[key] = self.now
                self.event("TISA_PARTIAL_READY", source, condition=condition)

    def wakeup(self):
        for queue in self.wq.values():
            for tid in queue:
                entry = self.entries[tid]
                if (
                    entry.wakeup is not None
                    or entry.dispatched + self.pipe.dispatch_latency > self.now
                ):
                    continue
                ready = entry.dispatched + self.pipe.dispatch_latency
                for dependency in self.descriptors[tid].dependencies:
                    source = self.entries[dependency.source.tisa_id]
                    feedback = self.feedback_ready.get(
                        (dependency.source.tisa_id, dependency.condition), source.completed
                    )
                    if feedback is None:
                        ready = math.inf
                        break
                    ready = max(ready, feedback + self.pipe.wakeup_latency)
                if ready <= self.now:
                    entry.wakeup = self.now
                    self.event("TISA_WAKE_UP", tid)
                else:
                    self.stall(tid, "dependency_wait", "wakeup")

    def _address_block(self, tid):
        if not self.config.address_scoreboard:
            return None
        # Include older, not-yet-issued producers; an active-only table is insufficient.
        for older in self.instructions:
            if older == tid:
                break
            if older in self.completed or self.entries[older].received is None:
                continue
            conflict = address_conflict(
                self.descriptors[older], self.descriptors[tid]
            )
            if conflict is not None:
                kind, region = conflict
                observation = address_observation(
                    self.descriptors[older], self.descriptors[tid], kind, region
                )
                self.hazards[(older, tid, kind)] = observation
                return observation
        return None

    def issue(self):
        candidates = [
            tid
            for queue in self.iq.values()
            for tid in queue
            if self.entries[tid].selected + self.pipe.select_latency <= self.now
        ]
        candidates.sort(key=self._priority)
        used = Counter()
        issued_now = 0
        for tid in candidates:
            entry = self.entries[tid]
            resource = entry.resource
            if self.policy != "dynamic_ready_queue":
                next_tid = next(
                    (item for item in self.static_order if item not in self.issued),
                    None,
                )
                if tid != next_tid or (
                    self.policy == "sequential" and self.issued - self.retired
                ):
                    self.stall(tid, "static_order", "issue")
                    continue
            if (
                issued_now >= self.pipe.issue_width
                or used[resource] >= self.units[resource].issue_width
            ):
                self.stall(tid, "issue_bandwidth", "issue")
                continue
            tile = self.instructions[tid].tile_id
            if (
                tile not in self.active_tiles
                and len(self.active_tiles) >= self.config.max_inflight_tiles
            ):
                self.stall(tid, "tile_window_full", "issue")
                continue
            if (
                sum(len(self.instructions[item].operands) for item in self.fu[resource])
                + len(self.instructions[tid].operands)
                > self.pipe.inflight_entries
            ):
                self.stall(tid, "fu_table_full", "issue")
                continue
            hazard = self._address_block(tid)
            if hazard:
                self.stall(tid, "address_hazard", "issue", hazard=hazard)
                continue
            active_accesses = {item: self.accesses[item] for item in self.running}
            if self.config.memory_bank_scoreboard:
                conflict = memory_port_conflict(
                    active_accesses, self.accesses[tid], self.machine
                )
                if conflict:
                    self.stall(
                        tid, "memory_bank_port_conflict", "issue", conflict=conflict
                    )
                    continue
            descriptor = self.descriptors[tid]
            can_accept, rejection = self.execution.can_accept(descriptor, self.now)
            if not can_accept:
                self.stall(tid, "fu_busy", "issue")
                continue
            receipt = self.execution.issue(IssueRequest(descriptor, self.now))
            if not receipt.accepted:
                self.stall(
                    tid,
                    "fu_busy",
                    "issue",
                    execution_rejection=receipt.rejection_reason or rejection,
                )
                continue
            entry.instance = int(receipt.instance)
            entry.issued = self.now
            entry.done = int(receipt.expected_done_cycle)
            self.iq[resource].remove(tid)
            self.fu[resource].add(tid)
            self.running[tid] = entry.done
            self.issued.add(tid)
            self.active_tiles.add(tile)
            issued_now += 1
            used[resource] += 1
            self.event(
                "TISA_ISSUE",
                tid,
                runtime_operands=[
                    item.to_dict() for item in self.runtime_operands[tid]
                ],
            )

    def _priority(self, tid):
        if self.policy != "dynamic_ready_queue":
            return 0, self.submission_order[tid]
        if self.config.dynamic_priority in {"critical_path", "oracle_critical_path"}:
            return -self.priorities[tid], self.submission_order[tid]
        if self.config.dynamic_priority == "compiler_hint":
            hint = self.instructions[tid].attributes.get("scheduler_hint", {})
            value = hint.get("priority", 0) if isinstance(hint, dict) else 0
            return -float(value), self.submission_order[tid]
        return 0, self.submission_order[tid]

    def select(self):
        candidates = [
            tid
            for queue in self.wq.values()
            for tid in queue[: self.config.dependency_window]
            if self.entries[tid].wakeup is not None
        ]
        # Static admission also follows submission order, preventing a younger
        # ready entry from filling IQ while its older dependency is still in WQ.
        candidates.sort(
            key=self._priority
            if self.policy == "dynamic_ready_queue"
            else self.submission_order.__getitem__
        )
        selected = 0
        iq_capacity = min(self.pipe.iq_entries, self.config.ready_queue_depth)
        for tid in candidates:
            entry = self.entries[tid]
            hazard = self._address_block(tid)
            if hazard:
                self.stall(tid, "address_hazard", "select", hazard=hazard)
                continue
            if self.policy != "dynamic_ready_queue":
                older = [
                    item
                    for item in self.static_order[: self.static_order.index(tid)]
                    if self.entries[item].selected is None
                ]
                if older:
                    self.stall(tid, "static_order", "select")
                    continue
            if len(self.iq[entry.resource]) >= iq_capacity:
                self.stall(tid, "iq_full", "select")
                continue
            if selected >= self.pipe.select_width:
                self.stall(tid, "select_bandwidth", "select")
                continue
            entry.selected = self.now
            self.wq[entry.resource].remove(tid)
            self.iq[entry.resource].append(tid)
            selected += 1
            self.event("TISA_SELECT", tid)

    def dispatch(self):
        dispatched = 0
        while self.reception and dispatched < self.pipe.dispatch_width:
            tid = self.reception[0]
            resource = self.entries[tid].resource
            if len(self.rob) >= self.config.rob_entries:
                self.stall(tid, "rob_full", "dispatch")
                break
            if (
                len(self.wq[resource])
                >= self.units[resource].queue_depth * self.units[resource].count
            ):
                self.stall(tid, "wq_full", "dispatch")
                break
            self.reception.popleft()
            self.rob.append(tid)
            self.wq[resource].append(tid)
            self.entries[tid].dispatched = self.now
            dispatched += 1
            self.event("TISA_DISPATCH", tid)
        if self.reception and dispatched == self.pipe.dispatch_width:
            self.stall(self.reception[0], "dispatch_bandwidth", "dispatch")

    def receive(self):
        received = 0
        while self.next_receive < len(self.stream):
            ready, tid, chunk = self.stream[self.next_receive]
            if ready > self.now:
                break
            if len(self.reception) >= self.config.instruction_queue_depth:
                self.stall(tid, "reception_full", "receive")
                break
            if received >= self.pipe.receive_width:
                self.stall(tid, "receive_bandwidth", "receive")
                break
            self.reception.append(tid)
            self.entries[tid].received = self.now
            self.next_receive += 1
            received += 1
            self.event(
                "TISA_RECEIVE", tid, runtime_ready_cycle=ready, runtime_chunk_id=chunk
            )

    def snapshot(self):
        row = {
            "timestamp": self.now,
            "event": "CYCLE_END",
            "reception_queue": len(self.reception),
            "rob": len(self.rob),
            "inflight_tiles": len(self.active_tiles),
            "wq": {name: len(queue) for name, queue in self.wq.items()},
            "iq": {name: len(queue) for name, queue in self.iq.items()},
            "fu": {
                name: sum(len(self.instructions[tid].operands) for tid in entries)
                for name, entries in self.fu.items()
            },
            "completion_pending": sum(
                self.entries[tid].done <= self.now
                for tid in self.issued - self.completed
            ),
        }
        assert row["rob"] <= self.config.rob_entries
        assert row["reception_queue"] <= self.config.instruction_queue_depth
        assert all(value <= self.pipe.inflight_entries for value in row["fu"].values())
        assert all(
            row["wq"][name] <= unit.queue_depth * unit.count
            for name, unit in self.units.items()
        )
        assert all(
            value <= min(self.pipe.iq_entries, self.config.ready_queue_depth)
            for value in row["iq"].values()
        )
        queued = [
            *self.reception,
            *(tid for queue in self.wq.values() for tid in queue),
            *(tid for queue in self.iq.values() for tid in queue),
        ]
        assert len(queued) == len(set(queued)) and not set(queued).intersection(
            self.issued
        )
        assert set(self.rob) == {
            tid
            for tid, entry in self.entries.items()
            if entry.dispatched is not None and entry.retired is None
        }
        self.snapshots.append(row)
        for reason in self.cycle_reasons:
            self.stall_cycles[reason] += 1

    def run(self):
        while len(self.retired) < len(self.instructions):
            if self.now >= self.pipe.max_cycles:
                pending = [tid for tid in self.instructions if tid not in self.retired]
                raise RuntimeError(
                    f"cycle scheduler exceeded max_cycles={self.pipe.max_cycles}; pending={pending[:8]}"
                )
            self.stall_seen.clear()
            self.cycle_reasons.clear()
            previous_progress = self.progress
            self.retire()
            self.complete()
            self.wakeup()
            self.issue()
            self.select()
            self.dispatch()
            self.receive()
            self._close_inactive_stalls(self.now)
            self.snapshot()
            if len(self.retired) == len(self.instructions):
                break
            if self.progress == previous_progress and not self._future_transition():
                raise RuntimeError(
                    f"cycle scheduler deadlocked at cycle {self.now}; "
                    f"reception={list(self.reception)[:4]}, WQ={self.wq}, IQ={self.iq}, "
                    f"stalls={sorted(self.cycle_reasons)}"
                )
            self.now += 1
        return self.result()

    def _future_transition(self):
        if self.issued - self.completed:
            return True
        if self.rob:
            head = self.entries[self.rob[0]]
            if (
                head.completed is not None
                and head.completed + self.pipe.retire_latency > self.now
            ):
                return True
        if (
            self.next_receive < len(self.stream)
            and self.stream[self.next_receive][0] > self.now
        ):
            return True
        next_feedback = self.execution.next_feedback_cycle()
        if next_feedback is not None and next_feedback > self.now:
            return True
        if any(
            (cycle := self.execution.next_accept_cycle(self.descriptors[tid]))
            is not None
            and cycle > self.now
            for queue in self.iq.values()
            for tid in queue
        ):
            return True
        for entry in self.entries.values():
            if entry.retired is not None:
                continue
            if (
                entry.dispatched is not None
                and entry.dispatched + self.pipe.dispatch_latency > self.now
            ):
                return True
            if (
                entry.selected is not None
                and entry.selected + self.pipe.select_latency > self.now
            ):
                return True
            for dep in self.descriptors[entry.tisa_id].dependencies:
                feedback = self.feedback_ready.get(
                    (dep.source.tisa_id, dep.condition),
                    self.entries[dep.source.tisa_id].completed,
                )
                if (
                    feedback is not None
                    and feedback + self.pipe.wakeup_latency > self.now
                ):
                    return True
        return False

    def result(self):
        for key in tuple(self.active_stalls):
            self._close_stall(key, self.now + 1)
        execution_timings = tuple(self.execution.task_timings())
        self.timings = {item.task_id: item for item in execution_timings}
        self.events.extend(self.execution.trace_events())
        # Stable sort keeps the actual hardware phase order for equal timestamps.
        self.events.sort(key=lambda event: event.timestamp)
        start = min((item.issued for item in self.entries.values()), default=0)
        completion = max((item.completed for item in self.entries.values()), default=0)
        busy = Counter()
        for timing in self.timings.values():
            busy[timing.resource] += timing.duration
        timing_status = self.audit["timing_calibration_status"]
        payload_hash = self.audit["compile_package_sha256"]
        metrics = {
            "scheduler_target": "tisa",
            "scheduler_model": "cycle-stepped-v1",
            "event_backend": "cycle_event",
            "backend": self.audit["timing_provider_name"],
            "policy": self.policy,
            "calibration_status": "analytical",
            "scheduler_calibration_status": "uncalibrated",
            "timing_calibration_status": timing_status,
            "machine_calibration_status": self.machine.attributes.get(
                "calibration_status", "unspecified"
            ),
            "simulator_config": self.config.to_dict(),
            "compile_package_sha256": payload_hash,
            "phase_order": [
                "retire",
                "complete",
                "wakeup",
                "issue",
                "select",
                "dispatch",
                "receive",
            ],
            "retirement_order": "descriptor_submission_order",
            "payload_execution": "run_to_completion",
            "primitive_reordering_scope": "instruction_local",
            "fu_capacity_unit": "operand_entries_per_unit_class",
            "tisa_instruction_count": len(self.instructions),
            "payload_task_count": len(self.timings),
            "issued_instruction_count": len(self.issued),
            "completed_instruction_count": len(self.completed),
            "retired_instruction_count": len(self.retired),
            "tisa_decision_count": len(self.issued),
            "issued_task_count": len(self.timings),
            "completed_task_count": len(self.timings),
            "completed_tile_count": len(self.tile_remaining),
            "runtime_policy": self.loaded.attributes.get(
                "runtime_policy", "implicit_static"
            ),
            "runtime_launch_count": len(self.runtime_timings),
            "runtime_submit_cycles": self.submit_cycles,
            "runtime_submit_busy_cycles": self.submit_busy,
            "runtime_request_wait_cycles": self.request_wait,
            "runtime_synchronization_cycles": self.sync_cycles,
            "device_start_cycle": start,
            "device_finish_cycle": self.now,
            "device_cycles": self.now - start,
            "completion_finish_cycle": completion,
            "retirement_drain_cycles": self.now - completion,
            "total_cycles_including_runtime": self.now + self.sync_cycles,
            "stall_cycles": dict(self.stall_cycles),
            "stall_instruction_cycles": dict(self.stall_instruction_cycles),
            "queue_occupancy_timeline": self.snapshots,
            "instruction_pipeline": {
                tid: {
                    name: getattr(entry, name)
                    for name in (
                        "received",
                        "dispatched",
                        "wakeup",
                        "selected",
                        "issued",
                        "done",
                        "completed",
                        "retired",
                    )
                }
                for tid, entry in self.entries.items()
            },
            "resource_busy_cycles": dict(busy),
            "resource_utilization": {
                name: busy[name] / max(1, completion - start) / unit.count
                for name, unit in self.units.items()
            },
            "address_scoreboard_scope": "runtime_physical"
            if self.audit["runtime_submission_present"]
            else "compiler_logical",
            "address_hazards": list(self.hazards.values()),
            "address_hazard_count": len(self.hazards),
            "address_dependency_count": len(self.hazards),
            "partial_ready_event_count": len(self.feedback_ready),
            "partial_ready_dependency_count": sum(
                len(item.attributes.get("feedback_conditions", ()))
                for item in self.descriptors.values()
            ),
            "memory_bank_scoreboard": self.config.memory_bank_scoreboard,
            "priority_information": (
                "full_program_oracle"
                if self.config.dynamic_priority in {"critical_path", "oracle_critical_path"}
                else "compiler_descriptor_hint"
                if self.config.dynamic_priority == "compiler_hint"
                else "received_queue_age"
            ),
            "runtime_alias_dependency_count": sum(
                dependency.provenance.get("source") == "runtime_binding_alias"
                for descriptor in self.loaded.descriptors
                for dependency in descriptor.dependencies
            ),
        }
        for name, field in (
            ("rob_peak", "rob"),
            ("reception_queue_peak", "reception_queue"),
            ("inflight_tile_peak", "inflight_tiles"),
        ):
            metrics[name] = max((row[field] for row in self.snapshots), default=0)
        for name in ("wq", "iq", "fu"):
            metrics[name + "_peak"] = {
                unit: max((row[name][unit] for row in self.snapshots), default=0)
                for unit in self.units
            }
        if self.audit.get("timing_provider_coverage") is not None:
            metrics["timing_provider_coverage"] = self.audit[
                "timing_provider_coverage"
            ]
        backend_metrics = getattr(self.execution, "metrics", None)
        if callable(backend_metrics):
            metrics.update(backend_metrics())
        return SimulationResult(
            backend=self.audit["timing_provider_name"],
            policy=self.policy,
            graph_id=self.loaded.program_id,
            total_cycles=self.now + self.sync_cycles,
            timings=execution_timings,
            instruction_timings=tuple(
                self.instruction_timings[tid] for tid in self.instructions
            ),
            runtime_timings=tuple(self.runtime_timings),
            events=tuple(self.events),
            metrics=metrics,
        )


def schedule_loaded_cycle_program(
    loaded,
    machine,
    policy,
    execution,
    *,
    config=None,
    audit=None,
):
    """Run the clocked scheduler through the public loaded-device contract."""

    return _CycleScheduler(
        loaded,
        machine,
        policy,
        execution,
        config or SimulatorConfig(dynamic_priority="oldest_first"),
        dict(audit or {}),
    ).run()


def simulate_cycle_artifact(
    artifact,
    machine,
    policy="static_pipeline",
    *,
    timing_model=None,
    config=None,
    runtime_submission=None,
    execution_backend=None,
):
    if policy == "static_streams":
        from .device import DeviceSimulator

        return DeviceSimulator(
            artifact,
            machine,
            runtime_submission,
            timing_model,
        ).run("static_streams", model="cycle", config=config)
    timing_model = timing_model or AnalyticalTimingModel()
    config = config or SimulatorConfig(dynamic_priority="oldest_first")
    issues = (*artifact.validate(), *machine.validate())
    if runtime_submission is not None:
        issues += runtime_submission.validate(artifact.program)
        if runtime_submission.program_id != artifact.program.program_id:
            issues += ("runtime submission program does not match artifact",)
        if runtime_submission.artifact_id not in {None, artifact.artifact_id}:
            issues += ("runtime submission artifact does not match",)
    if hasattr(timing_model, "capabilities"):
        issues += validate_backend_capability(artifact, machine, timing_model)
    if issues:
        raise ValueError("; ".join(issues))
    loaded = (
        load_device_program(artifact, runtime_submission)
        if runtime_submission is not None
        else load_implicit_device_program(artifact)
    )
    execution = execution_backend or AnalyticalExecutionBackend(
        artifact, machine, timing_model
    )
    coverage = getattr(timing_model, "coverage", None)
    audit = {
        "timing_provider_name": timing_model.name,
        "timing_calibration_status": getattr(
            getattr(timing_model, "capabilities", None),
            "calibration_status",
            "analytical",
        ),
        "compile_package_sha256": hashlib.sha256(
            json.dumps(
                artifact.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "runtime_submission_present": runtime_submission is not None,
        "timing_provider_coverage": (
            dict(coverage(artifact.execution_graph.tasks))
            if callable(coverage)
            else None
        ),
    }
    return schedule_loaded_cycle_program(
        loaded,
        machine,
        policy,
        execution,
        config=config,
        audit=audit,
    )


__all__ = ["schedule_loaded_cycle_program", "simulate_cycle_artifact"]
