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

from .core import (
    AnalyticalTimingModel,
    SimulationResult,
    SimulatorConfig,
    TaskTiming,
    TraceEvent,
)
from .tisa import (
    _address_conflict,
    _critical_path_lengths,
    _memory_accesses,
    _memory_port_conflict,
    _payload_plan,
    _payload_readiness_task,
    _runtime_address_conflict,
    _tisa_address_observation,
    _tisa_dependency_details,
    _unit_map_matches,
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
    def __init__(self, artifact, machine, policy, timing_model, config, submission):
        if policy not in {"sequential", "static_pipeline", "dynamic_ready_queue"}:
            raise ValueError(f"unsupported scheduler policy '{policy}'")
        issues = (*artifact.validate(), *machine.validate())
        if submission is not None:
            issues += submission.validate(artifact.program)
            if submission.program_id != artifact.program.program_id:
                issues += ("runtime submission program does not match artifact",)
            if submission.artifact_id not in {None, artifact.artifact_id}:
                issues += ("runtime submission artifact does not match",)
        if issues:
            raise ValueError("; ".join(issues))
        self.config = config.resolved(machine)
        if self.config.static_pipeline is not None:
            raise ValueError(
                "primitive reservations are not supported by the TISA cycle scheduler"
            )
        self.pipe = self.config.pipeline
        if hasattr(timing_model, "capabilities"):
            issues = validate_backend_capability(artifact, machine, timing_model)
            if issues:
                raise ValueError("; ".join(issues))
        self.artifact, self.machine, self.policy = artifact, machine, policy
        self.provider, self.submission = timing_model, submission
        self.instructions = {
            item.tisa_id: item for item in artifact.program.instructions
        }
        self.program_order = {tid: index for index, tid in enumerate(self.instructions)}
        self.plans = {
            tid: _payload_plan(artifact, tid, machine, timing_model)
            for tid in self.instructions
        }
        self.entries = {
            tid: _Entry(tid, self.plans[tid].resource) for tid in self.instructions
        }
        self.units = {unit.name: unit for unit in machine.execution_units}
        self.runtime_operands = {tid: () for tid in self.instructions}
        if submission:
            self.runtime_operands = {
                tid: tuple(item for item in submission.operands if item.tisa_id == tid)
                for tid in self.instructions
            }
        self.accesses = {
            tid: _memory_accesses(item, machine, self.runtime_operands[tid])
            for tid, item in self.instructions.items()
        }
        self.ii = {}
        for tid, instruction in self.instructions.items():
            resource = self.entries[tid].resource
            if instruction.unit_map.quantity != 1 or not _unit_map_matches(
                instruction, resource
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
            intervals = [
                timing_model.timing(
                    artifact.execution_graph.task(task_id), machine
                ).initiation_interval_cycles
                for task_id in artifact.payloads[tid]
            ]
            if any(not math.isfinite(value) or value <= 0 for value in intervals):
                raise ValueError(
                    f"'{tid}' initiation intervals must be finite and positive"
                )
            self.ii[tid] = max(
                machine.unit(resource).initiation_interval_cycles, *intervals
            )

        self.priorities = _critical_path_lengths(
            artifact.program.instructions, self.plans
        )
        self.events, self.runtime_timings, self.snapshots = [], [], []
        self.timings, self.instruction_timings = {}, {}
        self.stall_cycles = Counter({reason: 0 for reason in STALL_REASONS})
        self.stall_instruction_cycles = Counter({reason: 0 for reason in STALL_REASONS})
        self.stall_seen, self.cycle_reasons = set(), set()
        self.hazards = {}
        self.stream = self._descriptor_stream()
        self.submission_order = {tid: i for i, (_, tid, _) in enumerate(self.stream)}
        self.next_receive = 0
        self.reception = deque()
        self.wq = {name: [] for name in self.units}
        self.iq = {name: [] for name in self.units}
        self.fu = {name: set() for name in self.units}
        self.running = {}
        self.next_issue = {
            (name, i): 0.0
            for name, unit in self.units.items()
            for i in range(unit.count)
        }
        self.rob = deque()
        self.completed, self.retired, self.issued = set(), set(), set()
        self.tile_remaining = Counter(
            item.tile_id for item in self.instructions.values()
        )
        self.active_tiles = set()
        self.feedback_ready = {}
        self.partial_offsets = {}
        for instruction in self.instructions.values():
            for dependency in instruction.dependencies:
                step = _payload_readiness_task(
                    dependency.condition, self.plans[dependency.source]
                )
                if step is not None:
                    self.partial_offsets[(dependency.source, dependency.condition)] = (
                        step.finish_offset
                    )
        self.now = 0
        self.progress = 0

    def _descriptor_stream(self):
        self.submit_cycles = self.submit_busy = self.request_wait = 0.0
        self.sync_cycles = (
            self.submission.synchronization_cycles if self.submission else 0.0
        )
        if self.submission is None:
            return [(0.0, tid, "implicit") for tid in self.instructions]
        stream = []
        for chunk in self.submission.commands:
            start = max(self.submit_cycles, chunk.availability_cycle)
            self.request_wait += start - self.submit_cycles
            finish = start + self.submission.launch_latency_cycles
            self.submit_busy += finish - start
            self.submit_cycles = finish
            stream.extend((finish, tid, chunk.chunk_id) for tid in chunk.tisa_ids)
            self.runtime_timings.append(
                TaskTiming(
                    chunk.chunk_id,
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
                "descriptor_count": len(chunk.tisa_ids),
                "runtime_policy": self.submission.policy,
            }
            self.events.extend(
                (
                    TraceEvent(
                        start,
                        "RUNTIME_SUBMIT_START",
                        chunk.chunk_id,
                        "Runtime/Submit",
                        details=details,
                    ),
                    TraceEvent(
                        finish,
                        "RUNTIME_SUBMIT_COMPLETE",
                        chunk.chunk_id,
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
                    "dependencies": _tisa_dependency_details(instruction),
                    **details,
                },
            )
        )

    def stall(self, tid, reason, stage, **details):
        key = (tid, reason)
        if key in self.stall_seen:
            return
        self.stall_seen.add(key)
        self.cycle_reasons.add(reason)
        self.stall_instruction_cycles[reason] += 1
        self.event("TISA_STALL", tid, reason=reason, stage=stage, **details)

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
        # Physical execution can finish before the completion bus accepts its result.
        for tid in tuple(self.running):
            entry = self.entries[tid]
            if entry.done <= self.now:
                self.running.pop(tid)
                self.event("TISA_EXECUTION_DONE", tid)
        due = [
            tid
            for tid in self.issued - self.completed
            if self.entries[tid].done + self.pipe.completion_latency <= self.now
        ]
        due.sort(key=lambda tid: (self.entries[tid].done, self.submission_order[tid]))
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
        # Partial readiness has its own modeled notification path; full results use the bus above.
        for key, offset in self.partial_offsets.items():
            source, condition = key
            issue = self.entries[source].issued
            if issue is None or key in self.feedback_ready:
                continue
            ready = math.ceil(issue + offset) + self.pipe.completion_latency
            if ready <= self.now:
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
                for dependency in self.instructions[tid].dependencies:
                    source = self.entries[dependency.source]
                    feedback = self.feedback_ready.get(
                        (dependency.source, dependency.condition), source.completed
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
            if older in self.completed:
                continue
            conflict = (
                _runtime_address_conflict(
                    self.runtime_operands[older], self.runtime_operands[tid]
                )
                if self.submission
                else _address_conflict(self.instructions[older], self.instructions[tid])
            )
            if conflict is not None:
                kind, region = conflict
                observation = _tisa_address_observation(
                    self.instructions[older], self.instructions[tid], kind, region
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
                    (item for _, item, _ in self.stream if item not in self.issued),
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
                conflict = _memory_port_conflict(
                    active_accesses, self.accesses[tid], self.machine
                )
                if conflict:
                    self.stall(
                        tid, "memory_bank_port_conflict", "issue", conflict=conflict
                    )
                    continue
            busy_instances = {
                self.entries[item].instance
                for item in self.running
                if self.entries[item].resource == resource
            }
            available = [
                i
                for i in range(self.units[resource].count)
                if i not in busy_instances
                and self.next_issue[(resource, i)] <= self.now
            ]
            if not available:
                self.stall(tid, "fu_busy", "issue")
                continue
            entry.instance, entry.issued = available[0], self.now
            entry.done = math.ceil(self.now + self.plans[tid].duration)
            self.next_issue[(resource, entry.instance)] = self.now + self.ii[tid]
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
            for step in self.plans[tid].steps:
                task = self.artifact.execution_graph.task(step.task_id)
                start, finish = (
                    self.now + step.start_offset,
                    self.now + step.finish_offset,
                )
                self.timings[task.task_id] = TaskTiming(
                    task.task_id,
                    resource,
                    entry.instance,
                    start,
                    start,
                    finish,
                    start,
                    start,
                )
                details = {
                    "primitive": task.primitive,
                    "parent_tisa_id": tid,
                    "operator_id": task.operator_id,
                    "tile_id": task.tile_id,
                    "dependencies": _tisa_dependency_details(self.instructions[tid]),
                }
                for timestamp, kind in (
                    (start, "ISSUE"),
                    (start, "START"),
                    (finish, "COMPLETE"),
                ):
                    self.events.append(
                        TraceEvent(
                            timestamp,
                            kind,
                            task.task_id,
                            resource,
                            entry.instance,
                            details,
                        )
                    )

    def _priority(self, tid):
        critical = (
            self.policy == "dynamic_ready_queue"
            and self.config.dynamic_priority == "critical_path"
        )
        return (-self.priorities[tid] if critical else 0, self.submission_order[tid])

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
                    for _, item, _ in self.stream[: self.submission_order[tid]]
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
        if any(cycle > self.now for cycle in self.next_issue.values()):
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
            for dep in self.instructions[entry.tisa_id].dependencies:
                feedback = self.feedback_ready.get(
                    (dep.source, dep.condition), self.entries[dep.source].completed
                )
                if (
                    feedback is not None
                    and feedback + self.pipe.wakeup_latency > self.now
                ):
                    return True
        return False

    def result(self):
        # Stable sort keeps the actual hardware phase order for equal timestamps.
        self.events.sort(key=lambda event: event.timestamp)
        start = min((item.issued for item in self.entries.values()), default=0)
        completion = max((item.completed for item in self.entries.values()), default=0)
        busy = Counter()
        for timing in self.timings.values():
            busy[timing.resource] += timing.duration
        timing_status = getattr(
            getattr(self.provider, "capabilities", None),
            "calibration_status",
            "analytical",
        )
        payload_hash = hashlib.sha256(
            json.dumps(
                self.artifact.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        metrics = {
            "scheduler_target": "tisa",
            "scheduler_model": "cycle-stepped-v1",
            "event_backend": "cycle_event",
            "backend": self.provider.name,
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
            "runtime_policy": self.submission.policy
            if self.submission
            else "implicit_static",
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
            if self.submission
            else "compiler_logical",
            "address_hazards": list(self.hazards.values()),
            "address_hazard_count": len(self.hazards),
            "address_dependency_count": len(self.hazards),
            "partial_ready_event_count": len(self.feedback_ready),
            "partial_ready_dependency_count": len(self.partial_offsets),
            "memory_bank_scoreboard": self.config.memory_bank_scoreboard,
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
        coverage = getattr(self.provider, "coverage", None)
        if callable(coverage):
            metrics["timing_provider_coverage"] = dict(
                coverage(self.artifact.execution_graph.tasks)
            )
        return SimulationResult(
            backend=self.provider.name,
            policy=self.policy,
            graph_id=self.artifact.program.program_id,
            total_cycles=self.now + self.sync_cycles,
            timings=tuple(
                self.timings[task.task_id]
                for task in self.artifact.execution_graph.tasks
            ),
            instruction_timings=tuple(
                self.instruction_timings[tid] for tid in self.instructions
            ),
            runtime_timings=tuple(self.runtime_timings),
            events=tuple(self.events),
            metrics=metrics,
        )


def simulate_cycle_artifact(
    artifact,
    machine,
    policy="static_pipeline",
    *,
    timing_model=None,
    config=None,
    runtime_submission=None,
):
    return _CycleScheduler(
        artifact,
        machine,
        policy,
        timing_model or AnalyticalTimingModel(),
        config or SimulatorConfig(),
        runtime_submission,
    ).run()
