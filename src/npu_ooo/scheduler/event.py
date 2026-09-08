"""Event-driven scheduler over bound descriptors and an execution interface."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.execution import ExecutionBackend, IssueRequest
from npu_ooo.ir import LoadedDeviceProgram
from npu_ooo.simulator.core import (
    SimulationResult,
    SimulatorConfig,
    TaskTiming,
    TraceEvent,
)

from .semantics import (
    address_conflict,
    address_observation,
    critical_path_lengths,
    dependency_details,
    memory_accesses,
    memory_port_conflict,
    unit_map_matches,
)


def schedule_loaded_event_program(
    loaded: LoadedDeviceProgram,
    machine: MachineConfig,
    policy: str,
    execution: ExecutionBackend,
    *,
    config: SimulatorConfig | None = None,
    audit: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Run the event baseline from loaded descriptors and execution feedback."""

    if policy not in {"sequential", "static_pipeline", "dynamic_ready_queue"}:
        raise ValueError(f"unsupported scheduler policy '{policy}'")
    issues = (*loaded.validate(), *machine.validate())
    if issues:
        raise ValueError("; ".join(issues))
    config = (
        config or SimulatorConfig(dynamic_priority="oldest_first")
    ).resolved(machine)
    if config.static_pipeline is not None:
        raise ValueError("primitive reservations are not valid for the TISA scheduler")
    audit = dict(audit or {})
    descriptors = {
        item.instruction.tisa_id: item for item in loaded.descriptors
    }
    plans = {
        tisa_id: execution.estimate(descriptor.payload_handle)
        for tisa_id, descriptor in descriptors.items()
    }
    for tisa_id, descriptor in descriptors.items():
        if descriptor.instruction.unit_map.quantity != 1 or not unit_map_matches(
            descriptor, plans[tisa_id].resource
        ):
            raise ValueError(
                f"descriptor '{descriptor.descriptor_id}' UnitMap does not match payload"
            )
    order = {tisa_id: item.submission_order for tisa_id, item in descriptors.items()}
    program_order = {tisa_id: item.program_order for tisa_id, item in descriptors.items()}
    static_order = tuple(
        item.tisa_id
        for item in sorted(
            loaded.static_schedule.entries, key=lambda row: row.order
        )
    ) if loaded.static_schedule is not None else tuple(
        sorted(descriptors, key=program_order.__getitem__)
    )
    envelope_by_descriptor = {
        item.descriptor_id: item for item in loaded.envelopes
    }
    stream = tuple(
        (
            envelope_by_descriptor[descriptor.descriptor_id].arrival_cycle,
            descriptor.instruction.tisa_id,
            envelope_by_descriptor[descriptor.descriptor_id].chunk_id,
        )
        for descriptor in sorted(
            loaded.descriptors, key=lambda item: item.submission_order
        )
    )
    oracle_priority = config.dynamic_priority in {
        "critical_path",
        "oracle_critical_path",
    }
    priorities = (
        critical_path_lengths(tuple(loaded.descriptors), execution)
        if oracle_priority
        else {tisa_id: 0.0 for tisa_id in descriptors}
    )

    def compiler_hint_priority(tisa_id: str) -> float:
        hint = descriptors[tisa_id].instruction.attributes.get("scheduler_hint", {})
        return float(hint.get("priority", 0)) if isinstance(hint, Mapping) else 0.0
    accesses = {
        tisa_id: memory_accesses(descriptor, machine)
        for tisa_id, descriptor in descriptors.items()
    }
    queue_depth = config.instruction_queue_depth or len(descriptors) or 1
    dependency_window = config.dependency_window or queue_depth
    ready_queue_depth = config.ready_queue_depth or queue_depth
    rob_limit = config.rob_entries or len(descriptors) or 1
    tile_limit = config.max_inflight_tiles or len(descriptors) or 1
    tile_instruction_count = Counter(
        item.instruction.tile_id for item in loaded.descriptors
    )
    tile_completed_count = Counter()
    waiting: list[str] = []
    received: set[str] = set()
    issued: set[str] = set()
    completed: set[str] = set()
    completed_tokens: set[str] = set()
    partial_ready: dict[tuple[str, str], float] = {}
    completion_time: dict[str, float] = {}
    instruction_timings: dict[str, TaskTiming] = {}
    active: dict[str, tuple[str, int]] = {}
    active_accesses: dict[str, tuple] = {}
    inflight_tiles: set[str] = set()
    next_receive = 0
    rob_occupancy = 0
    now = 0.0
    events: list[TraceEvent] = []
    runtime_timings: list[TaskTiming] = []
    hazards: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    wq_peak = {unit.name: 0 for unit in machine.execution_units}

    chunks: dict[str, list] = {}
    for envelope in loaded.envelopes:
        chunks.setdefault(envelope.chunk_id, []).append(envelope)
    if loaded.attributes.get("command_chunk_count", 0):
        for chunk_id, envelopes in sorted(
            chunks.items(), key=lambda item: min(row.chunk_order for row in item[1])
        ):
            finish = max(item.arrival_cycle for item in envelopes)
            start = finish - loaded.launch_latency_cycles
            runtime_timings.append(
                TaskTiming(chunk_id, "Runtime/Submit", 0, start, start, finish, start, start)
            )
            details = {
                "runtime_policy": loaded.attributes.get("runtime_policy"),
                "descriptor_count": len(envelopes),
            }
            events.extend(
                (
                    TraceEvent(start, "RUNTIME_SUBMIT_START", chunk_id, "Runtime/Submit", details=details),
                    TraceEvent(finish, "RUNTIME_SUBMIT_COMPLETE", chunk_id, "Runtime/Submit", details=details),
                )
            )

    metrics: dict[str, Any] = {
        "backend": audit.get("timing_provider_name", execution.name),
        "policy": policy,
        "scheduler_target": "bound_tisa_descriptor",
        "scheduler_model": "event-reference-v2",
        "payload_execution": "run_to_completion",
        "runtime_policy": loaded.attributes.get("runtime_policy", "implicit_static"),
        "runtime_launch_count": int(loaded.attributes.get("command_chunk_count", 0)),
        "runtime_submit_cycles": float(loaded.attributes.get("runtime_submit_cycles", 0.0)),
        "runtime_submit_busy_cycles": len(runtime_timings) * loaded.launch_latency_cycles,
        "runtime_request_wait_cycles": max(
            0.0,
            float(loaded.attributes.get("runtime_submit_cycles", 0.0))
            - len(runtime_timings) * loaded.launch_latency_cycles,
        ),
        "runtime_synchronization_cycles": loaded.synchronization_cycles,
        "dynamic_index_bindings": list(audit.get("dynamic_index_bindings", ())),
        "address_scoreboard_scope": "runtime_physical",
        "address_dependency_count": 0,
        "address_hazard_count": 0,
        "address_hazards": [],
        "primitive_reordering_scope": "instruction_local",
        "simulator_config": config.to_dict(),
        "tisa_instruction_count": len(descriptors),
        "payload_task_count": sum(len(item.steps) for item in plans.values()),
        "readiness_interpreter": "completion_token+partial_ready",
        "readiness_conditions": sorted(
            {dependency.condition for item in loaded.descriptors for dependency in item.dependencies}
        ),
        "partial_ready_dependency_count": sum(
            dependency.condition.startswith("payload_ready:")
            for item in loaded.descriptors
            for dependency in item.dependencies
        ),
        "partial_ready_event_count": 0,
        "issued_instruction_count": 0,
        "completed_instruction_count": 0,
        "issued_task_count": 0,
        "completed_task_count": 0,
        "tisa_decision_count": 0,
        "rob_peak": 0,
        "inflight_tile_peak": 0,
        "reception_queue_peak": 0,
        "wq_peak": wq_peak,
        "dependency_block_events": 0,
        "resource_block_events": 0,
        "rob_block_events": 0,
        "tile_window_block_events": 0,
        "address_scoreboard_block_events": 0,
        "memory_bank_scoreboard": config.memory_bank_scoreboard,
        "memory_bank_block_events": 0,
        "resource_busy_cycles": {},
        "queue_occupancy_timeline": [],
        "machine_calibration_status": machine.attributes.get(
            "calibration_status", "unspecified"
        ),
        "timing_calibration_status": audit.get(
            "timing_calibration_status", "unspecified"
        ),
        "calibration_status": audit.get("timing_calibration_status", "analytical"),
        "compile_package_sha256": audit.get("compile_package_sha256"),
        "scheduler_visibility": "loaded_descriptors_only",
        "priority_information": (
            "full_program_oracle"
            if oracle_priority
            else "compiler_descriptor_hint"
            if config.dynamic_priority == "compiler_hint"
            else "received_queue_age"
        ),
        "runtime_alias_dependency_count": sum(
            dependency.provenance.get("source") == "runtime_binding_alias"
            for descriptor in loaded.descriptors
            for dependency in descriptor.dependencies
        ),
    }
    if audit.get("timing_provider_coverage") is not None:
        metrics["timing_provider_coverage"] = audit["timing_provider_coverage"]

    def record_occupancy(event: str) -> None:
        metrics["queue_occupancy_timeline"].append(
            {
                "timestamp": now,
                "event": event,
                "reception_queue": len(waiting),
                "rob": rob_occupancy,
                "inflight_tiles": len(inflight_tiles),
                "wq": {
                    resource: sum(plans[item].resource == resource for item in waiting)
                    for resource in wq_peak
                },
            }
        )

    def receive() -> None:
        nonlocal next_receive
        while next_receive < len(stream) and len(waiting) < queue_depth:
            arrival, tisa_id, chunk_id = stream[next_receive]
            if arrival > now + 1e-9:
                break
            resource = plans[tisa_id].resource
            resource_waiting = sum(plans[item].resource == resource for item in waiting)
            capacity = machine.unit(resource).queue_depth * machine.unit(resource).count
            if resource_waiting >= capacity:
                break
            waiting.append(tisa_id)
            received.add(tisa_id)
            next_receive += 1
            wq_peak[resource] = max(wq_peak[resource], resource_waiting + 1)
            metrics["reception_queue_peak"] = max(
                metrics["reception_queue_peak"], len(waiting)
            )
            descriptor = descriptors[tisa_id]
            events.append(
                TraceEvent(
                    now,
                    "TISA_RECEIVE",
                    tisa_id,
                    f"TISA/{resource}",
                    details={
                        "op_type": descriptor.instruction.op_type,
                        "operator_id": descriptor.instruction.operator_id,
                        "tile_id": descriptor.instruction.tile_id,
                        "runtime_chunk_id": chunk_id,
                        "runtime_ready_cycle": arrival,
                        "dependencies": dependency_details(descriptor),
                        "runtime_operands": [item.to_dict() for item in descriptor.operands],
                    },
                )
            )

    def dependency_ready(tisa_id: str) -> bool:
        return all(
            dependency.source.token_id in completed_tokens
            or (
                (dependency.source.tisa_id, dependency.condition) in partial_ready
                and partial_ready[(dependency.source.tisa_id, dependency.condition)] <= now
            )
            for dependency in descriptors[tisa_id].dependencies
        )

    def dependency_ready_cycle(tisa_id: str) -> float:
        values = []
        for dependency in descriptors[tisa_id].dependencies:
            partial = partial_ready.get((dependency.source.tisa_id, dependency.condition))
            if partial is not None:
                values.append(partial)
            elif dependency.source.token_id in completed_tokens:
                values.append(completion_time[dependency.source.tisa_id])
            else:
                return math.inf
        return max(values, default=0.0)

    def blocked_by_address(tisa_id: str):
        if not config.address_scoreboard:
            return None
        older_received = [
            item
            for item in received - completed
            if program_order[item] < program_order[tisa_id]
        ]
        for active_id in sorted(older_received, key=program_order.__getitem__):
            conflict = address_conflict(descriptors[active_id], descriptors[tisa_id])
            if conflict is not None:
                kind, region = conflict
                return active_id, address_observation(
                    descriptors[active_id], descriptors[tisa_id], kind, region
                )
        return None

    receive()
    record_occupancy("INIT")
    while len(completed) < len(descriptors):
        for feedback in execution.advance(now):
            tisa_id = loaded.descriptor(feedback.descriptor_id).instruction.tisa_id
            if feedback.kind == "partial_ready":
                partial_ready[(tisa_id, feedback.condition)] = feedback.cycle
                events.append(
                    TraceEvent(
                        feedback.cycle,
                        "TISA_PARTIAL_READY",
                        tisa_id,
                        f"TISA/{feedback.resource}",
                        feedback.instance,
                        {"condition": feedback.condition, "ready_cycle": feedback.cycle},
                    )
                )
                metrics["partial_ready_event_count"] += 1
                continue
            descriptor = descriptors[tisa_id]
            completed.add(tisa_id)
            completed_tokens.add(descriptor.completion_token.token_id)
            completion_time[tisa_id] = feedback.cycle
            active.pop(tisa_id, None)
            active_accesses.pop(tisa_id, None)
            rob_occupancy -= 1
            tile = descriptor.instruction.tile_id
            tile_completed_count[tile] += 1
            if tile_completed_count[tile] == tile_instruction_count[tile]:
                inflight_tiles.discard(tile)
            events.append(
                TraceEvent(
                    feedback.cycle,
                    "TISA_COMPLETE",
                    tisa_id,
                    f"TISA/{feedback.resource}",
                    feedback.instance,
                    {
                        "op_type": descriptor.instruction.op_type,
                        "operator_id": descriptor.instruction.operator_id,
                        "tile_id": tile,
                        "payload_task_count": len(plans[tisa_id].steps),
                        "dependencies": dependency_details(descriptor),
                        "runtime_operands": [item.to_dict() for item in descriptor.operands],
                    },
                )
            )
            metrics["completed_instruction_count"] += 1
            metrics["completed_task_count"] += len(plans[tisa_id].steps)
            receive()
            record_occupancy("TISA_COMPLETE")

        receive()
        issued_at_now = 0
        while True:
            if policy == "sequential" and active:
                break
            if rob_occupancy >= rob_limit:
                metrics["rob_block_events"] += 1
                break
            visible = sorted(waiting, key=order.__getitem__)[:dependency_window]
            visible = visible[:ready_queue_depth]
            if policy in {"sequential", "static_pipeline"}:
                next_static = next(
                    (item for item in static_order if item not in issued), None
                )
                visible = [next_static] if next_static in waiting else []
            if not visible:
                break
            candidates = []
            dependency_blocked = resource_blocked = tile_blocked = False
            address_blocked = bank_blocked = False
            for tisa_id in visible:
                descriptor = descriptors[tisa_id]
                if not dependency_ready(tisa_id):
                    dependency_blocked = True
                    continue
                tile = descriptor.instruction.tile_id
                if tile not in inflight_tiles and len(inflight_tiles) >= tile_limit:
                    tile_blocked = True
                    continue
                conflict = blocked_by_address(tisa_id)
                if conflict is not None:
                    address_blocked = True
                    _active, observation = conflict
                    key = (
                        observation["predecessor"],
                        observation["successor"],
                        observation["kind"],
                        observation["tensor"],
                        observation["memory"],
                    )
                    hazards.setdefault(key, observation)
                    continue
                if config.memory_bank_scoreboard:
                    conflict = memory_port_conflict(
                        active_accesses, accesses[tisa_id], machine
                    )
                    if conflict is not None:
                        bank_blocked = True
                        continue
                accepted, _reason = execution.can_accept(descriptor, now)
                if not accepted:
                    resource_blocked = True
                    continue
                candidates.append(tisa_id)
            if not candidates:
                metrics["dependency_block_events"] += int(dependency_blocked)
                metrics["resource_block_events"] += int(resource_blocked)
                metrics["tile_window_block_events"] += int(tile_blocked)
                metrics["address_scoreboard_block_events"] += int(address_blocked)
                metrics["memory_bank_block_events"] += int(bank_blocked)
                break
            if policy == "dynamic_ready_queue":
                if config.dynamic_priority in {"critical_path", "oracle_critical_path"}:
                    selected = min(candidates, key=lambda item: (-priorities[item], order[item]))
                elif config.dynamic_priority == "compiler_hint":
                    selected = min(
                        candidates,
                        key=lambda item: (
                            -compiler_hint_priority(item),
                            order[item],
                        ),
                    )
                else:
                    selected = min(candidates, key=order.__getitem__)
            else:
                selected = candidates[0]
            descriptor = descriptors[selected]
            receipt = execution.issue(IssueRequest(descriptor, now))
            if not receipt.accepted:
                metrics["resource_block_events"] += 1
                break
            waiting.remove(selected)
            issued.add(selected)
            active[selected] = (str(receipt.resource), int(receipt.instance))
            active_accesses[selected] = accesses[selected]
            rob_occupancy += 1
            inflight_tiles.add(descriptor.instruction.tile_id)
            instruction_timings[selected] = TaskTiming(
                selected,
                f"TISA/{receipt.resource}",
                int(receipt.instance),
                now,
                now,
                float(receipt.expected_done_cycle),
                dependency_ready_cycle(selected),
                now,
            )
            events.append(
                TraceEvent(
                    now,
                    "TISA_ISSUE",
                    selected,
                    f"TISA/{receipt.resource}",
                    int(receipt.instance),
                    {
                        "op_type": descriptor.instruction.op_type,
                        "operator_id": descriptor.instruction.operator_id,
                        "tile_id": descriptor.instruction.tile_id,
                        "unit_map": descriptor.instruction.unit_map.unit,
                        "payload_task_count": len(plans[selected].steps),
                        "dependencies": dependency_details(descriptor),
                        "runtime_operands": [item.to_dict() for item in descriptor.operands],
                    },
                )
            )
            metrics["issued_instruction_count"] += 1
            metrics["issued_task_count"] += len(plans[selected].steps)
            metrics["tisa_decision_count"] += 1
            metrics["rob_peak"] = max(metrics["rob_peak"], rob_occupancy)
            metrics["inflight_tile_peak"] = max(
                metrics["inflight_tile_peak"], len(inflight_tiles)
            )
            issued_at_now += 1
            receive()
            record_occupancy("TISA_ISSUE")
            if issued_at_now > sum(unit.count for unit in machine.execution_units):
                break

        if len(completed) == len(descriptors):
            break
        future = [
            value
            for value in (
                execution.next_feedback_cycle(),
                stream[next_receive][0] if next_receive < len(stream) else None,
                *(
                    execution.next_accept_cycle(descriptors[item])
                    for item in waiting
                ),
            )
            if value is not None and value > now + 1e-9
        ]
        if not future:
            unresolved = sorted(waiting, key=order.__getitem__)
            next_static = next(
                (item for item in static_order if item not in issued), None
            )
            next_stream = (
                stream[next_receive] if next_receive < len(stream) else None
            )
            unmet = {
                item: [
                    dependency.source.token_id
                    for dependency in descriptors[item].dependencies
                    if dependency.source.token_id not in completed_tokens
                    and (dependency.source.tisa_id, dependency.condition)
                    not in partial_ready
                ]
                for item in unresolved
            }
            raise RuntimeError(
                f"event scheduler deadlocked at cycle {now}; waiting={unresolved}, "
                f"received={len(received)}, issued={len(issued)}, completed={len(completed)}, "
                f"next_static={next_static}, next_stream={next_stream}, rob={rob_occupancy}, "
                f"unmet={unmet}"
            )
        now = min(future)

    execution_timings = tuple(execution.task_timings())
    events.extend(execution.trace_events())
    event_order = {
        "RUNTIME_SUBMIT_START": 0,
        "RUNTIME_SUBMIT_COMPLETE": 1,
        "TISA_RECEIVE": 2,
        "TISA_ISSUE": 4,
        "ISSUE": 5,
        "START": 6,
        "COMPLETE": 7,
        "TISA_PARTIAL_READY": 8,
        "TISA_COMPLETE": 9,
    }
    events.sort(
        key=lambda event: (
            event.timestamp,
            event_order.get(event.event, 10),
            event.task_id,
            event.instance,
        )
    )
    device_finish = max(completion_time.values(), default=0.0)
    device_start = min(
        (item.issue for item in instruction_timings.values()), default=0.0
    )
    device_cycles = max(0.0, device_finish - device_start)
    busy = Counter()
    for timing in execution_timings:
        busy[timing.resource] += timing.duration
    metrics["resource_busy_cycles"] = dict(busy)
    metrics["resource_utilization"] = {
        unit.name: (
            busy[unit.name] / device_cycles / unit.count if device_cycles else 0.0
        )
        for unit in machine.execution_units
    }
    metrics["completed_tile_count"] = len(tile_instruction_count)
    metrics["device_start_cycle"] = device_start
    metrics["device_finish_cycle"] = device_finish
    metrics["device_cycles"] = device_cycles
    metrics["total_cycles_including_runtime"] = (
        device_finish + loaded.synchronization_cycles
    )
    metrics["address_dependency_count"] = len(hazards)
    metrics["address_hazard_count"] = len(hazards)
    metrics["address_hazards"] = [hazards[key] for key in sorted(hazards)]
    backend_metrics = getattr(execution, "metrics", None)
    if callable(backend_metrics):
        metrics.update(backend_metrics())
    return SimulationResult(
        backend=str(audit.get("timing_provider_name", execution.name)),
        policy=policy,
        graph_id=loaded.program_id,
        total_cycles=device_finish + loaded.synchronization_cycles,
        timings=execution_timings,
        instruction_timings=tuple(
            instruction_timings[item.instruction.tisa_id]
            for item in sorted(loaded.descriptors, key=lambda row: row.program_order)
        ),
        runtime_timings=tuple(runtime_timings),
        events=tuple(events),
        metrics=metrics,
    )


__all__ = ["schedule_loaded_event_program"]
