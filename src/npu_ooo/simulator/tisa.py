from __future__ import annotations

"""Hardware-like scheduling over TISA descriptors and bound backend payloads."""

from dataclasses import dataclass
import heapq
import math
from typing import Any

from npu_ooo.arch import ExecutionUnitConfig, MachineConfig
from npu_ooo.backend import validate_backend_capability
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    RuntimeOperandBinding,
    RuntimeSubmission,
    TISAInstruction,
    TISAOperand,
)

from .core import (
    AnalyticalTimingModel,
    SimulationResult,
    SimulatorConfig,
    TaskTiming,
    TimingModel,
    TraceEvent,
)


@dataclass(frozen=True)
class _PayloadStep:
    task_id: str
    start_offset: float
    finish_offset: float


@dataclass(frozen=True)
class _PayloadPlan:
    resource: str
    steps: tuple[_PayloadStep, ...]
    duration: float


@dataclass
class _UnitState:
    unit: ExecutionUnitConfig
    instance: int
    busy_until: float = 0.0

    def available(self, now: float) -> bool:
        return self.busy_until <= now + 1e-9


@dataclass(frozen=True)
class _MemoryAccess:
    memory: str
    bank: int
    mode: str


def _memory_accesses(
    instruction: TISAInstruction,
    machine: MachineConfig,
    runtime_operands: tuple[RuntimeOperandBinding, ...] = (),
) -> tuple[_MemoryAccess, ...]:
    """Map one TISA instruction's ranges to configured memory bank ports."""

    accesses: set[_MemoryAccess] = set()
    if runtime_operands:
        candidates = tuple(
            # Runtime bindings carry the concrete device address.  Use it for
            # bank selection so separate buffers with different base addresses
            # can exercise different banks even when their tile offsets match.
            (item.physical_scope, item.address, item.size_bytes, item.access_type)
            for item in runtime_operands
        )
    else:
        candidates = tuple(
            (
                operand.tile_mem.scope,
                operand.tile_mem.offset_bytes or 0,
                operand.tile_mem.size_bytes or 1,
                operand.normalized_access,
            )
            for operand in instruction.operands
        )
    for memory, offset, size, access_type in candidates:
        try:
            level = machine.memory(memory)
        except KeyError:
            # ``logical`` scopes are resolved by RuntimeSubmission later.
            continue
        bank_width = level.bank_width_bytes or 1
        first_bank = offset // bank_width
        last_bank = (offset + max(size - 1, 0)) // bank_width
        modes = (
            ("read", "write")
            if access_type == AccessType.READ_WRITE.value
            else ("read",)
            if access_type == AccessType.READ.value
            else ("write",)
        )
        for bank in range(first_bank, last_bank + 1):
            for mode in modes:
                accesses.add(_MemoryAccess(memory, bank % level.bank_count, mode))
    return tuple(sorted(accesses, key=lambda item: (item.memory, item.bank, item.mode)))


def _memory_port_conflict(
    active_accesses: dict[str, tuple[_MemoryAccess, ...]],
    candidate: tuple[_MemoryAccess, ...],
    machine: MachineConfig,
) -> tuple[str, int, str] | None:
    active_counts: dict[tuple[str, int, str], int] = {}
    for accesses in active_accesses.values():
        for access in accesses:
            key = (access.memory, access.bank, access.mode)
            active_counts[key] = active_counts.get(key, 0) + 1
    for access in candidate:
        level = machine.memory(access.memory)
        limit = level.read_ports if access.mode == "read" else level.write_ports
        if active_counts.get((access.memory, access.bank, access.mode), 0) >= limit:
            return access.memory, access.bank, access.mode
    return None


def _payload_plan(
    artifact: BackendArtifact,
    tisa_id: str,
    machine: MachineConfig,
    timing_model: TimingModel,
) -> _PayloadPlan:
    task_ids = artifact.payloads[tisa_id]
    tasks = {task_id: artifact.execution_graph.task(task_id) for task_id in task_ids}
    resources = {task.resource for task in tasks.values()}
    if len(resources) != 1:
        raise ValueError(
            f"TISA payload '{tisa_id}' must use exactly one backend resource"
        )
    resource = next(iter(resources))
    try:
        machine.unit(resource)
    except KeyError as exc:
        raise ValueError(
            f"TISA payload '{tisa_id}' references unknown machine resource '{resource}'"
        ) from exc

    remaining = {
        task_id: {predecessor for predecessor in task.predecessors if predecessor in tasks}
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
            raise ValueError(f"TISA payload '{tisa_id}' contains a local dependency cycle")
        selected = min(
            ready,
            key=lambda task_id: (tasks[task_id].program_order, task_id),
        )
        ordered.append(selected)

    offset = 0.0
    steps: list[_PayloadStep] = []
    for task_id in ordered:
        duration = timing_model.timing(tasks[task_id], machine).duration_cycles
        steps.append(_PayloadStep(task_id, offset, offset + duration))
        offset += duration
    return _PayloadPlan(resource=resource, steps=tuple(steps), duration=offset)


def _unit_map_matches(instruction: TISAInstruction, resource: str) -> bool:
    requested = instruction.unit_map.unit.lower()
    physical = resource.lower()
    if requested == physical:
        return True
    aliases = {
        "dma": {"dma", "gdma", "ldma", "de", "copy"},
        "tensor": {"mxu", "me", "tensor", "matrix"},
        "vector": {"aru", "ve", "vector", "vu"},
        "scalar": {"scalar", "cpu"},
    }
    return physical in aliases.get(requested, set())


def _reads(operand: TISAOperand) -> bool:
    return operand.normalized_access in {
        AccessType.READ.value,
        AccessType.READ_WRITE.value,
    }


def _writes(operand: TISAOperand) -> bool:
    return operand.normalized_access in {
        AccessType.WRITE.value,
        AccessType.READ_WRITE.value,
    }


def _overlaps(left: TISAOperand, right: TISAOperand) -> bool:
    left_mem = left.tile_mem
    right_mem = right.tile_mem
    left_base = left_mem.tensor or left_mem.base
    right_base = right_mem.tensor or right_mem.base
    if left_base != right_base or left_mem.scope != right_mem.scope:
        return False
    if (
        left_mem.offset_bytes is None
        or right_mem.offset_bytes is None
        or left_mem.size_bytes is None
        or right_mem.size_bytes is None
    ):
        return True
    return (
        left_mem.offset_bytes < right_mem.offset_bytes + right_mem.size_bytes
        and right_mem.offset_bytes < left_mem.offset_bytes + left_mem.size_bytes
    )


def _address_conflict(
    active: TISAInstruction,
    candidate: TISAInstruction,
) -> str | None:
    for left in active.operands:
        for right in candidate.operands:
            if not _overlaps(left, right):
                continue
            if _writes(left) and _reads(right):
                return "RAW"
            if _reads(left) and _writes(right):
                return "WAR"
            if _writes(left) and _writes(right):
                return "WAW"
    return None


def _runtime_reads(operand: RuntimeOperandBinding) -> bool:
    return operand.access_type in {
        AccessType.READ.value,
        AccessType.READ_WRITE.value,
    }


def _runtime_writes(operand: RuntimeOperandBinding) -> bool:
    return operand.access_type in {
        AccessType.WRITE.value,
        AccessType.READ_WRITE.value,
    }


def _runtime_address_conflict(
    active: tuple[RuntimeOperandBinding, ...],
    candidate: tuple[RuntimeOperandBinding, ...],
) -> str | None:
    for left in active:
        for right in candidate:
            if left.physical_scope != right.physical_scope:
                continue
            if not (
                left.address < right.address + right.size_bytes
                and right.address < left.address + left.size_bytes
            ):
                continue
            if _runtime_writes(left) and _runtime_reads(right):
                return "RAW"
            if _runtime_reads(left) and _runtime_writes(right):
                return "WAR"
            if _runtime_writes(left) and _runtime_writes(right):
                return "WAW"
    return None


def _critical_path_lengths(
    instructions: tuple[TISAInstruction, ...],
    plans: dict[str, _PayloadPlan],
) -> dict[str, float]:
    successors = {instruction.tisa_id: [] for instruction in instructions}
    for instruction in instructions:
        for dependency in instruction.dependencies:
            successors[dependency.source].append(instruction.tisa_id)
    lengths: dict[str, float] = {}
    for instruction in reversed(instructions):
        lengths[instruction.tisa_id] = plans[instruction.tisa_id].duration + max(
            (lengths[successor] for successor in successors[instruction.tisa_id]),
            default=0.0,
        )
    return lengths


def _payload_readiness_task(condition: str, plan: _PayloadPlan) -> _PayloadStep | None:
    """Resolve the opt-in partial-ready condition against one TISA payload.

    ``payload_ready:<task_id>`` means that a dependent instruction may use the
    source after the named backend step finishes.  All compiler-generated
    conditions remain completion-boundary conditions; this syntax is reserved
    for calibrated backends and simulator micro-tests.
    """

    prefix = "payload_ready:"
    if not condition.startswith(prefix):
        return None
    task_id = condition[len(prefix) :]
    if not task_id:
        raise ValueError("payload_ready condition must name a backend task")
    for step in plan.steps:
        if step.task_id == task_id:
            return step
    raise ValueError(
        f"payload_ready condition references task '{task_id}' outside the source payload"
    )


def simulate_tisa_artifact(
    artifact: BackendArtifact,
    machine: MachineConfig,
    policy: str = "static_pipeline",
    *,
    timing_model: TimingModel | None = None,
    config: SimulatorConfig | None = None,
    runtime_submission: RuntimeSubmission | None = None,
) -> SimulationResult:
    """Schedule TISA instructions and execute each bound payload atomically.

    Payload primitives are expanded only after their parent TISA instruction is
    issued. They run in local program order on one concrete EU instance, so no
    primitive can enter the global out-of-order window independently.
    """

    supported_policies = {"sequential", "static_pipeline", "dynamic_ready_queue"}
    if policy not in supported_policies:
        raise ValueError(f"unsupported scheduler policy '{policy}'")
    artifact_issues = artifact.validate()
    machine_issues = machine.validate()
    if artifact_issues or machine_issues:
        raise ValueError("; ".join((*artifact_issues, *machine_issues)))
    if runtime_submission is not None:
        submission_issues = runtime_submission.validate(artifact.program)
        if runtime_submission.artifact_id not in {None, artifact.artifact_id}:
            submission_issues = (
                *submission_issues,
                f"runtime submission artifact '{runtime_submission.artifact_id}' does not match "
                f"'{artifact.artifact_id}'",
            )
        if submission_issues:
            raise ValueError("; ".join(submission_issues))
    timing_model = timing_model or AnalyticalTimingModel()
    timing_calibration_status = getattr(
        getattr(timing_model, "capabilities", None),
        "calibration_status",
        "unspecified",
    )
    if not isinstance(timing_calibration_status, str) or not timing_calibration_status:
        timing_calibration_status = "unspecified"
    machine_calibration_status = machine.attributes.get(
        "calibration_status", "unspecified"
    )
    capability_issues = validate_backend_capability(artifact, machine, timing_model) \
        if hasattr(timing_model, "capabilities") else ()
    if capability_issues:
        raise ValueError("backend capability validation failed: " + "; ".join(capability_issues))
    config = (config or SimulatorConfig()).resolved(machine)
    if config.static_pipeline is not None:
        raise ValueError(
            "primitive StaticPipelineConfig reservations are not valid for the TISA scheduler"
        )

    instructions = artifact.program.instructions
    instruction_by_id = {
        instruction.tisa_id: instruction for instruction in instructions
    }
    if runtime_submission is None:
        descriptor_stream = tuple(
            (0.0, instruction.tisa_id, "implicit.chunk0000")
            for instruction in instructions
        )
        runtime_chunks = ()
        runtime_policy = "implicit_static"
        runtime_submit_cycles = 0.0
        synchronization_cycles = 0.0
    else:
        runtime_chunks = runtime_submission.commands
        runtime_policy = runtime_submission.policy
        launch_latency = runtime_submission.launch_latency_cycles
        runtime_chunk_windows: dict[str, tuple[float, float]] = {}
        runtime_cursor = 0.0
        runtime_request_wait_cycles = 0.0
        for chunk in runtime_chunks:
            start = max(runtime_cursor, chunk.availability_cycle)
            runtime_request_wait_cycles += start - runtime_cursor
            finish = start + launch_latency
            runtime_chunk_windows[chunk.chunk_id] = (start, finish)
            runtime_cursor = finish
        descriptor_stream = tuple(
            (
                runtime_chunk_windows[chunk.chunk_id][1],
                tisa_id,
                chunk.chunk_id,
            )
            for chunk in runtime_chunks
            for tisa_id in chunk.tisa_ids
        )
        runtime_submit_cycles = runtime_cursor
        runtime_submit_busy_cycles = len(runtime_chunks) * launch_latency
        synchronization_cycles = runtime_submission.synchronization_cycles
    order = {
        tisa_id: index
        for index, (_ready, tisa_id, _chunk_id) in enumerate(descriptor_stream)
    }
    plans = {
        instruction.tisa_id: _payload_plan(
            artifact, instruction.tisa_id, machine, timing_model
        )
        for instruction in instructions
    }
    for instruction in instructions:
        if instruction.unit_map.quantity != 1:
            raise ValueError(
                f"TISA instruction '{instruction.tisa_id}' requests quantity="
                f"{instruction.unit_map.quantity}; the analytical backend currently supports one EU"
            )
        if not _unit_map_matches(instruction, plans[instruction.tisa_id].resource):
            raise ValueError(
                f"TISA instruction '{instruction.tisa_id}' UnitMap '{instruction.unit_map.unit}' "
                f"does not match payload resource '{plans[instruction.tisa_id].resource}'"
            )

    critical_path = _critical_path_lengths(instructions, plans)
    resources = {
        unit.name: [_UnitState(unit, instance) for instance in range(unit.count)]
        for unit in machine.execution_units
    }
    dependency_specs = {
        instruction.tisa_id: tuple(instruction.dependencies)
        for instruction in instructions
    }
    readiness_dependencies: dict[str, list[Any]] = {}
    for instruction in instructions:
        for dependency in instruction.dependencies:
            if dependency.condition.startswith("payload_ready:"):
                readiness_dependencies.setdefault(dependency.source, []).append(dependency)
    runtime_operands_by_tisa: dict[str, tuple[RuntimeOperandBinding, ...]] = {}
    if runtime_submission is not None:
        for instruction in instructions:
            runtime_operands_by_tisa[instruction.tisa_id] = tuple(
                operand
                for operand in runtime_submission.operands
                if operand.tisa_id == instruction.tisa_id
            )
    tile_instruction_count: dict[str, int] = {}
    tile_completed_count: dict[str, int] = {}
    for instruction in instructions:
        tile_instruction_count[instruction.tile_id] = (
            tile_instruction_count.get(instruction.tile_id, 0) + 1
        )
        tile_completed_count[instruction.tile_id] = 0

    waiting: list[str] = []
    received: set[str] = set()
    issued: set[str] = set()
    completed: set[str] = set()
    active: dict[str, tuple[str, int]] = {}
    active_memory_accesses: dict[str, tuple[_MemoryAccess, ...]] = {}
    completion_time: dict[str, float] = {}
    payload_readiness_time: dict[tuple[str, str], float] = {}
    emitted_payload_readiness: set[tuple[str, str]] = set()
    instruction_timings: dict[str, TaskTiming] = {}
    primitive_timings: dict[str, TaskTiming] = {}
    runtime_timings: list[TaskTiming] = []
    events: list[TraceEvent] = []
    if runtime_submission is not None:
        for chunk in runtime_chunks:
            start, finish = runtime_chunk_windows[chunk.chunk_id]
            details = {
                "runtime_policy": runtime_submission.policy,
                "queue": chunk.queue,
                "submission_order": chunk.submission_order,
                "descriptor_count": len(chunk.tisa_ids),
                "availability_cycle": chunk.availability_cycle,
            }
            runtime_timings.append(
                TaskTiming(
                    task_id=chunk.chunk_id,
                    resource="Runtime/Submit",
                    instance=0,
                    issue=start,
                    start=start,
                    finish=finish,
                    dependency_ready=start,
                    resource_ready=start,
                )
            )
            events.extend(
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
    event_queue: list[tuple[float, int, str, str, int]] = []
    event_serial = 0
    next_receive = 0
    now = 0.0
    rob_occupancy = 0
    inflight_tiles: set[str] = set()
    queue_depth = config.instruction_queue_depth or len(instructions) or 1
    dependency_window = config.dependency_window or queue_depth
    ready_queue_depth = config.ready_queue_depth or queue_depth
    rob_limit = config.rob_entries or len(instructions) or 1
    tile_limit = config.max_inflight_tiles or len(tile_instruction_count) or 1
    wq_peak: dict[str, int] = {unit.name: 0 for unit in machine.execution_units}
    metrics: dict[str, Any] = {
        "backend": timing_model.name,
        "policy": policy,
        "scheduler_target": "tisa",
        "payload_execution": "run_to_completion",
        "runtime_policy": runtime_policy,
        "runtime_launch_count": len(runtime_chunks),
        "runtime_submit_cycles": runtime_submit_cycles,
        "runtime_submit_busy_cycles": (
            runtime_submit_busy_cycles if runtime_submission is not None else 0.0
        ),
        "runtime_request_wait_cycles": (
            runtime_request_wait_cycles if runtime_submission is not None else 0.0
        ),
        "runtime_synchronization_cycles": synchronization_cycles,
        "address_scoreboard_scope": (
            "runtime_physical" if runtime_submission is not None else "compiler_logical"
        ),
        "primitive_reordering_scope": "instruction_local",
        "simulator_config": config.to_dict(),
        "tisa_instruction_count": len(instructions),
        "payload_task_count": len(artifact.execution_graph.tasks),
        "readiness_interpreter": "completion_boundary+payload_ready",
        "readiness_conditions": sorted(
            {
                dependency.condition
                for instruction in instructions
                for dependency in instruction.dependencies
            }
        ),
        "partial_ready_dependency_count": sum(
            dependency.condition.startswith("payload_ready:")
            for instruction in instructions
            for dependency in instruction.dependencies
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
        "machine_calibration_status": machine_calibration_status,
        "timing_calibration_status": timing_calibration_status,
        "calibration_status": (
            timing_calibration_status
            if timing_calibration_status != "unspecified"
            else machine_calibration_status
        ),
    }
    coverage = getattr(timing_model, "coverage", None)
    if callable(coverage):
        metrics["timing_provider_coverage"] = dict(
            coverage(artifact.execution_graph.tasks)
        )

    def record_occupancy(event: str) -> None:
        metrics["queue_occupancy_timeline"].append(
            {
                "timestamp": now,
                "event": event,
                "reception_queue": len(waiting),
                "rob": rob_occupancy,
                "inflight_tiles": len(inflight_tiles),
                "wq": {
                    resource: sum(
                        plans[tisa_id].resource == resource for tisa_id in waiting
                    )
                    for resource in resources
                },
            }
        )

    def receive_descriptors() -> None:
        nonlocal next_receive
        while next_receive < len(descriptor_stream) and len(waiting) < queue_depth:
            ready_cycle, tisa_id, chunk_id = descriptor_stream[next_receive]
            if ready_cycle > now + 1e-9:
                break
            instruction = instruction_by_id[tisa_id]
            resource = plans[instruction.tisa_id].resource
            resource_waiting = sum(
                plans[tisa_id].resource == resource for tisa_id in waiting
            )
            wq_capacity = machine.unit(resource).queue_depth * machine.unit(resource).count
            if resource_waiting >= wq_capacity:
                break
            waiting.append(instruction.tisa_id)
            received.add(instruction.tisa_id)
            next_receive += 1
            wq_peak[resource] = max(wq_peak[resource], resource_waiting + 1)
            metrics["reception_queue_peak"] = max(
                metrics["reception_queue_peak"], len(waiting)
            )
            details = {
                "op_type": instruction.op_type,
                "operator_id": instruction.operator_id,
                "tile_id": instruction.tile_id,
                "unit_map": instruction.unit_map.unit,
                "runtime_chunk_id": chunk_id,
                "runtime_ready_cycle": ready_cycle,
            }
            events.append(
                TraceEvent(
                    now,
                    "TISA_RECEIVE",
                    instruction.tisa_id,
                    f"TISA/{resource}",
                    details=details,
                )
            )

    def dependency_ready(tisa_id: str) -> bool:
        for dependency in dependency_specs[tisa_id]:
            if dependency.source in completed:
                continue
            ready_time = payload_readiness_time.get(
                (dependency.source, dependency.condition)
            )
            if ready_time is None or ready_time > now + 1e-9:
                return False
        return True

    def dependency_ready_time(tisa_id: str) -> float:
        times: list[float] = []
        for dependency in dependency_specs[tisa_id]:
            partial_time = payload_readiness_time.get(
                (dependency.source, dependency.condition)
            )
            if partial_time is not None:
                times.append(partial_time)
            elif dependency.source in completed:
                times.append(completion_time[dependency.source])
            else:
                return math.inf
        return max(times, default=0.0)

    def resource_candidate(tisa_id: str) -> _UnitState | None:
        resource = plans[tisa_id].resource
        available = [state for state in resources[resource] if state.available(now)]
        if not available:
            return None
        return min(available, key=lambda state: state.instance)

    def address_block(tisa_id: str) -> tuple[str, str] | None:
        if not config.address_scoreboard:
            return None
        candidate = instruction_by_id[tisa_id]
        for active_id in sorted(active, key=order.__getitem__):
            if runtime_submission is None:
                kind = _address_conflict(instruction_by_id[active_id], candidate)
            else:
                kind = _runtime_address_conflict(
                    runtime_operands_by_tisa[active_id],
                    runtime_operands_by_tisa[tisa_id],
                )
            if kind is not None:
                return active_id, kind
        return None

    memory_accesses_by_tisa = {
        instruction.tisa_id: _memory_accesses(
            instruction,
            machine,
            runtime_operands_by_tisa.get(instruction.tisa_id, ()),
        )
        for instruction in instructions
    }

    def memory_block(tisa_id: str) -> tuple[str, int, str] | None:
        if not config.memory_bank_scoreboard:
            return None
        return _memory_port_conflict(
            active_memory_accesses,
            memory_accesses_by_tisa[tisa_id],
            machine,
        )

    receive_descriptors()
    record_occupancy("INIT")

    while len(completed) < len(instructions):
        while event_queue and event_queue[0][0] <= now + 1e-9:
            finish, _serial, tisa_id, resource, instance = heapq.heappop(event_queue)
            instruction = instruction_by_id[tisa_id]
            completed.add(tisa_id)
            completion_time[tisa_id] = finish
            active.pop(tisa_id)
            active_memory_accesses.pop(tisa_id, None)
            rob_occupancy -= 1
            tile_completed_count[instruction.tile_id] += 1
            if (
                tile_completed_count[instruction.tile_id]
                == tile_instruction_count[instruction.tile_id]
            ):
                inflight_tiles.discard(instruction.tile_id)
            details = {
                "op_type": instruction.op_type,
                "operator_id": instruction.operator_id,
                "tile_id": instruction.tile_id,
                "payload_task_count": len(artifact.payloads[tisa_id]),
            }
            events.append(
                TraceEvent(
                    finish,
                    "TISA_COMPLETE",
                    tisa_id,
                    f"TISA/{resource}",
                    instance,
                    details,
                )
            )
            metrics["completed_instruction_count"] += 1
            metrics["completed_task_count"] += len(artifact.payloads[tisa_id])
            receive_descriptors()
            record_occupancy("TISA_COMPLETE")

        for (source, condition), ready_time in sorted(payload_readiness_time.items()):
            if ready_time > now + 1e-9 or (source, condition) in emitted_payload_readiness:
                continue
            source_resource = plans[source].resource
            source_instance = active.get(source, (source_resource, 0))[1]
            events.append(
                TraceEvent(
                    now,
                    "TISA_PARTIAL_READY",
                    source,
                    f"TISA/{source_resource}",
                    source_instance,
                    details={"condition": condition, "ready_cycle": ready_time},
                )
            )
            emitted_payload_readiness.add((source, condition))
            metrics["partial_ready_event_count"] += 1

        receive_descriptors()

        issued_at_now = 0
        while True:
            if policy == "sequential" and rob_occupancy:
                break
            if rob_occupancy >= rob_limit:
                metrics["rob_block_events"] += 1
                break
            visible = sorted(waiting, key=order.__getitem__)[:dependency_window]
            visible = visible[:ready_queue_depth]
            if policy in {"sequential", "static_pipeline"}:
                visible = visible[:1]
            if not visible:
                break

            candidates: list[tuple[str, _UnitState]] = []
            dependency_blocked = False
            resource_blocked = False
            tile_blocked = False
            address_blocked = False
            memory_bank_blocked = False
            for tisa_id in visible:
                instruction = instruction_by_id[tisa_id]
                if not dependency_ready(tisa_id):
                    dependency_blocked = True
                    continue
                if (
                    instruction.tile_id not in inflight_tiles
                    and len(inflight_tiles) >= tile_limit
                ):
                    tile_blocked = True
                    continue
                conflict = address_block(tisa_id)
                if conflict is not None:
                    address_blocked = True
                    continue
                memory_conflict = memory_block(tisa_id)
                if memory_conflict is not None:
                    memory_bank_blocked = True
                    continue
                resource = resource_candidate(tisa_id)
                if resource is None:
                    resource_blocked = True
                    continue
                candidates.append((tisa_id, resource))

            if not candidates:
                if dependency_blocked:
                    metrics["dependency_block_events"] += 1
                if resource_blocked:
                    metrics["resource_block_events"] += 1
                if tile_blocked:
                    metrics["tile_window_block_events"] += 1
                if address_blocked:
                    metrics["address_scoreboard_block_events"] += 1
                if memory_bank_blocked:
                    metrics["memory_bank_block_events"] += 1
                break

            if policy == "dynamic_ready_queue":
                if config.dynamic_priority == "critical_path":
                    tisa_id, resource_state = min(
                        candidates,
                        key=lambda item: (
                            -critical_path[item[0]],
                            order[item[0]],
                            item[1].instance,
                        ),
                    )
                else:
                    tisa_id, resource_state = min(
                        candidates,
                        key=lambda item: (order[item[0]], item[1].instance),
                    )
            else:
                tisa_id, resource_state = candidates[0]

            instruction = instruction_by_id[tisa_id]
            plan = plans[tisa_id]
            finish = now + plan.duration
            resource_state.busy_until = finish
            waiting.remove(tisa_id)
            issued.add(tisa_id)
            active[tisa_id] = (plan.resource, resource_state.instance)
            active_memory_accesses[tisa_id] = memory_accesses_by_tisa[tisa_id]
            rob_occupancy += 1
            inflight_tiles.add(instruction.tile_id)
            for dependency in readiness_dependencies.get(tisa_id, ()):
                step = _payload_readiness_task(dependency.condition, plan)
                if step is not None:
                    payload_readiness_time[(tisa_id, dependency.condition)] = (
                        now + step.finish_offset
                    )
            dependency_ready_cycle = max(
                dependency_ready_time(tisa_id),
                0.0,
            )
            instruction_timings[tisa_id] = TaskTiming(
                task_id=tisa_id,
                resource=f"TISA/{plan.resource}",
                instance=resource_state.instance,
                issue=now,
                start=now,
                finish=finish,
                dependency_ready=dependency_ready_cycle,
                resource_ready=now,
            )
            details = {
                "primitive": "tisa_instruction",
                "op_type": instruction.op_type,
                "operator_id": instruction.operator_id,
                "tile_id": instruction.tile_id,
                "unit_map": instruction.unit_map.unit,
                "payload_task_count": len(plan.steps),
            }
            events.append(
                TraceEvent(
                    now,
                    "TISA_ISSUE",
                    tisa_id,
                    f"TISA/{plan.resource}",
                    resource_state.instance,
                    details,
                )
            )

            for step in plan.steps:
                task = artifact.execution_graph.task(step.task_id)
                start = now + step.start_offset
                task_finish = now + step.finish_offset
                task_details = {
                    "primitive": task.primitive,
                    "operator_id": task.operator_id,
                    "tile_id": task.tile_id,
                    "parent_tisa_id": tisa_id,
                    "backend": timing_model.name,
                }
                primitive_timings[task.task_id] = TaskTiming(
                    task_id=task.task_id,
                    resource=task.resource,
                    instance=resource_state.instance,
                    issue=start,
                    start=start,
                    finish=task_finish,
                    dependency_ready=start,
                    resource_ready=start,
                )
                events.extend(
                    (
                        TraceEvent(
                            start,
                            "ISSUE",
                            task.task_id,
                            task.resource,
                            resource_state.instance,
                            task_details,
                        ),
                        TraceEvent(
                            start,
                            "START",
                            task.task_id,
                            task.resource,
                            resource_state.instance,
                            task_details,
                        ),
                        TraceEvent(
                            task_finish,
                            "COMPLETE",
                            task.task_id,
                            task.resource,
                            resource_state.instance,
                            task_details,
                        ),
                    )
                )
                metrics["resource_busy_cycles"][task.resource] = (
                    metrics["resource_busy_cycles"].get(task.resource, 0.0)
                    + step.finish_offset
                    - step.start_offset
                )

            event_serial += 1
            heapq.heappush(
                event_queue,
                (
                    finish,
                    event_serial,
                    tisa_id,
                    plan.resource,
                    resource_state.instance,
                ),
            )
            metrics["issued_instruction_count"] += 1
            metrics["issued_task_count"] += len(plan.steps)
            metrics["tisa_decision_count"] += 1
            metrics["rob_peak"] = max(metrics["rob_peak"], rob_occupancy)
            metrics["inflight_tile_peak"] = max(
                metrics["inflight_tile_peak"], len(inflight_tiles)
            )
            issued_at_now += 1
            receive_descriptors()
            record_occupancy("TISA_ISSUE")
            if issued_at_now > sum(unit.count for unit in machine.execution_units):
                break

        if len(completed) >= len(instructions):
            break
        next_descriptor_time = (
            descriptor_stream[next_receive][0]
            if next_receive < len(descriptor_stream)
            else None
        )
        if not event_queue and next_descriptor_time is not None and next_descriptor_time > now + 1e-9:
            now = next_descriptor_time
            continue
        if not event_queue:
            unresolved = sorted(waiting, key=order.__getitem__)
            raise RuntimeError(
                f"TISA scheduler deadlocked at cycle {now}; waiting={unresolved}, "
                f"received={len(received)}, issued={len(issued)}, completed={len(completed)}"
            )
        next_time = event_queue[0][0]
        future_readiness = [
            ready_time
            for (source, _condition), ready_time in payload_readiness_time.items()
            if source in active and ready_time > now + 1e-9
        ]
        if future_readiness:
            next_time = min(next_time, min(future_readiness))
        if next_descriptor_time is not None and next_descriptor_time > now + 1e-9:
            next_time = min(next_time, next_descriptor_time)
        if next_time <= now + 1e-9:
            raise RuntimeError("TISA simulator made no time progress")
        now = next_time

    event_order = {
        "RUNTIME_SUBMIT_START": 0,
        "RUNTIME_SUBMIT_COMPLETE": 1,
        "TISA_RECEIVE": 2,
        "TISA_ISSUE": 3,
        "ISSUE": 4,
        "START": 5,
        "COMPLETE": 6,
        "TISA_PARTIAL_READY": 7,
        "TISA_COMPLETE": 8,
    }
    events.sort(
        key=lambda event: (
            event.timestamp,
            event_order[event.event],
            event.task_id,
            event.instance,
        )
    )
    device_finish_cycle = max(
        (timing.finish for timing in instruction_timings.values()), default=0.0
    )
    device_start_cycle = min(
        (timing.issue for timing in instruction_timings.values()), default=0.0
    )
    device_cycles = max(0.0, device_finish_cycle - device_start_cycle)
    total_cycles = device_finish_cycle + synchronization_cycles
    metrics["resource_utilization"] = {
        unit.name: (
            metrics["resource_busy_cycles"].get(unit.name, 0.0)
            / (device_cycles * unit.count)
            if device_cycles > 0
            else 0.0
        )
        for unit in machine.execution_units
    }
    metrics["completed_tile_count"] = len(tile_instruction_count)
    metrics["device_start_cycle"] = device_start_cycle
    metrics["device_finish_cycle"] = device_finish_cycle
    metrics["device_cycles"] = device_cycles
    metrics["total_cycles_including_runtime"] = total_cycles
    return SimulationResult(
        backend=timing_model.name,
        policy=policy,
        graph_id=artifact.program.program_id,
        total_cycles=total_cycles,
        timings=tuple(
            primitive_timings[task.task_id]
            for task in artifact.execution_graph.tasks
        ),
        instruction_timings=tuple(
            instruction_timings[instruction.tisa_id]
            for instruction in instructions
        ),
        runtime_timings=tuple(runtime_timings),
        events=tuple(events),
        metrics=metrics,
    )
