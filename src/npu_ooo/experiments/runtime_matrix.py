from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from npu_ooo.arch import MachineConfig
from npu_ooo.backend import EventBackend
from npu_ooo.ir import (
    BackendArtifact,
    BufferBinding,
    RuntimeSequence,
    RuntimeSubmission,
    create_runtime_sequence,
    create_runtime_state_registry,
    create_runtime_submission,
)
from npu_ooo.scheduler import SchedulerPolicy, schedule_tisa_program, schedule_tisa_sequence
from npu_ooo.simulator import (
    RuntimeSequenceSimulationResult,
    SimulationResult,
    SimulatorConfig,
    TimingModel,
)


@dataclass(frozen=True)
class RuntimeDeviceCase:
    """One cell in a runtime-policy by device-policy experiment matrix."""

    runtime_policy: str
    device_policy: str
    submission: RuntimeSubmission | RuntimeSequence
    result: SimulationResult | RuntimeSequenceSimulationResult

    @property
    def case_id(self) -> str:
        return f"runtime-{self.runtime_policy}__device-{self.device_policy}"

    def to_dict(self) -> dict[str, object]:
        if isinstance(self.submission, RuntimeSequence):
            invocations = self.submission.invocations
            program_id = self.submission.program_id
            artifact_id = self.submission.artifact_id
            command_count = sum(len(item.commands) for item in invocations)
        else:
            invocations = (self.submission,)
            program_id = self.submission.program_id
            artifact_id = self.submission.artifact_id
            command_count = len(self.submission.commands)
        return {
            "case_id": self.case_id,
            "runtime_policy": self.runtime_policy,
            "device_policy": self.device_policy,
            "program_id": program_id,
            "artifact_id": artifact_id,
            "request_count": len(invocations),
            "runtime_command_chunk_count": command_count,
            "runtime_submit_cycles": self.result.metrics.get("runtime_submit_cycles", 0.0),
            "runtime_submit_busy_cycles": self.result.metrics.get(
                "runtime_submit_busy_cycles", 0.0
            ),
            "runtime_request_wait_cycles": self.result.metrics.get(
                "runtime_request_wait_cycles", 0.0
            ),
            "runtime_synchronization_cycles": self.result.metrics.get(
                "runtime_synchronization_cycles", 0.0
            ),
            "device_start_cycle": self.result.metrics.get("device_start_cycle", 0.0),
            "device_finish_cycle": self.result.metrics.get(
                "device_finish_cycle", self.result.total_cycles
            ),
            "device_cycles": self.result.metrics.get(
                "device_cycles", self.result.total_cycles
            ),
            "total_cycles": self.result.total_cycles,
        }


def run_runtime_device_matrix(
    artifact: BackendArtifact,
    buffers: Iterable[BufferBinding],
    machine: MachineConfig,
    *,
    runtime_policies: Sequence[str] = ("static", "dynamic_ready_queue"),
    device_policies: Sequence[str | SchedulerPolicy] = (
        SchedulerPolicy.STATIC_PIPELINE,
        SchedulerPolicy.DYNAMIC_READY_QUEUE,
    ),
    chunk_size: int | None = None,
    launch_latency_cycles: float = 0.0,
    synchronization_cycles: float = 0.0,
    descriptor_available_cycles: Mapping[str, float] | None = None,
    timing_model: TimingModel | None = None,
    simulator_config: SimulatorConfig | None = None,
    event_backend: EventBackend | None = None,
    request_count: int = 1,
    inter_request_gap_cycles: float = 0.0,
) -> tuple[RuntimeDeviceCase, ...]:
    """Run policy combinations without recompiling or reallocating buffers."""

    normalized_buffers = tuple(buffers)
    if isinstance(request_count, bool) or not isinstance(request_count, int) or request_count <= 0:
        raise ValueError("request_count must be a positive integer")
    if inter_request_gap_cycles < 0:
        raise ValueError("inter_request_gap_cycles must be non-negative")
    cases: list[RuntimeDeviceCase] = []
    for runtime_policy in runtime_policies:
        if request_count == 1:
            submission: RuntimeSubmission | RuntimeSequence = create_runtime_submission(
                artifact,
                normalized_buffers,
                submission_id=f"submission.{artifact.program.program_id}.{runtime_policy}",
                policy=runtime_policy,
                chunk_size=chunk_size,
                launch_latency_cycles=launch_latency_cycles,
                synchronization_cycles=synchronization_cycles,
                descriptor_available_cycles=descriptor_available_cycles,
            )
        else:
            state_registry = create_runtime_state_registry(artifact, normalized_buffers)
            submission = create_runtime_sequence(
                artifact,
                state_registry,
                invocation_count=request_count,
                sequence_id=f"requests.{artifact.program.program_id}.{runtime_policy}",
                policy=runtime_policy,
                chunk_size=chunk_size,
                launch_latency_cycles=launch_latency_cycles,
                synchronization_cycles=synchronization_cycles,
                descriptor_available_cycles=descriptor_available_cycles,
                inter_invocation_gap_cycles=inter_request_gap_cycles,
            )
        for device_policy in device_policies:
            if isinstance(submission, RuntimeSequence):
                result = schedule_tisa_sequence(
                    artifact,
                    submission,
                    machine,
                    device_policy,
                    timing_model=timing_model,
                    simulator_config=simulator_config,
                    event_backend=event_backend,
                )
            else:
                result = schedule_tisa_program(
                    artifact,
                    machine,
                    device_policy,
                    timing_model=timing_model,
                    simulator_config=simulator_config,
                    runtime_submission=submission,
                    event_backend=event_backend,
                )
            cases.append(
                RuntimeDeviceCase(
                    runtime_policy=runtime_policy,
                    device_policy=result.policy,
                    submission=submission,
                    result=result,
                )
            )
    return tuple(cases)
