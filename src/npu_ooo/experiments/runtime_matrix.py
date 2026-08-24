from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import BackendArtifact, BufferBinding, RuntimeSubmission, create_runtime_submission
from npu_ooo.scheduler import SchedulerPolicy, schedule_tisa_program
from npu_ooo.simulator import SimulationResult, SimulatorConfig, TimingModel


@dataclass(frozen=True)
class RuntimeDeviceCase:
    """One cell in a runtime-policy by device-policy experiment matrix."""

    runtime_policy: str
    device_policy: str
    submission: RuntimeSubmission
    result: SimulationResult

    @property
    def case_id(self) -> str:
        return f"runtime-{self.runtime_policy}__device-{self.device_policy}"

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "runtime_policy": self.runtime_policy,
            "device_policy": self.device_policy,
            "program_id": self.submission.program_id,
            "artifact_id": self.submission.artifact_id,
            "runtime_command_chunk_count": len(self.submission.commands),
            "runtime_submit_cycles": self.result.metrics.get("runtime_submit_cycles", 0.0),
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
    timing_model: TimingModel | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> tuple[RuntimeDeviceCase, ...]:
    """Run policy combinations without recompiling or reallocating buffers."""

    normalized_buffers = tuple(buffers)
    cases: list[RuntimeDeviceCase] = []
    for runtime_policy in runtime_policies:
        submission = create_runtime_submission(
            artifact,
            normalized_buffers,
            submission_id=f"submission.{artifact.program.program_id}.{runtime_policy}",
            policy=runtime_policy,
            chunk_size=chunk_size,
            launch_latency_cycles=launch_latency_cycles,
            synchronization_cycles=synchronization_cycles,
        )
        for device_policy in device_policies:
            result = schedule_tisa_program(
                artifact,
                machine,
                device_policy,
                timing_model=timing_model,
                simulator_config=simulator_config,
                runtime_submission=submission,
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
