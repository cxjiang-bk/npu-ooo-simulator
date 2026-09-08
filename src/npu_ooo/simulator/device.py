"""Device-side orchestration over explicit loader, scheduler and execution APIs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.execution import AnalyticalExecutionBackend, ExecutionBackend
from npu_ooo.ir import BackendArtifact, LoadedDeviceProgram, RuntimeSubmission
from npu_ooo.runtime import load_device_program, load_implicit_device_program
from npu_ooo.scheduler.event import schedule_loaded_event_program

from .core import AnalyticalTimingModel, SimulationResult, SimulatorConfig, TimingModel


ExecutionFactory = Callable[
    [BackendArtifact, MachineConfig, TimingModel], ExecutionBackend
]


@dataclass(frozen=True)
class DeviceSimulator:
    """Connect a loaded descriptor stream to a scheduler and execution backend.

    Compiler artifacts are consumed only at this orchestration boundary.  The
    scheduler receives ``LoadedDeviceProgram`` descriptors; payload lookup,
    execution-unit availability and completion feedback remain owned by the
    execution backend.
    """

    artifact: BackendArtifact
    machine: MachineConfig
    runtime_submission: RuntimeSubmission | None = None
    timing_model: TimingModel | None = None
    execution_factory: ExecutionFactory = AnalyticalExecutionBackend

    def _prepare(self) -> tuple[LoadedDeviceProgram, ExecutionBackend, dict[str, Any]]:
        timing = self.timing_model or AnalyticalTimingModel()
        issues = (*self.artifact.validate(), *self.machine.validate())
        if hasattr(timing, "capabilities"):
            from npu_ooo.backend.contracts import validate_backend_capability

            issues += validate_backend_capability(
                self.artifact, self.machine, timing
            )
        if self.runtime_submission is not None:
            issues += self.runtime_submission.validate(self.artifact.program)
            if self.runtime_submission.program_id != self.artifact.program.program_id:
                issues += ("runtime submission program does not match artifact",)
            if self.runtime_submission.artifact_id not in {
                None,
                self.artifact.artifact_id,
            }:
                issues += ("runtime submission artifact does not match",)
        if issues:
            raise ValueError("; ".join(issues))
        loaded = (
            load_device_program(self.artifact, self.runtime_submission)
            if self.runtime_submission is not None
            else load_implicit_device_program(self.artifact)
        )
        execution = self.execution_factory(self.artifact, self.machine, timing)
        coverage = getattr(timing, "coverage", None)
        audit = {
            "timing_provider_name": timing.name,
            "timing_calibration_status": getattr(
                getattr(timing, "capabilities", None),
                "calibration_status",
                "analytical",
            ),
            "compile_package_sha256": hashlib.sha256(
                json.dumps(
                    self.artifact.to_dict(), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "runtime_submission_present": self.runtime_submission is not None,
            "dynamic_index_bindings": [
                item.to_dict()
                for item in (
                    self.runtime_submission.dynamic_indices
                    if self.runtime_submission is not None
                    else ()
                )
            ],
            "timing_provider_coverage": (
                dict(coverage(self.artifact.execution_graph.tasks))
                if callable(coverage)
                else None
            ),
        }
        return loaded, execution, audit

    def run(
        self,
        policy: str,
        *,
        model: str = "event",
        config: SimulatorConfig | None = None,
        audit: Mapping[str, Any] | None = None,
    ) -> SimulationResult:
        loaded, execution, default_audit = self._prepare()
        merged_audit = {**default_audit, **dict(audit or {})}
        if model == "event":
            return schedule_loaded_event_program(
                loaded,
                self.machine,
                policy,
                execution,
                config=config,
                audit=merged_audit,
            )
        if model == "cycle":
            from .cycle import schedule_loaded_cycle_program

            return schedule_loaded_cycle_program(
                loaded,
                self.machine,
                policy,
                execution,
                config=config,
                audit=merged_audit,
            )
        raise ValueError(f"unsupported device simulation model '{model}'")


__all__ = ["DeviceSimulator", "ExecutionFactory"]
