"""Cycle-stepped device scheduler with independently selectable payload timing."""

from dataclasses import dataclass, field

from .contracts import BackendCapabilities
from .registry import analytical_capabilities


@dataclass(frozen=True)
class CycleEventBackend:
    name: str = "cycle_event"
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            backend="cycle_event",
            supported_primitives=analytical_capabilities().supported_primitives,
            calibration_status="analytical",
            attributes={
                "scheduler": "cycle-stepped-v1",
                "scheduler_calibration": "uncalibrated",
            },
        )
    )

    def simulate(
        self,
        artifact,
        machine,
        policy,
        *,
        runtime_submission=None,
        timing_provider=None,
        simulator_config=None,
    ):
        from npu_ooo.simulator.device import DeviceSimulator

        issues = self.capabilities.validate_artifact(artifact, machine)
        if issues:
            raise ValueError("; ".join(issues))
        return DeviceSimulator(
            artifact,
            machine,
            runtime_submission,
            timing_provider,
        ).run(policy, model="cycle", config=simulator_config)
