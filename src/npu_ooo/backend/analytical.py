"""Concrete event backend backed by the repository's analytical TISA engine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import BackendArtifact, RuntimeSubmission
from npu_ooo.simulator.core import AnalyticalTimingModel, SimulatorConfig, TimingModel

from .contracts import BackendCapabilities, TimingProvider, validate_backend_capability
from .registry import analytical_capabilities


@dataclass(frozen=True)
class AnalyticalEventBackend:
    """Run TISA scheduling with the configurable analytical event engine.

    This adapter owns the device-event boundary only.  Primitive latency is
    supplied independently through ``timing_provider`` and is therefore
    replaceable without changing the scheduler policy or compiled artifact.
    """

    name: str = "analytical_event"
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            backend="analytical_event",
            supported_primitives=analytical_capabilities().supported_primitives,
            calibration_status="analytical",
            attributes={
                "engine": "simulate_tisa_artifact",
                "resource_contract": "machine_config",
            },
        )
    )

    def simulate(
        self,
        artifact: BackendArtifact,
        machine: MachineConfig,
        policy: str,
        *,
        runtime_submission: RuntimeSubmission | None = None,
        timing_provider: TimingProvider | None = None,
        simulator_config: SimulatorConfig | None = None,
    ) -> Any:
        issues = validate_backend_capability(artifact, machine, self.capabilities)
        if issues:
            raise ValueError(
                "event backend capability validation failed: " + "; ".join(issues)
            )
        provider: TimingModel = timing_provider or AnalyticalTimingModel()
        # Import lazily to keep backend contracts and registries independent of
        # the simulator package import order.
        from npu_ooo.simulator.tisa import simulate_tisa_artifact

        result = simulate_tisa_artifact(
            artifact,
            machine,
            policy,
            timing_model=provider,
            config=simulator_config,
            runtime_submission=runtime_submission,
        )
        return replace(
            result,
            metrics={
                **result.metrics,
                "event_backend": self.name,
                "event_backend_capabilities": self.capabilities.to_dict(),
            },
        )


__all__ = ["AnalyticalEventBackend"]
