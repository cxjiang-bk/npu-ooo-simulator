"""Hot-pluggable compiler/runtime/device backend contracts.

The contracts in this module are deliberately small.  They describe what a
backend must promise without forcing the analytical simulator, an external
MXU model, or an RTL adapter to share implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    BackendArtifact,
    OperatorGraph,
    RuntimeSubmission,
    ScheduleSpec,
    TISAProgram,
    TileGraph,
)
if TYPE_CHECKING:
    from npu_ooo.simulator.core import SimulationResult, SimulatorConfig, TaskTimingSpec


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared operations/resources/memory features of one backend."""

    backend: str
    supported_primitives: frozenset[str] = frozenset()
    supported_resources: frozenset[str] = frozenset()
    supported_memories: frozenset[str] = frozenset()
    calibration_status: str = "unspecified"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate_artifact(
        self,
        artifact: BackendArtifact,
        machine: MachineConfig,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if self.supported_primitives:
            unsupported_primitives = sorted(
                {
                    task.primitive
                    for task in artifact.execution_graph.tasks
                    if task.primitive not in self.supported_primitives
                }
            )
            if unsupported_primitives:
                issues.append(
                    f"backend '{self.backend}' does not support primitive(s): "
                    + ", ".join(unsupported_primitives)
                )
        if self.supported_resources:
            unsupported_resources = sorted(
                {
                    task.resource
                    for task in artifact.execution_graph.tasks
                    if task.resource not in self.supported_resources
                }
            )
            if unsupported_resources:
                issues.append(
                    f"backend '{self.backend}' does not support resource(s): "
                    + ", ".join(unsupported_resources)
                )
        if self.supported_memories:
            memories = {
                region.memory
                for task in artifact.execution_graph.tasks
                for region in (*task.reads, *task.writes)
            }
            unsupported_memories = sorted(memories - self.supported_memories)
            if unsupported_memories:
                issues.append(
                    f"backend '{self.backend}' does not support memory level(s): "
                    + ", ".join(unsupported_memories)
                )
        machine_resources = {unit.name for unit in machine.execution_units}
        if self.supported_resources and not machine_resources & self.supported_resources:
            issues.append(
                f"backend '{self.backend}' has no compatible execution unit in machine "
                f"'{machine.config_id}'"
            )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "supported_primitives": sorted(self.supported_primitives),
            "supported_resources": sorted(self.supported_resources),
            "supported_memories": sorted(self.supported_memories),
            "calibration_status": self.calibration_status,
            "attributes": dict(self.attributes),
        }


class TimingProvider(Protocol):
    """Task timing source consumed by the common scheduler."""

    name: str
    capabilities: BackendCapabilities

    def timing(self, task: Any, machine: MachineConfig) -> Any:
        ...


class CodegenBackend(Protocol):
    """Lower one semantic TISA program to a target execution artifact."""

    name: str
    capabilities: BackendCapabilities

    def lower(
        self,
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        tile_graph: TileGraph,
        machine: MachineConfig,
        *,
        program: TISAProgram,
    ) -> BackendArtifact:
        ...


class EventBackend(Protocol):
    """Device event engine; scheduler policy remains an input, not a backend choice."""

    name: str
    capabilities: BackendCapabilities

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
        ...


class SystemBackend(Protocol):
    """Optional host/device system model around RuntimeSubmission."""

    name: str
    capabilities: BackendCapabilities

    def submit(
        self,
        submission: RuntimeSubmission,
        machine: MachineConfig,
    ) -> Mapping[str, Any]:
        ...


def validate_backend_capability(
    artifact: BackendArtifact,
    machine: MachineConfig,
    backend: BackendCapabilities | TimingProvider | CodegenBackend | EventBackend | SystemBackend,
) -> tuple[str, ...]:
    """Validate a declared backend capability before simulation/codegen."""

    capabilities = getattr(backend, "capabilities", backend)
    if not isinstance(capabilities, BackendCapabilities):
        raise TypeError("backend must expose BackendCapabilities as 'capabilities'")
    return capabilities.validate_artifact(artifact, machine)


__all__ = [
    "BackendCapabilities",
    "CodegenBackend",
    "EventBackend",
    "SystemBackend",
    "TimingProvider",
    "validate_backend_capability",
]
