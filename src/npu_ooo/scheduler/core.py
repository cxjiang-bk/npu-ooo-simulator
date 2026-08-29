from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from npu_ooo.simulator import (
    AnalyticalTimingModel,
    SimulationResult,
    SimulatorConfig,
    RuntimeSequenceSimulationResult,
    StaticPipelineConfig,
    TaskTiming,
    TimingModel,
    TraceEvent,
    simulate_execution_graph,
)
from npu_ooo.ir import BackendArtifact, ExecutionGraph, RuntimeSequence, RuntimeSubmission
from npu_ooo.arch import MachineConfig

if TYPE_CHECKING:
    from npu_ooo.backend import EventBackend


class SchedulerPolicy(str, Enum):
    SEQUENTIAL = "sequential"
    STATIC_PIPELINE = "static_pipeline"
    DYNAMIC_READY_QUEUE = "dynamic_ready_queue"


# Keep the original public name while making the backend explicit.
ScheduleResult = SimulationResult


def schedule_execution_graph(
    graph: ExecutionGraph,
    machine: MachineConfig,
    policy: SchedulerPolicy | str = SchedulerPolicy.STATIC_PIPELINE,
    *,
    timing_model: TimingModel | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> ScheduleResult:
    normalized_policy = policy.value if isinstance(policy, SchedulerPolicy) else str(policy)
    return simulate_execution_graph(
        graph,
        machine,
        normalized_policy,
        timing_model=timing_model or AnalyticalTimingModel(),
        config=simulator_config,
    )


def schedule_tisa_program(
    artifact: BackendArtifact,
    machine: MachineConfig,
    policy: SchedulerPolicy | str = SchedulerPolicy.STATIC_PIPELINE,
    *,
    timing_model: TimingModel | None = None,
    simulator_config: SimulatorConfig | None = None,
    runtime_submission: RuntimeSubmission | None = None,
    event_backend: EventBackend | None = None,
) -> ScheduleResult:
    """Schedule TISA descriptors, activating only their bound payload group."""

    normalized_policy = policy.value if isinstance(policy, SchedulerPolicy) else str(policy)
    if event_backend is None:
        from npu_ooo.backend import default_event_backend_registry

        event_backend = default_event_backend_registry().create("analytical_event")
    return event_backend.simulate(
        artifact,
        machine,
        normalized_policy,
        timing_provider=timing_model or AnalyticalTimingModel(),
        simulator_config=simulator_config,
        runtime_submission=runtime_submission,
    )


def schedule_tisa_sequence(
    artifact: BackendArtifact,
    sequence: RuntimeSequence,
    machine: MachineConfig,
    policy: SchedulerPolicy | str = SchedulerPolicy.STATIC_PIPELINE,
    *,
    timing_model: TimingModel | None = None,
    simulator_config: SimulatorConfig | None = None,
    event_backend: EventBackend | None = None,
) -> RuntimeSequenceSimulationResult:
    """Schedule a multi-invocation runtime sequence with stable state."""

    normalized_policy = policy.value if isinstance(policy, SchedulerPolicy) else str(policy)
    from npu_ooo.simulator import simulate_tisa_sequence

    return simulate_tisa_sequence(
        artifact,
        sequence,
        machine,
        normalized_policy,
        timing_model=timing_model or AnalyticalTimingModel(),
        config=simulator_config,
        event_backend=event_backend,
    )
