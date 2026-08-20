from __future__ import annotations

from enum import Enum

from npu_ooo.simulator import (
    AnalyticalTimingModel,
    SimulationResult,
    SimulatorConfig,
    StaticPipelineConfig,
    TaskTiming,
    TimingModel,
    TraceEvent,
    simulate_execution_graph,
)
from npu_ooo.ir import ExecutionGraph
from npu_ooo.arch import MachineConfig


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
