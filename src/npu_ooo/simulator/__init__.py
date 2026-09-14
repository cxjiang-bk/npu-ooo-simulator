"""Configurable discrete-event backend for execution task graphs."""

from .core import (
    AnalyticalTimingModel,
    RuntimeSequenceSimulationResult,
    SimulationResult,
    SimulatorConfig,
    StaticPipelineConfig,
    TaskTimingSpec,
    TimingTableModel,
    TimingModel,
    TraceEvent,
    TaskTiming,
    simulate_execution_graph,
)
from .address import AddressConflict, AddressDependency, AddressHazardKind, AddressScoreboard, add_address_dependencies
from .tisa import simulate_tisa_artifact, simulate_tisa_sequence

__all__ = [
    "AnalyticalTimingModel",
    "AddressDependency",
    "AddressConflict",
    "AddressHazardKind",
    "AddressScoreboard",
    "SimulationResult",
    "RuntimeSequenceSimulationResult",
    "SimulatorConfig",
    "StaticPipelineConfig",
    "TaskTiming",
    "TaskTimingSpec",
    "TimingTableModel",
    "TimingModel",
    "TraceEvent",
    "add_address_dependencies",
    "simulate_execution_graph",
    "simulate_tisa_artifact",
    "simulate_tisa_sequence",
]
