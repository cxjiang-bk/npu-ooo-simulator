"""Configurable discrete-event backend for execution task graphs."""

from .core import (
    AnalyticalTimingModel,
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

__all__ = [
    "AnalyticalTimingModel",
    "AddressDependency",
    "AddressConflict",
    "AddressHazardKind",
    "AddressScoreboard",
    "SimulationResult",
    "SimulatorConfig",
    "StaticPipelineConfig",
    "TaskTiming",
    "TaskTimingSpec",
    "TimingTableModel",
    "TimingModel",
    "TraceEvent",
    "add_address_dependencies",
    "simulate_execution_graph",
]
