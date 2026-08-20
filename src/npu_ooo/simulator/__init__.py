"""Configurable discrete-event backend for execution task graphs."""

from .core import (
    AnalyticalTimingModel,
    SimulationResult,
    SimulatorConfig,
    TaskTimingSpec,
    TimingModel,
    TraceEvent,
    TaskTiming,
    simulate_execution_graph,
)
from .address import AddressDependency, AddressHazardKind, add_address_dependencies

__all__ = [
    "AnalyticalTimingModel",
    "AddressDependency",
    "AddressHazardKind",
    "SimulationResult",
    "SimulatorConfig",
    "TaskTiming",
    "TaskTimingSpec",
    "TimingModel",
    "TraceEvent",
    "add_address_dependencies",
    "simulate_execution_graph",
]
