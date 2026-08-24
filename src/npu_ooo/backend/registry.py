"""Registry for timing providers used by the simulator CLI and API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from npu_ooo.simulator.core import AnalyticalTimingModel, TimingModel, TimingTableModel

from .contracts import BackendCapabilities, TimingProvider


_ANALYTICAL_PRIMITIVES = frozenset(
    {
        "load",
        "load_transpose",
        "store",
        "matmul",
        "elementwise",
        "reduce",
        "reduce_max",
        "reduce_sum",
        "reduce_sum_square",
        "exp",
        "normalize",
        "square",
        "center",
        "rmsnorm",
        "layernorm_mean",
        "layernorm",
    }
)


def analytical_capabilities(*, backend: str = "analytical") -> BackendCapabilities:
    return BackendCapabilities(
        backend=backend,
        supported_primitives=_ANALYTICAL_PRIMITIVES,
        calibration_status="analytical",
        attributes={"source": "MachineConfig plus task/timing table"},
    )


@dataclass(frozen=True)
class TimingProviderAdapter:
    """Add capability metadata to an existing structural TimingModel."""

    provider: TimingModel
    capabilities: BackendCapabilities

    @property
    def name(self) -> str:
        return self.provider.name

    def timing(self, task, machine):
        return self.provider.timing(task, machine)


TimingFactory = Callable[[Path | None], TimingProvider]


class TimingProviderRegistry:
    def __init__(self, factories: Mapping[str, TimingFactory] | None = None) -> None:
        self._factories: dict[str, TimingFactory] = dict(factories or {})

    def register(self, name: str, factory: TimingFactory) -> None:
        if not name or not callable(factory):
            raise ValueError("timing provider name and factory must be valid")
        if name in self._factories:
            raise ValueError(f"timing provider '{name}' is already registered")
        self._factories[name] = factory

    def create(self, name: str, path: Path | None = None) -> TimingProvider:
        try:
            provider = self._factories[name](path)
        except KeyError as exc:
            raise ValueError(
                f"unknown timing provider '{name}'; available: {', '.join(sorted(self._factories))}"
            ) from exc
        if not hasattr(provider, "capabilities"):
            raise TypeError(f"timing provider '{name}' does not expose capabilities")
        return provider

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_timing_provider_registry() -> TimingProviderRegistry:
    registry = TimingProviderRegistry()
    registry.register(
        "analytical",
        lambda _path: TimingProviderAdapter(
            AnalyticalTimingModel(), analytical_capabilities()
        ),
    )
    def create_timing_table(path: Path | None) -> TimingProvider:
        if path is None:
            raise ValueError("timing_table provider requires --timing-config")
        return TimingProviderAdapter(
            TimingTableModel.from_path(path),
            analytical_capabilities(backend="timing_table"),
        )

    registry.register("timing_table", create_timing_table)
    return registry


__all__ = [
    "TimingProviderAdapter",
    "TimingProviderRegistry",
    "analytical_capabilities",
    "default_timing_provider_registry",
]
