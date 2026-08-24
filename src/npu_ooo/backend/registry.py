"""Registry for timing providers used by the simulator CLI and API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from npu_ooo.simulator.core import AnalyticalTimingModel, TimingModel, TimingTableModel

from .contracts import BackendCapabilities, CodegenBackend, EventBackend, TimingProvider


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
EventBackendFactory = Callable[[], EventBackend]
CodegenBackendFactory = Callable[..., CodegenBackend]


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


class EventBackendRegistry:
    """Factory registry for device event engines."""

    def __init__(self, factories: Mapping[str, EventBackendFactory] | None = None) -> None:
        self._factories: dict[str, EventBackendFactory] = dict(factories or {})

    def register(self, name: str, factory: EventBackendFactory) -> None:
        if not name or not callable(factory):
            raise ValueError("event backend name and factory must be valid")
        if name in self._factories:
            raise ValueError(f"event backend '{name}' is already registered")
        self._factories[name] = factory

    def create(self, name: str) -> EventBackend:
        try:
            backend = self._factories[name]()
        except KeyError as exc:
            raise ValueError(
                f"unknown event backend '{name}'; available: {', '.join(sorted(self._factories))}"
            ) from exc
        if not hasattr(backend, "capabilities"):
            raise TypeError(f"event backend '{name}' does not expose capabilities")
        return backend

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


class CodegenBackendRegistry:
    """Factory registry for TISA-to-payload code generators."""

    def __init__(self, factories: Mapping[str, CodegenBackendFactory] | None = None) -> None:
        self._factories: dict[str, CodegenBackendFactory] = dict(factories or {})

    def register(self, name: str, factory: CodegenBackendFactory) -> None:
        if not name or not callable(factory):
            raise ValueError("codegen backend name and factory must be valid")
        if name in self._factories:
            raise ValueError(f"codegen backend '{name}' is already registered")
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> CodegenBackend:
        try:
            backend = self._factories[name](**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"unknown codegen backend '{name}'; available: {', '.join(sorted(self._factories))}"
            ) from exc
        if not hasattr(backend, "capabilities"):
            raise TypeError(f"codegen backend '{name}' does not expose capabilities")
        return backend

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

    def create_systolic_mxu_profile(path: Path | None) -> TimingProvider:
        if path is None:
            raise ValueError("systolic_mxu_profile provider requires --timing-config")
        from .mxu_profile import SystolicMXUProfileTimingProvider

        return SystolicMXUProfileTimingProvider.from_path(path)

    registry.register("systolic_mxu_profile", create_systolic_mxu_profile)
    return registry


def default_event_backend_registry() -> EventBackendRegistry:
    # Keep the import lazy: the analytical adapter delegates to simulator.tisa,
    # while simulator.tisa imports backend capability validation.
    from .analytical import AnalyticalEventBackend

    registry = EventBackendRegistry()
    registry.register("analytical_event", AnalyticalEventBackend)
    return registry


def default_codegen_backend_registry() -> CodegenBackendRegistry:
    from .codegen import AnalyticalCodegenBackend

    registry = CodegenBackendRegistry()
    registry.register("analytical", AnalyticalCodegenBackend)
    return registry


__all__ = [
    "TimingProviderAdapter",
    "CodegenBackendRegistry",
    "EventBackendRegistry",
    "TimingProviderRegistry",
    "analytical_capabilities",
    "default_codegen_backend_registry",
    "default_event_backend_registry",
    "default_timing_provider_registry",
]
