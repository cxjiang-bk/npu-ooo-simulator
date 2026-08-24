import tempfile
import unittest
from pathlib import Path

from npu_ooo.arch import minimal_machine_config
from npu_ooo.backend import (
    AnalyticalEventBackend,
    BackendCapabilities,
    EventBackendRegistry,
    default_event_backend_registry,
    default_timing_provider_registry,
    validate_backend_capability,
)
from npu_ooo.benchmarks import build_elementwise_case, build_elementwise_model
from npu_ooo.compiler import compile_model_instance
from npu_ooo.simulator import TimingTableModel


class BackendContractTest(unittest.TestCase):
    def test_default_event_registry_exposes_analytical_backend(self) -> None:
        registry = default_event_backend_registry()
        self.assertEqual(registry.names(), ("analytical_event",))
        backend = registry.create("analytical_event")
        self.assertIsInstance(backend, AnalyticalEventBackend)
        self.assertEqual(backend.capabilities.calibration_status, "analytical")

    def test_event_registry_rejects_duplicates_and_unknown_names(self) -> None:
        registry = EventBackendRegistry()
        registry.register("analytical_event", AnalyticalEventBackend)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("analytical_event", AnalyticalEventBackend)
        with self.assertRaisesRegex(ValueError, "unknown event backend"):
            registry.create("missing")

    def test_default_registry_exposes_analytical_and_timing_table(self) -> None:
        registry = default_timing_provider_registry()
        self.assertEqual(registry.names(), ("analytical", "timing_table"))
        analytical = registry.create("analytical")
        self.assertEqual(analytical.capabilities.calibration_status, "analytical")
        self.assertEqual(analytical.name, "analytical")

    def test_timing_table_provider_requires_a_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --timing-config"):
            default_timing_provider_registry().create("timing_table")

    def test_capability_validation_rejects_unsupported_payload(self) -> None:
        model = build_elementwise_model(rows=8, cols=8)
        instance = model.instantiate(build_elementwise_case())
        compiled = compile_model_instance(instance, minimal_machine_config(), tile_size=4)
        capability = BackendCapabilities(
            backend="mxu-only",
            supported_primitives=frozenset({"matmul"}),
            calibration_status="source-derived",
        )
        issues = validate_backend_capability(
            compiled.backend_artifact,
            minimal_machine_config(),
            capability,
        )
        self.assertTrue(any("elementwise" in issue for issue in issues))

    def test_timing_table_provider_preserves_provider_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.json"
            path.write_text(
                '{"name":"probe","entries":{"elementwise":{"duration_cycles":3,"initiation_interval_cycles":1}}}',
                encoding="utf-8",
            )
            provider = default_timing_provider_registry().create("timing_table", path)
            self.assertEqual(provider.name, "probe")
            self.assertIsInstance(provider.provider, TimingTableModel)


if __name__ == "__main__":
    unittest.main()
