import tempfile
import unittest
from pathlib import Path

from npu_ooo.arch import minimal_machine_config
from npu_ooo.backend import (
    AnalyticalCodegenBackend,
    AnalyticalEventBackend,
    BackendCapabilities,
    CodegenBackendRegistry,
    EventBackendRegistry,
    default_codegen_backend_registry,
    default_event_backend_registry,
    default_timing_provider_registry,
    validate_backend_capability,
)
from npu_ooo.benchmarks import build_elementwise_case, build_elementwise_model
from npu_ooo.compiler import compile_model_instance
from npu_ooo.ir import ExecutionTask
from npu_ooo.simulator import TimingTableModel


class BackendContractTest(unittest.TestCase):
    def test_default_codegen_registry_exposes_analytical_backend(self) -> None:
        registry = default_codegen_backend_registry()
        self.assertEqual(registry.names(), ("analytical",))
        backend = registry.create("analytical")
        self.assertIsInstance(backend, AnalyticalCodegenBackend)
        self.assertEqual(backend.capabilities.calibration_status, "analytical")

    def test_codegen_registry_rejects_duplicates_and_unknown_names(self) -> None:
        registry = CodegenBackendRegistry()
        registry.register("analytical", AnalyticalCodegenBackend)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("analytical", AnalyticalCodegenBackend)
        with self.assertRaisesRegex(ValueError, "unknown codegen backend"):
            registry.create("missing")

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
        self.assertEqual(
            registry.names(),
            ("analytical", "systolic_mxu_profile", "timing_table"),
        )
        analytical = registry.create("analytical")
        self.assertEqual(analytical.capabilities.calibration_status, "analytical")
        self.assertEqual(analytical.name, "analytical")

    def test_timing_table_provider_requires_a_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --timing-config"):
            default_timing_provider_registry().create("timing_table")

    def test_systolic_mxu_profile_requires_a_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --timing-config"):
            default_timing_provider_registry().create("systolic_mxu_profile")

    def test_systolic_mxu_profile_matches_exact_tile_shape(self) -> None:
        path = Path("configs/timing/systolic_mxu_matmul_example.json")
        provider = default_timing_provider_registry().create(
            "systolic_mxu_profile", path
        )
        task = ExecutionTask(
            "matmul.0",
            "tile.0",
            "matmul",
            "matmul",
            "MXU",
            attributes={"batch_tile": [], "m_tile": 4, "n_tile": 12, "k_tile": 8},
        )

        timing = provider.timing(task, minimal_machine_config())
        self.assertEqual(timing.duration_cycles, 23)
        self.assertEqual(timing.initiation_interval_cycles, 3)
        self.assertEqual(
            provider.capabilities.calibration_status,
            "mixed:source-derived-example+analytical-fallback",
        )
        self.assertEqual(
            provider.metadata["profile_calibration_status"],
            "source-derived-example",
        )
        coverage = provider.coverage(
            (
                task,
                ExecutionTask(
                    "matmul.unmatched",
                    "tile.1",
                    "matmul",
                    "matmul",
                    "MXU",
                    attributes={"m_tile": 4, "n_tile": 8, "k_tile": 8},
                ),
                ExecutionTask(
                    "vector.0",
                    "tile.2",
                    "elementwise",
                    "elementwise",
                    "ARU",
                ),
            )
        )
        self.assertEqual(coverage["calibrated_matmul_task_count"], 1)
        self.assertEqual(coverage["unmatched_matmul_task_count"], 1)
        self.assertEqual(coverage["non_matmul_analytical_task_count"], 1)
        self.assertEqual(provider.metadata["profile_count"], 1)

    def test_systolic_mxu_profile_strictly_rejects_unmatched_matmul(self) -> None:
        from npu_ooo.backend import SystolicMXUProfileTimingProvider

        provider = SystolicMXUProfileTimingProvider.from_dict(
            {
                "format": "npu_ooo.systolic_mxu_profile.v1",
                "unmatched_matmul": "error",
                "matmul_profiles": [
                    {
                        "shape": {"m": 4, "n": 12, "k": 8},
                        "duration_cycles": 23,
                        "initiation_interval_cycles": 3,
                    }
                ],
            }
        )
        task = ExecutionTask(
            "matmul.unmatched",
            "tile.1",
            "matmul",
            "matmul",
            "MXU",
            attributes={"m_tile": 8, "n_tile": 12, "k_tile": 8},
        )

        with self.assertRaisesRegex(ValueError, "no calibrated matmul tile"):
            provider.timing(task, minimal_machine_config())

    def test_descriptor_interval_profile_cannot_be_used_for_isolated_matmul(self) -> None:
        from npu_ooo.backend import SystolicMXUProfileTimingProvider

        provider = SystolicMXUProfileTimingProvider.from_dict(
            {
                "format": "npu_ooo.systolic_mxu_profile.v1",
                "metadata": {"interval": "descriptor_issue_to_done"},
                "matmul_profiles": [
                    {
                        "shape": {"m": 4, "n": 12, "k": 8},
                        "duration_cycles": 31,
                        "initiation_interval_cycles": 10,
                    }
                ],
            }
        )
        task = ExecutionTask(
            "matmul.full-interval",
            "tile.0",
            "matmul",
            "matmul",
            "MXU",
            attributes={"m_tile": 4, "n_tile": 12, "k_tile": 8},
        )
        self.assertFalse(provider.capabilities.attributes["isolated_matmul_compatible"])
        with self.assertRaisesRegex(ValueError, "cannot be applied to the isolated matmul"):
            provider.timing(task, minimal_machine_config())

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

    def test_explicit_codegen_backend_matches_default_artifact(self) -> None:
        model = build_elementwise_model(rows=8, cols=8)
        instance = model.instantiate(build_elementwise_case())
        machine = minimal_machine_config()
        default = compile_model_instance(instance, machine, tile_size=4)
        explicit = compile_model_instance(
            instance,
            machine,
            tile_size=4,
            codegen_backend=AnalyticalCodegenBackend(),
        )

        self.assertEqual(explicit.tisa_program, default.tisa_program)
        self.assertEqual(explicit.backend_artifact, default.backend_artifact)
        self.assertEqual(explicit.attributes["codegen_backend"], "analytical")
        self.assertEqual(
            explicit.attributes["codegen_backend_capabilities"]["calibration_status"],
            "analytical",
        )

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
