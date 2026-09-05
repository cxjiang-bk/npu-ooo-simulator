import importlib.util
import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module
from npu_ooo.experiments import run_runtime_device_matrix
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available
from npu_ooo.ir import allocate_buffer_bindings


FRONTEND_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and torch_xla_available()
    and official_stablehlo_available()
)


@unittest.skipUnless(FRONTEND_AVAILABLE, "requires the production PyTorch frontend")
class RuntimeDeviceMatrixTest(unittest.TestCase):
    def test_four_policy_cells_share_compiled_artifact_and_buffers(self) -> None:
        import torch

        from examples.torch_models import ResidualAdd

        machine = minimal_machine_config()
        compiled = compile_torch_module(
            ResidualAdd(),
            (torch.randn(8, 8), torch.randn(8, 8)),
            machine,
            model_id="runtime-matrix",
            tile_size=4,
        )
        buffers = allocate_buffer_bindings(compiled.graph.tensors)
        cases = run_runtime_device_matrix(
            compiled.backend_artifact,
            buffers,
            machine,
            chunk_size=4,
            launch_latency_cycles=2,
            synchronization_cycles=3,
        )

        self.assertEqual(len(cases), 4)
        self.assertEqual(
            {(case.runtime_policy, case.device_policy) for case in cases},
            {
                ("static", "static_pipeline"),
                ("static", "dynamic_ready_queue"),
                ("dynamic_ready_queue", "static_pipeline"),
                ("dynamic_ready_queue", "dynamic_ready_queue"),
            },
        )
        self.assertEqual(
            {case.submission.artifact_id for case in cases},
            {compiled.backend_artifact.artifact_id},
        )
        self.assertTrue(all(case.submission.buffers == buffers for case in cases))
        self.assertTrue(all(case.result.runtime_timings for case in cases))
        self.assertTrue(
            all(case.result.metrics["runtime_synchronization_cycles"] == 3 for case in cases)
        )


if __name__ == "__main__":
    unittest.main()
