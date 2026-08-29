import unittest
from dataclasses import replace
import importlib.util

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import RecoverStableHLOKVCachePass, compile_operator_graph
from npu_ooo.compiler import compile_torch_module
from npu_ooo.frontend import FrontendImportError, official_stablehlo_available, torch_xla_available
from npu_ooo.frontend.stablehlo import StableHLOAdapter
from npu_ooo.frontend.stablehlo_official import OfficialStableHLOModule
from npu_ooo.ir import (
    allocate_buffer_bindings,
    create_runtime_sequence,
    create_runtime_state_registry,
    create_runtime_submission,
)
from npu_ooo.scheduler import (
    SchedulerPolicy,
    schedule_tisa_program,
    schedule_tisa_sequence,
)


FRONTEND_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and torch_xla_available()
    and official_stablehlo_available()
)


_KV_TEXT = """
module {
  func.func @main(
    %cache: tensor<1x2x4x4xf32>,
    %update: tensor<1x2x1x4xf32>
  ) -> tensor<1x2x4x4xf32> {
    %slice = stablehlo.slice %cache starts = [0, 0, 1, 0] limits = [1, 2, 4, 4] strides = [1, 1, 1, 1] : (tensor<1x2x4x4xf32>) -> tensor<1x2x3x4xf32>
    %output = stablehlo.concatenate %slice, %update dim = 2 : (tensor<1x2x3x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x4x4xf32>
    return %output : tensor<1x2x4x4xf32>
  }
}
"""


def _compile_text(text: str):
    frontend = StableHLOAdapter.from_text(text, model_id="kv-text")
    stablehlo = OfficialStableHLOModule(
        text=text,
        canonical_text=text,
        model_id="kv-text",
    )
    return compile_operator_graph(
        frontend.graph,
        minimal_machine_config(),
        frontend=frontend,
        source_frontend=frontend,
        stablehlo=stablehlo,
        tile_size=2,
    )


class KVCacheContractTest(unittest.TestCase):
    def test_importer_preserves_slice_and_concatenate_attributes(self) -> None:
        imported = StableHLOAdapter.from_text(_KV_TEXT, model_id="kv-import")
        self.assertEqual(
            [operator.normalized_type for operator in imported.graph.operators],
            ["slice", "concatenate"],
        )
        sliced, concatenated = imported.graph.operators
        self.assertEqual(sliced.attributes["slice_starts"], [0, 0, 1, 0])
        self.assertEqual(sliced.attributes["slice_limits"], [1, 2, 4, 4])
        self.assertEqual(sliced.attributes["slice_strides"], [1, 1, 1, 1])
        self.assertEqual(concatenated.attributes["concatenate_dimension"], 2)
        self.assertEqual(sliced.iteration_dims, (("d0", 1), ("d1", 2), ("d2", 3), ("d3", 4)))
        self.assertEqual(concatenated.iteration_dims, (("d0", 1), ("d1", 2), ("d2", 4), ("d3", 4)))

    def test_importer_rejects_slice_bounds_outside_operand(self) -> None:
        text = _KV_TEXT.replace("limits = [1, 2, 4, 4]", "limits = [1, 2, 5, 4]")
        with self.assertRaisesRegex(FrontendImportError, "invalid bounds"):
            StableHLOAdapter.from_text(text)

    def test_importer_rejects_incompatible_concatenate_shape(self) -> None:
        text = _KV_TEXT.replace(
            "%update: tensor<1x2x1x4xf32>",
            "%update: tensor<1x3x1x4xf32>",
        )
        with self.assertRaisesRegex(FrontendImportError, "incompatible shapes"):
            StableHLOAdapter.from_text(text)

    def test_recovery_produces_persistent_state_alias(self) -> None:
        compiled = _compile_text(_KV_TEXT)
        self.assertEqual(len(compiled.graph.operators), 1)
        operator = compiled.graph.operators[0]
        self.assertEqual(operator.normalized_type, "kv_cache_update")
        self.assertEqual(operator.attributes["state_id"], "cache")
        self.assertEqual(operator.attributes["state_buffer"], "cache")
        self.assertEqual(operator.attributes["cache_axis"], 2)
        self.assertEqual(operator.attributes["cache_window"], 4)
        self.assertEqual(operator.attributes["update_length"], 1)
        output = next(tensor for tensor in compiled.graph.tensors if tensor.name == "output")
        self.assertEqual(output.attributes["alias_of"], "cache")

        state_instructions = compiled.tisa_program.instructions
        self.assertTrue(state_instructions)
        self.assertTrue(all(item.attributes["stateful"] for item in state_instructions))
        self.assertTrue(all(item.attributes["state_id"] == "cache" for item in state_instructions))
        self.assertEqual(
            [item.op_type for item in state_instructions],
            ["load", "kv_cache_update", "store"],
        )

    def test_runtime_binds_alias_and_exports_state_contract(self) -> None:
        compiled = _compile_text(_KV_TEXT)
        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        cache = next(item for item in bindings if item.tensor == "cache")
        output = next(item for item in bindings if item.tensor == "output")
        self.assertEqual(cache.base_address, output.base_address)
        self.assertTrue(cache.attributes["persistent"])

        submission = create_runtime_submission(
            compiled.backend_artifact,
            bindings,
            policy="dynamic_ready_queue",
        )
        self.assertEqual(submission.validate(compiled.tisa_program), ())
        self.assertEqual(submission.attributes["state_contract"], "persistent_buffer_v1")
        self.assertEqual(submission.attributes["state_buffers"][0]["state_id"], "cache")

    def test_runtime_rejects_nonpersistent_state_binding(self) -> None:
        compiled = _compile_text(_KV_TEXT)
        bindings = list(allocate_buffer_bindings(compiled.graph.tensors))
        index = next(index for index, item in enumerate(bindings) if item.tensor == "cache")
        bindings[index] = replace(
            bindings[index],
            attributes={"allocation_policy": "linear"},
        )
        with self.assertRaisesRegex(ValueError, "must be bound as persistent"):
            create_runtime_submission(compiled.backend_artifact, bindings)

    def test_state_registry_keeps_full_bindings_and_stable_address(self) -> None:
        compiled = _compile_text(_KV_TEXT)
        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        registry = create_runtime_state_registry(compiled.backend_artifact, bindings)
        self.assertEqual(registry.state_ids(), ("cache",))
        self.assertEqual(
            {item.tensor for item in registry.runtime_buffers()},
            {item.tensor for item in bindings},
        )
        self.assertEqual(
            registry.binding("cache").base_address,
            next(item for item in bindings if item.tensor == "cache").base_address,
        )

    def test_two_invocation_sequence_preserves_state_and_records_dependency(self) -> None:
        compiled = _compile_text(_KV_TEXT)
        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        registry = create_runtime_state_registry(compiled.backend_artifact, bindings)
        sequence = create_runtime_sequence(
            compiled.backend_artifact,
            registry,
            invocation_count=2,
            sequence_id="kv.decode",
            policy="dynamic_ready_queue",
            chunk_size=2,
            inter_invocation_gap_cycles=3,
        )
        self.assertEqual(sequence.validate(compiled.tisa_program), ())
        self.assertEqual(len(sequence.invocations), 2)
        self.assertEqual(len(sequence.dependencies), 1)
        dependency = sequence.dependencies[0]
        self.assertEqual(dependency.condition, "state_complete")
        self.assertEqual(dependency.state_ids, ("cache",))
        cache_bases = [
            next(item for item in invocation.buffers if item.tensor == "cache").base_address
            for invocation in sequence.invocations
        ]
        self.assertEqual(cache_bases, [cache_bases[0], cache_bases[0]])

        result = schedule_tisa_sequence(
            compiled.backend_artifact,
            sequence,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        expected_cycles = sum(result.metrics["invocation_total_cycles"]) + 3
        self.assertEqual(result.total_cycles, expected_cycles)
        self.assertEqual(result.metrics["state_dependency_count"], 1)
        self.assertEqual(result.metrics["state_wait_cycles"], 3)
        self.assertTrue(any(event.event == "STATE_RELEASE" for event in result.events))
        self.assertTrue(any(event.event == "STATE_READY" for event in result.events))
        self.assertTrue(
            all(".invocation" in timing.task_id for timing in result.instruction_timings)
        )

    def test_scheduler_maps_update_stage_to_vector_unit(self) -> None:
        compiled = _compile_text(_KV_TEXT)
        self.assertEqual(
            [item.unit_map.unit for item in compiled.tisa_program.instructions],
            ["dma", "vector", "dma"],
        )
        result = schedule_tisa_program(
            compiled.backend_artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertGreater(result.total_cycles, 0)

    @unittest.skipUnless(
        FRONTEND_AVAILABLE,
        "requires PyTorch, Torch-XLA and official StableHLO",
    )
    def test_real_torch_xla_kv_cache_reaches_canonical_state_contract(self) -> None:
        import torch

        class KVModule(torch.nn.Module):
            def forward(self, key_cache, value_cache, key, value):
                return (
                    torch.cat((key_cache[..., 1:, :], key), dim=-2),
                    torch.cat((value_cache[..., 1:, :], value), dim=-2),
                )

        cache_shape = (1, 2, 4, 4)
        update_shape = (1, 2, 1, 4)
        inputs = (
            torch.randn(*cache_shape),
            torch.randn(*cache_shape),
            torch.randn(*update_shape),
            torch.randn(*update_shape),
        )
        compiled = compile_torch_module(
            KVModule().eval(),
            inputs,
            minimal_machine_config(),
            model_id="kv-torch-xla",
            tile_size=2,
        )

        self.assertEqual(
            compiled.attributes["frontend_path"],
            "torch_export->torch_xla->official_stablehlo->canonical",
        )
        self.assertTrue(compiled.stablehlo.verified)
        self.assertEqual(
            [operator.normalized_type for operator in compiled.graph.operators],
            ["kv_cache_update", "kv_cache_update"],
        )
        state_ids = {
            operator.attributes["state_id"]
            for operator in compiled.graph.operators
        }
        self.assertEqual(state_ids, {"arg1", "arg3"})
        self.assertEqual(
            {
                instruction.attributes["state_transition"]
                for instruction in compiled.tisa_program.instructions
            },
            {"drop_oldest_append_new"},
        )

        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        submission = create_runtime_submission(
            compiled.backend_artifact,
            bindings,
            policy="dynamic_ready_queue",
        )
        self.assertEqual(submission.validate(compiled.tisa_program), ())
        self.assertEqual(len(submission.attributes["state_buffers"]), 2)

    def test_recovery_rejects_non_fixed_window(self) -> None:
        text = _KV_TEXT.replace("[0, 0, 1, 0]", "[0, 0, 0, 0]")
        text = text.replace("strides = [1, 1, 1, 1]", "strides = [1, 1, 2, 1]")
        text = text.replace(
            "-> tensor<1x2x3x4xf32>",
            "-> tensor<1x2x2x4xf32>",
            1,
        )
        text = text.replace(
            "tensor<1x2x3x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x4x4xf32>",
            "tensor<1x2x2x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x3x4xf32>",
        )
        text = text.replace(
            ") -> tensor<1x2x4x4xf32> {",
            ") -> tensor<1x2x3x4xf32> {",
            1,
        )
        text = text.replace(
            "return %output : tensor<1x2x4x4xf32>",
            "return %output : tensor<1x2x3x4xf32>",
        )
        imported = StableHLOAdapter.from_text(text)
        result = RecoverStableHLOKVCachePass().run(imported.graph)
        self.assertEqual(
            [operator.normalized_type for operator in result.graph.operators],
            ["slice", "concatenate"],
        )


if __name__ == "__main__":
    unittest.main()
