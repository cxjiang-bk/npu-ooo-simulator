import importlib.util
import unittest

from examples.paper_benchmarks import build_paper_benchmark, paper_benchmark_specs
from examples.paper_benchmarks.llama2 import build_decode
from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available
from npu_ooo.ir import (
    allocate_buffer_bindings,
    create_runtime_sequence,
    create_runtime_state_registry,
)
from npu_ooo.scheduler import SchedulerPolicy, schedule_tisa_sequence


FRONTEND_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and torch_xla_available()
    and official_stablehlo_available()
)


class PaperBenchmarkRegistryTest(unittest.TestCase):
    def test_table_ix_registry_contains_six_rows_in_paper_order(self) -> None:
        specs = paper_benchmark_specs()
        self.assertEqual(
            [spec.case_id for spec in specs],
            [
                "resnet50",
                "bert-base",
                "gpt-j-6b-oneblk",
                "llama2-13b-oneblk",
                "deepseek-r1-16b-prefill",
                "deepseek-r1-16b-decode",
            ],
        )
        self.assertEqual(specs[0].reference_a100_ms, 9.3)
        self.assertEqual(specs[-1].phase, "decode")

    def test_each_row_builds_an_independent_real_pytorch_workload(self) -> None:
        for spec in paper_benchmark_specs():
            workload = build_paper_benchmark(spec.case_id, variant="micro")
            self.assertEqual(workload.spec, spec)
            self.assertTrue(workload.module.training is False)
            self.assertTrue(workload.inputs)
            self.assertTrue(all(value.numel() > 0 for value in workload.inputs))


@unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
class PaperBenchmarkFrontendTest(unittest.TestCase):
    def test_transformer_rows_reach_tisa(self) -> None:
        for case_id in ("bert-base", "gpt-j-6b-oneblk", "llama2-13b-oneblk"):
            workload = build_paper_benchmark(case_id, variant="micro")
            compiled = compile_torch_module(
                workload.module,
                workload.inputs,
                minimal_machine_config(),
                model_id=case_id,
                tile_size=4,
            )
            self.assertTrue(compiled.tisa_program.instructions)
            regions = [
                region
                for region in compiled.graph.attributes.get("semantic_regions", ())
                if region.get("semantic_family") == "attention"
            ]
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0]["semantic_family"], "attention")
            self.assertFalse(regions[0]["opaque"])
            if case_id == "llama2-13b-oneblk":
                rotary_regions = [
                    region
                    for region in compiled.graph.attributes.get("semantic_regions", ())
                    if region.get("semantic_family") == "rotary_embedding"
                ]
                self.assertEqual(len(rotary_regions), 1)
                rotary_region = rotary_regions[0]
                self.assertFalse(rotary_region["opaque"])
                self.assertEqual(set(rotary_region["roles"]), {
                    "query",
                    "key",
                    "cosine",
                    "sine",
                    "rotation_matrix",
                })
                self.assertEqual(rotary_region["algorithm"], "rotate_half")
                rotary_ops = [
                    operator
                    for operator in compiled.graph.operators
                    if operator.attributes.get("semantic_region_family")
                    == "rotary_embedding"
                ]
                self.assertTrue(rotary_ops)
                self.assertIn(
                    "rotate_half",
                    {operator.attributes["semantic_region_role"] for operator in rotary_ops},
                )
                rotary_tisa = [
                    instruction
                    for instruction in compiled.tisa_program.instructions
                    if instruction.attributes.get("semantic_region_family")
                    == "rotary_embedding"
                ]
                self.assertTrue(rotary_tisa)
                self.assertTrue(
                    all(
                        instruction.attributes.get("rotary_algorithm") == "rotate_half"
                        for instruction in rotary_tisa
                    )
                )
                swiglu = [
                    operator
                    for operator in compiled.graph.operators
                    if operator.normalized_type == "swiglu"
                ]
                self.assertEqual(len(swiglu), 1)
                self.assertEqual(
                    swiglu[0].attributes["conversion_steps"],
                    [
                        {"source_dtype": "f32", "target_dtype": "f16"},
                        {"source_dtype": "f16", "target_dtype": "f32"},
                    ],
                )
                self.assertIn(
                    "dtype_convert",
                    {
                        task.primitive
                        for task in compiled.backend_artifact.execution_graph.tasks
                        if task.operator_id == swiglu[0].op_id
                    },
                )
            self.assertEqual(compiled.validate(), ())

    def test_resnet_conv2d_reaches_tisa_and_backend(self) -> None:
        workload = build_paper_benchmark("resnet50", variant="micro")
        compiled = compile_torch_module(
            workload.module,
            workload.inputs,
            minimal_machine_config(),
            model_id="resnet50",
            tile_size=4,
        )
        convolution_ops = [
            operator
            for operator in compiled.graph.operators
            if operator.normalized_type == "conv2d"
        ]
        self.assertEqual(len(convolution_ops), 4)
        self.assertEqual(
            sum(operator.normalized_type == "batch_norm" for operator in compiled.graph.operators),
            4,
        )
        self.assertEqual(
            sum(operator.normalized_type == "pool" for operator in compiled.graph.operators),
            1,
        )
        self.assertTrue(all(operator.attributes["convolution_dimension_numbers"] ==
                            "nchw_oihw_nchw" for operator in convolution_ops))
        self.assertTrue(all(operator.attributes["padding"] in ([0, 0, 0, 0], [1, 1, 1, 1])
                            for operator in convolution_ops))
        spatial_conv = next(
            operator
            for operator in convolution_ops
            if operator.attributes["kernel_shape"] == [3, 3]
        )
        conv_tiles = [
            tile for tile in compiled.tile_graph.tiles
            if tile.operator_id == spatial_conv.op_id
        ]
        self.assertTrue(conv_tiles)
        # A 3x3 convolution at the left boundary consumes a halo wider than
        # the output tile, so the TileGraph must retain region dependencies
        # from adjacent producer tiles.
        input_tensor = spatial_conv.inputs[0]
        halo_dependencies = [
            dependency
            for dependency in compiled.tile_graph.dependencies
            if dependency.tensor == input_tensor
            and dependency.consumer == f"{spatial_conv.op_id}.t0000"
        ]
        self.assertGreaterEqual(len(halo_dependencies), 4)
        conv_instructions = [
            instruction
            for instruction in compiled.tisa_program.instructions
            if instruction.attributes.get("semantic_op_type") == "conv2d"
        ]
        self.assertTrue(conv_instructions)
        self.assertIn("conv2d", {instruction.op_type for instruction in conv_instructions})
        self.assertTrue(compiled.backend_artifact.execution_graph.tasks)
        self.assertEqual(compiled.validate(), ())

    def test_deepseek_dense_one_block_reaches_tisa(self) -> None:
        for case_id in ("deepseek-r1-16b-prefill", "deepseek-r1-16b-decode"):
            workload = build_paper_benchmark(case_id, variant="micro")
            compiled = compile_torch_module(
                workload.module,
                workload.inputs,
                minimal_machine_config(),
                model_id=case_id,
                tile_size=4,
            )
            self.assertTrue(compiled.tisa_program.instructions)
            self.assertTrue(
                any(operator.normalized_type == "swiglu" for operator in compiled.graph.operators)
            )
            self.assertTrue(
                any(operator.normalized_type == "rmsnorm" for operator in compiled.graph.operators)
            )
            self.assertEqual(compiled.validate(), ())

    def test_llama2_decode_reaches_runtime_sequence_with_two_cache_states(self) -> None:
        workload = build_decode()
        self.assertEqual(workload.variant, "decode_micro")
        self.assertEqual(workload.attributes["phase"], "decode")
        compiled = compile_torch_module(
            workload.module,
            workload.inputs,
            minimal_machine_config(),
            model_id="llama2-13b-decode-micro",
            tile_size=4,
        )
        cache_updates = [
            operator
            for operator in compiled.graph.operators
            if operator.normalized_type == "kv_cache_update"
        ]
        self.assertEqual(len(cache_updates), 2)
        self.assertEqual(
            {operator.attributes["state_transition"] for operator in cache_updates},
            {"drop_oldest_append_new"},
        )
        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        registry = create_runtime_state_registry(compiled.backend_artifact, bindings)
        self.assertEqual(registry.state_ids(), ("arg11", "arg18"))
        sequence = create_runtime_sequence(
            compiled.backend_artifact,
            registry,
            invocation_count=2,
            sequence_id="llama2.decode.micro",
            policy="dynamic_ready_queue",
            chunk_size=16,
        )
        self.assertEqual(sequence.validate(compiled.tisa_program), ())
        self.assertEqual(len(sequence.dependencies), 1)
        static_result = schedule_tisa_sequence(
            compiled.backend_artifact,
            sequence,
            minimal_machine_config(),
            SchedulerPolicy.STATIC_PIPELINE,
        )
        result = schedule_tisa_sequence(
            compiled.backend_artifact,
            sequence,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(result.metrics["state_ids"], ["arg11", "arg18"])
        self.assertEqual(result.metrics["state_dependency_count"], 1)
        self.assertEqual(result.metrics["invocation_count"], 2)
        self.assertGreater(result.total_cycles, 0)
        self.assertGreater(static_result.total_cycles, 0)
        self.assertEqual(
            static_result.metrics["state_ids"],
            result.metrics["state_ids"],
        )
        self.assertTrue(any(event.event == "STATE_RELEASE" for event in result.events))


if __name__ == "__main__":
    unittest.main()
