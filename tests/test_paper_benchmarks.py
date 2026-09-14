import importlib.util
import unittest

from examples.paper_benchmarks import build_paper_benchmark, get_paper_benchmark, paper_benchmark_specs
from examples.paper_benchmarks.llama2 import build_decode
from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available
from npu_ooo.experiments import run_runtime_device_matrix
from npu_ooo.ir import (
    RuntimeSequence,
    allocate_memory_plan_bindings,
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

    def test_resnet_unsupported_features_describe_model_omissions(self) -> None:
        unsupported = set(get_paper_benchmark("resnet50").unsupported_features)
        self.assertNotIn("stablehlo.convolution", unsupported)
        self.assertNotIn("stablehlo.batch_norm_inference", unsupported)
        self.assertNotIn("pooling", unsupported)
        self.assertIn("full_model_depth", unsupported)
        self.assertIn("classification_head", unsupported)

    def test_each_row_builds_an_independent_real_pytorch_workload(self) -> None:
        for spec in paper_benchmark_specs():
            workload = build_paper_benchmark(spec.case_id, variant="micro")
            self.assertEqual(workload.spec, spec)
            self.assertTrue(workload.module.training is False)
            self.assertTrue(workload.inputs)
            self.assertTrue(all(value.numel() > 0 for value in workload.inputs))

    def test_transformer_depth_proxy_repeats_blocks_with_same_input_contract(self) -> None:
        workload = build_paper_benchmark("bert-base", variant="micro", layer_count=3)
        self.assertEqual(workload.attributes["layer_count"], 3)
        self.assertEqual(workload.attributes["model_depth_proxy"], "sequential_repeated_blocks")
        self.assertEqual(len(workload.module.layers), 3)
        self.assertEqual(
            [tuple(value.shape) for value in workload.inputs],
            [(1, 4, 8), (1, 1, 4, 4)],
        )

    def test_one_block_keeps_the_original_module_boundary(self) -> None:
        workload = build_paper_benchmark("bert-base", variant="micro")
        self.assertEqual(type(workload.module).__name__, "BertBaseOneBlock")
        self.assertEqual(workload.attributes["materialized_scope"], "one_block")
        self.assertEqual(workload.attributes["paper_evaluation_scope"], "full_model")
        self.assertEqual(workload.attributes["reference_layer_count"], 12)

    def test_resnet_depth_proxy_requires_model_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_scope='model_proxy'"):
            build_paper_benchmark("resnet50", variant="micro", layer_count=2)

    def test_transformer_model_proxy_materializes_embeddings_mask_and_head(self) -> None:
        workload = build_paper_benchmark(
            "llama2-13b-oneblk",
            variant="micro",
            layer_count=2,
            model_scope="model_proxy",
        )
        self.assertEqual(workload.attributes["materialized_scope"], "model_proxy")
        self.assertEqual(workload.attributes["embedding"], "token")
        self.assertEqual(workload.attributes["causal_mask"], "additive_upper_triangle")
        self.assertTrue(workload.attributes["output_head"])
        self.assertEqual([tuple(value.shape) for value in workload.inputs], [(1, 4)])
        output = workload.module(*workload.inputs)
        self.assertEqual(tuple(output.shape), (1, 4, 32))
        self.assertGreater(workload.attributes["model_statistics"]["parameter_count"], 0)
        self.assertEqual(workload.attributes["model_components"]["transformer_blocks"], 2)

    def test_bert_model_proxy_materializes_three_embedding_inputs(self) -> None:
        workload = build_paper_benchmark(
            "bert-base",
            variant="micro",
            model_scope="model_proxy",
        )
        self.assertEqual(
            [tuple(value.shape) for value in workload.inputs],
            [(1, 4), (1, 4), (1, 4)],
        )
        self.assertEqual(workload.attributes["remaining_features"], ["exact_gelu", "full_model_depth"])
        output = workload.module(*workload.inputs)
        self.assertEqual(tuple(output.shape), (1, 4, 8))

    def test_deepseek_moe_proxy_consumes_sparse_external_routing_weights(self) -> None:
        workload = build_paper_benchmark(
            "deepseek-r1-16b-prefill",
            variant="micro",
            deepseek_mode="moe_proxy",
        )
        routing = workload.inputs[-1]
        self.assertEqual(tuple(routing.shape), (1, 4, 4))
        self.assertTrue(workload.attributes["moe"]["enabled"])
        self.assertEqual(workload.attributes["moe"]["top_k"], 2)
        self.assertEqual(
            workload.attributes["moe"]["dispatch_contract"],
            "mask_weighted_all_experts",
        )
        self.assertEqual(workload.attributes["model_statistics"]["routing_nonzero_count"], 8)
        self.assertIn("dynamic_topk_selection", workload.attributes["remaining_features"])
        self.assertIn("token_compaction_and_capacity", workload.attributes["remaining_features"])
        output = workload.module(*workload.inputs)
        self.assertEqual(tuple(output.shape), (1, 4, 8))

    def test_deepseek_decode_is_one_token_with_explicit_cache_state(self) -> None:
        workload = build_paper_benchmark(
            "deepseek-r1-16b-decode",
            variant="micro",
            model_scope="model_proxy",
            deepseek_mode="moe_proxy",
        )
        self.assertEqual(tuple(workload.inputs[0].shape), (1, 1))
        self.assertEqual(tuple(workload.inputs[1].shape), (1, 2, 4, 4))
        self.assertEqual(workload.attributes["phase"], "decode")
        self.assertEqual(workload.attributes["kv_cache"]["update_length"], 1)
        output, key_cache, value_cache = workload.module(*workload.inputs)
        self.assertEqual(tuple(output.shape), (1, 1, 32))
        self.assertEqual(tuple(key_cache.shape), (1, 2, 4, 4))
        self.assertEqual(tuple(value_cache.shape), (1, 2, 4, 4))

    def test_deepseek_decode_rejects_shared_cache_across_repeated_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "each layer owns distinct KV state"):
            build_paper_benchmark(
                "deepseek-r1-16b-decode",
                variant="micro",
                layer_count=2,
            )

    def test_resnet_model_proxy_materializes_stem_repetition_pool_and_head(self) -> None:
        workload = build_paper_benchmark(
            "resnet50",
            variant="micro",
            layer_count=2,
            model_scope="model_proxy",
        )
        self.assertEqual(workload.attributes["materialized_scope"], "model_proxy")
        self.assertEqual(len(workload.module.blocks), 2)
        output = workload.module(*workload.inputs)
        self.assertEqual(tuple(output.shape), (1, 10))
        self.assertEqual(workload.attributes["model_components"]["residual_bottlenecks"], 2)
        self.assertTrue(workload.attributes["model_components"]["classification_head"])



@unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
class PaperBenchmarkFrontendTest(unittest.TestCase):
    def test_deepseek_decode_request_replay_preserves_two_cache_states(self) -> None:
        workload = build_paper_benchmark(
            "deepseek-r1-16b-decode",
            variant="micro",
            model_scope="model_proxy",
        )
        compiled = compile_torch_module(
            workload.module,
            workload.inputs,
            minimal_machine_config(),
            model_id="deepseek-decode-request-replay",
            tile_size=4,
        )
        cache_updates = [
            operator
            for operator in compiled.graph.operators
            if operator.normalized_type == "kv_cache_update"
        ]
        self.assertEqual(len(cache_updates), 2)
        buffers = allocate_memory_plan_bindings(
            compiled.backend_artifact.memory_plan, minimal_machine_config()
        )
        cases = run_runtime_device_matrix(
            compiled.backend_artifact,
            buffers,
            minimal_machine_config(),
            runtime_policies=("static",),
            device_policies=(SchedulerPolicy.DYNAMIC_READY_QUEUE,),
            request_count=2,
            inter_request_gap_cycles=3,
        )
        self.assertEqual(len(cases), 1)
        self.assertIsInstance(cases[0].submission, RuntimeSequence)
        self.assertEqual(len(cases[0].submission.state_registry.state_ids()), 2)
        self.assertEqual(len(cases[0].submission.dependencies), 1)
        self.assertEqual(cases[0].result.metrics["invocation_count"], 2)
        self.assertEqual(cases[0].result.metrics["state_dependency_count"], 1)
        self.assertEqual(cases[0].result.metrics["inter_invocation_gap_cycles"], 3)

    def test_model_proxy_embeddings_and_repeated_depth_reach_tisa(self) -> None:
        one_block = build_paper_benchmark(
            "bert-base",
            variant="micro",
            model_scope="model_proxy",
        )
        repeated = build_paper_benchmark(
            "bert-base",
            variant="micro",
            model_scope="model_proxy",
            layer_count=2,
        )
        compiled_one = compile_torch_module(
            one_block.module,
            one_block.inputs,
            minimal_machine_config(),
            model_id="bert-model-proxy-one",
            tile_size=4,
        )
        compiled_repeated = compile_torch_module(
            repeated.module,
            repeated.inputs,
            minimal_machine_config(),
            model_id="bert-model-proxy-two",
            tile_size=4,
        )

        embedding_ops = [
            operator
            for operator in compiled_one.graph.operators
            if operator.normalized_type == "embedding"
        ]
        self.assertEqual(len(embedding_ops), 3)
        self.assertTrue(
            any(item.op_type == "gather" for item in compiled_one.tisa_program.instructions)
        )
        self.assertGreater(
            len(compiled_repeated.tisa_program.instructions),
            len(compiled_one.tisa_program.instructions),
        )
        self.assertEqual(compiled_one.validate(), ())
        self.assertEqual(compiled_repeated.validate(), ())

    def test_deepseek_moe_proxy_expert_branches_reach_tisa(self) -> None:
        workload = build_paper_benchmark(
            "deepseek-r1-16b-prefill",
            variant="micro",
            model_scope="model_proxy",
            deepseek_mode="moe_proxy",
        )
        compiled = compile_torch_module(
            workload.module,
            workload.inputs,
            minimal_machine_config(),
            model_id="deepseek-moe-proxy",
            tile_size=4,
        )

        matmul_ops = [
            operator
            for operator in compiled.graph.operators
            if operator.normalized_type in {"matmul", "batched_matmul", "gemv"}
        ]
        self.assertGreaterEqual(len(matmul_ops), 12)
        self.assertTrue(
            any(item.op_type == "gather" for item in compiled.tisa_program.instructions)
        )
        moe_regions = [
            region
            for region in compiled.graph.attributes.get("semantic_regions", ())
            if region.get("semantic_family") == "moe_dispatch"
        ]
        self.assertEqual(len(moe_regions), 1)
        self.assertGreaterEqual(len(moe_regions[0]["roles"]["expert_weights"]), 2)
        self.assertTrue(
            any(
                item.attributes.get("semantic_region_family") == "moe_dispatch"
                for item in compiled.tisa_program.instructions
            )
        )
        self.assertEqual(compiled.validate(), ())

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
        batch_norm_operator = next(
            operator
            for operator in compiled.graph.operators
            if operator.normalized_type == "batch_norm"
        )
        batch_norm_instruction = next(
            instruction
            for instruction in compiled.tisa_program.instructions
            if instruction.operator_id == batch_norm_operator.op_id
            and instruction.op_type == "load"
        )
        statistics_operands = [
            operand
            for operand in batch_norm_instruction.operands
            if operand.tile_mem.tensor in batch_norm_operator.inputs[1:]
        ]
        self.assertTrue(statistics_operands)
        self.assertTrue(all(operand.tile_mem.offset_bytes is not None for operand in statistics_operands))
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
        bindings = allocate_memory_plan_bindings(
            compiled.backend_artifact.memory_plan, minimal_machine_config()
        )
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
