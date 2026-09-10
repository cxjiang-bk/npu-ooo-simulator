import unittest

from npu_ooo.compiler import (
    SemanticFusionPattern,
    SemanticFusionPatternRegistry,
    MoEDispatchRegionPass,
    SoftmaxFusionPass,
    default_pass_manager,
    default_semantic_fusion_registry,
)
from npu_ooo.ir import DataEdge, OperatorGraph, OperatorSpec, TensorSpec


class SemanticFusionPatternRegistryTest(unittest.TestCase):
    def test_moe_region_keeps_router_expert_and_combine_members_visible(self) -> None:
        tensor_shapes = {
            "router_logits": (2, 4),
            "routing_mask": (2, 4),
            "router_probs": (2, 4),
            "routed": (2, 4),
            "weight0": (2, 1),
            "weight1": (2, 1),
            "x": (2, 8),
            "expert_weight0": (8, 8),
            "expert_weight1": (8, 8),
            "expert0": (2, 8),
            "expert1": (2, 8),
            "weighted0": (2, 8),
            "weighted1": (2, 8),
            "combined": (2, 8),
        }
        operators = (
            OperatorSpec("router", "softmax", ("router_logits",), ("router_probs",), (("d0", 2),), (("d1", 4),)),
            OperatorSpec("mask", "elementwise", ("router_probs", "routing_mask"), ("routed",), (("d0", 2), ("d1", 4)), attributes={"frontend_target": "stablehlo.multiply"}),
            OperatorSpec("slice0", "slice", ("routed",), ("weight0",), (("d0", 2), ("d1", 1))),
            OperatorSpec("slice1", "slice", ("routed",), ("weight1",), (("d0", 2), ("d1", 1))),
            OperatorSpec("expert0", "matmul", ("x", "expert_weight0"), ("expert0",), (("M", 2), ("N", 8)), (("K", 8),)),
            OperatorSpec("expert1", "matmul", ("x", "expert_weight1"), ("expert1",), (("M", 2), ("N", 8)), (("K", 8),)),
            OperatorSpec("weighted0", "elementwise", ("weight0", "expert0"), ("weighted0",), (("d0", 2), ("d1", 8)), attributes={"frontend_target": "stablehlo.multiply"}),
            OperatorSpec("weighted1", "elementwise", ("weight1", "expert1"), ("weighted1",), (("d0", 2), ("d1", 8)), attributes={"frontend_target": "stablehlo.multiply"}),
            OperatorSpec("combine", "elementwise", ("weighted0", "weighted1"), ("combined",), (("d0", 2), ("d1", 8)), attributes={"frontend_target": "stablehlo.add"}),
        )
        producer = {
            output: operation.op_id
            for operation in operators
            for output in operation.outputs
        }
        edges = tuple(
            DataEdge(producer[tensor], operation.op_id, tensor)
            for operation in operators
            for tensor in operation.inputs
            if tensor in producer
        )
        graph = OperatorGraph(
            "moe-region-test",
            tuple(TensorSpec(name, shape, "f32") for name, shape in tensor_shapes.items()),
            operators,
            edges,
        )

        result = MoEDispatchRegionPass().run(graph)

        regions = result.graph.attributes["semantic_regions"]
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["semantic_family"], "moe_dispatch")
        self.assertEqual(set(regions[0]["roles"]["expert_outputs"]), {"expert0", "expert1"})
        roles = {
            operation.op_id: operation.attributes.get("semantic_region_role")
            for operation in result.graph.operators
        }
        self.assertEqual(roles["router"], "router_softmax")
        self.assertEqual(roles["mask"], "topk_mask")
        self.assertEqual(roles["weighted0"], "expert_weight")
        self.assertEqual(roles["combine"], "combine")
        self.assertEqual(len(result.graph.operators), len(graph.operators))

    def test_default_registry_exposes_existing_semantic_patterns(self) -> None:
        patterns = default_semantic_fusion_registry().patterns()

        self.assertEqual(
            [pattern.name for pattern in patterns],
            [
                "recover_stablehlo_layernorm",
                "fuse_layernorm",
                "fuse_rmsnorm",
                "fuse_softmax",
                "recover_stablehlo_kv_cache",
                "recover_rotary_embedding",
                "recover_attention_region",
                "fuse_swiglu",
                "recover_moe_dispatch_region",
            ],
        )
        self.assertEqual(
            [pattern.semantic_family for pattern in patterns],
            [
                "layernorm",
                "layernorm",
                "rmsnorm",
                "softmax",
                "kv_cache",
                "rotary_embedding",
                "attention",
                "swiglu",
                "moe_dispatch",
            ],
        )
        self.assertEqual(
            [graph_pass.name for graph_pass in default_semantic_fusion_registry().create_passes()],
            [pattern.name for pattern in patterns],
        )

    def test_duplicate_pattern_name_is_rejected(self) -> None:
        pattern = SemanticFusionPattern(
            name="fuse_softmax",
            semantic_family="softmax",
            graph_pass=SoftmaxFusionPass(),
        )
        registry = SemanticFusionPatternRegistry((pattern,))

        with self.assertRaisesRegex(ValueError, "duplicate semantic fusion pattern"):
            registry.register(pattern)

    def test_equal_priorities_keep_registration_order(self) -> None:
        first = SemanticFusionPattern(
            name="first",
            semantic_family="test",
            graph_pass=_NoOpPass("first"),
            priority=10,
        )
        second = SemanticFusionPattern(
            name="second",
            semantic_family="test",
            graph_pass=_NoOpPass("second"),
            priority=10,
        )
        earlier = SemanticFusionPattern(
            name="earlier",
            semantic_family="test",
            graph_pass=_NoOpPass("earlier"),
            priority=5,
        )

        registry = SemanticFusionPatternRegistry((first, second, earlier))

        self.assertEqual(
            [pattern.name for pattern in registry.patterns()],
            ["earlier", "first", "second"],
        )

    def test_default_pass_manager_preserves_gc_pipeline_order(self) -> None:
        self.assertEqual(
            [graph_pass.name for graph_pass in default_pass_manager().passes],
            [
                "canonicalize",
                "decompose_linear",
                "recover_stablehlo_layernorm",
                "recover_stablehlo_flattened_linear",
                "fold_transpose_into_matmul",
                "fuse_layernorm",
                "fuse_rmsnorm",
                "fuse_softmax",
                "recover_stablehlo_kv_cache",
                "recover_rotary_embedding",
                "recover_attention_region",
                "fuse_swiglu",
                "recover_moe_dispatch_region",
            ],
        )

    def test_custom_registry_controls_only_semantic_passes(self) -> None:
        registry = SemanticFusionPatternRegistry(
            (
                SemanticFusionPattern(
                    name="custom_semantic",
                    semantic_family="custom",
                    graph_pass=_NoOpPass("custom_semantic"),
                    priority=35,
                ),
            )
        )

        self.assertEqual(
            [graph_pass.name for graph_pass in default_pass_manager(fusion_registry=registry).passes],
            [
                "canonicalize",
                "decompose_linear",
                "recover_stablehlo_flattened_linear",
                "custom_semantic",
                "fold_transpose_into_matmul",
            ],
        )


class _NoOpPass:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, graph):
        raise AssertionError("registry ordering tests must not execute graph passes")


if __name__ == "__main__":
    unittest.main()
