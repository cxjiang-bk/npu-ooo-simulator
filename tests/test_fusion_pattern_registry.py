import unittest

from npu_ooo.compiler import (
    SemanticFusionPattern,
    SemanticFusionPatternRegistry,
    SoftmaxFusionPass,
    default_pass_manager,
    default_semantic_fusion_registry,
)


class SemanticFusionPatternRegistryTest(unittest.TestCase):
    def test_default_registry_exposes_existing_semantic_patterns(self) -> None:
        patterns = default_semantic_fusion_registry().patterns()

        self.assertEqual(
            [pattern.name for pattern in patterns],
            [
                "recover_stablehlo_layernorm",
                "fuse_layernorm",
                "fuse_rmsnorm",
                "fuse_softmax",
                "recover_rotary_embedding",
                "recover_attention_region",
                "fuse_swiglu",
            ],
        )
        self.assertEqual(
            [pattern.semantic_family for pattern in patterns],
            [
                "layernorm",
                "layernorm",
                "rmsnorm",
                "softmax",
                "rotary_embedding",
                "attention",
                "swiglu",
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
                "recover_rotary_embedding",
                "recover_attention_region",
                "fuse_swiglu",
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
