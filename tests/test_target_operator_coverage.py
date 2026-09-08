import unittest
from dataclasses import replace

from npu_ooo.arch import lpu_like_machine_config, minimal_machine_config
from npu_ooo.compiler import compile_operator_graph
from npu_ooo.frontend import FrontendImport, OfficialStableHLOModule
from npu_ooo.ir import OperatorGraph, OperatorSpec, TensorSpec
from npu_ooo.lowering import default_lowering_registry


class TargetOperatorCoverageTest(unittest.TestCase):
    def test_every_registered_operator_has_an_explicit_target_rule(self):
        specialized = {"matmul", "batched_matmul", "gemv"}
        supported = set(default_lowering_registry().supported_types)
        for machine in (minimal_machine_config(), lpu_like_machine_config()):
            generic = {
                operation
                for rule in machine.operation_class_placements
                for operation in rule.operations
            }
            self.assertEqual(supported, specialized | generic)
            self.assertEqual(
                len(generic),
                sum(len(rule.operations) for rule in machine.operation_class_placements),
            )

    def test_missing_target_class_is_an_explicit_compile_error(self):
        graph = OperatorGraph(
            "missing-target-rule",
            (TensorSpec("x", (4,)), TensorSpec("y", (4,))),
            (
                OperatorSpec(
                    "neg",
                    "elementwise",
                    ("x",),
                    ("y",),
                    (("d0", 4),),
                    attributes={"semantic_op": "negate"},
                ),
            ),
        )
        frontend = FrontendImport(
            graph=graph,
            model_id=graph.graph_id,
            variant="test",
            frontend="stablehlo",
        )
        stablehlo = OfficialStableHLOModule(
            text="module {}", canonical_text="module {}", model_id=graph.graph_id
        )
        machine = replace(minimal_machine_config(), operation_class_placements=())
        with self.assertRaisesRegex(ValueError, "no explicit target class.*elementwise"):
            compile_operator_graph(
                graph,
                machine,
                frontend=frontend,
                source_frontend=frontend,
                stablehlo=stablehlo,
            )


if __name__ == "__main__":
    unittest.main()
