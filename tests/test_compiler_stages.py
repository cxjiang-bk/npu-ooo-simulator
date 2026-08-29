import unittest
from dataclasses import replace

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_operator_graph
from npu_ooo.frontend import FrontendImport, OfficialStableHLOModule
from npu_ooo.ir import OperatorGraph, OperatorSpec, TensorSpec
from npu_ooo.scheduler import SchedulerPolicy, schedule_tisa_program


def _stable_frontend(graph: OperatorGraph) -> tuple[FrontendImport, OfficialStableHLOModule]:
    frontend = FrontendImport(
        graph=graph,
        model_id=graph.graph_id,
        variant="test",
        frontend="stablehlo",
    )
    stablehlo = OfficialStableHLOModule(
        text="module {}",
        canonical_text="module {}",
        model_id=graph.graph_id,
    )
    return frontend, stablehlo


class CompilerStageContractTest(unittest.TestCase):
    def test_gc_keeps_pass_snapshots_and_memory_plan_metadata(self) -> None:
        graph = OperatorGraph(
            graph_id="gc-metadata-test",
            tensors=(TensorSpec("x", (4, 8)), TensorSpec("y", (4, 8))),
            operators=(
                OperatorSpec(
                    op_id="softmax",
                    op_type="softmax",
                    inputs=("x",),
                    outputs=("y",),
                    iteration_dims=(("row", 4),),
                    reduction_dims=(("col", 8),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        machine = minimal_machine_config()
        machine = replace(
            machine,
            memory_levels=tuple(
                replace(level, capacity_bytes=16) if level.name == "SRAM" else level
                for level in machine.memory_levels
            ),
        )
        compiled = compile_operator_graph(
            graph,
            machine,
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=4,
        )

        snapshots = compiled.gc_artifact.pass_dumps
        self.assertTrue(snapshots)
        self.assertEqual(
            [snapshot.pass_index for snapshot in snapshots],
            list(range(len(snapshots))),
        )
        self.assertEqual(snapshots[0].input_graph.to_dict(), graph.to_dict())
        self.assertEqual(snapshots[-1].output_graph.to_dict(), compiled.gc_artifact.graph.to_dict())
        self.assertEqual(compiled.gc_artifact.attributes["pass_count"], len(snapshots))

        operator_schedule = compiled.schedule.for_operator("softmax")
        ping_pong = operator_schedule.attributes["ping_pong"]
        self.assertTrue(ping_pong["enabled"])
        self.assertEqual(ping_pong["buffer_count"], 2)
        self.assertEqual(operator_schedule.attributes["residency_overflow_tensors"], ["x", "y"])

    def test_gc_fc_and_tisa_generator_have_distinct_contracts(self) -> None:
        graph = OperatorGraph(
            graph_id="stage-test",
            tensors=(
                TensorSpec("lhs", (4, 4)),
                TensorSpec("rhs", (4, 4)),
                TensorSpec("out", (4, 4)),
            ),
            operators=(
                OperatorSpec(
                    op_id="mm",
                    op_type="matmul",
                    inputs=("lhs", "rhs"),
                    outputs=("out",),
                    iteration_dims=(("m", 4), ("n", 4)),
                    reduction_dims=(("k", 4),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=2,
        )

        self.assertEqual(compiled.gc_artifact.to_dict()["paper_stage"], "GC")
        self.assertEqual(compiled.tisa_dialect.to_dict()["paper_stage"], "FC")
        self.assertEqual(compiled.tisa_program.attributes["paper_stage"], "TISA_GENERATOR")
        self.assertEqual(compiled.validate(), ())

    def test_composite_payload_is_not_scheduler_visible(self) -> None:
        graph = OperatorGraph(
            graph_id="softmax-stage-test",
            tensors=(TensorSpec("x", (4, 8)), TensorSpec("y", (4, 8))),
            operators=(
                OperatorSpec(
                    op_id="softmax",
                    op_type="softmax",
                    inputs=("x",),
                    outputs=("y",),
                    iteration_dims=(("row", 4),),
                    reduction_dims=(("col", 8),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=4,
        )

        visible_ops = {instruction.op_type for instruction in compiled.tisa_program.instructions}
        payload_ops = {
            task.primitive for task in compiled.backend_artifact.execution_graph.tasks
        }
        self.assertIn("softmax", visible_ops)
        self.assertNotIn("reduce_max", visible_ops)
        self.assertTrue({"reduce_max", "exp", "reduce_sum", "normalize"}.issubset(payload_ops))
        readiness = {
            instruction.op_type: instruction.attributes["readiness_condition"]
            for instruction in compiled.tisa_program.instructions
            if instruction.tile_id == "softmax.t0000"
        }
        self.assertEqual(
            readiness,
            {
                "load": "input_region_ready",
                "softmax": "semantic_tile_ready",
                "store": "output_region_ready",
            },
        )
        dependency_conditions = {
            dependency.condition
            for instruction in compiled.tisa_program.instructions
            for dependency in instruction.dependencies
        }
        self.assertIn("semantic_tile_ready", dependency_conditions)
        self.assertEqual(compiled.backend_artifact.validate(), ())

    def test_online_softmax_uses_forward_state_payload(self) -> None:
        graph = OperatorGraph(
            graph_id="online-softmax-stage-test",
            tensors=(TensorSpec("x", (4, 8)), TensorSpec("y", (4, 8))),
            operators=(
                OperatorSpec(
                    op_id="softmax",
                    op_type="softmax",
                    inputs=("x",),
                    outputs=("y",),
                    iteration_dims=(("row", 4),),
                    reduction_dims=(("col", 8),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        machine = replace(
            minimal_machine_config(),
            attributes={"softmax_algorithm": "online", "calibration_status": "analytical"},
        )
        compiled = compile_operator_graph(
            graph,
            machine,
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=4,
        )

        compute = [
            instruction
            for instruction in compiled.tisa_program.instructions
            if instruction.op_type == "softmax"
        ]
        self.assertEqual(len(compute), 2)
        self.assertTrue(
            all(
                instruction.attributes["softmax_algorithm"] == "online"
                for instruction in compute
            )
        )
        self.assertTrue(
            all(
                instruction.attributes["payload_primitives"] == ["online_update"]
                for instruction in compute
            )
        )
        self.assertEqual(
            compute[1].dependencies[0].kind,
            "STATE",
        )
        payload_primitives = {
            task.primitive for task in compiled.backend_artifact.execution_graph.tasks
        }
        self.assertIn("online_update", payload_primitives)
        self.assertNotIn("reduce_max", payload_primitives)
        self.assertEqual(compiled.validate(), ())

        result = schedule_tisa_program(
            compiled.backend_artifact,
            machine,
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertGreater(result.total_cycles, 0)
        self.assertEqual(result.metrics["calibration_status"], "analytical")

    def test_planner_ranks_tile_size_candidates_and_records_costs(self) -> None:
        graph = OperatorGraph(
            graph_id="candidate-plan-test",
            tensors=(
                TensorSpec("lhs", (16, 16)),
                TensorSpec("rhs", (16, 16)),
                TensorSpec("out", (16, 16)),
            ),
            operators=(
                OperatorSpec(
                    op_id="mm",
                    op_type="matmul",
                    inputs=("lhs", "rhs"),
                    outputs=("out",),
                    iteration_dims=(("M", 16), ("N", 16)),
                    reduction_dims=(("K", 16),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=4,
            tile_size_candidates=(2, 4, 8),
        )

        candidate_costs = compiled.schedule.attributes["candidate_costs"]
        self.assertEqual(set(candidate_costs), {"2", "4", "8"})
        self.assertEqual(compiled.schedule.attributes["selected_tile_size"], 8)
        self.assertEqual(
            compiled.schedule.for_operator("mm").tile_size_map,
            {"M": 8, "N": 8, "K": 8},
        )
        self.assertEqual(
            compiled.schedule.attributes["selected_tile_size"],
            min(
                (int(size) for size, cost in candidate_costs.items()),
                key=lambda size: (candidate_costs[str(size)]["score"], size),
            ),
        )
        self.assertEqual(compiled.validate(), ())

    def test_scalar_elementwise_operand_preserves_rank_zero_metadata(self) -> None:
        graph = OperatorGraph(
            graph_id="scalar-elementwise-test",
            tensors=(
                TensorSpec("value", (4, 3), dtype="f32"),
                TensorSpec(
                    "scalar",
                    (),
                    dtype="f32",
                    attributes={"source_kind": "constant", "constant_value": 1.0},
                ),
                TensorSpec("out", (4, 3), dtype="f32"),
            ),
            operators=(
                OperatorSpec(
                    op_id="add",
                    op_type="elementwise",
                    inputs=("value", "scalar"),
                    outputs=("out",),
                    iteration_dims=(("m", 4), ("n", 3)),
                    attributes={
                        "semantic_op": "add",
                        "frontend_target": "stablehlo.add",
                        "operand_arity": 2,
                    },
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=2,
        )

        scalar_operands = [
            operand
            for instruction in compiled.tisa_program.instructions
            for operand in instruction.operands
            if operand.tile_mem.tensor == "scalar"
        ]
        self.assertTrue(scalar_operands)
        self.assertTrue(all(operand.tile_shape == () for operand in scalar_operands))
        self.assertTrue(all(operand.tile_mem.strides_bytes == () for operand in scalar_operands))
        self.assertTrue(all(operand.tile_mem.size_bytes == 4 for operand in scalar_operands))
        self.assertEqual(compiled.validate(), ())

    def test_opaque_layout_uses_conservative_tisa_interval(self) -> None:
        graph = OperatorGraph(
            graph_id="layout-metadata-test",
            tensors=(
                TensorSpec(
                    "value",
                    (2, 3),
                    dtype="f32",
                    attributes={
                        "layout_source": "stablehlo_encoding",
                        "layout_encoding": "#row_major",
                    },
                ),
                TensorSpec("out", (2, 3), dtype="f32"),
            ),
            operators=(
                OperatorSpec(
                    op_id="negate",
                    op_type="elementwise",
                    inputs=("value",),
                    outputs=("out",),
                    iteration_dims=(("m", 2), ("n", 3)),
                    attributes={
                        "semantic_op": "negate",
                        "frontend_target": "stablehlo.negate",
                        "operand_arity": 1,
                    },
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=2,
        )
        encoded = [
            operand.tile_mem
            for instruction in compiled.tisa_program.instructions
            for operand in instruction.operands
            if operand.tile_mem.tensor == "value"
        ]
        self.assertTrue(encoded)
        self.assertTrue(all(item.layout == "stablehlo:#row_major" for item in encoded))
        self.assertTrue(all(item.offset_bytes is None and item.size_bytes is None for item in encoded))
        self.assertTrue(all(item.strides_bytes is None for item in encoded))
        self.assertEqual(compiled.validate(), ())


if __name__ == "__main__":
    unittest.main()
