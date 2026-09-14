import unittest
from dataclasses import replace

from npu_ooo.arch import lpu_like_machine_config, minimal_machine_config
from npu_ooo.compiler import compile_operator_graph
from npu_ooo.frontend import FrontendImport, OfficialStableHLOModule
from npu_ooo.ir import (
    DataEdge,
    OperatorGraph,
    OperatorSpec,
    TensorSpec,
    allocate_memory_plan_bindings,
    create_runtime_submission,
)
from npu_ooo.scheduler import SchedulerPolicy, schedule_tisa_program


def _attention_graph(*, fanout=False):
    tensors = [
        TensorSpec("q", (1, 4, 8), "f32"),
        TensorSpec("k", (1, 4, 8), "f32"),
        TensorSpec("scores", (1, 4, 4), "f32"),
        TensorSpec("prob", (1, 4, 4), "f32"),
    ]
    operators = [
        OperatorSpec(
            "qk",
            "batched_matmul",
            ("q", "k"),
            ("scores",),
            (("B0", 1), ("M", 4), ("N", 4)),
            (("K", 8),),
            attributes={"rhs_transposed": True},
        ),
        OperatorSpec(
            "softmax",
            "softmax",
            ("scores",),
            ("prob",),
            (("d0", 1), ("d1", 4)),
            (("d2", 4),),
            attributes={"axes": [2]},
        ),
    ]
    edges = [DataEdge("qk", "softmax", "scores")]
    if fanout:
        tensors.append(TensorSpec("prob2", (1, 4, 4), "f32"))
        operators.append(
            OperatorSpec(
                "softmax2",
                "softmax",
                ("scores",),
                ("prob2",),
                (("d0", 1), ("d1", 4)),
                (("d2", 4),),
                attributes={"axes": [2]},
            )
        )
        edges.append(DataEdge("qk", "softmax2", "scores"))
    return OperatorGraph("attention-onchip", tuple(tensors), tuple(operators), tuple(edges))


def _compile(machine, *, fanout=False):
    graph = _attention_graph(fanout=fanout)
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
    return compile_operator_graph(
        graph,
        machine,
        frontend=frontend,
        source_frontend=frontend,
        stablehlo=stablehlo,
        tile_size=4,
    )


class OnchipHandoffTest(unittest.TestCase):
    def test_attention_handoff_elides_root_roundtrip_and_binds_one_local_buffer(self):
        base_machine = lpu_like_machine_config()
        optimized_machine = replace(
            base_machine,
            attributes={
                **base_machine.attributes,
                "onchip_handoff_policy": "attention_single_consumer",
            },
        )
        baseline = _compile(base_machine)
        optimized = _compile(optimized_machine)
        self.assertEqual(
            baseline.virtual_tisa_program.to_dict(),
            optimized.virtual_tisa_program.to_dict(),
        )
        handoff, = optimized.backend_artifact.target_plan.attributes["onchip_handoffs"]
        self.assertEqual((handoff["producer"], handoff["consumer"]), ("qk", "softmax"))
        self.assertEqual(handoff["memory"], "UB")
        self.assertEqual(
            len(baseline.tisa_program.instructions) - len(optimized.tisa_program.instructions),
            2,
        )
        self.assertFalse(
            any(
                buffer.tensor == "scores" and buffer.memory == "GM"
                for buffer in optimized.backend_artifact.memory_plan.buffers
            )
        )

        submission = create_runtime_submission(
            optimized.backend_artifact,
            allocate_memory_plan_bindings(
                optimized.backend_artifact.memory_plan, optimized_machine
            ),
        )
        uses = [
            operand
            for operand in submission.operands
            if operand.buffer_id == handoff["buffer_id"]
        ]
        self.assertEqual({item.access_type for item in uses}, {"read", "write"})
        self.assertEqual(
            len({(item.address, item.address + item.size_bytes) for item in uses}), 1
        )

    def test_minimal_target_uses_sram_and_static_dynamic_share_package(self):
        machine = minimal_machine_config()
        machine = replace(
            machine,
            attributes={
                **machine.attributes,
                "onchip_handoff_policy": "attention_single_consumer",
            },
        )
        compiled = _compile(machine)
        handoff, = compiled.backend_artifact.target_plan.attributes["onchip_handoffs"]
        self.assertEqual(handoff["memory"], "SRAM")
        submission = create_runtime_submission(
            compiled.backend_artifact,
            allocate_memory_plan_bindings(compiled.backend_artifact.memory_plan, machine),
        )
        results = tuple(
            schedule_tisa_program(
                compiled.backend_artifact,
                machine,
                policy,
                runtime_submission=submission,
            )
            for policy in (
                SchedulerPolicy.STATIC_PIPELINE,
                SchedulerPolicy.DYNAMIC_READY_QUEUE,
            )
        )
        self.assertEqual(
            results[0].metrics["compile_package_sha256"],
            results[1].metrics["compile_package_sha256"],
        )
        self.assertEqual(
            {item.task_id for item in results[0].instruction_timings},
            {item.task_id for item in results[1].instruction_timings},
        )

    def test_fanout_rejects_optimization_and_keeps_root_handoff(self):
        machine = lpu_like_machine_config()
        machine = replace(
            machine,
            attributes={
                **machine.attributes,
                "onchip_handoff_policy": "attention_single_consumer",
            },
        )
        compiled = _compile(machine, fanout=True)
        plan = compiled.backend_artifact.target_plan
        self.assertEqual(plan.attributes["onchip_handoffs"], [])
        self.assertEqual(
            {item["reason"] for item in plan.attributes["onchip_handoff_rejections"]},
            {"fanout_is_not_one"},
        )
        self.assertTrue(
            any(
                buffer.tensor == "scores" and buffer.memory == "GM"
                for buffer in compiled.backend_artifact.memory_plan.buffers
            )
        )


if __name__ == "__main__":
    unittest.main()
