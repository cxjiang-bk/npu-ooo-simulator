import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_attention_case, build_attention_model
from npu_ooo.ir import default_mixed_schedule
from npu_ooo.lowering import lower_mixed_model
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class AttentionFragmentTest(unittest.TestCase):
    def test_qk_softmax_pv_graph_has_cross_stage_handoffs(self) -> None:
        model = build_attention_model(query_tokens=64, key_tokens=64, head_dim=32)
        instance = model.instantiate(build_attention_case())
        lowered = lower_mixed_model(instance, minimal_machine_config(), default_mixed_schedule(instance.graph))
        self.assertEqual(lowered.execution_graph.validate(), ())
        self.assertEqual(lowered.statistics["tile_count"], 12)
        self.assertGreater(lowered.statistics["cross_operator_dependency_count"], 0)
        primitives = {task.primitive for task in lowered.execution_graph.tasks}
        self.assertTrue({"matmul", "reduce_max", "exp", "reduce_sum", "normalize"}.issubset(primitives))
        softmax_loads = [
            task
            for task in lowered.execution_graph.tasks
            if task.operator_id == "attention_softmax" and task.primitive == "load"
        ]
        self.assertTrue(softmax_loads)
        self.assertTrue(
            all(
                any(pred.startswith("attention_scores.") and pred.endswith(".store") for pred in task.predecessors)
                for task in softmax_loads
            )
        )

    def test_attention_runs_in_dynamic_scheduler(self) -> None:
        model = build_attention_model(query_tokens=32, key_tokens=32, head_dim=16)
        instance = model.instantiate(build_attention_case(query_tokens=32))
        lowered = lower_mixed_model(instance, minimal_machine_config(), default_mixed_schedule(instance.graph))
        result = schedule_execution_graph(
            lowered.execution_graph,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(result.metrics["completed_tile_count"], len(lowered.tile_graph.tiles))
        self.assertGreater(result.metrics["resource_busy_cycles"]["MXU"], 0)
        self.assertGreater(result.metrics["resource_busy_cycles"]["ARU"], 0)


if __name__ == "__main__":
    unittest.main()
