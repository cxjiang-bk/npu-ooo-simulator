import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_layernorm_case, build_layernorm_model
from npu_ooo.ir import default_layernorm_schedule
from npu_ooo.lowering import lower_layernorm
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class LayerNormLoweringTest(unittest.TestCase):
    def test_layernorm_keeps_mean_and_variance_barriers(self) -> None:
        model = build_layernorm_model(rows=64, cols=96)
        instance = model.instantiate(build_layernorm_case())
        lowered = lower_layernorm(instance, minimal_machine_config(), default_layernorm_schedule(instance.graph))
        self.assertEqual(lowered.statistics["tile_count"], 6)
        self.assertEqual(lowered.statistics["task_count"], 38)
        self.assertEqual(lowered.execution_graph.validate(), ())
        tasks = {task.task_id: task for task in lowered.execution_graph.tasks}
        means = [task for task in lowered.execution_graph.tasks if task.primitive == "layernorm_mean"]
        self.assertEqual(len(means), 2)
        self.assertIn("layernorm0.t0002.reduce_sum", means[0].predecessors)
        normalize = tasks["layernorm0.t0000.layernorm"]
        self.assertTrue(any("reduce_sum_square" in pred for pred in normalize.predecessors))

    def test_layernorm_runs_through_dynamic_scheduler(self) -> None:
        model = build_layernorm_model(rows=64, cols=96)
        instance = model.instantiate(build_layernorm_case())
        lowered = lower_layernorm(instance, minimal_machine_config(), default_layernorm_schedule(instance.graph))
        result = schedule_execution_graph(
            lowered.execution_graph,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(result.metrics["completed_tile_count"], 6)
        self.assertGreater(result.metrics["resource_busy_cycles"]["ARU"], 0)


if __name__ == "__main__":
    unittest.main()
