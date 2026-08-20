import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_transformer_block_case, build_transformer_block_model
from npu_ooo.ir import default_mixed_schedule
from npu_ooo.lowering import lower_mixed_model
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class TransformerBlockTest(unittest.TestCase):
    def test_block_graph_lowers_all_attention_and_mlp_stages(self) -> None:
        model = build_transformer_block_model(tokens=32, sequence=32, head_dim=16, intermediate=32)
        instance = model.instantiate(build_transformer_block_case(tokens=32))
        lowered = lower_mixed_model(instance, minimal_machine_config(), default_mixed_schedule(instance.graph))
        self.assertEqual(lowered.execution_graph.validate(), ())
        primitives = {task.primitive for task in lowered.execution_graph.tasks}
        self.assertTrue({"layernorm", "matmul", "reduce_max", "normalize", "elementwise"}.issubset(primitives))
        self.assertEqual(len(instance.graph.operators), 9)
        self.assertEqual(lowered.statistics["cross_operator_dependency_count"], 8)

    def test_block_runs_dynamic_and_completes_every_tile(self) -> None:
        model = build_transformer_block_model(tokens=32, sequence=32, head_dim=16, intermediate=32)
        instance = model.instantiate(build_transformer_block_case(tokens=32))
        lowered = lower_mixed_model(instance, minimal_machine_config(), default_mixed_schedule(instance.graph))
        result = schedule_execution_graph(
            lowered.execution_graph,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(result.metrics["completed_tile_count"], len(lowered.tile_graph.tiles))
        self.assertGreater(result.total_cycles, 0)


if __name__ == "__main__":
    unittest.main()
