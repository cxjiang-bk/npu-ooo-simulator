import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.ir import AccessType, BufferRegion, ExecutionGraph, ExecutionTask
from npu_ooo.scheduler import SchedulerPolicy, SimulatorConfig, schedule_execution_graph


class EventSimulatorTest(unittest.TestCase):
    def test_dynamic_critical_path_priority_changes_issue_order(self) -> None:
        graph = ExecutionGraph(
            graph_id="critical_path_micro",
            tasks=(
                ExecutionTask(
                    task_id="short_dma",
                    tile_id="tile_short",
                    operator_id="micro",
                    primitive="load",
                    resource="DMA",
                    duration_cycles=10,
                    program_order=0,
                ),
                ExecutionTask(
                    task_id="critical_dma",
                    tile_id="tile_critical",
                    operator_id="micro",
                    primitive="load",
                    resource="DMA",
                    duration_cycles=10,
                    program_order=1,
                ),
                ExecutionTask(
                    task_id="critical_compute",
                    tile_id="tile_critical",
                    operator_id="micro",
                    primitive="matmul",
                    resource="MXU",
                    duration_cycles=100,
                    predecessors=("critical_dma",),
                    program_order=2,
                ),
            ),
        )
        machine = minimal_machine_config()
        static = schedule_execution_graph(graph, machine, SchedulerPolicy.STATIC_PIPELINE)
        dynamic = schedule_execution_graph(graph, machine, SchedulerPolicy.DYNAMIC_READY_QUEUE)
        self.assertEqual(static.timing("short_dma").issue, 0)
        self.assertEqual(static.timing("critical_dma").issue, 1)
        self.assertEqual(dynamic.timing("critical_dma").issue, 0)
        self.assertEqual(dynamic.timing("short_dma").issue, 1)
        self.assertLess(dynamic.total_cycles, static.total_cycles)

    def test_window_one_limits_runtime_out_of_order_overlap(self) -> None:
        graph = ExecutionGraph(
            graph_id="window_micro",
            tasks=(
                ExecutionTask("a", "tile_a", "micro", "load", "DMA", duration_cycles=5, program_order=0),
                ExecutionTask("b", "tile_b", "micro", "load", "DMA", duration_cycles=5, program_order=1),
                ExecutionTask("c", "tile_b", "micro", "matmul", "MXU", duration_cycles=5, predecessors=("b",), program_order=2),
            ),
        )
        machine = minimal_machine_config()
        open_result = schedule_execution_graph(graph, machine, SchedulerPolicy.DYNAMIC_READY_QUEUE)
        constrained_result = schedule_execution_graph(
            graph,
            machine,
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=SimulatorConfig(
                instruction_queue_depth=1,
                rob_entries=1,
                max_inflight_tiles=1,
                dependency_window=1,
                ready_queue_depth=1,
            ),
        )
        self.assertGreaterEqual(constrained_result.total_cycles, open_result.total_cycles)
        self.assertEqual(constrained_result.metrics["rob_peak"], 1)
        self.assertLessEqual(constrained_result.metrics["visible_ready_peak"], 1)

    def test_address_scoreboard_blocks_cross_resource_raw_hazard(self) -> None:
        produced = BufferRegion(
            tensor="X",
            memory="SRAM",
            shape=(4,),
            starts=(0,),
            access=AccessType.WRITE,
            size_bytes=8,
        )
        consumed = BufferRegion(
            tensor="X",
            memory="SRAM",
            shape=(4,),
            starts=(0,),
            access=AccessType.READ,
            size_bytes=8,
        )
        graph = ExecutionGraph(
            graph_id="address_micro",
            tasks=(
                ExecutionTask(
                    "writer",
                    "tile_writer",
                    "micro",
                    "store",
                    "DMA",
                    writes=(produced,),
                    duration_cycles=20,
                    program_order=0,
                ),
                ExecutionTask(
                    "reader",
                    "tile_reader",
                    "micro",
                    "matmul",
                    "MXU",
                    reads=(consumed,),
                    duration_cycles=1,
                    program_order=1,
                ),
            ),
        )
        machine = minimal_machine_config()
        without_scoreboard = schedule_execution_graph(graph, machine, SchedulerPolicy.STATIC_PIPELINE)
        with_scoreboard = schedule_execution_graph(
            graph,
            machine,
            SchedulerPolicy.STATIC_PIPELINE,
            simulator_config=SimulatorConfig(address_scoreboard=True),
        )
        self.assertEqual(without_scoreboard.timing("reader").issue, 0)
        self.assertEqual(with_scoreboard.timing("reader").issue, 20)
        self.assertEqual(with_scoreboard.timing("reader").start, 20)
        self.assertEqual(with_scoreboard.metrics["address_dependency_count"], 1)
        self.assertGreater(with_scoreboard.total_cycles, without_scoreboard.total_cycles)


if __name__ == "__main__":
    unittest.main()
