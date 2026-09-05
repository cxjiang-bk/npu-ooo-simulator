import unittest
from pathlib import Path
import tempfile

from npu_ooo.arch import minimal_machine_config
from npu_ooo.ir import AccessType, BufferRegion, ExecutionGraph, ExecutionTask
from npu_ooo.scheduler import (
    SchedulerPolicy,
    SimulatorConfig,
    StaticPipelineConfig,
    schedule_execution_graph,
)
from npu_ooo.simulator import TimingTableModel
from npu_ooo.trace import write_svg


class EventSimulatorTest(unittest.TestCase):
    def test_swimlane_svg_contains_primitive_legend_and_cycle_axis(self) -> None:
        graph = ExecutionGraph(
            graph_id="trace_micro",
            tasks=(
                ExecutionTask(
                    "load_a",
                    "tile0",
                    "micro",
                    "load_transpose",
                    "DMA",
                    duration_cycles=5,
                    program_order=0,
                ),
                ExecutionTask(
                    "compute",
                    "tile0",
                    "micro",
                    "matmul",
                    "MXU",
                    predecessors=("load_a",),
                    duration_cycles=12,
                    program_order=1,
                ),
            ),
        )
        result = schedule_execution_graph(
            graph,
            minimal_machine_config(),
            SchedulerPolicy.STATIC_PIPELINE,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "swimlane.svg"
            write_svg(result, output, width=800)
            svg = (output.parent / "07_trace" / output.name).read_text(encoding="utf-8")

        self.assertIn('id="legend"', svg)
        self.assertIn(">load_transpose</text>", svg)
        self.assertIn(">matmul</text>", svg)
        self.assertIn('id="cycle-axis"', svg)
        self.assertIn(">Cycle</text>", svg)
        self.assertIn(">5</text>", svg)
        self.assertIn("issue=", svg)

    def test_timing_table_overrides_primitive_and_keeps_backend_name(self) -> None:
        graph = ExecutionGraph(
            graph_id="timing_table_micro",
            tasks=(
                ExecutionTask(
                    "compute",
                    "tile0",
                    "micro",
                    "matmul",
                    "MXU",
                    duration_cycles=100,
                    initiation_interval_cycles=100,
                ),
            ),
        )
        result = schedule_execution_graph(
            graph,
            minimal_machine_config(),
            SchedulerPolicy.SEQUENTIAL,
            timing_model=TimingTableModel.from_dict(
                {
                    "name": "rtl_probe_v0",
                    "entries": {
                        "matmul": {
                            "duration_cycles": 7,
                            "initiation_interval_cycles": 2,
                        }
                    },
                }
            ),
        )
        self.assertEqual(result.backend, "rtl_probe_v0")
        self.assertEqual(result.total_cycles, 7)
        self.assertEqual(result.timing("compute").duration, 7)

    def test_static_pipeline_reservations_and_drain_match_hand_schedule(self) -> None:
        graph = ExecutionGraph(
            graph_id="static_dual_micro",
            tasks=(
                ExecutionTask(
                    "load0",
                    "tile0",
                    "micro",
                    "load",
                    "DMA",
                    duration_cycles=5,
                    stage_id=0,
                    program_order=0,
                    attributes={"iteration": 0},
                ),
                ExecutionTask(
                    "compute0",
                    "tile0",
                    "micro",
                    "matmul",
                    "MXU",
                    predecessors=("load0",),
                    duration_cycles=5,
                    stage_id=1,
                    program_order=1,
                    attributes={"iteration": 0},
                ),
                ExecutionTask(
                    "load1",
                    "tile1",
                    "micro",
                    "load",
                    "DMA",
                    duration_cycles=5,
                    stage_id=0,
                    program_order=2,
                    attributes={"iteration": 1},
                ),
                ExecutionTask(
                    "compute1",
                    "tile1",
                    "micro",
                    "matmul",
                    "MXU",
                    predecessors=("load1",),
                    duration_cycles=5,
                    stage_id=1,
                    program_order=3,
                    attributes={"iteration": 1},
                ),
            ),
        )
        static = schedule_execution_graph(
            graph,
            minimal_machine_config(),
            SchedulerPolicy.STATIC_PIPELINE,
            simulator_config=SimulatorConfig(
                static_pipeline=StaticPipelineConfig(
                    stage_count=2,
                    stage_offsets=(0, 10),
                    initiation_interval_cycles=20,
                )
            ),
        )
        self.assertEqual(static.timing("load0").issue, 0)
        self.assertEqual(static.timing("compute0").issue, 10)
        self.assertEqual(static.timing("load1").issue, 20)
        self.assertEqual(static.timing("compute1").issue, 30)
        self.assertEqual(static.total_cycles, 35)
        self.assertEqual(static.metrics["static_reservation_count"], 4)
        self.assertEqual(static.metrics["pipeline_drain_cycles"], 5)
        self.assertEqual(static.metrics["completed_tile_count"], 2)
        self.assertIn("resource_utilization", static.metrics)
        self.assertEqual(static.metrics["queue_peak_occupancy"]["rob"], 1)

    def test_static_triple_stage_reservation_uses_stage_two(self) -> None:
        graph = ExecutionGraph(
            graph_id="static_triple_micro",
            tasks=(
                ExecutionTask(
                    "stage0",
                    "tile0",
                    "micro",
                    "load",
                    "DMA",
                    duration_cycles=5,
                    stage_id=0,
                    attributes={"iteration": 0},
                ),
                ExecutionTask(
                    "stage1",
                    "tile1",
                    "micro",
                    "matmul",
                    "MXU",
                    predecessors=("stage0",),
                    duration_cycles=5,
                    stage_id=1,
                    attributes={"iteration": 0},
                ),
                ExecutionTask(
                    "stage2",
                    "tile2",
                    "micro",
                    "store",
                    "DMA",
                    predecessors=("stage1",),
                    duration_cycles=5,
                    stage_id=2,
                    attributes={"iteration": 0},
                ),
            ),
        )
        triple = schedule_execution_graph(
            graph,
            minimal_machine_config(),
            SchedulerPolicy.STATIC_PIPELINE,
            simulator_config=SimulatorConfig(
                static_pipeline=StaticPipelineConfig(
                    stage_count=3,
                    stage_offsets=(0, 10, 20),
                    initiation_interval_cycles=30,
                )
            ),
        )
        self.assertEqual([triple.timing(task_id).issue for task_id in ("stage0", "stage1", "stage2")], [0, 10, 20])
        self.assertEqual(triple.total_cycles, 25)
        self.assertEqual(triple.metrics["pipeline_drain_cycles"], 5)

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

    def test_instruction_queue_override_controls_visible_ready_depth(self) -> None:
        graph = ExecutionGraph(
            graph_id="instruction_queue_micro",
            tasks=tuple(
                ExecutionTask(
                    f"load_{index}",
                    f"tile_{index}",
                    "micro",
                    "load",
                    "DMA",
                    duration_cycles=1,
                    program_order=index,
                )
                for index in range(3)
            ),
        )
        result = schedule_execution_graph(
            graph,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=SimulatorConfig(instruction_queue_depth=1),
        )
        self.assertEqual(result.metrics["simulator_config"]["ready_queue_depth"], 1)
        self.assertLessEqual(result.metrics["visible_ready_peak"], 1)

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

    def test_address_scoreboard_blocks_war_and_waw_hazards(self) -> None:
        region = BufferRegion(
            tensor="X",
            memory="SRAM",
            shape=(4,),
            starts=(0,),
            access=AccessType.READ,
            size_bytes=8,
        )
        write = BufferRegion(
            tensor="X",
            memory="SRAM",
            shape=(4,),
            starts=(0,),
            access=AccessType.WRITE,
            size_bytes=8,
        )
        war_graph = ExecutionGraph(
            graph_id="address_war_micro",
            tasks=(
                ExecutionTask("reader", "tile_r", "micro", "load", "MXU", reads=(region,), duration_cycles=10, program_order=0),
                ExecutionTask("writer", "tile_w", "micro", "store", "DMA", writes=(write,), duration_cycles=1, program_order=1),
            ),
        )
        waw_graph = ExecutionGraph(
            graph_id="address_waw_micro",
            tasks=(
                ExecutionTask("writer0", "tile_0", "micro", "store", "DMA", writes=(write,), duration_cycles=10, program_order=0),
                ExecutionTask("writer1", "tile_1", "micro", "store", "MXU", writes=(write,), duration_cycles=1, program_order=1),
            ),
        )
        machine = minimal_machine_config()
        war = schedule_execution_graph(
            war_graph,
            machine,
            SchedulerPolicy.STATIC_PIPELINE,
            simulator_config=SimulatorConfig(address_scoreboard=True),
        )
        waw = schedule_execution_graph(
            waw_graph,
            machine,
            SchedulerPolicy.STATIC_PIPELINE,
            simulator_config=SimulatorConfig(address_scoreboard=True),
        )
        self.assertEqual(war.timing("writer").issue, 10)
        self.assertEqual(waw.timing("writer1").issue, 10)
        self.assertEqual(war.metrics["address_hazards"][0]["kind"], "WAR")
        self.assertEqual(waw.metrics["address_hazards"][0]["kind"], "WAW")

    def test_trace_and_address_hazard_keep_dependency_provenance(self) -> None:
        write = BufferRegion(
            tensor="X",
            memory="SRAM",
            shape=(4,),
            starts=(0,),
            access=AccessType.WRITE,
            size_bytes=8,
        )
        read = BufferRegion(
            tensor="X",
            memory="SRAM",
            shape=(4,),
            starts=(0,),
            access=AccessType.READ,
            size_bytes=8,
        )
        graph = ExecutionGraph(
            graph_id="provenance_micro",
            tasks=(
                ExecutionTask(
                    "producer", "tile_p", "micro", "store", "DMA",
                    writes=(write,), duration_cycles=4, program_order=0,
                ),
                ExecutionTask(
                    "consumer", "tile_c", "micro", "load", "DMA",
                    reads=(read,), predecessors=("producer",), duration_cycles=1,
                    program_order=1,
                    attributes={
                        "dependency_provenance": {
                            "producer": {
                                "source": "gc_tile_dependency",
                                "edges": [{
                                    "tensor": "X",
                                    "kind": "region_data",
                                    "hazard_kind": "RAW",
                                    "condition": "full_region_ready",
                                    "provenance": {"source": "operator_graph_edge"},
                                }],
                            }
                        }
                    },
                ),
            ),
        )
        result = schedule_execution_graph(
            graph, minimal_machine_config(), SchedulerPolicy.STATIC_PIPELINE,
            simulator_config=SimulatorConfig(address_scoreboard=True),
        )
        complete = next(event for event in result.events if event.event == "COMPLETE" and event.task_id == "consumer")
        self.assertEqual(complete.details["dependencies"][0]["source"], "gc_tile_dependency")
        self.assertEqual(complete.details["dependencies"][0]["edges"][0]["condition"], "full_region_ready")


if __name__ == "__main__":
    unittest.main()
