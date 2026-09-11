"""Hand-derived timelines for the clock-edge TISA scheduler."""

from dataclasses import replace
import csv
import contextlib
import io
import json
import random
from pathlib import Path
import tempfile
import unittest

from npu_ooo.arch import MachineConfig, SchedulerPipelineConfig, minimal_machine_config
from npu_ooo.backend import CycleEventBackend
from npu_ooo.cli import _simulation_config, build_parser, main
from npu_ooo.execution import AnalyticalExecutionBackend
from npu_ooo.ir import (
    BackendArtifact,
    BufferBinding,
    ExecutionGraph,
    ExecutionTask,
    MemoryBuffer,
    MemoryPlan,
    OperatorGraph,
    TensorSpec,
    TISADependency,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TargetInstructionPlan,
    TargetPlan,
    TileMem,
    UnitMap,
    create_runtime_submission,
    create_runtime_sequence,
    create_runtime_state_registry,
)
from npu_ooo.runtime import load_implicit_device_program
from npu_ooo.scheduler import schedule_tisa_program, schedule_tisa_sequence
from npu_ooo.scheduler.semantics import semantic_conflict
from npu_ooo.simulator import SimulatorConfig
from npu_ooo.simulator.cycle import schedule_loaded_cycle_program
from npu_ooo.trace import write_instruction_csv, write_json, write_svg
from npu_ooo.simulator.tisa import _access_banks


def artifact(rows):
    """Rows: (id, concrete unit, payload duration, predecessor ids)."""
    instructions, tasks, payloads = [], [], {}
    for tid, unit, duration, deps in rows:
        instructions.append(
            TISAInstruction(
                tid,
                f"tile.{tid}",
                "micro",
                "load" if unit == "DMA" else "matmul",
                (
                    TISAOperand(
                        tid,
                        (4,),
                        TileMem(tid, "SRAM", tensor=tid, offset_bytes=0, size_bytes=8),
                        "read",
                    ),
                ),
                UnitMap(unit),
                tuple(TISADependency(source) for source in deps),
                payload_ref=f"payload.{tid}",
            )
        )
        tasks.append(
            ExecutionTask(
                f"{tid}.task",
                f"tile.{tid}",
                "micro",
                "load" if unit == "DMA" else "matmul",
                unit,
                duration_cycles=duration,
                initiation_interval_cycles=1,
                program_order=len(tasks),
            )
        )
        payloads[tid] = (f"{tid}.task",)
    return BackendArtifact(
        "micro",
        TISAProgram("micro.tisa", tuple(instructions)),
        ExecutionGraph("micro.execution", tuple(tasks)),
        payloads,
    )


def run(
    program,
    *,
    machine=None,
    policy="dynamic_ready_queue",
    config=None,
    submission=None,
    **pipeline,
):
    cfg = config or SimulatorConfig(dynamic_priority="oldest_first")
    if pipeline:
        cfg = replace(cfg, pipeline=SchedulerPipelineConfig(**pipeline))
    return schedule_tisa_program(
        program,
        machine or minimal_machine_config(),
        policy,
        simulator_config=cfg,
        runtime_submission=submission,
        event_backend=CycleEventBackend(),
    )


class CycleSchedulerTest(unittest.TestCase):
    def test_continuous_stall_is_one_compact_interval(self):
        result = run(
            artifact(
                (
                    ("source", "DMA", 5, ()),
                    ("consumer", "MXU", 1, ("source",)),
                )
            ),
            receive_width=2,
            dispatch_width=2,
        )
        stalls = [
            event
            for event in result.events
            if event.event == "TISA_STALL"
            and event.task_id == "consumer"
            and event.details["reason"] == "dependency_wait"
        ]
        self.assertEqual(len(stalls), 1)
        details = stalls[0].details
        self.assertEqual(
            set(details),
            {"reason", "stage", "dependency_count", "start", "end", "duration"},
        )
        self.assertEqual(details["stage"], "wakeup")
        self.assertEqual(details["dependency_count"], 1)
        self.assertEqual(details["duration"], details["end"] - details["start"])
        self.assertEqual(
            details["duration"],
            result.metrics["stall_instruction_cycles"]["dependency_wait"],
        )
        perfetto_stall = next(
            event
            for event in result.perfetto_trace()["traceEvents"]
            if event.get("args", {}).get("reason") == "dependency_wait"
        )
        self.assertEqual(perfetto_stall["ph"], "X")
        self.assertEqual(perfetto_stall["dur"], details["duration"])

    def test_default_summary_contains_aggregates_and_timings_not_raw_trace(self):
        result = run(artifact((("a", "DMA", 2, ()),)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "06_simulation" / "summary.json"
            write_json(result, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary_schema_version"], 2)
        self.assertNotIn("events", payload)
        self.assertTrue(payload["timings"])
        self.assertTrue(payload["instruction_timings"])
        self.assertNotIn("queue_occupancy_timeline", payload["metrics"])
        self.assertNotIn("address_hazards", payload["metrics"])
        self.assertNotIn("semantic_conflicts", payload["metrics"])
        self.assertFalse(payload["trace"]["events_embedded"])
        self.assertEqual(payload["trace"]["event_count"], len(result.events))

    def test_default_swimlane_hides_duplicate_tisa_spans(self):
        result = run(artifact((("a", "DMA", 2, ()),)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "07_trace"
            physical = root / "physical.svg"
            detailed = root / "detailed.svg"
            write_svg(result, physical)
            write_svg(result, detailed, include_tisa_lanes=True)
            physical_text = physical.read_text(encoding="utf-8")
            detailed_text = detailed.read_text(encoding="utf-8")
        self.assertNotIn("TISA instruction</text>", physical_text)
        self.assertIn("TISA instruction</text>", detailed_text)

    def test_seeded_dags_preserve_dependencies_and_bandwidth_limits(self):
        for seed in range(20):
            rng = random.Random(seed)
            rows = [
                (
                    str(i),
                    rng.choice(("DMA", "MXU", "ARU")),
                    rng.randint(1, 5),
                    tuple(str(j) for j in range(i) if rng.random() < 0.3),
                )
                for i in range(6)
            ]
            program = artifact(rows)
            pipeline = dict(
                receive_width=2,
                dispatch_width=2,
                select_width=2,
                issue_width=2,
                wakeup_latency=seed % 3,
                completion_width=1,
                retire_width=1,
            )
            for policy in ("static_pipeline", "dynamic_ready_queue"):
                with self.subTest(seed=seed, policy=policy):
                    result = run(program, policy=policy, **pipeline)
                    stages = result.metrics["instruction_pipeline"]
                    for tid, unit, duration, deps in rows:
                        self.assertGreaterEqual(
                            stages[tid]["issued"], stages[tid]["selected"] + 1
                        )
                        for pred in deps:
                            self.assertGreaterEqual(
                                stages[tid]["wakeup"],
                                stages[pred]["completed"] + seed % 3,
                            )
                    retire_cycles = [stages[str(i)]["retired"] for i in range(6)]
                    self.assertEqual(retire_cycles, sorted(set(retire_cycles)))
                    for stage, width in (
                        ("issued", 2),
                        ("completed", 1),
                        ("retired", 1),
                    ):
                        cycles = [entry[stage] for entry in stages.values()]
                        self.assertTrue(
                            all(cycles.count(cycle) <= width for cycle in cycles)
                        )
                    for left in result.timings:
                        for right in result.timings:
                            if (
                                left.task_id != right.task_id
                                and left.resource == right.resource
                                and left.instance == right.instance
                            ):
                                self.assertTrue(
                                    left.finish <= right.start
                                    or right.finish <= left.start
                                )

    def test_bank_mapping_covers_element_crossing_bank_boundary_and_large_ranges(self):
        self.assertEqual(
            _access_banks(3, 4, (1,), (4,), bank_width=4, bank_count=4), (0, 1)
        )
        self.assertEqual(
            _access_banks(0, 10**12, (), (), bank_width=4, bank_count=4), (0, 1, 2, 3)
        )

    def test_registered_stages_and_raw_wakeup_latency(self):
        program = artifact(
            (("producer", "DMA", 2, ()), ("consumer", "MXU", 1, ("producer",)))
        )
        result = run(program)
        self.assertEqual(
            result.metrics["instruction_pipeline"]["producer"],
            {
                "received": 0,
                "dispatched": 1,
                "wakeup": 2,
                "selected": 2,
                "issued": 3,
                "done": 5,
                "completed": 5,
                "retired": 6,
            },
        )
        consumer = result.metrics["instruction_pipeline"]["consumer"]
        self.assertEqual(
            (
                consumer["wakeup"],
                consumer["selected"],
                consumer["issued"],
                consumer["retired"],
            ),
            (6, 6, 7, 9),
        )
        delayed = run(program, wakeup_latency=3)
        self.assertEqual(delayed.instruction_timing("consumer").issue, 9)
        self.assertEqual(result.total_cycles, 9)

    def test_completion_tag_broadcast_wakes_cross_unit_dependents(self):
        program = artifact(
            (
                ("source", "DMA", 2, ()),
                ("tensor", "MXU", 1, ("source",)),
                ("vector", "ARU", 1, ("source",)),
            )
        )
        result = run(
            program,
            receive_width=3,
            dispatch_width=3,
            select_width=3,
            issue_width=3,
        )
        source_complete = next(
            event
            for event in result.events
            if event.event == "TISA_COMPLETE" and event.task_id == "source"
        )
        self.assertEqual(
            source_complete.details["broadcast"], "global_completion_tag"
        )
        self.assertEqual(
            source_complete.details["broadcast_conditions"],
            ["full_region_ready"],
        )
        self.assertEqual(source_complete.details["matched_dependencies"], 2)
        self.assertEqual(source_complete.details["cross_unit_notifications"], 2)
        self.assertEqual(
            source_complete.details["dependency_ready_consumer_count"], 2
        )
        stages = result.metrics["instruction_pipeline"]
        for consumer in ("tensor", "vector"):
            self.assertEqual(
                stages[consumer]["wakeup"],
                stages["source"]["completed"] + 1,
            )
        self.assertEqual(
            result.metrics["dependency_tracking"],
            "global_completion_tag_broadcast+pending_mask",
        )
        self.assertEqual(result.metrics["broadcast_match_count"], 2)
        self.assertEqual(result.metrics["cross_unit_notification_count"], 2)

    def test_broadcast_pending_mask_waits_for_every_dependency(self):
        program = artifact(
            (
                ("short", "DMA", 2, ()),
                ("long", "ARU", 4, ()),
                ("join", "MXU", 1, ("short", "long")),
            )
        )
        result = run(
            program,
            receive_width=3,
            dispatch_width=3,
            select_width=3,
            issue_width=3,
        )
        stages = result.metrics["instruction_pipeline"]
        self.assertEqual(
            stages["join"]["wakeup"],
            max(stages["short"]["completed"], stages["long"]["completed"]) + 1,
        )
        broadcasts = {
            event.task_id: event.details
            for event in result.events
            if event.event == "TISA_COMPLETE"
            and event.task_id in {"short", "long"}
        }
        self.assertEqual(
            broadcasts["short"]["dependency_ready_consumer_count"], 0
        )
        self.assertEqual(
            broadcasts["long"]["dependency_ready_consumer_count"], 1
        )

    def test_completed_tag_history_wakes_a_late_consumer(self):
        program = artifact(
            (("source", "DMA", 2, ()), ("consumer", "MXU", 1, ("source",)))
        )
        submission = create_runtime_submission(
            program,
            (
                BufferBinding("source", 4096, 8, "SRAM", "SRAM"),
                BufferBinding("consumer", 8192, 8, "SRAM", "SRAM"),
            ),
            chunk_size=1,
            descriptor_available_cycles={"source": 0, "consumer": 10},
        )
        result = run(program, submission=submission)
        stages = result.metrics["instruction_pipeline"]
        self.assertLess(stages["source"]["completed"], stages["consumer"]["received"])
        self.assertEqual(stages["consumer"]["wakeup"], 12)
        self.assertGreaterEqual(result.metrics["late_ready_dependency_count"], 1)

    def test_semantic_conflict_describes_and_blocks_local_fu_overlap(self):
        program = artifact((("older", "DMA", 5, ()), ("younger", "DMA", 1, ())))

        def shared(access):
            return (
                TISAOperand(
                    "shared",
                    (4,),
                    TileMem("shared", "SRAM", offset_bytes=0, size_bytes=8),
                    access,
                ),
            )

        program = replace(
            program,
            program=replace(
                program.program,
                instructions=(
                    replace(program.program.instructions[0], operands=shared("write")),
                    replace(program.program.instructions[1], operands=shared("read")),
                ),
            ),
        )
        machine = minimal_machine_config()
        machine = replace(
            machine,
            execution_units=tuple(
                replace(unit, count=2, issue_width=2)
                if unit.name == "DMA"
                else unit
                for unit in machine.execution_units
            ),
        )
        loaded = load_implicit_device_program(program)
        older = loaded.descriptor_for_tisa("older")
        younger = loaded.descriptor_for_tisa("younger")
        observation = semantic_conflict(older, younger)
        self.assertIsNotNone(observation)
        self.assertEqual(observation.kind, "RAW")
        self.assertEqual(observation.memory, "SRAM")
        self.assertEqual(observation.size_bytes, 8)

        # Remove the loader's alias edge only to isolate the local Fu checker.
        loaded = replace(
            loaded,
            descriptors=tuple(
                replace(item, dependencies=())
                if item.instruction.tisa_id == "younger"
                else item
                for item in loaded.descriptors
            ),
            static_schedule=None,
        )
        result = schedule_loaded_cycle_program(
            loaded,
            machine,
            "dynamic_ready_queue",
            AnalyticalExecutionBackend(program, machine),
            config=SimulatorConfig(
                dynamic_priority="oldest_first",
                pipeline=SchedulerPipelineConfig(
                    receive_width=2,
                    dispatch_width=2,
                    select_width=2,
                    issue_width=2,
                ),
            ),
            audit={
                "timing_provider_name": "analytical",
                "timing_calibration_status": "analytical",
                "compile_package_sha256": "test",
                "runtime_submission_present": False,
            },
        )
        self.assertGreater(result.metrics["stall_cycles"]["semantic_conflict"], 0)
        self.assertEqual(result.metrics["semantic_conflict_count"], 1)
        self.assertGreaterEqual(
            result.instruction_timing("younger").issue,
            result.instruction_timing("older").finish,
        )

    def test_semantic_conflict_classifies_access_and_scope_rules(self):
        def observation(
            older_access,
            younger_access,
            *,
            younger_scope="SRAM",
            younger_offset=0,
        ):
            program = artifact(
                (("older", "DMA", 2, ()), ("younger", "DMA", 1, ()))
            )

            def operand(access, scope, offset):
                return (
                    TISAOperand(
                        "shared",
                        (4,),
                        TileMem(
                            "shared",
                            scope,
                            offset_bytes=offset,
                            size_bytes=8,
                        ),
                        access,
                    ),
                )

            program = replace(
                program,
                program=replace(
                    program.program,
                    instructions=(
                        replace(
                            program.program.instructions[0],
                            operands=operand(older_access, "SRAM", 0),
                        ),
                        replace(
                            program.program.instructions[1],
                            operands=operand(
                                younger_access,
                                younger_scope,
                                younger_offset,
                            ),
                        ),
                    ),
                ),
            )
            loaded = load_implicit_device_program(program)
            return semantic_conflict(
                loaded.descriptor_for_tisa("older"),
                loaded.descriptor_for_tisa("younger"),
            )

        for older_access, younger_access, kind in (
            ("write", "read", "RAW"),
            ("read", "write", "WAR"),
            ("write", "write", "WAW"),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(
                    observation(older_access, younger_access).kind,
                    kind,
                )
        self.assertIsNone(observation("read", "read"))
        self.assertIsNone(
            observation("write", "read", younger_scope="OTHER_MEMORY")
        )
        self.assertIsNone(
            observation("write", "read", younger_offset=16)
        )

    def test_explicit_control_latencies_and_physical_rounding(self):
        program = artifact((("a", "DMA", 1.5, ()),))
        result = run(
            program,
            dispatch_latency=3,
            select_latency=2,
            completion_latency=2,
            retire_latency=3,
        )
        self.assertEqual(
            result.metrics["instruction_pipeline"]["a"],
            {
                "received": 0,
                "dispatched": 1,
                "wakeup": 4,
                "selected": 4,
                "issued": 6,
                "done": 8,
                "completed": 10,
                "retired": 13,
            },
        )
        self.assertEqual(result.timing("a.task").finish, 7.5)
        self.assertEqual(result.total_cycles, 13)

    def test_different_units_overlap_and_completion_bus_serializes_feedback(self):
        program = artifact((("a", "DMA", 1, ()), ("b", "MXU", 1, ())))
        wide = dict(receive_width=2, dispatch_width=2, select_width=2, issue_width=2)
        result = run(program, **wide, completion_width=1)
        self.assertEqual([t.issue for t in result.instruction_timings], [3, 3])
        self.assertEqual([t.finish for t in result.instruction_timings], [4, 5])
        self.assertEqual(result.metrics["stall_cycles"]["completion_bandwidth"], 1)
        self.assertEqual(
            run(program, **wide, completion_width=2, retire_width=2).total_cycles, 5
        )

    def test_retirement_waits_for_older_instruction(self):
        program = artifact((("old", "DMA", 5, ()), ("young", "MXU", 1, ())))
        result = run(
            program, receive_width=2, dispatch_width=2, select_width=2, issue_width=2
        )
        stages = result.metrics["instruction_pipeline"]
        self.assertEqual(stages["young"]["completed"], 4)
        self.assertEqual(stages["old"]["retired"], 9)
        self.assertEqual(stages["young"]["retired"], 10)
        self.assertGreater(result.metrics["stall_cycles"]["retire_backpressure"], 0)
        constrained = run(program, config=SimulatorConfig(rob_entries=1))
        self.assertGreater(constrained.metrics["stall_cycles"]["rob_full"], 0)
        self.assertLessEqual(constrained.metrics["rob_peak"], 1)

    def test_dynamic_bypasses_blocked_same_unit_candidate_with_identical_artifact(self):
        program = artifact(
            (
                ("source", "MXU", 10, ()),
                ("blocked", "DMA", 2, ("source",)),
                ("ready", "DMA", 2, ()),
            )
        )
        before = json.dumps(program.to_dict(), sort_keys=True)
        static = run(program, policy="static_pipeline")
        dynamic = run(program)
        self.assertLess(
            dynamic.instruction_timing("ready").issue,
            dynamic.instruction_timing("blocked").issue,
        )
        self.assertLess(dynamic.total_cycles, static.total_cycles)
        self.assertEqual(
            static.metrics["compile_package_sha256"],
            dynamic.metrics["compile_package_sha256"],
        )
        self.assertEqual(before, json.dumps(program.to_dict(), sort_keys=True))
        narrow = run(program, config=SimulatorConfig(dependency_window=1))
        self.assertGreaterEqual(
            narrow.instruction_timing("ready").issue,
            narrow.instruction_timing("blocked").issue,
        )

    def test_wq_iq_and_reception_backpressure(self):
        machine = minimal_machine_config()
        machine = replace(
            machine,
            execution_units=tuple(
                replace(unit, queue_depth=1) for unit in machine.execution_units
            ),
        )
        program = artifact(tuple((f"a{i}", "DMA", 8, ()) for i in range(6)))
        cfg = SimulatorConfig(
            instruction_queue_depth=1,
            ready_queue_depth=1,
            dynamic_priority="oldest_first",
        )
        result = run(program, machine=machine, config=cfg)
        for reason in ("reception_full", "wq_full", "iq_full", "fu_busy"):
            self.assertGreater(result.metrics["stall_cycles"][reason], 0, reason)
        for row in result.metrics["queue_occupancy_timeline"]:
            self.assertLessEqual(row["reception_queue"], 1)
            self.assertLessEqual(row["wq"]["DMA"], 1)
            self.assertLessEqual(row["iq"]["DMA"], 1)

    def test_fu_tracks_operands_until_feedback_while_execution_unit_can_finish(self):
        program = artifact((("a", "DMA", 1, ()), ("b", "DMA", 1, ())))
        result = run(program, completion_latency=4, inflight_entries=1)
        self.assertGreater(result.metrics["stall_cycles"]["fu_table_full"], 0)
        self.assertEqual(result.metrics["fu_peak"]["DMA"], 1)
        self.assertGreaterEqual(
            result.instruction_timing("b").issue, result.instruction_timing("a").finish
        )

    def test_unit_issue_width_and_initiation_interval(self):
        machine = minimal_machine_config()
        machine = replace(
            machine,
            execution_units=tuple(
                replace(unit, count=2, issue_width=1) if unit.name == "DMA" else unit
                for unit in machine.execution_units
            ),
        )
        program = artifact((("a", "DMA", 5, ()), ("b", "DMA", 5, ())))
        result = run(
            program,
            machine=machine,
            receive_width=2,
            dispatch_width=2,
            select_width=2,
            issue_width=2,
        )
        self.assertEqual([t.issue for t in result.instruction_timings], [3, 4])
        self.assertEqual(result.metrics["stall_cycles"]["issue_bandwidth"], 1)
        slow = replace(
            program,
            execution_graph=replace(
                program.execution_graph,
                tasks=tuple(
                    replace(task, duration_cycles=1, initiation_interval_cycles=8)
                    for task in program.execution_graph.tasks
                ),
            ),
        )
        limited = run(slow)
        self.assertGreaterEqual(
            limited.instruction_timing("b").issue
            - limited.instruction_timing("a").issue,
            8,
        )

    def test_runtime_alias_tokens_protect_older_unissued_cross_unit_hazards(self):
        for first_access, second_access in (
            ("write", "read"),
            ("read", "write"),
            ("write", "write"),
        ):
            with self.subTest(first=first_access, second=second_access):
                program = artifact(
                    (
                        ("gate", "MXU", 8, ()),
                        ("old", "DMA", 2, ("gate",)),
                        ("young", "ARU", 1, ()),
                    )
                )
                operands = lambda access: (
                    TISAOperand(
                        "shared",
                        (4,),
                        TileMem("shared", "SRAM", offset_bytes=0, size_bytes=8),
                        access,
                    ),
                )
                instructions = (
                    program.program.instructions[0],
                    replace(
                        program.program.instructions[1], operands=operands(first_access)
                    ),
                    replace(
                        program.program.instructions[2],
                        operands=operands(second_access),
                    ),
                )
                program = replace(
                    program, program=replace(program.program, instructions=instructions)
                )
                result = run(
                    program,
                    config=SimulatorConfig(
                        address_scoreboard=True, dynamic_priority="oldest_first"
                    ),
                )
                self.assertGreaterEqual(
                    result.instruction_timing("young").issue,
                    result.instruction_timing("old").finish,
                )
                self.assertGreater(result.metrics["stall_cycles"]["dependency_wait"], 0)
                self.assertGreaterEqual(
                    result.metrics["runtime_alias_dependency_count"], 1
                )

    def test_memory_ports_and_tile_window_limit_overlap(self):
        program = artifact((("a", "DMA", 4, ()), ("b", "MXU", 4, ())))
        wide = dict(receive_width=2, dispatch_width=2, select_width=2, issue_width=2)
        for config, reason in (
            (SimulatorConfig(memory_bank_scoreboard=True), "memory_bank_port_conflict"),
            (SimulatorConfig(max_inflight_tiles=1), "tile_window_full"),
        ):
            with self.subTest(reason=reason):
                result = run(program, config=config, **wide)
                self.assertGreater(result.metrics["stall_cycles"][reason], 0)
                self.assertGreaterEqual(
                    result.instruction_timing("b").issue, result.timing("a.task").finish
                )

    def test_partial_ready_applies_wakeup_delay(self):
        program = artifact(
            (("source", "DMA", 2, ()), ("consumer", "MXU", 1, ("source",)))
        )
        tail = replace(
            program.execution_graph.tasks[0],
            task_id="source.tail",
            duration_cycles=8,
            program_order=2,
        )
        instructions = (
            program.program.instructions[0],
            replace(
                program.program.instructions[1],
                dependencies=(
                    TISADependency("source", condition="payload_ready:source.task"),
                ),
            ),
        )
        program = replace(
            program,
            program=replace(program.program, instructions=instructions),
            execution_graph=replace(
                program.execution_graph, tasks=(*program.execution_graph.tasks, tail)
            ),
            payloads={
                "source": ("source.task", "source.tail"),
                "consumer": ("consumer.task",),
            },
        )
        result = run(program, wakeup_latency=2)
        self.assertEqual(result.instruction_timing("consumer").issue, 8)
        self.assertEqual(result.metrics["partial_ready_event_count"], 1)
        self.assertLess(
            result.instruction_timing("consumer").finish,
            result.instruction_timing("source").finish,
        )

    def test_runtime_arrival_and_synchronization(self):
        program = artifact((("a", "DMA", 2, ()),))
        submission = create_runtime_submission(
            program,
            (BufferBinding("a", 4096, 8, "SRAM", "SRAM"),),
            descriptor_available_cycles={"a": 2.5},
            launch_latency_cycles=2,
            synchronization_cycles=3,
        )
        result = run(program, submission=submission)
        self.assertEqual(result.metrics["instruction_pipeline"]["a"]["received"], 5)
        self.assertEqual(result.total_cycles, 14)
        self.assertEqual(result.metrics["runtime_request_wait_cycles"], 2.5)

    def test_trace_contains_stages_and_csv_lifecycle(self):
        result = run(artifact((("a", "DMA", 2, ()),)))
        expected = {
            "TISA_RECEIVE",
            "TISA_DISPATCH",
            "TISA_WAKE_UP",
            "TISA_SELECT",
            "TISA_ISSUE",
            "TISA_EXECUTION_DONE",
            "TISA_COMPLETE",
            "TISA_RETIRE",
        }
        self.assertTrue(expected.issubset({event.event for event in result.events}))
        trace = result.perfetto_trace()["traceEvents"]
        self.assertTrue(any(event["ph"] == "i" and event["ts"] == 6 for event in trace))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "06_simulation" / "tisa_instructions.csv"
            write_instruction_csv(result, path)
            record = next(csv.DictReader(io.StringIO(path.read_text())))
        self.assertEqual(
            (record["receive"], record["dispatch"], record["select"], record["retire"]),
            ("0", "1", "2", "6"),
        )

    def test_invalid_capacity_condition_and_cycle_limit_fail_explicitly(self):
        for payload in (
            {"issue_width": 0},
            {"wakeup_latency": -1},
            {"receive_width": True},
            {"typo": 3},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                SchedulerPipelineConfig.from_dict(payload)
        with self.assertRaisesRegex(RuntimeError, "max_cycles"):
            run(artifact((("a", "DMA", 8, ()),)), max_cycles=2)
        program = artifact((("a", "DMA", 1, ()), ("b", "MXU", 1, ("a",))))
        bad = replace(
            program.program.instructions[1],
            dependencies=(TISADependency("a", condition="payload_ready:bad"),),
        )
        with self.assertRaisesRegex(ValueError, "outside the source payload"):
            run(
                replace(
                    program,
                    program=replace(
                        program.program,
                        instructions=(program.program.instructions[0], bad),
                    ),
                )
            )

    def test_machine_roundtrip_and_cli_configuration(self):
        machine = minimal_machine_config()
        machine = replace(
            machine,
            scheduler=replace(
                machine.scheduler, pipeline=SchedulerPipelineConfig(dispatch_latency=4)
            ),
        )
        self.assertEqual(MachineConfig.from_dict(machine.to_dict()), machine)
        result = run(artifact((("a", "DMA", 1, ()),)), machine=machine)
        self.assertEqual(result.instruction_timing("a").issue, 6)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "scheduler.json"
            config_path.write_text(json.dumps({"wakeup_latency": 3}))
            args = build_parser().parse_args(
                [
                    "simulate",
                    "--compile-dir",
                    directory,
                    "--event-backend",
                    "cycle_event",
                    "--scheduler-config",
                    str(config_path),
                ]
            )
            self.assertEqual(_simulation_config(args).pipeline.wakeup_latency, 3)
            args.event_backend = "analytical_event"
            with self.assertRaisesRegex(ValueError, "requires.*cycle_event"):
                _simulation_config(args)

    def test_standalone_simulate_writes_cycle_results_without_frontend(self):
        program = artifact((("a", "DMA", 2, ()),))
        inst = program.program.instructions[0]
        operand = replace(
            inst.operands[0],
            tile_mem=replace(inst.operands[0].tile_mem, scope="logical"),
        )
        program = replace(
            program,
            program=replace(
                program.program,
                instructions=(
                    replace(
                        inst,
                        operands=(
                            replace(
                                operand,
                                tile_mem=replace(
                                    operand.tile_mem,
                                    scope="SRAM",
                                    buffer_id="a@SRAM",
                                    allocation_id="SRAM.alloc0000",
                                    valid_bytes=8,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            memory_plan=MemoryPlan(
                "micro.memory",
                "minimal",
                minimal_machine_config().topology_hash(),
                (
                    MemoryBuffer(
                        "a@SRAM",
                        "SRAM.alloc0000",
                        "a",
                        "SRAM",
                        0,
                        8,
                        8,
                        1,
                        "dense",
                    ),
                ),
            ),
        )
        target_plan = TargetPlan(
            "micro.target-plan",
            "micro.abstract",
            "minimal",
            minimal_machine_config().topology_hash(),
            (
                TargetInstructionPlan(
                    "micro.abstract.a",
                    program.program.instructions[0],
                    "direct",
                ),
            ),
            {"a": ("a@SRAM",)},
            attributes={"target_program_id": program.program.program_id},
        )
        program = replace(
            program,
            program=target_plan.program,
            target_plan=target_plan,
        )
        graph = OperatorGraph("micro", (TensorSpec("a", (4,)),), ())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("01_gc", "04_backend"):
                (root / folder).mkdir()
            files = {
                "01_gc/canonical_graph.json": graph.to_dict(),
                "04_backend/backend_artifact.json": program.to_dict(),
                "04_backend/machine.json": minimal_machine_config().to_dict(),
            }
            for name, value in files.items():
                (root / name).write_text(json.dumps(value))
            before = {name: (root / name).read_bytes() for name in files}
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "simulate",
                        "--compile-dir",
                        str(root),
                        "--event-backend",
                        "cycle_event",
                        "--output-dir",
                        str(root / "run"),
                    ]
                )
            self.assertEqual(status, 0)
            result = json.loads((root / "run/06_simulation/summary.json").read_text())
            self.assertEqual(result["metrics"]["retired_instruction_count"], 1)
            self.assertEqual(result["total_cycles"], 6)
            self.assertEqual(
                before, {name: (root / name).read_bytes() for name in files}
            )
            self.assertTrue((root / "run/07_trace/perfetto.json").is_file())

    def test_reordered_descriptor_aliases_fail_with_deadlock_diagnostic(self):
        program = artifact((("a", "DMA", 2, ()), ("b", "MXU", 1, ())))
        shared = TISAOperand(
            "shared",
            (4,),
            TileMem("shared", "SRAM", tensor="shared", offset_bytes=0, size_bytes=8),
            "write",
        )
        program = replace(
            program,
            program=replace(
                program.program,
                instructions=tuple(
                    replace(inst, operands=(shared,))
                    for inst in program.program.instructions
                ),
            ),
        )
        submission = create_runtime_submission(
            program,
            (BufferBinding("shared", 4096, 8, "SRAM", "SRAM"),),
            policy="dynamic_ready_queue",
            chunk_size=1,
            descriptor_available_cycles={"a": 2, "b": 0},
        )
        with self.assertRaisesRegex(
            ValueError, "descriptor order.*physical alias dependencies"
        ):
            run(
                program,
                submission=submission,
                config=SimulatorConfig(rob_entries=1, address_scoreboard=True),
            )

    def test_sequence_keeps_each_invocation_pipeline_cycles_and_counts(self):
        program = artifact((("a", "DMA", 2, ()),))
        buffers = (BufferBinding("a", 4096, 8, "SRAM", "SRAM"),)
        sequence = create_runtime_sequence(
            program,
            create_runtime_state_registry(program, buffers),
            invocation_count=2,
            inter_invocation_gap_cycles=3,
        )
        result = schedule_tisa_sequence(
            program,
            sequence,
            minimal_machine_config(),
            event_backend=CycleEventBackend(),
        )
        self.assertEqual(result.total_cycles, 15)
        self.assertEqual(result.metrics["retired_instruction_count"], 2)
        self.assertEqual(result.metrics["completed_task_count"], 2)
        self.assertEqual(result.metrics["completion_finish_cycle"], 14)
        self.assertEqual(result.metrics["retirement_drain_cycles"], 1)
        self.assertEqual(
            result.metrics["instruction_pipeline"][
                sequence.invocations[1].submission_id + "/a"
            ]["retired"],
            15,
        )
        self.assertEqual(
            result.metrics["queue_occupancy_timeline"][-1]["timestamp"], 15
        )
        self.assertEqual(result.metrics["stall_cycles"]["dependency_wait"], 0)
        summary = result.summary_dict()
        self.assertNotIn("events", summary)
        self.assertNotIn("queue_occupancy_timeline", summary["metrics"])
        self.assertEqual(len(summary["invocations"]), 2)
        self.assertTrue(all("events" not in item for item in summary["invocations"]))


if __name__ == "__main__":
    unittest.main()
