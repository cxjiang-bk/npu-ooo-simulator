from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from npu_ooo.analysis import analyze_run, build_buffer_lifecycle, compare_runs, write_analysis_report
from npu_ooo.arch import SchedulerPipelineConfig, minimal_machine_config
from npu_ooo.backend import CycleEventBackend
from npu_ooo.ir import allocate_memory_plan_bindings, create_runtime_submission
from npu_ooo.runtime import load_device_program, load_implicit_device_program
from npu_ooo.scheduler import SimulatorConfig, schedule_tisa_program
from npu_ooo.trace import (
    ensure_output_layout,
    program_hierarchy,
    write_artifact_json,
    write_json,
    write_tisa_graph_dot,
)

from test_static_scheduling import _artifact
from test_target_memory_mapping import _compile


class OfflineAnalysisTest(unittest.TestCase):
    def test_hierarchy_links_operator_tile_tisa_payload_and_control_views(self):
        compiled = _compile(minimal_machine_config())
        hierarchy = program_hierarchy(compiled.backend_artifact)
        instructions = [
            instruction
            for operator in hierarchy["operators"]
            for tile in operator["tiles"]
            for instruction in tile["instructions"]
        ]
        self.assertEqual(
            len(instructions), len(compiled.backend_artifact.program.instructions)
        )
        self.assertTrue(all(item["payload"] for item in instructions))
        self.assertIsNotNone(hierarchy["static_control"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "08_analysis"
            static = root / "static.dot"
            dynamic = root / "dynamic.dot"
            write_tisa_graph_dot(compiled.backend_artifact, static)
            write_tisa_graph_dot(
                compiled.backend_artifact,
                dynamic,
                include_static_control=False,
            )
            static_text = static.read_text(encoding="utf-8")
            dynamic_text = dynamic.read_text(encoding="utf-8")
        self.assertIn("static order", static_text)
        self.assertNotIn("static order", dynamic_text)

    def _write_run(self, root: Path, artifact, result, loaded) -> None:
        ensure_output_layout(root)
        write_json(result, root / "summary.json")
        write_artifact_json(result.perfetto_trace(), root / "perfetto.json")
        write_artifact_json(loaded, root / "bound_device_program.json")
        write_artifact_json(
            {
                "artifact_kind": "simulation_result",
                "compile_artifact_id": artifact.artifact_id,
                "policy": result.policy,
                "machine_hash": minimal_machine_config().stable_hash(),
                "timing_provider": "analytical",
                "event_backend": "cycle_event",
            },
            root / "manifest.json",
        )

    def test_saved_run_reports_physical_bubbles_waits_and_fair_comparison(self):
        artifact = _artifact()
        machine = minimal_machine_config()
        static = schedule_tisa_program(
            artifact, machine, "static_streams", event_backend=CycleEventBackend()
        )
        dynamic = schedule_tisa_program(
            artifact, machine, "dynamic_ready_queue", event_backend=CycleEventBackend()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "static", root / "dynamic"
            self._write_run(first, artifact, static, load_implicit_device_program(artifact))
            self._write_run(second, artifact, dynamic, load_implicit_device_program(artifact))
            static_analysis = analyze_run(first)
            dynamic_analysis = analyze_run(second)
            comparison = compare_runs(static_analysis, dynamic_analysis)
            output = first / "08_analysis"
            write_analysis_report(static_analysis, output, comparison=comparison)
            self.assertTrue(static_analysis.resources)
            self.assertTrue(static_analysis.bubbles)
            self.assertTrue(static_analysis.waits)
            self.assertTrue(static_analysis.critical_wait_chain)
            self.assertTrue(comparison["same_shared_workload"])
            self.assertTrue((output / "analysis.json").is_file())
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "report.html").is_file())
            report_html = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn('class="cycle-grid-major"', report_html)
            self.assertIn("WQ occupancy / IQ occupancy by EU", report_html)
            resource_names = {str(item["resource"]) for item in static_analysis.resources}
            for resource in resource_names:
                self.assertIn(f"WQ[{resource}]", report_html)
                self.assertIn(f"IQ[{resource}]", report_html)
            self.assertIn("ROB occupancy", report_html)
            self.assertIn("Completion pending", report_html)
            self.assertIn("Protected bytes", report_html)
            self.assertIn("Retained bytes", report_html)
            self.assertIn("independent y-scale per memory", report_html)

    def test_control_width_change_can_have_no_effect_on_single_instruction(self):
        artifact = _artifact()
        artifact = replace(
            artifact,
            program=replace(artifact.program, instructions=artifact.program.instructions[:1]),
            execution_graph=replace(
                artifact.execution_graph,
                tasks=artifact.execution_graph.tasks[:1],
            ),
            payloads={"source": ("source.task",)},
            static_control=None,
        )
        from npu_ooo.compiler import attach_static_control

        artifact = attach_static_control(artifact, minimal_machine_config())
        results = []
        for width in (1, 8):
            results.append(
                schedule_tisa_program(
                    artifact,
                    minimal_machine_config(),
                    "static_streams",
                    simulator_config=SimulatorConfig(
                        pipeline=SchedulerPipelineConfig(control_width=width)
                    ),
                    event_backend=CycleEventBackend(),
                )
            )
        self.assertEqual(results[0].total_cycles, results[1].total_cycles)


class BufferLifecycleAnalysisTest(unittest.TestCase):
    def test_allocation_occupancy_uses_actual_consumers_and_no_alias_double_count(self):
        machine = minimal_machine_config()
        compiled = _compile(machine, shape=(8, 8, 8), tile_size=4)
        buffers = allocate_memory_plan_bindings(compiled.backend_artifact.memory_plan, machine)
        submission = create_runtime_submission(compiled.backend_artifact, buffers)
        loaded = load_device_program(compiled.backend_artifact, submission)
        result = schedule_tisa_program(
            compiled.backend_artifact,
            machine,
            "dynamic_ready_queue",
            runtime_submission=submission,
            event_backend=CycleEventBackend(),
        )
        report = build_buffer_lifecycle(
            compiled.backend_artifact, loaded, result, machine
        )
        expected_allocated = {}
        seen = set()
        for buffer in compiled.backend_artifact.memory_plan.buffers:
            if buffer.allocation_id in seen:
                continue
            seen.add(buffer.allocation_id)
            expected_allocated[buffer.memory] = (
                expected_allocated.get(buffer.memory, 0) + buffer.allocation_bytes
            )
        self.assertEqual(
            {memory: item["allocated_bytes"] for memory, item in report.memories.items()},
            expected_allocated,
        )
        for memory, item in report.memories.items():
            self.assertLessEqual(item["protected_peak_bytes"], item["allocated_bytes"])
            self.assertLessEqual(item["retained_peak_bytes"], item["allocated_bytes"])
            self.assertLessEqual(
                item["retained_peak_bytes"], item["protected_peak_bytes"]
            )
            if item["capacity_bytes"] is not None:
                self.assertLessEqual(item["allocated_bytes"], item["capacity_bytes"])
        completion = {item.task_id: item.finish for item in result.instruction_timings}
        for version in report.versions:
            self.assertGreaterEqual(version["valid_cycle"], version["protect_start"])
            self.assertGreaterEqual(
                version["release_cycle"],
                max((completion[item] for item in version["consumers"]), default=version["valid_cycle"]),
            )

    def test_persistent_allocation_is_retained_to_invocation_end(self):
        machine = minimal_machine_config()
        compiled = _compile(machine)
        plan = compiled.backend_artifact.memory_plan
        persistent_buffers = tuple(
            replace(
                item,
                attributes={**dict(item.attributes), "persistent": True},
            )
            for item in plan.buffers
        )
        artifact = replace(
            compiled.backend_artifact,
            memory_plan=replace(plan, buffers=persistent_buffers),
            target_plan=replace(
                compiled.backend_artifact.target_plan,
                memory_plan=replace(plan, buffers=persistent_buffers),
            ),
        )
        buffers = allocate_memory_plan_bindings(artifact.memory_plan, machine)
        submission = create_runtime_submission(artifact, buffers)
        loaded = load_device_program(artifact, submission)
        result = schedule_tisa_program(
            artifact,
            machine,
            "dynamic_ready_queue",
            runtime_submission=submission,
            event_backend=CycleEventBackend(),
        )
        report = build_buffer_lifecycle(artifact, loaded, result, machine)
        self.assertTrue(report.versions)
        self.assertTrue(all(item["release_cycle"] == result.total_cycles for item in report.versions))


if __name__ == "__main__":
    unittest.main()
