import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_operator_graph
from npu_ooo.frontend.stablehlo import StableHLOAdapter
from npu_ooo.frontend.stablehlo_official import OfficialStableHLOModule
from npu_ooo.ir import (
    DynamicIndexBinding,
    allocate_buffer_bindings,
    create_runtime_sequence,
    create_runtime_state_registry,
    create_runtime_submission,
)
from npu_ooo.scheduler import SchedulerPolicy, SimulatorConfig, schedule_tisa_program


def _compile(text: str, model_id: str, tile_size: int):
    frontend = StableHLOAdapter.from_text(text, model_id=model_id)
    stablehlo = OfficialStableHLOModule(
        text=text,
        canonical_text=text,
        model_id=model_id,
    )
    return compile_operator_graph(
        frontend.graph,
        minimal_machine_config(),
        frontend=frontend,
        source_frontend=frontend,
        stablehlo=stablehlo,
        tile_size=tile_size,
    )


class DynamicRuntimeContractTest(unittest.TestCase):
    def test_dynamic_slice_resolves_clamped_strided_window(self) -> None:
        text = """
        module {
          func.func @main(%source: tensor<8x4xf32>, %row: tensor<i32>, %column: tensor<i32>) -> tensor<2x3xf32> {
            %0 = stablehlo.dynamic_slice %source, %row, %column, sizes = [2, 3] : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """
        compiled = _compile(text, "dynamic-slice-runtime", tile_size=2)
        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        submission = create_runtime_submission(
            compiled.backend_artifact,
            bindings,
            policy="dynamic_ready_queue",
            dynamic_index_bindings=(DynamicIndexBinding("0.index", (99, 99)),),
        )
        static_submission = create_runtime_submission(
            compiled.backend_artifact,
            bindings,
            policy="static",
            dynamic_index_bindings=(DynamicIndexBinding("0.index", (99, 99)),),
        )
        self.assertEqual(static_submission.program_id, submission.program_id)
        self.assertEqual(
            compiled.backend_artifact.program.to_dict(),
            compiled.tisa_program.to_dict(),
        )
        dynamic_operands = [
            operand
            for operand in submission.operands
            if operand.attributes.get("dynamic_region") is not None
        ]
        self.assertEqual(len(dynamic_operands), 2)
        self.assertTrue(
            all(operand.attributes["resolved_index_values"] == [6, 1] for operand in dynamic_operands)
        )
        self.assertTrue(
            all(operand.attributes["resolved_offset_bytes"] == 100 for operand in dynamic_operands)
        )
        self.assertTrue(all(operand.size_bytes == 28 for operand in dynamic_operands))

        result = schedule_tisa_program(
            compiled.backend_artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=SimulatorConfig(address_scoreboard=True),
            runtime_submission=submission,
        )
        self.assertEqual(result.metrics["dynamic_index_bindings"][0]["values"], [99, 99])
        self.assertEqual(result.metrics["address_scoreboard_scope"], "runtime_physical")
        issue = next(
            event
            for event in result.events
            if event.event == "TISA_ISSUE"
        )
        self.assertTrue(
            any(
                operand.get("attributes", {}).get("dynamic_region")
                for operand in issue.details["runtime_operands"]
            )
        )

    def test_dynamic_index_accepts_signed_values_before_clamp(self) -> None:
        text = """
        module {
          func.func @main(%source: tensor<8x4xf32>, %row: tensor<i32>, %column: tensor<i32>) -> tensor<2x3xf32> {
            %0 = stablehlo.dynamic_slice %source, %row, %column, sizes = [2, 3] : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """
        compiled = _compile(text, "dynamic-slice-signed", tile_size=2)
        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        submission = create_runtime_submission(
            compiled.backend_artifact,
            bindings,
            dynamic_index_bindings=(DynamicIndexBinding("0.index", (-2, 99)),),
        )
        region = next(
            operand.attributes["dynamic_region"]
            for operand in submission.operands
            if operand.attributes.get("dynamic_region") is not None
        )
        self.assertEqual(region["resolved_index_values"], [0, 1])

    def test_dynamic_update_slice_binds_state_window_and_sequence_values(self) -> None:
        text = """
        module {
          func.func @main(%cache: tensor<4x4xf32>, %update: tensor<1x2xf32>, %row: tensor<i32>, %column: tensor<i32>) -> tensor<4x4xf32> {
            %0 = stablehlo.dynamic_update_slice %cache, %update, %row, %column : (tensor<4x4xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
            return %0 : tensor<4x4xf32>
          }
        }
        """
        compiled = _compile(text, "dynamic-update-runtime", tile_size=4)
        self.assertEqual(
            compiled.tisa_program.instructions[1].attributes["state_region"]["window_shape"],
            [1, 2],
        )
        bindings = allocate_buffer_bindings(compiled.graph.tensors)
        registry = create_runtime_state_registry(compiled.backend_artifact, bindings)
        sequence = create_runtime_sequence(
            compiled.backend_artifact,
            registry,
            invocation_count=2,
            policy="dynamic_ready_queue",
            invocation_dynamic_indices=(
                (DynamicIndexBinding("0.index", (0, 0)),),
                (DynamicIndexBinding("0.index", (3, 2)),),
            ),
        )
        self.assertEqual(sequence.validate(compiled.tisa_program), ())
        first = sequence.invocations[0]
        second = sequence.invocations[1]
        first_regions = [
            operand.attributes["dynamic_region"]
            for operand in first.operands
            if operand.attributes.get("dynamic_region") is not None
        ]
        second_regions = [
            operand.attributes["dynamic_region"]
            for operand in second.operands
            if operand.attributes.get("dynamic_region") is not None
        ]
        self.assertEqual({tuple(item["starts"]) for item in first_regions}, {(0, 0)})
        self.assertEqual({tuple(item["starts"]) for item in second_regions}, {(3, 2)})
        self.assertEqual(first.dynamic_indices[0].values, (0, 0))
        self.assertEqual(second.dynamic_indices[0].values, (3, 2))


if __name__ == "__main__":
    unittest.main()
