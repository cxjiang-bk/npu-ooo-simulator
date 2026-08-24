import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_two_matmul_case, build_two_matmul_model
from npu_ooo.compiler import compile_model_instance
from npu_ooo.ir import (
    BufferBinding,
    RuntimeSubmission,
    allocate_buffer_bindings,
    create_runtime_submission,
)


class RuntimeSubmissionTest(unittest.TestCase):
    def test_linear_allocator_is_aligned_and_non_overlapping(self) -> None:
        model = build_two_matmul_model()
        instance = model.instantiate(build_two_matmul_case())
        bindings = allocate_buffer_bindings(
            instance.graph.tensors,
            base_address=0x2000,
            alignment_bytes=256,
        )
        self.assertEqual(len(bindings), len(instance.graph.tensors))
        self.assertTrue(all(item.base_address % 256 == 0 for item in bindings))
        for left, right in zip(bindings, bindings[1:]):
            self.assertLessEqual(left.end_address, right.base_address)

    def test_submission_binds_operands_and_chunks_without_changing_program(self) -> None:
        model = build_two_matmul_model()
        instance = model.instantiate(build_two_matmul_case())
        compiled = compile_model_instance(instance, minimal_machine_config(), tile_size=32)
        buffers = allocate_buffer_bindings(compiled.graph.tensors, base_address=0x100000)
        submission = create_runtime_submission(
            compiled.backend_artifact,
            buffers,
            policy="dynamic_ready_queue",
            chunk_size=3,
        )
        self.assertIsInstance(submission, RuntimeSubmission)
        self.assertEqual(submission.validate(compiled.tisa_program), ())
        submitted_ids = [item for chunk in submission.commands for item in chunk.tisa_ids]
        program_ids = [item.tisa_id for item in compiled.tisa_program.instructions]
        self.assertCountEqual(submitted_ids, program_ids)
        submitted_index = {tisa_id: index for index, tisa_id in enumerate(submitted_ids)}
        for instruction in compiled.tisa_program.instructions:
            for dependency in instruction.dependencies:
                self.assertLess(submitted_index[dependency.source], submitted_index[instruction.tisa_id])
        self.assertEqual(len(submission.operands), sum(len(item.operands) for item in compiled.tisa_program.instructions))
        self.assertEqual(submission.attributes["device_issue_order"], "independent")
        self.assertTrue(all(item.attributes["address_source"] == "tile_mem" for item in submission.operands))
        self.assertTrue(all(item.attributes["size_source"] == "tile_mem" for item in submission.operands))

    def test_submission_rejects_out_of_range_explicit_operand_offset(self) -> None:
        model = build_two_matmul_model()
        instance = model.instantiate(build_two_matmul_case())
        compiled = compile_model_instance(instance, minimal_machine_config(), tile_size=32)
        buffers = allocate_buffer_bindings(compiled.graph.tensors, base_address=0x100000)
        instruction = compiled.tisa_program.instructions[0]
        operand = instruction.operands[0]
        with self.assertRaisesRegex(ValueError, "offset exceeds buffer"):
            create_runtime_submission(
                compiled.backend_artifact,
                buffers,
                operand_offsets={(instruction.tisa_id, operand.name): 10**9},
            )

    def test_buffer_binding_validation_rejects_unaligned_address(self) -> None:
        binding = BufferBinding(
            tensor="x",
            base_address=3,
            size_bytes=16,
            alignment_bytes=4,
        )
        self.assertIn("not aligned", " ".join(binding.validate()))


if __name__ == "__main__":
    unittest.main()
