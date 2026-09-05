import importlib.util
import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available
from npu_ooo.ir import (
    BufferBinding,
    RuntimeSubmission,
    allocate_buffer_bindings,
    create_runtime_submission,
    derive_tensor_lifetimes,
    derive_tensor_reuse_pairs,
)


FRONTEND_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and torch_xla_available()
    and official_stablehlo_available()
)


@unittest.skipUnless(FRONTEND_AVAILABLE, "requires the production PyTorch frontend")
class RuntimeSubmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from examples.torch_models import TwoMatmul

        cls.compiled = compile_torch_module(
            TwoMatmul(),
            tuple(torch.randn(8, 8) for _ in range(3)),
            minimal_machine_config(),
            model_id="runtime-two-matmul",
            tile_size=4,
        )

    def test_linear_allocator_is_aligned_and_non_overlapping(self) -> None:
        bindings = allocate_buffer_bindings(
            self.compiled.graph.tensors,
            base_address=0x2000,
            alignment_bytes=256,
        )
        self.assertEqual(len(bindings), len(self.compiled.graph.tensors))
        self.assertTrue(all(item.base_address % 256 == 0 for item in bindings))
        for left, right in zip(bindings, bindings[1:]):
            self.assertLessEqual(left.end_address, right.base_address)

    def test_submission_binds_operands_and_chunks_without_changing_program(self) -> None:
        buffers = allocate_buffer_bindings(self.compiled.graph.tensors, base_address=0x100000)
        submission = create_runtime_submission(
            self.compiled.backend_artifact,
            buffers,
            policy="dynamic_ready_queue",
            chunk_size=3,
        )
        self.assertIsInstance(submission, RuntimeSubmission)
        self.assertEqual(submission.validate(self.compiled.tisa_program), ())
        submitted_ids = [item for chunk in submission.commands for item in chunk.tisa_ids]
        program_ids = [item.tisa_id for item in self.compiled.tisa_program.instructions]
        self.assertCountEqual(submitted_ids, program_ids)
        submitted_index = {tisa_id: index for index, tisa_id in enumerate(submitted_ids)}
        for instruction in self.compiled.tisa_program.instructions:
            for dependency in instruction.dependencies:
                self.assertLess(submitted_index[dependency.source], submitted_index[instruction.tisa_id])

    def test_submission_rejects_out_of_range_explicit_operand_offset(self) -> None:
        buffers = allocate_buffer_bindings(self.compiled.graph.tensors, base_address=0x100000)
        instruction = self.compiled.tisa_program.instructions[0]
        operand = instruction.operands[0]
        with self.assertRaisesRegex(ValueError, "offset exceeds buffer"):
            create_runtime_submission(
                self.compiled.backend_artifact,
                buffers,
                operand_offsets={(instruction.tisa_id, operand.name): 10**9},
            )

    def test_lifetime_allocator_does_not_infer_whole_buffer_order_from_tile_order(self) -> None:
        lifetimes = derive_tensor_lifetimes(self.compiled.tisa_program)
        reuse_pairs = derive_tensor_reuse_pairs(self.compiled.tisa_program)
        bindings = allocate_buffer_bindings(
            self.compiled.graph.tensors,
            lifetimes=lifetimes,
            reuse_buffers=True,
            reuse_pairs=reuse_pairs,
        )
        reused = [binding for binding in bindings if binding.attributes.get("reused_from")]
        self.assertEqual(reuse_pairs, frozenset())
        self.assertFalse(reused)
        submission = create_runtime_submission(self.compiled.backend_artifact, bindings)
        self.assertEqual(submission.validate(self.compiled.tisa_program), ())

    def test_lifetime_allocator_requires_dependency_proof_for_reuse(self) -> None:
        bindings = allocate_buffer_bindings(
            self.compiled.graph.tensors,
            lifetimes=derive_tensor_lifetimes(self.compiled.tisa_program),
            reuse_buffers=True,
            reuse_pairs=frozenset(),
        )
        self.assertFalse(any(binding.attributes.get("reused_from") for binding in bindings))

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
