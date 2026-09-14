import unittest
from dataclasses import replace

from npu_ooo.arch import lpu_like_machine_config, minimal_machine_config
from npu_ooo.backend.memory import _add_physical_hazards
from npu_ooo.compiler import compile_operator_graph
from npu_ooo.frontend import FrontendImport, OfficialStableHLOModule
from npu_ooo.ir import (
    BackendArtifact,
    MemoryPlan,
    OperatorGraph,
    OperatorSpec,
    TensorSpec,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TileMem,
    UnitMap,
    allocate_memory_plan_bindings,
    create_runtime_submission,
)


def _compile(machine, *, shape=(4, 4, 4), tile_size=4, lhs_attributes=None):
    m, n, k = shape
    graph = OperatorGraph(
        "target-memory-matmul",
        (
            TensorSpec("lhs", (m, k), "f32", attributes=lhs_attributes or {}),
            TensorSpec("rhs", (k, n), "f32"),
            TensorSpec("out", (m, n), "f32"),
        ),
        (
            OperatorSpec(
                "mm",
                "matmul",
                ("lhs", "rhs"),
                ("out",),
                (("M", m), ("N", n)),
                (("K", k),),
            ),
        ),
    )
    frontend = FrontendImport(
        graph=graph,
        model_id=graph.graph_id,
        variant="test",
        frontend="stablehlo",
    )
    stablehlo = OfficialStableHLOModule(
        text="module {}",
        canonical_text="module {}",
        model_id=graph.graph_id,
    )
    return compile_operator_graph(
        graph,
        machine,
        frontend=frontend,
        source_frontend=frontend,
        stablehlo=stablehlo,
        tile_size=tile_size,
    )


class TargetMemoryMappingTest(unittest.TestCase):
    @staticmethod
    def _hazard_instruction(
        tisa_id: str,
        access: str,
        *,
        offset: int = 0,
        size: int = 8,
    ) -> TISAInstruction:
        return TISAInstruction(
            tisa_id=tisa_id,
            tile_id=f"tile.{tisa_id}",
            operator_id="hazard",
            op_type="elementwise",
            operands=(
                TISAOperand(
                    "buffer",
                    (size,),
                    TileMem(
                        "shared",
                        "SRAM",
                        tensor="shared",
                        buffer_id="shared@SRAM",
                        offset_bytes=offset,
                        size_bytes=size,
                    ),
                    access,
                ),
            ),
            unit_map=UnitMap("DMA"),
            payload_ref=f"payload:{tisa_id}",
        )

    @staticmethod
    def _hazard_sources(instruction: TISAInstruction) -> set[tuple[str, str]]:
        return {
            (dependency.source, dependency.kind)
            for dependency in instruction.dependencies
            if dependency.provenance.get("source") == "target_memory_hazard"
        }

    def test_target_memory_hazards_use_writer_reader_frontier(self):
        program = TISAProgram(
            "frontier",
            (
                self._hazard_instruction("writer", "write"),
                self._hazard_instruction("reader0", "read"),
                self._hazard_instruction("reader1", "read"),
                self._hazard_instruction("next_writer", "write"),
                self._hazard_instruction("last_writer", "write"),
            ),
        )
        lowered = _add_physical_hazards(program)
        self.assertEqual(
            self._hazard_sources(lowered.instructions[1]),
            {("writer", "RAW")},
        )
        self.assertEqual(
            self._hazard_sources(lowered.instructions[2]),
            {("writer", "RAW")},
        )
        self.assertEqual(
            self._hazard_sources(lowered.instructions[3]),
            {("reader0", "WAR"), ("reader1", "WAR")},
        )
        self.assertEqual(
            self._hazard_sources(lowered.instructions[4]),
            {("next_writer", "WAW")},
        )

    def test_minimal_matmul_uses_one_shared_descriptor_payload_contract(self):
        machine = minimal_machine_config()
        compiled = _compile(machine)
        self.assertEqual(
            compiled.tisa_program.to_dict(), compiled.backend_artifact.program.to_dict()
        )
        self.assertEqual(
            [instruction.unit_map.unit for instruction in compiled.tisa_program.instructions],
            ["DMA", "MXU", "DMA"],
        )
        compute = compiled.tisa_program.instructions[1]
        self.assertEqual(
            {(operand.tile_mem.physical_space, operand.normalized_access) for operand in compute.operands},
            {("SRAM", "read"), ("SRAM", "write")},
        )
        bindings = allocate_memory_plan_bindings(
            compiled.backend_artifact.memory_plan, machine
        )
        submission = create_runtime_submission(compiled.backend_artifact, bindings)
        self.assertEqual(submission.validate(compiled.tisa_program), ())
        target_ids = {
            instruction.tisa_id for instruction in compiled.tisa_program.instructions
        }
        abstract_ids = {
            instruction.tisa_id
            for instruction in compiled.virtual_tisa_program.instructions
        }
        self.assertTrue(
            all(
                task.attributes["target_tisa_id"] in target_ids
                and task.attributes["abstract_tisa_id"] in abstract_ids
                for task in compiled.backend_artifact.execution_graph.tasks
            )
        )
        by_id = {binding.buffer_id: binding for binding in submission.buffers}
        for operand in submission.operands:
            self.assertEqual(operand.physical_scope, by_id[operand.buffer_id].memory)
            self.assertGreaterEqual(operand.address, by_id[operand.buffer_id].base_address)
            self.assertLessEqual(
                operand.address + operand.size_bytes,
                by_id[operand.buffer_id].end_address,
            )

    def test_lpu_matmul_expands_multihop_routes_and_places_mxu_operands(self):
        compiled = _compile(lpu_like_machine_config())
        self.assertEqual(
            [instruction.unit_map.unit for instruction in compiled.tisa_program.instructions],
            ["GDMA", "LDMA", "MXU", "ARU", "GDMA"],
        )
        compute = next(
            instruction
            for instruction in compiled.tisa_program.instructions
            if instruction.unit_map.unit == "MXU"
        )
        accesses = {
            (operand.tile_mem.physical_space, operand.normalized_access)
            for operand in compute.operands
        }
        self.assertEqual(
            accesses,
            {("LMB", "read"), ("RMB", "read"), ("PSB", "write")},
        )
        self.assertFalse(any(scope in {"GM", "UB"} for scope, _ in accesses))
        paths = [
            (task.reads[0].memory, task.writes[0].memory, task.resource)
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.primitive != "matmul"
        ]
        self.assertEqual(
            paths,
            [
                ("GM", "UB", "GDMA"),
                ("GM", "UB", "GDMA"),
                ("UB", "LMB", "LDMA"),
                ("UB", "RMB", "LDMA"),
                ("PSB", "UB", "ARU"),
                ("UB", "GM", "GDMA"),
            ],
        )

    def test_multik_boundary_and_strided_source_keep_distinct_byte_meanings(self):
        compiled = _compile(
            minimal_machine_config(),
            shape=(5, 5, 6),
            tile_size=4,
            lhs_attributes={"strides_bytes": [32, 4]},
        )
        source = next(
            region
            for task in compiled.backend_artifact.execution_graph.tasks
            for region in task.reads
            if region.tensor == "lhs"
            and region.memory == "DRAM"
            and region.shape == (4, 4)
        )
        local = next(
            region
            for task in compiled.backend_artifact.execution_graph.tasks
            for region in task.writes
            if region.tensor == "lhs"
            and region.memory == "SRAM"
            and region.shape == (4, 4)
        )
        self.assertEqual((source.valid_bytes, source.size_bytes), (64, 112))
        self.assertEqual((local.valid_bytes, local.size_bytes), (64, 64))
        self.assertEqual(local.layout, "packed")

        computes = [
            instruction
            for instruction in compiled.tisa_program.instructions
            if instruction.unit_map.unit == "MXU"
        ]
        self.assertTrue(
            any(
                operand.normalized_access == "read_write"
                and operand.tile_mem.tensor == "out"
                for instruction in computes
                for operand in instruction.operands
            )
        )
        self.assertTrue(
            any(
                dependency.provenance.get("source") == "target_memory_hazard"
                and dependency.kind in {"WAR", "WAW"}
                for instruction in compiled.tisa_program.instructions
                for dependency in instruction.dependencies
            )
        )
        self.assertTrue(
            any(
                operand.tile_shape[-1] < 4
                for instruction in compiled.tisa_program.instructions
                for operand in instruction.operands
            )
        )

    def test_memory_plan_roundtrip_capacity_and_topology_guards(self):
        machine = minimal_machine_config()
        compiled = _compile(machine, shape=(8, 8, 8), tile_size=4)
        plan = compiled.backend_artifact.memory_plan
        self.assertEqual(MemoryPlan.from_dict(plan.to_dict()).to_dict(), plan.to_dict())
        self.assertEqual(
            BackendArtifact.from_dict(compiled.backend_artifact.to_dict()).to_dict(),
            compiled.backend_artifact.to_dict(),
        )

        timing_override = replace(
            machine,
            execution_units=tuple(
                replace(unit, count=unit.count + 1) if unit.name == "MXU" else unit
                for unit in machine.execution_units
            ),
        )
        self.assertEqual(machine.topology_hash(), timing_override.topology_hash())
        self.assertTrue(allocate_memory_plan_bindings(plan, timing_override))

        topology_override = replace(machine, operation_placements=())
        with self.assertRaisesRegex(ValueError, "topology/operand placement"):
            allocate_memory_plan_bindings(plan, topology_override)

        tiny = replace(
            lpu_like_machine_config(),
            memory_levels=tuple(
                replace(level, capacity_bytes=32) if level.name == "LMB" else level
                for level in lpu_like_machine_config().memory_levels
            ),
        )
        with self.assertRaisesRegex(ValueError, "overflow.*LMB"):
            _compile(tiny)


if __name__ == "__main__":
    unittest.main()
