import unittest
from dataclasses import replace

from npu_ooo.arch import (
    ExecutionUnitConfig,
    MemoryLevelConfig,
    OperandPlacementConfig,
    OperationPlacementConfig,
    TransferPathConfig,
    lpu_like_machine_config,
    minimal_machine_config,
)
from npu_ooo.backend.target_lowering import default_target_lowerer
from npu_ooo.compiler.fusion_compiler import default_fusion_compiler
from npu_ooo.compiler.graph_compiler import default_graph_compiler
from npu_ooo.compiler.tisa_generator import default_tisa_generator
from npu_ooo.ir import OperatorGraph, OperatorSpec, TargetPlan, TensorSpec


def _graph() -> OperatorGraph:
    return OperatorGraph(
        "target-boundary",
        (
            TensorSpec("lhs", (4, 4), "f32"),
            TensorSpec("rhs", (4, 4), "f32"),
            TensorSpec("out", (4, 4), "f32"),
        ),
        (
            OperatorSpec(
                "mm",
                "matmul",
                ("lhs", "rhs"),
                ("out",),
                (("M", 4), ("N", 4)),
                (("K", 4),),
            ),
        ),
    )


def _fc_fixture():
    graph = _graph()
    gc = default_graph_compiler().compile(
        graph, minimal_machine_config(), tile_size=4
    )
    fc = default_fusion_compiler().compile(gc, minimal_machine_config())
    virtual = default_tisa_generator().generate(fc)
    return graph, gc, fc, virtual


class TargetLoweringBoundaryTest(unittest.TestCase):
    def test_fc_is_symbolic_and_independent_of_target_route_hops(self):
        _graph_value, gc, fc_minimal, _virtual = _fc_fixture()
        fc_lpu = default_fusion_compiler().compile(gc, lpu_like_machine_config())

        self.assertEqual(fc_minimal.program.to_dict(), fc_lpu.program.to_dict())
        self.assertEqual(
            [item.attributes["tisa_stage"] for item in fc_minimal.program.instructions],
            ["load", "compute", "store"],
        )
        self.assertEqual(
            [item.unit_map.unit for item in fc_minimal.program.instructions],
            ["dma", "tensor", "dma"],
        )
        self.assertTrue(
            all(
                operand.tile_mem.memory_space is None
                for instruction in fc_minimal.program.instructions
                for operand in instruction.operands
            )
        )
        load = fc_minimal.program.instructions[0]
        self.assertEqual(
            {(operand.tile_mem.role, operand.tile_mem.visibility, operand.normalized_access)
             for operand in load.operands},
            {
                ("lhs", "Shared", "read"),
                ("lhs", "Private", "write"),
                ("rhs", "Shared", "read"),
                ("rhs", "Private", "write"),
            },
        )
        encoded = str(fc_minimal.to_dict())
        for target_name in ("GM", "UB", "LMB", "RMB", "PSB", "GDMA", "LDMA"):
            self.assertNotIn(target_name, encoded)

    def test_one_virtual_program_lowers_to_minimal_and_lpu(self):
        graph, gc, _fc, virtual = _fc_fixture()
        lowerer = default_target_lowerer()
        minimal = lowerer.lower(
            graph, gc.schedule, gc.tile_graph, minimal_machine_config(), virtual
        )
        lpu = lowerer.lower(
            graph, gc.schedule, gc.tile_graph, lpu_like_machine_config(), virtual
        )

        self.assertEqual(minimal.abstract_program_id, lpu.abstract_program_id)
        self.assertEqual(len(virtual.instructions), 3)
        self.assertEqual(len(minimal.instructions), 3)
        self.assertEqual(len(lpu.instructions), 5)
        minimal_compute = next(
            item.instruction
            for item in minimal.instructions
            if item.expansion_kind == "compute"
        )
        lpu_compute = next(
            item.instruction
            for item in lpu.instructions
            if item.expansion_kind == "compute"
        )
        self.assertEqual(
            {operand.tile_mem.physical_space for operand in minimal_compute.operands},
            {"SRAM"},
        )
        self.assertEqual(
            {operand.tile_mem.physical_space for operand in lpu_compute.operands},
            {"LMB", "RMB", "PSB"},
        )

    def test_route_and_engine_changes_only_target_plan(self):
        graph, gc, fc, virtual = _fc_fixture()
        base = minimal_machine_config()
        mid = MemoryLevelConfig(
            "MID", "DRAM", 4096, 32, 32,
            read_latency_cycles=1,
            write_latency_cycles=1,
        )
        routed = replace(
            base,
            config_id="minimal-with-mid",
            memory_levels=(*base.memory_levels, mid),
            execution_units=(
                *base.execution_units,
                ExecutionUnitConfig(
                    "AUX_DMA",
                    supported_ops=("load", "store", "copy"),
                    latency_cycles=2,
                ),
            ),
            transfer_paths=(
                *base.transfer_paths,
                TransferPathConfig(
                    "DRAM", "MID", "AUX_DMA", bandwidth_bytes_per_cycle=32
                ),
                TransferPathConfig("MID", "SRAM", "DMA", bandwidth_bytes_per_cycle=32),
            ),
            operation_placements=(
                OperationPlacementConfig(
                    "matmul",
                    "MXU",
                    (
                        OperandPlacementConfig(
                            "lhs", "SRAM", ("DRAM", "MID", "SRAM"), "input"
                        ),
                        base.placement("matmul").operand("rhs"),
                        base.placement("matmul").operand("output"),
                    ),
                ),
            ),
        )
        self.assertEqual(routed.validate(), ())
        routed_fc = default_fusion_compiler().compile(gc, routed)
        self.assertEqual(fc.program.to_dict(), routed_fc.program.to_dict())

        baseline = default_target_lowerer().lower(
            graph, gc.schedule, gc.tile_graph, base, virtual
        )
        changed = default_target_lowerer().lower(
            graph, gc.schedule, gc.tile_graph, routed, virtual
        )
        abstract_load = virtual.instructions[0].tisa_id
        self.assertEqual(len(baseline.targets_for(abstract_load)), 1)
        self.assertEqual(len(changed.targets_for(abstract_load)), 3)
        self.assertTrue(
            any(
                operand.tile_mem.physical_space == "MID"
                for item in changed.targets_for(abstract_load)
                for operand in item.instruction.operands
            )
        )
        self.assertNotIn("MID", str(routed_fc.to_dict()))
        self.assertNotIn("AUX_DMA", str(routed_fc.to_dict()))
        self.assertIn(
            "AUX_DMA",
            {
                item.instruction.unit_map.unit
                for item in changed.targets_for(abstract_load)
            },
        )

    def test_target_provenance_and_route_dependencies_roundtrip(self):
        graph, gc, _fc, virtual = _fc_fixture()
        plan = default_target_lowerer().lower(
            graph, gc.schedule, gc.tile_graph, lpu_like_machine_config(), virtual
        )
        self.assertEqual(TargetPlan.from_dict(plan.to_dict()).to_dict(), plan.to_dict())
        for item in plan.instructions:
            self.assertEqual(
                item.instruction.attributes["abstract_tisa_id"],
                item.abstract_tisa_id,
            )
        abstract_load = virtual.instructions[0].tisa_id
        load_targets = plan.targets_for(abstract_load)
        self.assertEqual(len(load_targets), 2)
        self.assertEqual(
            {dependency.source for dependency in load_targets[1].instruction.dependencies},
            {load_targets[0].instruction.tisa_id},
        )
        compute = plan.targets_for(virtual.instructions[1].tisa_id)[0]
        self.assertIn(
            load_targets[-1].instruction.tisa_id,
            {dependency.source for dependency in compute.instruction.dependencies},
        )


if __name__ == "__main__":
    unittest.main()
