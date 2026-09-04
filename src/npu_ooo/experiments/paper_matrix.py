"""Batch orchestration for the paper benchmark registry.

This module deliberately sits above ``runtime_matrix``.  It owns model
construction and one-time compilation, while the lower-level matrix keeps the
runtime/device policy experiment independent of model names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from npu_ooo.arch import MachineConfig
from npu_ooo.backend import CodegenBackend, EventBackend
from npu_ooo.compiler import compile_torch_module
from npu_ooo.ir import allocate_buffer_bindings
from npu_ooo.ir import derive_tensor_lifetimes, derive_tensor_reuse_pairs
from npu_ooo.scheduler import SchedulerPolicy
from npu_ooo.simulator import SimulatorConfig, TimingModel

from .runtime_matrix import RuntimeDeviceCase, run_runtime_device_matrix


@dataclass(frozen=True)
class PaperBenchmarkRun:
    """One paper registry case and its policy matrix results."""

    case_id: str
    variant: str
    spec: Mapping[str, Any]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    artifact_id: str | None
    program_id: str | None
    tisa_instruction_count: int
    tile_count: int = 0
    primitive_task_count: int = 0
    stablehlo_variant: str | None = None
    stablehlo_exporter_version: str | None = None
    selected_tile_size: int | None = None
    workload_attributes: Mapping[str, Any] = field(default_factory=dict)
    compiled: Any | None = field(default=None, repr=False, compare=False)
    cases: tuple[RuntimeDeviceCase, ...] = ()
    error: str | None = None

    def to_records(self) -> tuple[dict[str, Any], ...]:
        reference = dict(self.spec.get("reference", {}))
        records: list[dict[str, Any]] = []
        for case in self.cases:
            policy_record = case.to_dict()
            record = {
                "case_id": self.case_id,
                "benchmark_id": self.case_id,
                "policy_case_id": policy_record.pop("case_id"),
                "variant": self.variant,
                "status": "ok",
                "artifact_id": self.artifact_id,
                "program_id": self.program_id,
                "tisa_instruction_count": self.tisa_instruction_count,
                "tile_count": self.tile_count,
                "primitive_task_count": self.primitive_task_count,
                "stablehlo_variant": self.stablehlo_variant,
                "stablehlo_exporter_version": self.stablehlo_exporter_version,
                "selected_tile_size": self.selected_tile_size,
                "input_shapes": [list(shape) for shape in self.input_shapes],
                "input_dtypes": list(self.input_dtypes),
                "model_name": self.spec.get("model_name"),
                "model_family": self.spec.get("model_family"),
                "phase": self.spec.get("phase"),
                "dtype": self.spec.get("dtype"),
                "workload_kind": self.spec.get("workload_kind"),
                "unsupported_features": self.spec.get("unsupported_features", []),
                "simulation_dimensions": self.workload_attributes.get("simulation_dimensions"),
                "reference": reference,
                **policy_record,
            }
            records.append(record)
        if not records:
            records.append(
                {
                    "case_id": self.case_id,
                    "benchmark_id": self.case_id,
                    "policy_case_id": None,
                    "variant": self.variant,
                    "status": "error" if self.error else "empty",
                    "error": self.error,
                    "artifact_id": self.artifact_id,
                    "program_id": self.program_id,
                    "tisa_instruction_count": self.tisa_instruction_count,
                    "tile_count": self.tile_count,
                    "primitive_task_count": self.primitive_task_count,
                    "stablehlo_variant": self.stablehlo_variant,
                    "stablehlo_exporter_version": self.stablehlo_exporter_version,
                    "selected_tile_size": self.selected_tile_size,
                    "input_shapes": [list(shape) for shape in self.input_shapes],
                    "input_dtypes": list(self.input_dtypes),
                    "model_name": self.spec.get("model_name"),
                    "model_family": self.spec.get("model_family"),
                    "phase": self.spec.get("phase"),
                    "dtype": self.spec.get("dtype"),
                    "workload_kind": self.spec.get("workload_kind"),
                    "unsupported_features": self.spec.get("unsupported_features", []),
                    "simulation_dimensions": self.workload_attributes.get("simulation_dimensions"),
                    "reference": reference,
                }
            )
        baseline = next(
            (
                float(record["total_cycles"])
                for record in records
                if record.get("runtime_policy") == "static"
                and record.get("device_policy") == SchedulerPolicy.STATIC_PIPELINE.value
            ),
            None,
        )
        for record in records:
            record["baseline_case_id"] = (
                f"{self.case_id}:runtime-static__device-static_pipeline"
                if baseline is not None
                else None
            )
            record["speedup_vs_static"] = (
                baseline / float(record["total_cycles"])
                if baseline is not None and record.get("total_cycles")
                else None
            )
        return tuple(records)


@dataclass(frozen=True)
class PaperBenchmarkMatrix:
    """Stable result container for a batch of paper benchmark cases."""

    variant: str
    runs: tuple[PaperBenchmarkRun, ...]

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(record for run in self.runs for record in run.to_records())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "variant": self.variant,
            "case_count": len(self.runs),
            "runs": [
                {
                    "case_id": run.case_id,
                    "variant": run.variant,
                    "spec": dict(run.spec),
                    "status": "error" if run.error else "ok",
                    "error": run.error,
                    "artifact_id": run.artifact_id,
                    "program_id": run.program_id,
                    "tisa_instruction_count": run.tisa_instruction_count,
                    "record_count": len(run.to_records()),
                    "simulation_dimensions": run.workload_attributes.get("simulation_dimensions"),
                }
                for run in self.runs
            ],
            "records": list(self.records()),
        }


def _case_workload(case_id: str, variant: str, *, layer_count: int = 1):
    from examples.paper_benchmarks import build_paper_benchmark
    return build_paper_benchmark(case_id, variant=variant, layer_count=layer_count)


def run_paper_benchmark_matrix(
    machine: MachineConfig,
    *,
    case_ids: Sequence[str] | None = None,
    variant: str = "micro",
    layer_count: int = 1,
    tile_size: int = 32,
    tile_size_candidates: Sequence[int] | None = None,
    runtime_chunk_size: int | None = None,
    runtime_launch_latency: float = 0.0,
    runtime_synchronization_cycles: float = 0.0,
    descriptor_available_cycles: Mapping[str, float] | None = None,
    runtime_base_address: int = 0x10000000,
    runtime_alignment: int = 256,
    runtime_buffer_policy: str = "linear",
    runtime_policies: Sequence[str] = ("static",),
    device_policies: Sequence[str | SchedulerPolicy] = (
        SchedulerPolicy.STATIC_PIPELINE,
        SchedulerPolicy.DYNAMIC_READY_QUEUE,
    ),
    timing_model: TimingModel | None = None,
    simulator_config: SimulatorConfig | None = None,
    event_backend: EventBackend | None = None,
    codegen_backend: CodegenBackend | None = None,
    softmax_algorithm: str | None = None,
    continue_on_error: bool = False,
) -> PaperBenchmarkMatrix:
    """Compile each selected registry case once and run policy combinations."""

    from examples.paper_benchmarks import paper_benchmark_specs

    selected = tuple(case_ids) if case_ids is not None else tuple(
        spec.case_id for spec in paper_benchmark_specs()
    )
    if not selected:
        raise ValueError("paper benchmark selection must contain at least one case")
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if len(set(selected)) != len(selected):
        raise ValueError("paper benchmark selection must not contain duplicate case ids")
    if not runtime_policies or not device_policies:
        raise ValueError("paper benchmark matrix requires at least one runtime and device policy")
    if runtime_buffer_policy not in {"linear", "lifetime_reuse"}:
        raise ValueError("runtime_buffer_policy must be 'linear' or 'lifetime_reuse'")
    if softmax_algorithm not in {None, "materialized", "online"}:
        raise ValueError("softmax_algorithm must be 'materialized' or 'online' when specified")
    compile_machine = machine
    if softmax_algorithm is not None:
        compile_machine = replace(
            machine,
            attributes={**dict(machine.attributes), "softmax_algorithm": softmax_algorithm},
        )
    known_specs = {spec.case_id: spec for spec in paper_benchmark_specs()}
    unknown = tuple(case_id for case_id in selected if case_id not in known_specs)
    if unknown:
        known = ", ".join(known_specs)
        raise ValueError(
            f"unknown paper benchmark(s): {', '.join(unknown)}; choose one of: {known}"
        )
    runs: list[PaperBenchmarkRun] = []
    for case_id in selected:
        workload = None
        try:
            workload = _case_workload(case_id, variant, layer_count=layer_count)
            compiled = compile_torch_module(
                workload.module,
                workload.inputs,
                compile_machine,
                model_id=case_id,
                tile_size=tile_size,
                tile_size_candidates=tile_size_candidates,
                codegen_backend=codegen_backend,
            )
            lifetimes = derive_tensor_lifetimes(compiled.tisa_program)
            reuse_pairs = derive_tensor_reuse_pairs(compiled.tisa_program)
            buffers = allocate_buffer_bindings(
                compiled.graph.tensors,
                base_address=runtime_base_address,
                alignment_bytes=runtime_alignment,
                lifetimes=lifetimes,
                reuse_buffers=runtime_buffer_policy == "lifetime_reuse",
                reuse_pairs=reuse_pairs,
            )
            cases = run_runtime_device_matrix(
                compiled.backend_artifact,
                buffers,
                compile_machine,
                runtime_policies=runtime_policies,
                device_policies=device_policies,
                chunk_size=runtime_chunk_size,
                launch_latency_cycles=runtime_launch_latency,
                synchronization_cycles=runtime_synchronization_cycles,
                descriptor_available_cycles=descriptor_available_cycles,
                timing_model=timing_model,
                simulator_config=simulator_config,
                event_backend=event_backend,
            )
            runs.append(
                PaperBenchmarkRun(
                    case_id=case_id,
                    variant=variant,
                    spec=workload.spec.to_dict(),
                    input_shapes=tuple(tuple(int(item) for item in value.shape) for value in workload.inputs),
                    input_dtypes=tuple(str(value.dtype).removeprefix("torch.") for value in workload.inputs),
                    artifact_id=compiled.backend_artifact.artifact_id,
                    program_id=compiled.tisa_program.program_id,
                    tisa_instruction_count=len(compiled.tisa_program.instructions),
                    tile_count=len(compiled.tile_graph.tiles),
                    primitive_task_count=len(compiled.backend_artifact.execution_graph.tasks),
                    stablehlo_variant=compiled.stablehlo.variant,
                    stablehlo_exporter_version=compiled.stablehlo.provenance.get("exporter_version"),
                    selected_tile_size=compiled.schedule.attributes.get("selected_tile_size"),
                    workload_attributes=dict(workload.attributes),
                    compiled=compiled,
                    cases=cases,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            spec = known_specs[case_id].to_dict()
            runs.append(
                PaperBenchmarkRun(
                    case_id=case_id,
                    variant=variant,
                    spec=spec,
                    input_shapes=(),
                    input_dtypes=(),
                    artifact_id=None,
                    program_id=None,
                    tisa_instruction_count=0,
                    workload_attributes=(dict(workload.attributes) if workload is not None else {}),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return PaperBenchmarkMatrix(variant=variant, runs=tuple(runs))


__all__ = ["PaperBenchmarkMatrix", "PaperBenchmarkRun", "run_paper_benchmark_matrix"]
