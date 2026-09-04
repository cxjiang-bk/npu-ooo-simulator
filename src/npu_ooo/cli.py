from __future__ import annotations

"""Command-line entry points for the PyTorch-to-TISA research flow."""

import argparse
import csv
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from npu_ooo.arch import (
    load_machine_config,
    lpu_like_machine_config,
    minimal_machine_config,
    wide_mxu_machine_config,
)
from npu_ooo.backend import (
    AGGREGATIONS,
    INTERVALS,
    default_codegen_backend_registry,
    default_event_backend_registry,
    default_timing_provider_registry,
    import_rtl_completion_trace,
    load_mxu_vcs_log,
)
from npu_ooo.compiler import compile_torch_module
from npu_ooo.experiments import run_paper_benchmark_matrix, run_runtime_device_matrix
from npu_ooo.ir import (
    BackendArtifact,
    DynamicIndexBinding,
    OperatorGraph,
    RuntimeLayoutBinding,
    allocate_buffer_bindings,
    create_runtime_sequence,
    create_runtime_state_registry,
    create_runtime_submission,
    derive_tensor_lifetimes,
    derive_tensor_reuse_pairs,
)
from npu_ooo.scheduler import (
    SchedulerPolicy,
    SimulatorConfig,
    schedule_tisa_program,
    schedule_tisa_sequence,
)
from npu_ooo.trace import (
    ensure_output_layout,
    write_artifact_index,
    write_artifact_json,
    write_csv,
    write_execution_graph_dot,
    write_instruction_csv,
    write_json,
    write_operator_graph_dot,
    write_operator_graph_svg,
    write_png,
    write_svg,
    write_tile_graph_dot,
)


def _machine(name: str, config_path: Path | None = None):
    if config_path is not None:
        return load_machine_config(config_path)
    factories = {
        "minimal": minimal_machine_config,
        "wide-mxu": wide_mxu_machine_config,
        "lpu-like": lpu_like_machine_config,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unknown architecture profile '{name}'") from exc


def _timing_model(path: Path | None, provider: str | None = None):
    selected = provider or ("timing_table" if path is not None else "analytical")
    return default_timing_provider_registry().create(selected, path)


def _parse_positive_int_list(value: str, *, name: str) -> tuple[int, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    try:
        numbers = tuple(int(item) for item in items)
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of positive integers") from exc
    if any(number <= 0 for number in numbers):
        raise ValueError(f"{name} must contain only positive integers")
    return numbers


def _descriptor_availability(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"runtime availability config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid runtime availability JSON '{path}': {exc}") from exc
    if isinstance(payload, dict) and "descriptor_available_cycles" in payload:
        payload = payload["descriptor_available_cycles"]
    if not isinstance(payload, dict):
        raise ValueError("runtime availability config must be a TISA-id to cycle mapping")
    result: dict[str, float] = {}
    for tisa_id, cycle in payload.items():
        if not isinstance(tisa_id, str) or not tisa_id:
            raise ValueError("runtime availability TISA ids must be non-empty strings")
        if (
            isinstance(cycle, bool)
            or not isinstance(cycle, (int, float))
            or not math.isfinite(cycle)
            or cycle < 0
        ):
            raise ValueError(f"runtime availability for '{tisa_id}' must be non-negative")
        result[tisa_id] = float(cycle)
    return result


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _runtime_bindings(
    payload: dict[str, Any],
) -> tuple[
    tuple[DynamicIndexBinding, ...],
    tuple[RuntimeLayoutBinding, ...],
]:
    """Decode invocation-level dynamic bindings from a simulation manifest."""

    raw_indices = payload.get("dynamic_indices", ())
    if isinstance(raw_indices, dict):
        raw_indices = [
            {"expression_id": expression_id, "values": values}
            for expression_id, values in raw_indices.items()
        ]
    if not isinstance(raw_indices, (list, tuple)):
        raise ValueError("runtime_config.dynamic_indices must be a list or object")
    indices: list[DynamicIndexBinding] = []
    for item in raw_indices:
        if not isinstance(item, dict) or "expression_id" not in item or "values" not in item:
            raise ValueError("each dynamic index binding needs expression_id and values")
        try:
            binding = DynamicIndexBinding(
                str(item["expression_id"]),
                tuple(int(value) for value in item["values"]),
                attributes=item.get("attributes", {}),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid dynamic index binding") from exc
        issues = binding.validate()
        if issues:
            raise ValueError("invalid dynamic index binding: " + "; ".join(issues))
        indices.append(binding)

    raw_layouts = payload.get("dynamic_layouts", ())
    if isinstance(raw_layouts, dict):
        raw_layouts = [
            {"tensor": tensor, **spec} if isinstance(spec, dict) else {"tensor": tensor}
            for tensor, spec in raw_layouts.items()
        ]
    if not isinstance(raw_layouts, (list, tuple)):
        raise ValueError("runtime_config.dynamic_layouts must be a list or object")
    layouts: list[RuntimeLayoutBinding] = []
    for item in raw_layouts:
        if not isinstance(item, dict):
            raise ValueError("each dynamic layout binding must be an object")
        required = ("tensor", "shape", "strides_bytes")
        if any(key not in item for key in required):
            raise ValueError("each dynamic layout binding needs tensor, shape and strides_bytes")
        try:
            binding = RuntimeLayoutBinding(
                tensor=str(item["tensor"]),
                shape=tuple(int(value) for value in item["shape"]),
                strides_bytes=tuple(int(value) for value in item["strides_bytes"]),
                layout=str(item.get("layout", "runtime")),
                offset_bytes=int(item.get("offset_bytes", 0)),
                attributes=item.get("attributes", {}),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid dynamic layout binding") from exc
        issues = binding.validate()
        if issues:
            raise ValueError("invalid dynamic layout binding: " + "; ".join(issues))
        layouts.append(binding)
    return tuple(indices), tuple(layouts)


def _add_compile_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: str = "out/compile",
) -> None:
    parser.add_argument(
        "--torch-module",
        required=True,
        metavar="MODULE:CLASS_OR_FACTORY",
        help="import path to a zero-argument nn.Module class or factory",
    )
    parser.add_argument(
        "--input-shape",
        action="append",
        required=True,
        metavar="D0,D1,...",
        help="example input shape; repeat once per module input",
    )
    parser.add_argument(
        "--input-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument(
        "--tile-size-candidates",
        help="comma-separated tile sizes ranked by the GC cost model",
    )
    parser.add_argument(
        "--softmax-algorithm",
        choices=("materialized", "online"),
        default=None,
        help=(
            "Softmax payload strategy: materialized row-wise reductions or the "
            "analytical online state chain"
        ),
    )
    parser.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    parser.add_argument("--machine-config", type=Path)
    parser.add_argument(
        "--codegen-backend",
        choices=default_codegen_backend_registry().names(),
        default="analytical",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(default_output_dir))


def _add_simulation_options(
    parser: argparse.ArgumentParser,
    *,
    include_architecture: bool,
    include_output_dir: bool,
    include_runtime_device_matrix: bool = False,
    manifest_overrides: bool = False,
) -> None:
    """Add options consumed after compilation by runtime/device simulation."""

    if include_architecture:
        parser.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"))
        parser.add_argument("--machine-config", type=Path)
    parser.add_argument("--timing-config", type=Path)
    parser.add_argument(
        "--timing-provider",
        choices=default_timing_provider_registry().names(),
        default=None,
    )
    parser.add_argument(
        "--event-backend",
        choices=default_event_backend_registry().names(),
        default="analytical_event",
    )
    parser.add_argument(
        "--policy",
        choices=tuple(policy.value for policy in SchedulerPolicy),
        default=SchedulerPolicy.STATIC_PIPELINE.value,
    )
    if include_output_dir:
        parser.add_argument("--output-dir", type=Path, default=Path("out/simulate"))
    parser.add_argument("--instruction-queue-depth", type=int)
    parser.add_argument("--rob-entries", type=int)
    parser.add_argument("--max-inflight-tiles", type=int)
    parser.add_argument("--dependency-window", type=int)
    parser.add_argument("--ready-queue-depth", type=int)
    parser.add_argument("--address-scoreboard", action="store_true")
    parser.add_argument(
        "--memory-bank-scoreboard",
        action="store_true",
        help="model configured memory bank and read/write port conflicts",
    )
    parser.add_argument(
        "--dynamic-priority",
        choices=("critical_path", "oldest_first"),
        default="critical_path",
    )
    parser.add_argument(
        "--runtime-policy",
        choices=("static", "dynamic_ready_queue"),
        default=None if manifest_overrides else "static",
    )
    parser.add_argument("--runtime-chunk-size", type=int)
    parser.add_argument(
        "--runtime-base-address",
        type=lambda value: int(value, 0),
        default=None if manifest_overrides else 0x10000000,
    )
    parser.add_argument(
        "--runtime-alignment",
        type=int,
        default=None if manifest_overrides else 256,
    )
    parser.add_argument(
        "--runtime-buffer-policy",
        choices=("linear", "lifetime_reuse"),
        default=None if manifest_overrides else "linear",
    )
    parser.add_argument("--runtime-availability-config", type=Path)
    parser.add_argument(
        "--runtime-launch-latency",
        type=float,
        default=None if manifest_overrides else 0.0,
    )
    parser.add_argument(
        "--runtime-synchronization-cycles",
        type=float,
        default=None if manifest_overrides else 0.0,
    )
    parser.add_argument(
        "--runtime-invocations",
        type=int,
        default=None if manifest_overrides else 1,
        help="number of repeated invocations sharing persistent runtime state",
    )
    parser.add_argument(
        "--runtime-inter-invocation-gap",
        type=float,
        default=None if manifest_overrides else 0.0,
        help="cycles between state completion and the next invocation",
    )
    if include_runtime_device_matrix:
        parser.add_argument(
            "--runtime-device-matrix",
            action="store_true",
            help="run the four runtime/device static-dynamic policy combinations",
        )


def _add_compile_and_sim_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: str = "out/compile-and-sim",
) -> None:
    """Compose compile inputs with runtime/device options for the one-shot flow."""

    _add_compile_arguments(parser, default_output_dir=default_output_dir)
    _add_simulation_options(
        parser,
        include_architecture=False,
        include_output_dir=False,
        include_runtime_device_matrix=True,
    )


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options that affect runtime submission or device simulation only."""

    parser.add_argument("--compile-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    _add_simulation_options(
        parser,
        include_architecture=True,
        include_output_dir=True,
        manifest_overrides=True,
    )


def _add_paper_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="all or a comma-separated list of paper benchmark case ids",
    )
    parser.add_argument(
        "--variant",
        choices=("micro", "paper_shape"),
        default="micro",
        help="benchmark workload scale; micro is the default reproducible proxy",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=1,
        help="transformer depth proxy; 1 preserves the one-block benchmark rows",
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument(
        "--tile-size-candidates",
        help="comma-separated tile sizes ranked by the GC cost model",
    )
    parser.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    parser.add_argument("--machine-config", type=Path)
    parser.add_argument("--timing-config", type=Path)
    parser.add_argument(
        "--timing-provider",
        choices=default_timing_provider_registry().names(),
        default=None,
    )
    parser.add_argument(
        "--event-backend",
        choices=default_event_backend_registry().names(),
        default="analytical_event",
    )
    parser.add_argument(
        "--codegen-backend",
        choices=default_codegen_backend_registry().names(),
        default="analytical",
    )
    parser.add_argument(
        "--softmax-algorithm",
        choices=("materialized", "online"),
        default=None,
        help="softmax strategy applied during GC/FC compilation",
    )
    parser.add_argument(
        "--runtime-device-matrix",
        action="store_true",
        help="also vary runtime submission policy, producing four combinations per case",
    )
    parser.add_argument(
        "--device-policies",
        default="static_pipeline,dynamic_ready_queue",
        help="comma-separated device scheduler policies",
    )
    parser.add_argument("--instruction-queue-depth", type=int)
    parser.add_argument("--rob-entries", type=int)
    parser.add_argument("--max-inflight-tiles", type=int)
    parser.add_argument("--dependency-window", type=int)
    parser.add_argument("--ready-queue-depth", type=int)
    parser.add_argument("--address-scoreboard", action="store_true")
    parser.add_argument("--memory-bank-scoreboard", action="store_true")
    parser.add_argument(
        "--dynamic-priority",
        choices=("critical_path", "oldest_first"),
        default="critical_path",
    )
    parser.add_argument("--runtime-chunk-size", type=int)
    parser.add_argument("--runtime-launch-latency", type=float, default=0.0)
    parser.add_argument("--runtime-synchronization-cycles", type=float, default=0.0)
    parser.add_argument("--runtime-availability-config", type=Path)
    parser.add_argument("--runtime-base-address", type=lambda value: int(value, 0), default=0x10000000)
    parser.add_argument("--runtime-alignment", type=int, default=256)
    parser.add_argument(
        "--runtime-buffer-policy",
        choices=("linear", "lifetime_reuse"),
        default="linear",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record an explicit error row and continue compiling other cases",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("out/paper-matrix"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile PyTorch modules to TISA and run scheduling experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_only = commands.add_parser(
        "compile",
        help="PyTorch -> torch.export -> Torch-XLA -> StableHLO -> TISA/backend compile package",
    )
    _add_compile_arguments(compile_only, default_output_dir="out/compile")

    compile_and_sim = commands.add_parser(
        "compile-and-sim",
        help="compile a PyTorch module and immediately run the simulator",
    )
    _add_compile_and_sim_arguments(
        compile_and_sim,
        default_output_dir="out/compile-and-sim",
    )

    simulate = commands.add_parser(
        "simulate",
        help="load a compile package, bind runtime values, and run the simulator",
    )
    _add_simulation_arguments(simulate)

    paper_matrix = commands.add_parser(
        "paper-matrix",
        help="compile the paper benchmark registry once per case and compare device policies",
    )
    _add_paper_matrix_arguments(paper_matrix)

    rtl_trace = commands.add_parser(
        "import-rtl-trace",
        help="convert an RTL completion trace into an MXU timing profile",
    )
    rtl_trace.add_argument("--input", type=Path, required=True)
    rtl_trace.add_argument("--output", type=Path, required=True)
    rtl_trace.add_argument("--interval", choices=INTERVALS, default=INTERVALS[0])
    rtl_trace.add_argument("--aggregation", choices=AGGREGATIONS, default="median")
    rtl_trace.add_argument("--unmatched-matmul", choices=("error", "analytical"), default="error")
    rtl_trace.add_argument("--name", default="systolic_mxu_profile")

    rtl_log = commands.add_parser(
        "import-rtl-log",
        help="parse the repository MXU VCS log into completion-trace JSON",
    )
    rtl_log.add_argument("--input", type=Path, required=True)
    rtl_log.add_argument("--output", type=Path, required=True)
    rtl_log.add_argument("--k-per-tile", type=int, default=8)
    return parser


def run_import_rtl_trace(args: argparse.Namespace) -> int:
    profile = import_rtl_completion_trace(
        args.input,
        interval=args.interval,
        aggregation=args.aggregation,
        unmatched_matmul=args.unmatched_matmul,
        name=args.name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "shape_count": len(profile["matmul_profiles"])},
            sort_keys=True,
        )
    )
    return 0


def run_import_rtl_log(args: argparse.Namespace) -> int:
    trace = load_mxu_vcs_log(args.input, k_per_tile=args.k_per_tile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "record_count": len(trace["records"])}, sort_keys=True))
    return 0


def _load_torch_module(specification: str, input_shapes: list[str], dtype_name: str):
    if ":" not in specification:
        raise ValueError("--torch-module must use MODULE:CLASS_OR_FACTORY syntax")
    module_name, factory_name = specification.split(":", 1)
    try:
        python_module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError(f"cannot import PyTorch module '{module_name}': {exc}") from exc
    constructor: Any = python_module
    try:
        for attribute in factory_name.split("."):
            constructor = getattr(constructor, attribute)
    except AttributeError as exc:
        raise ValueError(f"PyTorch module class or factory '{specification}' does not exist") from exc
    if not callable(constructor):
        raise ValueError(f"PyTorch module class or factory '{specification}' is not callable")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ValueError("compile/compile-and-sim requires PyTorch") from exc
    try:
        module = constructor()
    except Exception as exc:
        raise ValueError(f"PyTorch module class or factory '{specification}' failed: {exc}") from exc
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"PyTorch module class or factory '{specification}' did not return nn.Module")

    torch.manual_seed(0)
    dtype = getattr(torch, dtype_name)
    shapes = tuple(_parse_positive_int_list(value, name="--input-shape") for value in input_shapes)
    example_inputs = tuple(torch.randn(*shape, dtype=dtype) for shape in shapes)
    return module.eval(), example_inputs, factory_name.rsplit(".", 1)[-1]


def _write_policy_matrix(
    root: Path,
    compiled,
    runtime_buffers,
    machine,
    args: argparse.Namespace,
    timing_model,
    simulator_config: SimulatorConfig,
    event_backend,
    descriptor_availability: dict[str, float],
) -> None:
    matrix_root = root / "policy_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)
    cases = run_runtime_device_matrix(
        compiled.backend_artifact,
        runtime_buffers,
        machine,
        chunk_size=args.runtime_chunk_size,
        launch_latency_cycles=args.runtime_launch_latency,
        synchronization_cycles=args.runtime_synchronization_cycles,
        descriptor_available_cycles=descriptor_availability,
        timing_model=timing_model,
        simulator_config=simulator_config,
        event_backend=event_backend,
    )
    baseline = next(
        case.result.total_cycles
        for case in cases
        if case.runtime_policy == "static"
        and case.device_policy == SchedulerPolicy.STATIC_PIPELINE.value
    )
    records: list[dict[str, object]] = []
    for case in cases:
        case_dir = matrix_root / case.case_id
        ensure_output_layout(case_dir)
        write_artifact_json(case.submission, case_dir / "runtime_submission.json")
        write_json(case.result, case_dir / "summary.json")
        write_csv(case.result, case_dir / "tasks.csv")
        write_instruction_csv(case.result, case_dir / "tisa_instructions.csv")
        write_svg(case.result, case_dir / "swimlane.svg")
        write_png(case.result, case_dir / "swimlane.png")
        write_artifact_json(case.result.perfetto_trace(), case_dir / "perfetto.json")
        record = {
            **case.to_dict(),
            "speedup_vs_static_runtime_static_device": (
                baseline / case.result.total_cycles if case.result.total_cycles else None
            ),
        }
        records.append(record)
        write_artifact_json(record, case_dir / "manifest.json")
    records.sort(key=lambda item: (str(item["runtime_policy"]), str(item["device_policy"])))
    with (matrix_root / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (matrix_root / "sweep.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _paper_case_ids(value: str) -> tuple[str, ...] | None:
    if value.strip().lower() == "all":
        return None
    case_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not case_ids:
        raise ValueError("--benchmarks must be 'all' or a comma-separated list of case ids")
    return case_ids


def _paper_device_policies(value: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in value.split(",") if item.strip())
    valid = {policy.value for policy in SchedulerPolicy}
    if not policies or any(policy not in valid for policy in policies):
        raise ValueError(
            "--device-policies must contain one or more of: " + ", ".join(sorted(valid))
        )
    if len(set(policies)) != len(policies):
        raise ValueError("--device-policies must not contain duplicates")
    return policies


def _write_compiled_case_artifacts(compiled, case_dir: Path, machine) -> None:
    """Write the shared staged compiler artifacts for one matrix case."""

    ensure_output_layout(case_dir)
    write_artifact_json(compiled.source_frontend, case_dir / "source_frontend_import.json")
    write_artifact_json(compiled.stablehlo, case_dir / "stablehlo_module.json")
    (case_dir / "00_frontend" / "generated.mlir").write_text(
        compiled.stablehlo.text, encoding="utf-8"
    )
    write_artifact_json(compiled.frontend, case_dir / "frontend_import.json")
    write_artifact_json(compiled.graph, case_dir / "canonical_graph.json")
    if compiled.gc_artifact is not None:
        write_artifact_json(compiled.gc_artifact, case_dir / "gc_artifact.json")
        pass_dump_dir = case_dir / "01_gc" / "pass_dumps"
        for snapshot in compiled.gc_artifact.pass_dumps:
            write_artifact_json(
                snapshot,
                pass_dump_dir / f"{snapshot.pass_index:02d}_{snapshot.pass_name}.json",
            )
    write_artifact_json(compiled.schedule, case_dir / "schedule.json")
    write_artifact_json(compiled.attributes["compile_statistics"], case_dir / "compile_statistics.json")
    write_artifact_json(compiled.tile_graph, case_dir / "tile_graph.json")
    if compiled.tisa_dialect is not None:
        write_artifact_json(compiled.tisa_dialect, case_dir / "tisa_dialect.json")
        write_artifact_json(compiled.tisa_dialect.attributes, case_dir / "fc_diagnostics.json")
    write_artifact_json(compiled.tisa_program, case_dir / "tisa_program.json")
    write_artifact_json(compiled, case_dir / "compiled_artifact.json")
    write_artifact_json(compiled.backend_artifact, case_dir / "backend_artifact.json")
    write_artifact_json(compiled.backend_artifact.execution_graph, case_dir / "execution_graph.json")
    write_artifact_json(machine, case_dir / "machine.json")
    write_operator_graph_dot(compiled.graph, case_dir / "operator_graph.dot")
    write_operator_graph_svg(compiled.graph, case_dir / "operator_graph.svg")
    write_tile_graph_dot(compiled.tile_graph, case_dir / "tile_graph.dot")
    write_execution_graph_dot(compiled.backend_artifact.execution_graph, case_dir / "execution_graph.dot")


def _write_paper_policy_artifacts(case_dir: Path, case) -> None:
    """Write one policy's runtime, simulation and trace artifacts."""

    policy_dir = case_dir / "policy_matrix" / case.case_id
    (policy_dir / "05_runtime").mkdir(parents=True, exist_ok=True)
    (policy_dir / "06_simulation").mkdir(parents=True, exist_ok=True)
    (policy_dir / "07_trace").mkdir(parents=True, exist_ok=True)
    write_artifact_json(case.submission, policy_dir / "05_runtime" / "runtime_submission.json")
    write_json(case.result, policy_dir / "06_simulation" / "summary.json")
    write_csv(case.result, policy_dir / "06_simulation" / "tasks.csv")
    write_instruction_csv(case.result, policy_dir / "06_simulation" / "tisa_instructions.csv")
    write_svg(case.result, policy_dir / "07_trace" / "swimlane.svg")
    write_png(case.result, policy_dir / "07_trace" / "swimlane.png")
    write_artifact_json(case.result.perfetto_trace(), policy_dir / "07_trace" / "perfetto.json")


def _write_paper_matrix(root: Path, matrix, machine) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    index_entries: list[dict[str, Any]] = []
    for run in matrix.runs:
        case_dir = root / run.case_id / run.variant
        case_dir.mkdir(parents=True, exist_ok=True)
        if run.compiled is not None:
            _write_compiled_case_artifacts(run.compiled, case_dir, machine)
        for case in run.cases:
            _write_paper_policy_artifacts(case_dir, case)
        case_records = [dict(record) for record in run.to_records()]
        for record in case_records:
            record["case_output_dir"] = str(case_dir.relative_to(root))
            record["policy_output_dir"] = str(
                (case_dir / "policy_matrix" / str(record["policy_case_id"])).relative_to(root)
            ) if record.get("policy_case_id") else None
        records.extend(case_records)
        (case_dir / "summary.json").write_text(
            json.dumps(case_records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (case_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_id": run.case_id,
                    "variant": run.variant,
                    "spec": dict(run.spec),
                    "status": "error" if run.error else "ok",
                    "error": run.error,
                    "artifact_id": run.artifact_id,
                    "program_id": run.program_id,
                    "tisa_instruction_count": run.tisa_instruction_count,
                    "tile_count": run.tile_count,
                    "primitive_task_count": run.primitive_task_count,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_artifact_index(case_dir)
        index_entries.append(
            {
                "benchmark_id": run.case_id,
                "variant": run.variant,
                "status": "error" if run.error else "ok",
                "case_output_dir": str(case_dir.relative_to(root)),
                "policy_output_dirs": [
                    str((case_dir / "policy_matrix" / case.case_id).relative_to(root))
                    for case in run.cases
                ],
            }
        )
    if records:
        fieldnames = tuple(sorted({key for record in records for key in record}))
        with (root / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in record.items()
                    }
                )
    (root / "sweep.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "matrix_index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": matrix.variant,
                "case_count": len(index_entries),
                "cases": index_entries,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# 论文模型矩阵\n\n"
        "每个 benchmark 只编译一次，然后在同一份 TISA/backend artifact 和 buffer binding 上比较设备调度策略。"
        "默认 runtime 固定为 static，只改变 device policy；使用 `--runtime-device-matrix` 才会展开四组合。\n\n"
        "`sweep.csv/json` 是跨 case 汇总；每个 `<benchmark>/<variant>/` 目录保存该 case 的汇总和 manifest。"
        "`matrix_index.json` 是本次运行实际 case 的权威索引；复用 output 目录时，旧 case 目录可能仍存在，"
        "应以该索引而不是目录枚举为准。当前 workload 是 scaled micro 或 representative paper_shape proxy，"
        "reference 字段来自论文，不能与 analytical cycle 混为绝对性能。\n",
        encoding="utf-8",
    )
    return records


def run_paper_matrix(args: argparse.Namespace) -> int:
    if args.variant == "paper_shape":
        print(
            "warning: paper_shape uses representative large inputs and may require substantial "
            "compile time and memory; it is not a full-model or absolute paper-performance run",
            file=sys.stderr,
        )
    machine = _machine(args.arch, args.machine_config)
    if args.softmax_algorithm is not None:
        machine = replace(
            machine,
            attributes={
                **dict(machine.attributes),
                "softmax_algorithm": args.softmax_algorithm,
            },
        )
    tile_size_candidates = (
        _parse_positive_int_list(args.tile_size_candidates, name="--tile-size-candidates")
        if args.tile_size_candidates
        else None
    )
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        memory_bank_scoreboard=args.memory_bank_scoreboard,
        dynamic_priority=args.dynamic_priority,
    )
    timing_model = _timing_model(args.timing_config, args.timing_provider)
    event_backend = default_event_backend_registry().create(args.event_backend)
    codegen_backend = default_codegen_backend_registry().create(args.codegen_backend)
    device_policies = _paper_device_policies(args.device_policies)
    descriptor_availability = _descriptor_availability(args.runtime_availability_config)
    runtime_policies = (
        ("static", "dynamic_ready_queue")
        if args.runtime_device_matrix
        else ("static",)
    )
    matrix = run_paper_benchmark_matrix(
        machine,
        case_ids=_paper_case_ids(args.benchmarks),
        variant=args.variant,
        layer_count=args.layer_count,
        tile_size=args.tile_size,
        tile_size_candidates=tile_size_candidates,
        runtime_policies=runtime_policies,
        device_policies=device_policies,
        runtime_chunk_size=args.runtime_chunk_size,
        runtime_launch_latency=args.runtime_launch_latency,
        runtime_synchronization_cycles=args.runtime_synchronization_cycles,
        descriptor_available_cycles=descriptor_availability,
        runtime_base_address=args.runtime_base_address,
        runtime_alignment=args.runtime_alignment,
        runtime_buffer_policy=args.runtime_buffer_policy,
        softmax_algorithm=args.softmax_algorithm,
        timing_model=timing_model,
        simulator_config=simulator_config,
        event_backend=event_backend,
        codegen_backend=codegen_backend,
        continue_on_error=args.continue_on_error,
    )
    records = _write_paper_matrix(args.output_dir, matrix, machine)
    manifest = {
        "schema_version": 1,
        "variant": args.variant,
        "layer_count": args.layer_count,
        "benchmarks": [run.case_id for run in matrix.runs],
        "architecture": args.arch,
        "machine_hash": machine.stable_hash(),
        "tile_size": args.tile_size,
        "tile_size_candidates": list(tile_size_candidates or (args.tile_size,)),
        "softmax_algorithm": machine.attributes.get("softmax_algorithm", "materialized"),
        "codegen_backend": codegen_backend.name,
        "runtime_policies": list(runtime_policies),
        "device_policies": list(device_policies),
        "timing_provider": getattr(timing_model, "name", "analytical"),
        "event_backend": event_backend.name,
        "record_count": len(records),
        "failed_cases": [run.case_id for run in matrix.runs if run.error],
    }
    (args.output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "case_count": len(matrix.runs),
                "record_count": len(records),
                "failed_cases": manifest["failed_cases"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if manifest["failed_cases"] else 0


def _compile_from_args(args: argparse.Namespace):
    module, example_inputs, factory_name = _load_torch_module(
        args.torch_module,
        args.input_shape,
        args.input_dtype,
    )
    machine = _machine(args.arch, args.machine_config)
    tile_size_candidates = (
        _parse_positive_int_list(args.tile_size_candidates, name="--tile-size-candidates")
        if args.tile_size_candidates
        else None
    )
    if args.softmax_algorithm is not None:
        machine = replace(
            machine,
            attributes={
                **dict(machine.attributes),
                "softmax_algorithm": args.softmax_algorithm,
            },
        )
    codegen_backend = default_codegen_backend_registry().create(args.codegen_backend)
    compiled = compile_torch_module(
        module,
        example_inputs,
        machine,
        model_id=args.model_id or factory_name,
        tile_size=args.tile_size,
        tile_size_candidates=tile_size_candidates,
        codegen_backend=codegen_backend,
    )

    return compiled, machine, factory_name


def _write_compile_artifacts(compiled, machine, output_dir: Path) -> None:
    """Persist the compiler-owned stages used by both CLI execution modes."""

    ensure_output_layout(output_dir)
    write_artifact_json(compiled.source_frontend, output_dir / "source_frontend_import.json")
    write_artifact_json(compiled.stablehlo, output_dir / "stablehlo_module.json")
    (output_dir / "00_frontend" / "generated.mlir").write_text(
        compiled.stablehlo.text,
        encoding="utf-8",
    )
    write_artifact_json(compiled.frontend, output_dir / "frontend_import.json")
    write_artifact_json(compiled.graph, output_dir / "canonical_graph.json")
    if compiled.gc_artifact is not None:
        write_artifact_json(compiled.gc_artifact, output_dir / "gc_artifact.json")
        pass_dump_dir = output_dir / "01_gc" / "pass_dumps"
        for snapshot in compiled.gc_artifact.pass_dumps:
            filename = f"{snapshot.pass_index:02d}_{snapshot.pass_name}.json"
            write_artifact_json(snapshot, pass_dump_dir / filename)
    write_artifact_json(compiled.schedule, output_dir / "schedule.json")
    write_artifact_json(
        compiled.attributes["compile_statistics"],
        output_dir / "compile_statistics.json",
    )
    write_artifact_json(compiled.tile_graph, output_dir / "tile_graph.json")
    if compiled.tisa_dialect is not None:
        write_artifact_json(compiled.tisa_dialect, output_dir / "tisa_dialect.json")
        write_artifact_json(
            compiled.tisa_dialect.attributes,
            output_dir / "fc_diagnostics.json",
        )
    write_artifact_json(compiled.tisa_program, output_dir / "tisa_program.json")
    write_artifact_json(compiled, output_dir / "compiled_artifact.json")
    write_artifact_json(compiled.backend_artifact, output_dir / "backend_artifact.json")
    write_artifact_json(compiled.backend_artifact.execution_graph, output_dir / "execution_graph.json")
    write_artifact_json(machine, output_dir / "machine.json")
    write_operator_graph_dot(compiled.graph, output_dir / "operator_graph.dot")
    write_operator_graph_svg(compiled.graph, output_dir / "operator_graph.svg")
    write_tile_graph_dot(compiled.tile_graph, output_dir / "tile_graph.dot")
    write_execution_graph_dot(compiled.backend_artifact.execution_graph, output_dir / "execution_graph.dot")


def _compile_manifest(compiled, machine) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "compile_package",
        "compiler_pipeline": compiled.attributes["compiler_pipeline"],
        "frontend_path": compiled.attributes["frontend_path"],
        "model_id": compiled.source_frontend.model_id,
        "stablehlo_exporter": "torch-xla",
        "stablehlo_exporter_version": compiled.attributes["stablehlo_exporter_version"],
        "stablehlo_verified": True,
        "stablehlo_version": compiled.attributes["stablehlo_version"],
        "architecture": machine.config_id,
        "machine_hash": machine.stable_hash(),
        "codegen_backend": compiled.attributes["codegen_backend"],
        "tisa_program_id": compiled.tisa_program.program_id,
        "artifact_id": compiled.backend_artifact.artifact_id,
        "compile_artifacts": {
            "backend_artifact": "04_backend/backend_artifact.json",
            "canonical_graph": "01_gc/canonical_graph.json",
            "machine": "04_backend/machine.json",
            "tisa_program": "03_tisa/tisa_program.json",
        },
    }


def run_compile(args: argparse.Namespace) -> int:
    compiled, machine, _factory_name = _compile_from_args(args)
    _write_compile_artifacts(compiled, machine, args.output_dir)
    manifest = _compile_manifest(compiled, machine)
    write_artifact_json(manifest, args.output_dir / "manifest.json")
    write_artifact_index(args.output_dir)
    print(json.dumps({
        "artifact_id": compiled.backend_artifact.artifact_id,
        "model_id": compiled.source_frontend.model_id,
        "tisa_instructions": len(compiled.tisa_program.instructions),
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


def run_compile_and_sim(args: argparse.Namespace) -> int:
    compiled, machine, _factory_name = _compile_from_args(args)
    _write_compile_artifacts(compiled, machine, args.output_dir)

    lifetimes = derive_tensor_lifetimes(compiled.tisa_program)
    reuse_pairs = derive_tensor_reuse_pairs(compiled.tisa_program)
    runtime_buffers = allocate_buffer_bindings(
        compiled.graph.tensors,
        base_address=args.runtime_base_address,
        alignment_bytes=args.runtime_alignment,
        lifetimes=lifetimes,
        reuse_buffers=args.runtime_buffer_policy == "lifetime_reuse",
        reuse_pairs=reuse_pairs,
    )
    descriptor_availability = _descriptor_availability(args.runtime_availability_config)
    if args.runtime_invocations <= 0:
        raise ValueError("--runtime-invocations must be a positive integer")
    runtime_sequence = None
    if args.runtime_invocations > 1:
        if args.runtime_device_matrix:
            raise ValueError(
                "--runtime-device-matrix currently supports one invocation; "
                "run a RuntimeSequence separately for multi-step decode"
            )
        state_registry = create_runtime_state_registry(
            compiled.backend_artifact,
            runtime_buffers,
        )
        if not state_registry.state_ids():
            raise ValueError(
                "--runtime-invocations > 1 requires a compiled persistent state contract"
            )
        runtime_sequence = create_runtime_sequence(
            compiled.backend_artifact,
            state_registry,
            invocation_count=args.runtime_invocations,
            sequence_id=f"sequence.{compiled.tisa_program.program_id}",
            policy=args.runtime_policy,
            chunk_size=args.runtime_chunk_size,
            launch_latency_cycles=args.runtime_launch_latency,
            synchronization_cycles=args.runtime_synchronization_cycles,
            descriptor_available_cycles=descriptor_availability,
            inter_invocation_gap_cycles=args.runtime_inter_invocation_gap,
        )
        runtime_submission = runtime_sequence.invocations[0]
        write_artifact_json(runtime_sequence, args.output_dir / "runtime_sequence.json")
    else:
        runtime_submission = create_runtime_submission(
            compiled.backend_artifact,
            runtime_buffers,
            submission_id=f"submission.{compiled.tisa_program.program_id}",
            policy=args.runtime_policy,
            chunk_size=args.runtime_chunk_size,
            launch_latency_cycles=args.runtime_launch_latency,
            synchronization_cycles=args.runtime_synchronization_cycles,
            descriptor_available_cycles=descriptor_availability,
        )
    write_artifact_json(runtime_submission, args.output_dir / "runtime_submission.json")

    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        memory_bank_scoreboard=args.memory_bank_scoreboard,
        dynamic_priority=args.dynamic_priority,
    )
    timing_model = _timing_model(args.timing_config, args.timing_provider)
    event_backend = default_event_backend_registry().create(args.event_backend)
    if runtime_sequence is not None:
        result = schedule_tisa_sequence(
            compiled.backend_artifact,
            runtime_sequence,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            event_backend=event_backend,
        )
    else:
        result = schedule_tisa_program(
            compiled.backend_artifact,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            runtime_submission=runtime_submission,
            event_backend=event_backend,
        )
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_instruction_csv(result, args.output_dir / "tisa_instructions.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        result.metrics.get("address_hazards", []),
        args.output_dir / "address_dependencies.json",
    )

    allocation_span = (
        max(buffer.end_address for buffer in runtime_buffers)
        - min(buffer.base_address for buffer in runtime_buffers)
        if runtime_buffers
        else 0
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "simulation_result",
        "compiler_pipeline": compiled.attributes["compiler_pipeline"],
        "frontend_path": compiled.attributes["frontend_path"],
        "model_id": compiled.source_frontend.model_id,
        "stablehlo_exporter": "torch-xla",
        "stablehlo_exporter_version": compiled.attributes["stablehlo_exporter_version"],
        "stablehlo_verified": True,
        "stablehlo_version": compiled.attributes["stablehlo_version"],
        "architecture": machine.config_id,
        "softmax_algorithm": machine.attributes.get("softmax_algorithm", "materialized"),
        "machine_hash": machine.stable_hash(),
        "codegen_backend": compiled.attributes["codegen_backend"],
        "timing_provider": getattr(timing_model, "name", "analytical"),
        "event_backend": event_backend.name,
        "runtime_policy": runtime_submission.policy,
        "runtime_buffer_policy": args.runtime_buffer_policy,
        "runtime_command_chunk_count": len(runtime_submission.commands),
        "runtime_buffer_count": len(runtime_submission.buffers),
        "runtime_invocation_count": (
            len(runtime_sequence.invocations) if runtime_sequence is not None else 1
        ),
        "runtime_state_contract": (
            runtime_sequence.attributes.get("state_contract")
            if runtime_sequence is not None
            else runtime_submission.attributes.get("state_contract")
        ),
        "runtime_state_ids": (
            list(runtime_sequence.state_registry.state_ids())
            if runtime_sequence is not None
            else [
                item["state_id"]
                for item in runtime_submission.attributes.get("state_buffers", ())
                if isinstance(item, dict) and "state_id" in item
            ]
        ),
        "runtime_state_dependency_count": (
            len(runtime_sequence.dependencies) if runtime_sequence is not None else 0
        ),
        "runtime_allocation_span_bytes": allocation_span,
        "policy": result.policy,
        "scheduler_target": "tisa",
        "tisa_instruction_count": len(compiled.tisa_program.instructions),
        "primitive_task_count": len(compiled.backend_artifact.execution_graph.tasks),
        "total_cycles": result.total_cycles,
        "total_cycles_including_runtime": result.metrics.get(
            "total_cycles_including_runtime", result.total_cycles
        ),
        "calibration_status": result.metrics["calibration_status"],
        "simulator_config": simulator_config.to_dict(),
    }
    write_artifact_json(manifest, args.output_dir / "manifest.json")

    if args.runtime_device_matrix:
        _write_policy_matrix(
            args.output_dir,
            compiled,
            runtime_buffers,
            machine,
            args,
            timing_model,
            simulator_config,
            event_backend,
            descriptor_availability,
        )
    write_artifact_index(args.output_dir)
    print(
        json.dumps(
            {
                "model_id": compiled.source_frontend.model_id,
                "tisa_instructions": len(compiled.tisa_program.instructions),
                "total_cycles": result.total_cycles,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_compile_package(root: Path):
    """Load the portable compiler-owned files without importing PyTorch."""

    root = root.expanduser().resolve()
    backend_path = root / "04_backend" / "backend_artifact.json"
    graph_path = root / "01_gc" / "canonical_graph.json"
    machine_path = root / "04_backend" / "machine.json"
    for path in (backend_path, graph_path, machine_path):
        if not path.is_file():
            raise ValueError(f"compile package is missing required artifact: {path}")
    backend = BackendArtifact.from_dict(_read_json_object(backend_path, description="backend artifact"))
    graph = OperatorGraph.from_dict(_read_json_object(graph_path, description="canonical graph"))
    from npu_ooo.arch import MachineConfig

    machine = MachineConfig.from_dict(_read_json_object(machine_path, description="machine config"))
    manifest_path = root / "manifest.json"
    manifest = (
        _read_json_object(manifest_path, description="compile manifest")
        if manifest_path.is_file()
        else {}
    )
    if manifest.get("artifact_kind") not in {None, "compile_package", "simulation_result"}:
        raise ValueError(f"'{manifest_path}' is not a compatible compile/simulation manifest")
    manifest_artifact_id = manifest.get("artifact_id", manifest.get("compile_artifact_id"))
    if manifest_artifact_id not in {None, backend.artifact_id}:
        raise ValueError("compile manifest artifact_id does not match backend artifact")
    return root, backend, graph, machine, manifest


def _simulation_config(args: argparse.Namespace) -> SimulatorConfig:
    return SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        memory_bank_scoreboard=args.memory_bank_scoreboard,
        dynamic_priority=args.dynamic_priority,
    )


def run_simulate(args: argparse.Namespace) -> int:
    compile_root, artifact, graph, compiled_machine, compile_manifest = _load_compile_package(
        args.compile_dir
    )
    runtime_payload = (
        _read_json_object(args.runtime_config, description="runtime config")
        if args.runtime_config is not None
        else {}
    )
    machine = (
        _machine(args.arch, args.machine_config)
        if args.arch is not None or args.machine_config is not None
        else compiled_machine
    )
    runtime_policy = args.runtime_policy or str(runtime_payload.get("runtime_policy", "static"))
    chunk_size = args.runtime_chunk_size
    if chunk_size is None and runtime_payload.get("runtime_chunk_size") is not None:
        chunk_size = int(runtime_payload["runtime_chunk_size"])
    base_address = args.runtime_base_address
    if base_address is None:
        base_address = int(runtime_payload.get("runtime_base_address", 0x10000000))
    alignment = args.runtime_alignment
    if alignment is None:
        alignment = int(runtime_payload.get("runtime_alignment", 256))
    buffer_policy = args.runtime_buffer_policy or str(
        runtime_payload.get("runtime_buffer_policy", "linear")
    )
    launch_latency = args.runtime_launch_latency
    if launch_latency is None:
        launch_latency = float(runtime_payload.get("runtime_launch_latency", 0.0))
    synchronization = args.runtime_synchronization_cycles
    if synchronization is None:
        synchronization = float(runtime_payload.get("runtime_synchronization_cycles", 0.0))
    gap = args.runtime_inter_invocation_gap
    if gap is None:
        gap = float(runtime_payload.get("runtime_inter_invocation_gap", 0.0))
    invocation_count = args.runtime_invocations
    if invocation_count is None:
        invocation_count = int(runtime_payload.get("runtime_invocations", 1))
    if invocation_count <= 0:
        raise ValueError("runtime invocation count must be positive")
    availability_path = args.runtime_availability_config
    if availability_path is None and runtime_payload.get("runtime_availability_config"):
        availability_path = (
            args.runtime_config.parent
            / str(runtime_payload["runtime_availability_config"])
        ).resolve()
    if availability_path is not None:
        descriptor_availability = _descriptor_availability(availability_path)
    else:
        inline_availability = runtime_payload.get("descriptor_available_cycles", {})
        if not isinstance(inline_availability, dict):
            raise ValueError("runtime_config.descriptor_available_cycles must be an object")
        descriptor_availability = {}
        for tisa_id, cycle in inline_availability.items():
            if not isinstance(tisa_id, str) or not tisa_id:
                raise ValueError("descriptor availability TISA ids must be non-empty strings")
            if (
                isinstance(cycle, bool)
                or not isinstance(cycle, (int, float))
                or not math.isfinite(cycle)
                or cycle < 0
            ):
                raise ValueError(f"descriptor availability for '{tisa_id}' must be non-negative")
            descriptor_availability[tisa_id] = float(cycle)
    lifetimes = derive_tensor_lifetimes(artifact.program)
    reuse_pairs = derive_tensor_reuse_pairs(artifact.program)
    buffers = allocate_buffer_bindings(
        graph.tensors,
        base_address=base_address,
        alignment_bytes=alignment,
        lifetimes=lifetimes,
        reuse_buffers=buffer_policy == "lifetime_reuse",
        reuse_pairs=reuse_pairs,
    )

    invocation_payloads = runtime_payload.get("invocations")
    if invocation_payloads is not None:
        if not isinstance(invocation_payloads, list) or len(invocation_payloads) != invocation_count:
            raise ValueError("runtime_config.invocations length must equal runtime invocation count")
    else:
        invocation_payloads = [runtime_payload] * invocation_count
    invocation_indices: list[tuple[DynamicIndexBinding, ...]] = []
    invocation_layouts: list[tuple[RuntimeLayoutBinding, ...]] = []
    for item in invocation_payloads:
        if not isinstance(item, dict):
            raise ValueError("each runtime invocation must be an object")
        indices, layouts = _runtime_bindings(item)
        invocation_indices.append(indices)
        invocation_layouts.append(layouts)

    runtime_sequence = None
    if invocation_count > 1:
        state_registry = create_runtime_state_registry(artifact, buffers)
        if not state_registry.state_ids():
            raise ValueError("multiple runtime invocations require a persistent state contract")
        runtime_sequence = create_runtime_sequence(
            artifact,
            state_registry,
            invocation_count=invocation_count,
            sequence_id=f"sequence.{artifact.program.program_id}",
            policy=runtime_policy,
            chunk_size=chunk_size,
            launch_latency_cycles=launch_latency,
            synchronization_cycles=synchronization,
            descriptor_available_cycles=descriptor_availability,
            inter_invocation_gap_cycles=gap,
            invocation_dynamic_indices=invocation_indices,
            invocation_dynamic_layouts=invocation_layouts,
        )
        runtime_submission = runtime_sequence.invocations[0]
    else:
        runtime_submission = create_runtime_submission(
            artifact,
            buffers,
            submission_id=f"submission.{artifact.program.program_id}",
            policy=runtime_policy,
            chunk_size=chunk_size,
            launch_latency_cycles=launch_latency,
            synchronization_cycles=synchronization,
            descriptor_available_cycles=descriptor_availability,
            dynamic_index_bindings=invocation_indices[0],
            dynamic_layout_bindings=invocation_layouts[0],
        )

    timing_model = _timing_model(args.timing_config, args.timing_provider)
    event_backend = default_event_backend_registry().create(args.event_backend)
    simulator_config = _simulation_config(args)
    if runtime_sequence is not None:
        result = schedule_tisa_sequence(
            artifact,
            runtime_sequence,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            event_backend=event_backend,
        )
    else:
        result = schedule_tisa_program(
            artifact,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            runtime_submission=runtime_submission,
            event_backend=event_backend,
        )

    ensure_output_layout(args.output_dir)
    write_artifact_json(runtime_submission, args.output_dir / "runtime_submission.json")
    if runtime_sequence is not None:
        write_artifact_json(runtime_sequence, args.output_dir / "runtime_sequence.json")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_instruction_csv(result, args.output_dir / "tisa_instructions.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        result.metrics.get("address_hazards", []),
        args.output_dir / "address_dependencies.json",
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "simulation_result",
        "compile_package": str(compile_root),
        "compile_artifact_id": artifact.artifact_id,
        "compile_program_id": artifact.program.program_id,
        "compile_manifest": compile_manifest,
        "architecture": machine.config_id,
        "machine_hash": machine.stable_hash(),
        "timing_provider": getattr(timing_model, "name", "analytical"),
        "event_backend": event_backend.name,
        "policy": result.policy,
        "runtime_policy": runtime_submission.policy,
        "runtime_invocation_count": invocation_count,
        "dynamic_index_binding_count": sum(len(item) for item in invocation_indices),
        "dynamic_layout_binding_count": sum(len(item) for item in invocation_layouts),
        "total_cycles": result.total_cycles,
        "total_cycles_including_runtime": result.metrics.get(
            "total_cycles_including_runtime", result.total_cycles
        ),
        "simulator_config": simulator_config.to_dict(),
    }
    write_artifact_json(manifest, args.output_dir / "manifest.json")
    write_artifact_index(args.output_dir)
    print(json.dumps({
        "compile_artifact_id": artifact.artifact_id,
        "total_cycles": result.total_cycles,
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runners = {
        "compile": run_compile,
        "compile-and-sim": run_compile_and_sim,
        "simulate": run_simulate,
        "paper-matrix": run_paper_matrix,
        "import-rtl-trace": run_import_rtl_trace,
        "import-rtl-log": run_import_rtl_log,
    }
    return runners[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
