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
from npu_ooo.experiments import run_runtime_device_matrix
from npu_ooo.ir import (
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


def _add_compile_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--policy",
        choices=tuple(policy.value for policy in SchedulerPolicy),
        default=SchedulerPolicy.STATIC_PIPELINE.value,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("out/compile-model"))
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
        default="static",
    )
    parser.add_argument("--runtime-chunk-size", type=int)
    parser.add_argument(
        "--runtime-base-address",
        type=lambda value: int(value, 0),
        default=0x10000000,
    )
    parser.add_argument("--runtime-alignment", type=int, default=256)
    parser.add_argument(
        "--runtime-buffer-policy",
        choices=("linear", "lifetime_reuse"),
        default="linear",
    )
    parser.add_argument("--runtime-availability-config", type=Path)
    parser.add_argument("--runtime-launch-latency", type=float, default=0.0)
    parser.add_argument("--runtime-synchronization-cycles", type=float, default=0.0)
    parser.add_argument(
        "--runtime-invocations",
        type=int,
        default=1,
        help="number of repeated invocations sharing persistent runtime state",
    )
    parser.add_argument(
        "--runtime-inter-invocation-gap",
        type=float,
        default=0.0,
        help="cycles between state completion and the next invocation",
    )
    parser.add_argument(
        "--runtime-device-matrix",
        action="store_true",
        help="run the four runtime/device static-dynamic policy combinations",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile PyTorch modules to TISA and run scheduling experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_model = commands.add_parser(
        "compile-model",
        help="PyTorch -> torch.export -> Torch-XLA -> StableHLO -> TISA -> simulator",
    )
    _add_compile_arguments(compile_model)

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
    print(json.dumps({"output": str(args.output), "shape_count": len(profile["matmul_profiles"])}, sort_keys=True))
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
        raise ValueError("compile-model requires PyTorch") from exc
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


def run_compile_model(args: argparse.Namespace) -> int:
    module, example_inputs, factory_name = _load_torch_module(
        args.torch_module,
        args.input_shape,
        args.input_dtype,
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
    codegen_backend = default_codegen_backend_registry().create(args.codegen_backend)
    compiled = compile_torch_module(
        module,
        example_inputs,
        machine,
        model_id=args.model_id or factory_name,
        tile_size=args.tile_size,
        codegen_backend=codegen_backend,
    )

    ensure_output_layout(args.output_dir)
    write_artifact_json(compiled.source_frontend, args.output_dir / "source_frontend_import.json")
    write_artifact_json(compiled.stablehlo, args.output_dir / "stablehlo_module.json")
    (args.output_dir / "00_frontend" / "generated.mlir").write_text(
        compiled.stablehlo.text,
        encoding="utf-8",
    )
    write_artifact_json(compiled.frontend, args.output_dir / "frontend_import.json")
    write_artifact_json(compiled.graph, args.output_dir / "canonical_graph.json")
    if compiled.gc_artifact is not None:
        write_artifact_json(compiled.gc_artifact, args.output_dir / "gc_artifact.json")
        pass_dump_dir = args.output_dir / "01_gc" / "pass_dumps"
        for snapshot in compiled.gc_artifact.pass_dumps:
            filename = f"{snapshot.pass_index:02d}_{snapshot.pass_name}.json"
            write_artifact_json(snapshot, pass_dump_dir / filename)
    write_artifact_json(compiled.schedule, args.output_dir / "schedule.json")
    write_artifact_json(
        compiled.attributes["compile_statistics"],
        args.output_dir / "compile_statistics.json",
    )
    write_artifact_json(compiled.tile_graph, args.output_dir / "tile_graph.json")
    if compiled.tisa_dialect is not None:
        write_artifact_json(compiled.tisa_dialect, args.output_dir / "tisa_dialect.json")
        write_artifact_json(
            compiled.tisa_dialect.attributes,
            args.output_dir / "fc_diagnostics.json",
        )
    write_artifact_json(compiled.tisa_program, args.output_dir / "tisa_program.json")
    write_artifact_json(compiled, args.output_dir / "compiled_artifact.json")
    write_artifact_json(compiled.backend_artifact, args.output_dir / "backend_artifact.json")
    write_artifact_json(compiled.backend_artifact.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_operator_graph_dot(compiled.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(compiled.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(compiled.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(compiled.backend_artifact.execution_graph, args.output_dir / "execution_graph.dot")

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
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")

    allocation_span = (
        max(buffer.end_address for buffer in runtime_buffers)
        - min(buffer.base_address for buffer in runtime_buffers)
        if runtime_buffers
        else 0
    )
    manifest = {
        "schema_version": 1,
        "compiler_pipeline": compiled.attributes["compiler_pipeline"],
        "frontend_path": compiled.attributes["frontend_path"],
        "model_id": compiled.source_frontend.model_id,
        "stablehlo_exporter": "torch-xla",
        "stablehlo_exporter_version": compiled.attributes["stablehlo_exporter_version"],
        "stablehlo_verified": True,
        "stablehlo_version": compiled.attributes["stablehlo_version"],
        "architecture": args.arch,
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runners = {
        "compile-model": run_compile_model,
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
