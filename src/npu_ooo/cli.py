from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from itertools import product
from pathlib import Path

from npu_ooo.arch import load_machine_config, lpu_like_machine_config, minimal_machine_config, wide_mxu_machine_config
from npu_ooo.backend import (
    AGGREGATIONS,
    INTERVALS,
    default_codegen_backend_registry,
    default_event_backend_registry,
    default_timing_provider_registry,
    import_rtl_completion_trace,
    load_mxu_vcs_log,
)
from npu_ooo.compiler import (
    compile_frontend_import,
    compile_stablehlo_file,
    compile_torch_module,
    compile_torch_module_through_stablehlo,
)
from npu_ooo.frontend import JsonGraphAdapter
from npu_ooo.experiments import run_runtime_device_matrix
from npu_ooo.benchmarks import (
    build_decoder_block_case,
    build_decoder_block_model,
    build_attention_case,
    build_attention_model,
    available_model_presets,
    build_model_preset,
    build_transformer_block_case,
    build_transformer_block_model,
    build_elementwise_case,
    build_elementwise_model,
    build_layernorm_case,
    build_layernorm_model,
    build_reduce_case,
    build_reduce_model,
    build_softmax_case,
    build_softmax_model,
    build_rmsnorm_case,
    build_rmsnorm_model,
    build_two_matmul_case,
    build_two_matmul_model,
)
from npu_ooo.ir import (
    allocate_buffer_bindings,
    create_runtime_submission,
    derive_tensor_lifetimes,
    derive_tensor_reuse_pairs,
    default_elementwise_schedule,
    default_layernorm_schedule,
    default_mixed_schedule,
    default_reduce_schedule,
    default_softmax_schedule,
    default_rmsnorm_schedule,
    default_two_matmul_schedule,
)
from npu_ooo.lowering import (
    lower_elementwise,
    lower_layernorm,
    lower_mixed_model,
    lower_reduce,
    lower_rmsnorm,
    lower_softmax,
    lower_two_matmul,
)
from npu_ooo.scheduler import (
    SchedulerPolicy,
    SimulatorConfig,
    StaticPipelineConfig,
    schedule_execution_graph,
    schedule_tisa_program,
)
from npu_ooo.trace import (
    write_artifact_json,
    write_csv,
    write_instruction_csv,
    write_execution_graph_dot,
    write_json,
    write_png,
    write_operator_graph_dot,
    write_operator_graph_svg,
    write_svg,
    write_tile_graph_dot,
    ensure_output_layout,
    write_artifact_index,
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


def _event_backend(name: str):
    return default_event_backend_registry().create(name)


def _codegen_backend(name: str):
    return default_codegen_backend_registry().create(name)


def run_import_rtl_trace(args) -> int:
    """Convert an offline RTL completion trace into an MXU timing profile."""

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
            {
                "input": str(args.input),
                "output": str(args.output),
                "format": profile["format"],
                "interval": args.interval,
                "aggregation": args.aggregation,
                "shape_count": len(profile["matmul_profiles"]),
                "record_count": profile["metadata"]["record_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def run_import_rtl_log(args) -> int:
    """Convert the repository MXU VCS console log into trace JSON."""

    trace = load_mxu_vcs_log(args.input, k_per_tile=args.k_per_tile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "format": trace["format"],
                "record_count": len(trace["records"]),
                "intervals_available": trace["metadata"]["intervals_available"],
            },
            sort_keys=True,
        )
    )
    return 0


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


def _parse_offsets(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        offsets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--static-stage-offsets must be a comma-separated list of numbers") from exc
    if not offsets:
        raise ValueError("--static-stage-offsets must contain at least one offset")
    if any(offset < 0 for offset in offsets):
        raise ValueError("--static-stage-offsets values must be non-negative")
    return offsets


def _parse_list(value: str, *, name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    return items


def _parse_positive_int_list(value: str, *, name: str) -> tuple[int, ...]:
    items = _parse_list(value, name=name)
    try:
        numbers = tuple(int(item) for item in items)
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of positive integers") from exc
    if any(number <= 0 for number in numbers):
        raise ValueError(f"{name} must contain only positive integers")
    return numbers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configurable NPU tile scheduling experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rtl_trace = subparsers.add_parser(
        "import-rtl-trace",
        help="convert a versioned RTL completion trace into an MXU timing profile",
    )
    rtl_trace.add_argument("--input", type=Path, required=True, help="RTL completion trace JSON or CSV")
    rtl_trace.add_argument("--output", type=Path, required=True, help="output systolic MXU profile JSON")
    rtl_trace.add_argument("--interval", choices=INTERVALS, default=INTERVALS[0])
    rtl_trace.add_argument("--aggregation", choices=AGGREGATIONS, default="median")
    rtl_trace.add_argument(
        "--unmatched-matmul",
        choices=("error", "analytical"),
        default="error",
        help="policy for compiled shapes absent from the generated profile",
    )
    rtl_trace.add_argument("--name", default="systolic_mxu_profile")
    rtl_log = subparsers.add_parser(
        "import-rtl-log",
        help="parse the repository MXU testbench VCS console log into trace JSON",
    )
    rtl_log.add_argument("--input", type=Path, required=True, help="tb_mxu VCS console log")
    rtl_log.add_argument("--output", type=Path, required=True, help="rtl_completion_trace.v1 JSON")
    rtl_log.add_argument(
        "--k-per-tile",
        type=int,
        default=8,
        help="physical K elements represented by one MXU K1 tile (default: RTL K0=8)",
    )
    compile_model = subparsers.add_parser(
        "compile-model",
        help="import a canonical graph and compile it through the unified frontend/backend pipeline",
    )
    compile_input = compile_model.add_mutually_exclusive_group(required=True)
    compile_input.add_argument(
        "--graph-json",
        type=Path,
        help="canonical OperatorGraph JSON (operator_graph.json or a graph wrapper)",
    )
    compile_input.add_argument(
        "--stablehlo-file",
        type=Path,
        help="StableHLO textual MLIR input",
    )
    compile_input.add_argument(
        "--torch-module",
        metavar="MODULE:FACTORY",
        help="import a zero-argument PyTorch module factory",
    )
    compile_model.add_argument(
        "--input-shape",
        action="append",
        default=[],
        metavar="D0,D1,...",
        help="generated PyTorch tensor shape; repeat for multiple module inputs",
    )
    compile_model.add_argument(
        "--input-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float32",
        help="dtype for generated PyTorch inputs",
    )
    compile_model.add_argument("--model-id", default=None)
    compile_model.add_argument("--variant", default="imported-v0")
    compile_model.add_argument(
        "--through-stablehlo",
        action="store_true",
        help="for a PyTorch module, export StableHLO and re-import it before canonical compilation",
    )
    compile_model.add_argument(
        "--stablehlo-backend",
        choices=("official", "auto", "textual"),
        default="official",
        help="StableHLO implementation for --through-stablehlo (default: official)",
    )
    compile_model.add_argument(
        "--stablehlo-exporter",
        choices=("project", "torch-xla"),
        default="project",
        help="PyTorch-to-StableHLO exporter for --through-stablehlo (default: project)",
    )
    compile_model.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="SYMBOL=VALUE",
        help="override a symbolic graph dimension; repeat for multiple symbols",
    )
    compile_model.add_argument("--tile-size", type=int, default=32)
    compile_model.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    compile_model.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    compile_model.add_argument(
        "--timing-config",
        type=Path,
        help="load timing-table overrides or an external timing-provider profile JSON",
    )
    compile_model.add_argument(
        "--timing-provider",
        choices=default_timing_provider_registry().names(),
        default=None,
        help="timing backend registry entry; defaults to timing_table when --timing-config is set",
    )
    compile_model.add_argument(
        "--event-backend",
        choices=default_event_backend_registry().names(),
        default="analytical_event",
        help="device event-engine backend for the TISA scheduler",
    )
    compile_model.add_argument(
        "--codegen-backend",
        choices=default_codegen_backend_registry().names(),
        default="analytical",
        help="TISA-to-payload code-generation backend",
    )
    compile_model.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    compile_model.add_argument(
        "--scheduler-target",
        choices=("tisa", "primitive"),
        default="tisa",
        help="schedule TISA instructions (default) or the legacy global primitive graph",
    )
    compile_model.add_argument("--output-dir", type=Path, default=Path("out/compile-model"))
    compile_model.add_argument("--instruction-queue-depth", type=int)
    compile_model.add_argument("--rob-entries", type=int)
    compile_model.add_argument("--max-inflight-tiles", type=int)
    compile_model.add_argument("--dependency-window", type=int)
    compile_model.add_argument("--ready-queue-depth", type=int)
    compile_model.add_argument("--address-scoreboard", action="store_true")
    compile_model.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    compile_model.add_argument(
        "--runtime-policy",
        choices=("static", "dynamic_ready_queue"),
        default="static",
        help="software submission policy for RuntimeSubmission (device policy remains --policy)",
    )
    compile_model.add_argument(
        "--runtime-chunk-size",
        type=int,
        default=None,
        help="number of TISA descriptors per runtime command chunk; default submits one chunk",
    )
    compile_model.add_argument(
        "--runtime-base-address",
        type=lambda value: int(value, 0),
        default=0x10000000,
        help="base physical address for the runtime linear allocator (decimal or 0x-prefixed)",
    )
    compile_model.add_argument("--runtime-alignment", type=int, default=256)
    compile_model.add_argument(
        "--runtime-buffer-policy",
        choices=("linear", "lifetime_reuse"),
        default="linear",
        help="physical buffer allocation policy; lifetime_reuse requires TISA dependency proof",
    )
    compile_model.add_argument(
        "--runtime-availability-config",
        type=Path,
        help="JSON mapping from compiled TISA id to earliest host submission cycle",
    )
    compile_model.add_argument(
        "--runtime-launch-latency",
        type=float,
        default=0.0,
        help="software launch latency per command chunk in simulator cycles",
    )
    compile_model.add_argument(
        "--runtime-synchronization-cycles",
        type=float,
        default=0.0,
        help="host/device synchronization cost after device completion",
    )
    compile_model.add_argument(
        "--runtime-device-matrix",
        action="store_true",
        help="run static/dynamic runtime x static/dynamic TISA device policies on one compiled artifact",
    )
    two_mm = subparsers.add_parser("two-mm", help="compile and schedule the 2mm benchmark")
    two_mm.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    two_mm.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    two_mm.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    two_mm.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    two_mm.add_argument("--output-dir", type=Path, default=Path("out/two-mm"))
    two_mm.add_argument("--instruction-queue-depth", type=int)
    two_mm.add_argument("--rob-entries", type=int)
    two_mm.add_argument("--max-inflight-tiles", type=int)
    two_mm.add_argument("--dependency-window", type=int)
    two_mm.add_argument("--ready-queue-depth", type=int)
    two_mm.add_argument("--address-scoreboard", action="store_true")
    two_mm.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    two_mm.add_argument(
        "--static-stage-offsets",
        help="comma-separated static stage reservation offsets; enables explicit static pipeline reservations",
    )
    two_mm.add_argument("--static-stage-ii", type=float, default=1.0)
    elementwise = subparsers.add_parser("elementwise", help="compile and schedule a residual-add benchmark")
    elementwise.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    elementwise.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    elementwise.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    elementwise.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    elementwise.add_argument("--output-dir", type=Path, default=Path("out/elementwise"))
    elementwise.add_argument("--instruction-queue-depth", type=int)
    elementwise.add_argument("--rob-entries", type=int)
    elementwise.add_argument("--max-inflight-tiles", type=int)
    elementwise.add_argument("--dependency-window", type=int)
    elementwise.add_argument("--ready-queue-depth", type=int)
    elementwise.add_argument("--address-scoreboard", action="store_true")
    elementwise.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    elementwise.add_argument("--static-stage-offsets")
    elementwise.add_argument("--static-stage-ii", type=float, default=1.0)
    reduce = subparsers.add_parser("reduce", help="compile and schedule a row-reduction benchmark")
    reduce.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    reduce.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    reduce.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    reduce.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    reduce.add_argument("--output-dir", type=Path, default=Path("out/reduce"))
    reduce.add_argument("--instruction-queue-depth", type=int)
    reduce.add_argument("--rob-entries", type=int)
    reduce.add_argument("--max-inflight-tiles", type=int)
    reduce.add_argument("--dependency-window", type=int)
    reduce.add_argument("--ready-queue-depth", type=int)
    reduce.add_argument("--address-scoreboard", action="store_true")
    reduce.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    reduce.add_argument("--static-stage-offsets")
    reduce.add_argument("--static-stage-ii", type=float, default=1.0)
    softmax = subparsers.add_parser("softmax", help="compile and schedule a row-softmax benchmark")
    softmax.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    softmax.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    softmax.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    softmax.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    softmax.add_argument("--output-dir", type=Path, default=Path("out/softmax"))
    softmax.add_argument("--instruction-queue-depth", type=int)
    softmax.add_argument("--rob-entries", type=int)
    softmax.add_argument("--max-inflight-tiles", type=int)
    softmax.add_argument("--dependency-window", type=int)
    softmax.add_argument("--ready-queue-depth", type=int)
    softmax.add_argument("--address-scoreboard", action="store_true")
    softmax.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    softmax.add_argument("--static-stage-offsets")
    softmax.add_argument("--static-stage-ii", type=float, default=1.0)
    rmsnorm = subparsers.add_parser("rmsnorm", help="compile and schedule an RMSNorm benchmark")
    rmsnorm.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    rmsnorm.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    rmsnorm.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    rmsnorm.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    rmsnorm.add_argument("--output-dir", type=Path, default=Path("out/rmsnorm"))
    rmsnorm.add_argument("--instruction-queue-depth", type=int)
    rmsnorm.add_argument("--rob-entries", type=int)
    rmsnorm.add_argument("--max-inflight-tiles", type=int)
    rmsnorm.add_argument("--dependency-window", type=int)
    rmsnorm.add_argument("--ready-queue-depth", type=int)
    rmsnorm.add_argument("--address-scoreboard", action="store_true")
    rmsnorm.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    rmsnorm.add_argument("--static-stage-offsets")
    rmsnorm.add_argument("--static-stage-ii", type=float, default=1.0)
    decoder_block = subparsers.add_parser(
        "decoder-block",
        help="compile and schedule an RMSNorm -> Matmul -> ResidualAdd fragment",
    )
    decoder_block.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    decoder_block.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    decoder_block.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    decoder_block.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    decoder_block.add_argument("--output-dir", type=Path, default=Path("out/decoder-block"))
    decoder_block.add_argument("--instruction-queue-depth", type=int)
    decoder_block.add_argument("--rob-entries", type=int)
    decoder_block.add_argument("--max-inflight-tiles", type=int)
    decoder_block.add_argument("--dependency-window", type=int)
    decoder_block.add_argument("--ready-queue-depth", type=int)
    decoder_block.add_argument("--address-scoreboard", action="store_true")
    decoder_block.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    decoder_block.add_argument("--static-stage-offsets")
    decoder_block.add_argument("--static-stage-ii", type=float, default=1.0)
    attention = subparsers.add_parser(
        "attention",
        help="compile and schedule a single-head QK-softmax-PV attention fragment",
    )
    attention.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    attention.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    attention.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    attention.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    attention.add_argument("--output-dir", type=Path, default=Path("out/attention"))
    attention.add_argument("--instruction-queue-depth", type=int)
    attention.add_argument("--rob-entries", type=int)
    attention.add_argument("--max-inflight-tiles", type=int)
    attention.add_argument("--dependency-window", type=int)
    attention.add_argument("--ready-queue-depth", type=int)
    attention.add_argument("--address-scoreboard", action="store_true")
    attention.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    attention.add_argument("--static-stage-offsets")
    attention.add_argument("--static-stage-ii", type=float, default=1.0)
    transformer_block = subparsers.add_parser(
        "transformer-block",
        help="compile and schedule a LayerNorm + attention + MLP decoder block skeleton",
    )
    transformer_block.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    transformer_block.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    transformer_block.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    transformer_block.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    transformer_block.add_argument("--output-dir", type=Path, default=Path("out/transformer-block"))
    transformer_block.add_argument("--instruction-queue-depth", type=int)
    transformer_block.add_argument("--rob-entries", type=int)
    transformer_block.add_argument("--max-inflight-tiles", type=int)
    transformer_block.add_argument("--dependency-window", type=int)
    transformer_block.add_argument("--ready-queue-depth", type=int)
    transformer_block.add_argument("--address-scoreboard", action="store_true")
    transformer_block.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    transformer_block.add_argument("--static-stage-offsets")
    transformer_block.add_argument("--static-stage-ii", type=float, default=1.0)
    model_block = subparsers.add_parser(
        "model-block",
        help="compile and schedule a named model's one-block proxy benchmark",
    )
    model_block.add_argument("--model-preset", choices=available_model_presets(), required=True)
    model_block.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    model_block.add_argument("--tokens", type=int)
    model_block.add_argument("--sequence", type=int)
    model_block.add_argument("--head-dim", type=int)
    model_block.add_argument("--intermediate", type=int)
    model_block.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    model_block.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    model_block.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    model_block.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    model_block.add_argument("--output-dir", type=Path, default=Path("out/model-block"))
    model_block.add_argument("--instruction-queue-depth", type=int)
    model_block.add_argument("--rob-entries", type=int)
    model_block.add_argument("--max-inflight-tiles", type=int)
    model_block.add_argument("--dependency-window", type=int)
    model_block.add_argument("--ready-queue-depth", type=int)
    model_block.add_argument("--address-scoreboard", action="store_true")
    model_block.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    model_block.add_argument("--static-stage-offsets")
    model_block.add_argument("--static-stage-ii", type=float, default=1.0)
    layernorm = subparsers.add_parser("layernorm", help="compile and schedule a row-LayerNorm benchmark")
    layernorm.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    layernorm.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    layernorm.add_argument("--timing-config", type=Path, help="load primitive timing overrides from JSON")
    layernorm.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    layernorm.add_argument("--output-dir", type=Path, default=Path("out/layernorm"))
    layernorm.add_argument("--instruction-queue-depth", type=int)
    layernorm.add_argument("--rob-entries", type=int)
    layernorm.add_argument("--max-inflight-tiles", type=int)
    layernorm.add_argument("--dependency-window", type=int)
    layernorm.add_argument("--ready-queue-depth", type=int)
    layernorm.add_argument("--address-scoreboard", action="store_true")
    layernorm.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    layernorm.add_argument("--static-stage-offsets")
    layernorm.add_argument("--static-stage-ii", type=float, default=1.0)
    sweep = subparsers.add_parser("sweep-two-mm", help="sweep 2mm architecture and scheduler parameters")
    sweep.add_argument("--architectures", default="minimal,wide-mxu")
    sweep.add_argument("--machine-config", type=Path, help="use one canonical MachineConfig JSON for every case")
    sweep.add_argument("--timing-config", type=Path, help="use one timing table JSON for every case")
    sweep.add_argument(
        "--policies",
        default=",".join(policy.value for policy in SchedulerPolicy),
    )
    sweep.add_argument("--windows", default="4,8")
    sweep.add_argument("--robs", default="4,8")
    sweep.add_argument("--tile-sizes", default="32")
    sweep.add_argument("--output-dir", type=Path, default=Path("out/sweep-two-mm"))
    sweep.add_argument("--address-scoreboard", action="store_true")
    sweep.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    sweep.add_argument("--static-stage-offsets")
    sweep.add_argument("--static-stage-ii", type=float, default=1.0)
    workload_sweep = subparsers.add_parser(
        "sweep-workloads",
        help="sweep multiple workloads, architectures and scheduler capacities",
    )
    workload_sweep.add_argument(
        "--workloads",
        default="two-mm,elementwise,reduce,softmax,rmsnorm,layernorm,decoder-block,attention,transformer-block",
    )
    workload_sweep.add_argument("--architectures", default="minimal,wide-mxu")
    workload_sweep.add_argument("--machine-config", type=Path, help="use one canonical MachineConfig JSON for every case")
    workload_sweep.add_argument("--timing-config", type=Path, help="use one timing table JSON for every case")
    workload_sweep.add_argument(
        "--policies",
        default=",".join(policy.value for policy in SchedulerPolicy),
    )
    workload_sweep.add_argument("--windows", default="4,8")
    workload_sweep.add_argument("--robs", default="4,8")
    workload_sweep.add_argument("--tile-sizes", default="32")
    workload_sweep.add_argument("--model-tokens", type=int, help="override token rows for named model presets")
    workload_sweep.add_argument("--model-sequence", type=int, help="override sequence columns for named model presets")
    workload_sweep.add_argument("--model-head-dim", type=int, help="override attention head dimension for named model presets")
    workload_sweep.add_argument("--model-intermediate", type=int, help="override MLP intermediate dimension for named model presets")
    workload_sweep.add_argument("--output-dir", type=Path, default=Path("out/sweep-workloads"))
    workload_sweep.add_argument("--address-scoreboard", action="store_true")
    workload_sweep.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    workload_sweep.add_argument("--dynamic-priorities", default="critical_path,oldest_first")
    workload_sweep.add_argument("--static-stage-offsets")
    workload_sweep.add_argument("--static-stage-ii", type=float, default=1.0)
    return parser


def _parse_shape_overrides(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--shape must use SYMBOL=VALUE")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        try:
            value = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"--shape value for '{name}' must be an integer") from exc
        if not name or value <= 0:
            raise ValueError("--shape symbols and values must be positive")
        if name in result and result[name] != value:
            raise ValueError(f"--shape symbol '{name}' is specified more than once")
        result[name] = value
    return result


def run_compile_model(args: argparse.Namespace) -> int:
    shape_environment = _parse_shape_overrides(args.shape)
    machine = _machine(args.arch, args.machine_config)
    codegen_backend = _codegen_backend(args.codegen_backend)
    if args.runtime_device_matrix and args.scheduler_target != "tisa":
        raise ValueError("--runtime-device-matrix requires --scheduler-target tisa")
    if args.through_stablehlo and args.torch_module is None:
        raise ValueError("--through-stablehlo is only valid with --torch-module")
    if args.stablehlo_exporter != "project" and not args.through_stablehlo:
        raise ValueError("--stablehlo-exporter is only valid with --through-stablehlo")
    if args.graph_json is not None:
        try:
            payload = json.loads(args.graph_json.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"graph JSON does not exist: {args.graph_json}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid graph JSON '{args.graph_json}': {exc}") from exc
        imported = JsonGraphAdapter.from_payload(
            payload,
            model_id=args.model_id,
            variant=args.variant,
            shape_environment=shape_environment,
        )
        compiled = compile_frontend_import(
            imported,
            machine,
            tile_size=args.tile_size,
            codegen_backend=codegen_backend,
        )
    elif args.stablehlo_file is not None:
        compiled = compile_stablehlo_file(
            args.stablehlo_file,
            machine,
            model_id=args.model_id or args.stablehlo_file.stem,
            variant=args.variant,
            shape_environment=shape_environment,
            tile_size=args.tile_size,
            stablehlo_backend=args.stablehlo_backend,
            codegen_backend=codegen_backend,
        )
        imported = compiled.frontend
    else:
        if not args.input_shape:
            raise ValueError("--torch-module requires at least one --input-shape")
        if ":" not in args.torch_module:
            raise ValueError("--torch-module must use MODULE:FACTORY syntax")
        module_name, factory_name = args.torch_module.split(":", 1)
        try:
            python_module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ValueError(f"cannot import PyTorch module '{module_name}': {exc}") from exc
        factory: object = python_module
        try:
            for attribute in factory_name.split("."):
                factory = getattr(factory, attribute)
        except AttributeError as exc:
            raise ValueError(
                f"PyTorch module factory '{args.torch_module}' does not exist"
            ) from exc
        if not callable(factory):
            raise ValueError(f"PyTorch module factory '{args.torch_module}' is not callable")
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ValueError("--torch-module requires PyTorch") from exc
        try:
            module = factory()
        except Exception as exc:
            raise ValueError(f"PyTorch module factory '{args.torch_module}' failed: {exc}") from exc
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"PyTorch module factory '{args.torch_module}' did not return nn.Module")
        dtype = getattr(torch, args.input_dtype)
        torch.manual_seed(0)
        input_shapes = tuple(
            _parse_positive_int_list(value, name="--input-shape")
            for value in args.input_shape
        )
        example_inputs = tuple(torch.randn(*shape, dtype=dtype) for shape in input_shapes)
        compile_args = {
            "model_id": args.model_id or factory_name.rsplit(".", 1)[-1],
            "variant": args.variant,
            "shape_environment": shape_environment,
            "tile_size": args.tile_size,
            "codegen_backend": codegen_backend,
        }
        if args.through_stablehlo:
            compiled = compile_torch_module_through_stablehlo(
                module.eval(),
                example_inputs,
                machine,
                stablehlo_backend=args.stablehlo_backend,
                stablehlo_exporter=args.stablehlo_exporter,
                **compile_args,
            )
        else:
            compiled = compile_torch_module(
                module.eval(), example_inputs, machine, **compile_args
            )
        imported = compiled.frontend
    ensure_output_layout(args.output_dir)
    write_artifact_json(imported, args.output_dir / "frontend_import.json")
    if compiled.source_frontend is not None:
        write_artifact_json(compiled.source_frontend, args.output_dir / "source_frontend_import.json")
    if compiled.stablehlo is not None:
        write_artifact_json(compiled.stablehlo, args.output_dir / "stablehlo_module.json")
        stablehlo_target = args.output_dir / "00_frontend" / "generated.mlir"
        stablehlo_target.write_text(compiled.stablehlo.text, encoding="utf-8")
        write_artifact_index(args.output_dir)
    write_artifact_json(compiled.graph, args.output_dir / "canonical_graph.json")
    write_artifact_json(compiled.schedule, args.output_dir / "schedule.json")
    write_artifact_json(compiled.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(compiled.tisa_program, args.output_dir / "tisa_program.json")
    write_artifact_json(compiled.backend_artifact, args.output_dir / "backend_artifact.json")
    write_artifact_json(compiled.backend_artifact.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(compiled, args.output_dir / "compiled_artifact.json")
    runtime_lifetimes = derive_tensor_lifetimes(compiled.tisa_program)
    runtime_reuse_pairs = derive_tensor_reuse_pairs(compiled.tisa_program)
    runtime_buffers = allocate_buffer_bindings(
        compiled.graph.tensors,
        base_address=args.runtime_base_address,
        alignment_bytes=args.runtime_alignment,
        lifetimes=runtime_lifetimes,
        reuse_buffers=args.runtime_buffer_policy == "lifetime_reuse",
        reuse_pairs=runtime_reuse_pairs,
    )
    runtime_descriptor_availability = _descriptor_availability(
        args.runtime_availability_config
    )
    runtime_allocation_span = (
        max(buffer.end_address for buffer in runtime_buffers)
        - min(buffer.base_address for buffer in runtime_buffers)
        if runtime_buffers
        else 0
    )
    runtime_submission = create_runtime_submission(
        compiled.backend_artifact,
        runtime_buffers,
        submission_id=f"submission.{compiled.tisa_program.program_id}",
        policy=args.runtime_policy,
        chunk_size=args.runtime_chunk_size,
        launch_latency_cycles=args.runtime_launch_latency,
        synchronization_cycles=args.runtime_synchronization_cycles,
        descriptor_available_cycles=runtime_descriptor_availability,
    )
    write_artifact_json(runtime_submission, args.output_dir / "runtime_submission.json")
    write_operator_graph_dot(compiled.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(compiled.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(compiled.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(compiled.backend_artifact.execution_graph, args.output_dir / "execution_graph.dot")

    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
    )
    timing_model = _timing_model(args.timing_config, args.timing_provider)
    event_backend = _event_backend(args.event_backend)
    if args.scheduler_target == "tisa":
        result = schedule_tisa_program(
            compiled.backend_artifact,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            runtime_submission=runtime_submission,
            event_backend=event_backend,
        )
    else:
        result = schedule_execution_graph(
            compiled.backend_artifact.execution_graph,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
        )
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    if result.instruction_timings:
        write_instruction_csv(result, args.output_dir / "tisa_instructions.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "compiler_pipeline": compiled.attributes["compiler_pipeline"],
            "frontend": imported.frontend.value if hasattr(imported.frontend, "value") else str(imported.frontend),
            "frontend_path": compiled.attributes.get("frontend_path", "direct_canonical"),
            "stablehlo_variant": compiled.attributes.get("stablehlo_variant"),
            "stablehlo_backend": compiled.attributes.get("stablehlo_backend"),
            "stablehlo_exporter": compiled.attributes.get("stablehlo_exporter"),
            "stablehlo_exporter_version": compiled.attributes.get("stablehlo_exporter_version"),
            "stablehlo_verified": compiled.attributes.get("stablehlo_verified", False),
            "stablehlo_producer": compiled.attributes.get("stablehlo_producer"),
            "stablehlo_verifier": compiled.attributes.get("stablehlo_verifier"),
            "stablehlo_version": compiled.attributes.get("stablehlo_version"),
            "stablehlo_fallback": compiled.attributes.get("stablehlo_fallback", False),
            "stablehlo_fallback_reason": compiled.attributes.get("stablehlo_fallback_reason"),
            "timing_provider": getattr(timing_model, "name", "analytical"),
            "timing_backend_capabilities": (
                timing_model.capabilities.to_dict()
                if hasattr(timing_model, "capabilities")
                else None
            ),
            "timing_provider_metadata": dict(
                getattr(timing_model, "metadata", {})
            ),
            "timing_provider_coverage": result.metrics.get(
                "timing_provider_coverage"
            ),
            "event_backend": (
                event_backend.name if args.scheduler_target == "tisa" else None
            ),
            "event_backend_capabilities": (
                event_backend.capabilities.to_dict()
                if args.scheduler_target == "tisa"
                else None
            ),
            "codegen_backend": compiled.attributes.get(
                "codegen_backend", compiled.backend_artifact.backend
            ),
            "codegen_backend_capabilities": compiled.attributes.get(
                "codegen_backend_capabilities"
            ),
            "runtime_backend": "builtin_runtime_submission",
            "model_id": imported.model_id,
            "architecture": args.arch,
            "policy": result.policy,
            "scheduler_target": result.metrics.get(
                "scheduler_target", args.scheduler_target
            ),
            "total_cycles": result.total_cycles,
            "tisa_instruction_count": len(compiled.tisa_program.instructions),
            "primitive_task_count": len(compiled.backend_artifact.execution_graph.tasks),
            "tisa_decision_count": result.metrics.get("tisa_decision_count"),
            "payload_execution": result.metrics.get("payload_execution"),
            "runtime_policy": runtime_submission.policy,
            "runtime_applied_to_device": args.scheduler_target == "tisa",
            "runtime_buffer_policy": args.runtime_buffer_policy,
            "runtime_command_chunk_count": len(runtime_submission.commands),
            "runtime_buffer_count": len(runtime_submission.buffers),
            "runtime_allocation_span_bytes": runtime_allocation_span,
            "runtime_submit_cycles": result.metrics.get("runtime_submit_cycles", 0.0),
            "runtime_submit_busy_cycles": result.metrics.get(
                "runtime_submit_busy_cycles", 0.0
            ),
            "runtime_request_wait_cycles": result.metrics.get(
                "runtime_request_wait_cycles", 0.0
            ),
            "runtime_descriptor_availability_count": len(
                runtime_descriptor_availability
            ),
            "runtime_synchronization_cycles": result.metrics.get(
                "runtime_synchronization_cycles", 0.0
            ),
            "device_start_cycle": result.metrics.get("device_start_cycle", 0.0),
            "device_finish_cycle": result.metrics.get(
                "device_finish_cycle", result.total_cycles
            ),
            "device_cycles": result.metrics.get("device_cycles", result.total_cycles),
            "total_cycles_including_runtime": result.metrics.get(
                "total_cycles_including_runtime", result.total_cycles
            ),
            "calibration_status": result.metrics["calibration_status"],
        },
        args.output_dir / "manifest.json",
    )
    if args.runtime_device_matrix:
        matrix_root = args.output_dir / "policy_matrix"
        matrix_root.mkdir(parents=True, exist_ok=True)
        matrix_cases = run_runtime_device_matrix(
            compiled.backend_artifact,
            runtime_buffers,
            machine,
            chunk_size=args.runtime_chunk_size,
            launch_latency_cycles=args.runtime_launch_latency,
            synchronization_cycles=args.runtime_synchronization_cycles,
            descriptor_available_cycles=runtime_descriptor_availability,
            timing_model=timing_model,
            simulator_config=simulator_config,
            event_backend=event_backend,
        )
        matrix_records: list[dict[str, object]] = []
        static_baseline = next(
            case.result.total_cycles
            for case in matrix_cases
            if case.runtime_policy == "static"
            and case.device_policy == SchedulerPolicy.STATIC_PIPELINE.value
        )
        for case in matrix_cases:
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
                    static_baseline / case.result.total_cycles
                    if case.result.total_cycles
                    else None
                ),
            }
            matrix_records.append(record)
            write_artifact_json(
                {
                    "schema_version": 1,
                    **record,
                    "shared_compiled_artifact": "../../03_tisa/compiled_artifact.json",
                    "shared_backend_artifact": "../../04_backend/backend_artifact.json",
                    "machine_hash": machine.stable_hash(),
                    "codegen_backend": compiled.attributes.get(
                        "codegen_backend", compiled.backend_artifact.backend
                    ),
                    "codegen_backend_capabilities": compiled.attributes.get(
                        "codegen_backend_capabilities"
                    ),
                    "runtime_backend": "builtin_runtime_submission",
                    "event_backend": event_backend.name,
                    "event_backend_capabilities": event_backend.capabilities.to_dict(),
                    "timing_provider": getattr(timing_model, "name", "analytical"),
                    "timing_provider_metadata": dict(
                        getattr(timing_model, "metadata", {})
                    ),
                    "timing_provider_coverage": case.result.metrics.get(
                        "timing_provider_coverage"
                    ),
                    "calibration_status": case.result.metrics["calibration_status"],
                },
                case_dir / "manifest.json",
            )
        matrix_records.sort(
            key=lambda item: (str(item["runtime_policy"]), str(item["device_policy"]))
        )
        with (matrix_root / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(matrix_records[0]))
            writer.writeheader()
            writer.writerows(matrix_records)
        write_artifact_json(matrix_records, matrix_root / "sweep.json")
        (matrix_root / "README.md").write_text(
            "# Runtime x Device policy matrix\n\n"
            "四个 case 共享父目录中的 compiled/backend artifact 和 physical buffer allocation。\n"
            "每个 case 只保存独立的 RuntimeSubmission、simulation result 和 trace。\n",
            encoding="utf-8",
        )
        write_artifact_index(args.output_dir)
    print(json.dumps({
        "frontend": imported.frontend.value if hasattr(imported.frontend, "value") else str(imported.frontend),
        "tisa_instructions": len(compiled.tisa_program.instructions),
        "primitive_tasks": len(compiled.backend_artifact.execution_graph.tasks),
        "scheduler_target": result.metrics.get("scheduler_target", args.scheduler_target),
        "total_cycles": result.total_cycles,
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


def _simulator_config(
    *,
    dependency_window: int | None,
    rob_entries: int | None,
    address_scoreboard: bool,
    static_stage_offsets: tuple[float, ...] = (),
    static_stage_ii: float = 1.0,
    dynamic_priority: str = "critical_path",
) -> SimulatorConfig:
    return SimulatorConfig(
        dependency_window=dependency_window,
        rob_entries=rob_entries,
        address_scoreboard=address_scoreboard,
        dynamic_priority=dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(static_stage_offsets),
                stage_offsets=static_stage_offsets,
                initiation_interval_cycles=static_stage_ii,
            )
            if static_stage_offsets
            else None
        ),
    )


def run_two_mm(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_two_matmul_model()
    case = build_two_matmul_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_two_matmul_schedule(instance.graph)
    lowered = lower_two_matmul(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    effective_execution_graph = lowered.execution_graph
    result = schedule_execution_graph(
        effective_execution_graph,
        machine,
        args.policy,
        timing_model=_timing_model(args.timing_config),
        simulator_config=simulator_config,
    )
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(effective_execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(
        result.metrics.get("address_hazards", []),
        args.output_dir / "address_dependencies.json",
    )
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(
        json.dumps(
            {
                "architecture": args.arch,
                "policy": result.policy,
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def run_elementwise(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_elementwise_model()
    case = build_elementwise_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_elementwise_schedule(instance.graph)
    lowered = lower_elementwise(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(
        lowered.execution_graph,
        machine,
        args.policy,
        timing_model=_timing_model(args.timing_config),
        simulator_config=simulator_config,
    )
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_reduce(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_reduce_model()
    case = build_reduce_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_reduce_schedule(instance.graph)
    lowered = lower_reduce(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, timing_model=_timing_model(args.timing_config), simulator_config=simulator_config)
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_softmax(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_softmax_model()
    case = build_softmax_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_softmax_schedule(instance.graph)
    lowered = lower_softmax(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, timing_model=_timing_model(args.timing_config), simulator_config=simulator_config)
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_rmsnorm(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_rmsnorm_model()
    case = build_rmsnorm_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_rmsnorm_schedule(instance.graph)
    lowered = lower_rmsnorm(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, timing_model=_timing_model(args.timing_config), simulator_config=simulator_config)
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_decoder_block(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_decoder_block_model()
    case = build_decoder_block_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_mixed_schedule(instance.graph)
    lowered = lower_mixed_model(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(
        lowered.execution_graph,
        machine,
        args.policy,
        timing_model=_timing_model(args.timing_config),
        simulator_config=simulator_config,
    )
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(
        json.dumps(
            {
                "architecture": args.arch,
                "policy": result.policy,
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def run_attention(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_attention_model()
    case = build_attention_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_mixed_schedule(instance.graph)
    lowered = lower_mixed_model(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, timing_model=_timing_model(args.timing_config), simulator_config=simulator_config)
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(
        json.dumps(
            {
                "architecture": args.arch,
                "policy": result.policy,
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def run_transformer_block(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    if getattr(args, "model_preset", None):
        model, case = build_model_preset(
            args.model_preset,
            architecture_profile=args.arch,
            scheduler_profile=args.policy,
            tokens=args.tokens,
            sequence=args.sequence,
            head_dim=args.head_dim,
            intermediate=args.intermediate,
            phase=args.phase,
        )
    else:
        model = build_transformer_block_model()
        case = build_transformer_block_case(
            architecture_profile=args.arch,
            scheduler_profile=args.policy,
        )
    instance = model.instantiate(case)
    schedule = default_mixed_schedule(instance.graph)
    lowered = lower_mixed_model(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, timing_model=_timing_model(args.timing_config), simulator_config=simulator_config)
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(
        json.dumps(
            {
                "architecture": args.arch,
                "policy": result.policy,
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def run_layernorm(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_layernorm_model()
    case = build_layernorm_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_layernorm_schedule(instance.graph)
    lowered = lower_layernorm(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, timing_model=_timing_model(args.timing_config), simulator_config=simulator_config)
    ensure_output_layout(args.output_dir)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    print(
        json.dumps(
            {
                "architecture": args.arch,
                "policy": result.policy,
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def run_sweep_two_mm(args: argparse.Namespace) -> int:
    architectures = _parse_list(args.architectures, name="--architectures")
    policies = _parse_list(args.policies, name="--policies")
    supported_architectures = {"minimal", "wide-mxu", "lpu-like"}
    unknown_architectures = sorted(set(architectures) - supported_architectures)
    if unknown_architectures and args.machine_config is None:
        raise ValueError(f"unknown architecture profile(s): {', '.join(unknown_architectures)}")
    supported_policies = {policy.value for policy in SchedulerPolicy}
    unknown_policies = sorted(set(policies) - supported_policies)
    if unknown_policies:
        raise ValueError(f"unsupported scheduler policy(s): {', '.join(unknown_policies)}")
    windows = _parse_positive_int_list(args.windows, name="--windows")
    robs = _parse_positive_int_list(args.robs, name="--robs")
    tile_sizes = _parse_positive_int_list(args.tile_sizes, name="--tile-sizes")
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    timing_model = _timing_model(args.timing_config)
    ensure_output_layout(args.output_dir)

    model = build_two_matmul_model()
    records: list[dict[str, object]] = []
    results: dict[tuple[str, str, int, int, int], object] = {}
    lowered_by_arch: dict[tuple[str, int], tuple[object, object, object]] = {}
    for architecture, policy, dependency_window, rob_entries, tile_size in product(architectures, policies, windows, robs, tile_sizes):
        machine = _machine(architecture, args.machine_config)
        lowering_key = (architecture, tile_size)
        if lowering_key not in lowered_by_arch:
            case = build_two_matmul_case(
                architecture_profile=architecture,
                scheduler_profile=SchedulerPolicy.STATIC_PIPELINE.value,
            )
            instance = model.instantiate(case)
            schedule = default_two_matmul_schedule(instance.graph, tile_size=tile_size)
            lowered_by_arch[lowering_key] = (instance, schedule, lower_two_matmul(instance, machine, schedule))
        instance, schedule, lowered = lowered_by_arch[lowering_key]
        case = build_two_matmul_case(
            architecture_profile=architecture,
            scheduler_profile=policy,
        )
        static_config = stage_offsets if policy == SchedulerPolicy.STATIC_PIPELINE.value else ()
        simulator_config = _simulator_config(
            dependency_window=dependency_window,
            rob_entries=rob_entries,
            address_scoreboard=args.address_scoreboard,
            static_stage_offsets=static_config,
            static_stage_ii=args.static_stage_ii,
            dynamic_priority=args.dynamic_priority,
        )
        result = schedule_execution_graph(
            lowered.execution_graph,
            machine,
            policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
        )
        key = (architecture, policy, dependency_window, rob_entries, tile_size)
        results[key] = result
        case_id = f"{architecture}__{policy}__tile{tile_size}__window{dependency_window}__rob{rob_entries}"
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        ensure_output_layout(case_dir)
        write_artifact_json(model, case_dir / "model_spec.json")
        write_artifact_json(case, case_dir / "benchmark_case.json")
        write_artifact_json(instance, case_dir / "model_instance.json")
        write_artifact_json(instance.graph, case_dir / "operator_graph.json")
        write_artifact_json(schedule, case_dir / "schedule.json")
        write_artifact_json(lowered.tile_graph, case_dir / "tile_graph.json")
        write_artifact_json(lowered.execution_graph, case_dir / "execution_graph.json")
        write_artifact_json(machine, case_dir / "machine.json")
        write_operator_graph_dot(instance.graph, case_dir / "operator_graph.dot")
        write_operator_graph_svg(instance.graph, case_dir / "operator_graph.svg")
        write_tile_graph_dot(lowered.tile_graph, case_dir / "tile_graph.dot")
        write_execution_graph_dot(lowered.execution_graph, case_dir / "execution_graph.dot")
        write_json(result, case_dir / "summary.json")
        write_csv(result, case_dir / "tasks.csv")
        write_svg(result, case_dir / "swimlane.svg")
        write_png(result, case_dir / "swimlane.png")
        write_artifact_json(result.perfetto_trace(), case_dir / "perfetto.json")
        write_artifact_json(result.metrics.get("address_hazards", []), case_dir / "address_dependencies.json")
        write_artifact_json(
            {
                "schema_version": 1,
                "case_id": case_id,
                "benchmark": case.case_id,
                "architecture": architecture,
                "policy": policy,
                "machine_hash": machine.stable_hash(),
                "backend": result.backend,
                "calibration_status": result.metrics["calibration_status"],
                "simulator_config": result.metrics["simulator_config"],
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
            },
            case_dir / "manifest.json",
        )

    static_cycles = {
        (architecture, dependency_window, rob_entries, tile_size): results[(architecture, SchedulerPolicy.STATIC_PIPELINE.value, dependency_window, rob_entries, tile_size)].total_cycles
        for architecture, dependency_window, rob_entries, tile_size in product(architectures, windows, robs, tile_sizes)
        if (architecture, SchedulerPolicy.STATIC_PIPELINE.value, dependency_window, rob_entries, tile_size) in results
    }
    for (architecture, policy, dependency_window, rob_entries, tile_size), result in results.items():
        baseline = static_cycles.get((architecture, dependency_window, rob_entries, tile_size))
        metrics = result.metrics
        records.append(
            {
                "case_id": f"{architecture}__{policy}__tile{tile_size}__window{dependency_window}__rob{rob_entries}",
                "architecture": architecture,
                "policy": policy,
                "tile_size": tile_size,
                "dependency_window": dependency_window,
                "rob_entries": rob_entries,
                "total_cycles": result.total_cycles,
                "speedup_vs_static": (baseline / result.total_cycles if baseline is not None and result.total_cycles else None),
                "address_scoreboard": args.address_scoreboard,
                "dynamic_priority": args.dynamic_priority,
                "address_hazard_count": metrics.get("address_hazard_count", 0),
                "rob_peak": metrics.get("rob_peak", 0),
                "ready_set_peak": metrics.get("ready_set_peak", 0),
                "queue_wait_cycles": metrics.get("queue_wait_cycles", 0.0),
                "resource_utilization": json.dumps(metrics.get("resource_utilization", {}), sort_keys=True),
                "stall_by_reason": json.dumps(metrics.get("stall_by_reason", {}), sort_keys=True),
                "pipeline_drain_cycles": metrics.get("pipeline_drain_cycles", 0.0),
                "completed_tile_count": metrics.get("completed_tile_count", 0),
            }
        )
    records.sort(key=lambda record: (str(record["architecture"]), int(record["tile_size"]), str(record["policy"]), int(record["dependency_window"]), int(record["rob_entries"])))
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_artifact_json(records, args.output_dir / "sweep.json")
    print(json.dumps({"case_count": len(records), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def _named_model_preset_builders(
    name: str,
    *,
    tokens: int | None = None,
    sequence: int | None = None,
    head_dim: int | None = None,
    intermediate: int | None = None,
):
    def model_builder():
        return build_model_preset(
            name,
            tokens=tokens,
            sequence=sequence,
            head_dim=head_dim,
            intermediate=intermediate,
        )[0]

    def case_builder(*, architecture_profile: str = "minimal", scheduler_profile: str = "sequential"):
        return build_model_preset(
            name,
            architecture_profile=architecture_profile,
            scheduler_profile=scheduler_profile,
            tokens=tokens,
            sequence=sequence,
            head_dim=head_dim,
            intermediate=intermediate,
        )[1]

    return model_builder, case_builder


def _workload_builders(
    *,
    model_tokens: int | None = None,
    model_sequence: int | None = None,
    model_head_dim: int | None = None,
    model_intermediate: int | None = None,
):
    builders = {
        "two-mm": (build_two_matmul_model, build_two_matmul_case),
        "elementwise": (build_elementwise_model, build_elementwise_case),
        "reduce": (build_reduce_model, build_reduce_case),
        "softmax": (build_softmax_model, build_softmax_case),
        "rmsnorm": (build_rmsnorm_model, build_rmsnorm_case),
        "layernorm": (build_layernorm_model, build_layernorm_case),
        "decoder-block": (build_decoder_block_model, build_decoder_block_case),
        "attention": (build_attention_model, build_attention_case),
        "transformer-block": (build_transformer_block_model, build_transformer_block_case),
    }
    builders.update(
        {
            name: _named_model_preset_builders(
                name,
                tokens=model_tokens,
                sequence=model_sequence,
                head_dim=model_head_dim,
                intermediate=model_intermediate,
            )
            for name in available_model_presets()
        }
    )
    return builders


def run_sweep_workloads(args: argparse.Namespace) -> int:
    workloads = _parse_list(args.workloads, name="--workloads")
    architectures = _parse_list(args.architectures, name="--architectures")
    policies = _parse_list(args.policies, name="--policies")
    builders = _workload_builders(
        model_tokens=args.model_tokens,
        model_sequence=args.model_sequence,
        model_head_dim=args.model_head_dim,
        model_intermediate=args.model_intermediate,
    )
    unknown_workloads = sorted(set(workloads) - set(builders))
    if unknown_workloads:
        raise ValueError(f"unknown workload(s): {', '.join(unknown_workloads)}")
    supported_architectures = {"minimal", "wide-mxu", "lpu-like"}
    unknown_architectures = sorted(set(architectures) - supported_architectures)
    if unknown_architectures and args.machine_config is None:
        raise ValueError(f"unknown architecture profile(s): {', '.join(unknown_architectures)}")
    supported_policies = {policy.value for policy in SchedulerPolicy}
    unknown_policies = sorted(set(policies) - supported_policies)
    if unknown_policies:
        raise ValueError(f"unsupported scheduler policy(s): {', '.join(unknown_policies)}")
    windows = _parse_positive_int_list(args.windows, name="--windows")
    robs = _parse_positive_int_list(args.robs, name="--robs")
    tile_sizes = _parse_positive_int_list(args.tile_sizes, name="--tile-sizes")
    dynamic_priorities = _parse_list(args.dynamic_priorities, name="--dynamic-priorities")
    supported_priorities = {"critical_path", "oldest_first"}
    unknown_priorities = sorted(set(dynamic_priorities) - supported_priorities)
    if unknown_priorities:
        raise ValueError(f"unsupported dynamic priority(s): {', '.join(unknown_priorities)}")
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    timing_model = _timing_model(args.timing_config)
    ensure_output_layout(args.output_dir)

    lowered_cache: dict[tuple[str, str, int], tuple[object, object, object, object]] = {}
    results: dict[tuple[str, str, str, int, int, int, str], object] = {}
    records: list[dict[str, object]] = []
    for workload, architecture, policy, dependency_window, rob_entries, tile_size, dynamic_priority in product(
        workloads, architectures, policies, windows, robs, tile_sizes, dynamic_priorities
    ):
        cache_key = (workload, architecture, tile_size)
        machine = _machine(architecture, args.machine_config)
        if cache_key not in lowered_cache:
            model_builder, _case_builder = builders[workload]
            model = model_builder()
            compile_case = builders[workload][1](
                architecture_profile=architecture,
                scheduler_profile=SchedulerPolicy.STATIC_PIPELINE.value,
            )
            instance = model.instantiate(compile_case)
            schedule = default_mixed_schedule(instance.graph, tile_size=tile_size)
            lowered = lower_mixed_model(instance, machine, schedule)
            lowered_cache[cache_key] = (model, instance, schedule, lowered)
        model, instance, schedule, lowered = lowered_cache[cache_key]
        case = builders[workload][1](
            architecture_profile=architecture,
            scheduler_profile=policy,
        )
        static_config = stage_offsets if policy == SchedulerPolicy.STATIC_PIPELINE.value else ()
        simulator_config = _simulator_config(
            dependency_window=dependency_window,
            rob_entries=rob_entries,
            address_scoreboard=args.address_scoreboard,
            static_stage_offsets=static_config,
            static_stage_ii=args.static_stage_ii,
            dynamic_priority=dynamic_priority,
        )
        result = schedule_execution_graph(
            lowered.execution_graph,
            machine,
            policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
        )
        key = (workload, architecture, policy, dependency_window, rob_entries, tile_size, dynamic_priority)
        results[key] = result
        case_id = (
            f"{workload}__{architecture}__{policy}__tile{tile_size}"
            f"__window{dependency_window}__rob{rob_entries}__priority{dynamic_priority}"
        )
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        write_artifact_json(model, case_dir / "model_spec.json")
        write_artifact_json(case, case_dir / "benchmark_case.json")
        write_artifact_json(instance, case_dir / "model_instance.json")
        write_artifact_json(instance.graph, case_dir / "operator_graph.json")
        write_artifact_json(schedule, case_dir / "schedule.json")
        write_artifact_json(lowered.tile_graph, case_dir / "tile_graph.json")
        write_artifact_json(lowered.execution_graph, case_dir / "execution_graph.json")
        write_artifact_json(machine, case_dir / "machine.json")
        write_json(result, case_dir / "summary.json")
        write_csv(result, case_dir / "tasks.csv")
        write_svg(result, case_dir / "swimlane.svg")
        write_png(result, case_dir / "swimlane.png")
        write_artifact_json(result.perfetto_trace(), case_dir / "perfetto.json")
        write_artifact_json(result.metrics.get("address_hazards", []), case_dir / "address_dependencies.json")
        write_artifact_json(
            {
                "schema_version": 1,
                "case_id": case_id,
                "benchmark": case.case_id,
                "workload": workload,
                "architecture": architecture,
                "policy": policy,
                "dynamic_priority": dynamic_priority,
                "machine_hash": machine.stable_hash(),
                "backend": result.backend,
                "calibration_status": result.metrics["calibration_status"],
                "simulator_config": result.metrics["simulator_config"],
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
            },
            case_dir / "manifest.json",
        )

    static_cycles = {
        (workload, architecture, dependency_window, rob_entries, tile_size, dynamic_priority): results[
            (
                workload,
                architecture,
                SchedulerPolicy.STATIC_PIPELINE.value,
                dependency_window,
                rob_entries,
                tile_size,
                dynamic_priority,
            )
        ].total_cycles
        for workload, architecture, dependency_window, rob_entries, tile_size, dynamic_priority in product(
            workloads, architectures, windows, robs, tile_sizes, dynamic_priorities
        )
        if (
            workload,
            architecture,
            SchedulerPolicy.STATIC_PIPELINE.value,
            dependency_window,
            rob_entries,
            tile_size,
            dynamic_priority,
        ) in results
    }
    for (workload, architecture, policy, dependency_window, rob_entries, tile_size, dynamic_priority), result in results.items():
        baseline = static_cycles.get((workload, architecture, dependency_window, rob_entries, tile_size, dynamic_priority))
        metrics = result.metrics
        records.append(
            {
                "workload": workload,
                "architecture": architecture,
                "policy": policy,
                "tile_size": tile_size,
                "dependency_window": dependency_window,
                "rob_entries": rob_entries,
                "total_cycles": result.total_cycles,
                "speedup_vs_static": (
                    baseline / result.total_cycles
                    if baseline is not None and result.total_cycles
                    else None
                ),
                "address_scoreboard": args.address_scoreboard,
                "dynamic_priority": dynamic_priority,
                "address_hazard_count": metrics.get("address_hazard_count", 0),
                "rob_peak": metrics.get("rob_peak", 0),
                "ready_set_peak": metrics.get("ready_set_peak", 0),
                "queue_wait_cycles": metrics.get("queue_wait_cycles", 0.0),
                "stall_by_reason": json.dumps(metrics.get("stall_by_reason", {}), sort_keys=True),
                "pipeline_drain_cycles": metrics.get("pipeline_drain_cycles", 0.0),
                "completed_tile_count": metrics.get("completed_tile_count", 0),
            }
        )
    records.sort(
        key=lambda record: (
            str(record["workload"]),
            str(record["architecture"]),
            int(record["tile_size"]),
            str(record["policy"]),
            int(record["dependency_window"]),
            int(record["rob_entries"]),
            str(record["dynamic_priority"]),
        )
    )
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_artifact_json(records, args.output_dir / "sweep.json")
    print(json.dumps({"case_count": len(records), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import-rtl-log":
        return run_import_rtl_log(args)
    if args.command == "import-rtl-trace":
        return run_import_rtl_trace(args)
    if args.command == "compile-model":
        return run_compile_model(args)
    if args.command == "two-mm":
        return run_two_mm(args)
    if args.command == "elementwise":
        return run_elementwise(args)
    if args.command == "reduce":
        return run_reduce(args)
    if args.command == "softmax":
        return run_softmax(args)
    if args.command == "rmsnorm":
        return run_rmsnorm(args)
    if args.command == "decoder-block":
        return run_decoder_block(args)
    if args.command == "attention":
        return run_attention(args)
    if args.command == "transformer-block":
        return run_transformer_block(args)
    if args.command == "model-block":
        return run_transformer_block(args)
    if args.command == "layernorm":
        return run_layernorm(args)
    if args.command == "sweep-two-mm":
        return run_sweep_two_mm(args)
    if args.command == "sweep-workloads":
        return run_sweep_workloads(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
