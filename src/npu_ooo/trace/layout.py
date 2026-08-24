"""Structured output layout for compiler and simulator artifacts.

Numbered stage directories are the only locations for stage artifacts.  The
layout layer also removes flattened files and compatibility symlinks produced
by older versions when an output directory is reused.
"""

from __future__ import annotations

import json
from pathlib import Path


STAGE_DIRECTORIES: tuple[tuple[str, str], ...] = (
    ("00_frontend", "前端导入与模型输入"),
    ("01_graph_ir", "规范化算子图与图可视化"),
    ("02_schedule_tile", "调度规划与 tile 实例"),
    ("03_tisa", "TISA 语义指令与编译产物"),
    ("04_backend", "后端执行图、机器配置与 payload"),
    ("05_runtime", "runtime 地址绑定与依赖观察"),
    ("06_simulation", "离散事件仿真、周期与任务结果"),
    ("07_trace", "泳道图与 Perfetto trace"),
)

_STAGE_BY_FILENAME: dict[str, str] = {
    "frontend_import.json": "00_frontend",
    "source_frontend_import.json": "00_frontend",
    "stablehlo_module.json": "00_frontend",
    "generated.mlir": "00_frontend",
    "model_spec.json": "00_frontend",
    "benchmark_case.json": "00_frontend",
    "model_instance.json": "00_frontend",
    "canonical_graph.json": "01_graph_ir",
    "operator_graph.json": "01_graph_ir",
    "operator_graph.dot": "01_graph_ir",
    "operator_graph.svg": "01_graph_ir",
    "schedule.json": "02_schedule_tile",
    "tile_graph.json": "02_schedule_tile",
    "tile_graph.dot": "02_schedule_tile",
    "tisa_program.json": "03_tisa",
    "compiled_artifact.json": "03_tisa",
    "backend_artifact.json": "04_backend",
    "machine.json": "04_backend",
    "execution_graph.json": "04_backend",
    "execution_graph.dot": "04_backend",
    "address_dependencies.json": "05_runtime",
    "summary.json": "06_simulation",
    "tasks.csv": "06_simulation",
    "tisa_instructions.csv": "06_simulation",
    "perfetto.json": "07_trace",
    "swimlane.svg": "07_trace",
    "swimlane.png": "07_trace",
}


def _root_readme() -> str:
    rows = [
        "# 本次运行输出",
        "",
        "规范 artifact 按编译和仿真阶段分目录保存。单次实验顶层只保留",
        "`README.md`、`artifact_index.json` 和 `manifest.json`；阶段 artifact 不创建",
        "顶层副本或兼容性符号链接。批量实验根目录还会保留 `sweep.csv/json`。",
        "",
        "| 目录 | 内容 |",
        "| --- | --- |",
    ]
    rows.extend(f"| `{name}/` | {description} |" for name, description in STAGE_DIRECTORIES)
    rows.extend(
        [
            "",
            "典型查看顺序：先看 `00_frontend` 确认输入，再看 `01_graph_ir` 和",
            "`02_schedule_tile` 检查图与切分，接着看 `03_tisa`/`04_backend`，最后在",
            "`06_simulation`、`07_trace` 中比较周期和泳道。",
            "启用 StableHLO 路径时，可读程序是 `00_frontend/generated.mlir`；",
            "`stablehlo_module.json` 保存程序文本、producer、版本、验证状态和 provenance。",
            "旧的 primitive baseline 命令尚未经过 TISA codegen 时，`03_tisa/` 可能为空；",
            "使用 `compile-model` 或后续 TISA target pipeline 时该目录会出现 descriptor。",
        ]
    )
    return "\n".join(rows) + "\n"


def _remove_legacy_flat_artifacts(root: Path) -> None:
    """Remove known root-level artifacts created by pre-staged layouts."""

    for filename in _STAGE_BY_FILENAME:
        candidate = root / filename
        if candidate.is_symlink() or candidate.is_file():
            candidate.unlink()


def ensure_output_layout(root: str | Path) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    _remove_legacy_flat_artifacts(root_path)
    for directory, _description in STAGE_DIRECTORIES:
        (root_path / directory).mkdir(parents=True, exist_ok=True)
    readme = root_path / "README.md"
    readme.write_text(_root_readme(), encoding="utf-8")
    return root_path


def _is_stage_directory(path: Path) -> bool:
    return path.parent.name in {name for name, _description in STAGE_DIRECTORIES}


def artifact_path(path: str | Path) -> tuple[Path, Path | None]:
    """Return ``(canonical_path, legacy_flat_path)`` for an artifact.

    Unknown files such as sweep summaries remain at their requested location.
    A path already inside a stage directory is never rewritten.
    """

    requested = Path(path)
    stage = _STAGE_BY_FILENAME.get(requested.name)
    if stage is None or _is_stage_directory(requested):
        return requested, None
    root = ensure_output_layout(requested.parent)
    canonical = root / stage / requested.name
    return canonical, requested


def finalize_artifact(canonical: Path, compatibility: Path | None) -> None:
    """Remove any legacy flat artifact and refresh the root index."""

    root = compatibility.parent if compatibility is not None else canonical.parent
    if canonical.parent.name in {name for name, _description in STAGE_DIRECTORIES}:
        root = canonical.parent.parent
    if compatibility is not None and compatibility != canonical:
        if compatibility.is_symlink() or compatibility.is_file():
            compatibility.unlink()
    if any((root / name).is_dir() for name, _description in STAGE_DIRECTORIES):
        write_artifact_index(root)


def write_artifact_index(root: str | Path) -> None:
    root_path = Path(root)
    entries: dict[str, list[str]] = {}
    for directory, _description in STAGE_DIRECTORIES:
        files = sorted(
            str(path.relative_to(root_path))
            for path in (root_path / directory).iterdir()
            if path.is_file()
        )
        if files:
            entries[directory] = files
    payload = {
        "schema_version": 1,
        "layout": "staged",
        "stages": [
            {"directory": name, "description": description, "files": entries.get(name, [])}
            for name, description in STAGE_DIRECTORIES
        ],
        "top_level_indexes": [
            name
            for name in ("README.md", "manifest.json", "artifact_index.json", "sweep.csv", "sweep.json")
            if (root_path / name).exists()
        ],
    }
    (root_path / "artifact_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_stage_paths(root: str | Path) -> dict[str, Path]:
    """Expose the canonical directories for callers that write custom files."""

    root_path = ensure_output_layout(root)
    return {name: root_path / name for name, _description in STAGE_DIRECTORIES}
