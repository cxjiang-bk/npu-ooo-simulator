"""Adapter for the console trace emitted by the repository MXU testbench.

``rtl/unit_test/mxu/tb_mxu.sv`` prints prepared dimensions, instruction
acceptance, and ``done_if.vld`` events.  This adapter preserves that boundary
as descriptor-to-completion timing.  It deliberately does not manufacture a
compute-start marker: the testbench does not print one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .rtl_trace import TRACE_FORMAT


_PREPARED_RE = re.compile(
    r"\[(?P<cycle>[-+]?\d+(?:\.\d+)?)\]\s+Prepared instruction:\s+"
    r"M=(?P<m>\d+),\s+N=(?P<n>\d+),\s+K1=(?P<k1>\d+),\s+"
    r"psum_en=(?P<psum>[01]),\s+bias_en=(?P<bias>[01])"
)
_ACCEPTED_RE = re.compile(
    r"\[(?P<cycle>[-+]?\d+(?:\.\d+)?)\]\s+Test\s+.*?instruction accepted"
)
_DONE_RE = re.compile(
    r"\[(?P<cycle>[-+]?\d+(?:\.\d+)?)\]\s+Done Signal\s+#(?P<count>\d+):\s+"
    r"instr_idx=(?P<instr_idx>\d+),\s+task_done=(?P<task_done>[01])"
)


def _cycle(match: re.Match[str]) -> float:
    return float(match.group("cycle"))


def parse_mxu_vcs_log(
    text: str,
    *,
    k_per_tile: int = 8,
    source: str = "rtl/unit_test/mxu/tb_mxu.sv",
) -> dict[str, Any]:
    """Parse the MXU testbench's ``$display`` lines into trace records.

    The testbench calls its K field ``K1`` and stores the number of K tiles;
    the current RTL parameter is ``K0=8``.  ``k_per_tile`` is therefore an
    explicit argument rather than an assumption hidden in the parser.
    """

    if isinstance(k_per_tile, bool) or not isinstance(k_per_tile, int) or k_per_tile <= 0:
        raise ValueError("k_per_tile must be a positive integer")
    prepared: list[dict[str, Any]] = []
    accepted: list[float] = []
    done: list[dict[str, Any]] = []
    ignored_end_acceptance_count = 0
    ignored_end_done_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _PREPARED_RE.search(line)
        if match:
            prepared.append(
                {
                    "prepared_cycle": _cycle(match),
                    "m": int(match.group("m")),
                    "n": int(match.group("n")),
                    "k1": int(match.group("k1")),
                    "psum_en": int(match.group("psum")),
                    "bias_en": int(match.group("bias")),
                    "line": line_number,
                }
            )
            continue
        match = _ACCEPTED_RE.search(line)
        if match:
            if "END instruction accepted" in line:
                ignored_end_acceptance_count += 1
                continue
            accepted.append(_cycle(match))
            continue
        match = _DONE_RE.search(line)
        if match:
            if match.group("task_done") == "1":
                ignored_end_done_count += 1
                continue
            done.append(
                {
                    "cycle": _cycle(match),
                    "instr_idx": int(match.group("instr_idx")),
                    "task_done": int(match.group("task_done")),
                }
            )

    if not prepared:
        raise ValueError("MXU VCS log contains no 'Prepared instruction' records")
    if len(accepted) < len(prepared):
        raise ValueError(
            f"MXU VCS log has {len(prepared)} prepared instructions but only "
            f"{len(accepted)} acceptance events"
        )
    if len(done) < len(prepared):
        raise ValueError(
            f"MXU VCS log has {len(prepared)} prepared instructions but only "
            f"{len(done)} done events"
        )

    records: list[dict[str, Any]] = []
    for index, instruction in enumerate(prepared):
        accepted_cycle = accepted[index]
        completion = done[index]
        if completion["cycle"] <= accepted_cycle:
            raise ValueError(
                f"MXU VCS log instruction {index} completes before acceptance "
                f"({completion['cycle']:g} <= {accepted_cycle:g})"
            )
        records.append(
            {
                "instruction_id": f"mxu-vcs-{index}",
                "batch": 1,
                "m": instruction["m"],
                "n": instruction["n"],
                "k": instruction["k1"] * k_per_tile,
                "descriptor_issue_cycle": accepted_cycle,
                "psb_write_done_cycle": completion["cycle"],
                "attributes": {
                    "source_line": instruction["line"],
                    "k1": instruction["k1"],
                    "k_per_tile": k_per_tile,
                    "psum_en": instruction["psum_en"],
                    "bias_en": instruction["bias_en"],
                    "rtl_instr_idx": completion["instr_idx"],
                    "task_done": completion["task_done"],
                },
            }
        )

    return {
        "format": TRACE_FORMAT,
        "metadata": {
            "source": source,
            "source_format": "npu_ooo.mxu_vcs_console.v1",
            "intervals_available": ["descriptor_issue_to_done"],
            "descriptor_issue_event": "instruction accepted",
            "completion_event": "done_if.vld / Done Signal",
            "compute_start_available": False,
            "k_semantics": "k = K1 * k_per_tile",
            "k_per_tile": k_per_tile,
            "calibration_status": "rtl-observed",
            "prepared_count": len(prepared),
            "accepted_count": len(accepted),
            "done_count": len(done),
            "ignored_end_acceptance_count": ignored_end_acceptance_count,
            "ignored_end_done_count": ignored_end_done_count,
        },
        "records": records,
    }


def load_mxu_vcs_log(
    path: str | Path,
    *,
    k_per_tile: int = 8,
    source: str | None = None,
) -> dict[str, Any]:
    log_path = Path(path)
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot load MXU VCS log '{log_path}': {exc}") from exc
    return parse_mxu_vcs_log(
        text,
        k_per_tile=k_per_tile,
        source=source or str(log_path),
    )


__all__ = ["load_mxu_vcs_log", "parse_mxu_vcs_log"]
