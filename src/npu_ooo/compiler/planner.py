from __future__ import annotations

"""Automatic schedule planning boundary.

The current implementation intentionally reuses the project's deterministic
32-element heuristic.  Keeping it behind a planner object gives later passes a
stable place to add architecture-aware cost models without restoring
benchmark-specific ``default_*_schedule`` calls to the frontend path.
"""

from dataclasses import replace

from npu_ooo.ir import OperatorGraph, ScheduleSpec, plan_uniform_tiles


class SchedulePlanner:
    """Plan a resolved graph with a deterministic, shape-aware heuristic."""

    name = "heuristic-v1"

    def plan(self, graph: OperatorGraph, *, tile_size: int = 32) -> ScheduleSpec:
        schedule = plan_uniform_tiles(graph, tile_size=tile_size)
        return replace(
            schedule,
            attributes={
                **dict(schedule.attributes),
                "source": "automatic-planner",
                "planner": self.name,
            },
        )


def default_schedule_planner() -> SchedulePlanner:
    return SchedulePlanner()


__all__ = ["SchedulePlanner", "default_schedule_planner"]
