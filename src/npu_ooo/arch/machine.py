from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class MemoryLevelConfig:
    name: str
    parent: str | None
    capacity_bytes: int | None
    read_bandwidth_bytes_per_cycle: float
    write_bandwidth_bytes_per_cycle: float
    read_latency_cycles: float = 0.0
    write_latency_cycles: float = 0.0
    read_ports: int = 1
    write_ports: int = 1
    bank_count: int = 1
    bank_width_bytes: int | None = None
    alignment_bytes: int = 1
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name:
            issues.append("memory level name must not be empty")
        for label, value in (
            ("capacity_bytes", self.capacity_bytes),
            ("read_bandwidth_bytes_per_cycle", self.read_bandwidth_bytes_per_cycle),
            ("write_bandwidth_bytes_per_cycle", self.write_bandwidth_bytes_per_cycle),
            ("read_latency_cycles", self.read_latency_cycles),
            ("write_latency_cycles", self.write_latency_cycles),
            ("read_ports", self.read_ports),
            ("write_ports", self.write_ports),
            ("bank_count", self.bank_count),
            ("alignment_bytes", self.alignment_bytes),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
                issues.append(f"memory '{self.name}' {label} must be positive")
        if self.capacity_bytes == 0:
            issues.append(f"memory '{self.name}' capacity_bytes must be positive or None")
        if self.bank_width_bytes is not None and self.bank_width_bytes <= 0:
            issues.append(f"memory '{self.name}' bank_width_bytes must be positive when specified")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "capacity_bytes": self.capacity_bytes,
            "read_bandwidth_bytes_per_cycle": self.read_bandwidth_bytes_per_cycle,
            "write_bandwidth_bytes_per_cycle": self.write_bandwidth_bytes_per_cycle,
            "read_latency_cycles": self.read_latency_cycles,
            "write_latency_cycles": self.write_latency_cycles,
            "read_ports": self.read_ports,
            "write_ports": self.write_ports,
            "bank_count": self.bank_count,
            "bank_width_bytes": self.bank_width_bytes,
            "alignment_bytes": self.alignment_bytes,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ExecutionUnitConfig:
    name: str
    count: int = 1
    supported_ops: tuple[str, ...] = ()
    queue_depth: int = 1
    issue_width: int = 1
    pipeline_depth: int = 1
    latency_cycles: float = 1.0
    initiation_interval_cycles: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name:
            issues.append("execution unit name must not be empty")
        for label, value in (
            ("count", self.count),
            ("queue_depth", self.queue_depth),
            ("issue_width", self.issue_width),
            ("pipeline_depth", self.pipeline_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                issues.append(f"unit '{self.name}' {label} must be positive")
        for label, value in (
            ("latency_cycles", self.latency_cycles),
            ("initiation_interval_cycles", self.initiation_interval_cycles),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                issues.append(f"unit '{self.name}' {label} must be positive")
        if len(set(self.supported_ops)) != len(self.supported_ops):
            issues.append(f"unit '{self.name}' supported_ops must be unique")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "supported_ops": list(self.supported_ops),
            "queue_depth": self.queue_depth,
            "issue_width": self.issue_width,
            "pipeline_depth": self.pipeline_depth,
            "latency_cycles": self.latency_cycles,
            "initiation_interval_cycles": self.initiation_interval_cycles,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class TransferPathConfig:
    source: str
    target: str
    engine: str
    channel_count: int = 1
    bandwidth_bytes_per_cycle: float = 1.0
    setup_latency_cycles: float = 0.0
    transform: str | None = None
    transform_latency_cycles: float = 0.0
    can_overlap: bool = True
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.source or not self.target:
            issues.append("transfer path source and target must not be empty")
        if not self.engine:
            issues.append("transfer path engine must not be empty")
        if isinstance(self.channel_count, bool) or self.channel_count <= 0:
            issues.append(f"transfer path {self.source}->{self.target} channel_count must be positive")
        for label, value in (
            ("bandwidth_bytes_per_cycle", self.bandwidth_bytes_per_cycle),
            ("setup_latency_cycles", self.setup_latency_cycles),
            ("transform_latency_cycles", self.transform_latency_cycles),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (
                label == "bandwidth_bytes_per_cycle" and value == 0
            ):
                issues.append(f"transfer path {self.source}->{self.target} {label} is invalid")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "engine": self.engine,
            "channel_count": self.channel_count,
            "bandwidth_bytes_per_cycle": self.bandwidth_bytes_per_cycle,
            "setup_latency_cycles": self.setup_latency_cycles,
            "transform": self.transform,
            "transform_latency_cycles": self.transform_latency_cycles,
            "can_overlap": self.can_overlap,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class SchedulerCapacityConfig:
    instruction_queue_depth: int = 16
    rob_entries: int = 8
    max_inflight_tiles: int = 8
    dependency_window: int = 8

    def validate(self) -> tuple[str, ...]:
        return tuple(
            f"scheduler {name} must be positive"
            for name, value in (
                ("instruction_queue_depth", self.instruction_queue_depth),
                ("rob_entries", self.rob_entries),
                ("max_inflight_tiles", self.max_inflight_tiles),
                ("dependency_window", self.dependency_window),
            )
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "instruction_queue_depth": self.instruction_queue_depth,
            "rob_entries": self.rob_entries,
            "max_inflight_tiles": self.max_inflight_tiles,
            "dependency_window": self.dependency_window,
        }


@dataclass(frozen=True)
class MachineConfig:
    config_id: str
    memory_levels: tuple[MemoryLevelConfig, ...]
    execution_units: tuple[ExecutionUnitConfig, ...]
    transfer_paths: tuple[TransferPathConfig, ...]
    scheduler: SchedulerCapacityConfig = field(default_factory=SchedulerCapacityConfig)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.config_id:
            issues.append("machine config id must not be empty")
        memory_names = {level.name for level in self.memory_levels}
        if len(memory_names) != len(self.memory_levels):
            issues.append("memory level names must be unique")
        for level in self.memory_levels:
            issues.extend(level.validate())
            if level.parent is not None and level.parent not in memory_names:
                issues.append(f"memory '{level.name}' references unknown parent '{level.parent}'")
        issues.extend(self._validate_memory_acyclic())

        unit_names = {unit.name for unit in self.execution_units}
        if len(unit_names) != len(self.execution_units):
            issues.append("execution unit names must be unique")
        for unit in self.execution_units:
            issues.extend(unit.validate())
        for path in self.transfer_paths:
            issues.extend(path.validate())
            if path.source not in memory_names:
                issues.append(f"transfer path references unknown source memory '{path.source}'")
            if path.target not in memory_names:
                issues.append(f"transfer path references unknown target memory '{path.target}'")
            if path.engine not in unit_names:
                issues.append(f"transfer path references unknown engine '{path.engine}'")
        issues.extend(self.scheduler.validate())
        return tuple(issues)

    def _validate_memory_acyclic(self) -> tuple[str, ...]:
        parents = {level.name: level.parent for level in self.memory_levels}
        for name in parents:
            seen: set[str] = set()
            current: str | None = name
            while current is not None:
                if current in seen:
                    return (f"memory hierarchy contains a cycle at '{current}'",)
                seen.add(current)
                current = parents.get(current)
        return ()

    def memory(self, name: str) -> MemoryLevelConfig:
        for level in self.memory_levels:
            if level.name == name:
                return level
        raise KeyError(name)

    def unit(self, name: str) -> ExecutionUnitConfig:
        for unit in self.execution_units:
            if unit.name == name:
                return unit
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "memory_levels": [level.to_dict() for level in self.memory_levels],
            "execution_units": [unit.to_dict() for unit in self.execution_units],
            "transfer_paths": [path.to_dict() for path in self.transfer_paths],
            "scheduler": self.scheduler.to_dict(),
            "attributes": dict(self.attributes),
        }

    def stable_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def minimal_machine_config() -> MachineConfig:
    config = MachineConfig(
        config_id="minimal",
        memory_levels=(
            MemoryLevelConfig("DRAM", None, None, 16, 16, read_latency_cycles=40, write_latency_cycles=40),
            MemoryLevelConfig("SRAM", "DRAM", 256 * 1024, 64, 64, read_latency_cycles=2, write_latency_cycles=2),
            MemoryLevelConfig("RF", "SRAM", 32 * 1024, 256, 256, read_latency_cycles=1, write_latency_cycles=1),
        ),
        execution_units=(
            ExecutionUnitConfig(
                "DMA",
                supported_ops=("load", "store", "copy", "transpose"),
                queue_depth=8,
                latency_cycles=2,
                initiation_interval_cycles=1,
            ),
            ExecutionUnitConfig(
                "MXU",
                supported_ops=("matmul", "batched_matmul", "gemv"),
                queue_depth=4,
                latency_cycles=16,
                initiation_interval_cycles=4,
            ),
        ),
        transfer_paths=(
            TransferPathConfig("DRAM", "SRAM", "DMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=2),
            TransferPathConfig("SRAM", "RF", "DMA", bandwidth_bytes_per_cycle=64, setup_latency_cycles=1),
            TransferPathConfig("RF", "SRAM", "DMA", bandwidth_bytes_per_cycle=64, setup_latency_cycles=1),
            TransferPathConfig("SRAM", "DRAM", "DMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=2),
        ),
    )
    _raise_if_invalid(config)
    return config


def wide_mxu_machine_config() -> MachineConfig:
    base = minimal_machine_config()
    mxu = base.unit("MXU")
    config = replace(
        base,
        config_id="wide-mxu",
        execution_units=tuple(
            replace(mxu, count=2, issue_width=2, initiation_interval_cycles=2)
            if unit.name == "MXU"
            else unit
            for unit in base.execution_units
        ),
    )
    _raise_if_invalid(config)
    return config


def lpu_like_machine_config() -> MachineConfig:
    config = MachineConfig(
        config_id="lpu-like",
        memory_levels=(
            MemoryLevelConfig("GM", None, None, 16, 16, read_latency_cycles=40, write_latency_cycles=40, alignment_bytes=4096),
            MemoryLevelConfig("UB", "GM", 1024 * 1024, 256, 256, read_latency_cycles=2, write_latency_cycles=2, bank_count=16, bank_width_bytes=16),
            MemoryLevelConfig("LMB", "UB", 64 * 1024, 32, 32, read_latency_cycles=1, write_latency_cycles=1, bank_count=16, bank_width_bytes=16),
            MemoryLevelConfig("RMB", "UB", 64 * 1024, 16, 16, read_latency_cycles=1, write_latency_cycles=1, bank_count=8, bank_width_bytes=16),
            MemoryLevelConfig("PSB", "UB", 256 * 1024, 512, 512, read_latency_cycles=1, write_latency_cycles=1, bank_count=4, bank_width_bytes=128),
            MemoryLevelConfig("ARB", "UB", 2 * 1024, 8, 8, read_latency_cycles=1, write_latency_cycles=1),
        ),
        execution_units=(
            ExecutionUnitConfig("GDMA", supported_ops=("load", "store", "copy"), queue_depth=8, latency_cycles=4, initiation_interval_cycles=1),
            ExecutionUnitConfig("LDMA", supported_ops=("load", "store", "transpose"), queue_depth=8, latency_cycles=4, initiation_interval_cycles=1),
            ExecutionUnitConfig("MXU", supported_ops=("matmul", "batched_matmul", "gemv"), queue_depth=8, pipeline_depth=4, latency_cycles=32, initiation_interval_cycles=4, attributes={"rows": 16, "cols": 8, "k": 8}),
            ExecutionUnitConfig("ARU", supported_ops=("softmax", "layernorm", "rmsnorm", "reduce", "elementwise"), queue_depth=8, latency_cycles=8, initiation_interval_cycles=2),
        ),
        transfer_paths=(
            TransferPathConfig("GM", "UB", "GDMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=4),
            TransferPathConfig("UB", "LMB", "LDMA", bandwidth_bytes_per_cycle=32, setup_latency_cycles=4),
            TransferPathConfig("UB", "RMB", "LDMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=4, transform="transpose", transform_latency_cycles=4),
            TransferPathConfig("PSB", "UB", "ARU", bandwidth_bytes_per_cycle=8, setup_latency_cycles=2),
            TransferPathConfig("UB", "GM", "GDMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=4),
        ),
        scheduler=SchedulerCapacityConfig(instruction_queue_depth=32, rob_entries=8, max_inflight_tiles=8, dependency_window=8),
        attributes={"source": "analytical-lpu-like", "calibration_status": "analytical"},
    )
    _raise_if_invalid(config)
    return config


def _raise_if_invalid(config: MachineConfig) -> None:
    issues = config.validate()
    if issues:
        raise ValueError("; ".join(issues))
