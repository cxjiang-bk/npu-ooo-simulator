from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .scheduler import SchedulerPipelineConfig
from .placement import (
    OperandPlacementConfig,
    OperationClassPlacementConfig,
    OperationPlacementConfig,
)


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
    pipeline: SchedulerPipelineConfig = field(default_factory=SchedulerPipelineConfig)

    def validate(self) -> tuple[str, ...]:
        return self.pipeline.validate() + tuple(
            f"scheduler {name} must be positive"
            for name, value in (
                ("instruction_queue_depth", self.instruction_queue_depth),
                ("rob_entries", self.rob_entries),
                ("max_inflight_tiles", self.max_inflight_tiles),
                ("dependency_window", self.dependency_window),
            )
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_queue_depth": self.instruction_queue_depth,
            "rob_entries": self.rob_entries,
            "max_inflight_tiles": self.max_inflight_tiles,
            "dependency_window": self.dependency_window,
            "pipeline": self.pipeline.to_dict(),
        }


@dataclass(frozen=True)
class MachineConfig:
    config_id: str
    memory_levels: tuple[MemoryLevelConfig, ...]
    execution_units: tuple[ExecutionUnitConfig, ...]
    transfer_paths: tuple[TransferPathConfig, ...]
    scheduler: SchedulerCapacityConfig = field(default_factory=SchedulerCapacityConfig)
    operation_placements: tuple[OperationPlacementConfig, ...] = ()
    operation_class_placements: tuple[OperationClassPlacementConfig, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MachineConfig":
        return machine_config_from_dict(payload)

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
        placements = {item.operation: item for item in self.operation_placements}
        if len(placements) != len(self.operation_placements):
            issues.append("operation placement operation names must be unique")
        paths = {(item.source, item.target): item for item in self.transfer_paths}
        roots = {level.name for level in self.memory_levels if level.parent is None}
        for placement in self.operation_placements:
            issues.extend(placement.validate())
            if placement.unit not in unit_names:
                issues.append(
                    f"operation placement '{placement.operation}' references unknown unit "
                    f"'{placement.unit}'"
                )
            elif placement.operation not in self.unit(placement.unit).supported_ops:
                issues.append(
                    f"unit '{placement.unit}' does not support placed operation "
                    f"'{placement.operation}'"
                )
            for operand in placement.operands:
                if operand.memory not in memory_names:
                    issues.append(
                        f"operand placement '{placement.operation}.{operand.role}' references "
                        f"unknown memory '{operand.memory}'"
                    )
                for source, target in zip(operand.route, operand.route[1:]):
                    if (source, target) not in paths:
                        issues.append(
                            f"operand placement '{placement.operation}.{operand.role}' has no "
                            f"transfer path {source}->{target}"
                        )
                if operand.direction == "input" and operand.route and operand.route[0] not in roots:
                    issues.append(
                        f"input placement '{placement.operation}.{operand.role}' route must "
                        "start at a root memory"
                    )
                if operand.direction in {"output", "state"} and operand.route and operand.route[-1] not in roots:
                    issues.append(
                        f"{operand.direction} placement '{placement.operation}.{operand.role}' "
                        "route must end at a root memory"
                    )
        class_ids = [item.class_id for item in self.operation_class_placements]
        if len(set(class_ids)) != len(class_ids):
            issues.append("operation class placement ids must be unique")
        class_operations: dict[str, str] = {}
        for rule in self.operation_class_placements:
            issues.extend(rule.validate())
            if rule.unit not in unit_names:
                issues.append(
                    f"operation class '{rule.class_id}' references unknown unit '{rule.unit}'"
                )
            if rule.local_memory not in memory_names:
                issues.append(
                    f"operation class '{rule.class_id}' references unknown local memory "
                    f"'{rule.local_memory}'"
                )
            for operation in rule.operations:
                previous = class_operations.setdefault(operation, rule.class_id)
                if previous != rule.class_id:
                    issues.append(
                        f"operation '{operation}' belongs to target classes '{previous}' and "
                        f"'{rule.class_id}'"
                    )
            for route in (rule.input_route, rule.output_route):
                for source, target in zip(route, route[1:]):
                    if (source, target) not in paths:
                        issues.append(
                            f"operation class '{rule.class_id}' has no transfer path "
                            f"{source}->{target}"
                        )
            if rule.input_route and rule.input_route[0] not in roots:
                issues.append(
                    f"operation class '{rule.class_id}' input route must start at root memory"
                )
            if rule.output_route and rule.output_route[-1] not in roots:
                issues.append(
                    f"operation class '{rule.class_id}' output route must end at root memory"
                )
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

    def placement(self, operation: str) -> OperationPlacementConfig:
        for item in self.operation_placements:
            if item.operation == operation:
                return item
        raise KeyError(operation)

    def operation_class(self, operation: str) -> OperationClassPlacementConfig:
        matches = [
            item
            for item in self.operation_class_placements
            if operation in item.operations
        ]
        if len(matches) != 1:
            raise KeyError(operation)
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "memory_levels": [level.to_dict() for level in self.memory_levels],
            "execution_units": [unit.to_dict() for unit in self.execution_units],
            "transfer_paths": [path.to_dict() for path in self.transfer_paths],
            "scheduler": self.scheduler.to_dict(),
            "operation_placements": [
                item.to_dict() for item in self.operation_placements
            ],
            "operation_class_placements": [
                item.to_dict() for item in self.operation_class_placements
            ],
            "attributes": dict(self.attributes),
        }

    def stable_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def topology_hash(self) -> str:
        """Hash the parts of the machine that determine compiled placement.

        Capacity, bandwidth, latency and unit counts are simulation parameters:
        they may be changed when replaying an existing compile package.  Memory
        identities/parentage, transfer connectivity, operation placement and
        supported operation families change the meaning of the generated
        payload and therefore require recompilation.
        """

        payload = {
            "memory_levels": [
                {
                    "name": level.name,
                    "parent": level.parent,
                    "alignment_bytes": level.alignment_bytes,
                }
                for level in self.memory_levels
            ],
            "execution_units": [
                {
                    "name": unit.name,
                    "supported_ops": list(unit.supported_ops),
                }
                for unit in self.execution_units
            ],
            "transfer_paths": [
                {
                    "source": path.source,
                    "target": path.target,
                    "engine": path.engine,
                    "transform": path.transform,
                }
                for path in self.transfer_paths
            ],
            "operation_placements": [
                placement.to_dict() for placement in self.operation_placements
            ],
            "operation_class_placements": [
                placement.to_dict()
                for placement in self.operation_class_placements
            ],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
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
                supported_ops=("load", "store", "copy", "transpose", "gather"),
                queue_depth=8,
                latency_cycles=2,
                initiation_interval_cycles=1,
                attributes={"gather_timing": "root_bandwidth_proxy"},
            ),
            ExecutionUnitConfig(
                "MXU",
                supported_ops=("matmul", "batched_matmul", "gemv", "conv2d"),
                queue_depth=4,
                latency_cycles=16,
                initiation_interval_cycles=4,
            ),
            ExecutionUnitConfig(
                "ARU",
                supported_ops=(
                    "elementwise",
                    "residual_add",
                    "reduce",
                    "softmax",
                    "rmsnorm",
                    "layernorm",
                    "swiglu",
                    "kv_cache_update",
                    "batch_norm",
                    "pool",
                ),
                queue_depth=8,
                latency_cycles=4,
                initiation_interval_cycles=2,
                attributes={"elements_per_cycle": 16},
            ),
        ),
        transfer_paths=(
            TransferPathConfig("DRAM", "SRAM", "DMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=2),
            TransferPathConfig("SRAM", "RF", "DMA", bandwidth_bytes_per_cycle=64, setup_latency_cycles=1),
            TransferPathConfig("RF", "SRAM", "DMA", bandwidth_bytes_per_cycle=64, setup_latency_cycles=1),
            TransferPathConfig("SRAM", "DRAM", "DMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=2),
        ),
        operation_placements=(
            OperationPlacementConfig(
                "matmul",
                "MXU",
                (
                    OperandPlacementConfig("lhs", "SRAM", ("DRAM", "SRAM"), "input"),
                    OperandPlacementConfig("rhs", "SRAM", ("DRAM", "SRAM"), "input"),
                    OperandPlacementConfig("output", "SRAM", ("SRAM", "DRAM"), "output"),
                ),
            ),
        ),
        operation_class_placements=(
            OperationClassPlacementConfig(
                "vector-local",
                (
                    "elementwise",
                    "residual_add",
                    "reduce",
                    "softmax",
                    "rmsnorm",
                    "layernorm",
                    "swiglu",
                    "kv_cache_update",
                    "batch_norm",
                    "pool",
                ),
                "ARU",
                "SRAM",
                ("DRAM", "SRAM"),
                ("SRAM", "DRAM"),
            ),
            OperationClassPlacementConfig(
                "tensor-local",
                ("conv2d",),
                "MXU",
                "SRAM",
                ("DRAM", "SRAM"),
                ("SRAM", "DRAM"),
            ),
            OperationClassPlacementConfig(
                "root-transfer",
                ("reshape", "transpose", "slice", "embedding"),
                "DMA",
                "DRAM",
                ("DRAM",),
                ("DRAM",),
                direct=True,
            ),
        ),
        attributes={"source": "hand-written", "calibration_status": "analytical"},
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
            ExecutionUnitConfig(
                "GDMA",
                supported_ops=("load", "store", "copy", "gather"),
                queue_depth=8,
                latency_cycles=4,
                initiation_interval_cycles=1,
                attributes={
                    "gather_timing": "root_bandwidth_proxy",
                    "hardware_mapping": "unvalidated",
                },
            ),
            ExecutionUnitConfig("LDMA", supported_ops=("load", "store", "transpose"), queue_depth=8, latency_cycles=4, initiation_interval_cycles=1),
            ExecutionUnitConfig("MXU", supported_ops=("matmul", "batched_matmul", "gemv", "conv2d"), queue_depth=8, pipeline_depth=4, latency_cycles=32, initiation_interval_cycles=4, attributes={"rows": 16, "cols": 8, "k": 8}),
            ExecutionUnitConfig(
                "ARU",
                supported_ops=(
                    "softmax",
                    "layernorm",
                    "rmsnorm",
                    "swiglu",
                    "reduce",
                    "elementwise",
                    "residual_add",
                    "kv_cache_update",
                    "batch_norm",
                    "pool",
                ),
                queue_depth=8,
                latency_cycles=8,
                initiation_interval_cycles=2,
            ),
        ),
        transfer_paths=(
            TransferPathConfig("GM", "UB", "GDMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=4),
            TransferPathConfig("UB", "LMB", "LDMA", bandwidth_bytes_per_cycle=32, setup_latency_cycles=4),
            TransferPathConfig("UB", "RMB", "LDMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=4, transform="transpose", transform_latency_cycles=4),
            TransferPathConfig("PSB", "UB", "ARU", bandwidth_bytes_per_cycle=8, setup_latency_cycles=2),
            TransferPathConfig("UB", "GM", "GDMA", bandwidth_bytes_per_cycle=16, setup_latency_cycles=4),
        ),
        operation_placements=(
            OperationPlacementConfig(
                "matmul",
                "MXU",
                (
                    OperandPlacementConfig("lhs", "LMB", ("GM", "UB", "LMB"), "input"),
                    OperandPlacementConfig("rhs", "RMB", ("GM", "UB", "RMB"), "input"),
                    OperandPlacementConfig("output", "PSB", ("PSB", "UB", "GM"), "output"),
                ),
            ),
        ),
        operation_class_placements=(
            OperationClassPlacementConfig(
                "vector-local",
                (
                    "elementwise",
                    "residual_add",
                    "reduce",
                    "softmax",
                    "rmsnorm",
                    "layernorm",
                    "swiglu",
                    "kv_cache_update",
                    "batch_norm",
                    "pool",
                ),
                "ARU",
                "UB",
                ("GM", "UB"),
                ("UB", "GM"),
            ),
            OperationClassPlacementConfig(
                "tensor-local",
                ("conv2d",),
                "MXU",
                "UB",
                ("GM", "UB"),
                ("UB", "GM"),
            ),
            OperationClassPlacementConfig(
                "root-transfer",
                ("reshape", "transpose", "slice", "embedding"),
                "GDMA",
                "GM",
                ("GM",),
                ("GM",),
                direct=True,
            ),
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


def machine_config_from_dict(payload: Mapping[str, Any]) -> MachineConfig:
    """Decode the canonical MachineConfig JSON schema without binding to a profile name."""

    if not isinstance(payload, Mapping):
        raise ValueError("machine config payload must be an object")
    try:
        memory_levels = tuple(
            MemoryLevelConfig(
                name=item["name"],
                parent=item.get("parent"),
                capacity_bytes=item.get("capacity_bytes"),
                read_bandwidth_bytes_per_cycle=item["read_bandwidth_bytes_per_cycle"],
                write_bandwidth_bytes_per_cycle=item["write_bandwidth_bytes_per_cycle"],
                read_latency_cycles=item.get("read_latency_cycles", 0.0),
                write_latency_cycles=item.get("write_latency_cycles", 0.0),
                read_ports=item.get("read_ports", 1),
                write_ports=item.get("write_ports", 1),
                bank_count=item.get("bank_count", 1),
                bank_width_bytes=item.get("bank_width_bytes"),
                alignment_bytes=item.get("alignment_bytes", 1),
                attributes=item.get("attributes", {}),
            )
            for item in payload["memory_levels"]
        )
        execution_units = tuple(
            ExecutionUnitConfig(
                name=item["name"],
                count=item.get("count", 1),
                supported_ops=tuple(item.get("supported_ops", ())),
                queue_depth=item.get("queue_depth", 1),
                issue_width=item.get("issue_width", 1),
                pipeline_depth=item.get("pipeline_depth", 1),
                latency_cycles=item.get("latency_cycles", 1.0),
                initiation_interval_cycles=item.get("initiation_interval_cycles", 1.0),
                attributes=item.get("attributes", {}),
            )
            for item in payload["execution_units"]
        )
        transfer_paths = tuple(
            TransferPathConfig(
                source=item["source"],
                target=item["target"],
                engine=item["engine"],
                channel_count=item.get("channel_count", 1),
                bandwidth_bytes_per_cycle=item.get("bandwidth_bytes_per_cycle", 1.0),
                setup_latency_cycles=item.get("setup_latency_cycles", 0.0),
                transform=item.get("transform"),
                transform_latency_cycles=item.get("transform_latency_cycles", 0.0),
                can_overlap=item.get("can_overlap", True),
                attributes=item.get("attributes", {}),
            )
            for item in payload["transfer_paths"]
        )
        scheduler_payload = payload.get("scheduler", {})
        scheduler = SchedulerCapacityConfig(
            instruction_queue_depth=scheduler_payload.get("instruction_queue_depth", 16),
            rob_entries=scheduler_payload.get("rob_entries", 8),
            max_inflight_tiles=scheduler_payload.get("max_inflight_tiles", 8),
            dependency_window=scheduler_payload.get("dependency_window", 8),
            pipeline=SchedulerPipelineConfig.from_dict(scheduler_payload.get("pipeline", {})),
        )
        config = MachineConfig(
            config_id=payload["config_id"],
            memory_levels=memory_levels,
            execution_units=execution_units,
            transfer_paths=transfer_paths,
            scheduler=scheduler,
            operation_placements=tuple(
                OperationPlacementConfig.from_dict(item)
                for item in payload.get("operation_placements", ())
            ),
            operation_class_placements=tuple(
                OperationClassPlacementConfig.from_dict(item)
                for item in payload.get("operation_class_placements", ())
            ),
            attributes=payload.get("attributes", {}),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid machine config payload") from exc
    _raise_if_invalid(config)
    return config


def load_machine_config(path: str | Path) -> MachineConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load machine config '{config_path}': {exc}") from exc
    return machine_config_from_dict(payload)
