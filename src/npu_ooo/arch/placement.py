"""Target operation placement and explicit transfer-route contracts."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class OperandPlacementConfig:
    role: str
    memory: str
    route: tuple[str, ...]
    direction: str
    layout: str = "packed"
    alignment_bytes: int | None = None

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.role or not self.memory:
            issues.append("operand placement role and memory must not be empty")
        if self.direction not in {"input", "output", "state"}:
            issues.append(
                f"operand placement '{self.role}' direction must be input, output or state"
            )
        if not self.route:
            issues.append(f"operand placement '{self.role}' route must not be empty")
        elif self.direction == "input" and self.route[-1] != self.memory:
            issues.append(
                f"input placement '{self.role}' route must end at '{self.memory}'"
            )
        elif self.direction in {"output", "state"} and self.route[0] != self.memory:
            issues.append(
                f"{self.direction} placement '{self.role}' route must start at '{self.memory}'"
            )
        if self.layout not in {"packed", "source"}:
            issues.append(f"operand placement '{self.role}' layout must be packed or source")
        if self.alignment_bytes is not None and (
            isinstance(self.alignment_bytes, bool)
            or not isinstance(self.alignment_bytes, int)
            or self.alignment_bytes <= 0
        ):
            issues.append(
                f"operand placement '{self.role}' alignment_bytes must be positive"
            )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "memory": self.memory,
            "route": list(self.route),
            "direction": self.direction,
            "layout": self.layout,
            "alignment_bytes": self.alignment_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperandPlacementConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("operand placement must be a JSON object")
        try:
            result = cls(
                role=str(payload["role"]),
                memory=str(payload["memory"]),
                route=tuple(str(item) for item in payload["route"]),
                direction=str(payload["direction"]),
                layout=str(payload.get("layout", "packed")),
                alignment_bytes=(
                    int(payload["alignment_bytes"])
                    if payload.get("alignment_bytes") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid operand placement") from exc
        issues = result.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return result


@dataclass(frozen=True)
class OperationPlacementConfig:
    operation: str
    unit: str
    operands: tuple[OperandPlacementConfig, ...]

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.operation or not self.unit:
            issues.append("operation placement operation and unit must not be empty")
        roles = [item.role for item in self.operands]
        if len(set(roles)) != len(roles):
            issues.append(f"operation placement '{self.operation}' roles must be unique")
        for item in self.operands:
            issues.extend(item.validate())
        return tuple(issues)

    def operand(self, role: str) -> OperandPlacementConfig:
        for item in self.operands:
            if item.role == role:
                return item
        raise KeyError(role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "unit": self.unit,
            "operands": [item.to_dict() for item in self.operands],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationPlacementConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("operation placement must be a JSON object")
        try:
            result = cls(
                operation=str(payload["operation"]),
                unit=str(payload["unit"]),
                operands=tuple(
                    OperandPlacementConfig.from_dict(item)
                    for item in payload.get("operands", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid operation placement") from exc
        issues = result.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return result


@dataclass(frozen=True)
class OperationClassPlacementConfig:
    """Explicit target rule for operations sharing one storage pattern."""

    class_id: str
    operations: tuple[str, ...]
    unit: str
    local_memory: str
    input_route: tuple[str, ...]
    output_route: tuple[str, ...]
    direct: bool = False

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.class_id or not self.unit or not self.local_memory:
            issues.append("operation class placement identities must not be empty")
        if not self.operations or len(set(self.operations)) != len(self.operations):
            issues.append(
                f"operation class placement '{self.class_id}' operations must be non-empty and unique"
            )
        if not self.input_route or not self.output_route:
            issues.append(
                f"operation class placement '{self.class_id}' routes must not be empty"
            )
        if self.direct:
            if len(self.input_route) != 1 or len(self.output_route) != 1:
                issues.append(
                    f"direct operation class '{self.class_id}' routes must contain one memory"
                )
        else:
            if self.input_route[-1:] != (self.local_memory,):
                issues.append(
                    f"operation class '{self.class_id}' input route must end at local memory"
                )
            if self.output_route[:1] != (self.local_memory,):
                issues.append(
                    f"operation class '{self.class_id}' output route must start at local memory"
                )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "operations": list(self.operations),
            "unit": self.unit,
            "local_memory": self.local_memory,
            "input_route": list(self.input_route),
            "output_route": list(self.output_route),
            "direct": self.direct,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationClassPlacementConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("operation class placement must be a JSON object")
        try:
            result = cls(
                class_id=str(payload["class_id"]),
                operations=tuple(str(item) for item in payload["operations"]),
                unit=str(payload["unit"]),
                local_memory=str(payload["local_memory"]),
                input_route=tuple(str(item) for item in payload["input_route"]),
                output_route=tuple(str(item) for item in payload["output_route"]),
                direct=bool(payload.get("direct", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid operation class placement") from exc
        issues = result.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return result
