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
