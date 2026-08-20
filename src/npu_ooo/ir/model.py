from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .operator import OperatorGraph, ShapeValue, TensorSpec


class ModelFamily(str, Enum):
    SYNTHETIC = "synthetic"
    CNN_RESIDUAL = "cnn_residual"
    ENCODER_TRANSFORMER = "encoder_transformer"
    DECODER_TRANSFORMER = "decoder_transformer"
    DECODER_REASONING = "decoder_reasoning"
    MOE_DECODER = "moe_decoder"


class ExecutionPhase(str, Enum):
    INFERENCE = "inference"
    PREFILL = "prefill"
    DECODE = "decode"
    TRAIN = "train"


class EvaluationScope(str, Enum):
    ONE_BLOCK = "one_block"
    LAYER = "layer"
    FULL_MODEL = "full_model"


def _pairs_to_dict(values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {name: value for name, value in values}


@dataclass(frozen=True)
class PersistentStateSpec:
    name: str
    shape: tuple[ShapeValue, ...]
    dtype: str = "fp16"
    layout: str = "dense"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        return TensorSpec(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            layout=self.layout,
            attributes=self.attributes,
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class GraphTemplate:
    template_id: str
    graph: OperatorGraph
    parameters: tuple[str, ...] = ()
    repeat_count: int = 1
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.graph.validate())
        if not self.template_id:
            issues.append("graph template id must not be empty")
        if self.repeat_count <= 0:
            issues.append(f"graph template '{self.template_id}' repeat_count must be positive")
        if len(set(self.parameters)) != len(self.parameters):
            issues.append(f"graph template '{self.template_id}' parameters must be unique")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "graph": self.graph.to_dict(),
            "parameters": list(self.parameters),
            "repeat_count": self.repeat_count,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    model_id: str
    evaluation_scope: EvaluationScope | str
    phase: ExecutionPhase | str
    batch: int = 1
    sequence_length: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    dtype: str = "fp16"
    quantization: str | None = None
    shape_overrides: tuple[tuple[str, int], ...] = ()
    architecture_profile: str = "minimal"
    scheduler_profile: str = "sequential"
    warmup: int = 0
    repetitions: int = 1
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_scope(self) -> str:
        return self.evaluation_scope.value if isinstance(self.evaluation_scope, EvaluationScope) else str(self.evaluation_scope)

    @property
    def normalized_phase(self) -> str:
        return self.phase.value if isinstance(self.phase, ExecutionPhase) else str(self.phase)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.case_id:
            issues.append("benchmark case id must not be empty")
        if not self.model_id:
            issues.append("benchmark case model_id must not be empty")
        if self.normalized_scope not in {item.value for item in EvaluationScope}:
            issues.append(f"unsupported evaluation scope '{self.normalized_scope}'")
        if self.normalized_phase not in {item.value for item in ExecutionPhase}:
            issues.append(f"unsupported execution phase '{self.normalized_phase}'")
        for name, value in (
            ("batch", self.batch),
            ("warmup", self.warmup),
            ("repetitions", self.repetitions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name != "warmup" else 0):
                issues.append(f"benchmark case {name} has an invalid value {value!r}")
        for name, value in (
            ("sequence_length", self.sequence_length),
            ("image_height", self.image_height),
            ("image_width", self.image_width),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                issues.append(f"benchmark case {name} must be positive when specified")
        overrides = _pairs_to_dict(self.shape_overrides)
        if len(overrides) != len(self.shape_overrides):
            issues.append("benchmark shape_overrides must have unique symbols")
        for name, value in self.shape_overrides:
            if not name or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                issues.append(f"benchmark shape override '{name}' must be a positive integer")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model_id": self.model_id,
            "evaluation_scope": self.normalized_scope,
            "phase": self.normalized_phase,
            "batch": self.batch,
            "sequence_length": self.sequence_length,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "shape_overrides": {name: value for name, value in self.shape_overrides},
            "architecture_profile": self.architecture_profile,
            "scheduler_profile": self.scheduler_profile,
            "warmup": self.warmup,
            "repetitions": self.repetitions,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: ModelFamily | str
    variant: str
    shape_symbols: tuple[tuple[str, int], ...]
    templates: tuple[GraphTemplate, ...]
    top_level_template: str | None = None
    dtype_policy: str = "fp16"
    layout_policy: str = "dense"
    persistent_state: tuple[PersistentStateSpec, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_family(self) -> str:
        return self.family.value if isinstance(self.family, ModelFamily) else str(self.family)

    @property
    def shape_environment(self) -> dict[str, int]:
        return _pairs_to_dict(self.shape_symbols)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.model_id:
            issues.append("model id must not be empty")
        if not self.variant:
            issues.append(f"model '{self.model_id}' variant must not be empty")
        if len(self.shape_environment) != len(self.shape_symbols):
            issues.append(f"model '{self.model_id}' shape symbols must be unique")
        for name, value in self.shape_symbols:
            if not name or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                issues.append(f"model shape symbol '{name}' must be a positive integer")
        template_ids = {template.template_id for template in self.templates}
        if len(template_ids) != len(self.templates):
            issues.append(f"model '{self.model_id}' template ids must be unique")
        if self.top_level_template is not None and self.top_level_template not in template_ids:
            issues.append(f"model '{self.model_id}' references unknown top-level template '{self.top_level_template}'")
        for template in self.templates:
            issues.extend(template.validate())
        state_names = {state.name for state in self.persistent_state}
        if len(state_names) != len(self.persistent_state):
            issues.append(f"model '{self.model_id}' persistent state names must be unique")
        for state in self.persistent_state:
            issues.extend(state.validate())
        return tuple(issues)

    def instantiate(
        self,
        case: BenchmarkCase,
        *,
        template_id: str | None = None,
    ) -> "ModelInstance":
        model_issues = self.validate()
        case_issues = case.validate()
        if model_issues or case_issues:
            raise ValueError("; ".join((*model_issues, *case_issues)))
        if case.model_id != self.model_id:
            raise ValueError(
                f"benchmark case '{case.case_id}' targets model '{case.model_id}', "
                f"not '{self.model_id}'"
            )
        selected_id = template_id or self.top_level_template
        if selected_id is None:
            if len(self.templates) != 1:
                raise ValueError("template_id is required when a model has multiple templates")
            selected_id = self.templates[0].template_id
        templates = {template.template_id: template for template in self.templates}
        template = templates.get(selected_id)
        if template is None:
            raise ValueError(f"model '{self.model_id}' has no template '{selected_id}'")
        environment = dict(self.shape_environment)
        environment.update(_pairs_to_dict(case.shape_overrides))
        graph = template.graph.resolve(environment)
        return ModelInstance(
            model_id=self.model_id,
            model_variant=self.variant,
            case_id=case.case_id,
            template_id=template.template_id,
            graph=graph,
            persistent_state=self.persistent_state,
            provenance={
                "model_id": self.model_id,
                "template_id": template.template_id,
                "evaluation_scope": case.normalized_scope,
                "phase": case.normalized_phase,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.normalized_family,
            "variant": self.variant,
            "shape_symbols": {name: value for name, value in self.shape_symbols},
            "templates": [template.to_dict() for template in self.templates],
            "top_level_template": self.top_level_template,
            "dtype_policy": self.dtype_policy,
            "layout_policy": self.layout_policy,
            "persistent_state": [state.to_dict() for state in self.persistent_state],
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ModelInstance:
    model_id: str
    model_variant: str
    case_id: str
    template_id: str
    graph: OperatorGraph
    persistent_state: tuple[PersistentStateSpec, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.graph.validate())
        if not self.model_id or not self.case_id or not self.template_id:
            issues.append("model instance identifiers must not be empty")
        for state in self.persistent_state:
            issues.extend(state.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "case_id": self.case_id,
            "template_id": self.template_id,
            "graph": self.graph.to_dict(),
            "persistent_state": [state.to_dict() for state in self.persistent_state],
            "provenance": dict(self.provenance),
        }
