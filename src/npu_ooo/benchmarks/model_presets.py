"""Named model-level benchmark presets for one-block scheduling studies.

The presets intentionally use a bounded proxy shape so that a complete
compile/simulate/trace run stays quick.  Native model dimensions and
architecture assumptions are kept in ``ModelSpec.attributes`` and are
therefore visible in every exported manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from npu_ooo.ir import BenchmarkCase, ExecutionPhase, ModelFamily, ModelSpec

from .transformer_block import build_transformer_block_case, build_transformer_block_model


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_id: str
    family: ModelFamily
    variant: str
    tokens: int
    sequence: int
    head_dim: int
    intermediate: int
    native_config: dict[str, Any]
    assumptions: tuple[str, ...]
    phase: ExecutionPhase = ExecutionPhase.PREFILL

    def build(
        self,
        *,
        architecture_profile: str = "minimal",
        scheduler_profile: str = "sequential",
        tokens: int | None = None,
        sequence: int | None = None,
        head_dim: int | None = None,
        intermediate: int | None = None,
        phase: ExecutionPhase | str | None = None,
    ) -> tuple[ModelSpec, BenchmarkCase]:
        effective_tokens = self.tokens if tokens is None else tokens
        effective_sequence = self.sequence if sequence is None else sequence
        effective_head_dim = self.head_dim if head_dim is None else head_dim
        effective_intermediate = self.intermediate if intermediate is None else intermediate
        base_model = build_transformer_block_model(
            tokens=effective_tokens,
            sequence=effective_sequence,
            head_dim=effective_head_dim,
            intermediate=effective_intermediate,
        )
        model_attributes = dict(base_model.attributes)
        model_attributes.update(
            {
                "preset": self.name,
                "benchmark_status": "proxy",
                "native_config": dict(self.native_config),
                "proxy_shape": {
                    "tokens": effective_tokens,
                    "sequence": effective_sequence,
                    "head_dim": effective_head_dim,
                    "intermediate": effective_intermediate,
                },
                "assumptions": list(self.assumptions),
            }
        )
        model = replace(
            base_model,
            model_id=self.model_id,
            family=self.family,
            variant=self.variant,
            attributes=model_attributes,
        )

        selected_phase = self.phase if phase is None else phase
        normalized_phase = selected_phase.value if isinstance(selected_phase, ExecutionPhase) else str(selected_phase)
        case = build_transformer_block_case(
            tokens=effective_tokens,
            architecture_profile=architecture_profile,
            scheduler_profile=scheduler_profile,
        )
        case_attributes = dict(case.attributes)
        case_attributes.update(
            {
                "preset": self.name,
                "benchmark_status": "proxy",
                "native_config": dict(self.native_config),
                "proxy_shape": {
                    "tokens": effective_tokens,
                    "sequence": effective_sequence,
                    "head_dim": effective_head_dim,
                    "intermediate": effective_intermediate,
                },
                "assumptions": list(self.assumptions),
            }
        )
        benchmark_case = replace(
            case,
            case_id=f"{self.model_id}_{normalized_phase}_one_block",
            model_id=self.model_id,
            phase=selected_phase,
            sequence_length=effective_sequence,
            attributes=case_attributes,
        )
        return model, benchmark_case


MODEL_PRESETS: dict[str, ModelPreset] = {
    "bert-base": ModelPreset(
        name="bert-base",
        model_id="bert_base",
        family=ModelFamily.ENCODER_TRANSFORMER,
        variant="bert-base-attention-mlp-proxy-v0",
        tokens=128,
        sequence=128,
        head_dim=64,
        intermediate=256,
        native_config={"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072},
        assumptions=(
            "uses the pre-norm skeleton as a scheduling proxy; BERT's exact norm placement is not modeled",
            "multi-head projections are collapsed into one shape-only attention path",
        ),
    ),
    "gpt-j": ModelPreset(
        name="gpt-j",
        model_id="gptj",
        family=ModelFamily.DECODER_TRANSFORMER,
        variant="gpt-j-attention-mlp-proxy-v0",
        tokens=128,
        sequence=128,
        head_dim=128,
        intermediate=512,
        native_config={"hidden_size": 4096, "num_attention_heads": 16, "intermediate_size": 16384},
        assumptions=(
            "Q/K/V projections, RoPE, causal mask and KV-cache traffic are not yet lowered",
            "the one-block graph uses a single collapsed attention path",
        ),
    ),
    "llama2-7b": ModelPreset(
        name="llama2-7b",
        model_id="llama2_7b",
        family=ModelFamily.DECODER_TRANSFORMER,
        variant="llama2-attention-mlp-proxy-v0",
        tokens=128,
        sequence=128,
        head_dim=128,
        intermediate=512,
        native_config={"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 32, "intermediate_size": 11008},
        assumptions=(
            "Q/K/V projections, RoPE, causal mask and KV-cache traffic are not yet lowered",
            "the one-block graph uses a single collapsed attention path",
        ),
    ),
    "deepseek-r1-16b": ModelPreset(
        name="deepseek-r1-16b",
        model_id="deepseek_r1_16b_proxy",
        family=ModelFamily.DECODER_REASONING,
        variant="deepseek-r1-attention-mlp-proxy-v0",
        tokens=128,
        sequence=128,
        head_dim=128,
        intermediate=512,
        native_config={"parameter_count": "16B", "hidden_size": "model-config-required"},
        assumptions=(
            "the public benchmark configuration must be supplied before claiming dense or MoE behavior",
            "this preset is a dense shape-only proxy and does not model expert routing",
            "Q/K/V projections, RoPE, causal mask and KV-cache traffic are not yet lowered",
        ),
    ),
}


def available_model_presets() -> tuple[str, ...]:
    return tuple(MODEL_PRESETS)


def build_model_preset(
    name: str,
    *,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
    tokens: int | None = None,
    sequence: int | None = None,
    head_dim: int | None = None,
    intermediate: int | None = None,
    phase: ExecutionPhase | str | None = None,
) -> tuple[ModelSpec, BenchmarkCase]:
    try:
        preset = MODEL_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(available_model_presets())
        raise ValueError(f"unknown model preset '{name}'; choose from: {choices}") from exc
    return preset.build(
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
        tokens=tokens,
        sequence=sequence,
        head_dim=head_dim,
        intermediate=intermediate,
        phase=phase,
    )


__all__ = ["MODEL_PRESETS", "ModelPreset", "available_model_presets", "build_model_preset"]
