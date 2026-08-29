from __future__ import annotations

"""Official StableHLO verification and canonical-IR import.

Torch-XLA is the only StableHLO producer in the production frontend.  This
module owns the next boundary: official dialect registration, MLIR
parse/verify, and projection into the semantic subset understood by the NPU
compiler.
"""

from dataclasses import dataclass, field
import importlib.metadata
import re
from typing import Any, Mapping

from .bridge import FrontendImport, FrontendImportError, FrontendKind
from .stablehlo import StableHLOAdapter


def _bindings() -> tuple[Any, Any, Any]:
    try:
        from mlir.ir import Context, Module
        import mlir.dialects.stablehlo as stablehlo_dialect
    except ModuleNotFoundError as exc:
        raise FrontendImportError(
            "official StableHLO bindings are unavailable; install the OpenXLA "
            "StableHLO wheel (see docs/install-stablehlo.md)"
        ) from exc
    return Context, Module, stablehlo_dialect


def official_stablehlo_available() -> bool:
    try:
        _bindings()
    except FrontendImportError:
        return False
    return True


def official_stablehlo_version() -> str | None:
    for distribution in ("stablehlo", "mlir-python-bindings"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@dataclass(frozen=True)
class OfficialStableHLOModule:
    """StableHLO text and provenance after official MLIR verification."""

    text: str
    canonical_text: str
    model_id: str
    variant: str = "stablehlo-torch-xla-v1"
    stablehlo_version: str | None = None
    verified: bool = True
    producer: str = "external-stablehlo"
    verifier: str = "official-stablehlo-mlir"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.text.strip():
            issues.append("official StableHLO text must not be empty")
        if not self.canonical_text.strip():
            issues.append("official StableHLO canonical text must not be empty")
        if not self.model_id:
            issues.append("official StableHLO model_id must not be empty")
        if not self.verified:
            issues.append("official StableHLO module must be verified")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "canonical_text": self.canonical_text,
            "model_id": self.model_id,
            "variant": self.variant,
            "stablehlo_version": self.stablehlo_version,
            "verified": self.verified,
            "producer": self.producer,
            "verifier": self.verifier,
            "provenance": dict(self.provenance),
        }


def _parse_verified(text: str) -> tuple[Any, str, Any]:
    Context, Module, dialect = _bindings()
    with Context() as context:
        dialect.register_dialect(context)
        try:
            module = Module.parse(text)
            module.operation.verify()
        except Exception as exc:
            raise FrontendImportError(f"official StableHLO parse/verify failed: {exc}") from exc
        return module, str(module), context


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"-?\d+", text))


def _attribute_ints(value: Any) -> tuple[int, ...]:
    """Extract integer payloads without treating MLIR element types as values."""

    text = str(value)
    dense_scalar = re.search(r"(?:array|dense)<\s*(-?\d+)\s*>", text)
    if dense_scalar is not None:
        return (int(dense_scalar.group(1)),)
    match = re.search(r"(?:array|dense)<(?:i\d+|ui\d+)?\s*:\s*([^>]+)>", text)
    if match is not None:
        text = match.group(1)
    else:
        # Scalar MLIR integers print as ``2 : i64``.  The type suffix is not
        # part of the attribute value and must not be returned as ``64``.
        scalar = re.fullmatch(r"\s*(-?\d+)\s*:\s*(?:ui|i)\d+\s*", text)
        if scalar is not None:
            text = scalar.group(1)
    return _ints(text)


def _attribute_int(value: Any) -> int:
    values = _attribute_ints(value)
    if not values:
        raise FrontendImportError(f"MLIR integer attribute has no value: {value}")
    return values[-1]


def _project_module(module: Any) -> str:
    """Project verified MLIR operations to the semantic importer's readable form."""

    functions = [
        operation
        for operation in module.operation.regions[0].blocks[0]
        if operation.operation.name == "func.func"
    ]
    if not functions:
        raise FrontendImportError("official StableHLO module has no func.func entry point")
    function = functions[0]
    block = function.regions[0].blocks[0]
    value_names: dict[Any, str] = {
        argument: f"arg{index}" for index, argument in enumerate(block.arguments)
    }
    lines = ["module {", "  func.func @main("]
    lines.append(",\n".join(f"    %{value_names[arg]}: {arg.type}" for arg in block.arguments))
    return_type = str(function.attributes["function_type"]).split("->", 1)[-1].strip().strip(")")
    lines.append(f") -> {return_type} {{")
    counter = 0
    omitted_results: set[Any] = set()

    for operation in block:
        name = operation.name
        if name == "func.return":
            if any(value in omitted_results for value in operation.operands):
                raise FrontendImportError(
                    "official StableHLO projection does not support returning a secondary operation result"
                )
            returns = [value_names[value] for value in operation.operands]
            return_types = ", ".join(str(value.type) for value in operation.operands)
            lines.append(f"    return {', '.join('%' + item for item in returns)} : {return_types}")
            continue
        if not name.startswith("stablehlo.") or not operation.results:
            continue

        if len(operation.results) > 1 and name != "stablehlo.batch_norm_training":
            raise FrontendImportError(
                f"official StableHLO projection does not support multi-result operation "
                f"'{name}' ({len(operation.results)} results); add a multi-result canonical "
                "capability before compiling this graph"
            )

        result_names: list[str] = []
        for result_index, operation_result in enumerate(operation.results):
            result_name = f"v{counter}"
            counter += 1
            value_names[operation_result] = result_name
            result_names.append(result_name)
            if result_index:
                omitted_results.add(operation_result)
        result = operation.results[0]
        result_name = result_names[0]
        if any(value in omitted_results for value in operation.operands):
            raise FrontendImportError(
                f"official StableHLO projection does not support consuming a secondary result of '{name}'"
            )
        operands = [value_names[value] for value in operation.operands]
        operand_types = [str(value.type) for value in operation.operands]
        result_type = str(result.type)

        if name == "stablehlo.broadcast_in_dim":
            dimensions_text = str(operation.attributes["broadcast_dimensions"])
            dimensions_match = re.search(r"array<i64:\s*([^>]*)>", dimensions_text)
            dimensions = _ints(
                dimensions_match.group(1) if dimensions_match else dimensions_text
            )
            lines.append(
                f"    %{result_name} = stablehlo.broadcast_in_dim %{operands[0]}, "
                f"dims = [{', '.join(map(str, dimensions))}] : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.constant":
            dense = str(operation.attributes["value"]).split(":", 1)[0].strip()
            lines.append(f"    %{result_name} = stablehlo.constant {dense} : {result_type}")
            continue
        if name == "stablehlo.reduce":
            dims_text = str(operation.attributes["dimensions"])
            dims_match = re.search(r"array<i64:\s*([^>]*)>", dims_text)
            dims = _ints(dims_match.group(1) if dims_match else dims_text)
            reducer = "add"
            if operation.regions and operation.regions[0].blocks:
                for region_op in operation.regions[0].blocks[0]:
                    if region_op.name.startswith("stablehlo.") and region_op.name != "stablehlo.return":
                        reducer = region_op.name.removeprefix("stablehlo.")
                        break
            lines.append(
                f"    %{result_name} = stablehlo.reduce %{operands[0]}, %{operands[1]} "
                f"dimensions = [{', '.join(map(str, dims))}] reducer = {reducer} : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.dot_general":
            attributes = str(operation.attributes["dot_dimension_numbers"])

            def values(attribute: str) -> str:
                match = re.search(rf"{attribute}\s*=\s*\[([^]]*)\]", attributes)
                return match.group(1).strip() if match else ""

            lines.append(
                f"    %{result_name} = stablehlo.dot_general %{operands[0]}, %{operands[1]}, "
                f"batching_dims = [{values('lhs_batching_dimensions')}] x "
                f"[{values('rhs_batching_dimensions')}], contracting_dims = "
                f"[{values('lhs_contracting_dimensions')}] x "
                f"[{values('rhs_contracting_dimensions')}] : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.convolution":
            dimension_numbers = str(operation.attributes["dimension_numbers"])
            window = str(operation.attributes["window_strides"]) if "window_strides" in operation.attributes else ""
            padding = str(operation.attributes["padding"]) if "padding" in operation.attributes else ""
            lhs_dilation = str(operation.attributes["lhs_dilation"]) if "lhs_dilation" in operation.attributes else ""
            rhs_dilation = str(operation.attributes["rhs_dilation"]) if "rhs_dilation" in operation.attributes else ""
            feature_groups = str(operation.attributes["feature_group_count"]) if "feature_group_count" in operation.attributes else "1"
            batch_groups = str(operation.attributes["batch_group_count"]) if "batch_group_count" in operation.attributes else "1"

            def array_values(value: str) -> str:
                match = re.search(r"(?:array|dense)<[^:>]*:\s*([^>]+)>", value)
                if match is not None:
                    return match.group(1).replace("[", "").replace("]", "")
                return value.split(":", 1)[0].strip()

            # StableHLO commonly prints a symmetric padding attribute as
            # ``dense<1> : tensor<2x2xi64>``.  Parse the payload only: the
            # dimensions and element type in the trailing MLIR type are not
            # padding values.  A scalar payload is expanded to one low/high
            # pair for each spatial dimension (NCHW has two spatial dims).
            dense_scalar = re.search(r"dense<\s*(-?\d+)\s*>", padding)
            if dense_scalar is not None:
                padding_values = [dense_scalar.group(1)] * 4
            else:
                payload_match = re.search(r"(?:array|dense)<[^:>]*:\s*([^>]+)>", padding)
                if payload_match is not None:
                    payload = payload_match.group(1)
                else:
                    payload = padding.split(":", 1)[0].strip()
                padding_values = [str(value) for value in _ints(payload)]
            lines.append(
                f"    %{result_name} = stablehlo.convolution %{operands[0]}, %{operands[1]} "
                f"dimension_numbers = {dimension_numbers} "
                f"window_strides = [{array_values(window)}] "
                f"padding = [{', '.join(padding_values)}] "
                f"lhs_dilation = [{array_values(lhs_dilation)}] "
                f"rhs_dilation = [{array_values(rhs_dilation)}] "
                f"feature_group_count = {feature_groups.split(':', 1)[0].strip()} "
                f"batch_group_count = {batch_groups.split(':', 1)[0].strip()} : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.transpose":
            permutation_text = str(operation.attributes["permutation"])
            permutation_match = re.search(r"array<i64:\s*([^>]*)>", permutation_text)
            permutation = _ints(permutation_match.group(1) if permutation_match else permutation_text)
            lines.append(
                f"    %{result_name} = stablehlo.transpose %{operands[0]}, "
                f"dimensions = [{', '.join(map(str, permutation))}] : {result_type}"
            )
            continue
        if name == "stablehlo.slice":
            starts = _attribute_ints(operation.attributes["start_indices"])
            limits = _attribute_ints(operation.attributes["limit_indices"])
            strides_attribute = (
                operation.attributes["strides"]
                if "strides" in operation.attributes
                else ""
            )
            strides = _attribute_ints(strides_attribute)
            lines.append(
                f"    %{result_name} = stablehlo.slice %{operands[0]} "
                f"starts = [{', '.join(map(str, starts))}] "
                f"limits = [{', '.join(map(str, limits))}] "
                f"strides = [{', '.join(map(str, strides))}] : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.concatenate":
            dimension = _attribute_int(operation.attributes["dimension"])
            lines.append(
                f"    %{result_name} = stablehlo.concatenate "
                f"{', '.join('%' + item for item in operands)} dim = {dimension} : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.batch_norm_training":
            feature_index = str(operation.attributes["feature_index"]).split(":", 1)[0].strip()
            epsilon = str(operation.attributes["epsilon"]).split(":", 1)[0].strip()
            lines.append(
                f"    %{result_name} = stablehlo.batch_norm_training "
                f"{', '.join('%' + item for item in operands)} "
                f"feature_index = {feature_index} epsilon = {epsilon} : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.batch_norm_inference":
            feature_index = str(operation.attributes["feature_index"]).split(":", 1)[0].strip()
            epsilon = str(operation.attributes["epsilon"]).split(":", 1)[0].strip()
            lines.append(
                f"    %{result_name} = stablehlo.batch_norm_inference "
                f"{', '.join('%' + item for item in operands)} "
                f"feature_index = {feature_index} epsilon = {epsilon} : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        if name == "stablehlo.reduce_window":
            window_dimensions = _attribute_ints(operation.attributes["window_dimensions"])
            window_strides = _attribute_ints(operation.attributes["window_strides"])
            padding_values = _attribute_ints(operation.attributes["padding"])
            base_dilations = _attribute_ints(
                operation.attributes["base_dilations"]
                if "base_dilations" in operation.attributes
                else ""
            )
            window_dilations = _attribute_ints(
                operation.attributes["window_dilations"]
                if "window_dilations" in operation.attributes
                else ""
            )
            if len(padding_values) == 1:
                padding_values *= 8
            reducer = "add"
            if operation.regions and operation.regions[0].blocks:
                for region_op in operation.regions[0].blocks[0]:
                    if region_op.name.startswith("stablehlo.") and region_op.name != "stablehlo.return":
                        reducer = region_op.name.removeprefix("stablehlo.")
                        break
            lines.append(
                f"    %{result_name} = stablehlo.reduce_window %{operands[0]}, %{operands[1]} "
                f"window_dimensions = [{', '.join(map(str, window_dimensions))}] "
                f"window_strides = [{', '.join(map(str, window_strides))}] "
                f"padding = [{', '.join(map(str, padding_values))}] "
                f"base_dilations = [{', '.join(map(str, base_dilations))}] "
                f"window_dilations = [{', '.join(map(str, window_dilations))}] "
                f"reducer = {reducer} : ({', '.join(operand_types)}) -> {result_type}"
            )
            continue
        target = name.removeprefix("stablehlo.")
        lines.append(
            f"    %{result_name} = stablehlo.{target} "
            f"{', '.join('%' + item for item in operands)} : {result_type}"
        )

    lines.extend(("  }", "}"))
    return "\n".join(lines) + "\n"


class OfficialStableHLOAdapter:
    """Verify and import Torch-XLA StableHLO through official MLIR bindings."""

    kind = FrontendKind.STABLEHLO

    @classmethod
    def parse_text(
        cls,
        text: str,
        *,
        model_id: str = "stablehlo_model",
        variant: str = "stablehlo-torch-xla-v1",
    ) -> OfficialStableHLOModule:
        if not isinstance(text, str) or not text.strip():
            raise FrontendImportError("official StableHLO text must be a non-empty string")
        _, canonical, _ = _parse_verified(text)
        return OfficialStableHLOModule(
            text=text,
            canonical_text=canonical,
            model_id=model_id,
            variant=variant,
            stablehlo_version=official_stablehlo_version(),
            provenance={"source": "official-stablehlo-text", "verifier": "mlir.ir.Operation.verify"},
        )

    @classmethod
    def import_text(
        cls,
        text: str,
        *,
        model_id: str = "stablehlo_model",
        variant: str = "stablehlo-torch-xla-v1",
        shape_environment: Mapping[str, int] | None = None,
    ) -> FrontendImport:
        module_obj, canonical, _ = _parse_verified(text)
        projected = _project_module(module_obj)
        imported = StableHLOAdapter.from_text(
            projected,
            model_id=model_id,
            variant=variant,
            shape_environment=shape_environment,
        )
        return FrontendImport(
            graph=imported.graph,
            model_id=imported.model_id,
            variant=imported.variant,
            shape_environment=imported.shape_environment,
            frontend=imported.frontend,
            provenance={
                **dict(imported.provenance),
                "source": "official-stablehlo-bindings",
                "canonical_assembly": canonical,
                "verifier": "mlir.ir.Operation.verify",
                "stablehlo_version": official_stablehlo_version(),
            },
            family=imported.family,
        )


__all__ = [
    "OfficialStableHLOAdapter",
    "OfficialStableHLOModule",
    "official_stablehlo_available",
    "official_stablehlo_version",
]
