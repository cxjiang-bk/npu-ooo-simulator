"""Shared dtype names and storage widths for frontend, compiler, and runtime."""

from __future__ import annotations

from typing import Any


# Keep source spellings visible while accepting names emitted by StableHLO and
# PyTorch.  The canonical family is used for capability and byte-width lookup.
_CANONICAL: dict[str, str] = {
    "bool": "bool",
    "pred": "bool",
    "i1": "bool",
    "ui1": "bool",
    "int8": "i8",
    "i8": "i8",
    "uint8": "ui8",
    "ui8": "ui8",
    "int16": "i16",
    "i16": "i16",
    "uint16": "ui16",
    "ui16": "ui16",
    "int32": "i32",
    "i32": "i32",
    "uint32": "ui32",
    "ui32": "ui32",
    "int64": "i64",
    "i64": "i64",
    "uint64": "ui64",
    "ui64": "ui64",
    "float16": "f16",
    "fp16": "f16",
    "f16": "f16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float32": "f32",
    "fp32": "f32",
    "f32": "f32",
    "float64": "f64",
    "fp64": "f64",
    "f64": "f64",
}

_BYTES: dict[str, int] = {
    "bool": 1,
    "i8": 1,
    "ui8": 1,
    "i16": 2,
    "ui16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "ui32": 4,
    "f32": 4,
    "i64": 8,
    "ui64": 8,
    "f64": 8,
}


def normalize_dtype(dtype: Any) -> str:
    """Return a stable lowercase source spelling."""

    return str(dtype).lower().replace("torch.", "")


def canonical_dtype(dtype: Any) -> str | None:
    """Return the capability/storage family for a dtype alias."""

    return _CANONICAL.get(normalize_dtype(dtype))


def is_known_dtype(dtype: Any) -> bool:
    return canonical_dtype(dtype) is not None


def known_dtype_names() -> frozenset[str]:
    return frozenset(_CANONICAL)


def dtype_bytes(dtype: Any, *, default: int | None = None) -> int:
    """Return storage width, optionally using an explicit unknown-type fallback."""

    canonical = canonical_dtype(dtype)
    if canonical is not None:
        return _BYTES[canonical]
    if default is not None:
        return default
    raise ValueError(f"unsupported dtype '{dtype}'")


__all__ = [
    "canonical_dtype",
    "dtype_bytes",
    "is_known_dtype",
    "known_dtype_names",
    "normalize_dtype",
]
