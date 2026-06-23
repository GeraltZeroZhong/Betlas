from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

BETLAS_PREFIX = "betlas_"
_COMPATIBILITY_PREFIX = "cz_"


class SchemaCompatibilityWarning(UserWarning):
    """Warning raised when compatibility feature columns are adapted."""


def is_betlas_column(name: str) -> bool:
    return str(name).startswith(BETLAS_PREFIX)


def _is_compatibility_column(name: str) -> bool:
    return str(name).startswith(_COMPATIBILITY_PREFIX)


def canonical_feature_name(name: str) -> str:
    """Return the canonical Betlas feature name for a supported input name."""

    text = str(name)
    if text.startswith(_COMPATIBILITY_PREFIX):
        return f"{BETLAS_PREFIX}{text[len(_COMPATIBILITY_PREFIX):]}"
    return text


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _scalars_conflict(left: Any, right: Any) -> bool:
    left_missing = _is_missing_scalar(left)
    right_missing = _is_missing_scalar(right)
    if left_missing and right_missing:
        return False
    if left_missing or right_missing:
        return False
    try:
        return not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left) != str(right)


def _series_conflict_mask(left: pd.Series, right: pd.Series) -> pd.Series:
    left_text = left.astype(str)
    right_text = right.astype(str)
    left_missing = left.isna() | left_text.eq("")
    right_missing = right.isna() | right_text.eq("")
    comparable = ~(left_missing | right_missing)

    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_numeric = comparable & left_num.notna() & right_num.notna()
    numeric_conflicts = both_numeric & ((left_num - right_num).abs() > 1e-12)
    text_conflicts = comparable & ~both_numeric & left_text.ne(right_text)
    return numeric_conflicts | text_conflicts


def normalize_feature_mapping(
    values: Mapping[str, Any],
    *,
    strict: bool = False,
    keep_compatibility_columns: bool = False,
) -> dict[str, Any]:
    """Return a copy with supported input feature keys mapped to Betlas names."""

    normalized = dict(values)
    for source_key in [key for key in values if _is_compatibility_column(str(key))]:
        betlas_key = canonical_feature_name(str(source_key))
        if betlas_key in normalized:
            if _scalars_conflict(normalized[betlas_key], values[source_key]):
                message = (
                    "compatibility feature column conflicts with the canonical Betlas "
                    f"column {betlas_key!r}; using the canonical column"
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, SchemaCompatibilityWarning, stacklevel=2)
            if not keep_compatibility_columns:
                normalized.pop(source_key, None)
            continue
        if keep_compatibility_columns:
            normalized[betlas_key] = values[source_key]
        else:
            normalized[betlas_key] = normalized.pop(source_key)
    return normalized


def normalize_feature_columns(
    frame: pd.DataFrame,
    *,
    strict: bool = False,
    keep_compatibility_columns: bool = False,
) -> pd.DataFrame:
    """Return a copy with supported input feature columns mapped to Betlas names."""

    normalized = frame.copy()
    for source_column in [column for column in frame.columns if _is_compatibility_column(str(column))]:
        betlas_column = canonical_feature_name(str(source_column))
        if betlas_column in normalized.columns:
            conflicts = _series_conflict_mask(normalized[betlas_column], normalized[source_column])
            conflict_count = int(conflicts.sum())
            if conflict_count:
                message = (
                    "compatibility feature column conflicts with the canonical Betlas "
                    f"column {betlas_column!r} in {conflict_count} row(s); "
                    "using the canonical column"
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, SchemaCompatibilityWarning, stacklevel=2)
            if not keep_compatibility_columns:
                normalized = normalized.drop(columns=[source_column])
            continue
        if keep_compatibility_columns:
            normalized[betlas_column] = normalized[source_column]
        else:
            normalized = normalized.rename(columns={source_column: betlas_column})
    return normalized


__all__ = [
    "BETLAS_PREFIX",
    "SchemaCompatibilityWarning",
    "canonical_feature_name",
    "is_betlas_column",
    "normalize_feature_columns",
    "normalize_feature_mapping",
]
