from __future__ import annotations

import math

from ..constants import FOLD_LABELS
from ..schema import _is_compatibility_column, normalize_feature_mapping

GRAMMAR_RULE_INPUT_COLUMNS = (
    "betlas_axis_best_slice_coverage_median",
    "betlas_axis_best_angular_coverage",
    "betlas_axis_best_largest_gap_fraction",
    "betlas_pca_elongation",
    "betlas_pca_flatness",
    "betlas_sheet_count",
    "betlas_sheet_face_count",
    "betlas_beta_strand_count",
    "betlas_beta_run_count",
    "betlas_helix_count",
    "betlas_alpha_shell_radial_delta",
    "betlas_beta_alpha_alternation_fraction",
    "betlas_angular_fft_k3_8_max",
    "betlas_angular_fft_k3_8_best_k",
    "betlas_axis_periodicity_score",
    "betlas_sheet_angular_span_max",
    "betlas_barrel_wall_continuity_score",
    "betlas_sheet_pair_bilayer_score",
    "betlas_sandwich_lobe_guard_score",
    "betlas_angular_fft_k2_dominance",
    "betlas_sheet_pair_top2_fraction",
    "betlas_sheet_pair_size_balance",
    "betlas_sheet_seq_greek_key_proxy",
    "betlas_jelly_roll_order_nonlocal_score",
    "betlas_sheet_seq_top2_interleave_score",
    "betlas_sheet_seq_top2_order_displacement",
    "betlas_z_continuity_fraction",
)


def _f(features: dict[str, float | int | str], key: str, default: float = 0.0) -> float:
    try:
        value = features.get(key, default)
        if value in {"", None}:
            return default
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except (TypeError, ValueError):
        return default


def _bell(value: float, center: float, width: float) -> float:
    if width <= 0:
        return 0.0
    return math.exp(-((value - center) / width) ** 2)


def _format_rule_input_problems(problems: dict[str, str]) -> str:
    details = ", ".join(f"{name}={reason}" for name, reason in sorted(problems.items())[:8])
    suffix = " ..." if len(problems) > 8 else ""
    return f"{details}{suffix}"


def _parse_ok_value(features: dict[str, object]) -> int | None:
    if "betlas_parse_ok" not in features:
        return None
    try:
        value = float(features.get("betlas_parse_ok", 0))
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(value):
        return 0
    return 1 if value == 1.0 else 0


def _noninformative_slice_count(features: dict[str, object]) -> bool:
    if "betlas_axis_best_slice_count" not in features:
        return False
    try:
        value = float(features.get("betlas_axis_best_slice_count", 0))
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(value)) or value <= 0.0


def _rule_scores_permissive(features: dict[str, float | int | str]) -> dict[str, float]:
    features = normalize_feature_mapping(features)
    closure = _f(features, "betlas_axis_best_slice_coverage_median")
    all_coverage = _f(features, "betlas_axis_best_angular_coverage")
    gap = _f(features, "betlas_axis_best_largest_gap_fraction")
    elongation = _f(features, "betlas_pca_elongation")
    flatness = _f(features, "betlas_pca_flatness")
    sheet_count = _f(features, "betlas_sheet_count")
    face_count = _f(features, "betlas_sheet_face_count")
    strand_count = _f(features, "betlas_beta_strand_count")
    run_count = _f(features, "betlas_beta_run_count", strand_count)
    helix_count = _f(features, "betlas_helix_count")
    alpha_shell = _f(features, "betlas_alpha_shell_radial_delta")
    alternation = _f(features, "betlas_beta_alpha_alternation_fraction")
    prop_fft = _f(features, "betlas_angular_fft_k3_8_max")
    best_k = _f(features, "betlas_angular_fft_k3_8_best_k")
    solenoid = _f(features, "betlas_axis_periodicity_score")
    sheet_span = _f(features, "betlas_sheet_angular_span_max")
    barrel_wall = _f(features, "betlas_barrel_wall_continuity_score")
    bilayer = _f(features, "betlas_sheet_pair_bilayer_score")
    lobe_guard = _f(features, "betlas_sandwich_lobe_guard_score")
    k2_dominance = _f(features, "betlas_angular_fft_k2_dominance")
    top2_fraction = _f(features, "betlas_sheet_pair_top2_fraction")
    top2_balance = _f(features, "betlas_sheet_pair_size_balance")
    greek_proxy = _f(features, "betlas_sheet_seq_greek_key_proxy")
    nonlocal_order = _f(features, "betlas_jelly_roll_order_nonlocal_score")
    interleave = _f(features, "betlas_sheet_seq_top2_interleave_score")
    order_displacement = _f(features, "betlas_sheet_seq_top2_order_displacement")
    two_sheet = _bell(sheet_count, 2.0, 1.2)
    three_face = _bell(face_count, 3.0, 1.0)
    eight_strands = _bell(strand_count, 8.0, 2.2)
    eight_runs = _bell(run_count, 8.0, 2.4)

    scores = {
        "beta_barrel": 1.1 * closure
        + 1.0 * all_coverage
        + 0.7 * barrel_wall
        + 0.4 * (1.0 - gap)
        + 0.2 * sheet_span
        - 0.35 * lobe_guard
        - 0.20 * k2_dominance,
        "beta_prism": 1.2 * three_face + 0.7 * _bell(best_k, 3.0, 1.0) + 0.4 * flatness + 0.3 * all_coverage,
        "beta_propeller": 1.1 * prop_fft + 0.9 * _bell(best_k, 6.0, 2.5) + 0.5 * all_coverage + 0.3 * flatness,
        "jelly_roll": 0.8 * eight_strands
        + 0.9 * eight_runs
        + 0.6 * two_sheet
        + 0.5 * top2_fraction
        + 0.6 * greek_proxy
        + 0.7 * nonlocal_order
        + 0.5 * interleave
        + 0.4 * order_displacement
        + 0.3 * (1.0 - closure),
        "beta_solenoid": 1.3 * solenoid + 1.0 * min(1.5, elongation / 3.0) + 0.5 * _f(features, "betlas_z_continuity_fraction"),
        "beta_sandwich": 0.8 * two_sheet
        + 1.0 * bilayer
        + 0.7 * lobe_guard
        + 0.4 * top2_balance
        + 0.5 * (1.0 - closure)
        + 0.4 * flatness
        + 0.2 * min(1.0, face_count / 3.0),
        "tim_like_beta_alpha_barrel": 1.0 * _bell(strand_count, 8.0, 2.5)
        + 0.9 * min(1.0, helix_count / 8.0)
        + 0.8 * alternation
        + 0.5 * max(0.0, alpha_shell / 6.0)
        + 0.4 * closure,
    }
    return {label: float(scores.get(label, 0.0)) for label in FOLD_LABELS}


def grammar_rule_scores(
    features: dict[str, float | int | str],
    *,
    strict: bool = True,
) -> dict[str, float]:
    """Return transparent Betlas fold-rule scores.

    Strict validation is the public default. Pass ``strict=False`` only for
    compatibility or exploratory scoring of incomplete feature dictionaries;
    that path preserves the historical zero-fill behavior.
    """

    if strict and any(_is_compatibility_column(str(key)) for key in features):
        raise ValueError(
            "grammar_rule_scores requires canonical betlas_* columns by default; "
            "pass strict=False only for compatibility scoring of older feature names"
        )
    normalized = normalize_feature_mapping(features)
    parse_ok = _parse_ok_value(normalized)
    if strict:
        if parse_ok is None:
            raise ValueError(
                "grammar_rule_scores strict validation requires betlas_parse_ok=1; "
                "pass strict=False only for compatibility scoring of older feature tables"
            )
        if parse_ok != 1:
            raise ValueError(
                "grammar_rule_scores strict validation refused parse-failed feature row: "
                "betlas_parse_ok is not 1"
            )
        if _noninformative_slice_count(normalized):
            raise ValueError(
                "grammar_rule_scores strict validation refused feature row with no informative slices: "
                "betlas_axis_best_slice_count must be > 0"
            )
        problems = missing_or_invalid_rule_inputs(normalized)
        if problems:
            raise ValueError(
                "grammar_rule_scores strict validation failed; required Betlas rule-score "
                f"inputs are missing or invalid: {_format_rule_input_problems(problems)}"
            )
    return _rule_scores_permissive(normalized)


def missing_or_invalid_rule_inputs(features: dict[str, object]) -> dict[str, str]:
    """Return rule-score input columns that are absent, blank, or nonnumeric."""

    normalized = normalize_feature_mapping(features)
    problems: dict[str, str] = {}
    for column in GRAMMAR_RULE_INPUT_COLUMNS:
        if column not in normalized:
            problems[column] = "missing"
            continue
        value = normalized[column]
        if value in {"", None}:
            problems[column] = "blank"
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            problems[column] = "nonnumeric"
            continue
        if not math.isfinite(numeric):
            problems[column] = "nonfinite"
    return problems


__all__ = [
    "GRAMMAR_RULE_INPUT_COLUMNS",
    "grammar_rule_scores",
    "missing_or_invalid_rule_inputs",
]
