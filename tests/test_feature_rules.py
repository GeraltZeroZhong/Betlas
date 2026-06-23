from __future__ import annotations

import pytest

from betlas.features.extract import assert_no_diagnostic_label_leakage
from betlas.features.geometry import entropy
from betlas.features.rules import GRAMMAR_RULE_INPUT_COLUMNS, grammar_rule_scores


def _complete_rule_inputs(**updates: float) -> dict[str, float]:
    row = {column: 0.0 for column in GRAMMAR_RULE_INPUT_COLUMNS}
    row["betlas_parse_ok"] = 1.0
    row["betlas_axis_best_slice_count"] = 1.0
    row.update(updates)
    return row


def test_rule_scores_emit_all_labels() -> None:
    scores = grammar_rule_scores(
        _complete_rule_inputs(
            betlas_axis_best_slice_coverage_median=0.9,
            betlas_axis_best_angular_coverage=0.9,
            betlas_axis_best_largest_gap_fraction=0.1,
            betlas_beta_strand_count=8,
            betlas_helix_count=8,
            betlas_beta_alpha_alternation_fraction=0.8,
            betlas_alpha_shell_radial_delta=4.0,
        )
    )
    assert set(scores) == {
        "beta_barrel",
        "beta_prism",
        "beta_propeller",
        "jelly_roll",
        "beta_solenoid",
        "beta_sandwich",
        "tim_like_beta_alpha_barrel",
    }


def test_diagnostic_columns_cannot_populate_labels() -> None:
    assert_no_diagnostic_label_leakage({"fold_label_final": "beta_barrel", "betlas_top_fold": "beta_barrel"})
    try:
        assert_no_diagnostic_label_leakage({"fold_label_final": "betlas_top_fold"})
    except ValueError:
        return
    raise AssertionError("expected diagnostic label leakage to be rejected")


def test_entropy_is_bounded_for_degenerate_inputs() -> None:
    assert entropy(["same", "same"]) == 0.0
    assert 0.0 <= entropy(["a", "b", "b"]) <= 1.0


def test_rule_scores_default_strict_rejects_incomplete_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="requires betlas_parse_ok"):
        grammar_rule_scores({"betlas_axis_best_slice_coverage_median": 0.5})

    row = _complete_rule_inputs(betlas_axis_best_slice_coverage_median=float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        grammar_rule_scores(row)


def test_rule_scores_default_strict_rejects_zero_informative_slices() -> None:
    row = _complete_rule_inputs(betlas_axis_best_slice_count=0.0)

    with pytest.raises(ValueError, match="no informative slices"):
        grammar_rule_scores(row)


def test_rule_scores_compatibility_mode_treats_nan_as_missing() -> None:
    scores = grammar_rule_scores({"betlas_axis_best_slice_coverage_median": float("nan")}, strict=False)
    assert all(value == value for value in scores.values())
