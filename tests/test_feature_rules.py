from __future__ import annotations

from betlas.features.extract import assert_no_diagnostic_label_leakage
from betlas.features.geometry import entropy
from betlas.features.rules import grammar_rule_scores


def test_rule_scores_emit_all_labels() -> None:
    scores = grammar_rule_scores(
        {
            "cz_axis_best_slice_coverage_median": 0.9,
            "cz_axis_best_angular_coverage": 0.9,
            "cz_axis_best_largest_gap_fraction": 0.1,
            "cz_beta_strand_count": 8,
            "cz_helix_count": 8,
            "cz_beta_alpha_alternation_fraction": 0.8,
            "cz_alpha_shell_radial_delta": 4.0,
        }
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
    assert_no_diagnostic_label_leakage({"fold_label_final": "beta_barrel", "cz_top_fold": "beta_barrel"})
    try:
        assert_no_diagnostic_label_leakage({"fold_label_final": "cz_top_fold"})
    except ValueError:
        return
    raise AssertionError("expected diagnostic label leakage to be rejected")


def test_entropy_is_bounded_for_degenerate_inputs() -> None:
    assert entropy(["same", "same"]) == 0.0
    assert 0.0 <= entropy(["a", "b", "b"]) <= 1.0


def test_rule_scores_treat_nan_as_missing() -> None:
    scores = grammar_rule_scores({"cz_axis_best_slice_coverage_median": float("nan")})

    assert all(value == value for value in scores.values())
