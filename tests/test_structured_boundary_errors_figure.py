from __future__ import annotations

import math

import pandas as pd
import pytest

from figures.scripts.generate_structured_boundary_errors_figure import (
    boundary_enrichment,
    misclassification_flows,
    normalize_confusion,
    odds_ratio_ci,
    validate_record_alignment,
)


def test_validate_record_alignment_accepts_one_to_one_subset() -> None:
    predictions = pd.DataFrame({"record_id": ["a", "b"], "true_label": ["x", "y"]})
    diagnostics = pd.DataFrame({"record_id": ["a", "b", "c"]})

    validate_record_alignment(predictions, diagnostics)


def test_validate_record_alignment_rejects_duplicates_and_missing() -> None:
    with pytest.raises(ValueError, match="prediction record_id values are duplicated"):
        validate_record_alignment(pd.DataFrame({"record_id": ["a", "a"]}), pd.DataFrame({"record_id": ["a"]}))

    with pytest.raises(ValueError, match="diagnostics record_id values are duplicated"):
        validate_record_alignment(pd.DataFrame({"record_id": ["a"]}), pd.DataFrame({"record_id": ["a", "a"]}))

    with pytest.raises(ValueError, match="diagnostics are missing"):
        validate_record_alignment(pd.DataFrame({"record_id": ["a", "b"]}), pd.DataFrame({"record_id": ["a"]}))


def test_normalize_confusion_preserves_zero_rows() -> None:
    cm = pd.DataFrame([[2, 2], [0, 0]], index=["a", "b"], columns=["a", "b"])

    observed = normalize_confusion(cm)

    assert observed.loc["a", "a"] == 0.5
    assert observed.loc["a", "b"] == 0.5
    assert observed.loc["b", "a"] == 0.0
    assert observed.loc["b", "b"] == 0.0


def test_misclassification_flows_thresholds_off_diagonal_errors() -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "record_id": f"a{index}",
                "true_label": "beta_barrel",
                "pred_label": "beta_sandwich",
                "error": True,
                "boundary_region": 1,
                "mixed_topology": 0,
                "cz_topology_ambiguity_score": 0.4,
            }
        )
    rows.append(
        {
            "record_id": "b0",
            "true_label": "jelly_roll",
            "pred_label": "beta_barrel",
            "error": True,
            "boundary_region": 0,
            "mixed_topology": 1,
            "cz_topology_ambiguity_score": 0.2,
        }
    )
    rows.append(
        {
            "record_id": "c0",
            "true_label": "beta_barrel",
            "pred_label": "beta_barrel",
            "error": False,
            "boundary_region": 0,
            "mixed_topology": 0,
            "cz_topology_ambiguity_score": 0.1,
        }
    )

    flows = misclassification_flows(pd.DataFrame(rows), min_count=2)

    included = flows[flows["included"]]
    assert included[["true_label", "pred_label", "n"]].to_dict("records") == [
        {"true_label": "beta_barrel", "pred_label": "beta_sandwich", "n": 4}
    ]
    assert int(flows["n"].sum()) == 5


def test_odds_ratio_ci_matches_hand_calculation() -> None:
    observed = odds_ratio_ci(30, 10, 40, 60)

    assert math.isclose(observed["odds_ratio"], 4.5)
    assert math.isclose(observed["log2_odds_ratio"], math.log2(4.5))
    assert observed["ci_low"] < observed["odds_ratio"] < observed["ci_high"]


def test_boundary_enrichment_uses_correct_predictions_as_background() -> None:
    joined = pd.DataFrame(
        [
            {"record_id": "e1", "true_label": "beta_barrel", "pred_label": "beta_sandwich", "error": True, "boundary_region": 1},
            {"record_id": "e2", "true_label": "beta_barrel", "pred_label": "beta_sandwich", "error": True, "boundary_region": 1},
            {"record_id": "e3", "true_label": "beta_barrel", "pred_label": "beta_sandwich", "error": True, "boundary_region": 0},
            {"record_id": "c1", "true_label": "beta_barrel", "pred_label": "beta_barrel", "error": False, "boundary_region": 1},
            {"record_id": "c2", "true_label": "beta_barrel", "pred_label": "beta_barrel", "error": False, "boundary_region": 0},
            {"record_id": "c3", "true_label": "beta_barrel", "pred_label": "beta_barrel", "error": False, "boundary_region": 0},
            {"record_id": "c4", "true_label": "beta_barrel", "pred_label": "beta_barrel", "error": False, "boundary_region": 0},
        ]
    )
    flows = misclassification_flows(joined.assign(mixed_topology=0, cz_topology_ambiguity_score=0.0), min_count=1)

    enrichment = boundary_enrichment(joined, flows)
    all_errors = enrichment[enrichment["label"] == "All errors"].iloc[0]

    assert all_errors["background_correct_boundary_n"] == 1
    assert all_errors["background_correct_n"] == 4
    assert math.isclose(all_errors["odds_ratio"], (2 / 1) / (1 / 3))
