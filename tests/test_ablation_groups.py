from __future__ import annotations

import pandas as pd
import pytest

from betlas.constants import FOLD_LABELS
from betlas.ml.ablations import feature_group_for, run_ablation_suite
from betlas.ml.benchmark import (
    RuleScoreClassifier,
    _align_predict_proba,
    load_benchmark_config,
    run_grouped_benchmark,
)
from betlas.ml.splits import make_grouped_splits


def test_new_boundary_features_are_grouped() -> None:
    assert feature_group_for("betlas_top2_sheet_order_nonlocal_fraction") == "sheet_sequence_topology"
    assert feature_group_for("betlas_jelly_roll_order_nonlocal_score") == "sheet_sequence_topology"
    assert feature_group_for("betlas_contact8_seq_gap_mean") == "contact_graph"
    assert feature_group_for("betlas_sheet_pair_bilayer_score") == "sheet_pair_packing"


def test_rule_score_classifier_requires_rule_score_columns() -> None:
    classifier = RuleScoreClassifier(labels=FOLD_LABELS)

    with pytest.raises(ValueError, match="requires precomputed Betlas rule-score columns"):
        classifier.predict_proba(pd.DataFrame({"betlas_axis_best_angular_coverage": [0.5]}))


def test_default_benchmark_config_is_raw_geometry_sklearn_cold_start() -> None:
    config = load_benchmark_config()

    assert config["features"]["set"] == "raw_geometry"
    assert "grammar_rules" not in config["models"]["include"]
    assert "xgboost_tuned" not in config["models"]["include"]


def test_benchmark_preflight_fails_fast_for_missing_rule_scores(tmp_path) -> None:
    features = tmp_path / "features.csv"
    out_dir = tmp_path / "benchmark"
    pd.DataFrame(
        [
            {
                "record_id": "r1",
                "pdb_id": "p001",
                "domain_id": "d1",
                "fold_label_final": FOLD_LABELS[0],
                "betlas_parse_ok": 1,
                "betlas_axis_best_angular_coverage": 0.5,
            }
        ]
    ).to_csv(features, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        run_grouped_benchmark(
            features,
            out_dir,
            config={"models": {"include": ["grammar_rules"]}},
        )

    preflight = pd.read_json(out_dir / "benchmark_preflight.json", typ="series")
    assert preflight["feature_set"] == "raw_geometry"
    assert preflight["rule_score_columns_present"] is False
    assert preflight["group_columns_priority"] == ["cath_s35_cluster_id", "cath_homology_code", "pdb_id"]


def test_grouped_splits_fail_when_test_folds_cannot_cover_observed_classes() -> None:
    x = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
    y = [0, 1, 2]
    groups = ["g0", "g1", "g2"]

    with pytest.raises(ValueError, match="complete class coverage"):
        make_grouped_splits(x, y, groups, n_splits=3, random_state=13)


def test_benchmark_writes_preflight_when_grouped_splits_fail(tmp_path) -> None:
    features = tmp_path / "features.csv"
    out_dir = tmp_path / "benchmark"
    pd.DataFrame(
        [
            {
                "record_id": f"r{idx}",
                "pdb_id": f"p{idx}",
                "domain_id": f"d{idx}",
                "cath_s35_cluster_id": f"g{idx}",
                "fold_label_final": label,
                "betlas_parse_ok": 1,
                "betlas_axis_best_angular_coverage": float(idx + 1),
            }
            for idx, label in enumerate(FOLD_LABELS[:3])
        ]
    ).to_csv(features, index=False)

    with pytest.raises(ValueError, match="complete class coverage"):
        run_grouped_benchmark(features, out_dir, n_splits=3)

    preflight = pd.read_json(out_dir / "benchmark_preflight.json", typ="series")
    assert preflight["effective_split_strategy"] == "split_failed"
    assert "complete class coverage" in preflight["split_error"]


def test_benchmark_and_ablation_write_failed_preflight_for_invalid_feature_schema(tmp_path) -> None:
    features = tmp_path / "invalid.csv"
    pd.DataFrame(
        [
            {
                "record_id": "r1",
                "pdb_id": "p001",
                "domain_id": "d1",
                "fold_label_final": FOLD_LABELS[0],
                "betlas_parse_ok": 1,
            }
        ]
    ).to_csv(features, index=False)

    benchmark_out = tmp_path / "benchmark_invalid"
    with pytest.raises(ValueError, match="no numeric betlas"):
        run_grouped_benchmark(features, benchmark_out)
    benchmark_preflight = pd.read_json(benchmark_out / "benchmark_preflight.json", typ="series")
    assert benchmark_preflight["status"] == "failed"
    assert benchmark_preflight["failure_stage"] == "feature_schema"

    ablation_out = tmp_path / "ablation_invalid"
    with pytest.raises(ValueError, match="no raw geometry"):
        run_ablation_suite(features, ablation_out)
    ablation_preflight = pd.read_json(ablation_out / "ablation_preflight.json", typ="series")
    assert ablation_preflight["status"] == "failed"
    assert ablation_preflight["failure_stage"] == "feature_schema"


def test_ablation_writes_preflight_when_grouped_splits_fail(tmp_path) -> None:
    features = tmp_path / "features.csv"
    out_dir = tmp_path / "ablation"
    pd.DataFrame(
        [
            {
                "record_id": f"r{idx}",
                "pdb_id": f"p{idx}",
                "domain_id": f"d{idx}",
                "cath_s35_cluster_id": f"g{idx}",
                "fold_label_final": label,
                "betlas_parse_ok": 1,
                "betlas_axis_best_angular_coverage": float(idx + 1),
            }
            for idx, label in enumerate(FOLD_LABELS[:3])
        ]
    ).to_csv(features, index=False)

    with pytest.raises(ValueError, match="complete class coverage"):
        run_ablation_suite(features, out_dir, n_splits=3)

    preflight = pd.read_json(out_dir / "ablation_preflight.json", typ="series")
    assert preflight["effective_split_strategy"] == "split_failed"
    assert "complete class coverage" in preflight["split_error"]


def test_predict_proba_alignment_maps_estimator_classes_to_global_columns() -> None:
    class Estimator:
        classes_ = [0, 2]

    aligned, warnings = _align_predict_proba(
        pd.DataFrame([[0.75, 0.25]]).to_numpy(),
        Estimator(),
        labels=["a", "b", "c"],
        model_name="unit",
        fold=1,
    )

    assert aligned.tolist() == [[0.75, 0.0, 0.25]]
    assert any("omitted classes" in warning for warning in warnings)


def test_benchmark_writes_preflight_before_xgboost_missing_error(tmp_path, monkeypatch) -> None:
    features = tmp_path / "features.csv"
    out_dir = tmp_path / "benchmark"
    rows = []
    for group_id in range(2):
        for label in FOLD_LABELS:
            rows.append(
                {
                    "record_id": f"{label}_{group_id}",
                    "pdb_id": f"p{group_id}{label[:2]}",
                    "domain_id": f"{label}_{group_id}",
                    "cath_s35_cluster_id": f"{label}_g{group_id}",
                    "fold_label_final": label,
                    "betlas_parse_ok": 1,
                    "betlas_axis_best_angular_coverage": float(group_id + 1),
                }
            )
    pd.DataFrame(rows).to_csv(features, index=False)
    monkeypatch.setattr("betlas.ml.benchmark._xgboost_available", lambda: False)

    with pytest.raises(ValueError, match="xgboost_tuned"):
        run_grouped_benchmark(
            features,
            out_dir,
            n_splits=2,
            config={"models": {"include": ["xgboost_tuned"], "allow_model_skip": False}},
        )

    preflight = pd.read_json(out_dir / "benchmark_preflight.json", typ="series")
    assert preflight["model_dependency_status"]["xgboost_tuned"].startswith("unavailable")
