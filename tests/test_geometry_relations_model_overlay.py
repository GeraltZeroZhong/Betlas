from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betlas.geometry_relations.model_overlay import SYSTEMS, run_model_overlay


def _source(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    root = tmp_path / "source"
    geometry = pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4"],
        "A": ["2.40", "2.60", "2.10", "2.60"],
        "T": ["2.40.10", "2.60.120", "2.10.10", "2.60.999"],
        "H": ["2.40.10.1", "2.60.120.1", "2.10.10.1", "2.60.999.1"],
        "role": ["discovery", "confirmation", "confirmation", "confirmation"],
        "label_overlap": [False, pd.NA, True, True],
        "label_overlap_fraction": [0.0, np.nan, 0.9, 0.5],
    })
    for task in ("architecture", "topology"):
        for system in SYSTEMS:
            directory = root / "classification" / "assessment" / task / system
            directory.mkdir(parents=True)
            if task == "architecture":
                classes = np.array(["2.40", "2.60", "2.10"])
                targets = geometry.A.tolist()
                probabilities = np.array([[0.7, 0.2, 0.1], [np.nan] * 3, [0.0, 0.0, 1.0], [0.2, 0.45, 0.35]])
            else:
                classes = np.array(["2.40.10", "2.60.120", "2.60.40", "2.10.10"])
                targets = geometry["T"].tolist()
                probabilities = np.array([[0.5, 0.25, 0.15, 0.1], [np.nan] * 4, [0.0, 0.0, 0.0, 1.0], [0.1, 0.3, 0.4, 0.2]])
            if "random_forest" in system:
                probabilities[0] = [0.2, 0.7, 0.1] if task == "architecture" else [0.1, 0.7, 0.15, 0.05]
            rows = []
            for i, record_id in enumerate(geometry.record_id):
                rows.append({
                    "record_id": record_id, "split_id": "fold-0", "target_label": targets[i],
                    "role": "train" if record_id == "r3" else "test",
                    "prediction_status": "feature_failure" if record_id == "r2" else "ok",
                    "prediction": "" if record_id == "r2" else classes[np.argmax(probabilities[i])],
                })
            pd.DataFrame(rows).to_csv(directory / "predictions.csv", index=False)
            # Intentionally include a train row: the overlay must ignore it even
            # though a plausible probability vector and a prediction are present.
            np.savez(
                directory / "probabilities-00.npz", record_id=geometry.record_id.to_numpy(dtype=str),
                split_id=np.array(["fold-0"]), class_ids=classes,
                prediction_available=np.array([True, False, True, True]), probabilities=probabilities,
            )
    return root, geometry


def test_overlay_preserves_denominators_and_never_uses_training_predictions(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    original = geometry.copy(deep=True)
    output = tmp_path / "output"
    protocol = run_model_overlay(source, geometry, output)
    rows = pd.read_csv(output / "domain-summaries.csv", dtype={"target_label": str})
    primary = rows.loc[rows.task.eq("architecture") & rows.system.eq(SYSTEMS[0])].set_index("record_id")
    assert len(rows) == 24
    assert primary.loc["r3", "target_label"] == "2.10"
    assert primary.loc["r3", "overlay_status"] == "not_in_test_set"
    assert not primary.loc["r3", "prediction_available"]
    assert np.isnan(primary.loc["r3", "top_probability"])
    assert primary.loc["r3", "geometry_role"] == "confirmation"
    assert primary.loc["r2", "overlay_status"] == "source_prediction_unavailable"
    assert pd.isna(primary.loc["r2", "geometry_label_overlap"])
    assert primary.loc["r2", "overlap_stratum"] == "overlap_unavailable"
    summary = pd.read_csv(output / "summary.csv")
    first = summary.loc[summary.task.eq("architecture") & summary.system.eq(SYSTEMS[0])].iloc[0]
    assert first["cohort_n"] == 4 and first["test_n"] == 3 and first["predicted_n"] == 2
    assert first["accuracy_test_effectiveness"] == pytest.approx(2 / 3)
    assert first["accuracy_prediction_available"] == 1
    assert first["prediction_coverage_cohort"] == 0.5
    assert primary.loc["r1", "log_loss"] == pytest.approx(-np.log(0.7))
    assert primary.loc["r1", "brier"] == pytest.approx(0.14)
    assert protocol["sources"][0]["npz_non_test_rows_ignored"] == 1
    pd.testing.assert_frame_equal(geometry, original)


def test_topology_aggregation_is_one_probability_space_with_explicit_support(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    output = tmp_path / "output"
    run_model_overlay(source, geometry, output)
    rows = pd.read_csv(output / "domain-summaries.csv")
    primary = rows.loc[rows.task.eq("topology") & rows.system.eq(SYSTEMS[0])].set_index("record_id")
    first = primary.loc["r1"]
    assert first["mass_B"] == 0.5
    assert first["mass_J"] == 0.25
    assert first["mass_S_other"] == 0.15
    assert first["mass_other"] == 0.1
    assert first["three_group_mass"] == pytest.approx(0.9)
    assert first["conditional_J"] == pytest.approx(0.25 / 0.9)
    assert first["support_J_classes"] == 1
    unsupported = primary.loc["r4"]
    assert unsupported["overlay_status"] == "target_outside_probability_space"
    assert unsupported["probability_available"]
    assert not unsupported["target_supported"]
    assert np.isnan(unsupported["brier"]) and np.isnan(unsupported["log_loss"])
    assert unsupported["top2_margin"] == pytest.approx(0.1)
    support = pd.read_csv(output / "class-support.csv", dtype={"class_label": str})
    absent = support.loc[support.task.eq("topology") & support.class_label.eq("2.60.999")]
    assert not absent.class_supported.any()


def test_fixed_strata_calibration_and_head_disagreement(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    output = tmp_path / "output"
    run_model_overlay(source, geometry, output)
    strata = pd.read_csv(output / "uncertainty-overlap-strata.csv")
    first = strata.loc[strata.task.eq("architecture") & strata.system.eq(SYSTEMS[0])]
    assert first.cohort_n.sum() == 4
    assert first.test_n.sum() == 3
    low_overlap = first.loc[first.overlap_stratum.eq("label_overlap") & first.margin_stratum.eq("margin_le_0.2")].iloc[0]
    assert low_overlap.cohort_n == 1
    calibration = pd.read_csv(output / "calibration.csv")
    bins = calibration.loc[calibration.task.eq("architecture") & calibration.system.eq(SYSTEMS[0]) & calibration.scope.eq("all_valid_test_probabilities")]
    assert len(bins) == 10 and bins.n.sum() == 2
    assert bins.loc[bins.bin.eq(7), "n"].iloc[0] == 1
    assert bins.loc[bins.bin.eq(4), "n"].iloc[0] == 1
    comparisons = pd.read_csv(output / "head-disagreement-summary.csv")
    pair = comparisons.loc[comparisons.task.eq("architecture") & comparisons.left_system.eq(SYSTEMS[0]) & comparisons.right_system.eq(SYSTEMS[2])].iloc[0]
    assert pair.both_prediction_n == 2
    assert pair.disagreement_fraction_available == 0.5
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["low_margin_threshold"] == 0.2


def test_invalid_probability_is_not_silently_normalized(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    path = source / "classification" / "assessment" / "architecture" / SYSTEMS[0] / "probabilities-00.npz"
    with np.load(path) as original:
        payload = {key: original[key].copy() for key in original.files}
    payload["probabilities"][0] = [0.7, 0.2, 0.2]
    np.savez(path, **payload)
    output = tmp_path / "output"
    run_model_overlay(source, geometry, output)
    rows = pd.read_csv(output / "domain-summaries.csv")
    row = rows.loc[rows.task.eq("architecture") & rows.system.eq(SYSTEMS[0]) & rows.record_id.eq("r1")].iloc[0]
    assert row.overlay_status == "invalid_probability_vector"
    assert not row.probability_available
    assert np.isnan(row.normalized_entropy)
    assert row.prediction_available  # The recorded native class call still exists.


def test_fractional_overlap_cannot_be_implicitly_binarized(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    geometry["label_overlap"] = [0.0, np.nan, 0.7, 1.0]
    with pytest.raises(ValueError, match="boolean or NA"):
        run_model_overlay(source, geometry, tmp_path / "output")


def test_duplicate_test_roles_are_rejected(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    path = source / "classification" / "assessment" / "architecture" / SYSTEMS[0] / "predictions.csv"
    predictions = pd.read_csv(path, dtype=str, keep_default_na=False)
    pd.concat([predictions, predictions.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="multiple test predictions"):
        run_model_overlay(source, geometry, tmp_path / "output")


def test_absent_j_class_is_explicit_and_not_a_conditional_zero_risk(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    directory = source / "classification" / "assessment" / "topology" / SYSTEMS[0]
    path = directory / "probabilities-00.npz"
    with np.load(path) as original:
        payload = {key: original[key].copy() for key in original.files}
    payload["class_ids"] = np.array(["2.40.10", "2.60.40", "2.10.10"])
    payload["probabilities"] = np.array([[0.6, 0.3, 0.1], [np.nan] * 3, [0, 0, 1], [0.2, 0.6, 0.2]])
    np.savez(path, **payload)
    output = tmp_path / "output"
    run_model_overlay(source, geometry, output)
    rows = pd.read_csv(output / "domain-summaries.csv")
    row = rows.loc[rows.task.eq("topology") & rows.system.eq(SYSTEMS[0]) & rows.record_id.eq("r1")].iloc[0]
    assert row.support_J_classes == 0
    assert row.support_J_status == "absent_from_fold_space"
    assert row.mass_J == 0  # Zero modeled mass, not a complete-catalog estimate.
    assert np.isnan(row.conditional_J)
    assert row.three_group_mass == pytest.approx(0.9)


def test_unit_confidence_enters_last_calibration_bin(tmp_path: Path) -> None:
    source, geometry = _source(tmp_path)
    directory = source / "classification" / "assessment" / "architecture" / SYSTEMS[0]
    path = directory / "probabilities-00.npz"
    with np.load(path) as original:
        payload = {key: original[key].copy() for key in original.files}
    payload["probabilities"][0] = [1.0, 0.0, 0.0]
    np.savez(path, **payload)
    output = tmp_path / "output"
    run_model_overlay(source, geometry, output)
    calibration = pd.read_csv(output / "calibration.csv")
    row = calibration.loc[
        calibration.task.eq("architecture") & calibration.system.eq(SYSTEMS[0])
        & calibration.scope.eq("all_valid_test_probabilities") & calibration.bin.eq(9)
    ].iloc[0]
    assert row.n == 1 and row.mean_confidence == 1 and row.accuracy == 1
