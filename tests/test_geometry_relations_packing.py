from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from betlas.geometry_relations.packing import (
    PACKING_BLOCKS,
    PackingMatrix,
    PackingScaler,
    knn_from_distances,
    pairwise_block_distances,
    prepare_packing_features,
)


def test_scale_uses_only_discovery_and_serializes_without_imputing() -> None:
    discovery = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "constant": [2.0] * 4, "empty": [np.nan] * 4})
    scaler = PackingScaler.fit(discovery, {"shape": ("x", "constant", "empty")})
    before = json.dumps(scaler.to_dict(), sort_keys=True)
    confirmation = pd.DataFrame({"x": [1000.0, np.nan], "constant": [900.0, 900.0]})
    matrix = scaler.transform(confirmation)
    assert matrix.columns == ("x",)
    assert matrix.values[0, 0] == pytest.approx((1000 - 1.5) / 1.5)
    assert np.isnan(matrix.values[1, 0])
    assert matrix.block_availability()["shape"].tolist() == [True, False]
    assert json.dumps(scaler.to_dict(), sort_keys=True) == before
    restored = PackingScaler.from_dict(json.loads(before))
    np.testing.assert_array_equal(restored.transform(confirmation).values, matrix.values)


def test_identifiers_labels_and_predictions_do_not_change_geometry() -> None:
    data = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0], "record_id": ["a", "b", "c", "d"],
        "cath_architecture": ["first"] * 4, "prediction_probability": [0.9] * 4,
    })
    changed = data.assign(record_id=["z", "w", "y", "q"], cath_architecture="second", prediction_probability=0.1)
    original = PackingScaler.fit(data, {"shape": ("x",)}).transform(data)
    modified = PackingScaler.fit(changed, {"shape": ("x",)}).transform(changed)
    np.testing.assert_array_equal(original.values, modified.values)
    np.testing.assert_array_equal(
        pairwise_block_distances(original).combine(), pairwise_block_distances(modified).combine(),
    )


def test_block_weighting_is_invariant_to_redundant_fields_inside_a_block() -> None:
    compact = PackingMatrix(np.array([[0, 0], [4, 2]], dtype=np.float32), ("a", "b"), {"a": (0,), "b": (1,)})
    duplicate = PackingMatrix(
        np.array([[0, 0, 0, 0], [4, 4, 4, 2]], dtype=np.float32),
        ("a1", "a2", "a3", "b"), {"a": (0, 1, 2), "b": (3,)},
    )
    for metric, expected in (("rms", np.sqrt(10)), ("l1", 3.0)):
        first = pairwise_block_distances(compact, block_size=1).combine(metric=metric)
        second = pairwise_block_distances(duplicate, block_size=2).combine(metric=metric)
        np.testing.assert_allclose(first, second)
        assert first[0, 1] == pytest.approx(expected)


def test_missing_block_is_not_zero_distance_and_removal_restores_coverage() -> None:
    matrix = PackingMatrix(
        np.array([[0, np.nan], [1, 3], [2, 4]], dtype=np.float32),
        ("shape", "repetition"), {"shape": (0,), "repetition": (1,)},
    )
    cache = pairwise_block_distances(matrix, block_size=2)
    full = cache.combine()
    assert np.isinf(full[0]).all()
    reduced = cache.combine(drop_blocks=("repetition",))
    assert reduced[0, 1] == 1
    assert reduced[0, 2] == 2
    assert knn_from_distances(full).finite_candidate_counts[0] == 0
    assert knn_from_distances(reduced).finite_candidate_counts[0] == 2
    with pytest.raises(ValueError, match="At least one"):
        cache.combine(drop_blocks=("shape", "repetition"))


def test_native_zero_placeholders_require_actual_measurement_support() -> None:
    frame = pd.DataFrame({column: [0.0] * 3 for columns in PACKING_BLOCKS.values() for column in columns})
    frame["betlas_parse_ok"] = [1, 1, 0]
    frame["betlas_beta_strand_count"] = [2, 4, 8]
    frame["betlas_beta_residue_count"] = [4, 40, 80]
    frame["betlas_sheet_face_count"] = [0, 2, 3]
    frame["betlas_axis_best_slice_count"] = [0, 5, 8]
    selected = prepare_packing_features(frame)
    assert selected.loc[0, "betlas_pca_flatness"] == 0
    assert np.isnan(selected.loc[0, "betlas_angular_circular_moment_k1"])
    assert np.isnan(selected.loc[0, "betlas_angular_circular_moment_k8"])
    assert np.isnan(selected.loc[0, "betlas_sheet_axis_dispersion_mean"])
    assert np.isnan(selected.loc[0, "betlas_axis_best_slice_coverage_median"])
    assert selected.loc[1].notna().all()
    assert selected.loc[2].isna().all()
    no_metadata = prepare_packing_features(frame.drop(columns="betlas_beta_strand_count"))
    assert no_metadata["betlas_angular_circular_moment_k3"].isna().all()


def test_filtering_original_top_k_never_fills_with_more_distant_records() -> None:
    positions = np.array([0, 1, 2, 10], dtype=float)
    distances = np.abs(positions[:, None] - positions[None, :])
    original = distances.copy()
    neighbors = knn_from_distances(distances, max_k=3)
    result = neighbors.neighborhood(2, group_ids=["same", "same", "other", "far"], pdb_ids=["a", "b", "A", "d"])
    assert result.coverage.loc[0, "base_neighbors"] == 2
    assert result.coverage.loc[0, "retained_neighbors"] == 0
    assert result.coverage.loc[0, "local_scale"] == 2
    assert not (result.edges["source"] == 0).any()
    assert len(result.excluded[result.excluded["source"] == 0]) == 2
    np.testing.assert_array_equal(distances, original)
    wider = neighbors.neighborhood(3, group_ids=["same", "same", "other", "far"], pdb_ids=["a", "b", "A", "d"])
    only = wider.edges[wider.edges["source"] == 0].iloc[0]
    assert only["target"] == 3 and only["rank"] == 3


def test_mutuality_uses_requested_k_and_reports_finite_coverage() -> None:
    distances = np.array([[0, 1, 3, np.inf], [1, 0, 2, np.inf], [3, 2, 0, np.inf], [np.inf] * 4])
    neighbors = knn_from_distances(distances, max_k=60)
    one = neighbors.neighborhood(1)
    assert one.edges["mutual"].tolist() == [True, True, False]
    assert one.coverage.loc[3, "finite_candidates"] == 0
    assert np.isnan(one.coverage.loc[3, "coverage_available"])
    ten = neighbors.neighborhood(10)
    assert ten.edges["mutual"].all()
    assert ten.coverage.loc[0, "base_neighbors"] == 2
    assert ten.coverage.loc[0, "coverage_requested"] == 0.2
    assert ten.coverage.loc[0, "coverage_available"] == 1
    assert np.isfinite(ten.edges["relative_distance"]).all()


def test_zero_iqr_fallback_records_rare_discovery_variation() -> None:
    scaler = PackingScaler.fit(pd.DataFrame({"x": [0.0] * 9 + [1.0]}), {"shape": ("x",)})
    assert scaler.field_status["x"]["status"] == "active_zero_iqr_std_fallback"
    assert scaler.scale["x"] == pytest.approx(0.3)
