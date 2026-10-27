"""Synthetic checks of rotation invariance references and sampling weights."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from betlas.geometry_relations.cohorts import A, H
from betlas.features.current import circular_moments
from betlas.geometry_relations.measurement_controls import (
    S35,
    block_decomposition,
    geometry_from_dict,
    representative_controls,
    rotate_geometry,
    rotation_matrices,
    weighted_quantile,
    weighted_summaries,
)
from betlas.models import DomainCandidate, ResidueRecord, StructureGeometry


def test_rigid_rotations_preserve_distances_annotation_and_identity_roundtrip():
    domain = DomainCandidate("toy", "toy", "A", "toy", "", "", "", "")
    coords = np.array([[1., 2., 3.], [-4., 2., 1.], [3., -2., 7.]])
    residues = tuple(ResidueRecord("toy", "A", i, i, "", "ALA", tuple(p)) for i, p in enumerate(coords))
    geometry = StructureGeometry(domain, residues, (), (), ())
    assert geometry_from_dict(geometry.to_dict()) == geometry
    for rotation in rotation_matrices(4):
        rotated = rotate_geometry(geometry, rotation)
        new = np.array([row.coord_ca for row in rotated.residues])
        assert np.allclose(np.linalg.norm(new[:, None] - new[None, :], axis=-1),
                           np.linalg.norm(coords[:, None] - coords[None, :], axis=-1))
        assert replace(rotated, residues=geometry.residues) == geometry
    with pytest.raises(ValueError):
        rotate_geometry(geometry, np.diag([1, 1, -1]))


def test_equal_block_weighting_is_not_equal_feature_weighting_and_preserves_na():
    left = np.array([[2., 0., 0.], [0., 0., 0.], [np.nan, 1., 2.]])
    result = block_decomposition(left, np.zeros_like(left), {"one": [0], "two": [1, 2]})
    assert result.packing_distance.iloc[0] == np.sqrt(2)
    assert result.one_fraction_of_squared_distance.iloc[0] == 1
    assert pd.isna(result.one_fraction_of_squared_distance.iloc[1])
    assert pd.isna(result.packing_distance.iloc[2])
    assert np.isfinite(result.two_rms.iloc[2])


def test_group_weighting_changes_total_opportunity_without_independence_claim():
    frame = pd.DataFrame({S35: ["a", "a", "b"], H: ["x", "x", "y"], "value": [0., 0., 9.]})
    summaries = weighted_summaries(frame, ["value"]).set_index("weighting")
    assert summaries.loc["domain_equal", "mean"] == 3
    assert summaries.loc["s35_equal", "mean"] == 4.5
    assert summaries.loc["h_equal", "mean"] == 4.5
    assert weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], [.5])[0] == 2.5


def test_representative_candidates_are_not_evaluated_on_themselves():
    frame = pd.DataFrame({"record_id": [f"r{i}" for i in range(9)],
                          A: ["2.40"] * 9, H: ["2.40.1.1"] * 9, S35: ["cluster"] * 9,
                          "packing_distance": np.arange(9, dtype=float),
                          "self_reference": [True] + [False] * 8})
    values = np.arange(9, dtype=float)[:, None]
    representatives = frame.iloc[:1]
    distances, manifests, coverage = representative_controls(
        frame, values, representatives, values[:1], {"one": [0]})
    assert len(distances) == 12
    assert coverage.selected_clusters.sum() == 1
    record = manifests[0]
    assert not set(record["candidate_ids"]) & set(record["evaluation_ids"])
    assert not set(record["training_ids"]) & set(record["evaluation_ids"])
    assert "r0" not in record["training_ids"] + record["evaluation_ids"]
    assert distances.groupby("method").record_id.apply(set).apply(
        lambda observed: observed == set(record["evaluation_ids"])).all()


def test_circular_moments_are_invariant_to_continuous_phase_and_axis_reversal():
    angles = np.array([-.9, .2, .8, 1.1, 2.6, 2.9])
    original = circular_moments(angles)
    for phase in (.017, .4, np.pi / 7, 4.9):
        np.testing.assert_allclose(circular_moments(angles + phase), original, atol=1e-14)
        np.testing.assert_allclose(circular_moments(-angles + phase), original, atol=1e-14)
    assert np.all((original >= 0) & (original <= 1))
    assert np.isnan(circular_moments([])).all()


def test_uniform_fourfold_pattern_has_exact_fourfold_moment():
    measured = circular_moments(np.arange(4) * np.pi / 2)
    np.testing.assert_allclose(measured, [0, 0, 0, 1, 0, 0, 0, 1], atol=1e-14)


