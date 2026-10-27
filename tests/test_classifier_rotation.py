"""Analytical and rigid-motion checks of the current descriptor contract."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betlas.features.current import (
    CIRCULAR_MOMENT_COLUMNS, FEATURE_COLUMNS, FEATURE_SCHEMA, VALID_PLANE_COLUMNS,
    circular_moments,
)
from betlas.features.extract import (
    extract_classifier_features, extract_signature, extract_structure_features,
    plane_applicability,
)
from betlas.ml.benchmark import raw_geometry_feature_columns
from betlas.models import DomainCandidate, ResidueRecord, SecondaryStructureElement, StructureGeometry


def geometry(strands=4, short_last=False):
    residues, nodes = [], []
    for index, angle in enumerate([-.8, .4, 1.5, 2.9][:strands]):
        coordinates = [[5*np.cos(angle)+.2*j*j, 3*np.sin(angle)+.1*j, 3*j+.3*index]
                       for j in range(2 if short_last and index == strands-1 else 4)]
        selected = []
        for coordinate in coordinates:
            position = len(residues)+1
            selected.append(position-1)
            residues.append(ResidueRecord("toy", "A", position, position, "", "ALA", tuple(coordinate)))
        nodes.append(SecondaryStructureElement(f"s{index}", "strand", "A", selected[0]+1, selected[-1]+1,
                                              tuple(selected), "B" if index == strands-1 else "A"))
    return StructureGeometry(DomainCandidate("toy", "toy", "A", "toy", "", "", "", ""),
                             tuple(residues), tuple(nodes), (), ())


def rotated(original, matrix):
    residues = tuple(replace(row, coord_ca=tuple(np.asarray(row.coord_ca) @ matrix.T + np.array([2.,-4.,9.])))
                     for row in original.residues)
    return replace(original, residues=residues)


def rotations():
    rng = np.random.default_rng(13)
    for _ in range(8):
        matrix, _ = np.linalg.qr(rng.normal(size=(3,3)))
        matrix[:,0] *= np.linalg.det(matrix)
        yield matrix


@pytest.mark.parametrize("kwargs", [{}, {"short_last":True}, {"strands":2}])
def test_current_measurements_are_rigid_motion_invariant(kwargs):
    original = geometry(**kwargs)
    baseline = extract_classifier_features(original)
    assert list(baseline)[:136] == list(FEATURE_COLUMNS)
    for matrix in rotations():
        measured = extract_classifier_features(rotated(original,matrix))
        np.testing.assert_allclose([measured[key] for key in FEATURE_COLUMNS],
                                   [baseline[key] for key in FEATURE_COLUMNS],
                                   atol=1e-8,rtol=1e-8,equal_nan=True)


def test_plane_and_sparse_axis_applicability_remain_explicit():
    original = geometry(short_last=True)
    assert not plane_applicability(original)["plane_applicable"]
    measured = extract_classifier_features(original)
    assert all(np.isnan(measured[key]) for key in VALID_PLANE_COLUMNS)
    sparse = extract_classifier_features(geometry(strands=2))
    assert sparse["betlas_beta_strand_count"] == 2
    assert sparse["betlas_parse_ok"] == 1
    assert np.isnan(sparse["betlas_axis_best_z_range"])
    assert sparse["axis_applicability"] == "too_few_beta_nodes"
    assert all(np.isnan(sparse[key]) for key in CIRCULAR_MOMENT_COLUMNS)


def test_angular_moments_of_regular_quadrilateral():
    moments = circular_moments(np.arange(4)*np.pi/2)
    np.testing.assert_allclose(moments, [0,0,0,1,0,0,0,1], atol=1e-14)
    np.testing.assert_allclose(circular_moments(np.arange(4)*np.pi/2+0.37),moments,atol=1e-14)


def test_empty_geometry_keeps_complete_missing_measurement_contract():
    row=extract_classifier_features(geometry(strands=0))
    assert all(key in row for key in FEATURE_COLUMNS)
    assert row["betlas_parse_ok"] == 0
    assert row["moment_status"] == "no_beta_segments"


def test_default_structure_api_uses_fixed_current_contract():
    row=extract_structure_features(Path(__file__).parents[1]/"examples/mini.cif",chain_id="A")
    assert row["feature_schema"] == FEATURE_SCHEMA
    assert raw_geometry_feature_columns(pd.DataFrame([row])) == list(FEATURE_COLUMNS)
    assert not any("fft" in key or "rule_score" in key for key in row)
    all_missing=pd.DataFrame({key:[np.nan] for key in FEATURE_COLUMNS})
    all_missing["betlas_sheet_pair_top_sheet_ids_equal"]="12"
    assert raw_geometry_feature_columns(all_missing)==list(FEATURE_COLUMNS)


def test_signature_and_classifier_interface_have_same_values():
    row=extract_classifier_features(geometry())
    signature=extract_signature(geometry())
    np.testing.assert_allclose([row[k] for k in FEATURE_COLUMNS],
                               [signature.features[k] for k in FEATURE_COLUMNS],equal_nan=True)
    assert not any(name.startswith("betlas_rule_score_") for name in signature.features)
