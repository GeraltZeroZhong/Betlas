from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betlas.geometry_relations.cohorts import A, H, T
from betlas.geometry_relations.connection_patterns_geometry import (
    PHYSICAL_COLUMNS,
    measure_pattern_geometry,
    run_pattern_geometry,
)


def _geometry():
    nodes, residues = [], []
    for node_index in range(4):
        sheet, strand = divmod(node_index, 2)
        uids = []
        for index in range(6):
            uid = f"A:{node_index * 10 + index}:"
            uids.append(uid)
            residues.append({"residue_uid": uid, "coordinate": [float(index), strand * 4.0 + (index % 2) * .2, sheet * 6.0]})
        nodes.append({"node_id": f"s{node_index}", "sheet_id": f"sheet{sheet}", "residue_uids": uids,
                      "label_positions": list(range(node_index * 10, node_index * 10 + 6))})
    return {"nodes": nodes}, residues


def test_physical_geometry_is_rigid_invariant_and_size_is_explicit():
    graph, residues = _geometry()
    first = measure_pattern_geometry(graph, residues)
    assert first["domain"]["coordinate_status"] == "complete_beta_coordinates"
    assert first["domain"]["largest_two_sheet_centroid_distance_angstrom"] == pytest.approx(6)
    assert first["domain"]["largest_two_sheet_normal_angle_degrees"] == pytest.approx(0)
    assert first["domain"]["strand_axis_bend_mean_degrees"] == pytest.approx(0)
    assert first["domain"]["n_sheet_pairs_measured"] == 1
    angle = .71
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    moved = [{**row, "coordinate": (rotation @ np.asarray(row["coordinate"]) + [17, -25, 8]).tolist()} for row in residues]
    second = measure_pattern_geometry(graph, moved)
    for field in PHYSICAL_COLUMNS:
        assert second["domain"][field] == pytest.approx(first["domain"][field], abs=1e-5)
    enlarged = [{**row, "coordinate": (np.asarray(row["coordinate"]) * 2).tolist()} for row in residues]
    third = measure_pattern_geometry(graph, enlarged)
    for field in PHYSICAL_COLUMNS:
        multiplier = 2 if field.endswith("angstrom") else 1
        assert third["domain"][field] == pytest.approx(first["domain"][field] * multiplier, abs=1e-5)


def test_missing_CA_and_inapplicable_planes_remain_NA():
    graph, residues = _geometry()
    missing = measure_pattern_geometry(graph, residues[1:])["domain"]
    assert missing["coordinate_status"] == "partial_beta_coordinates"
    assert missing["beta_residues_missing"] == 1
    assert all(missing[field] is None for field in PHYSICAL_COLUMNS)
    graph["nodes"][3]["sheet_id"] = "single_strand_sheet"
    measured = measure_pattern_geometry(graph, residues)
    assert measured["domain"]["n_sheets_plane_measured"] == 1
    assert measured["domain"]["n_sheet_pairs_measured"] == 0
    assert measured["domain"]["largest_two_sheet_centroid_distance_angstrom"] is None
    assert measured["domain"]["sheet_plane_rms_mean_angstrom"] == pytest.approx(0)


def test_sequence_gaps_do_not_masquerade_as_local_strand_bending():
    graph, residues = _geometry()
    graph["nodes"][0]["label_positions"][3] += 2
    measured = measure_pattern_geometry(graph, residues)
    strand = measured["strands"][0]
    assert not strand["polymer_positions_contiguous"]
    assert strand["axis_bend_degrees"] is None
    assert strand["ca_pseudodihedral_mean_absolute_degrees"] is None
    assert measured["domain"]["n_strands_bend_measured"] == 3
    assert measured["domain"]["beta_rg_angstrom"] is not None


def test_physical_distributions_use_all_requests_not_display_representatives(tmp_path: Path):
    graph, residues = _geometry()
    requests = pd.DataFrame({"record_id": [f"r{i}" for i in range(6)], "chain_pattern": ["Kchain-p00001"] * 6,
                             H: ["h1"] * 3 + ["h2"] * 3, A: ["2.60"] * 6, T: ["2.60.40"] * 6,
                             "pattern_population_n": [6] * 6, "pattern_requested_n": [6] * 6})
    requests.to_csv(tmp_path / "coordinate-requests.csv", index=False)
    requests.iloc[[0, 5]].to_csv(tmp_path / "series-representatives.csv", index=False)
    raw_path = tmp_path / "graphs.jsonl.gz"
    with gzip.open(raw_path, "wt") as handle:
        for record_id in requests.record_id:
            handle.write(json.dumps({"record_id": record_id, "graph": graph}) + "\n")
    prepared = {record_id: {"ca_identities": deepcopy(residues)} for record_id in requests.record_id}
    prepared["r5"]["ca_identities"].pop()
    protocol = run_pattern_geometry(raw_path, prepared, tmp_path)
    assert protocol["requested_domains"] == 6
    assert protocol["coordinate_status"]["complete_beta_coordinates"] == 5
    summary = pd.read_csv(tmp_path / "pattern-physical-distributions.csv").set_index("measurement")
    assert summary.loc["beta_rg_angstrom", "n_total"] == 6
    assert summary.loc["beta_rg_angstrom", "n_observed"] == 5
    assert summary.loc["beta_rg_angstrom", "H_observed"] == 2
