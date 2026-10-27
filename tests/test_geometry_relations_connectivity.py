"""Synthetic checks for graph semantics and exact annotation pairing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from Bio.PDB import MMCIFIO

from betlas.geometry_relations.annotations import (
    build_deposited_domain_geometry,
    build_dssp_domain_geometry,
    compare_annotations,
)
from betlas.geometry_relations.connectivity import summarize_connectivity
from betlas.models import (
    DomainCandidate,
    ResidueRecord,
    SecondaryStructureElement,
    StructureGeometry,
)


def _domain():
    return DomainCandidate("toy", "toy", "A", "toy", "", "unlabeled", "synthetic", "synthetic")


def _graph(*, edges=(), chains=("A", "A", "A", "A"), fragments=("F1",) * 4):
    residues, nodes = [], []
    counters = {}
    for index, (chain, fragment) in enumerate(zip(chains, fragments, strict=True)):
        start = counters.get(chain, 1)
        # Numeric author IDs deliberately run backwards; label IDs are polymer order.
        selected = []
        for position in range(start, start + 2):
            selected.append(len(residues))
            residues.append(ResidueRecord("toy", chain, 100 - position, position, "", "ALA",
                                          (float(position), index * 2.0, 0.0)))
        counters[chain] = start + 2
        nodes.append(SecondaryStructureElement(f"node-{9-index}", "strand", chain,
                                               100 - start, 99 - start, tuple(selected),
                                               "sheet", str(90 - index), fragment_id=fragment))
    return StructureGeometry(_domain(), tuple(residues), tuple(nodes), (), (), (),
                             tuple((nodes[i].element_id, nodes[j].element_id, sense)
                                   for i, j, sense in edges), "dssp")


def _annotation_data():
    data = {"data_": "synthetic"}
    atoms = []
    # Insertion-coded author numbering and a second chain are intentional.
    for chain, auth in (("A", [(9, ""), (10, ""), (10, "A"), (11, ""),
                              (12, ""), (13, ""), (14, ""), (15, "")]),
                        ("AA", [(1, ""), (2, ""), (3, ""), (4, "")])):
        for label, (number, insertion) in enumerate(auth, 1):
            atoms.append(["ATOM", str(len(atoms) + 1), "C", "CA", "ALA", chain, chain,
                          str(number), str(label), insertion or "?", ".", "1.0",
                          str(label), str(label % 2 * 3), "0" if chain == "A" else "5", "1"])
    columns = ("group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
               "auth_asym_id", "label_asym_id", "auth_seq_id", "label_seq_id", "pdbx_PDB_ins_code",
               "label_alt_id", "occupancy", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")
    for column, values in zip(columns, zip(*atoms, strict=True), strict=True):
        data[f"_atom_site.{column}"] = list(values)
    for key, value in {"id": ["zeta", "alpha", "middle"], "sheet_id": ["S", "S", "S"],
                       "beg_label_asym_id": ["A", "A", "AA"],
                       "end_label_asym_id": ["A", "A", "AA"],
                       "beg_label_seq_id": ["1", "5", "1"],
                       "end_label_seq_id": ["4", "8", "4"]}.items():
        data[f"_struct_sheet_range.{key}"] = value
    for key, value in {"sheet_id": ["S", "S"], "range_id_1": ["zeta", "zeta"],
                       "range_id_2": ["alpha", "middle"],
                       "sense": ["parallel", "anti-parallel"]}.items():
        data[f"_struct_sheet_order.{key}"] = value
    return data


def _segments():
    return [{"chain_id": "A", "start": 9, "end": 15},
            {"chain_id": "AA", "start": 1, "end": 4}]


def _write(data, path):
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))


def test_pairing_graph_is_distinct_from_contact_graph_and_sequence_order():
    geometry = _graph(edges=((0, 3, "anti-parallel"),))
    result = summarize_connectivity(geometry, include_contacts=True)
    assert result["metrics"]["pairing_edge_count"] == 1
    assert len(result["contact_edges"]) == 6
    assert result["metrics"]["pairing_nonlocal_fraction_same_chain"] == 1
    assert result["pairing_edges"][0]["strand_rank_gap"] == 3
    assert result["pairing_edges"][0]["observed_chain_path"]
    assert [n["label_start"] for n in result["nodes"]] == [1, 3, 5, 7]


def test_graph_statistics_ignore_identifiers_and_rigid_motion():
    geometry = _graph(edges=((0, 1, "parallel"), (1, 2, "parallel"),
                             (2, 3, "anti-parallel"), (3, 0, "anti-parallel")))
    original = summarize_connectivity(geometry, include_contacts=True)
    mapping = {node.element_id: f"rename{index * 17}" for index, node in enumerate(reversed(geometry.beta_segments))}
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    renamed = replace(geometry,
        residues=tuple(replace(r, coord_ca=tuple(rotation @ np.array(r.coord_ca) + [4, -3, 2])) for r in geometry.residues),
        beta_segments=tuple(replace(n, element_id=mapping[n.element_id], sheet_id="renamed-sheet",
                                    sheet_range_id="arbitrary") for n in reversed(geometry.beta_segments)),
        sheet_connections=tuple((mapping[a], mapping[b], s) for a, b, s in geometry.sheet_connections))
    revised = summarize_connectivity(renamed, include_contacts=True)
    assert revised["metrics"] == original["metrics"]
    assert revised["metrics"]["pairing_cycle_rank"] == 1
    assert revised["metrics"]["pairing_simple_cycle_component_node_fraction"] == 1
    assert sorted(e["min_ca_distance_angstrom"] for e in revised["contact_edges"]) == pytest.approx(
        sorted(e["min_ca_distance_angstrom"] for e in original["contact_edges"]))


def test_branch_cross_chain_and_fragment_states_are_preserved():
    geometry = _graph(edges=((0, 1, "parallel"), (0, 2, "anti-parallel"), (0, 3, "parallel")),
                      chains=("A", "A", "A", "B"), fragments=("F1", "F1", "F2", "F3"))
    geometry = replace(geometry, warnings=("clipped_ladder_pairing_unavailable",))
    result = summarize_connectivity(geometry)
    assert result["pairing_status"] == "partial_clipped_pairings"
    assert result["metrics"]["pairing_branch_fraction"] == .25
    assert result["metrics"]["pairing_cycle_rank"] == 0
    assert {e["sequence_context"] for e in result["pairing_edges"]} == {"same_fragment", "cross_fragment", "cross_chain"}
    cross = next(e for e in result["pairing_edges"] if e["sequence_context"] == "cross_chain")
    assert cross["strand_rank_gap"] is None and not cross["observed_chain_path"]
    assert all(e["sequence_context"] != "cross_chain" for e in result["sequence_relations"])


def test_empty_edges_do_not_become_observed_zero_orientation():
    result = summarize_connectivity(_graph())
    assert result["pairing_status"] == "no_retained_pairings"
    assert result["metrics"]["pairing_parallel_fraction_known"] is None
    missing = summarize_connectivity(replace(_graph(), warnings=("missing_deposited_sheet_order",)))
    assert missing["pairing_status"] == "annotation_pairing_missing"


def test_deposited_keeps_full_declared_edges_and_exact_insertions(tmp_path):
    data = _annotation_data()
    path = tmp_path / "synthetic.cif"
    _write(data, path)
    geometry = build_deposited_domain_geometry(_domain(), data, segments=_segments())
    from_file = build_deposited_domain_geometry(_domain(), path, segments=_segments())
    assert geometry == from_file
    assert len(geometry.sheet_connections) == 2
    assert len(geometry.residues) == 12
    assert any(r.insertion_code == "A" for r in geometry.residues)
    result = summarize_connectivity(geometry)
    assert result["metrics"]["pairing_edge_count"] == 2
    assert result["metrics"]["pairing_cross_chain_fraction"] == .5


def test_deposited_partial_range_does_not_invent_pairing():
    data = _annotation_data()
    segments = [{"chain_id": "A", "start": 10, "end": 15}]
    geometry = build_deposited_domain_geometry(_domain(), data, segments=segments)
    assert not geometry.sheet_connections
    assert "clipped_deposited_pairing_unavailable" in geometry.warnings
    assert summarize_connectivity(geometry)["pairing_status"] == "partial_clipped_pairings"


def test_dssp_path_mapping_parity_and_annotation_node_overlap(tmp_path):
    deposited_data = _annotation_data()
    deposited = build_deposited_domain_geometry(_domain(), deposited_data, segments=_segments())
    dssp = dict(deposited_data)
    dssp["_dssp_struct_summary.label_seq_id"] = ["1"]
    # Different beta boundaries are a measured annotation change, not an input change.
    dssp["_struct_sheet_range.beg_label_seq_id"] = ["2", "5", "1"]
    for key, value in {"id": ["ladder"], "sheet_id": ["S"], "range_id_1": ["zeta"],
                       "range_id_2": ["middle"], "type": ["anti-parallel"],
                       "beg_1_label_asym_id": ["A"], "beg_2_label_asym_id": ["AA"],
                       "beg_1_label_seq_id": ["2"], "end_1_label_seq_id": ["4"],
                       "beg_2_label_seq_id": ["4"], "end_2_label_seq_id": ["1"]}.items():
        dssp[f"_dssp_struct_ladder.{key}"] = value
    path = tmp_path / "annotated.cif"
    _write(dssp, path)
    geometry = build_dssp_domain_geometry(_domain(), dssp, segments=_segments())
    assert geometry == build_dssp_domain_geometry(_domain(), path, segments=_segments())
    comparison = compare_annotations(deposited, geometry)
    assert comparison["same_selected_coordinates"]
    assert comparison["left_beta_residue_count"] == 12
    assert comparison["right_beta_residue_count"] == 11
    assert comparison["beta_residue_jaccard"] == 11 / 12
    assert comparison["node_residue_overlaps"][0]["shared_residue_count"] == 3
    shifted = replace(geometry, residues=(replace(geometry.residues[0], coord_ca=(99., 0., 0.)), *geometry.residues[1:]))
    assert not compare_annotations(deposited, shifted)["same_selected_coordinates"]


def test_deposited_missing_annotation_keeps_residue_population():
    data = {key: value for key, value in _annotation_data().items() if not key.startswith("_struct_sheet")}
    geometry = build_deposited_domain_geometry(_domain(), data, segments=_segments())
    assert len(geometry.residues) == 12
    assert not geometry.beta_segments
    assert "missing_deposited_sheet_annotation" in geometry.warnings
    assert summarize_connectivity(geometry)["pairing_status"] == "no_beta_segments"


def test_deposited_author_endpoint_fallback_preserves_insertion_keys():
    data = _annotation_data()
    for key in list(data):
        if key.startswith("_struct_sheet_range.") and "label_" in key:
            del data[key]
    for key, values in {
        "beg_auth_asym_id": ["A", "A", "AA"], "end_auth_asym_id": ["A", "A", "AA"],
        "beg_auth_seq_id": ["9", "12", "1"], "end_auth_seq_id": ["10", "15", "4"],
        "pdbx_beg_PDB_ins_code": ["?", "?", "?"], "pdbx_end_PDB_ins_code": ["A", "?", "?"],
    }.items():
        data[f"_struct_sheet_range.{key}"] = values
    geometry = build_deposited_domain_geometry(_domain(), data, segments=_segments())
    first = geometry.beta_segments[0]
    assert len(first.residue_indices) == 3
    assert geometry.residues[first.residue_indices[-1]].insertion_code == "A"
    assert len(geometry.sheet_connections) == 2


def test_missing_ca_and_discontinuous_selection_do_not_create_chain_paths():
    data = _annotation_data()
    # Remove an interior CA. Domain residues and strand fragments remain distinct.
    for key in list(data):
        if key.startswith("_atom_site."):
            data[key] = [value for index, value in enumerate(data[key]) if index != 2]
    geometry = build_deposited_domain_geometry(_domain(), data, segments=_segments())
    result = summarize_connectivity(geometry)
    assert "missing_ca_within_domain" in geometry.warnings
    assert result["pairing_status"] == "partial_clipped_pairings"
    assert all(not edge["observed_chain_path"] for edge in result["sequence_relations"]
               if edge["sequence_context"] == "cross_fragment")
    discontinuous = build_deposited_domain_geometry(_domain(), _annotation_data(), segments=[
        {"chain_id": "A", "start": 9, "end": 10},
        {"chain_id": "A", "start": 12, "end": 15}])
    relationship = summarize_connectivity(discontinuous)["sequence_relations"][0]
    assert relationship["sequence_context"] == "cross_fragment"
    assert not relationship["observed_chain_path"]


def test_overlapping_annotation_nodes_have_unavailable_sequence_span():
    geometry = _graph(edges=((0, 1, "parallel"),))
    nodes = (geometry.beta_segments[0], replace(geometry.beta_segments[1],
             residue_indices=geometry.beta_segments[0].residue_indices), *geometry.beta_segments[2:])
    result = summarize_connectivity(replace(geometry, beta_segments=nodes))
    assert result["pairing_edges"][0]["sequence_context"] == "overlapping_polymer_ranges"
    assert result["metrics"]["pairing_strand_rank_gap_mean"] is None
