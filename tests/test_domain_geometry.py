from __future__ import annotations

import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from Bio.PDB import MMCIFIO

from betlas.features.extract import _beta_runs, extract_signature
from betlas.io.domain_geometry import (
    annotate_structure,
    build_domain_geometry,
    export_domain_from_source,
    export_domain_mmcif,
)
from betlas.io.mmcif import read_mmcif_dict
from betlas.models import DomainCandidate


def _domain():
    return DomainCandidate("toy", "toy", "A", "toy", "", "unlabeled", "synthetic", "synthetic")


def _fixture(path: Path, *, sheets: bool = True, models: bool = False):
    data = {"data_": "synthetic"}
    atoms = []
    for chain, auth_ids in (("A", [(9, ""), (10, ""), (10, "A"), (11, ""),
                                  (12, ""), (13, ""), (14, ""), (15, "")]),
                            ("AA", [(1, ""), (2, ""), (3, ""), (4, "")])):
        for label, (auth, insertion) in enumerate(auth_ids, 1):
            atoms.append(["ATOM", str(len(atoms) + 1), "C", "CA", "ALA", chain, chain,
                          str(auth), str(label), insertion or "?", ".", "1.0",
                          str(label), str(label % 2 * 3), "0" if chain == "A" else "5", "1"])
    if models:
        second = [row.copy() for row in atoms]
        for row in second:
            row[-1] = "2"
            row[-4] = str(float(row[-4]) + 100)
        atoms.extend(second)
    columns = ("group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
               "auth_asym_id", "label_asym_id", "auth_seq_id", "label_seq_id", "pdbx_PDB_ins_code",
               "label_alt_id", "occupancy", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")
    for column, values in zip(columns, zip(*atoms, strict=True), strict=True):
        data[f"_atom_site.{column}"] = list(values)
    data["_dssp_struct_summary.label_seq_id"] = ["1"]
    if sheets:
        for key, values in {
            "id": ["alpha", "omega"], "sheet_id": ["S", "S"],
            "beg_label_asym_id": ["A", "AA"], "end_label_asym_id": ["A", "AA"],
            "beg_label_seq_id": ["1", "1"], "end_label_seq_id": ["8", "4"],
        }.items():
            data[f"_struct_sheet_range.{key}"] = values
        for key, value in {
            "id": "ladder", "sheet_id": "S", "range_id_1": "alpha", "range_id_2": "omega",
            "type": "anti-parallel", "beg_1_label_asym_id": "A", "beg_2_label_asym_id": "AA",
            "beg_1_label_seq_id": "1", "end_1_label_seq_id": "8",
            "beg_2_label_seq_id": "4", "end_2_label_seq_id": "1",
        }.items():
            data[f"_dssp_struct_ladder.{key}"] = [value]
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    return data


def _segment(chain="A", start=9, end=15, start_ins="", end_ins=""):
    return dict(chain_id=chain, start=start, end=end, start_ins=start_ins, end_ins=end_ins)


def _save(data, path):
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))


def _scheme(data):
    for source, target in (("label_asym_id", "asym_id"), ("auth_asym_id", "pdb_strand_id"),
                           ("label_seq_id", "seq_id"), ("auth_seq_id", "pdb_seq_num"),
                           ("auth_seq_id", "auth_seq_num"), ("pdbx_PDB_ins_code", "pdb_ins_code")):
        data[f"_pdbx_poly_seq_scheme.{target}"] = data[f"_atom_site.{source}"].copy()


def _cath_zero_domain():
    return replace(_domain(), record_id="9toy000", domain_id="9toy000", pdb_id="9toy", chain_id="",
                   cath_architecture_code="2.40")


def test_cath_zero_token_resolves_literal_author_zero_for_features_and_raw_atoms(tmp_path):
    path, output = tmp_path / "zero.cif", tmp_path / "domain.cif"
    data = _fixture(path)
    data["_atom_site.auth_asym_id"] = ["0" if chain == "A" else chain
                                     for chain in data["_atom_site.auth_asym_id"]]
    _scheme(data)
    _save(data, path)
    geometry = build_domain_geometry(_cath_zero_domain(), path, segments=[_segment("")])
    assert len(geometry.residues) == 8
    assert {residue.chain_id for residue in geometry.residues} == {"0"}
    assert geometry.beta_segments[0].chain_id == "0"
    report = export_domain_from_source(_cath_zero_domain(), path, output, segments=[_segment("")])
    assert report["atom_count"] == 8
    assert set(read_mmcif_dict(output)["_atom_site.auth_asym_id"]) == {"0"}
    # A blank request without explicit CATH provenance must remain strict.
    with pytest.raises(ValueError, match="unmapped"):
        build_domain_geometry(replace(_cath_zero_domain(), cath_architecture_code=""), path,
                              segments=[_segment("")])


def test_cath_zero_literal_can_alias_blank_only_when_literal_zero_is_absent(tmp_path):
    path = tmp_path / "blank.cif"
    data = _fixture(path)
    data["_atom_site.auth_asym_id"] = ["?" if chain == "A" else chain
                                     for chain in data["_atom_site.auth_asym_id"]]
    _scheme(data)
    _save(data, path)
    geometry = build_domain_geometry(_cath_zero_domain(), path, segments=[_segment("0")])
    assert len(geometry.residues) == 8
    assert {residue.chain_id for residue in geometry.residues} == {""}
    with pytest.raises(ValueError, match="unmapped"):
        build_domain_geometry(replace(_cath_zero_domain(), cath_architecture_code=""), path,
                              segments=[_segment("0")])


def test_cath_zero_token_rejects_two_boundary_valid_author_chains(tmp_path):
    path = tmp_path / "ambiguous.cif"
    data = _fixture(path)
    data["_atom_site.auth_asym_id"] = ["0" if chain == "A" else "?"
                                     for chain in data["_atom_site.auth_asym_id"]]
    data["_atom_site.auth_seq_id"][8:] = ["9", "10", "11", "15"]
    _scheme(data)
    _save(data, path)
    with pytest.raises(ValueError, match="ambiguous CATH zero-chain"):
        build_domain_geometry(_cath_zero_domain(), path, segments=[_segment("")])
    with pytest.raises(ValueError, match="ambiguous CATH zero-chain"):
        export_domain_from_source(_cath_zero_domain(), path, tmp_path / "domain.cif",
                                  segments=[_segment("")])


def test_cath_zero_does_not_use_blank_when_present_literal_chain_has_wrong_boundaries(tmp_path):
    path = tmp_path / "different.cif"
    data = _fixture(path)
    data["_atom_site.auth_asym_id"] = ["0" if chain == "A" else "?"
                                     for chain in data["_atom_site.auth_asym_id"]]
    data["_atom_site.auth_seq_id"][8:] = ["21", "22", "23", "24"]
    _scheme(data)
    _save(data, path)
    with pytest.raises(ValueError, match="unmapped"):
        build_domain_geometry(_cath_zero_domain(), path, segments=[_segment("", 21, 24)])


def test_unobserved_endpoint_resolves_only_verified_polymer_scheme(tmp_path):
    path = tmp_path / "input.cif"
    data = _fixture(path)
    _scheme(data)
    # Author-provided scheme numbers need not match coordinate numbering.
    data["_pdbx_poly_seq_scheme.auth_seq_num"] = [str(1000 + i) for i in range(12)]
    for name in list(data):
        if name.startswith("_atom_site."):
            data[name] = data[name][1:]
    _save(data, path)
    geometry = build_domain_geometry(_domain(), path, segments=[_segment()])
    assert geometry.requested_residue_count == 8
    assert len(geometry.residues) == 7
    assert geometry.missing_boundary_ca_count == 1
    assert "boundary_ca_missing_resolved_by_polymer_scheme" in geometry.warnings
    with pytest.raises(ValueError, match="unmapped"):
        build_domain_geometry(_domain(), path, segments=[_segment(start=1000)])


def test_nonmonotonic_author_numbering_follows_polymer_order(tmp_path):
    path = tmp_path / "input.cif"
    data = _fixture(path)
    data["_atom_site.auth_seq_id"][:8] = ["100", "101", "102", "1", "2", "3", "4", "5"]
    _save(data, path)
    geometry = build_domain_geometry(_domain(), path, segments=[_segment(start=100, end=5)])
    assert [r.auth_seq_id for r in geometry.residues] == [100, 101, 102, 1, 2, 3, 4, 5]
    assert len(geometry.beta_segments) == 1
    assert _beta_runs(geometry)[0]["length"] == 8


def test_exact_domain_export_contains_all_atoms_and_no_other_domain(tmp_path):
    path, output = tmp_path / "input.cif", tmp_path / "domain.cif"
    data = _fixture(path)
    for key in list(data):
        if key.startswith("_atom_site."):
            data[key].append(data[key][0])
    data["_atom_site.label_atom_id"][-1] = "N"
    data["_atom_site.type_symbol"][-1] = "N"
    data["_atom_site.id"][-1] = "99"
    _save(data, path)
    report = export_domain_mmcif(_domain(), path, output,
        segments=[_segment(end=10), _segment("AA", 3, 4)])
    exported = read_mmcif_dict(output)
    assert report["requested_residue_count"] == 4
    assert report["atom_count"] == 5
    assert "N" in exported["_atom_site.label_atom_id"]
    assert set(zip(exported["_atom_site.auth_asym_id"], exported["_atom_site.auth_seq_id"], strict=True)) == {
        ("A", "9"), ("A", "10"), ("AA", "3"), ("AA", "4")}
    assert not any(key.startswith("_struct_sheet_range.") for key in exported)


def test_raw_export_needs_no_dssp_and_selects_exact_model_and_segments(tmp_path):
    path, output = tmp_path / "raw.cif", tmp_path / "domain.cif"
    data = _fixture(path, sheets=False, models=True)
    del data["_dssp_struct_summary.label_seq_id"]
    _save(data, path)
    report = export_domain_from_source(_domain(), path, output, model_id=1,
        segments=[_segment(start=10, start_ins="A", end=11), _segment("AA", 3, 4)])
    cropped = read_mmcif_dict(output)
    assert report["model_id"] == 1 and report["source_model_num"] == "2"
    assert report["atom_count"] == 4
    assert set(cropped["_atom_site.pdbx_PDB_model_num"]) == {"1"}
    assert [float(value) for value in cropped["_atom_site.Cartn_x"]] == [103, 104, 103, 104]
    assert list(zip(cropped["_atom_site.auth_asym_id"], cropped["_atom_site.auth_seq_id"],
                    cropped["_atom_site.pdbx_PDB_ins_code"], strict=True)) == [
        ("A", "10", "A"), ("A", "11", "?"), ("AA", "3", "?"), ("AA", "4", "?")]
    assert not any(key.startswith("_dssp") for key in cropped)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="unavailable"):
        export_domain_from_source(_domain(), path, output, segments=[_segment()], model_id=2)
    assert output.read_bytes() == original


def test_raw_export_resolves_unobserved_boundary_without_annotation(tmp_path):
    path, output = tmp_path / "raw.cif", tmp_path / "domain.cif"
    data = _fixture(path, sheets=False)
    del data["_dssp_struct_summary.label_seq_id"]
    _scheme(data)
    for key in list(data):
        if key.startswith("_atom_site."):
            data[key] = data[key][1:]
    _save(data, path)
    report = export_domain_from_source(_domain(), path, output, segments=[_segment(end=11)])
    assert report["requested_residue_count"] == 4
    assert report["missing_boundary_ca_count"] == 1
    assert report["atom_count"] == 3
    assert read_mmcif_dict(output)["_atom_site.label_seq_id"] == ["2", "3", "4"]


def test_exact_insertion_boundaries_use_polymer_order(tmp_path):
    path = tmp_path / "input.cif"
    _fixture(path)
    geometry = build_domain_geometry(_domain(), path,
        segments=[_segment(start=10, start_ins="A", end=12)])
    assert [(r.auth_seq_id, r.insertion_code) for r in geometry.residues] == [(10, "A"), (11, ""), (12, "")]
    assert len(geometry.beta_segments) == 1


def test_discontinuous_segments_never_bridge_strand_or_beta_runs(tmp_path):
    path = tmp_path / "input.cif"
    _fixture(path)
    geometry = build_domain_geometry(_domain(), path,
        segments=[_segment(end=10), _segment(start=12)])
    assert [len(s.residue_indices) for s in geometry.beta_segments] == [2, 4]
    assert len(_beta_runs(geometry)) == 2
    assert "clipped_ladder_pairing_unavailable" in geometry.warnings


def test_multichain_sheet_preserves_ladder_sense_without_linear_order(tmp_path):
    path = tmp_path / "input.cif"
    _fixture(path)
    geometry = build_domain_geometry(_domain(), path,
        segments=[_segment(), _segment("AA", 1, 4)])
    assert {res.chain_id for res in geometry.residues} == {"A", "AA"}
    assert geometry.sheet_connections == (("B001", "B002", "anti-parallel"),)
    signature = extract_signature(geometry)
    assert signature.features["betlas_parse_ok"] == 1
    assert signature.features["betlas_antiparallel_sheet_order_fraction"] == 1.0
    assert math.isnan(signature.features["betlas_sheet_order_nonlocal_fraction"])
    assert signature.features["betlas_sequence_order_status"] == "unavailable_across_chains"
    assert not any(name.startswith("betlas_rule_score_") for name in signature.features)


@pytest.mark.parametrize("segments,match", [
    ([], "at least one"), ([_segment(end=99)], "boundary"),
    ([_segment("B")], "boundary"), ([_segment(start=15, end=9)], "reversed"),
    ([_segment(), _segment(start=12)], "overlap"),
])
def test_bad_domain_boundaries_fail_without_whole_chain_fallback(tmp_path, segments, match):
    path = tmp_path / "input.cif"
    _fixture(path)
    with pytest.raises(ValueError, match=match):
        build_domain_geometry(_domain(), path, segments=segments)


def test_no_beta_is_coverage_status_not_parse_exception(tmp_path):
    path = tmp_path / "input.cif"
    _fixture(path, sheets=False)
    geometry = build_domain_geometry(_domain(), path, segments=[_segment()])
    assert "no_dssp_beta_segments" in geometry.warnings
    assert extract_signature(geometry).features["betlas_parse_ok"] == 0


def test_missing_ca_splits_polymer_continuity(tmp_path):
    path = tmp_path / "input.cif"
    data = _fixture(path)
    for name in list(data):
        if name.startswith("_atom_site."):
            data[name] = [value for i, value in enumerate(data[name]) if i != 3]
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    geometry = build_domain_geometry(_domain(), path, segments=[_segment()])
    assert [len(s.residue_indices) for s in geometry.beta_segments] == [3, 4]
    assert len(_beta_runs(geometry)) == 2


def test_branching_ladder_graph_does_not_create_total_order(tmp_path):
    path = tmp_path / "input.cif"
    _fixture(path)
    geometry = build_domain_geometry(_domain(), path, segments=[_segment(), _segment("AA", 1, 4)])
    base = geometry.beta_segments[0]
    geometry = replace(geometry,
        beta_segments=(*geometry.beta_segments, replace(base, element_id="B003"), replace(base, element_id="B004")),
        sheet_connections=(("B001", "B002", "parallel"), ("B001", "B003", "anti-parallel"),
                           ("B001", "B004", "parallel")))
    features = extract_signature(geometry).features
    assert features["betlas_dssp_ladder_edge_count"] == 3
    assert math.isnan(features["betlas_sheet_order_inversion_fraction_mean"])


def test_multimodel_annotation_input_is_rejected_by_geometry(tmp_path):
    path = tmp_path / "input.cif"
    _fixture(path, models=True)
    with pytest.raises(ValueError, match="exactly one model"):
        build_domain_geometry(_domain(), path, segments=[_segment()])


def test_annotation_selects_full_model_and_preserves_existing_output_on_failure(tmp_path, monkeypatch):
    source, output = tmp_path / "input.cif", tmp_path / "output.cif"
    _fixture(source, models=True)
    binary = tmp_path / "mkdssp"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    observed = {}

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "mkdssp version 4.6.1\n", "")
        selected = read_mmcif_dict(Path(command[-2]))
        observed.update(selected)
        Path(command[-1]).write_bytes(Path(command[-2]).read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = annotate_structure(source, output, dssp_binary=binary, model_id=1)
    assert report["source_model_num"] == "2"
    assert set(observed["_atom_site.auth_asym_id"]) == {"A", "AA"}
    assert set(observed["_atom_site.pdbx_PDB_model_num"]) == {"1"}
    assert min(map(float, observed["_atom_site.Cartn_x"])) > 100
    before = output.read_bytes()

    def failing_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "mkdssp version 4.6.1\n", "")
        return subprocess.CompletedProcess(command, 1, "", "synthetic failure")

    monkeypatch.setattr(subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        annotate_structure(source, output, dssp_binary=binary)
    assert output.read_bytes() == before
