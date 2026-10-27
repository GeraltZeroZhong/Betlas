from __future__ import annotations

import json
import math

import numpy as np
import pytest
from Bio.PDB import MMCIFIO

from betlas.geometry_relations.scaffold_alignment import (
    _fit_rigid,
    export_coordinates,
    load_graphs,
    measure_scaffold,
    parse_residue_correspondence,
    parse_transform,
    run_scaffold_alignments,
)


def _residues(xyz, *, chain="B", first=100):
    return [{"coordinate": list(map(float, coord)), "chain_id": chain,
             "author_residue_number": first + i, "insertion_code": "",
             "residue_name": "ALA", "label_seq_id": i + 1,
             "residue_uid": f"{chain}:{first + i}:", "index0": i}
            for i, coord in enumerate(xyz)]


def _graph(residues, groups):
    return {"nodes": [{"node_id": name, "sheet_id": sheet, "chain_id": residues[ids[0]]["chain_id"],
                       "residue_uids": [residues[i]["residue_uid"] for i in ids],
                       "label_positions": [residues[i]["label_seq_id"] for i in ids]}
                      for name, sheet, ids in groups], "pairing_edges": []}


def _pairs(mapping):
    return [{"query_index0": i, "target_index0": j} for i, j in mapping]


IDENTITY = {"rotation": np.eye(3).tolist(), "translation": [0., 0., 0.]}


def test_native_transform_orientation_and_reflection_rejection():
    matrix = "0 2 0 -1 0\n1 3 1 0 0\n2 4 0 0 1\n"
    parsed = parse_transform(matrix)
    assert np.array([1., 0., 0.]) @ np.array(parsed["rotation"]).T + parsed["translation"] == pytest.approx([2, 4, 4])
    with pytest.raises(ValueError, match="proper rigid"):
        parse_transform("0 0 -1 0 0\n1 0 0 1 0\n2 0 0 0 1")
    with pytest.raises(ValueError, match="complete"):
        parse_transform("0 0 1 0 0")


def test_nonsequential_native_correspondences_use_actual_identity_not_display_sequence():
    q = _residues(np.zeros((4, 3)))
    t = _residues(np.zeros((4, 3)), chain="C", first=800)
    stdout = "misleading sequence display AAAA\n" + "\n".join(
        f" CA  ALA A{i:4d} \t CA  ALA A{j:4d} \t 1.000" for i, j in [(4, 1), (2, 2), (1, 3)])
    parsed = parse_residue_correspondence(stdout, q, t)
    assert [(p["query_index0"], p["target_index0"]) for p in parsed] == [(3, 0), (1, 1), (0, 2)]
    assert parsed[0]["query_residue_uid"] == "B:103:"
    with pytest.raises(ValueError, match="one-to-one"):
        parse_residue_correspondence(stdout + "\n CA  ALA A   4 \t CA  ALA A   4 \t 1.000", q, t)


def test_beta_coverage_uses_two_full_denominators_and_beta_to_beta_only():
    q = _residues([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    t = _residues([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    qg = _graph(q, [("Q", "s", [0, 1, 2, 3])])
    tg = _graph(t, [("T", "s", [0, 1])])
    r = measure_scaffold(q, t, _pairs([(0, 0), (1, 1), (2, 2), (3, 3)]), IDENTITY, qg, tg)
    assert r["summary"]["query_beta_coverage_2"] == .5
    assert r["summary"]["target_beta_coverage_2"] == 1
    assert r["summary"]["min_full_strand_coverage_2"] == 0
    assert r["summary"]["beta_to_beta_pair_count"] == 2
    assert r["summary"]["fixed_transform_query_tm_style_beta_contribution"] == 1
    assert r["summary"]["fixed_transform_target_tm_style_other_contribution"] == pytest.approx(.4)


def test_distance_threshold_curves_and_no_beta_unavailable():
    q = _residues([[0, 0, 0], [0, 1, 0], [0, 2, 0]])
    t = _residues([[1, 0, 0], [3, 1, 0], [4, 2, 0]])
    qg, tg = _graph(q, [("Q", "a", [0, 1, 2])]), _graph(t, [("T", "b", [0, 1, 2])])
    r = measure_scaffold(q, t, _pairs([(i, i) for i in range(3)]), IDENTITY, qg, tg)["summary"]
    assert [r[f"min_beta_coverage_{s}"] for s in ["2", "3p5", "5"]] == pytest.approx([1/3, 2/3, 1])
    r = measure_scaffold(q, t, [], IDENTITY, {"nodes": []}, tg)["summary"]
    assert r["min_beta_coverage_2"] is None
    assert r["query_strand_coverage_2"] is None


def test_spatial_strand_mapping_preserves_partial_and_split_correspondence():
    q = _residues([[i, 0, 0] for i in range(6)])
    t = _residues([[i, 0, 0] for i in range(6)])
    qg = _graph(q, [("Q1", "a", [0, 1, 2, 3]), ("Q2", "a", [4, 5])])
    tg = _graph(t, [("T1", "b", [0, 1]), ("T2", "b", [2, 3]), ("T3", "b", [4, 5])])
    calls = []
    def compare(left, right, mapping):
        calls.append(mapping)
        return {"mapping_complete": False}
    r = measure_scaffold(q, t, _pairs([(i, i) for i in range(6)]), IDENTITY, qg, tg, graph_comparator=compare)
    assert r["spatial_strand_mapping"] == {"Q2": "T3"}
    assert r["summary"]["query_split_strand_fraction"] == .5
    assert calls == [{"Q2": "T3"}]
    assert not r["graph_mapping_comparison"]["mapping_complete"]
    assert not r["summary"]["graph_mapping_complete"]


def test_fragmentation_breaks_at_observed_polymer_gaps():
    q = _residues([[i, 0, 0] for i in range(4)])
    t = _residues([[i, 0, 0] for i in range(4)])
    for r, position in zip(q, [1, 2, 8, 9], strict=True):
        r["label_seq_id"] = position
    qg, tg = _graph(q, [("Q", "a", [0, 1, 2, 3])]), _graph(t, [("T", "a", [0, 1, 2, 3])])
    r = measure_scaffold(q, t, _pairs([(i, i) for i in range(4)]), IDENTITY, qg, tg)
    assert r["strand_correspondences"][0]["correspondence_run_count"] == 2
    assert r["strand_correspondences"][0]["longest_run_residues"] == 2


def test_independent_sheet_fit_does_not_imply_joint_scaffold_fit():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 5], [1, 0, 5], [0, 1, 5]], dtype=float)
    target = xyz.copy()
    target[3:] += [4, 0, 0]
    q, t = _residues(xyz), _residues(target)
    groups = [("s1", "a", [0, 1, 2]), ("s2", "b", [3, 4, 5])]
    r = measure_scaffold(q, t, _pairs([(i, i) for i in range(6)]), IDENTITY, _graph(q, groups), _graph(t, groups))
    assert r["summary"]["independent_sheet_fit_rmsd"] == pytest.approx(0, abs=1e-12)
    assert r["summary"]["common_sheet_fit_rmsd"] > .1
    assert r["summary"]["sheet_fit_residue_pair_count"] == 6
    assert r["summary"]["whole_sheet_bijection"]
    assert r["summary"]["one_to_one_sheet_fit_status"] == "multiple_sheets"
    assert r["summary"]["one_to_one_sheet_assembly_rmsd_gap"] > .1
    assert _fit_rigid([[0,0,0], [1,0,0], [2,0,0]], [[0,0,0], [1,0,0], [2,0,0]]) is None


def test_split_sheet_fits_cannot_masquerade_as_whole_sheet_assembly():
    xyz = [[0,0,0], [1,0,0], [0,1,0], [5,0,0], [6,0,0], [5,1,0]]
    q, t = _residues(xyz), _residues(xyz)
    qg = _graph(q, [("Q", "whole", list(range(6)))])
    tg = _graph(t, [("T1", "left", [0,1,2]), ("T2", "right", [3,4,5])])
    result = measure_scaffold(q, t, _pairs([(i,i) for i in range(6)]), IDENTITY, qg, tg)
    summary = result["summary"]
    assert summary["fitted_sheet_pair_count"] == 2
    assert summary["query_split_sheet_count"] == 1
    assert not summary["whole_sheet_bijection"]
    assert summary["one_to_one_sheet_pair_count"] == 0
    assert summary["one_to_one_sheet_assembly_rmsd_gap"] is None


def test_annotation_identity_mismatch_is_not_silently_a_nonbeta_residue():
    q = _residues([[0,0,0], [1,0,0], [0,1,0]])
    graph = _graph(q, [("Q", "a", [0,1,2])])
    graph["nodes"][0]["label_positions"][0] = 99
    with pytest.raises(ValueError, match="polymer identities"):
        measure_scaffold(q, q, _pairs([(0,0)]), IDENTITY, graph, graph)


def _write_cif(tmp_path, name="q", *, alternate=False):
    xyz = [[i * 2.7, math.sin(i), math.cos(i)] for i in range(8)]
    data = {"data_": name, "_atom_site.group_PDB": ["ATOM"] * 8,
            "_atom_site.id": [str(i+1) for i in range(8)], "_atom_site.type_symbol": ["C"] * 8,
            "_atom_site.label_atom_id": ["CA"] * 8, "_atom_site.label_alt_id": ["."] * 8,
            "_atom_site.label_comp_id": ["ALA"] * 8, "_atom_site.label_asym_id": ["AB"] * 8,
            "_atom_site.label_seq_id": [str(i+20) for i in range(8)],
            "_atom_site.auth_seq_id": [str(i-4) for i in range(8)],
            "_atom_site.auth_asym_id": ["AB"] * 8, "_atom_site.pdbx_PDB_ins_code": ["?"] * 8,
            "_atom_site.pdbx_PDB_model_num": ["1"] * 8, "_atom_site.occupancy": ["1.0"] * 8}
    for column, axis in zip(["x", "y", "z"], range(3), strict=True):
        data[f"_atom_site.Cartn_{column}"] = [f"{r[axis]:.7f}" for r in xyz]
    if alternate:
        for key in data:
            if key.startswith("_atom_site."):
                data[key].append(data[key][0])
        data["_atom_site.label_alt_id"][0] = "A"
        data["_atom_site.label_alt_id"][-1] = "B"
        data["_atom_site.occupancy"][0] = ".8"
        data["_atom_site.occupancy"][-1] = ".2"
        data["_atom_site.Cartn_x"][0] = "0.1234567"
        data["_atom_site.Cartn_x"][-1] = "99.1"
    path = tmp_path / f"{name}.cif"
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    return {"record_id": name, "structure_path": str(path), "export_status": "ok"}


def test_export_preserves_precision_altloc_and_identity_despite_execution_renumbering(tmp_path):
    manifest = [_write_cif(tmp_path, alternate=True)]
    prepared = export_coordinates(["q"], manifest, tmp_path / "coordinates")
    q = prepared["q"]
    assert q["status"] == "ok"
    assert q["ca_identities"][0]["coordinate"][0] == .1234567
    assert q["ca_identities"][0]["residue_uid"] == "AB:-4:"
    assert q["ca_identities"][0]["label_seq_id"] == 20
    assert q["source_multiple_altloc_ca_count"] == 1
    assert "0.1234567" in open(q["execution_path"]).read()
    assert json.loads(open(q["identity_map_path"]).read())["ca_count"] == 8


def _fake(tmp_path, *, fail=False, no_mapping=False, timeout=False):
    binary = tmp_path / "USalign"
    header = "Length of Structure_1: 8 residues\nLength of Structure_2: 8 residues\nAligned length= 8, RMSD= 0.00, Seq_ID=n_identical/n_aligned= 1.000\nTM-score= 1.0 (normalized by length of Structure_1)\nTM-score= 1.0 (normalized by length of Structure_2)\n"
    mapping = "\n".join(f" CA  ALA A{i:4d} \t CA  ALA A{i:4d} \t 0.000" for i in range(1,9))
    binary.write_text("#!/usr/bin/env python3\nimport sys,pathlib,time\n"
                      "if '-v' in sys.argv: print('USalign fixture');sys.exit(0)\n"
                      + ("time.sleep(1)\n" if timeout else "")
                      + ("sys.exit(2)\n" if fail else "")
                      + "pathlib.Path(sys.argv[sys.argv.index('-m')+1]).write_text('0 0 1 0 0\\n1 0 0 1 0\\n2 0 0 0 1\\n')\n"
                      + f"print({header + ('' if no_mapping else mapping)!r})\n")
    binary.chmod(0o755)
    return str(binary)


@pytest.mark.parametrize("options,status", [({}, "ok"), ({"fail": True}, "execution_failed"),
    ({"no_mapping": True}, "alignment_parse_failed"), ({"timeout": True}, "timeout")])
def test_runner_preserves_modes_outputs_and_failure_states(tmp_path, options, status):
    manifest = [_write_cif(tmp_path)]
    prepared = export_coordinates(["q"], manifest, tmp_path / "prepared")
    residues = prepared["q"]["ca_identities"]
    graph = _graph(residues, [("Q", "s", list(range(8)))])
    output = tmp_path / "run"
    results = run_scaffold_alignments([{"pair_id": "self", "query_id": "q", "target_id": "q"}],
        manifest, {"q": graph}, output, modes=("SQ", "sNS", "fNS"), binary=_fake(tmp_path, **options),
        prepared_coordinates=prepared, timeout=.15 if options.get("timeout") else 3)
    assert [r["status"] for r in results] == [status] * 3
    assert [r["mode"] for r in results] == ["SQ", "sNS", "fNS"]
    for result in results:
        assert (output / result["stdout_path"]).exists()
        assert (output / result["stderr_path"]).exists()
        assert (output / result["matrix_path"]).exists()
        if status == "ok":
            assert result["min_beta_coverage_2"] == 1
            detail = json.loads((output / result["correspondence_path"]).read_text())
            assert len(detail["residue_correspondences"]) == 8
        else:
            assert result.get("min_beta_coverage_2") is None


def test_graph_jsonl_wrappers_are_read_without_changing_node_ids(tmp_path):
    path = tmp_path / "graphs.jsonl"
    path.write_text(json.dumps({"record_id": "q", "graph": {"nodes": [{"node_id": "B001"}]}}) + "\n")
    assert load_graphs(path, ["q"])["q"]["nodes"][0]["node_id"] == "B001"
    assert load_graphs(path, ["absent"]) == {}


def test_missing_annotation_keeps_alignment_without_fabricating_beta_coverage(tmp_path):
    manifest = [_write_cif(tmp_path)]
    out = tmp_path / "run"
    results = run_scaffold_alignments([{"query_id": "q", "target_id": "q"}], manifest,
                                      {}, out, modes=("SQ",), binary=_fake(tmp_path))
    row = results[0]
    assert row["status"] == "annotation_unavailable"
    assert row["alignment_status"] == "ok"
    assert row["min_tm_score"] == 1
    assert row.get("min_beta_coverage_2") is None
    raw = json.loads((out / row["correspondence_path"]).read_text())
    assert len(raw["residue_correspondences"]) == 8
    assert raw["transform"] == IDENTITY
