from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import pytest
from Bio.PDB import MMCIFIO

from betlas.geometry_relations.validation import (
    _canonical_data,
    _coordinate_data,
    build_review_package,
    inspect_coordinates,
    parse_alignment_output,
    run_pair_alignments,
    summarize_validation,
)


def _output(query=8, target=12, aligned=6, *, kind="Structure"):
    return f"""Name of {kind}_1: query.cif:A
Name of {kind}_2: target.cif:A
Length of {kind}_1: {query} residues
Length of {kind}_2: {target} residues
Aligned length= {aligned}, RMSD= 1.25, Seq_ID=n_identical/n_aligned= 0.250
TM-score= 0.60000 (normalized by length of {kind}_1: L={query}, d0=0.50)
TM-score= 0.40000 (normalized by length of {kind}_2: L={target}, d0=0.50)
"""


def _pdb(path: Path, n=8, *, chain="A", shift=0, extra_chain=False):
    lines = ["HEADER    RESEARCH_ID_MUST_NOT_REACH_REVIEWER\n"]
    for index in range(n):
        # Non-collinear geometry; periodic variation also avoids a straight-line fit.
        x, y, z = index * 2.4 + shift, 4 * math.sin(index), 3 * math.cos(index * .6)
        lines.append(f"ATOM  {index + 1:5d}  CA  ALA {chain}{index + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n")
    if extra_chain:
        lines.append("ATOM    999  CA  ALA B   1       1.000   2.000   3.000  1.00 20.00           C  \n")
    lines.append("END\n")
    path.write_text("".join(lines))
    return {"record_id": path.stem, "structure_path": str(path), "export_status": "ok"}


def _fake(tmp_path, *, stdout=None, returncode=0, pause=0):
    binary = tmp_path / "USalign"
    binary.write_text("#!/usr/bin/env python3\nimport sys,time\n"
                      "if '-v' in sys.argv:\n print('US-align test fixture');sys.exit(0)\n"
                      f"time.sleep({pause!r})\nprint({stdout!r})\nsys.exit({returncode})\n")
    binary.chmod(0o755)
    return str(binary)


@pytest.mark.parametrize("kind", ["Structure", "Chain"])
def test_parser_preserves_both_normalizations_and_length_asymmetry(kind):
    result = parse_alignment_output(_output(kind=kind))
    assert result["query_tm_score"] == .6
    assert result["target_tm_score"] == .4
    assert result["min_tm_score"] == .4
    assert result["query_coverage"] == .75
    assert result["target_coverage"] == .5
    assert result["short_long_length_ratio"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("text", ["", "No alignment found", _output(aligned=13),
                                  _output().replace("0.60000", "1.6"),
                                  _output().replace("0.60000", "nan"),
                                  _output().split("TM-score=")[0]])
def test_invalid_or_absent_alignments_never_become_zero_distance(text):
    with pytest.raises(ValueError):
        parse_alignment_output(text)


def test_runner_preserves_raw_output_and_all_requested_pairs(tmp_path):
    manifest = [_pdb(tmp_path / "a.pdb"), _pdb(tmp_path / "b.pdb", 12)]
    pairs = [{"pair_id": "valid", "query_id": "a", "target_id": "b"},
             {"pair_id": "absent", "query_id": "a", "target_id": "missing"}]
    binary = _fake(tmp_path, stdout=_output())
    output = tmp_path / "alignments"
    rows = run_pair_alignments(pairs, manifest, output, binary=binary, workers=2)
    assert [row["status"] for row in rows] == ["ok", "input_unavailable"]
    assert rows[1]["query_tm_score"] is None
    assert (output / rows[0]["stdout_path"]).read_text().strip() == _output().strip()
    inspection = json.loads((output / "coordinate-inspection.json").read_text())
    assert inspection[0]["ca_count"] == 8
    assert len(inspection[0]["ca_identities"]) == 8
    assert not Path(inspection[0]["execution_path"]).exists()
    command = json.loads((output / rows[0]["command_path"]).read_text())
    assert command["argv"][-4:] == ["-mol", "prot", "-mm", "0"]
    assert "-fast" not in command["argv"]
    assert "-TMcut" not in command["argv"]
    assert len(list(csv.DictReader((output / "pair-alignments.csv").open()))) == 2


@pytest.mark.parametrize("stdout,exitcode,pause,status", [
    (_output(query=7), 0, 0, "input_length_mismatch"),
    ("fatal", 2, 0, "execution_failed"),
    ("No alignment", 0, 0, "parse_failed"),
    (_output(), 0, .4, "timeout"),
])
def test_failed_or_partial_native_outputs_remain_explicit(tmp_path, stdout, exitcode, pause, status):
    manifest = [_pdb(tmp_path / "a.pdb"), _pdb(tmp_path / "b.pdb", 12)]
    binary = _fake(tmp_path, stdout=stdout, returncode=exitcode, pause=pause)
    rows = run_pair_alignments([{"query_id": "a", "target_id": "b"}], manifest,
                              tmp_path / "output", binary=binary, timeout=.1)
    assert rows[0]["status"] == status
    assert (tmp_path / "output" / rows[0]["stdout_path"]).exists()
    if status != "input_length_mismatch":
        assert rows[0]["min_tm_score"] is None


def test_multichain_domain_is_not_silently_reduced_or_concatenated(tmp_path):
    record = _pdb(tmp_path / "multi.pdb", extra_chain=True)
    inspected = inspect_coordinates(record["structure_path"])
    assert inspected["status"] == "unsupported_multiple_chains"
    assert inspected["ca_count"] == 9
    rows = run_pair_alignments([{"query_id": "multi", "target_id": "multi"}], [record],
                              tmp_path / "output", binary=_fake(tmp_path, stdout=_output()))
    assert rows[0]["status"] == "input_unavailable"
    assert rows[0]["returncode"] is None


def test_single_selected_altloc_is_usable_but_multiple_conformers_are_not(tmp_path):
    path = tmp_path / "alternative.pdb"
    _pdb(path)
    lines = path.read_text().splitlines(keepends=True)
    first = lines[1]
    lines[1] = first[:16] + "A" + first[17:]
    path.write_text("".join(lines))
    assert inspect_coordinates(path)["status"] == "ok"
    lines.insert(2, first[:6] + "  101" + first[11:16] + "B" + first[17:])
    path.write_text("".join(lines))
    assert inspect_coordinates(path)["status"] == "unresolved_ca_altloc"


def test_duplicate_keys_cannot_silently_replace_coordinate_or_pair(tmp_path):
    manifest = [_pdb(tmp_path / "a.pdb")]
    pair = {"pair_id": "duplicate", "query_id": "a", "target_id": "a"}
    binary = _fake(tmp_path, stdout=_output())
    with pytest.raises(ValueError, match="Duplicate coordinate"):
        run_pair_alignments([pair], manifest * 2, tmp_path / "a", binary=binary)
    with pytest.raises(ValueError, match="Duplicate pair"):
        run_pair_alignments([pair] * 2, manifest, tmp_path / "b", binary=binary)


def test_review_package_removes_identifiers_and_does_not_claim_human_validation(tmp_path):
    manifest = [_pdb(tmp_path / "private-domain-one.pdb", chain="Z"),
                _pdb(tmp_path / "private-domain-two.pdb", n=12)]
    pairs = [{"pair_id": "private-candidate", "query_id": "private-domain-one",
              "target_id": "private-domain-two", "cath_label": "do_not_export",
              "model_probability": .95}]
    output = tmp_path / "review"
    summary = build_review_package(pairs, manifest, output)
    assert summary["completed_independent_human_reviews"] == 0
    assert summary["review_status"] == "pending_human_review"
    combined = "".join(path.read_text() for path in (output / "reviewer").rglob("*") if path.is_file())
    for hidden in ("private-domain", "private-candidate", "do_not_export", "RESEARCH_ID_MUST_NOT"):
        assert hidden not in combined
    keys = list(csv.DictReader((output / "private" / "structure-key.csv").open()))
    assert {row["record_id"] for row in keys} == {row["record_id"] for row in manifest}
    for path in (output / "reviewer" / "coordinates").glob("*.cif"):
        coordinates = inspect_coordinates(path)
        assert coordinates["status"] == "ok"
        assert {row["chain_id"] for row in coordinates["ca_identities"]} == {"A"}


@pytest.mark.parametrize("aligner", ["USalign", "TMalign"])
def test_native_alignment_smoke_on_synthetic_rigid_transform(tmp_path, aligner):
    if shutil.which(aligner) is None:
        pytest.skip(f"Optional native {aligner} unavailable")
    manifest = [_pdb(tmp_path / "toy-original.pdb", n=24),
                _pdb(tmp_path / "toy-translated.pdb", n=24, shift=10)]
    rows = run_pair_alignments([{"query_id": "toy-original", "target_id": "toy-translated"}],
                              manifest, tmp_path / "native", binary=aligner, timeout=10)
    assert rows[0]["status"] == "ok"
    assert rows[0]["min_tm_score"] == pytest.approx(1, abs=1e-5)
    assert rows[0]["min_coverage"] == 1


def _summary_inputs():
    metadata = [
        {"pair_id": "c1", "query_id": "a", "target_id": "b", "relation_kind": "shape",
         "role": "candidate", "phase": "confirmation", "queryH": "h1", "targetH": "h2",
         "queryA": "a1", "targetA": "a2", "matched_pair_id": "m1"},
        {"pair_id": "n1", "query_id": "a", "target_id": "c", "relation_kind": "shape",
         "role": "control", "phase": "confirmation", "queryH": "h1", "targetH": "h3",
         "queryA": "a1", "targetA": "a2", "matched_pair_id": "m1"},
        {"pair_id": "c2", "query_id": "a", "target_id": "d", "relation_kind": "shape",
         "role": "candidate", "phase": "confirmation", "queryH": "h1", "targetH": "h4",
         "queryA": "a1", "targetA": "a2", "matched_pair_id": "m2"},
        {"pair_id": "n2", "query_id": "b", "target_id": "e", "relation_kind": "shape",
         "role": "control", "phase": "confirmation", "queryH": "h2", "targetH": "h5",
         "queryA": "a2", "targetA": "a3", "matched_pair_id": "m2"},
    ]
    alignments = []
    for index, row in enumerate(metadata):
        alignments.append({key: row[key] for key in ("pair_id", "query_id", "target_id")})
        alignments[-1].update(status="ok", reason="", min_tm_score=.8 - index * .1,
                              min_coverage=.9 - index * .1, short_long_length_ratio=.75)
    return alignments, metadata


def test_validation_summary_keeps_dependency_counts_and_declared_matched_differences(tmp_path):
    alignments, metadata = _summary_inputs()
    result = summarize_validation(alignments, metadata, tmp_path / "summary")
    assert result["n_pairs"] == 4
    assert result["n_unique_domains"] == 5
    assert result["n_reused_domains"] == 2
    assert result["max_pairs_per_domain"] == 3
    candidate_group = next(row for row in result["groups"] if row["role"] == "candidate")
    assert candidate_group["n_unique_homologies"] == 3
    assert candidate_group["n_reused_domains"] == 1
    assert candidate_group["n_pairs_with_reused_domain"] == 2
    assert candidate_group["min_tm_score_median"] == pytest.approx(.7)
    for contrast in result["matched_contrasts"]:
        assert contrast["contrast_kind"] == "one_to_one"
        assert contrast["min_tm_score_difference"] == pytest.approx(.1)
    assert (tmp_path / "summary" / "matched-contrasts.csv").is_file()
    assert not any("p_value" in key or "ci_" in key for row in result["groups"] for key in row)


def test_validation_summary_does_not_drop_missing_or_failed_matched_members():
    alignments, metadata = _summary_inputs()
    alignments[1]["status"] = "input_length_mismatch"
    alignments.pop()
    result = summarize_validation(alignments, metadata)
    assert result["n_pairs"] == 4
    assert result["n_available"] == 2
    assert result["statuses"] == {"ok": 2, "input_length_mismatch": 1, "alignment_missing": 1}
    assert {row["status"] for row in result["matched_contrasts"]} == {"incomplete_measurements"}
    assert all(row["min_tm_score_difference"] is None for row in result["matched_contrasts"])


def test_validation_summary_rejects_invalid_success_metrics_and_mislabeled_records():
    alignments, metadata = _summary_inputs()
    alignments[0]["min_tm_score"] = float("nan")
    result = summarize_validation(alignments, metadata)
    assert result["statuses"]["invalid_ok_record"] == 1
    alignments[0]["target_id"] = "different_domain"
    with pytest.raises(ValueError, match="identity differs"):
        summarize_validation(alignments, metadata)


def test_many_to_one_matched_contrasts_are_explicit_group_means_and_missing_roles_stay_na():
    alignments, metadata = _summary_inputs()
    metadata[2]["matched_pair_id"] = "m1"
    metadata[3]["matched_pair_id"] = "orphan_control"
    result = summarize_validation(alignments, metadata)
    matched = next(row for row in result["matched_contrasts"] if row["matched_pair_id"] == "m1")
    assert matched["contrast_kind"] == "difference_of_group_means"
    assert matched["n_candidates"] == 2
    assert matched["n_controls"] == 1
    assert matched["min_tm_score_difference"] == pytest.approx(0)
    orphan = next(row for row in result["matched_contrasts"] if row["matched_pair_id"] == "orphan_control")
    assert orphan["status"] == "missing_role"
    assert orphan["min_tm_score_difference"] is None


def test_validation_summary_accepts_csv_and_caller_defined_role_columns(tmp_path):
    alignments, metadata = _summary_inputs()
    for row in metadata:
        row["panel_role"] = row.pop("role")
    for filename, rows in (("alignments.csv", alignments), ("metadata.csv", metadata)):
        with (tmp_path / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = summarize_validation(tmp_path / "alignments.csv", tmp_path / "metadata.csv",
                                  role_column="panel_role", group_columns=("relation_kind", "panel_role", "phase"))
    assert result["n_available"] == 4
    assert result["matched_statuses"] == {"ok": 2}


def test_validation_summary_supports_matched_control_without_counting_panel_background():
    alignments, metadata = _summary_inputs()
    for row in metadata:
        if row["role"] == "control":
            row["role"] = "matched_control"
    metadata.append({"pair_id": "background", "query_id": "a", "target_id": "e",
                     "role": "panel_background", "relation_kind": "within_panel", "phase": "mixed",
                     "matched_pair_id": ""})
    alignments.append({"pair_id": "background", "query_id": "a", "target_id": "e", "status": "ok",
                       "min_tm_score": .1, "min_coverage": .2, "short_long_length_ratio": .4})
    result = summarize_validation(alignments, metadata, control_role="matched_control")
    assert result["n_matched_groups"] == 2
    assert result["matched_statuses"] == {"ok": 2}
    assert all(row["n_candidates"] == row["n_controls"] == 1 for row in result["matched_contrasts"])


@pytest.mark.parametrize("original_alt,original_occupancy,other_alt,other_occupancy,expected", [
    ("B", ".5", "A", ".5", 9.87654321),
    ("A", ".5", ".", ".5", 9.87654321),
    ("B", ".6", "A", ".4", 1.23456789),
])
def test_canonical_altloc_policy_matches_geometry_and_preserves_coordinate_tokens(
        tmp_path, original_alt, original_occupancy, other_alt, other_occupancy, expected):
    from betlas.io.domain_geometry import _ca_residues

    original = tmp_path / "original.pdb"
    _pdb(original)
    data = _coordinate_data(original)
    for key, values in data.items():
        if key.startswith("_atom_site."):
            values.append(values[0])
    data["_atom_site.id"][-1] = "999"
    for key, first, last in (("label_alt_id", original_alt, other_alt),
                             ("occupancy", original_occupancy, other_occupancy),
                             ("Cartn_x", "1.23456789", "9.87654321")):
        data[f"_atom_site.{key}"][0] = first
        data[f"_atom_site.{key}"][-1] = last
    source = tmp_path / "precision.cif"
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(source))
    canonical, inspection = _canonical_data(source)
    assert inspection["status"] == "ok"
    assert inspection["source_multiple_altloc_ca_count"] == 1
    assert inspection["ca_identities"][0]["coordinate"][0] == expected
    geometry, _ = _ca_residues(data, "synthetic")
    assert [row["coordinate"] for row in inspection["ca_identities"]] == [list(r.coord_ca) for r in geometry]
    assert canonical["_atom_site.Cartn_x"][0] == str(expected)
    assert set(canonical["_atom_site.label_alt_id"]) == {"."}
    manifest = [{"record_id": "precision", "structure_path": str(source), "export_status": "ok"}]
    result = build_review_package([{"query_id": "precision", "target_id": "precision"}], manifest,
                                  tmp_path / "anonymous")
    assert result["coordinate_statuses"] == {"ok": 1}
    anonymous = inspect_coordinates(tmp_path / "anonymous/reviewer/coordinates/S001.cif")
    assert anonymous["ca_identities"][0]["coordinate"][0] == expected
