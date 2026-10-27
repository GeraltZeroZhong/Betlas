"""Exact-domain export and worker checks against a synthetic SQLite cache."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from Bio.PDB import MMCIFIO

from betlas.benchmarks import preparation
from betlas.io.mmcif import read_mmcif_dict
from betlas.provenance import sha256_file


def _annotation(path: Path) -> bytes:
    columns = {
        "group_PDB": [], "id": [], "type_symbol": [], "label_atom_id": [],
        "auth_atom_id": [], "label_comp_id": [], "auth_comp_id": [],
        "auth_asym_id": [], "label_asym_id": [], "auth_seq_id": [], "label_seq_id": [],
        "pdbx_PDB_ins_code": [], "label_alt_id": [], "occupancy": [],
        "Cartn_x": [], "Cartn_y": [], "Cartn_z": [], "pdbx_PDB_model_num": [],
    }
    for chain, label_chain in (("A", "X"), ("a", "Y")):
        for residue in range(1, 5):
            for atom in ("CA", "N"):
                row = ("ATOM", str(len(columns["id"]) + 1), "C" if atom == "CA" else "N",
                       atom, atom, "ALA", "ALA", chain, label_chain, str(residue), str(residue),
                       "?", ".", "1.0", str(100 + residue), "0", "0" if chain == "A" else "5", "1")
                for key, value in zip(columns, row, strict=True):
                    columns[key].append(value)
    data = {"data_": "synthetic"}
    data.update({f"_atom_site.{key}": value for key, value in columns.items()})
    data["_dssp_struct_summary.label_seq_id"] = ["1"]
    for key, values in {"id": ["first", "second"], "sheet_id": ["S", "S"],
                        "beg_label_asym_id": ["X", "Y"], "end_label_asym_id": ["X", "Y"],
                        "beg_label_seq_id": ["1", "1"], "end_label_seq_id": ["4", "4"]}.items():
        data[f"_struct_sheet_range.{key}"] = values
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    return path.read_bytes()


@pytest.fixture
def cached_domains(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "_source_contract", lambda: {"synthetic.py": "fixed"})
    source = tmp_path / "source.cif"
    source.write_text("Synthetic coordinate identity; cached annotation is used.\n")
    payload = _annotation(tmp_path / "annotation.cif")
    rows = []
    for record_id, chain, start, end, code in (
        ("9toyA00", "A", 1, 2, "9toy"), ("9toya00", "a", 3, 4, "9toy"),
        ("9toyA01", "A", 1, 99, "9toy"), ("9misA00", "A", 1, 2, "9mis"),
    ):
        rows.append({"record_id": record_id, "domain_id": record_id, "pdb_id": code,
                     "chain_id": chain, "structure_path": str(source),
                     "coordinate_sha256": sha256_file(source), "coordinate_status": "available",
                     "boundary_status": "ok", "cath_architecture_code": "2.10",
                     "segments_json": json.dumps([{"chain_id": chain, "start": start, "start_ins": "",
                                                    "end": end, "end_ins": ""}])})
    manifest, checkpoint = tmp_path / "domains.csv", tmp_path / "checkpoint.sqlite"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    contract = {"manifest_sha256": sha256_file(manifest), "source": preparation._source_contract(), "model_id": 1}
    with sqlite3.connect(checkpoint) as conn:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("CREATE TABLE annotations (pdb_id TEXT PRIMARY KEY, content BLOB NOT NULL, metadata TEXT NOT NULL)")
        conn.execute("INSERT INTO metadata VALUES ('contract', ?)", (json.dumps(contract),))
        conn.execute("INSERT INTO annotations VALUES (?,?,?)", ("9toy", gzip.compress(payload), json.dumps({"model_id": 1})))
    return manifest, checkpoint, rows, payload


def test_export_preserves_case_exact_crops_and_failed_records(cached_domains, tmp_path):
    manifest, checkpoint, _rows, _payload = cached_domains
    out = tmp_path / "export"
    result = preparation.export_domains(manifest, checkpoint, out).set_index("record_id")
    successful = result[result.export_status == "ok"]
    assert set(successful.index) == {"9toyA00", "9toya00"}
    paths = [Path(path) for path in successful.structure_path]
    assert len({path.name.casefold() for path in paths}) == 2
    for record_id, expected_chain, expected_ids in (("9toyA00", "A", {"1", "2"}), ("9toya00", "a", {"3", "4"})):
        path = Path(result.loc[record_id, "structure_path"])
        assert bytes.fromhex(path.stem).decode() == record_id
        data = read_mmcif_dict(path)
        assert set(data["_atom_site.auth_asym_id"]) == {expected_chain}
        assert set(data["_atom_site.auth_seq_id"]) == expected_ids
        assert set(data["_atom_site.label_atom_id"]) == {"CA", "N"}
        assert result.loc[record_id, "atom_count"] == 4
    assert result.loc["9toyA01", "export_status"] == "mapping_failed"
    assert result.loc["9misA00", "export_status"] == "annotation_unavailable"
    assert result.loc["9toyA01", "structure_path"] == ""
    assert result.loc["9misA00", "structure_path"] == ""
    table = pd.read_csv(out / "domain_structures.csv", keep_default_na=False).set_index("record_id")
    assert len(table) == 4 and table.index.is_unique
    # Failed records must never retain the whole-structure source as baseline input.
    assert table.loc["9misA00", "structure_path"] == ""
    assert table.loc["9misA00", "structure_path_source"] != ""


@pytest.mark.parametrize("changed", ["manifest", "implementation"])
def test_export_rejects_changed_frozen_contract(cached_domains, tmp_path, monkeypatch, changed):
    manifest, checkpoint, _rows, _payload = cached_domains
    if changed == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    else:
        monkeypatch.setattr(preparation, "_source_contract", lambda: {"synthetic.py": "changed"})
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="differs"):
        preparation.export_domains(manifest, checkpoint, out)
    assert not (out / "coordinates").exists()


def test_bounded_export_retains_unprocessed_domains_as_pending(cached_domains, tmp_path):
    manifest, checkpoint, _rows, _payload = cached_domains
    result = preparation.export_domains(manifest, checkpoint, tmp_path / "export", limit_domains=1)
    assert result.export_status.tolist() == ["ok", "pending", "pending", "pending"]
    assert result.loc[result.export_status == "pending", "structure_path"].eq("").all()


def test_worker_uses_cache_exact_domains_and_requested_model(cached_domains, tmp_path, monkeypatch):
    _manifest, _checkpoint, rows, payload = cached_domains
    from betlas.io import domain_geometry

    def forbidden(*args, **kwargs):
        pytest.fail("A compatible cached annotation must not execute DSSP again")

    monkeypatch.setattr(domain_geometry, "annotate_structure", forbidden)
    result = preparation._extract_structure_job({
        "rows": rows[:3], "cached_annotation": gzip.compress(payload), "cached_meta": {"model_id": 1},
        "work_dir": str(tmp_path), "dssp_binary": "not-executed", "model_id": 1, "timeout": 1,
    })
    outputs = {row["record_id"]: row for row in result["rows"]}
    assert outputs["9toyA00"]["feature_status"] == "ok"
    assert outputs["9toya00"]["feature_status"] == "ok"
    assert outputs["9toyA01"]["feature_status"] == "mapping_failed"
    assert outputs["9toyA00"]["observed_residue_count"] == 2
    assert outputs["9toya00"]["domain_sequence"] == "AA"
    assert outputs["9toya00"]["model_id"] == 1
    assert result["metadata"]["model_id"] == 1
    assert gzip.decompress(result["annotation"]) == payload


def test_worker_annotation_receives_requested_model(cached_domains, tmp_path, monkeypatch):
    _manifest, _checkpoint, rows, payload = cached_domains
    from betlas.io import domain_geometry
    called = []

    def annotate(source, target, **kwargs):
        called.append(kwargs["model_id"])
        target.write_bytes(payload)
        return {"model_id": kwargs["model_id"], "source_model_num": "2"}

    monkeypatch.setattr(domain_geometry, "annotate_structure", annotate)
    result = preparation._extract_structure_job({
        "rows": rows[:1], "cached_annotation": None, "cached_meta": {},
        "work_dir": str(tmp_path), "dssp_binary": "not-executed", "model_id": 1, "timeout": 1,
    })
    assert called == [1]
    assert result["rows"][0]["feature_status"] == "ok"
    assert result["rows"][0]["model_id"] == 1


@pytest.mark.parametrize("fail_features", [False, True])
def test_multichain_worker_preserves_separate_sequences_without_artificial_junction(
    cached_domains, tmp_path, monkeypatch, fail_features,
):
    _manifest, _checkpoint, rows, payload = cached_domains
    from betlas.features import extract

    row = {**rows[0], "segments_json": json.dumps([
        {"chain_id": "A", "start": 1, "start_ins": "", "end": 2, "end_ins": ""},
        {"chain_id": "a", "start": 3, "start_ins": "", "end": 4, "end_ins": ""},
    ])}
    if fail_features:
        def failing_signature(*args, **kwargs):
            raise ValueError("Synthetic geometry feature failure")
        monkeypatch.setattr(extract, "extract_signature", failing_signature)
    result = preparation._extract_structure_job({
        "rows": [row], "cached_annotation": gzip.compress(payload), "cached_meta": {"model_id": 1},
        "work_dir": str(tmp_path), "dssp_binary": "not-executed", "model_id": 1, "timeout": 1,
    })
    output = result["rows"][0]
    assert output["feature_status"] == ("extraction_failed" if fail_features else "ok")
    assert output["domain_sequence"] == ""
    assert json.loads(output["chain_sequences_json"]) == {"A": "AA", "a": "AA"}
    assert output["domain_chain_count"] == 2
    assert output["sequence_status"] == "multichain_requires_explicit_pooling"
