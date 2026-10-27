"""Raw-coordinate export provenance and coverage with a synthetic backend."""

import json
from pathlib import Path

import pandas as pd
import pytest

from betlas.benchmarks import coordinate_exports
from betlas.io import domain_geometry
from betlas.provenance import sha256_file


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    source = tmp_path / "source.cif"
    source.write_text("Synthetic source coordinate bytes\n")
    wrong = tmp_path / "wrong.cif"
    wrong.write_text("Changed coordinate bytes\n")
    rows = []
    for identifier, code, path, status, digest in (
        ("9toyA00", "9toy", source, "available", sha256_file(source)),
        ("9toya00", "9toy", source, "available", sha256_file(source)),
        ("9toyF00", "9toy", source, "available", sha256_file(source)),
        ("9misA00", "9mis", "", "unavailable", ""),
        ("9badA00", "9bad", wrong, "available", "0" * 64),
    ):
        rows.append({"record_id": identifier, "domain_id": identifier, "pdb_id": code,
                     "chain_id": identifier[4], "structure_path": str(path),
                     "coordinate_status": status, "coordinate_sha256": digest,
                     "boundary_status": "ok", "cath_architecture_code": "2.10",
                     "segments_json": json.dumps([{"chain_id": identifier[4], "start": 1,
                                                    "start_ins": "", "end": 2, "end_ins": ""}])})
    manifest = tmp_path / "domains.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    calls = []

    def export(domain, raw_mmcif, target, *, segments, model_id):
        calls.append((domain.record_id, raw_mmcif, segments, model_id))
        assert domain.model_id == model_id
        if domain.record_id == "9toyF00":
            raise ValueError("Synthetic domain mapping failure")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{domain.record_id} model {model_id}\n")
        return {"atom_count": 4, "requested_residue_count": 2,
                "missing_boundary_ca_count": 0, "source_model_num": str(model_id + 1)}

    def forbidden(*args, **kwargs):
        pytest.fail("Raw domain exports must never execute DSSP")

    monkeypatch.setattr(domain_geometry, "export_domain_from_source", export)
    monkeypatch.setattr(domain_geometry, "annotate_structure", forbidden)
    return manifest, rows, calls


def test_bounded_raw_export_and_resume_keep_complete_ledger(catalog, tmp_path):
    manifest, rows, calls = catalog
    out = tmp_path / "exports"
    partial = coordinate_exports.export_coordinate_domains(manifest, out, model_id=2, limit_domains=1)
    assert partial.export_status.tolist() == ["ok", "pending", "pending", "input_unavailable", "pending"]
    assert len(calls) == 1
    result = coordinate_exports.export_coordinate_domains(manifest, out, model_id=2, resume=True)
    assert result.export_status.tolist() == ["ok", "ok", "mapping_failed", "input_unavailable", "coordinate_identity_failed"]
    assert [call[0] for call in calls] == ["9toyA00", "9toya00", "9toyF00"]
    assert all(call[3] == 2 for call in calls)
    assert calls[0][2] == json.loads(rows[0]["segments_json"])
    success = result[result.export_status.eq("ok")]
    paths = [Path(path) for path in success.structure_path]
    assert len({path.name.casefold() for path in paths}) == 2
    assert {bytes.fromhex(path.stem).decode() for path in paths} == {"9toyA00", "9toya00"}
    assert all(result.loc[result.export_status.ne("ok"), "structure_path"].eq(""))
    table = pd.read_csv(out / "domain_structures.csv", keep_default_na=False)
    assert table.record_id.tolist() == [row["record_id"] for row in rows]
    assert table.loc[0, "structure_path_source"] == rows[0]["structure_path"]
    assert table.loc[2, "structure_path"] == ""


def test_resume_checks_output_bytes_and_reexports_corrupted_success(catalog, tmp_path):
    manifest, _rows, calls = catalog
    out = tmp_path / "exports"
    result = coordinate_exports.export_coordinate_domains(manifest, out, limit_domains=1)
    target = Path(result.iloc[0].structure_path)
    target.write_text("Corrupted output\n")
    restored = coordinate_exports.export_coordinate_domains(manifest, out, limit_domains=1, resume=True)
    assert [call[0] for call in calls] == ["9toyA00", "9toyA00"]
    assert restored.iloc[0].structure_sha256 == sha256_file(target)
    assert restored.iloc[1].export_status == "pending"


def test_resume_does_not_reuse_success_if_source_bytes_changed(catalog, tmp_path):
    manifest, rows, calls = catalog
    out = tmp_path / "exports"
    coordinate_exports.export_coordinate_domains(manifest, out, limit_domains=1)
    Path(rows[0]["structure_path"]).write_text("Changed original source\n")
    result = coordinate_exports.export_coordinate_domains(manifest, out, resume=True)
    assert set(result.iloc[:3].export_status) == {"coordinate_identity_failed"}
    assert result.iloc[:3].structure_path.eq("").all()
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["manifest", "model", "implementation"])
def test_resume_rejects_changed_contract(catalog, tmp_path, monkeypatch, change):
    manifest, _rows, calls = catalog
    out = tmp_path / "exports"
    coordinate_exports.export_coordinate_domains(manifest, out, limit_domains=1)
    kwargs = {}
    if change == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    elif change == "model":
        kwargs["model_id"] = 1
    else:
        monkeypatch.setattr(coordinate_exports, "_source_contract", lambda: {"changed": "implementation"})
    with pytest.raises(ValueError, match="Resume contract changed"):
        coordinate_exports.export_coordinate_domains(manifest, out, resume=True, **kwargs)
    assert len(calls) == 1


def test_requires_digest_and_explicit_resume(catalog, tmp_path):
    manifest, _rows, _calls = catalog
    out = tmp_path / "exports"
    coordinate_exports.export_coordinate_domains(manifest, out, limit_domains=1)
    with pytest.raises(ValueError, match="--resume"):
        coordinate_exports.export_coordinate_domains(manifest, out)
    frame = pd.read_csv(manifest, keep_default_na=False)
    frame.loc[0, "coordinate_sha256"] = ""
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="coordinate_sha256"):
        coordinate_exports.export_coordinate_domains(manifest, tmp_path / "other")


def test_cli_exports_without_annotation_checkpoint(catalog, tmp_path, capsys):
    manifest, _rows, calls = catalog
    assert coordinate_exports.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "cli"),
                                    "--model-id", "1", "--limit-domains", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["statuses"]["ok"] == 1
    assert calls[0][3] == 1
