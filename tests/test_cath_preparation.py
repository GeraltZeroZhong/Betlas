"""Offline preparation and resume checks with synthetic source records."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from betlas.benchmarks import preparation


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_catalog_selects_representatives_from_input_qc_and_preserves_all_domains(tmp_path: Path) -> None:
    domains, boundaries, names = (tmp_path / name for name in ("domains.txt", "boundaries.txt", "names.txt"))
    domains.write_text(
        "9aaaA00 2 10 20 30 1 1 1 1 1 40 1.0\n"
        "9aabA01 2 10 20 30 1 1 1 1 2 39 2.0\n"
        "9aacA00 2 10 20 30 2 1 1 1 1 20 1.5\n"
        "9aadA01 2 10 20 30 3 1 1 1 1 20 2.0\n"
    )
    boundaries.write_text(
        "9aaaA D01 F00 1 A 1 - A 40 -\n"
        "9aabA D01 F00 1 A 1 - A 39 -\n"
        "9aacA D01 F00 1 A 80 - A 10 -\n"
    )
    names.write_text(
        "2 9aaaA00 :Mainly Beta\n2.10 9aaaA00 :Example architecture\n"
        "2.10.20 9aaaA00 :Example topology\n2.10.20.30 9aaaA00 :\n"
    )
    index_rows = []
    for code in ("9aaa", "9aab", "9aac", "9aad"):
        coordinate = tmp_path / f"{code}.cif"
        coordinate.write_text("synthetic input, never executed\n")
        index_rows.append({"pdb_id": code, "state": "downloaded", "path": coordinate.name,
                           "sha256": _digest(coordinate)})
    # The append-only index's final state overrides an earlier success.
    index_rows.append({"pdb_id": "9AAA", "state": "failed"})
    index = tmp_path / "coordinates.jsonl"
    index.write_text("\n".join(json.dumps(row) for row in index_rows) + "\n")
    out = tmp_path / "catalog"
    frame = preparation.make_catalog(domains, boundaries, names, index, out)
    assert len(frame) == 4
    assert set(frame.previously_used) == {"unknown"}
    reps = pd.read_csv(out / "representatives.csv", keep_default_na=False).set_index("record_id")
    assert set(reps.index) == {"9aabA01", "9aacA00", "9aadA01"}
    assert reps.loc["9aabA01", "official_representative_id"] == "9aaaA00"
    assert reps.loc["9aabA01", "representative_source"] == "availability_fallback"
    assert reps.loc["9aacA00", "boundary_status"] == "requires_coordinate_order"
    assert reps.loc["9aacA00", "representative_status"] == "available"
    assert reps.loc["9aadA01", "representative_status"] == "unavailable"
    assert frame.set_index("record_id").loc["9aaaA00", "coordinate_status"] == "unavailable"
    assert reps.loc["9aabA01", "structure_path"] == str(tmp_path / "9aab.cif")
    assert pd.read_csv(out / "architecture_support.csv").domain_count.sum() == 4
    metadata = json.loads((out / "catalog.json").read_text())
    assert metadata["domains"] == 4 and metadata["representatives"] == 3


@pytest.fixture
def extraction_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(preparation, "_source_contract", lambda: {"synthetic.py": "fixed-code-state"})
    monkeypatch.setattr(preparation, "_binary_version", lambda _: "synthetic DSSP version")
    rows = []
    for code, ids in (("9aaa", ["9aaaA01", "9aaaA02"]), ("9aab", ["9aabA01"])):
        source = tmp_path / f"{code}.cif"
        source.write_text(f"synthetic coordinate {code}\n")
        for record_id in ids:
            rows.append({
                "record_id": record_id, "domain_id": record_id, "pdb_id": code,
                "chain_id": "A", "cath_architecture_code": "2.10", "cath_topology_code": "2.10.20",
                "structure_path": str(source), "coordinate_status": "available",
                "coordinate_sha256": _digest(source), "boundary_status": "ok",
                "segments_json": json.dumps([{"chain_id": "A", "start": 1, "start_ins": "", "end": 20, "end_ins": ""}]),
            })
    rows.append({**rows[0], "record_id": "9aacA01", "domain_id": "9aacA01", "pdb_id": "9aac",
                 "structure_path": "", "coordinate_status": "unavailable", "coordinate_sha256": ""})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest, tmp_path / "output", tmp_path / "fake-dssp"


def _job_result(job: dict, *, status: str = "ok", annotation: bool = True) -> dict:
    code = job["rows"][0]["pdb_id"]
    return {
        "pdb_id": code,
        "annotation": gzip.compress(b"synthetic annotated coordinates", mtime=0) if annotation else None,
        "metadata": {"source": "synthetic"},
        "rows": [{"record_id": row["record_id"], "pdb_id": code, "feature_status": status,
                  "feature_reason": "" if status == "ok" else "synthetic failure",
                  "betlas_parse_ok": int(status == "ok"), "betlas_example": 2.0,
                  "cath_architecture_code": "incorrect worker metadata"} for row in job["rows"]],
    }


def test_bounded_extraction_keeps_pending_then_resumes_without_duplicates(
    extraction_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, out, binary = extraction_inputs
    seen = []

    def execute(job):
        seen.append(job["rows"][0]["pdb_id"])
        return _job_result(job)

    monkeypatch.setattr(preparation, "_extract_structure_job", execute)
    first = preparation.extract_catalog(manifest, out, dssp_binary=binary, limit_pdb=1)
    assert first["status_counts"] == {"ok": 2, "pending": 1, "data_unavailable": 1}
    assert first["complete"] is False
    table = pd.read_csv(out / "features.csv", dtype=str)
    assert len(table) == 4 and table.record_id.is_unique
    assert set(table.cath_architecture_code) == {"2.10"}
    second = preparation.extract_catalog(manifest, out, dssp_binary=binary, resume=True)
    assert second["status_counts"] == {"ok": 3, "data_unavailable": 1}
    assert second["complete"] is True
    assert seen == ["9aaa", "9aab"]
    third = preparation.extract_catalog(manifest, out, dssp_binary=binary, resume=True)
    assert third["structures_processed_this_run"] == 0
    assert seen == ["9aaa", "9aab"]
    with sqlite3.connect(out / "checkpoint.sqlite") as db:
        assert db.execute("SELECT count(*) FROM results").fetchone()[0] == 4
        assert db.execute("SELECT count(*) FROM annotations").fetchone()[0] == 2


@pytest.mark.parametrize("change", ["manifest", "model", "implementation", "binary"])
def test_changed_resume_contract_rejected_before_jobs(
    extraction_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    manifest, out, binary = extraction_inputs
    monkeypatch.setattr(preparation, "_extract_structure_job", _job_result)
    preparation.extract_catalog(manifest, out, dssp_binary=binary, limit_pdb=1)
    options = {}
    if change == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    elif change == "model":
        options["model_id"] = 1
    elif change == "implementation":
        monkeypatch.setattr(preparation, "_source_contract", lambda: {"synthetic.py": "changed"})
    else:
        monkeypatch.setattr(preparation, "_binary_version", lambda _: "different DSSP")

    def forbidden(_):
        pytest.fail("Changed contract must be rejected before any job")

    monkeypatch.setattr(preparation, "_extract_structure_job", forbidden)
    with pytest.raises(ValueError, match="Resume contract changed"):
        preparation.extract_catalog(manifest, out, dssp_binary=binary, resume=True, **options)


@pytest.mark.parametrize("failure, cached", [("annotation_failed", False), ("extraction_failed", True)])
def test_failed_jobs_require_retry_and_reuse_successful_annotation(
    extraction_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, failure: str, cached: bool,
) -> None:
    manifest, out, binary = extraction_inputs
    calls = []

    def fail(job):
        calls.append(job)
        return _job_result(job, status=failure, annotation=cached)

    monkeypatch.setattr(preparation, "_extract_structure_job", fail)
    preparation.extract_catalog(manifest, out, dssp_binary=binary, limit_pdb=1)
    assert len(calls) == 1
    monkeypatch.setattr(preparation, "_extract_structure_job", _job_result)
    # Ordinary resume processes the previously pending structure, not failed domains.
    preparation.extract_catalog(manifest, out, dssp_binary=binary, resume=True)
    table = pd.read_csv(out / "features.csv").set_index("record_id")
    assert table.loc["9aaaA01", "feature_status"] == failure

    def recover(job):
        assert job["rows"][0]["pdb_id"] == "9aaa"
        assert (job["cached_annotation"] is not None) is cached
        if cached:
            assert gzip.decompress(job["cached_annotation"]) == b"synthetic annotated coordinates"
            assert job["cached_meta"] == {"source": "synthetic"}
        return _job_result(job)

    monkeypatch.setattr(preparation, "_extract_structure_job", recover)
    final = preparation.extract_catalog(manifest, out, dssp_binary=binary, resume=True, retry_failed=True)
    assert final["status_counts"] == {"ok": 3, "data_unavailable": 1}


def test_completed_transactions_survive_interrupted_invocation(
    extraction_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, out, binary = extraction_inputs

    def interrupt(job):
        if job["rows"][0]["pdb_id"] == "9aab":
            raise RuntimeError("synthetic process interruption")
        return _job_result(job)

    monkeypatch.setattr(preparation, "_extract_structure_job", interrupt)
    with pytest.raises(RuntimeError, match="process interruption"):
        preparation.extract_catalog(manifest, out, dssp_binary=binary)
    with sqlite3.connect(out / "checkpoint.sqlite") as db:
        assert db.execute("SELECT count(*) FROM results").fetchone()[0] == 3
    resumed = []

    def recover(job):
        resumed.extend(row["record_id"] for row in job["rows"])
        return _job_result(job)

    monkeypatch.setattr(preparation, "_extract_structure_job", recover)
    final = preparation.extract_catalog(manifest, out, dssp_binary=binary, resume=True)
    assert final["complete"] is True
    assert resumed == ["9aabA01"]


def test_available_coordinates_require_content_identity(
    extraction_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, out, binary = extraction_inputs
    frame = pd.read_csv(manifest, keep_default_na=False)
    frame.loc[frame.pdb_id.eq("9aaa"), "coordinate_sha256"] = ""
    frame.to_csv(manifest, index=False)
    monkeypatch.setattr(preparation, "_extract_structure_job", _job_result)
    with pytest.raises(ValueError, match="(?i)(sha256|identity|checksum|digest)"):
        preparation.extract_catalog(manifest, out, dssp_binary=binary)


def test_worker_rejects_changed_source_before_annotation_or_cached_geometry(
    extraction_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from betlas.io import domain_geometry

    manifest, out, binary = extraction_inputs
    row = pd.read_csv(manifest, dtype=str, keep_default_na=False).iloc[0].to_dict()
    Path(row["structure_path"]).write_text("changed coordinate content\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Changed coordinate bytes must be rejected before annotation or geometry")

    monkeypatch.setattr(domain_geometry, "annotate_structure", forbidden)
    monkeypatch.setattr(domain_geometry, "build_domain_geometry", forbidden)
    result = preparation._extract_structure_job({
        "rows": [row], "cached_annotation": gzip.compress(b"old annotated content"), "cached_meta": {},
        "work_dir": str(out), "dssp_binary": str(binary), "model_id": 0, "timeout": 10,
    })
    assert result["annotation"] is None
    assert result["rows"][0]["feature_status"] != "ok"
    assert "Coordinate bytes differ" in result["rows"][0]["feature_reason"]
