from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from betlas.io.rcsb import mmcif_path_for, validate_pdb_id
from betlas.io.snapshots import (
    cath_source_snapshot,
    enrich_feature_table_with_mmcif_provenance,
    mmcif_source_snapshot,
)


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_source_snapshots_record_cath_and_mmcif_hashes(tmp_path: Path) -> None:
    cath_dir = tmp_path / "cath"
    _write_gzip(cath_dir / "cath-b-newest-all.gz", "# comment\n1abcA00 v4 2.40.10.10 1-10:A\n")
    _write_gzip(cath_dir / "cath-b-newest-names.gz", "2.40.10.10 Example\n")
    _write_gzip(cath_dir / "cath-b-s35-newest.gz", "1abcA00 v4 2.40.10.10.1\n")

    mmcif_dir = tmp_path / "mmcif"
    mmcif_dir.mkdir()
    (mmcif_dir / "1abc.cif.gz").write_bytes(b"example")
    labels = tmp_path / "labels.csv"
    pd.DataFrame([{"pdb_id": "1abc"}, {"pdb_id": "9zzz"}]).to_csv(labels, index=False)

    cath = cath_source_snapshot(cath_dir)
    mmcif = mmcif_source_snapshot(mmcif_dir, labels_csv=labels)

    assert cath["files"]["all"]["data_line_count"] == 1
    assert len(cath["files"]["all"]["sha256"]) == 64
    assert mmcif["present_count"] == 1
    assert mmcif["missing_count"] == 1
    assert len(mmcif["files"]["1abc"]["sha256"]) == 64


def test_enrich_feature_table_adds_row_level_mmcif_hash(tmp_path: Path) -> None:
    mmcif_dir = tmp_path / "mmcif"
    mmcif_dir.mkdir()
    (mmcif_dir / "1abc.cif.gz").write_bytes(b"example")
    features = tmp_path / "features.csv"
    out = tmp_path / "features_out.csv"
    pd.DataFrame([{"record_id": "r1", "pdb_id": "1abc"}]).to_csv(features, index=False)

    df = enrich_feature_table_with_mmcif_provenance(features, out, mmcif_dir=mmcif_dir)

    assert out.exists()
    assert df.loc[0, "source_mmcif_exists"] == 1
    assert len(str(df.loc[0, "source_mmcif_sha256"])) == 64


def test_mmcif_path_rejects_pdb_id_path_traversal(tmp_path: Path) -> None:
    assert validate_pdb_id("1ABC") == "1abc"
    assert mmcif_path_for("1abc", tmp_path) == tmp_path / "1abc.cif.gz"

    for bad in ["../../outside/evil", "/abs", "1abc/evil", "abc", "abcde"]:
        with pytest.raises(ValueError, match="invalid pdb_id"):
            mmcif_path_for(bad, tmp_path)


def test_snapshot_inputs_reject_unsafe_pdb_ids(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    pd.DataFrame([{"pdb_id": "../../outside"}]).to_csv(labels, index=False)

    with pytest.raises(ValueError, match="invalid pdb_id"):
        mmcif_source_snapshot(tmp_path / "mmcif", labels_csv=labels)

    features = tmp_path / "features.csv"
    out = tmp_path / "features_out.csv"
    pd.DataFrame([{"record_id": "r1", "pdb_id": "../../outside"}]).to_csv(features, index=False)
    with pytest.raises(ValueError, match="invalid pdb_id"):
        enrich_feature_table_with_mmcif_provenance(features, out, mmcif_dir=tmp_path / "mmcif")
