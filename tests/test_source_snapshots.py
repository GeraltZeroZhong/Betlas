from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from betlas.io.snapshots import (
    cath_source_snapshot,
    enrich_feature_table_with_mmcif_provenance,
    mmcif_source_snapshot,
)
from betlas.readouts.beta_barrel_staves.schema import validate_beta_barrel_staves_dataset
from betlas.science_report import repair_feature_table_metadata


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


def test_beta_barrel_staves_dataset_validation_reports_missing_manifest(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "required": [
                    "record_id",
                    "pdb_id",
                    "auth_chain_id",
                    "evidence_level",
                    "confidence_score",
                    "strand_count_final",
                    "strand_count_beta_barrel_staves",
                    "qc_status",
                    "audit_manifest",
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = tmp_path / "rows.csv"
    pd.DataFrame(
        [
            {
                "record_id": "1abc_A",
                "pdb_id": "1abc",
                "auth_chain_id": "A",
                "evidence_level": "gold",
                "confidence_score": 1.0,
                "strand_count_final": 8,
                "strand_count_beta_barrel_staves": 8,
                "qc_status": "pass",
                "audit_manifest": "data/audit/missing.json",
                "raw_structure_sha256": "not-a-sha",
            }
        ]
    ).to_csv(rows, index=False)

    report = validate_beta_barrel_staves_dataset(rows, schema_path=schema, repo_root=tmp_path)

    assert report["row_count"] == 1
    assert report["severity_counts"]["warning"] == 1
    assert report["severity_counts"]["error"] == 1


def test_repair_feature_table_preserves_cath_code_strings(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.csv"
    out = tmp_path / "features_out.csv"
    pd.DataFrame(
        [
            {
                "record_id": "1abcA00",
                "pdb_id": "1abc",
                "cath_architecture_code": "2.40",
                "cath_topology_code": "2.40.10",
                "cath_homology_code": "2.40.10.10",
                "fold_label_final": "beta_barrel",
            }
        ]
    ).to_csv(labels, index=False)
    pd.DataFrame(
        [
            {
                "record_id": "1abcA00",
                "pdb_id": "1abc",
                "cath_architecture_code": "2.4",
                "cath_topology_code": "2.40.10",
                "cath_homology_code": "2.40.10.10",
                "fold_label_final": "beta_barrel",
                "cz_sheet_size_entropy": "-0.000000001",
            }
        ]
    ).to_csv(features, index=False)

    repaired = repair_feature_table_metadata(features, labels, out)

    assert repaired.loc[0, "cath_architecture_code"] == "2.40"
    assert repaired.loc[0, "cz_sheet_size_entropy"] == "0.0"
