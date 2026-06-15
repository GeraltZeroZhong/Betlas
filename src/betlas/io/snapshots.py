from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from ..constants import CATH_URLS, DEFAULT_CATH_DIR, DEFAULT_MMCIF_DIR
from ..provenance import build_run_manifest, file_state, write_json
from .rcsb import mmcif_path_for


def _count_gzip_data_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                count += 1
    return count


def cath_source_snapshot(cath_dir: str | Path = DEFAULT_CATH_DIR) -> dict[str, Any]:
    base = Path(cath_dir)
    filenames = {
        "all": "cath-b-newest-all.gz",
        "names": "cath-b-newest-names.gz",
        "s35": "cath-b-s35-newest.gz",
    }
    files: dict[str, Any] = {}
    for key, filename in filenames.items():
        path = base / filename
        state = file_state(path)
        state["url"] = CATH_URLS[key]
        state["data_line_count"] = _count_gzip_data_lines(path) if state["exists"] else 0
        files[key] = state
    return {
        "source": "CATH-B daily-release newest",
        "release_policy": "daily-release/newest; freeze this manifest for publication archives",
        "cath_dir": str(base),
        "files": files,
    }


def _pdb_ids_from_labels(labels_csv: str | Path | None) -> list[str]:
    if labels_csv is None:
        return []
    labels = pd.read_csv(labels_csv, usecols=["pdb_id"], dtype=str, keep_default_na=False)
    return sorted({str(value).strip().lower() for value in labels["pdb_id"] if str(value).strip()})


def _scan_mmcif_ids(mmcif_dir: Path) -> list[str]:
    return sorted(path.name.removesuffix(".cif.gz").lower() for path in mmcif_dir.glob("*.cif.gz"))


def mmcif_source_snapshot(
    mmcif_dir: str | Path = DEFAULT_MMCIF_DIR,
    *,
    pdb_ids: Iterable[str] | None = None,
    labels_csv: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(mmcif_dir)
    ids = sorted({str(pdb_id).strip().lower() for pdb_id in (pdb_ids or []) if str(pdb_id).strip()})
    if not ids:
        ids = _pdb_ids_from_labels(labels_csv) or _scan_mmcif_ids(base)

    entries: dict[str, Any] = {}
    for pdb_id in ids:
        path = mmcif_path_for(pdb_id, base)
        state = file_state(path)
        state["url"] = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
        state["pdb_id"] = pdb_id
        state["status"] = "present" if state["exists"] and state["size"] > 0 else "missing"
        entries[pdb_id] = state

    present = sum(1 for entry in entries.values() if entry["status"] == "present")
    return {
        "source": "RCSB PDB mmCIF",
        "mmcif_dir": str(base),
        "labels_csv": str(labels_csv) if labels_csv is not None else "",
        "requested_count": len(ids),
        "present_count": present,
        "missing_count": len(ids) - present,
        "files": entries,
    }


def build_source_snapshot(
    *,
    cath_dir: str | Path = DEFAULT_CATH_DIR,
    mmcif_dir: str | Path = DEFAULT_MMCIF_DIR,
    labels_csv: str | Path | None = None,
) -> dict[str, Any]:
    return build_run_manifest(
        command="betlas source-snapshot",
        parameters={
            "cath_dir": str(cath_dir),
            "mmcif_dir": str(mmcif_dir),
            "labels_csv": str(labels_csv) if labels_csv is not None else "",
        },
        inputs={"labels_csv": labels_csv} if labels_csv is not None else {},
        metrics={},
        extra={
            "cath": cath_source_snapshot(cath_dir),
            "mmcif": mmcif_source_snapshot(mmcif_dir, labels_csv=labels_csv),
        },
    )


def write_source_snapshot(
    output_path: str | Path,
    *,
    cath_dir: str | Path = DEFAULT_CATH_DIR,
    mmcif_dir: str | Path = DEFAULT_MMCIF_DIR,
    labels_csv: str | Path | None = None,
) -> Path:
    return write_json(
        output_path,
        build_source_snapshot(cath_dir=cath_dir, mmcif_dir=mmcif_dir, labels_csv=labels_csv),
    )


def enrich_feature_table_with_mmcif_provenance(
    features_csv: str | Path,
    output_csv: str | Path,
    *,
    mmcif_dir: str | Path = DEFAULT_MMCIF_DIR,
) -> pd.DataFrame:
    features = pd.read_csv(features_csv, dtype=str, keep_default_na=False)
    base = Path(mmcif_dir)
    states: dict[str, dict[str, Any]] = {}
    for pdb_id in sorted({str(value).strip().lower() for value in features["pdb_id"] if str(value).strip()}):
        states[pdb_id] = file_state(mmcif_path_for(pdb_id, base))

    features["source_mmcif_path"] = features["pdb_id"].map(
        lambda value: states.get(str(value).strip().lower(), {}).get("path", "")
    )
    features["source_mmcif_sha256"] = features["pdb_id"].map(
        lambda value: states.get(str(value).strip().lower(), {}).get("sha256", "")
    )
    features["source_mmcif_size"] = features["pdb_id"].map(
        lambda value: states.get(str(value).strip().lower(), {}).get("size", 0)
    )
    features["source_mmcif_exists"] = features["pdb_id"].map(
        lambda value: int(bool(states.get(str(value).strip().lower(), {}).get("exists", False)))
    )

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return features
