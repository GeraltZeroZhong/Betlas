"""Export exact domain coordinates directly from local source mmCIF files.

No annotation or feature-extraction checkpoint is required. The complete output
ledger can feed structure baselines without conditioning their input on DSSP.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from betlas.models import DomainCandidate
from betlas.provenance import package_version, sha256_file

RESULT_COLUMNS = (
    "record_id", "pdb_id", "structure_path", "export_status", "export_reason",
    "atom_count", "model_id", "structure_sha256", "requested_residue_count",
    "missing_boundary_ca_count", "source_model_num",
)


def _source_contract() -> dict[str, str]:
    """Resume consumes these identities to avoid mixing coordinate definitions."""
    package = Path(__file__).resolve().parents[1]
    names = (
        "benchmarks/coordinate_exports.py", "io/domain_geometry.py", "io/mmcif.py",
        "models.py", "constants.py", "provenance.py",
    )
    return {name: sha256_file(package / name) for name in names}


def _read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"record_id", "pdb_id", "structure_path", "coordinate_status", "segments_json", "boundary_status"}
    if required - set(frame):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(frame))}")
    if frame.empty or frame.record_id.str.strip().eq("").any() or frame.record_id.duplicated().any():
        raise ValueError("Manifest must contain unique, nonempty domain records")
    if frame.pdb_id.str.strip().eq("").any():
        raise ValueError("Manifest pdb_id must be nonempty")
    ready = frame.coordinate_status.eq("available")
    if ready.any() and ("coordinate_sha256" not in frame or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None for value in frame.loc[ready, "coordinate_sha256"]
    )):
        raise ValueError("Available sources require a complete lowercase coordinate_sha256")
    for _, group in frame.groupby("pdb_id", sort=False):
        if group.structure_path.nunique() > 1 or ("coordinate_sha256" in group and group.coordinate_sha256.nunique() > 1):
            raise ValueError("Each PDB must have one coordinate source and content identity")
    return frame


def _empty_result(row: dict[str, Any], model_id: int) -> dict[str, Any]:
    result = dict.fromkeys(RESULT_COLUMNS, "")
    result.update(record_id=row["record_id"], pdb_id=row["pdb_id"],
                  export_status="pending", model_id=model_id)
    if (row["coordinate_status"] != "available" or not row["structure_path"]
            or row.get("representative_status", "") == "unavailable"):
        result.update(export_status="input_unavailable", export_reason="Coordinate source unavailable")
    elif row["boundary_status"] not in {"ok", "requires_coordinate_order"}:
        result.update(export_status="input_unavailable", export_reason="Boundary unavailable: " + row["boundary_status"])
    return result


def _write_ledger(frame: pd.DataFrame, results: dict[str, dict], target: Path) -> pd.DataFrame:
    exported = pd.DataFrame([results[identifier] for identifier in frame.record_id], columns=RESULT_COLUMNS)
    table = frame.merge(exported, on=["record_id", "pdb_id"], how="left",
                        suffixes=("_source", ""), validate="one_to_one")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix=".exports-", dir=target.parent,
                                     encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            table.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return exported


def export_coordinate_domains(
    manifest: Path, out_dir: Path, *, model_id: int = 0,
    limit_domains: int | None = None, resume: bool = False,
) -> pd.DataFrame:
    """Sequentially export domain atoms with a frozen, restartable provenance.

    Limits count new/retried coordinate exports, not verified cached successes.
    Resume revalidates source and output bytes; failed records are retried within
    the limit. SHA-256 identities control reuse rather than merely documenting it.
    All other records remain in the ledger with empty baseline structure paths.
    """
    if model_id < 0 or (limit_domains is not None and limit_domains < 1):
        raise ValueError("model_id must be nonnegative and limit_domains positive")
    manifest, out_dir = Path(manifest).resolve(), Path(out_dir).resolve()
    frame = _read_manifest(manifest)
    contract = {
        "schema": "betlas.raw-domain-coordinate-export.v1",
        "manifest_sha256": sha256_file(manifest), "model_id": model_id,
        "source": _source_contract(), "biopython": package_version("biopython"),
        "numpy": package_version("numpy"),
    }
    checkpoint = out_dir / "coordinate-export.sqlite"
    if checkpoint.exists() and not resume:
        raise ValueError("Checkpoint exists; choose --resume or a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = frame.to_dict("records")
    results = {row["record_id"]: _empty_result(row, model_id) for row in records}
    with sqlite3.connect(checkpoint) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS results (record_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        stored = connection.execute("SELECT payload FROM metadata WHERE key='contract'").fetchone()
        if stored is not None and json.loads(stored[0]) != contract:
            raise ValueError("Resume contract changed: manifest, model, implementation or environment differs")
        connection.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (json.dumps(contract, sort_keys=True),))
        connection.commit()
        existing = {key: json.loads(payload) for key, payload in connection.execute("SELECT record_id,payload FROM results")}
        verified_sources: dict[Path, tuple[str, str]] = {}
        attempted = 0
        try:
            for row in records:
                identifier = row["record_id"]
                result = results[identifier]
                target = out_dir / "coordinates" / (identifier.encode("utf-8").hex() + ".cif")
                if result["export_status"] == "pending" and (
                    limit_domains is None or attempted < limit_domains
                    or existing.get(identifier, {}).get("export_status") == "ok"
                ):
                    source = (manifest.parent / Path(row["structure_path"]).expanduser()).resolve()
                    if source not in verified_sources:
                        try:
                            verified_sources[source] = (sha256_file(source), "")
                        except OSError as exc:
                            verified_sources[source] = ("", f"{type(exc).__name__}: {exc}")
                    source_digest, source_error = verified_sources[source]
                    if source_error:
                        result.update(export_status="input_unavailable", export_reason=source_error)
                    elif source_digest != row["coordinate_sha256"]:
                        result.update(export_status="coordinate_identity_failed",
                                      export_reason="Coordinate bytes differ from the frozen manifest")
                    else:
                        previous = existing.get(identifier, {})
                        reusable = False
                        if previous.get("export_status") == "ok" and target.is_file():
                            reusable = sha256_file(target) == previous.get("structure_sha256")
                        if reusable:
                            result = {**previous, "structure_path": str(target)}
                            results[identifier] = result
                        elif limit_domains is None or attempted < limit_domains:
                            attempted += 1
                            try:
                                from betlas.io.domain_geometry import export_domain_from_source

                                domain = DomainCandidate.from_mapping({
                                    **row, "domain_id": row.get("domain_id", identifier), "model_id": model_id,
                                })
                                metadata = export_domain_from_source(
                                    domain, source, target, segments=json.loads(row["segments_json"]), model_id=model_id,
                                )
                                result.update(export_status="ok", structure_path=str(target),
                                              structure_sha256=sha256_file(target), atom_count=metadata["atom_count"])
                                for key in ("requested_residue_count", "missing_boundary_ca_count", "source_model_num"):
                                    result[key] = metadata.get(key, "")
                            except Exception as exc:
                                result.update(export_status="mapping_failed", structure_path="",
                                              export_reason=f"{type(exc).__name__}: {exc}")
                with connection:
                    connection.execute("INSERT OR REPLACE INTO results VALUES (?, ?)",
                                       (identifier, json.dumps(result, sort_keys=True)))
        finally:
            exported = _write_ledger(frame, results, out_dir / "domain_structures.csv")
    return exported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument("--limit-domains", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    exported = export_coordinate_domains(**vars(args))
    print(json.dumps({"domains": len(exported), "statuses": exported.export_status.value_counts().to_dict()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
