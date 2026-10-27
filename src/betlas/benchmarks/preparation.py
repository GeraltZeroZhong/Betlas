"""Prepare a frozen CATH release and extract domain features from local coordinates.

Coordinate acquisition is a separate operation. This module never downloads data.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended

from betlas.models import DomainCandidate
from betlas.features.current import FEATURE_COLUMNS, FEATURE_SCHEMA
from betlas.provenance import package_version, sha256_file, utc_now_iso

SUCCESS_STATES = {"downloaded", "reused_byte_verified", "reused_existing_collection", "ok"}


def read_coordinate_index(path: Path) -> dict[str, dict[str, Any]]:
    """Read an append-only JSONL acquisition index, retaining the latest state."""
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("pdb_id", "")).strip().lower()
        if not key:
            raise ValueError(f"Coordinate index line {line_number} has no pdb_id")
        if row.get("path"):
            row["path"] = str((path.parent / Path(row["path"]).expanduser()).resolve())
        records[key] = row
    return records


def make_catalog(
    domain_list: Path, boundaries: Path, names: Path, coordinate_index: Path,
    out_dir: Path, *, version: str = "4.4.0", class_id: int = 2,
    official_representatives: Path | None = None, previously_used: Path | None = None,
) -> pd.DataFrame:
    from betlas.io.cath_release import (
        build_release_manifest,
        select_s35_representatives,
        summarize_support,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_release_manifest(domain_list, boundaries, names, class_id=class_id, version=version)
    coordinates = read_coordinate_index(coordinate_index)
    paths, states, digests = [], [], []
    for code in frame["pdb_id"]:
        source = coordinates.get(str(code).lower(), {})
        path = str(source.get("path", ""))
        ready = source.get("state") in SUCCESS_STATES and bool(path) and Path(path).is_file()
        paths.append(path if ready else "")
        states.append("available" if ready else "unavailable")
        digests.append(str(source.get("sha256", "")) if ready else "")
    frame["structure_path"] = paths
    frame["coordinate_status"] = states
    frame["coordinate_sha256"] = digests
    frame["previously_used"] = "unknown"
    if previously_used is not None:
        used = {s.strip() for s in previously_used.read_text().splitlines() if s.strip()}
        frame["previously_used"] = frame["domain_id"].map(lambda key: "true" if key in used else "false")
    # Representativeness is decided from source availability, before algorithm output.
    available = pd.Series(
        ((frame["coordinate_status"] == "available") & frame["boundary_status"].isin(["ok", "requires_coordinate_order"])).to_numpy(),
        index=frame["record_id"], dtype=bool,
    )
    representatives = select_s35_representatives(frame, official_representatives, availability=available)
    frame.to_csv(out_dir / "domains.csv", index=False)
    representatives.to_csv(out_dir / "representatives.csv", index=False)
    for level in ("architecture", "topology"):
        summarize_support(frame, level=level).to_csv(out_dir / f"{level}_support.csv", index=False)
    inputs = [domain_list, boundaries, names, coordinate_index]
    if official_representatives is not None:
        inputs.append(official_representatives)
    if previously_used is not None:
        inputs.append(previously_used)
    manifest = {
        "schema": "betlas.cath-catalog.v1", "created_utc": utc_now_iso(), "version": version,
        "class_id": class_id, "domains": len(frame), "representatives": len(representatives),
        "coordinate_states": dict(Counter(states)),
        "inputs": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in inputs],
        "representative_policy": "Official S35 representative first; explicit input-availability fallback before feature extraction.",
        "previously_used_policy": "unknown unless a caller supplies a complete prior domain-ID list",
    }
    (out_dir / "catalog.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return frame


def _source_contract() -> dict[str, str]:
    """Fingerprint implementation files so resume cannot mix feature definitions."""
    package = Path(__file__).resolve().parents[1]
    paths = {p for directory in ("features", "io", "grammars") for p in (package / directory).rglob("*.py")}
    paths.update(package / name for name in (
        "__init__.py", "models.py", "constants.py", "slicing.py", "provenance.py",
        "schema.py", "specs.py", "benchmarks/preparation.py",
    ))
    return {str(p.relative_to(package)): sha256_file(p) for p in sorted(paths) if p.is_file()}


def _binary_version(path: Path) -> str:
    result = subprocess.run([str(path), "--version"], check=True, capture_output=True, text=True, timeout=20)
    return (result.stdout + result.stderr).strip()


def _failure(row: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        **dict.fromkeys(FEATURE_COLUMNS, float("nan")), "feature_schema": FEATURE_SCHEMA,
        "record_id": row["record_id"], "pdb_id": row["pdb_id"],
        "feature_status": status, "feature_reason": reason, "betlas_parse_ok": 0,
    }


def _extract_structure_job(job: dict[str, Any]) -> dict[str, Any]:
    """One local structure, one annotation, all requested domain crops."""
    from betlas.features.extract import extract_signature
    from betlas.io.domain_geometry import annotate_structure, build_domain_geometry

    rows = job["rows"]
    first = rows[0]
    source = Path(first["structure_path"])
    try:
        expected = str(first.get("coordinate_sha256", ""))
        if expected and sha256_file(source) != expected:
            return {
                "pdb_id": first["pdb_id"], "annotation": None, "metadata": {},
                "rows": [_failure(row, "coordinate_identity_failed", "Coordinate bytes differ from the acquisition manifest") for row in rows],
            }
        with tempfile.TemporaryDirectory(prefix="structure-", dir=job["work_dir"]) as tmp:
            annotated = Path(tmp) / "annotated.cif"
            if job["cached_annotation"] is not None:
                annotated.write_bytes(gzip.decompress(job["cached_annotation"]))
                annotation_meta = job["cached_meta"]
            else:
                annotation_meta = annotate_structure(
                    source, annotated, dssp_binary=Path(job["dssp_binary"]),
                    model_id=job["model_id"], timeout=job["timeout"],
                )
            payload = gzip.compress(annotated.read_bytes(), mtime=0)
            # Temporary command paths are not stable cache identifiers.
            annotation_meta = {**annotation_meta, "input_sha256": expected, "model_id": job["model_id"]}
            results = []
            for row in rows:
                try:
                    domain = DomainCandidate.from_mapping({
                        **row, "domain_id": row.get("domain_id", row["record_id"]),
                        "fold_label_final": row.get("cath_architecture_code", ""),
                        "model_id": job["model_id"], "label_source_primary": "CATH fixed release",
                    })
                    geometry = build_domain_geometry(domain, annotated, segments=json.loads(row["segments_json"]))
                    if not geometry.residues:
                        results.append(_failure(row, "mapping_failed", "No selected coordinate residues"))
                        continue
                except Exception as exc:
                    results.append(_failure(row, "mapping_failed", f"{type(exc).__name__}: {exc}"))
                    continue
                chain_sequences: dict[str, str] = {}
                for residue in geometry.residues:
                    chain_sequences[residue.chain_id] = chain_sequences.get(residue.chain_id, "") + protein_letters_3to1_extended.get(residue.residue_name, "X")
                domain_metadata = {
                    "domain_sequence": next(iter(chain_sequences.values())) if len(chain_sequences) == 1 else "",
                    "chain_sequences_json": json.dumps(chain_sequences, sort_keys=True),
                    "domain_chain_count": len(chain_sequences),
                    "sequence_status": "observed_single_chain_domain" if len(chain_sequences) == 1 else "multichain_requires_explicit_pooling",
                    "observed_residue_count": len(geometry.residues),
                    "requested_residue_count": geometry.requested_residue_count,
                    "missing_boundary_ca_count": geometry.missing_boundary_ca_count,
                    "secondary_structure_source": "dssp", "model_id": job["model_id"],
                }
                try:
                    signature = extract_signature(geometry)
                    output = signature.to_row()
                    output.update(
                        record_id=row["record_id"], pdb_id=row["pdb_id"],
                        feature_status="ok" if geometry.beta_segments else "no_beta_segments",
                        feature_reason=";".join(geometry.warnings),
                        **domain_metadata,
                    )
                    results.append(output)
                except Exception as exc:
                    results.append({**_failure(row, "extraction_failed", f"{type(exc).__name__}: {exc}"), **domain_metadata})
            return {"pdb_id": first["pdb_id"], "annotation": payload, "metadata": annotation_meta, "rows": results}
    except Exception as exc:
        return {
            "pdb_id": first["pdb_id"], "annotation": None, "metadata": {},
            "rows": [_failure(row, "annotation_failed", f"{type(exc).__name__}: {exc}") for row in rows],
        }


def extract_catalog(
    manifest: Path, out_dir: Path, *, dssp_binary: Path, workers: int = 1,
    model_id: int = 0, timeout: float = 180, resume: bool = False,
    retry_failed: bool = False, limit_pdb: int | None = None,
) -> dict[str, Any]:
    """Extract offline using an atomic SQLite checkpoint and compressed SSE cache.

    A bounded structure limit is useful for throughput and format validation.
    Unprocessed domains remain visible as pending in the exported feature table.
    """
    if workers < 1 or timeout <= 0 or model_id < 0 or (limit_pdb is not None and limit_pdb < 1):
        raise ValueError("workers/timeout/limit must be positive and model_id nonnegative")
    if retry_failed and not resume:
        raise ValueError("--retry-failed requires --resume")
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError("Manifest must contain at least one domain")
    required = {"record_id", "pdb_id", "structure_path", "segments_json", "boundary_status", "coordinate_status"}
    if required - set(frame):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(frame))}")
    if frame["record_id"].eq("").any() or frame["record_id"].duplicated().any():
        raise ValueError("Manifest record_id must be unique and nonempty")
    for _, group in frame.groupby("pdb_id", sort=False):
        if group["structure_path"].nunique() > 1 or ("coordinate_sha256" in group and group["coordinate_sha256"].nunique() > 1):
            raise ValueError("Each PDB must have one coordinate source and content identity")
    ready = frame["coordinate_status"].eq("available")
    if ready.any() and ("coordinate_sha256" not in frame or any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in frame.loc[ready, "coordinate_sha256"]
    )):
        raise ValueError("Available coordinates require a complete lowercase coordinate_sha256 identity")
    dssp_binary = dssp_binary.expanduser().resolve()
    contract = {
        "schema": "betlas.domain-extraction.v1", "manifest_sha256": sha256_file(manifest),
        "feature_schema": FEATURE_SCHEMA, "feature_columns": list(FEATURE_COLUMNS),
        "dssp_binary": str(dssp_binary), "dssp_version": _binary_version(dssp_binary),
        "model_id": model_id, "source": _source_contract(),
        "python": sys.version, "biopython": package_version("biopython"),
        "numpy": package_version("numpy"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.sqlite"
    if checkpoint.exists() and not resume:
        raise ValueError("Checkpoint exists; choose --resume or a new output directory")
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)
    with sqlite3.connect(checkpoint) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS results (record_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS annotations (pdb_id TEXT PRIMARY KEY, content BLOB NOT NULL, metadata TEXT NOT NULL)")
        stored = conn.execute("SELECT payload FROM metadata WHERE key='contract'").fetchone()
        if stored and json.loads(stored[0]) != contract:
            raise ValueError("Resume contract changed: input, implementation, environment or DSSP differs")
        conn.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (json.dumps(contract, sort_keys=True),))
        conn.commit()
        existing = {key: json.loads(payload) for key, payload in conn.execute("SELECT record_id,payload FROM results")}
        if retry_failed:
            for key, row in list(existing.items()):
                if row["feature_status"] != "ok":
                    conn.execute("DELETE FROM results WHERE record_id=?", (key,))
                    del existing[key]
            conn.commit()
        candidates = []
        for row in frame.to_dict("records"):
            if row["record_id"] in existing:
                continue
            reason = ""
            if row["boundary_status"] not in {"ok", "requires_coordinate_order"}:
                reason = "Boundary unavailable: " + row.get("boundary_reason", row["boundary_status"])
            elif row["coordinate_status"] != "available" or not row["structure_path"] or not Path(row["structure_path"]).is_file():
                reason = "Coordinate unavailable"
            if reason:
                result = _failure(row, "data_unavailable", reason)
                conn.execute("INSERT INTO results VALUES (?,?)", (row["record_id"], json.dumps(result)))
            else:
                candidates.append(row)
        conn.commit()
        grouped = {}
        for row in candidates:
            grouped.setdefault(row["pdb_id"], []).append(row)
        jobs = list(grouped.items())
        if limit_pdb is not None:
            jobs = jobs[:limit_pdb]

        def prepare_job(item):
            code, rows = item
            cache = conn.execute("SELECT content,metadata FROM annotations WHERE pdb_id=?", (code,)).fetchone()
            return {
                "rows": rows, "cached_annotation": cache[0] if cache else None,
                "cached_meta": json.loads(cache[1]) if cache else {},
                "work_dir": str(work_dir.resolve()), "dssp_binary": str(dssp_binary),
                "model_id": model_id, "timeout": timeout,
            }

        def save(result):
            with conn:
                if result["annotation"] is not None:
                    conn.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?)", (result["pdb_id"], result["annotation"], json.dumps(result["metadata"])))
                for row in result["rows"]:
                    conn.execute("INSERT OR REPLACE INTO results VALUES (?,?)", (row["record_id"], json.dumps(row)))

        if workers == 1:
            for item in jobs:
                save(_extract_structure_job(prepare_job(item)))
        else:
            iterator = iter(jobs)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending = {}
                for item in list(next(iterator, None) for _ in range(min(len(jobs), workers * 2))):
                    if item is not None:
                        pending[pool.submit(_extract_structure_job, prepare_job(item))] = item[0]
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        pending.pop(future)
                        save(future.result())
                        item = next(iterator, None)
                        if item is not None:
                            pending[pool.submit(_extract_structure_job, prepare_job(item))] = item[0]
        results = {key: json.loads(payload) for key, payload in conn.execute("SELECT record_id,payload FROM results")}
    result_rows = [results.get(row["record_id"], _failure(row, "pending", "Not processed in this invocation")) for row in frame.to_dict("records")]
    values = pd.DataFrame(result_rows)
    # Preserve authoritative source metadata when feature rows carry metadata fields.
    extra = [c for c in values if c not in frame or c == "record_id"]
    output = frame.merge(values[extra], on="record_id", how="left", validate="one_to_one")
    temporary = out_dir / "features.csv.partial"
    output.to_csv(temporary, index=False)
    os.replace(temporary, out_dir / "features.csv")
    counts = output["feature_status"].value_counts().to_dict()
    summary = {
        "created_utc": utc_now_iso(), "domains": len(frame), "status_counts": counts,
        "complete": counts.get("pending", 0) == 0, "structures_processed_this_run": len(jobs),
        "workers": workers, "limit_pdb": limit_pdb, "contract": contract,
    }
    (out_dir / "extraction.json").write_text(json.dumps(summary, indent=2) + "\n")
    for label in ("cath_architecture_code", "cath_topology_code"):
        if label in output:
            output.groupby([label, "feature_status"], dropna=False).size().rename("domains").reset_index().to_csv(out_dir / f"coverage_{label}.csv", index=False)
    return summary


def export_domains(manifest: Path, checkpoint: Path, out_dir: Path, *, limit_domains: int | None = None) -> pd.DataFrame:
    """Export exact observed domain atoms from the frozen annotation checkpoint.

    Hex-encoded names preserve case-sensitive CATH IDs on case-insensitive disks.
    The mapping table retains original IDs, failed exports and pending records.
    """
    from betlas.io.domain_geometry import export_domain_mmcif

    if limit_domains is not None and limit_domains < 1:
        raise ValueError("limit_domains must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if frame.empty or frame.record_id.duplicated().any():
        raise ValueError("Manifest must contain unique domain records")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with sqlite3.connect(f"file:{checkpoint.resolve()}?mode=ro", uri=True) as conn:
        stored = conn.execute("SELECT payload FROM metadata WHERE key='contract'").fetchone()
        if stored is None or json.loads(stored[0])["manifest_sha256"] != sha256_file(manifest):
            raise ValueError("Export manifest differs from the frozen extraction checkpoint")
        contract = json.loads(stored[0])
        if contract["source"] != _source_contract():
            raise ValueError("Export implementation differs from the extraction checkpoint")
        count = 0
        for code, group in frame.groupby("pdb_id", sort=False):
            cached = conn.execute("SELECT content FROM annotations WHERE pdb_id=?", (code,)).fetchone()
            with tempfile.TemporaryDirectory(prefix="export-", dir=out_dir) as tmp:
                annotated = Path(tmp) / "annotated.cif"
                if cached:
                    annotated.write_bytes(gzip.decompress(cached[0]))
                for row in group.to_dict("records"):
                    result = {"record_id": row["record_id"], "pdb_id": code, "structure_path": "", "export_status": "pending", "export_reason": ""}
                    if limit_domains is not None and count >= limit_domains:
                        rows.append(result)
                        continue
                    count += 1
                    if not cached:
                        result.update(export_status="annotation_unavailable", export_reason="No successful cached annotation")
                    else:
                        try:
                            domain = DomainCandidate.from_mapping({**row, "domain_id": row.get("domain_id", row["record_id"]), "model_id": contract["model_id"]})
                            target = out_dir / "coordinates" / (row["record_id"].encode("utf-8").hex() + ".cif")
                            metadata = export_domain_mmcif(domain, annotated, target, segments=json.loads(row["segments_json"]))
                            result.update(export_status="ok", structure_path=str(target.resolve()), atom_count=metadata["atom_count"], model_id=contract["model_id"])
                        except Exception as exc:
                            result.update(export_status="mapping_failed", export_reason=f"{type(exc).__name__}: {exc}")
                    rows.append(result)
    exported = pd.DataFrame(rows)
    frame.merge(exported, on=["record_id", "pdb_id"], how="left", suffixes=("_source", ""), validate="one_to_one").to_csv(out_dir / "domain_structures.csv", index=False)
    return exported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog", help="Join fixed-release classifications, boundaries and a local coordinate index")
    for name in ("domain-list", "boundaries", "names", "coordinate-index", "out-dir"):
        catalog.add_argument("--" + name, type=Path, required=True)
    catalog.add_argument("--official-representatives", type=Path)
    catalog.add_argument("--previously-used", type=Path, help="Complete prior domain-ID list; omitted means unknown")
    catalog.add_argument("--version", default="4.4.0")
    catalog.add_argument("--class-id", type=int, default=2)
    extract = sub.add_parser("extract", help="Offline precise-domain extraction with resumable local DSSP caching")
    for name in ("manifest", "out-dir", "dssp-binary"):
        extract.add_argument("--" + name, type=Path, required=True)
    extract.add_argument("--workers", type=int, default=1)
    extract.add_argument("--model-id", type=int, default=0, help="Zero-based coordinate model index")
    extract.add_argument("--timeout", type=float, default=180)
    extract.add_argument("--resume", action="store_true")
    extract.add_argument("--retry-failed", action="store_true")
    extract.add_argument("--limit-pdb", type=int, help="Bound this invocation; remaining domains stay pending")
    export = sub.add_parser("export", help="Export exact domain atoms from a frozen extraction checkpoint")
    for name in ("manifest", "checkpoint", "out-dir"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--limit-domains", type=int)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = {"catalog": make_catalog, "extract": extract_catalog, "export": export_domains}[command](**args)
    print(f"Catalog domains: {len(result)}" if isinstance(result, pd.DataFrame) else json.dumps({k: v for k, v in result.items() if k != "contract"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
