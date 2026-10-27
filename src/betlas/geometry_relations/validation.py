"""Bounded pairwise structural validation and anonymous review material.

This module consumes an already selected pair panel. It neither selects pairs
using labels nor treats a missing alignment as a structural distance. TM scores
are normalized by the two complete input lengths; coverage is the aligner's
reported aligned length divided by each input length. Sequence-order-preserving
alignment cannot rule out shared packing with different chain connectivity.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBException

from betlas.constants import STANDARD_AMINO_ACIDS

ALIGNMENT_SCHEMA_VERSION = "pairwise-global-tm-v1"
REVIEW_SCHEMA_VERSION = "anonymous-structure-review-v1"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_METRICS = (
    "query_length", "target_length", "aligned_length", "rmsd_angstrom",
    "aligned_sequence_identity", "query_tm_score", "target_tm_score",
    "min_tm_score", "query_coverage", "target_coverage", "min_coverage",
    "query_target_length_ratio", "short_long_length_ratio",
)


def _read_text(path: Path) -> str:
    """Use bounded reads, including for older WSL mounted-drive clients."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as handle:
        chunks = []
        while chunk := handle.read(8192):
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


def _records(table: Any) -> list[dict[str, Any]]:
    if isinstance(table, (str, Path)):
        return list(csv.DictReader(io.StringIO(_read_text(Path(table)))))
    if hasattr(table, "to_dict") and not isinstance(table, Mapping):
        return table.to_dict(orient="records")
    return [dict(row) for row in table]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest(table: Any) -> dict[str, dict[str, Any]]:
    output = {}
    for row in _records(table):
        record_id = str(row["record_id"])
        if record_id in output:
            raise ValueError(f"Duplicate coordinate record_id: {record_id}")
        output[record_id] = row
    return output


def _pairs(table: Any) -> list[dict[str, Any]]:
    rows = _records(table)
    seen = set()
    for index, row in enumerate(rows):
        for field in ("query_id", "target_id"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"Pair row {index} lacks {field}")
            row[field] = str(row[field])
        row["pair_id"] = str(row.get("pair_id", f"pair-{index:05d}"))
        if row["pair_id"] in seen:
            raise ValueError(f"Duplicate pair_id: {row['pair_id']}")
        seen.add(row["pair_id"])
    return rows


def parse_alignment_output(stdout: str) -> dict[str, int | float]:
    """Parse complete US-align/TM-align text; reject partial or invalid output."""
    lengths = {}
    scores = {}
    for number, role in ((1, "query"), (2, "target")):
        length = re.search(rf"Length of (?:Structure|Chain)_?{number}:\s*(\d+)", stdout)
        score = re.search(
            rf"TM-score\s*=\s*({_NUMBER})[^\n]*normalized by length of "
            rf"(?:Structure|Chain)_?{number}\b", stdout, flags=re.IGNORECASE,
        )
        if length is None or score is None:
            raise ValueError(f"Missing {role} length or normalized TM-score")
        lengths[role] = int(length.group(1))
        scores[role] = float(score.group(1))
    aligned = re.search(
        rf"Aligned length\s*=\s*(\d+)\s*,\s*RMSD\s*=\s*({_NUMBER})"
        rf"\s*,\s*Seq_ID[^=\n]*=[^=\n]*=\s*({_NUMBER})", stdout,
    )
    if aligned is None:
        # TM-align versions commonly print a single '=' after Seq_ID.
        aligned = re.search(
            rf"Aligned length\s*=\s*(\d+)\s*,\s*RMSD\s*=\s*({_NUMBER})"
            rf"\s*,\s*Seq_ID[^\n]*?=\s*({_NUMBER})\s*(?:\n|$)", stdout,
        )
    if aligned is None:
        raise ValueError("Missing aligned length, RMSD, or aligned sequence identity")
    count, rmsd, identity = int(aligned.group(1)), float(aligned.group(2)), float(aligned.group(3))
    query_length, target_length = lengths["query"], lengths["target"]
    if min(query_length, target_length) <= 0 or not 0 < count <= min(query_length, target_length):
        raise ValueError("Invalid input or aligned residue counts")
    if any(not math.isfinite(score) or not 0 <= score <= 1.00001 for score in scores.values()):
        raise ValueError("Normalized TM-score is not finite or outside [0, 1]")
    if not math.isfinite(rmsd) or rmsd < 0 or not 0 <= identity <= 1.00001:
        raise ValueError("Invalid RMSD or sequence identity")
    return {
        "query_length": query_length, "target_length": target_length,
        "aligned_length": count, "rmsd_angstrom": rmsd,
        "aligned_sequence_identity": identity, "query_tm_score": scores["query"],
        "target_tm_score": scores["target"], "min_tm_score": min(scores.values()),
        "query_coverage": count / query_length, "target_coverage": count / target_length,
        "min_coverage": count / max(query_length, target_length),
        "query_target_length_ratio": query_length / target_length,
        "short_long_length_ratio": min(query_length, target_length) / max(query_length, target_length),
    }


def _load_structure(path: Path):
    text = _read_text(path)
    suffixes = {suffix.lower() for suffix in path.suffixes}
    parser = MMCIFParser(QUIET=True) if suffixes & {".cif", ".mmcif"} else PDBParser(QUIET=True)
    return parser.get_structure("input", io.StringIO(text))


def _coordinate_data(path: Path) -> dict[str, Any]:
    if {suffix.lower() for suffix in path.suffixes} & {".cif", ".mmcif"}:
        return dict(MMCIF2Dict(io.StringIO(_read_text(path))))
    writer, handle = MMCIFIO(), io.StringIO()
    writer.set_structure(_load_structure(path))
    writer.save(handle)
    return dict(MMCIF2Dict(io.StringIO(handle.getvalue())))


def _atom_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    columns = {key.removeprefix("_atom_site."): value for key, value in data.items()
               if key.startswith("_atom_site.")}
    n = len(columns.get("id", []))
    if not n or any(not isinstance(values, list) or len(values) != n for values in columns.values()):
        raise ValueError("Missing or inconsistent atom_site records")
    return [{key: values[index] for key, values in columns.items()} for index in range(n)]


def _blank(value: str) -> str:
    return "" if value in {"", ".", "?", " "} else value


def _residue_key(row):
    return (row.get("pdbx_PDB_model_num", "1"), _blank(row["auth_asym_id"]),
            int(row["auth_seq_id"]), _blank(row.get("pdbx_PDB_ins_code", "?")))


def _atom_rank(row):
    alt = _blank(row.get("label_alt_id", "."))
    occupancy = _blank(row.get("occupancy", "1"))
    return float(occupancy or "1"), 2 if not alt else 1 if alt == "A" else 0


def _selected_ca(rows):
    chosen, counts = {}, Counter()
    for index, row in enumerate(rows):
        if (row["label_atom_id"] != "CA" or row["label_comp_id"] not in STANDARD_AMINO_ACIDS
                or not _blank(row.get("label_seq_id", "")) or row["group_PDB"] not in {"ATOM", "HETATM"}):
            continue
        if not all(math.isfinite(float(row[key])) for key in ("Cartn_x", "Cartn_y", "Cartn_z")):
            continue
        key = _residue_key(row)
        counts[key] += 1
        if key not in chosen or _atom_rank(row) > _atom_rank(rows[chosen[key]]):
            chosen[key] = index
    ordered = sorted(chosen.values(), key=lambda index: (
        rows[index].get("pdbx_PDB_model_num", "1"), rows[index]["label_asym_id"],
        int(rows[index]["label_seq_id"])))
    return ordered, counts


def _inspect_data(data: dict[str, Any]) -> dict[str, Any]:
    rows = _atom_rows(data)
    selected, counts = _selected_ca(rows)
    identities = []
    for index in selected:
        row = rows[index]
        model, chain, number, insertion = _residue_key(row)
        identities.append({"model_id": int(model) - 1, "model_serial": int(model),
                           "chain_id": chain, "hetero_flag": row["group_PDB"],
                           "author_residue_number": number, "insertion_code": insertion,
                           "label_seq_id": int(row["label_seq_id"]),
                           "residue_name": row["label_comp_id"], "altloc": _blank(row.get("label_alt_id", ".")),
                           "coordinate": [float(row[key]) for key in ("Cartn_x", "Cartn_y", "Cartn_z")]})
    model_count = len({row.get("pdbx_PDB_model_num", "1") for row in rows})
    alternate_ca = sum(count > 1 for count in counts.values())
    status, reason = "ok", ""
    chain_count = len({(row["model_id"], row["chain_id"]) for row in identities})
    if model_count != 1:
        status, reason = "unsupported_multiple_models", "Exact domain input must contain one model"
    elif chain_count != 1:
        status, reason = "unsupported_multiple_chains", "Monomer validation does not concatenate chains"
    elif alternate_ca:
        status, reason = "unresolved_ca_altloc", "Exact domain export must select one CA conformer"
    elif len(identities) < 3:
        status, reason = "too_few_ca", "At least three CA residues are needed"
    return {"status": status, "reason": reason, "ca_count": len(identities),
            "chain_count": chain_count, "model_count": model_count,
            "atom_count": len(rows), "ca_identities": identities, "n_multiple_altloc_ca": alternate_ca}


def inspect_coordinates(path: str | Path) -> dict[str, Any]:
    """Inspect CA identities using the geometry pipeline's occupancy/tie policy.

    Source mmCIF coordinate tokens retain their precision. Multiple source CA
    conformers remain explicitly flagged until canonical execution preparation.
    """
    return _inspect_data(_coordinate_data(Path(path)))


def _canonical_data(path: Path, *, anonymous_id: str | None = None):
    data = _coordinate_data(path)
    before = _inspect_data(data)
    if before["status"] not in {"ok", "unresolved_ca_altloc"}:
        return data, before
    rows = _atom_rows(data)
    ca_indices, _ = _selected_ca(rows)
    allowed = {_residue_key(rows[index]): rows[index]["label_comp_id"] for index in ca_indices}
    chosen = {}
    for index, row in enumerate(rows):
        residue = _residue_key(row)
        if allowed.get(residue) != row["label_comp_id"]:
            continue
        key = (*residue, row["label_atom_id"])
        if key not in chosen or _atom_rank(row) > _atom_rank(rows[chosen[key]]):
            chosen[key] = index
    selected = sorted(chosen.values(), key=lambda index: (
        rows[index].get("pdbx_PDB_model_num", "1"), rows[index]["label_asym_id"],
        int(rows[index]["label_seq_id"]), index))
    output = {"data_": anonymous_id or "domain"}
    for key, values in data.items():
        if key.startswith("_atom_site."):
            output[key] = [values[index] for index in selected]
    # Normalize a selected atom's altloc marker only; never alter its coordinates.
    output["_atom_site.label_alt_id"] = ["."] * len(selected)
    if anonymous_id:
        output["_atom_site.auth_asym_id"] = ["A"] * len(selected)
        output["_atom_site.label_asym_id"] = ["A"] * len(selected)
        output["_atom_site.pdbx_PDB_model_num"] = ["1"] * len(selected)
    after = _inspect_data(output)
    comparable = ("author_residue_number", "insertion_code", "label_seq_id", "residue_name", "coordinate")
    if len(before["ca_identities"]) != len(after["ca_identities"]) or any(
            left[key] != right[key] for left, right in zip(before["ca_identities"], after["ca_identities"], strict=True)
            for key in comparable):
        raise ValueError("Canonical export changed the geometry-selected CA identity or coordinate")
    after.update(source_ca_identities=before["ca_identities"], source_atom_count=before["atom_count"],
                 source_multiple_altloc_ca_count=before["n_multiple_altloc_ca"],
                 altloc_policy="highest occupancy; ties prefer blank, then A, then first other; matches domain_geometry._ca_residues",
                 coordinate_precision_policy="Preserve original mmCIF atom_site coordinate strings",
                 removed_atom_alternatives=before["atom_count"] - after["atom_count"],
                 canonical_ca_identity_check="exact_match")
    return output, after


def _prepare_coordinate(record_id: str, row: Mapping[str, Any] | None,
                        staged: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"record_id": record_id, "status": "coordinate_unavailable", "reason": ""}
    if row is None:
        result["reason"] = "Record absent from coordinate manifest"
        return result
    source = str(row.get("structure_path", ""))
    result["source_path"] = source
    if row.get("export_status", "ok") != "ok":
        result["reason"] = f"Upstream export status: {row.get('export_status')}"
        return result
    if not source or not Path(source).is_file():
        result["reason"] = "Coordinate file missing"
        return result
    try:
        actual = Path(source)
        data, inspected = _canonical_data(actual)
        if staged is not None:
            actual = staged / f"structure-{record_id.encode().hex()}.cif"
            writer = MMCIFIO()
            writer.set_dict(data)
            writer.save(str(actual))
        elif inspected["status"] == "ok" and inspected.get("source_multiple_altloc_ca_count", 0):
            # Callers disabling staging must not feed unresolved originals to the aligner.
            inspected.update(status="canonical_staging_required",
                             reason="Source has multiple CA conformers; enable canonical staging")
        result.update(inspected)
        result["execution_path"] = str(actual)
    except (OSError, ValueError, KeyError, IndexError, PDBException) as error:
        result.update(status="coordinate_error", reason=f"{type(error).__name__}: {error}")
    return result


def run_pair_alignments(pairs: Any, coordinate_manifest: Any, output_dir: str | Path, *,
                        binary: str = "USalign", workers: int = 1, timeout: float = 120,
                        stage_coordinates: bool = True) -> list[dict[str, Any]]:
    """Run a fixed pair panel, preserving raw outputs and every failure.

    ``pairs`` has ``pair_id, query_id, target_id``; ``coordinate_manifest`` has
    ``record_id, structure_path, export_status``. Both accept CSV paths, records,
    or DataFrames. The new output directory must not already exist. Optional
    staging reads selected coordinates in 8 KiB chunks and writes canonical
    single-conformer mmCIF into /tmp without rounding source coordinate tokens.
    """
    if workers < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    rows, manifest = _pairs(pairs), _manifest(coordinate_manifest)
    executable = shutil.which(str(binary))
    if executable is None:
        raise FileNotFoundError(f"Structural aligner unavailable: {binary}")
    is_usalign = "usalign" in Path(executable).name.lower()
    if not is_usalign and "tmalign" not in Path(executable).name.lower():
        raise ValueError("Expected a USalign or TMalign executable")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    version = subprocess.run([executable, "-v"], capture_output=True, text=True, timeout=20, check=False)
    (output / "version.stdout.txt").write_text(version.stdout, encoding="utf-8")
    (output / "version.stderr.txt").write_text(version.stderr, encoding="utf-8")
    ids = sorted({row[key] for row in rows for key in ("query_id", "target_id")})
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="betlas-pair-coordinates-") as scratch:
        prepared = {record_id: _prepare_coordinate(record_id, manifest.get(record_id),
                    Path(scratch) if stage_coordinates else None) for record_id in ids}
        _write_json(output / "coordinate-inspection.json", list(prepared.values()))

        def execute(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            index, pair = item
            query, target = prepared[pair["query_id"]], prepared[pair["target_id"]]
            record = {key: pair[key] for key in ("pair_id", "query_id", "target_id")}
            record.update({key: None for key in _METRICS})
            record.update(status="not_run", reason="", returncode=None, elapsed_seconds=0.0,
                          query_input_ca_count=query.get("ca_count"),
                          target_input_ca_count=target.get("ca_count"),
                          stdout_path=f"raw/{index:05d}.stdout.txt",
                          stderr_path=f"raw/{index:05d}.stderr.txt",
                          command_path=f"raw/{index:05d}.command.json")
            command = []
            stdout = stderr = ""
            failed = [(name, value) for name, value in (("query", query), ("target", target))
                      if value["status"] != "ok"]
            if failed:
                record.update(status="input_unavailable", reason="; ".join(
                    f"{name}: {value['status']}: {value['reason']}" for name, value in failed))
            else:
                command = [executable, query["execution_path"], target["execution_path"],
                           "-ter", "1", "-split", "0", "-het", "1", "-outfmt", "0"]
                if is_usalign:
                    command.extend(["-mol", "prot", "-mm", "0"])
                tick = time.monotonic()
                try:
                    process = subprocess.run(command, capture_output=True, text=True,
                                             timeout=timeout, check=False)
                    stdout, stderr = process.stdout, process.stderr
                    record["returncode"] = process.returncode
                    if process.returncode != 0:
                        record.update(status="execution_failed", reason=f"Exit code {process.returncode}")
                    else:
                        record.update(parse_alignment_output(stdout))
                        record.update(status="ok")
                        if (record["query_length"] != query["ca_count"] or
                                record["target_length"] != target["ca_count"]):
                            record.update(status="input_length_mismatch",
                                          reason="Aligner did not consume all inspected domain CA residues")
                except subprocess.TimeoutExpired as error:
                    stdout, stderr = error.stdout or "", error.stderr or ""
                    record.update(status="timeout", reason=f"Pair exceeded {timeout:g} seconds")
                except OSError as error:
                    record.update(status="execution_failed", reason=f"{type(error).__name__}: {error}")
                except ValueError as error:
                    record.update(status="parse_failed", reason=str(error))
                record["elapsed_seconds"] = time.monotonic() - tick
            for value, key in ((stdout, "stdout_path"), (stderr, "stderr_path")):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                (output / record[key]).write_text(value, encoding="utf-8")
            _write_json(output / record["command_path"], {"argv": command,
                        "query_source": query.get("source_path"), "target_source": target.get("source_path"),
                        "temporary_staging": stage_coordinates})
            _write_json(output / "raw" / f"{index:05d}.result.json", record)
            return record

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(execute, enumerate(rows)))
    fields = list(results[0]) if results else ["pair_id", "query_id", "target_id", "status", *_METRICS]
    _write_csv(output / "pair-alignments.csv", results, fields)
    _write_json(output / "summary.json", {
        "schema_version": ALIGNMENT_SCHEMA_VERSION, "pair_count": len(rows),
        "coordinate_count": len(ids), "statuses": dict(Counter(row["status"] for row in results)),
        "binary": executable, "workers": workers, "timeout_seconds": timeout,
        "elapsed_seconds": time.monotonic() - started, "mode": "monomer_sequence_order_preserving_global_TM",
        "coverage_definition": "reported aligned length / each complete input CA length",
        "missing_alignment_policy": "unavailable, never zero TM or infinite structural distance",
        "human_review_status": "not_performed",
    })
    return results


def build_review_package(pairs: Any, coordinate_manifest: Any, output_dir: str | Path, *,
                         seed: int = 13) -> dict[str, Any]:
    """Export only panel domains with anonymous headers and an isolated identity key.

    The reviewer folder has no CATH IDs, labels, model outputs, selection scores,
    or source paths. The private key is for the coordinator and must be withheld
    from reviewers. Structure identity remains inferable by external structure
    search; this is blinded presentation, not irreversible de-identification.
    """
    rows, manifest = _pairs(pairs), _manifest(coordinate_manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reviewer, private = output / "reviewer", output / "private"
    (reviewer / "coordinates").mkdir(parents=True)
    private.mkdir()
    ids = sorted({row[key] for row in rows for key in ("query_id", "target_id")})
    random.Random(seed).shuffle(ids)
    aliases = {record_id: f"S{index + 1:03d}" for index, record_id in enumerate(ids)}
    structures, identities, inspections = [], [], []
    for record_id in ids:
        alias = aliases[record_id]
        source = manifest.get(record_id)
        inspected = _prepare_coordinate(record_id, source, None)
        anonymous_data = None
        if inspected["status"] in {"ok", "canonical_staging_required"}:
            anonymous_data, checked = _canonical_data(Path(inspected["source_path"]), anonymous_id=alias)
            inspected.update(checked)
        inspections.append(inspected)
        row = {"structure_id": alias, "coordinate_path": "", "coordinate_status": inspected["status"],
               "ca_count": inspected.get("ca_count"), "atom_count": inspected.get("atom_count"),
               "review_status": "pending_human_review"}
        if inspected["status"] == "ok":
            old_chain = inspected["source_ca_identities"][0]["chain_id"]
            writer = MMCIFIO()
            writer.set_dict(anonymous_data)
            relative = f"coordinates/{alias}.cif"
            writer.save(str(reviewer / relative))
            checked = inspect_coordinates(reviewer / relative)
            before = inspected["source_ca_identities"]
            after = checked["ca_identities"]
            comparable = ("author_residue_number", "insertion_code", "residue_name", "coordinate")
            if len(before) != len(after) or any(
                    a[key] != b[key] for a, b in zip(before, after, strict=True) for key in comparable):
                raise ValueError(f"Anonymous export changed CA identities or coordinates: {record_id}")
            row["coordinate_path"] = relative
        else:
            old_chain = ""
        structures.append(row)
        identities.append({"structure_id": alias, "record_id": record_id,
                           "source_path": inspected.get("source_path", ""), "source_chain_id": old_chain})
    pair_order = list(rows)
    random.Random(seed + 1).shuffle(pair_order)
    review_pairs, pair_keys = [], []
    for index, pair in enumerate(pair_order):
        pair_id = f"P{index + 1:03d}"
        review_pairs.append({"review_pair_id": pair_id,
                             "structure_1": aliases[pair["query_id"]],
                             "structure_2": aliases[pair["target_id"]],
                             "reviewer_id": "", "review_status": "pending_human_review",
                             "sheet_arrangement": "", "pairing_evidence": "",
                             "closure_extent": "", "similarity_scope": "",
                             "missing_or_clipped_regions": "", "confidence": "", "notes": ""})
        pair_keys.append({"review_pair_id": pair_id, "pair_id": pair["pair_id"],
                          "query_id": pair["query_id"], "target_id": pair["target_id"]})
    _write_csv(reviewer / "structures.csv", structures,
               ["structure_id", "coordinate_path", "coordinate_status", "ca_count", "atom_count", "review_status"])
    _write_csv(reviewer / "pair-review.csv", review_pairs, list(review_pairs[0]) if review_pairs else
               ["review_pair_id", "structure_1", "structure_2", "review_status"])
    _write_csv(private / "structure-key.csv", identities,
               ["structure_id", "record_id", "source_path", "source_chain_id"])
    _write_csv(private / "pair-key.csv", pair_keys, ["review_pair_id", "pair_id", "query_id", "target_id"])
    _write_json(private / "coordinate-inspection.json", inspections)
    schema = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_status": "pending_human_review", "completed_independent_human_reviews": 0,
        "coordinate_scope": "selected panel only; source headers removed; one anonymous chain A",
        "instructions": [
            "Give each independent reviewer only a separate copy of the reviewer directory.",
            "Do not expose the private key, CATH labels, model output, or selection scores.",
            "Coordinate counts and export checks are automatic facts, not human validation.",
            "Record observed arrangement, pairing and closure extent with residue references.",
            "Separate local motifs from whole-domain similarity; report missing evidence.",
            "Do not force a fold label or regard a cycle alone as a geometric barrel.",
        ],
        "field_guidance": {
            "sheet_arrangement": "Observed relative sheet positions; concrete residue or strand references",
            "pairing_evidence": "Observed pairing source and explicit edges; unknown if not established",
            "closure_extent": "Local/global/absent/undetermined, with covered residues or strands",
            "similarity_scope": "whole_domain/local_motif/neither/undetermined, with explanation",
            "missing_or_clipped_regions": "Coordinate gaps, truncation, and non-evaluable relationships",
            "confidence": "high/moderate/low/undetermined; justified in notes",
        },
        "limitation": "Blinded presentation; structures can still be recognized or searched externally.",
    }
    _write_json(reviewer / "review-schema.json", schema)
    summary = {"schema_version": REVIEW_SCHEMA_VERSION, "structures": len(structures), "pairs": len(rows),
               "coordinate_statuses": dict(Counter(row["coordinate_status"] for row in structures)),
               "completed_independent_human_reviews": 0, "review_status": "pending_human_review"}
    _write_json(output / "summary.json", summary)
    return summary


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _descriptive(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values), "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
    }


def summarize_validation(alignments: Any, pair_metadata: Any, output_dir: str | Path | None = None, *,
                         group_columns: tuple[str, ...] = ("relation_kind", "role", "phase"),
                         role_column: str = "role", match_column: str = "matched_pair_id",
                         candidate_role: str = "candidate", control_role: str = "control",
                         homology_columns: tuple[str, str] = ("queryH", "targetH"),
                         architecture_columns: tuple[str, str] = ("queryA", "targetA")) -> dict[str, Any]:
    """Describe a fixed pair panel without treating pairs as independent samples.

    Every metadata row stays in the denominator. Only complete ``status=ok``
    measurements enter metric summaries. The role and grouping columns are
    caller-defined protocol fields, never derived from alignment outcomes.
    Matched contrasts are candidate minus control means within each declared
    ``matched_pair_id``. For a 1:1 match this is the ordinary paired difference;
    larger groups retain their counts and are explicitly marked as group means.
    No confidence intervals, significance tests, or effective sample sizes are
    inferred from node-sharing pairs.
    """
    metadata = _pairs(pair_metadata)
    alignment_rows = _records(alignments)
    lookup = {}
    for row in alignment_rows:
        pair_id = str(row["pair_id"])
        if pair_id in lookup:
            raise ValueError(f"Duplicate alignment pair_id: {pair_id}")
        lookup[pair_id] = row
    known = {row["pair_id"] for row in metadata}
    if unknown := set(lookup).difference(known):
        raise ValueError(f"Alignments absent from pair metadata: {sorted(unknown)[:5]}")
    metrics = ("min_tm_score", "min_coverage", "short_long_length_ratio")

    def value_string(value: Any) -> str:
        if value is None or isinstance(value, float) and not math.isfinite(value):
            return ""
        return str(value)

    ledger = []
    for metadata_row in metadata:
        pair_id = metadata_row["pair_id"]
        alignment = lookup.get(pair_id)
        row = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in metadata_row.items()}
        row.update({metric: None for metric in metrics})
        row.update(alignment_status="alignment_missing", alignment_reason="No result record",
                   measurement_available=False)
        if alignment is not None:
            for key in ("query_id", "target_id"):
                if str(alignment.get(key, "")) != str(metadata_row[key]):
                    raise ValueError(f"Alignment identity differs from pair metadata: {pair_id}/{key}")
            row["alignment_status"] = str(alignment.get("status", "status_missing"))
            row["alignment_reason"] = str(alignment.get("reason", ""))
            if row["alignment_status"] == "ok":
                values = {metric: _finite_number(alignment.get(metric)) for metric in metrics}
                if any(value is None or not 0 <= value <= 1.00001 for value in values.values()):
                    row.update(alignment_status="invalid_ok_record",
                               alignment_reason="Successful record lacks valid finite TM, coverage or length ratio")
                else:
                    row.update(values)
                    row["measurement_available"] = True
        ledger.append(row)

    grouped = {}
    for row in ledger:
        key = tuple(value_string(row.get(column)) for column in group_columns)
        grouped.setdefault(key, []).append(row)
    group_summaries = []
    for key, rows in sorted(grouped.items()):
        usable = [row for row in rows if row["measurement_available"]]
        node_counts = Counter(node for row in rows for node in {row["query_id"], row["target_id"]})
        support = {column: {value_string(row.get(column)) for row in rows
                            if value_string(row.get(column))}
                   for column in (*homology_columns, *architecture_columns)}
        item = dict(zip(group_columns, key, strict=True))
        item.update(n_pairs=len(rows), n_available=len(usable), n_unavailable=len(rows) - len(usable),
                    n_unique_domains=len(node_counts),
                    n_available_domains=len({row[column] for row in usable for column in ("query_id", "target_id")}),
                    n_unique_homologies=len(set().union(*(support[column] for column in homology_columns))),
                    n_unique_architectures=len(set().union(*(support[column] for column in architecture_columns))),
                    n_reused_domains=sum(count > 1 for count in node_counts.values()),
                    max_pairs_per_domain=max(node_counts.values(), default=0),
                    n_pairs_with_reused_domain=sum(any(node_counts[row[column]] > 1
                                                     for column in ("query_id", "target_id")) for row in rows),
                    statuses_json=json.dumps(dict(Counter(row["alignment_status"] for row in rows)), sort_keys=True))
        for metric in metrics:
            item.update({f"{metric}_{stat}": value
                         for stat, value in _descriptive([row[metric] for row in usable]).items()})
        group_summaries.append(item)

    matches = {}
    for row in ledger:
        match_id = value_string(row.get(match_column))
        if match_id:
            matches.setdefault(match_id, []).append(row)
    contrasts = []
    for match_id, rows in sorted(matches.items()):
        candidates = [row for row in rows if value_string(row.get(role_column)) == candidate_role]
        controls = [row for row in rows if value_string(row.get(role_column)) == control_role]
        ready_candidates = [row for row in candidates if row["measurement_available"]]
        ready_controls = [row for row in controls if row["measurement_available"]]
        item = {match_column: match_id,
                "candidate_pair_ids_json": json.dumps([row["pair_id"] for row in candidates]),
                "control_pair_ids_json": json.dumps([row["pair_id"] for row in controls]),
                "n_candidates": len(candidates), "n_controls": len(controls),
                "n_available_candidates": len(ready_candidates), "n_available_controls": len(ready_controls),
                "n_other_role_pairs": len(rows) - len(candidates) - len(controls),
                "n_unique_domains": len({row[column] for row in rows for column in ("query_id", "target_id")}),
                "contrast_kind": "one_to_one" if len(candidates) == len(controls) == 1 else "difference_of_group_means",
                "status": "ok", "reason": ""}
        for column in group_columns:
            if column != role_column and column != match_column:
                item[f"{column}_values_json"] = json.dumps(sorted({value_string(row.get(column)) for row in rows}))
        if not candidates or not controls:
            item.update(status="missing_role", reason="Declared matched group lacks candidate or control")
        elif len(ready_candidates) != len(candidates) or len(ready_controls) != len(controls):
            # Dropping failed members changes a declared matched comparison.
            item.update(status="incomplete_measurements", reason="At least one declared matched member is unavailable")
        for metric in metrics:
            item[f"{metric}_candidate_mean"] = None
            item[f"{metric}_control_mean"] = None
            item[f"{metric}_difference"] = None
            if item["status"] == "ok":
                candidate_mean = statistics.mean(row[metric] for row in ready_candidates)
                control_mean = statistics.mean(row[metric] for row in ready_controls)
                item[f"{metric}_candidate_mean"] = candidate_mean
                item[f"{metric}_control_mean"] = control_mean
                item[f"{metric}_difference"] = candidate_mean - control_mean
        contrasts.append(item)
    node_counts = Counter(node for row in ledger for node in {row["query_id"], row["target_id"]})
    summary = {
        "schema_version": "structural-validation-summary-v1", "n_pairs": len(ledger),
        "n_available": sum(row["measurement_available"] for row in ledger),
        "statuses": dict(Counter(row["alignment_status"] for row in ledger)),
        "n_unique_domains": len(node_counts), "n_reused_domains": sum(count > 1 for count in node_counts.values()),
        "max_pairs_per_domain": max(node_counts.values(), default=0),
        "group_columns": list(group_columns), "role_column": role_column, "match_column": match_column,
        "matched_contrast": "candidate mean minus control mean within declared matched group; complete groups only",
        "n_matched_groups": len(contrasts), "matched_statuses": dict(Counter(row["status"] for row in contrasts)),
        "inference_scope": "Descriptive fixed-panel evidence; shared nodes and families make pairs dependent",
        "uncertainty_policy": "No pair-independent CI, p-value, effective sample size, or population prevalence estimate",
        "groups": group_summaries, "matched_contrasts": contrasts, "pair_ledger": ledger,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        for filename, rows in (("group-summary.csv", group_summaries),
                               ("matched-contrasts.csv", contrasts), ("pair-ledger.csv", ledger)):
            fields = list(dict.fromkeys(key for row in rows for key in row))
            _write_csv(output / filename, rows, fields or ["status"])
        _write_json(output / "summary.json", {key: value for key, value in summary.items()
                                             if key not in {"groups", "matched_contrasts", "pair_ledger"}})
    return summary
