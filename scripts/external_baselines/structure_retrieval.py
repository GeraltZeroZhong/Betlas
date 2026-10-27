"""Local labelled-reference Foldseek readout for binary structure tasks.

References contain both classes. Query labels are never used during retrieval.
The score is the difference between the best positive and negative symmetric
TM scores, not a native Foldseek classifier probability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from Bio.PDB import MMCIFIO
from Bio.PDB.PDBExceptions import PDBConstructionException, PDBException

from scripts.external_baselines.beta_barrel_detection.foldseek.runner import (
    DEFAULT_FORMAT_FIELDS,
    FoldseekHit,
    load_hits_tsv,
)
from scripts.external_baselines.beta_barrel_detection.foldseek.structures import (
    _chain_residues,
    _parse_structure,
    _ProteinResidueSelect,
)

TASKS = ("beta_solenoid", "barrel_geometry", "membrane_beta_barrel")
REQUIRED = {"sample_id", "structure_path", "chain_id", "pdb_id", "group_id"}


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("structure_path"):
            row["structure_path"] = str((path.parent / row["structure_path"]).resolve())
    return rows


def validate_collections(queries: list[dict], references: list[dict]) -> None:
    """Reject declared identity, PDB, group and exact source overlap before search."""
    for name, rows in (("queries", queries), ("references", references)):
        if not rows:
            raise ValueError(f"Empty {name}")
        for row in rows:
            if REQUIRED - row.keys() or any(not str(row[k]).strip() for k in REQUIRED - {"chain_id"}):
                raise ValueError(f"{name} require sample, source, chain, PDB and biological group IDs")
            if int(row.get("model_id", 0)) < 0:
                raise ValueError("model_id must be nonnegative")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate sample IDs in {name}")
    if {int(row["target"]) for row in references} != {0, 1}:
        raise ValueError("References must contain both binary classes")
    for field in ("sample_id", "pdb_id", "group_id"):
        def key(row, field=field):
            return row[field].lower() if field == "pdb_id" else row[field]
        if {key(row) for row in queries} & {key(row) for row in references}:
            raise ValueError(f"Query/reference {field} overlap")
    def source_key(row):
        return (str(Path(row["structure_path"]).resolve()), row["chain_id"], int(row.get("model_id", 0)))
    if {source_key(row) for row in queries} & {source_key(row) for row in references}:
        raise ValueError("Query/reference coordinate source overlap")


def _aliases(rows: list[dict]) -> dict[str, str]:
    aliases = {}
    for row in rows:
        token = row["sample_id"].encode().hex()
        for name in (token, token + ".cif", token + "_A", token + ".cif_A"):
            aliases[name] = row["sample_id"]
    return aliases


def summarize_hits(hits: list[FoldseekHit], queries: list[dict], references: list[dict], *,
                   min_coverage: float = .6) -> list[dict]:
    """Use returned, coverage-eligible hits; no hit in either class is unavailable."""
    query_ids, reference_ids = _aliases(queries), _aliases(references)
    labels = {row["sample_id"]: int(row["target"]) for row in references}
    best: dict[str, dict[int, tuple[float, str]]] = {row["sample_id"]: {} for row in queries}
    for hit in hits:
        q, t = Path(hit.query).name, Path(hit.target).name
        if q not in query_ids or t not in reference_ids:
            raise ValueError("Foldseek hit outside the declared query/reference collections")
        values = (hit.qcov, hit.tcov, hit.qtmscore, hit.ttmscore)
        if any(v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Foldseek hit needs finite coverage and TM scores in [0,1]")
        if min(hit.qcov, hit.tcov) < min_coverage:
            continue
        qid, tid = query_ids[q], reference_ids[t]
        label, score = labels[tid], min(hit.qtmscore, hit.ttmscore)
        previous = best[qid].get(label)
        if previous is None or (-score, tid) < (-previous[0], previous[1]):
            best[qid][label] = (score, tid)
    rows = []
    for query in queries:
        evidence = best[query["sample_id"]]
        positive, negative = evidence.get(1), evidence.get(0)
        margin = ((positive[0] if positive else 0.) - (negative[0] if negative else 0.)) if evidence else None
        rows.append({"sample_id": query["sample_id"], "method": "Foldseek",
                     "status": "ok" if evidence else "unavailable",
                     "reason": "" if evidence else "no_coverage_eligible_hit",
                     "score": margin, "prediction": int(margin > 0) if margin is not None else None,
                     "positive_hit_available": positive is not None,
                     "negative_hit_available": negative is not None,
                     "best_positive_score": positive[0] if positive else None,
                     "best_negative_score": negative[0] if negative else None,
                     "best_positive_id": positive[1] if positive else None,
                     "best_negative_id": negative[1] if negative else None})
    return rows


def _prepare(rows: list[dict], folder: Path) -> tuple[list[dict], list[dict]]:
    folder.mkdir()
    ready, inspection = [], []
    for row in rows:
        item = dict(row, status="unavailable", reason="")
        try:
            structure = _parse_structure(Path(row["structure_path"]))
            models = list(structure.get_models())
            model = models[int(row.get("model_id", 0))]
            chain = model[row["chain_id"] or " "]
            residues = _chain_residues(chain)
            if len(residues) < 4:
                raise ValueError("Fewer than four protein CA residues")
            exported = folder / (row["sample_id"].encode().hex() + ".cif")
            copy = chain.copy()
            copy.id = "A"
            writer = MMCIFIO()
            writer.set_structure(copy)
            writer.save(str(exported), select=_ProteinResidueSelect())
            item.update(status="ok", n_observed_residues=len(residues), exported_path=str(exported),
                        exported_chain_id="A", model_serial=model.serial_num)
            ready.append(row)
        except (OSError, ValueError, KeyError, IndexError, PDBException, PDBConstructionException) as error:
            item["reason"] = f"{type(error).__name__}: {error}"
        inspection.append(item)
    return ready, inspection


def run(queries: list[dict], references: list[dict], output: Path, *, foldseek: str,
        task: str, threads: int = 1, timeout: float = 300, min_coverage: float = .6,
        exhaustive: bool = False, scratch_dir: Path | None = None,
        alignment_type: int = 1) -> dict:
    if task not in TASKS:
        raise ValueError("Declare a supported binary task")
    if threads < 1 or not math.isfinite(timeout) or timeout <= 0 or not 0 <= min_coverage <= 1:
        raise ValueError("Invalid resource or coverage settings")
    if alignment_type not in (1, 2):
        raise ValueError("alignment_type must be 1 (global TMalign) or 2 (local 3Di+AA)")
    validate_collections(queries, references)
    binary = shutil.which(foldseek)
    if binary is None:
        raise FileNotFoundError(f"Foldseek executable unavailable: {foldseek}")
    version = subprocess.run([binary, "version"], capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "queries.json").write_text(json.dumps(queries, indent=2) + "\n")
    (output / "references.json").write_text(json.dumps(references, indent=2) + "\n")
    ready_q, inspect_q = _prepare(queries, output / "query_coordinates")
    ready_r, inspect_r = _prepare(references, output / "reference_coordinates")
    (output / "input-inspection.json").write_text(json.dumps({"queries": inspect_q, "references": inspect_r}, indent=2) + "\n")
    results = {row["sample_id"]: {"sample_id": row["sample_id"], "method": "Foldseek",
               "status": "unavailable", "score": None, "prediction": None,
               "reason": row["reason"] or "reference_class_unavailable"} for row in inspect_q}
    command, execution = [], "not_run"
    started = time.monotonic()
    if ready_q and {int(row["target"]) for row in ready_r} == {0, 1}:
        # Foldseek creates symlinks: scratch defaults to Linux /tmp, independently
        # of the drive holding persistent biological inputs and outputs.
        with tempfile.TemporaryDirectory(prefix="betlas-specialist-foldseek-", dir=scratch_dir) as work:
            command = [binary, "easy-search", str(output / "query_coordinates"),
                       str(output / "reference_coordinates"), str(output / "hits.tsv"), work,
                       "--threads", str(threads), "--gpu", "0", "--alignment-type", str(alignment_type),
                       "--chain-name-mode", "1", "--format-output", ",".join(DEFAULT_FORMAT_FIELDS),
                       "--max-seqs", str(len(ready_r)), "-e", "10",
                       "--exhaustive-search", str(int(exhaustive))]
            (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            try:
                with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=timeout)
                if not (output / "hits.tsv").is_file():
                    raise RuntimeError("Foldseek succeeded without a hit table")
                parsed = summarize_hits(load_hits_tsv(output / "hits.tsv"), ready_q, ready_r,
                                        min_coverage=min_coverage)
                results.update({row["sample_id"]: row for row in parsed})
                execution = "ok"
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
                execution = "failed"
                for row in ready_q:
                    results[row["sample_id"]]["reason"] = f"search_failed: {type(error).__name__}: {error}"
    result_rows = []
    for i, row in enumerate(results.values()):
        row.update(task=task, input_regime="coordinates", score_is_probability=False,
                   score_semantics="best_positive_minus_best_negative_min_query_target_TM",
                   missing_class_score=0, decision_rule="score > 0; equal scores predict negative")
        result_path = output / f"result-{i:05d}.json"
        result_path.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        result_rows.append({"sample_id": row["sample_id"], "result_path": result_path.name})
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "result_path"])
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {"method": "Foldseek", "task": task, "version": version, "execution_status": execution,
               "elapsed_seconds": time.monotonic() - started, "n_queries": len(queries),
               "n_references": len(references), "n_available_references": len(ready_r),
               "statuses": dict(Counter(row["status"] for row in results.values())),
               "min_coverage": min_coverage, "exhaustive": exhaustive, "command": command,
               "alignment_type": alignment_type,
               "alignment_semantics": "global TMalign" if alignment_type == 1 else "local 3Di+AA with TM rescoring",
               "search_scope": "returned coverage-eligible hits; absent class has zero evidence",
               "reference_separation": "declared sample, PDB, biological group, exact coordinate source",
               "scope_limit": "Caller must audit upstream training overlap and full-catalog group links"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foldseek", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--min-coverage", type=float, default=.6)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--alignment-type", type=int, choices=(1, 2), default=1,
                        help="1: global TMalign; 2: native default local 3Di+AA with TM output")
    parser.add_argument("--scratch-dir", type=Path)
    args = parser.parse_args(argv)
    summary = run(read_manifest(args.queries), read_manifest(args.references), args.output_dir,
                  foldseek=args.foldseek, task=args.task, threads=args.threads,
                  timeout=args.timeout_seconds, min_coverage=args.min_coverage,
                  exhaustive=args.exhaustive, scratch_dir=args.scratch_dir,
                  alignment_type=args.alignment_type)
    print(json.dumps(summary, indent=2))
    return 0 if summary["execution_status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
