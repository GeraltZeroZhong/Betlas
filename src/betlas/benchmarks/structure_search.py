"""Training-only Foldseek nearest-neighbor baselines for frozen CATH splits.

Supply exact domain exports from the raw-coordinate export workflow. Each
search builds its target collection solely from available training domains.
This CPU workflow uses local 3Di+AA alignment and ranks hits by descending bit
score, ascending E-value, then target record ID. It never downloads a database.
Transient working databases require a filesystem supporting symbolic links;
``--scratch-dir`` controls their location independently of inputs and results.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from betlas.io.mmcif import read_mmcif_dict
from betlas.provenance import sha256_file

from .classification import (
    MANIFEST_COLUMNS,
    A,
    T,
    _manifest,
    _source_available,
    _validate_splits,
    summarize_predictions,
)


def _strings(data: dict, key: str) -> list[str]:
    values = data.get(key, [])
    return [str(value) for value in values] if isinstance(values, list) else [str(values)]


def _structure_availability(manifest: pd.DataFrame, exports: pd.DataFrame) -> pd.DataFrame:
    required = {"record_id", "structure_path", "export_status"}
    if required - set(exports):
        raise ValueError(f"Domain exports lack columns: {sorted(required - set(exports))}")
    if exports.record_id.duplicated().any():
        raise ValueError("Domain export record_id must be unique")
    if not set(exports.record_id).issubset(set(manifest.record_id)):
        raise ValueError("Domain exports include records outside the frozen manifest")
    by_id = exports.set_index("record_id")
    rows = []
    for record_id in manifest.record_id:
        row: dict[str, Any] = {"record_id": record_id, "structure_path": "", "structure_status": "export_unavailable",
                               "structure_reason": "No successful exact domain export", "structure_sha256": "",
                               "chain_id": "", "label_chain_id": ""}
        if record_id not in by_id.index or str(by_id.loc[record_id, "export_status"]) != "ok":
            rows.append(row)
            continue
        path = Path(str(by_id.loc[record_id, "structure_path"])).expanduser()
        if not path.is_file():
            row["structure_reason"] = "Exported coordinate file is unavailable"
            rows.append(row)
            continue
        try:
            data = read_mmcif_dict(path)
            labels = _strings(data, "_atom_site.label_seq_id")
            author = _strings(data, "_atom_site.auth_asym_id")
            chains = _strings(data, "_atom_site.label_asym_id")
            atom_names = _strings(data, "_atom_site.label_atom_id")
            if len({len(labels), len(author), len(chains), len(atom_names)}) != 1:
                raise ValueError("Inconsistent domain atom columns")
            selected = [i for i, value in enumerate(labels) if value not in {"", ".", "?"} and atom_names[i] == "CA"]
            author_chains = {author[i] for i in selected}
            label_chains = {chains[i] for i in selected}
            if not selected:
                raise ValueError("No polymer CA records in domain export")
            if len(author_chains) != 1 or len(label_chains) != 1:
                row.update(structure_status="multichain_requires_explicit_pooling",
                           structure_reason="The monomer baseline requires exactly one polymer chain")
            else:
                models = set(_strings(data, "_atom_site.pdbx_PDB_model_num")) - {"", ".", "?"}
                if len(models) > 1:
                    raise ValueError("Domain export contains multiple coordinate models")
                row.update(structure_path=str(path.resolve()), structure_status="ok", structure_reason="",
                           structure_sha256=sha256_file(path), chain_id=next(iter(author_chains)),
                           label_chain_id=next(iter(label_chains)))
        except (OSError, ValueError, KeyError) as exc:
            row.update(structure_status="invalid_structure", structure_reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return pd.DataFrame(rows)


def _aliases(table: pd.DataFrame) -> dict[str, str]:
    aliases = {}
    for row in table.itertuples():
        name = row.record_id.encode("utf-8").hex()
        candidates = {name, f"{name}.cif"}
        for chain in (row.chain_id, row.label_chain_id):
            candidates.update({f"{name}_{chain}", f"{name}.cif_{chain}"})
        for candidate in candidates:
            if candidate in aliases and aliases[candidate] != row.record_id:
                raise ValueError("Ambiguous Foldseek structure identifier")
            aliases[candidate] = row.record_id
    return aliases


def parse_best_hits(path: Path, queries: pd.DataFrame, training: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Parse only approved query/training identifiers; reject outside references."""
    query_ids, target_ids = _aliases(queries), _aliases(training)
    best = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"Foldseek hit line {number} must have query,target,bits,evalue")
        query_name, target_name, bits_text, evalue_text = fields
        query_name, target_name = Path(query_name).name, Path(target_name).name
        if query_name not in query_ids:
            raise ValueError(f"Foldseek query is outside the frozen test collection: {query_name}")
        if target_name not in target_ids:
            raise ValueError(f"Foldseek target is outside the frozen training collection: {target_name}")
        bits, evalue = float(bits_text), float(evalue_text)
        if not math.isfinite(bits) or not math.isfinite(evalue) or evalue < 0:
            raise ValueError(f"Invalid Foldseek score at line {number}")
        query, target = query_ids[query_name], target_ids[target_name]
        key = (-bits, evalue, target)
        if query not in best or key < best[query]["rank_key"]:
            best[query] = {"target_record_id": target, "bits": bits, "evalue": evalue, "rank_key": key}
    return best


def run_structure_search(
    manifest: pd.DataFrame,
    exports: pd.DataFrame,
    splits: pd.DataFrame,
    out_dir: str | Path,
    *,
    foldseek_binary: str | Path,
    label_column: str = A,
    threads: int = 2,
    evalue: float = 0.001,
    sensitivity: float = 9.5,
    max_seqs: int = 1000,
    timeout: float = 3600,
    scratch_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Evaluate nearest neighbors using each split's frozen training library.

All frozen test records stay in the denominator, including unavailable exports,
unsupported multichain crops, execution failures and queries with no accepted
hit. Unknown returned identifiers raise, preventing a contaminated comparison.
    """
    if label_column not in (A, T):
        raise ValueError("label_column must be native architecture or topology")
    if threads < 1 or max_seqs < 1 or timeout <= 0 or not math.isfinite(evalue) or evalue < 0:
        raise ValueError("Invalid thread, hit-count, timeout or E-value setting")
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("sensitivity must be finite and positive")
    frame = _manifest(manifest)
    frozen = splits.copy()
    _validate_splits(frame, frozen, label_column)
    binary = shutil.which(str(foldseek_binary))
    if binary is None:
        raise FileNotFoundError(f"Foldseek executable is unavailable: {foldseek_binary}")
    version_result = subprocess.run([binary, "version"], capture_output=True, text=True, check=True, timeout=20)
    version = (version_result.stdout + version_result.stderr).strip()
    out = Path(out_dir)
    scratch_root = Path(scratch_dir) if scratch_dir is not None else Path(tempfile.gettempdir())
    scratch_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "predictions.csv").exists() or (out / "splits").exists():
        raise ValueError("Structure-search output already exists; use a new output directory")
    available = _structure_availability(frame, exports)
    available.to_csv(out / "structure_availability.csv", index=False)
    source_columns = [column for column in ("representative_status", "coordinate_status") if column in frame]
    joined = frame[list(MANIFEST_COLUMNS) + source_columns].merge(
        available, on="record_id", how="left", validate="one_to_one").set_index("record_id")
    predictions, runs = [], []
    for split_id, part in frozen.groupby("split_id", sort=True):
        data = joined.loc[part.record_id].reset_index()
        for column in ("role", "group_id", "target_label"):
            data[column] = part[column].to_numpy()
        data["held_out"] = (
            part.held_out.astype(str).str.lower().isin(["true", "1"]).to_numpy()
            if "held_out" in part else data.role.eq("test")
        )
        data["pdb_policy"] = part.pdb_policy.to_numpy() if "pdb_policy" in part else "unspecified"
        data["input_source_available"] = _source_available(data)
        data["split_id"] = str(split_id)
        data["eligibility_reason"] = part.get("eligibility_reason", pd.Series("unspecified", index=part.index)).to_numpy()
        data["feature_available"] = data.structure_status.eq("ok")
        train = data[data.role.eq("train") & data.feature_available].copy()
        queries = data[data.role.eq("test") & data.feature_available].copy()
        labels = train.set_index("record_id")[label_column].to_dict()
        data["class_evaluation_status"] = data[label_column].isin(set(labels.values())).map(
            {True: "training_class_available", False: "training_class_unavailable"})
        data["prediction"], data["target_record_id"] = "", ""
        data["bits"], data["evalue"] = math.nan, math.nan
        data["prediction_status"] = data.role
        data.loc[data.role.eq("test"), "prediction_status"] = data.loc[data.role.eq("test"), "structure_status"]
        can_predict = data.role.eq("test") & data.feature_available
        data.loc[can_predict, "prediction_status"] = "no_training_structures" if train.empty else "no_hit"
        split_dir = out / "splits" / str(split_id).encode("utf-8").hex()
        split_dir.mkdir(parents=True)
        train.to_csv(split_dir / "training.csv", index=False)
        queries.to_csv(split_dir / "queries.csv", index=False)
        run = {"split_id": str(split_id), "n_train_planned": int(data.role.eq("train").sum()),
               "n_train_available": len(train), "n_test_planned": int(data.role.eq("test").sum()),
               "n_query_available": len(queries), "training_classes": sorted(set(labels.values())),
               "status": "not_run_empty_collection", "command": []}
        if not train.empty and not queries.empty:
            # Foldseek creates symbolic links in its working database tree.
            # Keep this transient tree on a supporting filesystem (e.g. Linux
            # /tmp), even when immutable coordinates and outputs use exFAT.
            with tempfile.TemporaryDirectory(prefix="search-", dir=split_dir) as temporary, \
                    tempfile.TemporaryDirectory(prefix="betlas-foldseek-", dir=scratch_root) as working:
                scratch = Path(temporary)
                for name, table in (("queries", queries), ("training", train)):
                    directory = scratch / name
                    directory.mkdir()
                    for row in table.itertuples():
                        shutil.copyfile(row.structure_path, directory / (row.record_id.encode("utf-8").hex() + ".cif"))
                hits = split_dir / "hits.tsv"
                command = [binary, "easy-search", str(scratch / "queries"), str(scratch / "training"), str(hits),
                           str(Path(working) / "work"), "--threads", str(threads), "--gpu", "0",
                           "--alignment-type", "2", "--sort-by-structure-bits", "0", "--chain-name-mode", "1",
                           "--format-output", "query,target,bits,evalue", "-e", str(evalue),
                           "-s", str(sensitivity), "--max-seqs", str(max_seqs)]
                run["command"] = command
                try:
                    with (split_dir / "stdout.log").open("w") as stdout, (split_dir / "stderr.log").open("w") as stderr:
                        subprocess.run(command, check=True, stdout=stdout, stderr=stderr, timeout=timeout)
                except (OSError, subprocess.SubprocessError) as exc:
                    run.update(status="search_failed", reason=f"{type(exc).__name__}: {exc}")
                    data.loc[can_predict, "prediction_status"] = "search_failed"
                else:
                    if not hits.is_file():
                        raise ValueError("Foldseek succeeded without a hit table")
                    # A bad/outside identifier is a protocol violation, not a
                    # no-hit result; fail before writing comparative predictions.
                    best = parse_best_hits(hits, queries, train)
                    run["status"] = "ok"
                    for query, hit in best.items():
                        mask = data.record_id.eq(query)
                        data.loc[mask, "prediction"] = labels[hit["target_record_id"]]
                        data.loc[mask, "prediction_status"] = "ok"
                        for key in ("target_record_id", "bits", "evalue"):
                            data.loc[mask, key] = hit[key]
        runs.append(run)
        (split_dir / "search.json").write_text(json.dumps(run, indent=2) + "\n")
        columns = list(MANIFEST_COLUMNS) + source_columns + [
                    "group_id", "split_id", "role", "held_out", "pdb_policy", "target_label", "eligibility_reason",
                    "input_source_available",
                    "feature_available", "prediction", "prediction_status", "class_evaluation_status",
                    "structure_status", "structure_reason", "target_record_id", "bits", "evalue"]
        predictions.append(data.loc[~data.role.eq("train"), columns])
    predicted = pd.concat(predictions, ignore_index=True)
    summary, per_class = summarize_predictions(predicted, labels=sorted(frame[label_column].unique()))
    summary.update(model="foldseek_nearest_neighbor", label_column=label_column, foldseek_binary=binary,
                   foldseek_version=version, alignment_type=2, ranking="descending bits, ascending evalue, target record_id",
                   evalue_threshold=evalue, sensitivity=sensitivity, max_seqs=max_seqs, threads=threads,
                   scratch_dir=str(scratch_root.resolve()),
                   multichain_policy="unavailable until explicit pooling is defined", fits=runs)
    predicted.to_csv(out / "predictions.csv", index=False)
    per_class.to_csv(out / "per_class.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return predicted, summary, per_class


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "exports", "splits", "out-dir", "foldseek-binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--label-column", choices=(A, T), default=A)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--evalue", type=float, default=0.001)
    parser.add_argument("--sensitivity", type=float, default=9.5)
    parser.add_argument("--max-seqs", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--scratch-dir", type=Path,
                        help="Transient Foldseek database directory on a filesystem supporting symbolic links (default: system temporary directory)")
    args = vars(parser.parse_args(argv))
    input_paths = {name: args[name] for name in ("manifest", "exports", "splits")}
    for name, path in input_paths.items():
        args[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
    _, summary, _ = run_structure_search(**args)
    protocol = {"schema": "betlas.foldseek-benchmark.v1", "command": "run", "inputs": {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in input_paths.items()},
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in args.items()
                       if key not in input_paths},
        "source_sha256": sha256_file(Path(__file__)),
        "foldseek_version": summary["foldseek_version"],
        "identity_consumer": "assert_comparable_protocols validates shared manifest, split and label identities"}
    (args["out_dir"] / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("n_test", "n_predictions", "prediction_coverage")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
