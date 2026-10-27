"""Run the genuine Cooper-Beta CLI and retain its native hard decision."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from Bio.PDB.PDBExceptions import PDBConstructionException, PDBException

from .inputs import select_structure


def normalize_row(row: dict, *, sample_id: str) -> dict:
    native = row.get("result")
    usable = native in {"BARREL", "NON_BARREL"} and str(row.get("degraded", "false")).lower() not in {"true", "1"}
    return {"method": "cooper_beta", "sample_id": sample_id,
            "task": "barrel_geometry", "input_regime": "coordinates",
            "status": "ok" if usable else "unavailable", "score": None,
            "prediction": int(native == "BARREL") if usable else None,
            "score_semantics": "native_hard_call_only",
            "reason": "" if usable else row.get("reason", "native_error"),
            "native_result": row}


def run_sample(structure_path: str | Path, *, sample_id: str, chain_id: str,
               output_dir: str | Path, python_executable: str = sys.executable,
               source_root: str | Path | None = None, dssp_executable: str | None = None,
               model_index: int = 0, timeout: float = 120.0) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"method": "cooper_beta", "sample_id": sample_id, "task": "barrel_geometry",
              "input_regime": "coordinates", "status": "unavailable", "score": None,
              "prediction": None, "reason": ""}
    try:
        selected, selection = select_structure(structure_path, output / "input", chain_id=chain_id,
                                               model_index=model_index)
        result["input"] = selection
        command = [python_executable, "-B", "-m", "cooper_beta", str(selected), "--workers", "1",
                   "--prepare-workers", "1", "--out", str(output / "native.csv"),
                   "runtime.prepare_cache_enabled=false", "output.hash_input_files=false"]
        if dssp_executable:
            command.append(f"runtime.dssp_bin_path={Path(dssp_executable).resolve()}")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if source_root is not None:
            environment["PYTHONPATH"] = str(Path(source_root).resolve() / "src")
            git = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=False)
            result["source"] = {"root": str(Path(source_root).resolve()), "git_commit": git.stdout.strip()}
        result["command"] = command
        completed = subprocess.run(command, cwd=output, env=environment, capture_output=True,
                                   text=True, timeout=timeout, check=False)
        (output / "stdout.txt").write_text(completed.stdout)
        (output / "stderr.txt").write_text(completed.stderr)
        result["returncode"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(f"Cooper-Beta exited with status {completed.returncode}; see native logs")
        with (output / "native.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0].get("author_chain_id") != chain_id:
            raise ValueError("Native output does not contain exactly the selected author chain")
        result.update(normalize_row(rows[0], sample_id=sample_id))
    except (OSError, ValueError, RuntimeError, PDBException, PDBConstructionException,
            subprocess.TimeoutExpired) as error:
        result.update(status="unavailable", prediction=None, score=None, reason=str(error))
    result["elapsed_seconds"] = time.monotonic() - started
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-root")
    parser.add_argument("--dssp")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    result = run_sample(args.structure, sample_id=args.sample_id, chain_id=args.chain,
                        model_index=args.model_index, output_dir=args.out_dir,
                        python_executable=args.python, source_root=args.source_root,
                        dssp_executable=args.dssp, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
