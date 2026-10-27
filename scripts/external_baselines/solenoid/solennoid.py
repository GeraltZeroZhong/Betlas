"""Normalize SOLeNNoID output and run its separately installed prediction API."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from Bio.PDB.PDBExceptions import PDBException

from .structures import inspect_structure

CLASS_NAMES = ("non", "beta", "alphabeta", "alpha")


@dataclass(frozen=True)
class ResiduePrediction:
    sample_id: str
    chain_id: str
    residue_number: int
    insertion_code: str
    predicted_class: int
    native_score_non: float
    native_score_beta: float
    native_score_alphabeta: float
    native_score_alpha: float
    omitted_probability_mass: float


def _prediction(sample_id: str, residue: dict, scores) -> ResiduePrediction:
    values = tuple(float(x) for x in scores)
    if len(values) != 4 or any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
        raise ValueError("Expected four finite native softmax channels in [0, 1]")
    if sum(values) > 1.00001:
        raise ValueError("Retained native channels sum to more than one")
    # Upstream takes argmax of the first four of six channels. Do not renormalize.
    label = max(range(4), key=lambda index: values[index])
    return ResiduePrediction(
        sample_id, residue["chain_id"], int(residue["residue_number"]),
        residue.get("insertion_code", ""), label, *values, max(0.0, 1 - sum(values)),
    )


def parse_native_csv(path: str | Path, *, sample_id: str, mapping: list[dict]) -> list[ResiduePrediction]:
    """Parse verified upstream CLI schema, only with an explicit identity mapping.

    The official CLI misassigns names/indices for gapped residue numbering. Such
    native CSVs cannot be rescued by positional relabelling; use the API worker.
    """
    expected = {}
    previous = {}
    for residue in mapping:
        key = (residue["chain_id"], int(residue["residue_number"]))
        if residue.get("insertion_code") or key in expected:
            raise ValueError("Native CSV cannot represent insertion codes or duplicate numbers")
        if key[0] in previous and key[1] != previous[key[0]] + 1:
            raise ValueError("Native CLI CSV requires contiguous residue numbers; use API worker")
        previous[key[0]] = key[1]
        expected[key] = residue
    if not expected:
        raise ValueError("An explicit nonempty residue mapping is required")
    parsed = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"chain", "index", "residue", "one_hot", "class"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Missing native SOLeNNoID CSV columns")
        for row in reader:
            key = (row["chain"], int(row["index"]))
            if key not in expected or key in parsed:
                raise ValueError("Unknown or duplicated native residue identity")
            residue = expected[key]
            if row["residue"] != residue["residue_name"]:
                raise ValueError("Native residue name does not match coordinate mapping")
            vector = row["one_hot"].strip()
            if not vector.startswith("[") or not vector.endswith("]"):
                raise ValueError("Malformed native score vector")
            values = [float(x) for x in re.split(r"[\s,]+", vector[1:-1].strip()) if x]
            prediction = _prediction(sample_id, residue, values)
            if row["class"] != CLASS_NAMES[prediction.predicted_class]:
                raise ValueError("Native class disagrees with retained-channel argmax")
            parsed[key] = prediction
    if parsed.keys() != expected.keys():
        raise ValueError("Missing native residues or silently skipped chains")
    return [parsed[key] for key in expected]


def map_span_scores(path: str | Path, *, sample_id: str, inspection: dict) -> list[ResiduePrediction]:
    """Map integer-span rows to observed residues; padded positions are not truth."""
    if inspection["status"] != "ready":
        raise ValueError("Cannot map scores for an unsupported structure")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != inspection["matrix_span"]:
        raise ValueError("Native score span does not match inspected coordinate numbering")
    for index, row in enumerate(rows):
        if int(row["span_index"]) != index:
            raise ValueError("Native span rows are reordered or duplicated")
    return [
        _prediction(sample_id, residue, [rows[residue["native_span_index"]][f"score_{i}"] for i in range(4)])
        for residue in inspection["residues"]
    ]


def domain_summary(predictions: list[ResiduePrediction], *, sample_id: str) -> dict:
    if any(row.sample_id != sample_id for row in predictions):
        raise ValueError("Cannot aggregate predictions from another sample")
    if len({(x.chain_id, x.residue_number, x.insertion_code) for x in predictions}) != len(predictions):
        raise ValueError("Duplicate residue identities in domain aggregation")
    n = len(predictions)
    return {
        "sample_id": sample_id, "method": "SOLeNNoID", "status": "ok" if n else "unavailable",
        "n_observed_predicted_residues": n,
        "beta_fraction": sum(x.predicted_class == 1 for x in predictions) / n if n else None,
        "any_solenoid_fraction": sum(x.predicted_class != 0 for x in predictions) / n if n else None,
        "aggregation": "fraction_of_observed_residue_hard_calls",
        "score_is_probability": False,
    }


def run_baseline(
    structure: str | Path, *, sample_id: str, source_dir: str | Path,
    model_path: str | Path, python_executable: str | Path, output_dir: str | Path,
    chain_id: str | None = None, max_span: int = 2048, timeout_seconds: float = 120,
    keras3_groups_one_compat: bool = False,
) -> dict:
    """Run one local CPU worker; preserve input files, diagnostics and missingness."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        inspection = inspect_structure(structure, chain_id=chain_id, max_span=max_span)
    except (OSError, ValueError, PDBException, StopIteration) as error:
        inspection = {"status": "unavailable", "reason": "coordinate_parse_failed",
                      "detail": str(error), "structure_path": str(Path(structure).resolve())}
    (output / "inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")
    result = {"sample_id": sample_id, "method": "SOLeNNoID", "status": "unavailable", "reason": inspection["reason"]}
    if inspection["status"] == "ready":
        source = Path(source_dir).resolve()
        model = Path(model_path).resolve()
        if not (source / "solenoid_detector_onehot_v3.py").is_file() or not model.is_file():
            result["reason"] = "missing_upstream_source_or_model"
        else:
            command = [str(python_executable), str(Path(__file__).with_name("worker.py")),
                       "--source-dir", str(source), "--model-path", str(model),
                       "--inspection", str(output / "inspection.json"), "--output-dir", str(output)]
            if keras3_groups_one_compat:
                command.append("--keras3-groups-one-compat")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTEROP_THREADS="1",
                       TF_NUM_INTRAOP_THREADS="1", OMP_NUM_THREADS="1")
            started = time.monotonic()
            (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
                try:
                    process = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                             timeout=timeout_seconds, check=False)
                    result["returncode"] = process.returncode
                    if process.returncode:
                        result["reason"] = "upstream_worker_failed"
                    else:
                        predictions = map_span_scores(output / "native_span_scores.csv", sample_id=sample_id, inspection=inspection)
                        result = domain_summary(predictions, sample_id=sample_id)
                        with (output / "residue_predictions.csv").open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(asdict(predictions[0])))
                            writer.writeheader()
                            writer.writerows(asdict(x) for x in predictions)
                except subprocess.TimeoutExpired:
                    result["reason"] = "timeout"
                except (OSError, ValueError, KeyError) as error:
                    result.update(reason="worker_or_output_error", detail=str(error))
            result["elapsed_seconds"] = time.monotonic() - started
    result["execution_mode"] = "local_cpu_api_worker"
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chain-id")
    parser.add_argument("--keras3-groups-one-compat", action="store_true")
    parser.add_argument("--max-span", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = vars(parser.parse_args())
    result = run_baseline(**args)
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
