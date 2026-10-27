"""Evaluate specialist methods on their native endpoints, with explicit coverage.

Domain scores are not probabilities. Contact predictions require a declared pair
universe and input regime; this module does not turn them into CATH classifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BinaryCase:
    sample_id: str
    target: int
    split: str
    group_id: str
    pdb_id: str


@dataclass(frozen=True)
class Score:
    sample_id: str
    score: float | None
    status: str
    prediction: int | None = None


def _cases(rows: list[BinaryCase]) -> dict[str, BinaryCase]:
    result = {}
    for row in rows:
        if (not row.sample_id or not row.group_id or not row.pdb_id
                or row.target not in {0, 1} or row.split not in {"development", "test"}):
            raise ValueError("Cases need IDs, binary targets, groups, PDB IDs and development/test split")
        if row.sample_id in result:
            raise ValueError(f"Duplicate case: {row.sample_id}")
        result[row.sample_id] = row
    if not result:
        raise ValueError("No cases")
    return result


def _scores(rows: list[Score], cases: dict[str, BinaryCase]) -> dict[str, Score]:
    result = {}
    for row in rows:
        if row.sample_id not in cases or row.sample_id in result:
            raise ValueError(f"Unknown or duplicate score ID: {row.sample_id}")
        if not row.status:
            raise ValueError("Score status is required")
        if row.prediction is not None and row.prediction not in {0, 1}:
            raise ValueError("Native predictions must be binary")
        if row.status == "ok":
            if row.score is None and row.prediction is None:
                raise ValueError("Successful records need a finite score or native prediction")
            if row.score is not None and not math.isfinite(row.score):
                raise ValueError("Successful scores must be finite")
        elif row.score is not None or row.prediction is not None:
            raise ValueError("Unavailable predictions must not carry numeric scores or calls")
        result[row.sample_id] = row
    return result


def _divide(a: float, b: float) -> float | None:
    return a / b if b else None


def _decision(score: float, threshold: float, operator: str) -> bool:
    if operator not in {"ge", "gt"}:
        raise ValueError("Threshold operator must be ge or gt")
    return score >= threshold if operator == "ge" else score > threshold


def binary_metrics(cases: list[BinaryCase], scores: list[Score], *,
                   threshold: float | None, operator: str = "ge") -> dict[str, Any]:
    """Keep unavailable predictions as abstentions, never as true negatives.

    Fixed-denominator recalls count abstentions as unsuccessful. Precision uses
    actual emitted calls. Ranking metrics are conditional on score availability.
    """
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    if threshold is not None:
        _decision(0, threshold, operator)
    cohort = _cases(cases)
    predictions = _scores(scores, cohort)
    confusion = [[0, 0, 0], [0, 0, 0]]  # negative, positive, unavailable
    statuses = Counter()
    rank_y, rank_scores = [], []
    by_group: dict[str, list[int]] = defaultdict(list)
    for key, case in cohort.items():
        row = predictions.get(key, Score(key, None, "missing"))
        statuses[row.status] += 1
        if row.status == "ok":
            if threshold is None:
                if row.prediction is None:
                    raise ValueError("Native evaluation requires an explicit prediction")
                column = row.prediction
            else:
                if row.score is None:
                    raise ValueError("Hard calls require native evaluation, not a score threshold")
                column = int(_decision(row.score, threshold, operator))
            if row.score is not None:
                rank_y.append(case.target)
                rank_scores.append(row.score)
        else:
            column = 2
        confusion[case.target][column] += 1
        by_group[case.group_id].append(int(column == case.target))
    supports = [sum(row) for row in confusion]
    recall = [_divide(confusion[c][c], supports[c]) for c in [0, 1]]
    precision = [_divide(confusion[c][c], sum(row[c] for row in confusion)) for c in [0, 1]]
    # 2 TP / (2 TP + FP + FN); FN includes unavailable true-class records.
    f1 = [_divide(2 * confusion[c][c], supports[c] + sum(row[c] for row in confusion))
          for c in [0, 1]]
    ranking = {"average_precision": None, "roc_auc": None}
    if len(set(rank_y)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        ranking = {"average_precision": float(average_precision_score(rank_y, rank_scores)),
                   "roc_auc": float(roc_auc_score(rank_y, rank_scores))}
    return {
        "n": len(cases), "threshold": threshold,
        "operator": operator if threshold is not None else "native",
        "confusion_columns": ["negative", "positive", "unavailable"],
        "confusion_by_true_class": confusion, "support": supports,
        "coverage": statuses["ok"] / len(cases), "statuses": dict(statuses),
        "accuracy": (confusion[0][0] + confusion[1][1]) / len(cases),
        "recall": recall, "precision": precision, "f1": f1,
        "balanced_accuracy": sum(recall) / 2 if all(x is not None for x in recall) else None,
        "macro_f1": sum(f1) / 2 if all(x is not None for x in f1) else None,
        "group_mean_accuracy": sum(sum(v) / len(v) for v in by_group.values()) / len(by_group),
        "score_available_n": len(rank_y),
        "score_available_positive_n": sum(rank_y),
        "ranking_scope": "score-available records; native scores, not calibrated probabilities",
        "ranking": ranking,
    }


def calibrate_threshold(cases: list[BinaryCase], scores: list[Score], *, method: str) -> dict:
    """Choose a deterministic threshold using development records only."""
    cohort = _cases(cases)
    if not method or any(row.split != "development" for row in cases):
        raise ValueError("Calibration requires a method and development-only cases")
    if {row.target for row in cases} != {0, 1}:
        raise ValueError("Development must contain both classes")
    predictions = _scores(scores, cohort)
    if any(row.status == "ok" and row.score is None for row in predictions.values()):
        raise ValueError("Native hard calls cannot be threshold-calibrated")
    values = sorted({row.score for row in predictions.values() if row.status == "ok"})
    if not values:
        raise ValueError("No available development scores")
    above = math.nextafter(values[-1], math.inf)
    if not math.isfinite(above):
        raise ValueError("Score magnitude prevents finite threshold calibration")
    # Includes all-positive/all-negative decisions. Missing predictions remain incorrect.
    candidates = [*values, above]
    support = Counter(row.target for row in cases)
    scored = [(row.score, cohort[key].target) for key, row in predictions.items() if row.status == "ok"]
    def criterion(threshold: float) -> tuple[float, float]:
        correct = Counter(label for score, label in scored if int(score >= threshold) == label)
        return (sum(correct[c] / support[c] for c in [0, 1]) / 2, threshold)
    threshold = max(candidates, key=criterion)
    return {
        "schema": "betlas.specialist-threshold.v1", "method": method,
        "threshold": threshold, "operator": "ge",
        "criterion": "fixed-denominator balanced accuracy; ties choose higher threshold",
        "development_sample_ids": sorted(cohort),
        "development_group_ids": sorted({row.group_id for row in cases}),
        "development_pdb_ids": sorted({row.pdb_id.lower() for row in cases}),
        "development_metrics": binary_metrics(cases, scores, threshold=threshold),
    }


def evaluate_frozen_threshold(cases: list[BinaryCase], scores: list[Score],
                              calibration: dict, *, method: str) -> dict:
    _cases(cases)
    if calibration.get("schema") != "betlas.specialist-threshold.v1" or calibration.get("method") != method:
        raise ValueError("Threshold schema/method mismatch")
    if any(row.split != "test" for row in cases):
        raise ValueError("Evaluation requires test-only cases")
    for key, present in [
        ("development_sample_ids", {row.sample_id for row in cases}),
        ("development_group_ids", {row.group_id for row in cases}),
        ("development_pdb_ids", {row.pdb_id.lower() for row in cases}),
    ]:
        if key not in calibration or present.intersection(calibration[key]):
            raise ValueError(f"Missing separation evidence or development/test overlap: {key}")
    return binary_metrics(cases, scores, threshold=float(calibration["threshold"]),
                          operator=calibration["operator"])




def _pair(row: dict) -> tuple[str, int, int]:
    i, j = int(row["residue_i"]), int(row["residue_j"])
    if i < 1 or j < 1 or i == j:
        raise ValueError("Pair indices must be distinct positive 1-based integers")
    return str(row["sample_id"]), min(i, j), max(i, j)


def pair_metrics(reference: list[dict], calls: list[dict], statuses: dict[str, str], *,
                 unit: str, input_regime: str) -> dict:
    """Score hard calls over an explicitly supplied, complete eligible universe.

    Input reference rows: sample_id,residue_i,residue_j,target,orientation.
    Indices denote residue positions or strand IDs according to ``unit``.
    Unlisted reference pairs are outside the endpoint, not assumed negative.
    Missing methods remain unavailable; no AP is invented for hard calls.
    """
    if unit not in {"residue_contact", "strand_pair"}:
        raise ValueError("Declare residue_contact or strand_pair")
    if input_regime not in {"sequence_predicted_ss", "sequence_observed_ss", "structure_reference"}:
        raise ValueError("Declare the input regime; do not pool unequal inputs")
    orientations = {"parallel", "antiparallel", "bridge", "unknown", ""}
    universe, predictions = {}, {}
    for row in reference:
        key = _pair(row)
        if key in universe or int(row["target"]) not in {0, 1}:
            raise ValueError("Duplicate reference pair or nonbinary reference")
        if row.get("orientation", "") not in orientations:
            raise ValueError("Unknown reference orientation")
        universe[key] = row
    if not universe:
        raise ValueError("Empty pair universe")
    sample_ids = {key[0] for key in universe}
    if set(statuses) - sample_ids or any(not status for status in statuses.values()):
        raise ValueError("Unknown sample or empty execution status")
    for row in calls:
        key = _pair(row)
        if key not in universe or key in predictions:
            raise ValueError("Duplicate predicted pair or pair outside declared universe")
        if statuses.get(key[0]) != "ok":
            raise ValueError("Calls require successful execution status")
        if row.get("orientation", "") not in orientations:
            raise ValueError("Unknown call orientation")
        predictions[key] = row
    records = {}
    for identifier in sorted(sample_ids):
        available = statuses.get(identifier, "missing") in {"ok", "empty"}
        positives = {k for k, v in universe.items() if k[0] == identifier and int(v["target"]) == 1}
        emitted = {k for k in predictions if k[0] == identifier}
        tp, fp, fn = len(positives & emitted), len(emitted - positives), len(positives - emitted)
        true_oriented = {k for k in positives if universe[k].get("orientation") in {"parallel", "antiparallel"}}
        oriented_correct = sum(predictions.get(k, {}).get("orientation") == universe[k]["orientation"]
                               for k in true_oriented)
        records[identifier] = {
            "status": statuses.get(identifier, "missing"), "tp": tp, "fp": fp, "fn": fn,
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, len(positives)),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
            "orientation_recall": _divide(oriented_correct, len(true_oriented)),
            "eligible_pairs": sum(k[0] == identifier for k in universe),
            "available": available,
        }
    def mean(key: str) -> float | None:
        values = [r[key] for r in records.values() if r[key] is not None]
        return sum(values) / len(values) if values else None
    return {"unit": unit, "input_regime": input_regime, "n": len(records),
            "coverage": sum(r["available"] for r in records.values()) / len(records),
            "macro_precision": mean("precision"), "macro_recall": mean("recall"), "macro_f1": mean("f1"),
            "macro_orientation_recall": mean("orientation_recall"), "per_sample": records,
            "reference_requirement": "complete eligible pair universe, independent endpoint definition",
            "comparison_limit": "compare predictors only within the same declared input regime"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="", buffering=8192) as handle:
        return list(csv.DictReader(handle))




def import_domain_scores(results: list[dict], *, method: str) -> list[dict]:
    """Bridge normalized specialist JSON to the common score CSV contract."""
    fields = {"SOLeNNoID": "beta_fraction", "RAPHAEL": "svm_score",
              "Foldseek": "score", "cooper_beta": "score",
              "cooper_beta_learned": "score",
              "isitabarrel_structure_map": "score", "pred_tmbb2_single_juchmme": "score",
              "betlas_barrel_geometry": "score"}
    if method not in fields:
        raise ValueError("Unsupported specialist method")
    for field in ("task", "input_regime"):
        if len({result[field] for result in results if result.get(field)}) > 1:
            raise ValueError(f"Import one {field} at a time")
    rows, seen = [], set()
    for result in results:
        identifier = result["sample_id"]
        if not identifier or identifier in seen or result.get("method") != method:
            raise ValueError("Duplicate/empty sample or incompatible method")
        seen.add(identifier)
        status = result["status"]
        prediction = result.get("prediction")
        if status == "ok":
            raw = result.get(fields[method])
            value = float(raw) if raw is not None else ""
            if (value != "" and not math.isfinite(value)) or (method == "SOLeNNoID" and
                    (value == "" or not 0 <= value <= 1)):
                raise ValueError("Invalid native domain score")
            if method == "RAPHAEL" and value == "":
                raise ValueError("RAPHAEL requires its native SVM score")
            if prediction is not None and prediction not in {0, 1}:
                raise ValueError("Native predictions must be binary")
            if value == "" and prediction is None:
                raise ValueError("Successful result needs a score or native prediction")
        else:
            value = ""
            prediction = None
        rows.append({"sample_id": identifier, "score": value, "status": status,
                     "reason": result.get("reason", "")})
        if method not in {"SOLeNNoID", "RAPHAEL"}:
            rows[-1]["prediction"] = prediction if prediction is not None else ""
            rows[-1]["task"] = result.get("task", "")
            rows[-1]["input_regime"] = result.get("input_regime", "")
    return rows


def import_betlas_barrel_rows(native: list[dict], samples: list[dict]) -> list[dict]:
    """Map native chain decisions through explicit source paths and author chains."""
    indexed = {}
    for row in native:
        key = (str(Path(row["source_path"]).resolve()), row["chain"])
        if key in indexed:
            raise ValueError("Duplicate native source/chain result")
        indexed[key] = row
    output, seen = [], set()
    for sample in samples:
        identifier = sample["sample_id"]
        if not identifier or identifier in seen:
            raise ValueError("Duplicate or empty sample ID")
        seen.add(identifier)
        key = (str(Path(sample["source_path"]).resolve()), sample["chain_id"])
        row = indexed.get(key)
        result = {"sample_id": identifier, "method": "betlas_barrel_geometry",
                  "task": "barrel_geometry", "input_regime": "coordinates",
                  "status": "unavailable", "score": None, "prediction": None,
                  "reason": "missing_native_source_chain"}
        if row is not None:
            usable = row.get("result_stage") == "decision" and row.get("result") in {"BARREL", "NON_BARREL"}
            result["reason"] = row.get("reason", "")
            if usable:
                value = float(row["decision_score"])
                if not math.isfinite(value):
                    raise ValueError("Native Betlas decision score must be finite")
                result.update(status="ok", score=value, prediction=int(row["result"] == "BARREL"))
        output.append(result)
    return import_domain_scores(output, method="betlas_barrel_geometry")


def import_pair_calls(results: list[dict], *, unit: str) -> tuple[list[dict], list[dict]]:
    """Convert native hard calls while retaining empty-success and bridge labels.

    Native strand IDs are not automatically equivalent across tools: callers must
    supply the corresponding fixed strand universe or explicitly map them first.
    """
    if unit not in {"residue_contact", "strand_pair"}:
        raise ValueError("Declare residue_contact or strand_pair")
    collection, left, right = (("contacts", "residue_i", "residue_j") if unit == "residue_contact"
                               else ("strand_pairs", "strand_i", "strand_j"))
    if len({r.get("tool") for r in results}) > 1:
        raise ValueError("Import one method at a time")
    calls, statuses, seen = [], [], set()
    for result in results:
        identifier = result["sample_id"]
        if not identifier or identifier in seen:
            raise ValueError("Duplicate or empty sample ID")
        seen.add(identifier)
        if result.get("normalized_index_base") != 1:
            raise ValueError("Import requires normalized one-based indices")
        if "beta_residue_positions" in result:
            raise ValueError("A BetaPro probability map is not a hard-call prediction")
        status = result["status"]
        records = result.get(collection, [])
        if records and status != "ok":
            raise ValueError("Calls require successful nonempty status")
        statuses.append({"sample_id": identifier, "status": status})
        calls.extend({"sample_id": identifier, "residue_i": row[left], "residue_j": row[right],
                      "orientation": row.get("orientation", "unknown")} for row in records)
    return calls, statuses


def _results_manifest(path: Path) -> list[dict]:
    results = []
    for row in read_csv(path):
        result = json.loads((path.parent / row["result_path"]).read_text(encoding="utf-8"))
        if result["sample_id"] != row["sample_id"]:
            raise ValueError("Result identity does not match manifest")
        results.append(result)
    return results


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["calibrate", "evaluate"]:
        command = sub.add_parser(name)
        command.add_argument("--labels", type=Path, required=True)
        command.add_argument("--scores", type=Path, required=True)
        command.add_argument("--method", required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "evaluate":
            threshold = command.add_mutually_exclusive_group(required=True)
            threshold.add_argument("--calibration", type=Path)
            threshold.add_argument("--fixed-threshold", type=float)
            threshold.add_argument("--native-decisions", action="store_true")
            command.add_argument("--operator", choices=["ge", "gt"], default="ge")
    pair = sub.add_parser("pairs")
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--calls", type=Path, required=True)
    pair.add_argument("--statuses", type=Path, required=True)
    pair.add_argument("--unit", choices=["residue_contact", "strand_pair"], required=True)
    pair.add_argument("--input-regime", required=True)
    pair.add_argument("--out", type=Path, required=True)
    barrel = sub.add_parser("betlas-barrel-scores")
    barrel.add_argument("--native-csv", type=Path, required=True)
    barrel.add_argument("--samples", type=Path, required=True,
                        help="CSV: sample_id,source_path,chain_id; paths relative to this CSV or absolute")
    barrel.add_argument("--out", type=Path, required=True)
    for name in ["import-scores", "import-pairs"]:
        command = sub.add_parser(name)
        command.add_argument("--results", type=Path, required=True,
                             help="CSV: sample_id,result_path (relative to this CSV or absolute)")
        command.add_argument("--out", type=Path, required=True)
        if name == "import-scores":
            command.add_argument("--method", choices=["SOLeNNoID", "RAPHAEL", "Foldseek",
                "cooper_beta", "cooper_beta_learned",
                "isitabarrel_structure_map", "pred_tmbb2_single_juchmme",
                "betlas_barrel_geometry"], required=True)
        else:
            command.add_argument("--unit", choices=["residue_contact", "strand_pair"], required=True)
            command.add_argument("--statuses-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output exists; choose a new path")
    if args.command == "betlas-barrel-scores":
        samples = read_csv(args.samples)
        for sample in samples:
            sample["source_path"] = str((args.samples.parent / sample["source_path"]).resolve())
        rows = import_betlas_barrel_rows(read_csv(args.native_csv), samples)
        _write_csv(args.out, rows, ["sample_id", "score", "status", "reason", "prediction", "task", "input_regime"])
        return 0
    if args.command == "import-scores":
        rows = import_domain_scores(_results_manifest(args.results), method=args.method)
        _write_csv(args.out, rows, ["sample_id", "score", "status", "reason", "prediction", "task", "input_regime"])
        return 0
    if args.command == "import-pairs":
        if args.statuses_out.exists() or args.statuses_out.resolve() == args.out.resolve():
            parser.error("Choose a new, separate statuses output path")
        calls, statuses = import_pair_calls(_results_manifest(args.results), unit=args.unit)
        _write_csv(args.out, calls, ["sample_id", "residue_i", "residue_j", "orientation"])
        _write_csv(args.statuses_out, statuses, ["sample_id", "status"])
        return 0
    if args.command == "pairs":
        statuses = read_csv(args.statuses)
        if len({row["sample_id"] for row in statuses}) != len(statuses):
            raise ValueError("Duplicate sample statuses")
        result = pair_metrics(read_csv(args.reference), read_csv(args.calls),
                              {row["sample_id"]: row["status"] for row in statuses},
                              unit=args.unit, input_regime=args.input_regime)
    else:
        score_rows = read_csv(args.scores)
        metadata = {}
        for field in ("task", "input_regime"):
            values = {row[field] for row in score_rows if row.get(field)}
            if len(values) > 1:
                raise ValueError(f"Evaluate one {field} at a time")
            if values:
                metadata[field] = next(iter(values))
        cases = [BinaryCase(row["sample_id"], int(row["target"]), row["split"],
                            row["group_id"], row["pdb_id"]) for row in read_csv(args.labels)]
        scores = [Score(row["sample_id"], float(row["score"]) if row["score"] else None,
                        row["status"], int(row["prediction"]) if row.get("prediction") else None)
                  for row in score_rows]
        if args.command == "calibrate":
            result = calibrate_threshold(cases, scores, method=args.method)
        elif args.calibration:
            result = evaluate_frozen_threshold(cases, scores, json.loads(args.calibration.read_text()),
                                               method=args.method)
        else:
            if any(case.split != "test" for case in cases):
                raise ValueError("Evaluation requires test-only cases")
            result = binary_metrics(cases, scores, threshold=args.fixed_threshold, operator=args.operator)
            result["threshold_scope"] = ("native hard decisions; no score threshold" if args.native_decisions
                else "caller-supplied fixed threshold; must be chosen before test access")
        result["method"] = args.method
        result.update(metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
