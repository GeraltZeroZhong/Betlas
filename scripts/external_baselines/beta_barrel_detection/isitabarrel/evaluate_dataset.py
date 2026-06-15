from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_baselines.beta_barrel_detection.isitabarrel.contact_maps import (
    DEFAULT_CA_CUTOFF,
    DEFAULT_LOCAL_EXCLUSION,
    DEFAULT_MIN_RESIDUES,
    GeneratedContactMap,
    GeneratedContactMapSet,
)
from scripts.external_baselines.beta_barrel_detection.isitabarrel.runner import (
    BASELINE_NAME,
    DEFAULT_DECISION_COLUMN,
    IsItABarrelResult,
)
from scripts.external_baselines.beta_barrel_detection.isitabarrel.structure_map import (
    run_structure_map_baseline,
)

RAW_CHAIN_FIELDS = [
    "filename",
    "chain",
    "result",
    "result_stage",
    "decision_score",
    "decision_basis",
    "decision_threshold",
    "score_raw",
    "score_adjust",
    "valid_layers",
    "scored_layers",
    "total_layers",
    "valid_layer_frac",
    "scored_layer_frac",
    "junk_layers",
    "invalid_layers",
    "avg_radius",
    "chain_residues",
    "sheet_residues",
    "informative_slices",
    "reason",
    "all_adjusted_layers",
    "all_layers",
    "y_true",
    "split",
    "is_error",
    "is_filtered_out",
    "is_skip",
    "pred_barrel",
    "use_for_metrics",
    "sample_id",
    "baseline",
    "source_file",
    "pdb_id",
    "cc2",
    "bss",
    "h4",
    "cc2_to_h4",
    "cc2_to_bss",
    "bss_to_cc2_to_bss",
    "decision_column",
    "original_split",
]
FILE_FIELDS = [
    "split",
    "y_true",
    "file_id",
    "filename",
    "source_file",
    "decision_score_max",
    "score_adjust_max",
    "pred_barrel_any",
    "any_filtered_out",
    "any_skip",
    "chains_n",
]
SUMMARY_FIELDS = [
    "scope",
    "level",
    "n_used",
    "TP",
    "FP",
    "TN",
    "FN",
    "recall",
    "precision",
    "f1",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "mcc",
]


@dataclass(frozen=True)
class SplitRun:
    split_name: str
    y_true: int
    generated: GeneratedContactMapSet
    results: list[IsItABarrelResult]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pdb_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[0].upper()


def _metadata_by_sample(generated: GeneratedContactMapSet) -> dict[str, GeneratedContactMap]:
    return {record.sample_id: record for record in generated.records}


def _chain_rows_for_split(run: SplitRun) -> list[dict[str, object]]:
    metadata = _metadata_by_sample(run.generated)
    result_by_sample = {result.sample_id: result for result in run.results}
    missing_samples = sorted(set(metadata) - set(result_by_sample))
    if missing_samples:
        preview = ", ".join(missing_samples[:10])
        suffix = " ..." if len(missing_samples) > 10 else ""
        raise ValueError(f"IsItABarrel did not return results for sample(s): {preview}{suffix}")

    rows: list[dict[str, object]] = []
    for sample_id, record in metadata.items():
        result = result_by_sample[sample_id]
        filename = Path(record.source_path).name
        pred_barrel = result.result == "BARREL"
        rows.append(
            {
                "filename": filename,
                "chain": record.chain_id,
                "result": result.result,
                "result_stage": BASELINE_NAME,
                "decision_score": result.score,
                "decision_basis": f"{result.decision_column}>0",
                "decision_threshold": 0.0,
                "score_raw": result.score,
                "score_adjust": result.score,
                "valid_layers": 0,
                "scored_layers": 0,
                "total_layers": 0,
                "valid_layer_frac": 0.0,
                "scored_layer_frac": 0.0,
                "junk_layers": 0,
                "invalid_layers": 0,
                "avg_radius": 0.0,
                "chain_residues": record.n_residues,
                "sheet_residues": 0,
                "informative_slices": record.n_contacts,
                "reason": f"{result.decision_column}>0",
                "all_adjusted_layers": "",
                "all_layers": "",
                "y_true": run.y_true,
                "split": "true" if run.y_true == 1 else "false",
                "is_error": False,
                "is_filtered_out": False,
                "is_skip": False,
                "pred_barrel": pred_barrel,
                "use_for_metrics": True,
                "sample_id": result.sample_id,
                "baseline": BASELINE_NAME,
                "source_file": record.source_path,
                "pdb_id": _pdb_id_from_filename(filename),
                "cc2": result.cc2,
                "bss": result.bss,
                "h4": result.h4,
                "cc2_to_h4": result.cc2_to_h4,
                "cc2_to_bss": result.cc2_to_bss,
                "bss_to_cc2_to_bss": result.bss_to_cc2_to_bss,
                "decision_column": result.decision_column,
                "original_split": run.split_name,
            }
        )
    return rows


def _file_rows_from_chain_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("use_for_metrics") is False:
            continue
        file_id = str(row.get("source_file") or row["filename"])
        grouped[file_id].append(row)

    file_rows: list[dict[str, object]] = []
    for file_id in sorted(grouped):
        group = grouped[file_id]
        decision_scores = [float(row["decision_score"]) for row in group]
        pred_any = any(_boolish(row["pred_barrel"]) for row in group)
        y_true = int(group[0]["y_true"])
        file_rows.append(
            {
                "split": "true" if y_true == 1 else "false",
                "y_true": y_true,
                "file_id": file_id,
                "filename": str(group[0]["filename"]),
                "source_file": str(group[0].get("source_file", "")),
                "decision_score_max": max(decision_scores) if decision_scores else 0.0,
                "score_adjust_max": max(decision_scores) if decision_scores else 0.0,
                "pred_barrel_any": pred_any,
                "any_filtered_out": any(_boolish(row["is_filtered_out"]) for row in group),
                "any_skip": any(_boolish(row["is_skip"]) for row in group),
                "chains_n": len(group),
            }
        )
    return file_rows


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _metrics(rows: Sequence[dict[str, object]], *, file_level: bool) -> dict[str, object]:
    tp = fp = tn = fn = 0
    for row in rows:
        y_true = int(row["y_true"])
        pred = _boolish(row["pred_barrel_any"] if file_level else row["pred_barrel"])
        if y_true == 1 and pred:
            tp += 1
        elif y_true == 0 and pred:
            fp += 1
        elif y_true == 0 and not pred:
            tn += 1
        elif y_true == 1 and not pred:
            fn += 1

    recall = tp / (tp + fn) if tp + fn else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    balanced_accuracy = (
        (recall + specificity) / 2
        if not math.isnan(recall) and not math.isnan(specificity)
        else math.nan
    )
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else math.nan

    return {
        "n_used": tp + fp + tn + fn,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def _summary_rows(
    chain_rows: Sequence[dict[str, object]],
    file_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for scope, level, rows, file_level in [
        ("raw", "chain", chain_rows, False),
        ("raw", "file", file_rows, True),
    ]:
        row = {"scope": scope, "level": level}
        row.update(_metrics(rows, file_level=file_level))
        output.append(row)
    return output


def _write_summary_md(path: Path, summary_rows: Sequence[dict[str, object]]) -> None:
    headers = SUMMARY_FIELDS
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(
    positive_dir: Path,
    negative_dir: Path,
    output_dir: Path,
    *,
    script_path: Path,
    min_residues: int,
    cutoff: float,
    local_exclusion: int,
    decision_column: str,
    python_executable: str,
    extra_args: Sequence[str] | None,
    timeout: float | None,
    tag: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_runs: list[SplitRun] = []
    for split_name, y_true, input_dir in [
        ("positive", 1, positive_dir),
        ("negative", 0, negative_dir),
    ]:
        split_output = output_dir / split_name
        run = run_structure_map_baseline(
            input_dir,
            split_output,
            script_path=script_path,
            output_path=split_output / "normalized.csv",
            cutoff=cutoff,
            local_exclusion=local_exclusion,
            min_residues=min_residues,
            decision_column=decision_column,
            python_executable=python_executable,
            extra_args=extra_args,
            timeout=timeout,
        )
        split_runs.append(SplitRun(split_name, y_true, run.generated_maps, run.results))

    chain_rows = [row for split_run in split_runs for row in _chain_rows_for_split(split_run)]
    file_rows = _file_rows_from_chain_rows(chain_rows)
    summary_rows = _summary_rows(chain_rows, file_rows)

    _write_csv(output_dir / f"eval_chain_results_{tag}_raw.csv", RAW_CHAIN_FIELDS, chain_rows)
    _write_csv(output_dir / f"eval_file_results_{tag}_raw.csv", FILE_FIELDS, file_rows)
    _write_csv(
        output_dir / f"{BASELINE_NAME}_summary_{tag}.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )
    _write_summary_md(output_dir / f"{BASELINE_NAME}_summary_{tag}.md", summary_rows)

    print(f"Raw chain rows: {len(chain_rows)}")
    print(f"Raw file rows: {len(file_rows)}")
    print(f"Output directory: {output_dir.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IsItABarrel on positive/negative structure directories."
    )
    parser.add_argument("--positive-dir", required=True, type=Path)
    parser.add_argument("--negative-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--script", required=True, type=Path, help="Official isitabarrel.py.")
    parser.add_argument("--tag", default="dataset")
    parser.add_argument("--min-residues", type=int, default=DEFAULT_MIN_RESIDUES)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CA_CUTOFF)
    parser.add_argument("--local-exclusion", type=int, default=DEFAULT_LOCAL_EXCLUSION)
    parser.add_argument("--decision-column", default=DEFAULT_DECISION_COLUMN)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float)
    return parser


def _parse_args_and_passthrough(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw_args:
        return parser.parse_args(raw_args), []
    passthrough_index = raw_args.index("--")
    args = parser.parse_args(raw_args[:passthrough_index])
    return args, raw_args[passthrough_index + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_args = _parse_args_and_passthrough(build_arg_parser(), argv)
    run_dataset(
        args.positive_dir,
        args.negative_dir,
        args.out_dir,
        script_path=args.script,
        min_residues=args.min_residues,
        cutoff=args.cutoff,
        local_exclusion=args.local_exclusion,
        decision_column=args.decision_column,
        python_executable=args.python,
        extra_args=extra_args,
        timeout=args.timeout,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
