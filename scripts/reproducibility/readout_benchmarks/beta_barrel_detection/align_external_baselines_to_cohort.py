from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_BENCHMARK_DIR = Path("data/readouts/beta_barrel_detection/full_mpstruc_767_neg800")


@dataclass(frozen=True)
class BaselineSpec:
    baseline: str
    display_name: str
    file_results_csv: Path
    runtime_seconds: float


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _path_key(value: object) -> str:
    return str(Path(str(value)).expanduser().resolve())


def _metrics(rows: pd.DataFrame) -> dict[str, object]:
    tp = fp = tn = fn = 0
    for row in rows.itertuples(index=False):
        y_true = int(row.y_true)
        pred = bool(row.pred_barrel_any)
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


def _default_specs(benchmark_dir: Path) -> list[BaselineSpec]:
    external = benchmark_dir / "external_baselines"
    return [
        BaselineSpec(
            baseline="isitabarrel_structure_map",
            display_name="IsItABarrel",
            file_results_csv=external
            / "isitabarrel_structure_map/eval_file_results_full_mpstruc_neg800_raw.csv",
            runtime_seconds=459.51,
        ),
        BaselineSpec(
            baseline="pred_tmbb2_single_juchmme",
            display_name="PRED-TMBB2/JUCHMME",
            file_results_csv=external
            / "pred_tmbb2_single_juchmme/eval_file_results_full_mpstruc_neg800_raw.csv",
            runtime_seconds=362.30,
        ),
        BaselineSpec(
            baseline="foldseek_tmalign_structure_search",
            display_name="Foldseek TMalign",
            file_results_csv=external
            / "foldseek_tmalign_structure_search/eval_file_results_full_mpstruc_neg800_raw.csv",
            runtime_seconds=9395.00,
        ),
    ]


def _load_baseline_rows(spec: BaselineSpec) -> dict[str, dict[str, object]]:
    if not spec.file_results_csv.exists():
        raise FileNotFoundError(f"Missing baseline file results: {spec.file_results_csv}")
    df = pd.read_csv(spec.file_results_csv)
    required = {
        "source_file",
        "filename",
        "y_true",
        "decision_score_max",
        "score_adjust_max",
        "pred_barrel_any",
        "any_filtered_out",
        "any_skip",
        "chains_n",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{spec.file_results_csv} missing column(s): {sorted(missing)}")
    df = df.copy()
    df["path_key"] = df["source_file"].map(_path_key)
    if df["path_key"].duplicated().any():
        duplicated = df.loc[df["path_key"].duplicated(), "path_key"].head(10).tolist()
        raise ValueError(f"Duplicate file-level baseline rows for {spec.baseline}: {duplicated}")
    return {str(row.path_key): row._asdict() for row in df.itertuples(index=False)}


def build_aligned_tables(benchmark_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_csv = benchmark_dir / "benchmark_cohort.csv"
    if not cohort_csv.exists():
        raise FileNotFoundError(f"Missing cohort CSV: {cohort_csv}")
    cohort = pd.read_csv(cohort_csv)
    required = {
        "record_id",
        "filename",
        "pdb_id",
        "corrected_split",
        "include_for_metrics",
        "y_true",
        "structure_path",
    }
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"{cohort_csv} missing column(s): {sorted(missing)}")
    cohort = cohort.loc[cohort["include_for_metrics"].map(_boolish)].copy()
    cohort["path_key"] = cohort["structure_path"].map(_path_key)

    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for spec in _default_specs(benchmark_dir):
        by_path = _load_baseline_rows(spec)
        for row in cohort.itertuples(index=False):
            hit = by_path.get(str(row.path_key))
            if hit is None:
                prediction_rows.append(
                    {
                        "baseline": spec.baseline,
                        "display_name": spec.display_name,
                        "record_id": row.record_id,
                        "filename": row.filename,
                        "pdb_id": row.pdb_id,
                        "split": row.corrected_split,
                        "y_true": int(row.y_true),
                        "structure_path": row.structure_path,
                        "prediction_status": "missing_prediction",
                        "pred_barrel_any": False,
                        "decision_score_max": 0.0,
                        "score_adjust_max": 0.0,
                        "chains_n": 0,
                        "any_filtered_out": False,
                        "any_skip": False,
                    }
                )
                continue

            prediction_rows.append(
                {
                    "baseline": spec.baseline,
                    "display_name": spec.display_name,
                    "record_id": row.record_id,
                    "filename": row.filename,
                    "pdb_id": row.pdb_id,
                    "split": row.corrected_split,
                    "y_true": int(row.y_true),
                    "structure_path": row.structure_path,
                    "prediction_status": "predicted",
                    "pred_barrel_any": _boolish(hit["pred_barrel_any"]),
                    "decision_score_max": float(hit["decision_score_max"]),
                    "score_adjust_max": float(hit["score_adjust_max"]),
                    "chains_n": int(hit["chains_n"]),
                    "any_filtered_out": _boolish(hit["any_filtered_out"]),
                    "any_skip": _boolish(hit["any_skip"]),
                }
            )

        baseline_rows = pd.DataFrame(
            [row for row in prediction_rows if row["baseline"] == spec.baseline]
        )
        metric_row = {
            "baseline": spec.baseline,
            "display_name": spec.display_name,
            "scope": "cohort_aligned_missing_as_non_barrel",
            "level": "file",
            "n_total": len(baseline_rows),
            "n_predicted": int((baseline_rows["prediction_status"] == "predicted").sum()),
            "n_missing_prediction": int(
                (baseline_rows["prediction_status"] == "missing_prediction").sum()
            ),
            "missing_positive": int(
                (
                    (baseline_rows["prediction_status"] == "missing_prediction")
                    & (baseline_rows["y_true"] == 1)
                ).sum()
            ),
            "missing_negative": int(
                (
                    (baseline_rows["prediction_status"] == "missing_prediction")
                    & (baseline_rows["y_true"] == 0)
                ).sum()
            ),
            "runtime_seconds": spec.runtime_seconds,
        }
        metric_row.update(_metrics(baseline_rows))
        summary_rows.append(metric_row)

    predictions = pd.DataFrame(prediction_rows)
    summary = pd.DataFrame(summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "aligned_external_baseline_predictions.csv", index=False)
    summary.to_csv(out_dir / "aligned_external_baseline_summary.csv", index=False)
    return predictions, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align external beta-barrel detection baselines to the full cohort."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR / "external_baselines",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    _, summary = build_aligned_tables(args.benchmark_dir.expanduser(), args.out_dir.expanduser())
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
