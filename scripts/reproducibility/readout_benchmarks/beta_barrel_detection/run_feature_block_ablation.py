from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from betlas.ml.splits import make_grouped_splits

DEFAULT_BENCHMARK_DIR = Path("data/readouts/beta_barrel_detection/full_mpstruc_767_neg800")
DEFAULT_INPUT_DIR = DEFAULT_BENCHMARK_DIR / "betlas_151_layer_radial16_official"
DEFAULT_OUT_DIR = DEFAULT_BENCHMARK_DIR / "feature_block_ablation_catboost"


def _require_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _validate_inputs(*, benchmark_dir: Path, input_dir: Path) -> None:
    _require_file(benchmark_dir / "benchmark_cohort.csv", label="benchmark cohort CSV")
    for filename in [
        "feature_columns_151.csv",
        "betlas_151_chain_features.csv",
        "layer_radial16_feature_manifest.csv",
        "layer_radial16_feature_values.csv",
        "esmc_mean_embeddings_aligned.npz",
    ]:
        _require_file(input_dir / filename, label=f"fixed-cohort input {filename}")


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "n_used": int(len(y_true)),
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
        "score_mean": float(np.mean(y_score)) if len(y_score) else 0.0,
    }


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out.astype(float)


def _make_model(*, iterations: int, seed: int, jobs: int):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "This fixed-cohort companion runner requires CatBoost. Install the "
            "reproducibility environment or `python -m pip install catboost`."
        ) from exc
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=int(iterations),
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=int(seed),
        thread_count=int(jobs),
        verbose=False,
        allow_writing_files=False,
    )


def _positive_class_probability(model: object, x_test: pd.DataFrame, *, fold: int) -> tuple[np.ndarray, str]:
    proba = np.asarray(model.predict_proba(x_test), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if proba.ndim != 2:
        return np.zeros(len(x_test), dtype=float), f"fold {fold}: predict_proba returned non-matrix output"
    if 1 in classes:
        return proba[:, classes.index(1)], ""
    if proba.shape[1] > 1 and not classes:
        return proba[:, 1], f"fold {fold}: classes_ unavailable; used column 1 as compatibility fallback"
    return np.zeros(proba.shape[0], dtype=float), f"fold {fold}: positive class 1 absent from estimator classes_"


def _load_esmc(
    path: Path,
    record_ids: list[str],
    folds: np.ndarray,
    *,
    pca_dim: int,
    seed: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], pd.Series, int, float]:
    start = time.perf_counter()
    loaded = np.load(path, allow_pickle=False)
    ids = [str(value) for value in loaded["record_id"].tolist()]
    embeddings = pd.DataFrame(loaded["embeddings"], index=ids).loc[record_ids]
    if "esmc_available" in loaded.files:
        available = pd.Series(loaded["esmc_available"].astype(float), index=ids).loc[record_ids]
    else:
        available = pd.Series(1.0, index=record_ids)

    n_train_min = len(folds) - int(pd.Series(folds).value_counts().max())
    actual_dim = min(int(pca_dim), n_train_min, embeddings.shape[1])
    train_pcs: dict[int, pd.DataFrame] = {}
    test_pcs: dict[int, pd.DataFrame] = {}
    for fold in sorted(int(value) for value in np.unique(folds)):
        train_index = np.where(folds != fold)[0]
        test_index = np.where(folds == fold)[0]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(embeddings.iloc[train_index])
        test_scaled = scaler.transform(embeddings.iloc[test_index])
        pca = PCA(n_components=actual_dim, random_state=seed)
        columns = [f"esmc_pc_{idx:03d}" for idx in range(actual_dim)]
        train_pcs[fold] = pd.DataFrame(pca.fit_transform(train_scaled), index=train_index, columns=columns)
        test_pcs[fold] = pd.DataFrame(pca.transform(test_scaled), index=test_index, columns=columns)
    return train_pcs, test_pcs, available.reset_index(drop=True), actual_dim, time.perf_counter() - start


def _parts_for_fold(
    *,
    selected_blocks: tuple[str, ...],
    row_index: np.ndarray,
    fold: int,
    is_train: bool,
    x151: pd.DataFrame,
    layer: pd.DataFrame,
    train_pcs: dict[int, pd.DataFrame],
    test_pcs: dict[int, pd.DataFrame],
    esmc_available: pd.Series,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if "betlas_151" in selected_blocks:
        parts.append(x151.iloc[row_index].reset_index(drop=True))
    if "layer_radial16" in selected_blocks:
        parts.append(layer.iloc[row_index].reset_index(drop=True))
    if "esmc" in selected_blocks:
        pcs = train_pcs[fold] if is_train else test_pcs[fold]
        parts.append(pcs.reset_index(drop=True))
        parts.append(pd.DataFrame({"esmc_available": esmc_available.iloc[row_index].to_numpy(float)}))
    if not parts:
        raise ValueError("At least one feature block must be selected.")
    out = pd.concat(parts, axis=1)
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out.astype(float)


def run_ablation(
    *,
    benchmark_dir: Path,
    input_dir: Path,
    out_dir: Path,
    iterations: int,
    seed: int,
    jobs: int,
    pca_dim: int,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_inputs(benchmark_dir=benchmark_dir, input_dir=input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(benchmark_dir / "benchmark_cohort.csv")
    cohort = cohort.loc[cohort["include_for_metrics"].map(_boolish)].reset_index(drop=True)
    y = cohort["y_true"].to_numpy(dtype=int)
    groups = cohort["pdb_id"].astype(str).to_numpy()

    feature_columns = pd.read_csv(input_dir / "feature_columns_151.csv")["feature"].astype(str).tolist()
    feature_rows = pd.read_csv(input_dir / "betlas_151_chain_features.csv")
    feature_rows["record_id"] = feature_rows["record_id"].astype(str)
    feature_rows = feature_rows.set_index("record_id").loc[cohort["record_id"].astype(str)].reset_index()
    x151 = _numeric_frame(feature_rows, feature_columns)

    layer_manifest = pd.read_csv(input_dir / "layer_radial16_feature_manifest.csv")
    layer_columns = layer_manifest["feature"].astype(str).tolist()
    layer_rows = pd.read_csv(input_dir / "layer_radial16_feature_values.csv")
    layer_rows["record_id"] = layer_rows["record_id"].astype(str)
    layer_rows = layer_rows.set_index("record_id").loc[cohort["record_id"].astype(str)].reset_index()
    layer = _numeric_frame(layer_rows, layer_columns)

    splits, split_strategy, effective_splits = make_grouped_splits(
        x151,
        y,
        groups,
        n_splits=int(n_splits),
        random_state=int(seed),
    )
    fold_ids = np.zeros(len(y), dtype=int)
    for fold, (_train_index, test_index) in enumerate(splits):
        fold_ids[test_index] = fold

    train_pcs, test_pcs, esmc_available, esmc_dim, esmc_prepare_runtime = _load_esmc(
        input_dir / "esmc_mean_embeddings_aligned.npz",
        cohort["record_id"].astype(str).tolist(),
        fold_ids,
        pca_dim=int(pca_dim),
        seed=int(seed),
    )

    specs: list[tuple[str, str, tuple[str, ...]]] = [
        ("betlas_151", "Betlas 151", ("betlas_151",)),
        ("layer_radial16", "LayerRadial16", ("layer_radial16",)),
        ("esmc_pca256", "ESM-C", ("esmc",)),
        (
            "betlas_151_layer_radial16",
            "Betlas 151 + LayerRadial16",
            ("betlas_151", "layer_radial16"),
        ),
        ("betlas_151_esmc", "Betlas 151 + ESM-C", ("betlas_151", "esmc")),
        ("layer_radial16_esmc", "LayerRadial16 + ESM-C", ("layer_radial16", "esmc")),
        (
            "betlas_151_layer_radial16_esmc",
            "Betlas 151 + LayerRadial16 + ESM-C",
            ("betlas_151", "layer_radial16", "esmc"),
        ),
    ]

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    per_record_rows: list[pd.DataFrame] = []
    probability_warnings: list[str] = []
    for ablation_id, display_name, selected_blocks in specs:
        start = time.perf_counter()
        predictions = np.zeros(len(y), dtype=int)
        scores = np.zeros(len(y), dtype=float)
        for fold, (train_index, test_index) in enumerate(splits):
            x_train = _parts_for_fold(
                selected_blocks=selected_blocks,
                row_index=train_index,
                fold=fold,
                is_train=True,
                x151=x151,
                layer=layer,
                train_pcs=train_pcs,
                test_pcs=test_pcs,
                esmc_available=esmc_available,
            )
            x_test = _parts_for_fold(
                selected_blocks=selected_blocks,
                row_index=test_index,
                fold=fold,
                is_train=False,
                x151=x151,
                layer=layer,
                train_pcs=train_pcs,
                test_pcs=test_pcs,
                esmc_available=esmc_available,
            )
            model = _make_model(iterations=iterations, seed=seed + fold, jobs=jobs)
            model.fit(x_train, y[train_index])
            fold_scores, warning = _positive_class_probability(model, x_test, fold=fold)
            if warning:
                probability_warnings.append(f"{ablation_id}: {warning}")
            fold_pred = (fold_scores >= 0.5).astype(int)
            predictions[test_index] = fold_pred
            scores[test_index] = fold_scores
            fold_row = _metric_row(y[test_index], fold_pred, fold_scores)
            fold_row.update(
                {
                    "ablation_id": ablation_id,
                    "display_name": display_name,
                    "outer_fold": int(fold),
                    "feature_blocks": "+".join(selected_blocks),
                    "split_strategy": split_strategy,
                    "effective_n_splits": int(effective_splits),
                    "probability_alignment_warning": warning,
                }
            )
            fold_rows.append(fold_row)
        runtime = time.perf_counter() - start
        feature_count = 0
        if "betlas_151" in selected_blocks:
            feature_count += x151.shape[1]
        if "layer_radial16" in selected_blocks:
            feature_count += layer.shape[1]
        if "esmc" in selected_blocks:
            feature_count += esmc_dim + 1
        summary = _metric_row(y, predictions, scores)
        summary.update(
            {
                "ablation_id": ablation_id,
                "display_name": display_name,
                "feature_blocks": "+".join(selected_blocks),
                "feature_count": int(feature_count),
                "uses_betlas_151": "betlas_151" in selected_blocks,
                "uses_layer_radial16": "layer_radial16" in selected_blocks,
                "uses_esmc": "esmc" in selected_blocks,
                "esmc_pca_dim": int(esmc_dim) if "esmc" in selected_blocks else 0,
                "runtime_seconds": round(runtime, 3),
                "shared_esmc_prepare_runtime_seconds": round(esmc_prepare_runtime, 3),
            }
        )
        summary_rows.append(summary)
        per = cohort.copy()
        per["ablation_id"] = ablation_id
        per["display_name"] = display_name
        per["outer_fold"] = fold_ids
        per["pred_barrel"] = predictions.astype(bool)
        per["prob_barrel"] = scores
        per["correct"] = predictions == y
        per_record_rows.append(per)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy", "f1", "mcc"], ascending=[False, False, False]
    )
    fold_df = pd.DataFrame(fold_rows)
    per_record_df = pd.concat(per_record_rows, ignore_index=True)
    summary_df.to_csv(out_dir / "feature_block_ablation_summary.csv", index=False)
    fold_df.to_csv(out_dir / "feature_block_ablation_fold_metrics.csv", index=False)
    per_record_df.to_csv(out_dir / "feature_block_ablation_per_record_predictions.csv", index=False)
    metadata = {
        "benchmark_dir": str(benchmark_dir),
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "n_records": int(len(cohort)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "iterations": int(iterations),
        "seed": int(seed),
        "jobs": int(jobs),
        "n_splits": int(n_splits),
        "effective_n_splits": int(effective_splits),
        "split_strategy": split_strategy,
        "probability_alignment_warnings": probability_warnings,
        "model_dependency_status": {"catboost": "required"},
        "esmc_pca_dim": int(esmc_dim),
        "shared_esmc_prepare_runtime_seconds": round(esmc_prepare_runtime, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "feature_block_ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_df, fold_df, per_record_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CatBoost feature-block ablations.",
        epilog=(
            "Required inputs:\n"
            "  --benchmark-dir/benchmark_cohort.csv\n"
            "  --input-dir/feature_columns_151.csv\n"
            "  --input-dir/betlas_151_chain_features.csv\n"
            "  --input-dir/layer_radial16_feature_manifest.csv\n"
            "  --input-dir/layer_radial16_feature_values.csv\n"
            "  --input-dir/esmc_mean_embeddings_aligned.npz\n\n"
            "Outputs: feature_block_ablation_summary.csv, fold metrics, per-record predictions, metadata JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--splits", type=int, default=5)
    return parser


def _main() -> int:
    args = build_arg_parser().parse_args()
    summary, _folds, _per_record = run_ablation(
        benchmark_dir=args.benchmark_dir.expanduser(),
        input_dir=args.input_dir.expanduser(),
        out_dir=args.out_dir.expanduser(),
        iterations=args.iterations,
        seed=args.seed,
        jobs=args.jobs,
        pca_dim=args.pca_dim,
        n_splits=args.splits,
    )
    print(summary.to_string(index=False))
    return 0


def main() -> int:
    try:
        return _main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
