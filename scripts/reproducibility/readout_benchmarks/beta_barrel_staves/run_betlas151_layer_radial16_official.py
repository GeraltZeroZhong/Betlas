#!/usr/bin/env python
"""Build official Betlas/LayerRadial16 stave-count readouts.

The formal display scope is:

1. Betlas 151
2. Betlas 151 + LayerRadial16
3. Betlas 151 + LayerRadial16 + ESM-C

The cohort denominator and outer-fold assignment are inherited from the aligned
Betlas 151 readout. LayerRadial16 is the named layer/radial stave-count
auxiliary feature block.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betlas.assets import asset_file_report, resolve_asset_path  # noqa: E402
from betlas.provenance import (  # noqa: E402
    file_state,
    git_state,
    runtime_state,
    source_tree_state,
    write_json,
)

DEFAULT_ALIGNED_DIR = (
    REPO_ROOT
    / "data/readouts/beta_barrel_staves/benchmark_508/mechanics/"
    "betlas_151_layer_radial16_inputs"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "data/readouts/beta_barrel_staves/benchmark_508/mechanics/"
    "betlas_151_layer_radial16_official"
)
DEFAULT_LAYER_VALUES_CSV = DEFAULT_ALIGNED_DIR / "layer_radial16_feature_values.csv"
DEFAULT_ASSET_ID = "betlas-beta-barrel-staves-official-v1"
COMPAT_REFERENCE_COUNT_COLUMN = "".join(("go", "ld"))
ALIGNED_ASSET_FILES = (
    "per_record_aligned_wide.csv",
    "feature_columns_151.csv",
    "betlas_151_chain_features.csv",
    "esmc_mean_embeddings_aligned.npz",
    "esmc_embedding_coverage.csv",
)


def _unique_in_order(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


LAYER_RADIAL_RAW: list[str] = [
    "layer_count_mode",
    "supporting_layers",
    "usable_layers",
    "total_layers",
    "layer_support_fraction",
    "usable_layer_fraction",
    "geometric_pass_layers",
    "geometric_pass_fraction",
    "trimmed_layers",
    "radial_outlier_layers",
    "radial_outlier_points",
    "layer_count_q75",
    "layer_count_q90",
    "layer_count_max",
    "layer_count_max_plateau",
    "layer_count_largest_drop",
]


@dataclass
class Evaluation:
    row: dict[str, Any]
    folds: pd.DataFrame
    per_record: pd.DataFrame
    estimators: dict[int, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Required local inputs without --asset-id/--download-assets:\n"
            "  --aligned-dir/per_record_aligned_wide.csv\n"
            "  --aligned-dir/feature_columns_151.csv\n"
            "  --aligned-dir/betlas_151_chain_features.csv\n"
            "  --aligned-dir/esmc_mean_embeddings_aligned.npz\n"
            "  --aligned-dir/esmc_embedding_coverage.csv\n"
            "  --layer-values-csv layer_radial16_feature_values.csv\n\n"
            "Outputs: official_summary.csv, official_fold_metrics.csv, official_per_record_predictions.csv, "
            "official_per_record_wide.csv, metadata.json, summary.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help="Optional Betlas asset id used to resolve aligned fixed-cohort inputs from the local asset cache.",
    )
    parser.add_argument(
        "--asset-cache-dir",
        type=Path,
        default=None,
        help="Optional Betlas asset cache directory. Defaults to BETLAS_ASSET_DIR.",
    )
    parser.add_argument(
        "--download-assets",
        action="store_true",
        help=(
            "Download and verify selected asset files before running. Requires a released payload "
            "or a BETLAS_ASSET_BASE_URL-compatible mirror."
        ),
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=DEFAULT_ALIGNED_DIR,
        help=(
            "Directory containing fixed-cohort aligned input CSV/NPZ files. "
            "Clean clones should prefer --download-assets or --asset-id after caching."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--layer-values-csv",
        type=Path,
        default=DEFAULT_LAYER_VALUES_CSV,
        help=(
            "LayerRadial16 feature-values CSV. Clean clones should prefer --download-assets "
            "or --asset-id after caching."
        ),
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--skip-permutation", action="store_true")
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _require_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}. This companion runner needs the fixed-cohort "
            "asset payload; use --download-assets, use --asset-id after caching, "
            "or pass explicit local input paths."
        )
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _validate_inputs(*, aligned_dir: Path, layer_values_csv: Path) -> None:
    for filename in [
        "per_record_aligned_wide.csv",
        "feature_columns_151.csv",
        "betlas_151_chain_features.csv",
        "esmc_mean_embeddings_aligned.npz",
        "esmc_embedding_coverage.csv",
    ]:
        _require_file(aligned_dir / filename, label=f"fixed-cohort input {filename}")
    _require_file(layer_values_csv, label="LayerRadial16 feature values CSV")


def layer_feature_name(raw: str) -> str:
    return f"layer_radial16__{raw}"


def metric_values(reference: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(pred, dtype=int) - np.asarray(reference, dtype=int)
    abs_errors = np.abs(errors)
    n = int(len(reference))
    exact = int(np.sum(errors == 0))
    within_2 = int(np.sum(abs_errors <= 2))
    return {
        "n": n,
        "valid_n": n,
        "rejected": 0,
        "exact": exact,
        "exact_rate": exact / n if n else 0.0,
        "exact_rate_valid": exact / n if n else 0.0,
        "within_2": within_2,
        "within_2_rate": within_2 / n if n else 0.0,
        "within_2_rate_valid": within_2 / n if n else 0.0,
        "mae": float(np.mean(abs_errors)) if n else 0.0,
        "rmse": math.sqrt(float(np.mean(errors.astype(float) ** 2))) if n else 0.0,
        "bias": float(np.mean(errors)) if n else 0.0,
        "over": int(np.sum(errors > 0)),
        "under": int(np.sum(errors < 0)),
    }


def load_rows(aligned_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(aligned_dir / "per_record_aligned_wide.csv")
    rows["record_id"] = rows["record_id"].astype(str)
    rows["pdb_id"] = rows["pdb_id"].astype(str).str.strip().str.upper()
    rows["auth_chain_id"] = rows["auth_chain_id"].astype(str)
    if "reference_count" not in rows.columns:
        if COMPAT_REFERENCE_COUNT_COLUMN not in rows.columns:
            raise SystemExit(
                "per_record_aligned_wide.csv must contain `reference_count` "
                "(an older label-count alias is accepted only as an input alias)."
            )
        rows["reference_count"] = rows[COMPAT_REFERENCE_COUNT_COLUMN]
    rows["reference_count"] = rows["reference_count"].astype(int)
    rows["outer_fold"] = rows["outer_fold"].astype(int)
    return rows[["record_id", "pdb_id", "auth_chain_id", "reference_count", "outer_fold"]].copy()


def _count_dict(values: pd.Series | np.ndarray) -> dict[str, int]:
    series = pd.Series(values)
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _check_unique_record_ids(values: pd.Series | list[str], *, frame_name: str) -> list[str]:
    ids = pd.Series(values).astype(str)
    if ids.duplicated().any():
        dupes = ids.loc[ids.duplicated()].head(8).tolist()
        raise ValueError(f"{frame_name} contains duplicate record_id values: {dupes}")
    return ids.tolist()


def _align_record_frame(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    path: Path,
    frame_name: str,
) -> pd.DataFrame:
    if "record_id" not in frame.columns:
        raise ValueError(f"{frame_name} lacks `record_id`: {path}")
    target_ids = _check_unique_record_ids(rows["record_id"], frame_name="fixed-cohort aligned rows")
    source = frame.copy()
    source["record_id"] = _check_unique_record_ids(source["record_id"], frame_name=frame_name)
    missing = sorted(set(target_ids) - set(source["record_id"]))
    extra = sorted(set(source["record_id"]) - set(target_ids))
    if missing or extra:
        raise ValueError(
            f"{frame_name} record_id set does not match aligned rows "
            f"(missing={missing[:8]}, extra={extra[:8]}): {path}"
        )
    return source.set_index("record_id").loc[target_ids].reset_index()


def build_fold_preflight(rows: pd.DataFrame) -> dict[str, Any]:
    folds = rows["outer_fold"].astype(int)
    counts = rows["reference_count"].astype(int)
    global_count_values = sorted(int(value) for value in counts.unique())
    pdb_keys = rows["pdb_id"].astype(str).str.strip().str.upper()
    leakage_rows: list[dict[str, Any]] = []
    for pdb_id, part in rows.groupby(pdb_keys, dropna=False):
        fold_values = sorted(int(value) for value in part["outer_fold"].astype(int).unique())
        if len(fold_values) > 1:
            leakage_rows.append(
                {
                    "pdb_id": str(pdb_id),
                    "input_pdb_id_values": sorted(part["pdb_id"].astype(str).unique().tolist()),
                    "outer_folds": fold_values,
                    "record_ids": part["record_id"].astype(str).head(8).tolist(),
                }
            )
    duplicate_record_ids = sorted(
        str(record_id)
        for record_id, n in rows["record_id"].astype(str).value_counts().items()
        if int(n) > 1
    )
    fold_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []
    for fold in sorted(int(value) for value in folds.unique()):
        test_mask = folds == fold
        train_mask = ~test_mask
        train_counts = sorted(int(value) for value in counts.loc[train_mask].unique())
        test_counts = sorted(int(value) for value in counts.loc[test_mask].unique())
        missing_train = sorted(set(global_count_values) - set(train_counts))
        missing_test = sorted(set(global_count_values) - set(test_counts))
        if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
            failures.append(f"fold {fold}: empty train/test partition")
        if missing_train:
            failures.append(f"fold {fold}: train partition lacks reference_count classes {missing_train}")
        if missing_test:
            failures.append(f"fold {fold}: test partition lacks reference_count classes {missing_test}")
        fold_rows.append(
            {
                "outer_fold": fold,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_reference_count_classes": train_counts,
                "test_reference_count_classes": test_counts,
                "missing_train_reference_count_classes": missing_train,
                "missing_test_reference_count_classes": missing_test,
                "train_reference_count_histogram": _count_dict(counts.loc[train_mask]),
                "test_reference_count_histogram": _count_dict(counts.loc[test_mask]),
            }
        )
    if leakage_rows:
        failures.append(
            "pdb_id appears in more than one outer_fold: "
            + "; ".join(f"{row['pdb_id']}->{row['outer_folds']}" for row in leakage_rows[:8])
        )
    if duplicate_record_ids:
        failures.append(f"duplicate record_id values: {duplicate_record_ids[:8]}")
    status = "failed" if failures else "ok"
    return {
        "schema": "betlas.fixed-cohort-fold-preflight.v1",
        "status": status,
        "failure_stage": "fold_contract" if failures else "",
        "error": "; ".join(failures[:8]),
        "warnings": warnings,
        "n_records": int(len(rows)),
        "outer_folds": sorted(int(value) for value in folds.unique()),
        "reference_count_classes": global_count_values,
        "reference_count_histogram": _count_dict(counts),
        "pdb_fold_leakage": leakage_rows,
        "duplicate_record_ids": duplicate_record_ids,
        "folds": fold_rows,
    }


def write_fold_preflight(rows: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    preflight = build_fold_preflight(rows)
    write_json(out_dir / "fold_preflight.json", preflight)
    if preflight["status"] != "ok":
        raise ValueError(f"fixed-cohort outer_fold preflight failed: {preflight['error']}")
    return preflight


def load_151(aligned_dir: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = pd.read_csv(aligned_dir / "feature_columns_151.csv")["feature"].astype(str).tolist()
    feature_path = aligned_dir / "betlas_151_chain_features.csv"
    features = _align_record_frame(
        pd.read_csv(feature_path),
        rows,
        path=feature_path,
        frame_name="Betlas 151 chain feature table",
    )
    x151 = features.loc[:, columns].copy()
    for column in columns:
        x151[column] = pd.to_numeric(x151[column], errors="coerce").fillna(0.0)
    return x151.reset_index(drop=True).astype(float), columns


def load_esmc_pca_by_fold(
    aligned_dir: Path,
    rows: pd.DataFrame,
    *,
    pca_dim: int,
    seed: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], pd.Series, int]:
    loaded = np.load(aligned_dir / "esmc_mean_embeddings_aligned.npz", allow_pickle=False)
    record_ids = _check_unique_record_ids(
        loaded["record_id"].astype(str).tolist(),
        frame_name="ESM-C embedding cache",
    )
    target_ids = _check_unique_record_ids(rows["record_id"], frame_name="fixed-cohort aligned rows")
    missing_embeddings = sorted(set(target_ids) - set(record_ids))
    if missing_embeddings:
        raise ValueError(f"ESM-C embedding cache is missing record_id values: {missing_embeddings[:8]}")
    embeddings = pd.DataFrame(loaded["embeddings"], index=record_ids)
    embeddings = embeddings.loc[target_ids]
    coverage = pd.read_csv(aligned_dir / "esmc_embedding_coverage.csv")
    coverage["record_id"] = _check_unique_record_ids(
        coverage["record_id"],
        frame_name="ESM-C embedding coverage table",
    )
    missing_coverage = sorted(set(target_ids) - set(coverage["record_id"]))
    if missing_coverage:
        raise ValueError(f"ESM-C embedding coverage is missing record_id values: {missing_coverage[:8]}")
    esmc_available = coverage.set_index("record_id").loc[target_ids, "esmc_available"].astype(float)
    folds = rows["outer_fold"].astype(int).to_numpy()
    n_train_min = len(rows) - int(pd.Series(folds).value_counts().max())
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
        train_pc = pca.fit_transform(train_scaled)
        test_pc = pca.transform(test_scaled)
        columns = [f"esmc_pc_{idx:03d}" for idx in range(actual_dim)]
        train_pcs[fold] = pd.DataFrame(train_pc, index=train_index, columns=columns)
        test_pcs[fold] = pd.DataFrame(test_pc, index=test_index, columns=columns)
    return train_pcs, test_pcs, esmc_available.reset_index(drop=True), actual_dim


def load_layer_radial16(
    *,
    layer_values_csv: Path,
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = _align_record_frame(
        pd.read_csv(layer_values_csv),
        rows,
        path=layer_values_csv,
        frame_name="LayerRadial16 feature table",
    )

    layer_columns = [layer_feature_name(raw) for raw in LAYER_RADIAL_RAW]
    missing_features = [feature for feature in layer_columns if feature not in source.columns]
    if missing_features:
        raise SystemExit(f"LayerRadial16 source features missing: {missing_features}")

    layer = source.loc[:, layer_columns].copy()
    for column in layer.columns:
        layer[column] = pd.to_numeric(layer[column], errors="coerce").fillna(0.0)
    layer = layer.reset_index(drop=True).astype(float)

    manifest = pd.DataFrame(
        [
            {
                "feature": layer_feature_name(raw),
                "source_feature": raw,
                "feature_block": "LayerRadial16",
                "feature_family": "layer_radial",
                "official_block": "Betlas LayerRadial16 auxiliary stave-count block",
                "source_status": "fixed_cohort_auxiliary_feature_block",
            }
            for raw in LAYER_RADIAL_RAW
        ]
    )
    return layer, manifest


def dependency_status() -> dict[str, str]:
    return {
        "catboost": "available" if importlib.util.find_spec("catboost") is not None else "missing",
        "numpy": "available" if importlib.util.find_spec("numpy") is not None else "missing",
        "pandas": "available" if importlib.util.find_spec("pandas") is not None else "missing",
    }


def require_catboost(status: dict[str, str]) -> None:
    if status.get("catboost") == "available":
        return
    raise RuntimeError(
        "This fixed-cohort companion runner requires CatBoost. Install the "
        "reproducibility environment or `python -m pip install catboost`."
    )


def make_model(*, iterations: int, seed: int, jobs: int) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This fixed-cohort companion runner requires CatBoost. Install the "
            "reproducibility environment or `python -m pip install catboost`."
        ) from exc

    return CatBoostClassifier(
        loss_function="MultiClass",
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


def build_parts(
    *,
    row_index: np.ndarray,
    fold: int,
    is_train: bool,
    x151: pd.DataFrame,
    layer: pd.DataFrame,
    layer_columns: list[str],
    include_esmc: bool,
    train_pcs: dict[int, pd.DataFrame],
    test_pcs: dict[int, pd.DataFrame],
    esmc_available: pd.Series,
) -> pd.DataFrame:
    parts = [x151.iloc[row_index].reset_index(drop=True)]
    if layer_columns:
        parts.append(layer.iloc[row_index][layer_columns].reset_index(drop=True))
    if include_esmc:
        pcs = train_pcs[fold] if is_train else test_pcs[fold]
        parts.append(pcs.reset_index(drop=True))
        parts.append(
            pd.DataFrame({"esmc_available": esmc_available.iloc[row_index].to_numpy(dtype=float)})
        )
    out = pd.concat(parts, axis=1)
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out.astype(float)


def evaluate(
    *,
    experiment: str,
    display_label: str,
    x151: pd.DataFrame,
    layer: pd.DataFrame,
    layer_columns: list[str],
    include_esmc: bool,
    rows: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    train_pcs: dict[int, pd.DataFrame],
    test_pcs: dict[int, pd.DataFrame],
    esmc_available: pd.Series,
    pca_dim: int,
    iterations: int,
    seed: int,
    jobs: int,
) -> Evaluation:
    predictions = np.zeros(len(y), dtype=int)
    probabilities = np.zeros(len(y), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    estimators: dict[int, Any] = {}
    start = time.perf_counter()
    for fold in sorted(int(value) for value in np.unique(folds)):
        train_index = np.where(folds != fold)[0]
        test_index = np.where(folds == fold)[0]
        x_train = build_parts(
            row_index=train_index,
            fold=fold,
            is_train=True,
            x151=x151,
            layer=layer,
            layer_columns=layer_columns,
            include_esmc=include_esmc,
            train_pcs=train_pcs,
            test_pcs=test_pcs,
            esmc_available=esmc_available,
        )
        x_test = build_parts(
            row_index=test_index,
            fold=fold,
            is_train=False,
            x151=x151,
            layer=layer,
            layer_columns=layer_columns,
            include_esmc=include_esmc,
            train_pcs=train_pcs,
            test_pcs=test_pcs,
            esmc_available=esmc_available,
        )
        model = make_model(iterations=iterations, seed=seed, jobs=jobs)
        model.fit(x_train, y[train_index])
        fold_pred = np.asarray(model.predict(x_test)).ravel().astype(int)
        predictions[test_index] = fold_pred
        probabilities[test_index] = np.asarray(model.predict_proba(x_test)).max(axis=1)
        fold_row = metric_values(y[test_index], fold_pred)
        fold_row.update({"experiment": experiment, "display_label": display_label, "outer_fold": fold})
        fold_rows.append(fold_row)
        estimators[fold] = model

    metrics = metric_values(y, predictions)
    feature_count = int(x151.shape[1] + len(layer_columns) + (pca_dim + 1 if include_esmc else 0))
    metrics.update(
        {
            "experiment": experiment,
            "display_label": display_label,
            "iterations": int(iterations),
            "feature_count": feature_count,
            "betlas_feature_count": int(x151.shape[1]),
            "layer_radial16_feature_count": int(len(layer_columns)),
            "include_esmc": bool(include_esmc),
            "esmc_pca_dim": int(pca_dim) if include_esmc else 0,
            "runtime_seconds": round(time.perf_counter() - start, 3),
            "layer_radial16_features": ";".join(layer_columns),
        }
    )
    per_record = rows.copy().reset_index(drop=True)
    per_record["experiment"] = experiment
    per_record["display_label"] = display_label
    per_record["model_pred"] = predictions
    per_record["model_err"] = predictions - y
    per_record["abs_err"] = np.abs(predictions - y)
    per_record["probability_max"] = probabilities
    per_record["probability_source"] = "catboost_predict_proba"
    per_record["probability_calibration_status"] = "model_reported_uncalibrated"
    per_record["include_esmc"] = bool(include_esmc)
    per_record["layer_radial16_feature_count"] = int(len(layer_columns))
    return Evaluation(metrics, pd.DataFrame(fold_rows), per_record, estimators)


def permutation_importance_layer_radial16(
    *,
    baseline: Evaluation,
    layer_features: list[str],
    x151: pd.DataFrame,
    layer: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    train_pcs: dict[int, pd.DataFrame],
    test_pcs: dict[int, pd.DataFrame],
    esmc_available: pd.Series,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if repeats <= 0:
        return pd.DataFrame()
    baseline_metrics = metric_values(y, baseline.per_record["model_pred"].to_numpy(dtype=int))
    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, Any]] = []
    for feature in layer_features:
        exact_values: list[float] = []
        mae_values: list[float] = []
        rmse_values: list[float] = []
        start = time.perf_counter()
        for _repeat in range(repeats):
            permuted_pred = baseline.per_record["model_pred"].to_numpy(dtype=int, copy=True)
            for fold, model in baseline.estimators.items():
                test_index = np.where(folds == fold)[0]
                x_test = build_parts(
                    row_index=test_index,
                    fold=fold,
                    is_train=False,
                    x151=x151,
                    layer=layer,
                    layer_columns=layer_features,
                    include_esmc=True,
                    train_pcs=train_pcs,
                    test_pcs=test_pcs,
                    esmc_available=esmc_available,
                )
                shuffled = x_test[feature].to_numpy(copy=True)
                rng.shuffle(shuffled)
                x_test[feature] = shuffled
                permuted_pred[test_index] = np.asarray(model.predict(x_test)).ravel().astype(int)
            metrics = metric_values(y, permuted_pred)
            exact_values.append(float(metrics["exact"]))
            mae_values.append(float(metrics["mae"]))
            rmse_values.append(float(metrics["rmse"]))
        out_rows.append(
            {
                "feature": feature,
                "source_feature": feature.removeprefix("layer_radial16__"),
                "feature_block": "LayerRadial16",
                "feature_family": "layer_radial",
                "repeats": int(repeats),
                "permuted_exact_mean": float(np.mean(exact_values)),
                "permuted_exact_std": float(np.std(exact_values)),
                "exact_drop_mean": float(baseline_metrics["exact_rate"] - np.mean(exact_values) / len(y)),
                "exact_drop_std": float(np.std(np.asarray(exact_values) / len(y))),
                "mae_increase_mean": float(np.mean(mae_values) - baseline_metrics["mae"]),
                "mae_increase_std": float(np.std(mae_values)),
                "rmse_increase_mean": float(np.mean(rmse_values) - baseline_metrics["rmse"]),
                "rmse_increase_std": float(np.std(rmse_values)),
                "runtime_seconds": round(time.perf_counter() - start, 3),
            }
        )
    out = pd.DataFrame(out_rows).sort_values(
        ["exact_drop_mean", "mae_increase_mean", "feature"],
        ascending=[False, False, True],
    )
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def write_summary(
    path: Path,
    *,
    official: pd.DataFrame,
    permutation: pd.DataFrame,
    pca_dim: int,
    n_records: int,
) -> None:
    lines = [
        "# Betlas 151 + LayerRadial16 Official Stave Readout",
        "",
        "- Official display scope: Betlas 151; Betlas 151 + LayerRadial16; Betlas 151 + LayerRadial16 + ESM-C.",
        f"- Classifier: CatBoost multiclass with the fixed {n_records}-row mechanics denominator and fixed outer folds.",
        f"- ESM-C branch: cached mean embeddings, fold-local PCA{pca_dim}, and an embedding-availability indicator.",
        f"- LayerRadial16 input features are loaded from the active {n_records}-row LayerRadial16 input cache.",
        "- `probability_max` is the estimator-reported, uncalibrated probability assigned to the predicted stave-count class.",
        "",
        "## Official Models",
        "",
        "| model | features | exact | within +/-2 | MAE | RMSE | bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in official.itertuples(index=False):
        lines.append(
            f"| `{row.display_label}` | {int(row.feature_count)} | "
            f"{int(row.exact)}/{n_records} ({float(row.exact_rate):.3f}) | "
            f"{int(row.within_2)}/{n_records} ({float(row.within_2_rate):.3f}) | "
            f"{float(row.mae):.3f} | {float(row.rmse):.3f} | {float(row.bias):+.3f} |"
        )
    if not permutation.empty:
        lines.extend(["", "## LayerRadial16 Permutation Contributions", ""])
        lines.append("| rank | feature | exact-rate drop | MAE increase |")
        lines.append("| ---: | --- | ---: | ---: |")
        for row in permutation.head(16).itertuples(index=False):
            lines.append(
                f"| {int(row.rank)} | `{row.source_feature}` | "
                f"{float(row.exact_drop_mean):.4f} | {float(row.mae_increase_mean):.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "model_dependency_status": dependency_status(),
        "required_model_dependency": "catboost",
        "status": "ok",
    }
    if preflight["model_dependency_status"].get("catboost") != "available":
        preflight["status"] = "failed"
        (out_dir / "dependency_preflight.json").write_text(
            json.dumps(preflight, indent=2) + "\n",
            encoding="utf-8",
        )
        require_catboost(preflight["model_dependency_status"])
    (out_dir / "dependency_preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )

    asset_id = args.asset_id or (DEFAULT_ASSET_ID if args.download_assets else None)
    asset_cache_dir = args.asset_cache_dir.expanduser().resolve() if args.asset_cache_dir else None
    aligned_dir = args.aligned_dir.expanduser().resolve()
    layer_values_csv = args.layer_values_csv.expanduser().resolve()
    asset_files_used: list[str] = []
    asset_manifest_verification: dict[str, object] = {}
    if asset_id:
        if args.aligned_dir == DEFAULT_ALIGNED_DIR:
            asset_files_used.extend(ALIGNED_ASSET_FILES)
            aligned_dir = resolve_asset_path(
                asset_id,
                cache_dir=asset_cache_dir,
                download=bool(args.download_assets),
            ).resolve()
        if args.layer_values_csv == DEFAULT_LAYER_VALUES_CSV:
            asset_files_used.append("layer_radial16_feature_values.csv")
            layer_values_csv = resolve_asset_path(
                asset_id,
                "layer_radial16_feature_values.csv",
                cache_dir=asset_cache_dir,
                download=bool(args.download_assets),
            ).resolve()
        selected_asset_files = _unique_in_order(asset_files_used)
        if selected_asset_files:
            asset_manifest_verification = asset_file_report(
                asset_id,
                cache_dir=asset_cache_dir,
                filenames=selected_asset_files,
                strict=True,
            )
    _validate_inputs(aligned_dir=aligned_dir, layer_values_csv=layer_values_csv)

    rows = load_rows(aligned_dir)
    fold_preflight = write_fold_preflight(rows, out_dir)
    x151, columns151 = load_151(aligned_dir, rows)
    layer, manifest = load_layer_radial16(
        layer_values_csv=layer_values_csv,
        rows=rows,
    )
    train_pcs, test_pcs, esmc_available, pca_dim = load_esmc_pca_by_fold(
        aligned_dir,
        rows,
        pca_dim=int(args.pca_dim),
        seed=int(args.seed),
    )
    y = rows["reference_count"].to_numpy(dtype=int)
    folds = rows["outer_fold"].to_numpy(dtype=int)
    layer_features = [layer_feature_name(raw) for raw in LAYER_RADIAL_RAW]
    if len(layer_features) != 16:
        raise SystemExit(f"Expected LayerRadial16 to contain 16 features, found {len(layer_features)}")

    manifest.to_csv(out_dir / "layer_radial16_feature_manifest.csv", index=False)
    pd.concat(
        [rows[["record_id", "pdb_id", "auth_chain_id", "reference_count", "outer_fold"]], layer],
        axis=1,
    ).to_csv(out_dir / "layer_radial16_feature_values.csv", index=False)

    common = {
        "x151": x151,
        "layer": layer,
        "rows": rows,
        "y": y,
        "folds": folds,
        "train_pcs": train_pcs,
        "test_pcs": test_pcs,
        "esmc_available": esmc_available,
        "pca_dim": int(pca_dim),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "jobs": int(args.jobs),
    }
    official_specs = [
        ("betlas_151_catboost", "Betlas 151", [], False),
        ("betlas_151_layer_radial16_catboost", "Betlas 151 + LayerRadial16", layer_features, False),
        (
            f"betlas_151_layer_radial16_esmc_pca{pca_dim}_catboost",
            "Betlas 151 + LayerRadial16 + ESM-C",
            layer_features,
            True,
        ),
    ]

    official_evals: list[Evaluation] = []
    n_records = int(len(rows))
    print("Running official Betlas/LayerRadial16 CatBoost readouts")
    for experiment, display_label, layer_columns, include_esmc in official_specs:
        result = evaluate(
            experiment=experiment,
            display_label=display_label,
            layer_columns=layer_columns,
            include_esmc=include_esmc,
            **common,
        )
        official_evals.append(result)
        print(
            f"  {experiment}: exact={result.row['exact']}/{n_records} "
            f"mae={result.row['mae']:.3f} rmse={result.row['rmse']:.3f}"
        )

    official = pd.DataFrame([evaluation.row for evaluation in official_evals])
    official.to_csv(out_dir / "official_summary.csv", index=False)
    pd.concat([evaluation.folds for evaluation in official_evals], ignore_index=True).to_csv(
        out_dir / "official_fold_metrics.csv",
        index=False,
    )
    pd.concat([evaluation.per_record for evaluation in official_evals], ignore_index=True).to_csv(
        out_dir / "official_per_record_predictions.csv",
        index=False,
    )
    wide = rows.copy().reset_index(drop=True)
    for evaluation in official_evals:
        key = evaluation.row["experiment"]
        wide[f"{key}__pred"] = evaluation.per_record["model_pred"].to_numpy(dtype=int)
        wide[f"{key}__err"] = evaluation.per_record["model_err"].to_numpy(dtype=int)
        wide[f"{key}__probability_max"] = evaluation.per_record["probability_max"].to_numpy(float)
        wide[f"{key}__probability_calibration_status"] = evaluation.per_record[
            "probability_calibration_status"
        ].to_numpy(dtype=object)
    wide.to_csv(out_dir / "official_per_record_wide.csv", index=False)

    permutation = pd.DataFrame()
    if not args.skip_permutation:
        print("Running LayerRadial16 permutation importance for the ESM-C-augmented readout")
        permutation = permutation_importance_layer_radial16(
            baseline=official_evals[-1],
            layer_features=layer_features,
            x151=x151,
            layer=layer,
            y=y,
            folds=folds,
            train_pcs=train_pcs,
            test_pcs=test_pcs,
            esmc_available=esmc_available,
            repeats=int(args.permutation_repeats),
            seed=int(args.seed) + 2000,
        )
        permutation.to_csv(out_dir / "layer_radial16_permutation_importance.csv", index=False)

    metadata = {
        "asset_id": asset_id or "",
        "asset_cache_dir": display_path(asset_cache_dir) if asset_cache_dir is not None else "",
        "asset_manifest_verification": asset_manifest_verification,
        "aligned_dir": display_path(aligned_dir),
        "out_dir": display_path(out_dir),
        "n_records": int(len(rows)),
        "n_features_betlas": int(len(columns151)),
        "n_features_layer_radial16": int(len(layer_features)),
        "formal_scope": [spec[1] for spec in official_specs],
        "model_family": "CatBoostClassifier MultiClass",
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "pca_dim": int(pca_dim),
        "permutation_repeats": int(args.permutation_repeats),
        "esmc_cache_status": "mean embeddings reused from aligned Betlas cache",
        "layer_feature_source": display_path(layer_values_csv),
        "probability_max_definition": "Estimator-reported, uncalibrated probability assigned to the predicted stave-count class.",
        "probability_calibration_status": "model_reported_uncalibrated",
        "dependency_preflight": "dependency_preflight.json",
        "fold_preflight": "fold_preflight.json",
        "fold_preflight_status": fold_preflight["status"],
        "model_dependency_status": preflight["model_dependency_status"],
        "inputs": {
            filename: file_state(aligned_dir / filename)
            for filename in ALIGNED_ASSET_FILES
        }
        | {"layer_values_csv": file_state(layer_values_csv)},
        "git": git_state(),
        "source_tree": source_tree_state(),
        "runtime": runtime_state(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_summary(
        out_dir / "summary.md",
        official=official,
        permutation=permutation,
        pca_dim=int(pca_dim),
        n_records=n_records,
    )
    print(f"Wrote official LayerRadial16 outputs to {display_path(out_dir)}")
    return 0


def main() -> int:
    try:
        return _main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
