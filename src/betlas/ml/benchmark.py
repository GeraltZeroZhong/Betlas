from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..constants import FOLD_LABELS
from ..provenance import build_run_manifest, write_json
from .splits import fold_label_counts, make_grouped_splits

ID_COLUMNS = {
    "record_id",
    "pdb_id",
    "assembly_id",
    "model_id",
    "chain_id",
    "domain_id",
    "residue_ranges",
    "fold_label_final",
    "fold_label_candidates",
    "evidence_level",
    "label_source_primary",
    "label_source_supporting",
    "manual_curator",
    "curation_date",
    "label_conflict_notes",
    "qc_status",
    "allowed_for_publication_benchmark",
    "discovered_by_betlas",
    "cath_status",
    "cath_code",
    "cath_architecture_code",
    "cath_topology_code",
    "cath_homology_code",
    "cath_s35_cluster_id",
    "cath_s35_source",
    "cath_name",
    "cz_top_fold",
    "cz_fold_scores_json",
    "cz_warnings",
    "cz_error",
    "cz_axis_best_name",
}

DEPRECATED_SCIENTIFIC_FEATURE_COLUMNS = {
    # Older feature tables computed this from hemisphere-stabilized SSE axes, so
    # the sign does not encode N-to-C strand direction.
    "cz_strand_direction_parallel_fraction",
}

READOUT_FEATURE_COLUMNS = {
    "cz_boundary_region_flag",
    "cz_jelly_rollness",
    "cz_sandwichness",
    "cz_barrel_likeness",
    "cz_jelly_sandwich_overlap",
    "cz_barrel_sandwich_overlap",
    "cz_barrel_jelly_overlap",
    "cz_mixed_topology_score",
    "cz_mixed_topology_flag",
    "cz_secondary_topology_score",
}

READOUT_FEATURE_PREFIXES = (
    "cz_topology_ambiguity_",
    "cz_probability_",
    "cz_rule_top",
    "cz_rule_probability_",
    "cz_rule_label_conflict",
    "cz_neighbor_",
    "cz_boundary_neighbor_",
    "cz_mixed_topology_",
    "cz_manual_boundary_audit_",
)


CONFIG_RESOURCE = "conf/default.yaml"


def load_benchmark_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = OmegaConf.create(files(__package__).joinpath(CONFIG_RESOURCE).read_text())
    if config_path is not None:
        base = OmegaConf.merge(base, OmegaConf.load(Path(config_path)))
    if overrides is not None:
        base = OmegaConf.merge(base, dict(overrides))
    return dict(OmegaConf.to_container(base, resolve=True))


def _cfg_get(config: Mapping[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _cfg_int(config: Mapping[str, Any], dotted_key: str, default: int) -> int:
    return int(_cfg_get(config, dotted_key, default))


def _cfg_float(config: Mapping[str, Any], dotted_key: str, default: float) -> float:
    return float(_cfg_get(config, dotted_key, default))


def _included_models(config: Mapping[str, Any]) -> tuple[str, ...]:
    include = _cfg_get(config, "models.include", ())
    return tuple(str(model) for model in include)


def is_readout_feature_column(column: str) -> bool:
    return column in READOUT_FEATURE_COLUMNS or any(
        column.startswith(prefix) for prefix in READOUT_FEATURE_PREFIXES
    )


class RuleScoreClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, labels: tuple[str, ...] = FOLD_LABELS):
        self.labels = tuple(labels)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RuleScoreClassifier:
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score_cols = [f"cz_rule_score_{label}" for label in self.labels]
        raw = np.column_stack(
            [
                pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
                if col in X
                else np.zeros(len(X))
                for col in score_cols
            ]
        )
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


@dataclass(frozen=True)
class BenchmarkResult:
    metrics_summary: pd.DataFrame
    per_class_metrics: pd.DataFrame
    confusion_matrices: dict[str, pd.DataFrame]
    oof_predictions: pd.DataFrame
    tuning_trace: pd.DataFrame


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if is_readout_feature_column(col):
            continue
        if col in DEPRECATED_SCIENTIFIC_FEATURE_COLUMNS:
            continue
        if not col.startswith("cz_"):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        columns.append(col)
    return columns


def _first_nonempty_series(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    out = pd.Series("", index=df.index, dtype=object)
    for column in columns:
        if column not in df:
            continue
        values = df[column].astype(str).str.strip()
        out = out.mask(out.astype(str).str.strip() == "", values)
    return out


def _make_models(
    random_state: int,
    labels: tuple[str, ...],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    include = set(_included_models(config))
    models: dict[str, Any] = {}
    if "grammar_rules" in include:
        models["grammar_rules"] = RuleScoreClassifier(labels=labels)
    if "logistic_l2" in include:
        class_weight = _cfg_get(config, "logistic_l2.class_weight", "balanced")
        models["logistic_l2"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=_cfg_int(config, "logistic_l2.max_iter", 2000),
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        models["logistic_l2"].set_params(model__C=_cfg_float(config, "logistic_l2.C", 1.0))
    if "random_forest" in include:
        rf_class_weight = _cfg_get(config, "random_forest.class_weight", "balanced_subsample")
        models["random_forest"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=_cfg_int(config, "random_forest.n_estimators", 500),
                        max_features=_cfg_get(config, "random_forest.max_features", "sqrt"),
                        min_samples_leaf=_cfg_int(config, "random_forest.min_samples_leaf", 2),
                        class_weight=rf_class_weight,
                        n_jobs=_cfg_int(config, "random_forest.n_jobs", -1),
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if "extra_trees" in include:
        et_class_weight = _cfg_get(config, "extra_trees.class_weight", "balanced")
        models["extra_trees"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=_cfg_int(config, "extra_trees.n_estimators", 700),
                        max_features=_cfg_get(config, "extra_trees.max_features", "sqrt"),
                        min_samples_leaf=_cfg_int(config, "extra_trees.min_samples_leaf", 1),
                        class_weight=et_class_weight,
                        n_jobs=_cfg_int(config, "extra_trees.n_jobs", -1),
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if "hist_gradient_boosting" in include:
        models["hist_gradient_boosting"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=_cfg_float(
                            config, "hist_gradient_boosting.learning_rate", 0.05
                        ),
                        max_leaf_nodes=_cfg_int(
                            config, "hist_gradient_boosting.max_leaf_nodes", 31
                        ),
                        l2_regularization=_cfg_float(
                            config, "hist_gradient_boosting.l2_regularization", 0.05
                        ),
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def _xgb_param_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = _cfg_get(config, "xgboost.param_grid", ())
    params = [dict(item) for item in grid]
    if not params:
        raise ValueError("xgboost.param_grid must contain at least one candidate")
    return params


def _make_xgb_pipeline(
    params: dict[str, Any],
    random_state: int,
    n_classes: int,
    config: Mapping[str, Any],
) -> Pipeline:
    from xgboost import XGBClassifier

    xgb_kwargs: dict[str, Any] = {
        "n_estimators": _cfg_int(config, "xgboost.n_estimators", 450),
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "tree_method": str(_cfg_get(config, "xgboost.tree_method", "hist")),
        "n_jobs": _cfg_int(config, "xgboost.n_jobs", -1),
        "random_state": random_state,
        **params,
    }
    device = _cfg_get(config, "xgboost.device", None)
    if device not in (None, ""):
        xgb_kwargs["device"] = str(device)
    verbosity = _cfg_get(config, "xgboost.verbosity", None)
    if verbosity is not None:
        xgb_kwargs["verbosity"] = int(verbosity)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(**xgb_kwargs),
            ),
        ]
    )


def _inner_tune_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
    n_classes: int,
    outer_fold: int,
    config: Mapping[str, Any],
) -> tuple[Pipeline, list[dict[str, Any]]]:
    unique_groups = np.unique(groups)
    n_splits = min(_cfg_int(config, "xgboost.inner_splits", 3), len(unique_groups))
    param_grid = _xgb_param_grid(config)
    trace: list[dict[str, Any]] = []
    if n_splits < 2:
        params = param_grid[0]
        return _make_xgb_pipeline(params, random_state, n_classes, config), [
            {"outer_fold": outer_fold, "mean_macro_f1": np.nan, **params}
        ]

    splits, split_strategy, n_splits = make_grouped_splits(
        X,
        y,
        groups,
        n_splits=n_splits,
        random_state=random_state,
    )
    best_score = -1.0
    best_params = param_grid[0]
    for candidate_index, params in enumerate(param_grid, start=1):
        scores: list[float] = []
        for train_idx, val_idx in splits:
            model = _make_xgb_pipeline(params, random_state, n_classes, config)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(f1_score(y[val_idx], pred, average="macro"))
        mean_score = float(np.mean(scores))
        trace.append(
            {
                "outer_fold": outer_fold,
                "candidate_index": int(candidate_index),
                "inner_split_strategy": split_strategy,
                "inner_n_splits": int(n_splits),
                "mean_macro_f1": mean_score,
                **params,
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return _make_xgb_pipeline(best_params, random_state, n_classes, config), trace


def _top2_accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return float("nan")
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    return float(np.mean([truth in row for truth, row in zip(y_true, top2, strict=False)]))


def run_grouped_benchmark(
    features_csv: Path,
    out_dir: Path,
    *,
    n_splits: int = 5,
    random_state: int = 13,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_config = load_benchmark_config(config_path, config)
    OmegaConf.save(
        config=OmegaConf.create(benchmark_config),
        f=out_dir / "benchmark_config_resolved.yaml",
    )
    df = pd.read_csv(features_csv, dtype=str, keep_default_na=False)
    df = df[df["fold_label_final"].isin(FOLD_LABELS)].copy()
    df = df[pd.to_numeric(df.get("cz_parse_ok", 0), errors="coerce").fillna(0).astype(int) == 1]
    feature_cols = numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("no numeric cz_* feature columns were found")

    labels = sorted(FOLD_LABELS)
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(df["fold_label_final"])
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    groups = _first_nonempty_series(
        df, ("cath_s35_cluster_id", "cath_homology_code", "pdb_id")
    ).to_numpy()
    splits, split_strategy, n_splits = make_grouped_splits(
        X,
        y,
        groups,
        n_splits=n_splits,
        random_state=random_state,
    )
    models = _make_models(random_state, tuple(encoder.classes_), benchmark_config)
    include = set(_included_models(benchmark_config))
    if (
        "xgboost_tuned" in include
        and bool(_cfg_get(benchmark_config, "xgboost.enabled", True))
        and _xgboost_available()
    ):
        models["xgboost_tuned"] = "TUNED_XGBOOST"
    if not models:
        raise ValueError("benchmark config did not enable any available models")

    oof_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    model_true_pred: dict[str, tuple[list[int], list[int]]] = {
        model_name: ([], []) for model_name in models
    }

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        for model_name, estimator in models.items():
            if estimator == "TUNED_XGBOOST":
                model, trace = _inner_tune_xgb(
                    X.iloc[train_idx],
                    y[train_idx],
                    groups[train_idx],
                    random_state=random_state,
                    n_classes=len(labels),
                    outer_fold=fold,
                    config=benchmark_config,
                )
                tuning_rows.extend(trace)
            else:
                model = clone(estimator)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = np.asarray(model.predict(X.iloc[test_idx]), dtype=int)
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X.iloc[test_idx]))
            else:
                proba = np.zeros((len(test_idx), len(labels)), dtype=float)
                proba[np.arange(len(test_idx)), pred] = 1.0
            y_true_list, y_pred_list = model_true_pred[model_name]
            y_true_list.extend(y[test_idx].tolist())
            y_pred_list.extend(pred.tolist())
            for local_i, row_index in enumerate(test_idx):
                probabilities = proba[local_i] if proba.ndim == 2 else np.zeros(len(labels))
                top2_idx = np.argsort(probabilities)[-2:][::-1] if len(probabilities) else []
                top1_probability = float(probabilities[top2_idx[0]]) if len(top2_idx) else 0.0
                top2_probability = float(probabilities[top2_idx[1]]) if len(top2_idx) > 1 else 0.0
                probability_row = {
                    f"prob_{label}": float(probabilities[label_index])
                    for label_index, label in enumerate(labels)
                }
                oof_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "record_id": df.iloc[row_index]["record_id"],
                        "pdb_id": df.iloc[row_index]["pdb_id"],
                        "domain_id": df.iloc[row_index]["domain_id"],
                        "group": groups[row_index],
                        "true_label": encoder.inverse_transform([y[row_index]])[0],
                        "pred_label": encoder.inverse_transform([pred[local_i]])[0],
                        "top2_labels": ";".join(encoder.inverse_transform(top2_idx).tolist())
                        if len(top2_idx)
                        else "",
                        "pred_probability": top1_probability,
                        "top2_probability": top2_probability,
                        "top2_margin": top1_probability - top2_probability,
                        **probability_row,
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, pd.DataFrame] = {}
    oof_df = pd.DataFrame(oof_rows)
    for model_name, (truth, pred) in model_true_pred.items():
        truth_arr = np.asarray(truth, dtype=int)
        pred_arr = np.asarray(pred, dtype=int)
        model_oof = oof_df[oof_df["model"] == model_name]
        # Top-2 is stored as labels for audit; recompute from stored labels where possible.
        top2_acc = float(np.mean([
            row.true_label in str(row.top2_labels).split(";") for row in model_oof.itertuples()
        ]))
        summary_rows.append(
            {
                "model": model_name,
                "n": int(len(truth_arr)),
                "accuracy": accuracy_score(truth_arr, pred_arr),
                "balanced_accuracy": balanced_accuracy_score(truth_arr, pred_arr),
                "macro_f1": f1_score(truth_arr, pred_arr, average="macro"),
                "weighted_f1": f1_score(truth_arr, pred_arr, average="weighted"),
                "top2_accuracy": top2_acc,
                "feature_count": len(feature_cols),
            }
        )
        report = classification_report(
            truth_arr,
            pred_arr,
            labels=np.arange(len(labels)),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        for label in labels:
            row = dict(report[label])
            row.update({"model": model_name, "fold_label": label})
            per_class_rows.append(row)
        cm = confusion_matrix(truth_arr, pred_arr, labels=np.arange(len(labels)))
        confusion_matrices[model_name] = pd.DataFrame(cm, index=labels, columns=labels)

    metrics_summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    per_class_metrics = pd.DataFrame(per_class_rows)
    tuning_trace = pd.DataFrame(tuning_rows)

    metrics_summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    per_class_metrics.to_csv(out_dir / "per_class_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    tuning_trace.to_csv(out_dir / "xgboost_tuning.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "feature_columns.csv", index=False)
    for model_name, cm_df in confusion_matrices.items():
        cm_df.to_csv(out_dir / f"confusion_matrix_{model_name}.csv")

    fold_assignments = oof_df[["fold", "record_id", "group"]].drop_duplicates().sort_values(["fold", "record_id"])
    fold_assignments.to_csv(out_dir / "fold_assignments.csv", index=False)
    fold_counts = fold_label_counts(
        record_ids=df["record_id"].to_numpy(),
        labels=df["fold_label_final"].to_numpy(),
        groups=groups,
        splits=splits,
    )
    fold_counts.to_csv(out_dir / "fold_label_counts.csv", index=False)
    manifest = build_run_manifest(
        command="betlas benchmark",
        parameters={
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "models": list(models),
            "xgboost_available": _xgboost_available(),
            "split_strategy": split_strategy,
            "config_path": str(config_path) if config_path is not None else None,
        },
        inputs={"features_csv": features_csv},
        outputs={
            "metrics_summary": out_dir / "metrics_summary.csv",
            "per_class_metrics": out_dir / "per_class_metrics.csv",
            "oof_predictions": out_dir / "oof_predictions.csv",
            "feature_columns": out_dir / "feature_columns.csv",
            "fold_assignments": out_dir / "fold_assignments.csv",
            "fold_label_counts": out_dir / "fold_label_counts.csv",
            "benchmark_config": out_dir / "benchmark_config_resolved.yaml",
        },
        metrics={
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "best_macro_f1": float(metrics_summary.iloc[0]["macro_f1"]),
            "best_model": str(metrics_summary.iloc[0]["model"]),
        },
        extra={
            "labels": labels,
            "feature_columns": feature_cols,
            "split_strategy": split_strategy,
            "fold_label_counts": fold_counts.to_dict(orient="records"),
            "benchmark_config": benchmark_config,
        },
    )
    write_json(out_dir / "benchmark_manifest.json", manifest)
    return BenchmarkResult(metrics_summary, per_class_metrics, confusion_matrices, oof_df, tuning_trace)
