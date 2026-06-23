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
from ..schema import normalize_feature_columns
from .splits import fold_label_counts, make_grouped_splits

BENCHMARK_REQUIRED_COLUMNS = ("record_id", "pdb_id", "domain_id", "fold_label_final")
BENCHMARK_GROUP_COLUMNS = ("cath_s35_cluster_id", "cath_homology_code", "pdb_id")
RULE_SCORE_COLUMNS = tuple(f"betlas_rule_score_{label}" for label in FOLD_LABELS)

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
    "allowed_for_benchmark",
    "discovered_by_betlas",
    "cath_status",
    "cath_code",
    "cath_architecture_code",
    "cath_topology_code",
    "cath_homology_code",
    "cath_s35_cluster_id",
    "cath_s35_source",
    "cath_name",
    "betlas_top_fold",
    "betlas_fold_scores_json",
    "betlas_warnings",
    "betlas_error",
    "betlas_axis_best_name",
}

DEPRECATED_SCIENTIFIC_FEATURE_COLUMNS = {
    # Older feature tables computed this from hemisphere-stabilized SSE axes, so
    # the sign does not encode N-to-C strand direction.
    "betlas_strand_direction_parallel_fraction",
}

READOUT_FEATURE_COLUMNS = {
    "betlas_boundary_region_flag",
    "betlas_jelly_rollness",
    "betlas_sandwichness",
    "betlas_barrel_likeness",
    "betlas_jelly_sandwich_overlap",
    "betlas_barrel_sandwich_overlap",
    "betlas_barrel_jelly_overlap",
    "betlas_mixed_topology_score",
    "betlas_mixed_topology_flag",
    "betlas_secondary_topology_score",
}

READOUT_FEATURE_PREFIXES = (
    "betlas_topology_ambiguity_",
    "betlas_probability_",
    "betlas_rule_top",
    "betlas_rule_probability_",
    "betlas_rule_label_conflict",
    "betlas_neighbor_",
    "betlas_boundary_neighbor_",
    "betlas_mixed_topology_",
    "betlas_manual_boundary_audit_",
)
AGGREGATE_FEATURE_PREFIXES = ("betlas_rule_score_",)
AGGREGATE_FEATURES = {
    "betlas_rule_margin",
    "betlas_parse_ok",
}
FEATURE_SETS = ("raw_geometry", "raw_plus_rule_scores", "rules_only")


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
        score_cols = [f"betlas_rule_score_{label}" for label in self.labels]
        missing = [column for column in score_cols if column not in X]
        if missing:
            raise ValueError(
                "grammar_rules benchmark requires precomputed Betlas rule-score columns; "
                f"missing columns: {', '.join(missing)}"
            )
        raw = np.column_stack(
            [
                pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
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
        if not col.startswith("betlas_"):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        columns.append(col)
    return columns


def raw_geometry_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in numeric_feature_columns(df):
        if column in AGGREGATE_FEATURES:
            continue
        if any(column.startswith(prefix) for prefix in AGGREGATE_FEATURE_PREFIXES):
            continue
        columns.append(column)
    return columns


def feature_columns_for_set(df: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "raw_geometry":
        return raw_geometry_feature_columns(df)
    if feature_set == "raw_plus_rule_scores":
        return [
            column
            for column in numeric_feature_columns(df)
            if column != "betlas_parse_ok" and column != "betlas_rule_margin"
        ]
    if feature_set == "rules_only":
        return [column for column in RULE_SCORE_COLUMNS if column in df.columns]
    available = ", ".join(FEATURE_SETS)
    raise ValueError(f"unknown benchmark feature set {feature_set!r}; expected one of: {available}")


def _connected_group_series(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Return grouping components linked by any non-empty public group column."""

    n = len(df)
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    seen_tokens: dict[str, int] = {}
    row_has_token = [False] * n
    for column in columns:
        if column not in df:
            continue
        values = df[column].astype(str).str.strip().str.upper().reset_index(drop=True)
        for index, value in enumerate(values):
            if not value:
                continue
            row_has_token[index] = True
            token = f"{column}={value}"
            previous = seen_tokens.get(token)
            if previous is None:
                seen_tokens[token] = index
            else:
                union(previous, index)

    missing_indices = [index for index, has_token in enumerate(row_has_token) if not has_token]
    if missing_indices:
        examples = ", ".join(str(df.index[index]) for index in missing_indices[:10])
        more = f"; plus {len(missing_indices) - 10} more" if len(missing_indices) > 10 else ""
        raise ValueError(
            "grouped cross-validation requires every retained row to have at least one "
            f"non-empty group identifier in {', '.join(columns)}; missing row indices: "
            f"{examples}{more}"
        )

    labels: list[str] = []
    root_to_label: dict[int, str] = {}
    for index in range(n):
        root = find(index)
        label = root_to_label.get(root)
        if label is None:
            tokens = sorted(token for token, token_index in seen_tokens.items() if find(token_index) == root)
            label = "|".join(tokens[:4]) + (f"|plus_{len(tokens) - 4}_more" if len(tokens) > 4 else "")
            root_to_label[root] = label
        labels.append(label)
    return pd.Series(labels, index=df.index, dtype=object)


def _group_presence_mask(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    present = pd.Series(False, index=df.index)
    for column in columns:
        if column not in df:
            continue
        present = present | df[column].astype(str).str.strip().ne("")
    return present


def _missing_group_examples(df: pd.DataFrame, mask: pd.Series, *, limit: int = 10) -> list[str]:
    if "record_id" in df.columns:
        values = df.loc[~mask, "record_id"].astype(str).str.strip()
        values = values[values.ne("")]
        if not values.empty:
            return values.head(limit).tolist()
    return [str(index) for index in df.index[~mask][:limit]]


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} requires column(s): {', '.join(missing)}")


def _group_source_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    remaining = pd.Series(True, index=df.index)
    for column in BENCHMARK_GROUP_COLUMNS:
        if column not in df:
            counts[column] = 0
            continue
        values = df[column].astype(str).str.strip()
        selected = remaining & values.ne("")
        counts[column] = int(selected.sum())
        remaining = remaining & ~selected
    counts["empty"] = int(remaining.sum())
    return counts


def _validate_rule_score_values(df: pd.DataFrame) -> None:
    missing = [column for column in RULE_SCORE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "grammar_rules benchmark requires precomputed Betlas rule-score columns; "
            f"missing columns: {', '.join(missing)}. "
            "Run 'betlas grammar score' and join the score columns, or remove grammar_rules from the benchmark config."
        )
    invalid: list[str] = []
    for column in RULE_SCORE_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            invalid.append(column)
    if invalid:
        raise ValueError(
            "grammar_rules benchmark requires finite numeric rule-score values; "
            f"invalid columns: {', '.join(invalid)}"
        )


def _split_summary(
    splits: list[tuple[np.ndarray, np.ndarray]],
    labels: pd.Series,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_labels = sorted(FOLD_LABELS)
    label_values = labels.to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_counts = pd.Series(label_values[train_idx]).value_counts().to_dict()
        test_counts = pd.Series(label_values[test_idx]).value_counts().to_dict()
        rows.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_group_count": int(pd.Series(groups[train_idx]).astype(str).nunique()),
                "test_group_count": int(pd.Series(groups[test_idx]).astype(str).nunique()),
                "train_class_counts": {label: int(train_counts.get(label, 0)) for label in all_labels},
                "test_class_counts": {label: int(test_counts.get(label, 0)) for label in all_labels},
                "missing_test_classes": [label for label in all_labels if int(test_counts.get(label, 0)) == 0],
            }
        )
    return rows


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


def _align_predict_proba(
    probabilities: np.ndarray,
    estimator: Any,
    *,
    labels: list[str],
    model_name: str,
    fold: int,
) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2:
        warnings.append(f"{model_name} fold {fold}: predict_proba returned non-matrix output")
        return np.zeros((len(proba), len(labels)), dtype=float), warnings

    classes = getattr(estimator, "classes_", None)
    if classes is None or len(classes) == 0:
        if proba.shape[1] == len(labels):
            return proba, warnings
        warnings.append(
            f"{model_name} fold {fold}: predict_proba had {proba.shape[1]} columns but estimator classes_ was unavailable"
        )
        aligned = np.zeros((proba.shape[0], len(labels)), dtype=float)
        cols = min(proba.shape[1], len(labels))
        aligned[:, :cols] = proba[:, :cols]
        return aligned, warnings

    aligned = np.zeros((proba.shape[0], len(labels)), dtype=float)
    class_values = list(classes)
    if len(class_values) != proba.shape[1]:
        warnings.append(
            f"{model_name} fold {fold}: predict_proba column count {proba.shape[1]} "
            f"does not match classes_ length {len(class_values)}"
        )
    used_classes: set[int] = set()
    for source_col, class_value in enumerate(class_values[: proba.shape[1]]):
        try:
            target_col = int(class_value)
        except (TypeError, ValueError):
            warnings.append(f"{model_name} fold {fold}: non-integer class label {class_value!r}")
            continue
        if 0 <= target_col < len(labels):
            aligned[:, target_col] = proba[:, source_col]
            used_classes.add(target_col)
        else:
            warnings.append(f"{model_name} fold {fold}: class index {target_col} outside global labels")
    missing = [labels[index] for index in range(len(labels)) if index not in used_classes]
    if missing:
        warnings.append(
            f"{model_name} fold {fold}: predict_proba omitted classes filled with 0: {', '.join(missing)}"
        )
    return aligned, warnings


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
    df_all = normalize_feature_columns(pd.read_csv(features_csv, dtype=str, keep_default_na=False))
    _require_columns(df_all, BENCHMARK_REQUIRED_COLUMNS, context="benchmark")
    label_mask = df_all["fold_label_final"].isin(FOLD_LABELS)
    df = df_all[label_mask].copy()
    parse_mask = pd.to_numeric(df.get("betlas_parse_ok", 0), errors="coerce").fillna(0).astype(int) == 1
    df = df[parse_mask].copy()
    feature_set = str(_cfg_get(benchmark_config, "features.set", "raw_geometry"))
    feature_cols = feature_columns_for_set(df, feature_set)
    if not feature_cols:
        write_json(
            out_dir / "benchmark_preflight.json",
            {
                "status": "failed",
                "failure_stage": "feature_schema",
                "error": "no numeric betlas_* feature columns were found",
                "input_rows": int(len(df_all)),
                "label_filtered_rows": int(label_mask.sum()),
                "parse_ok_rows": int(len(df)),
                "feature_set": feature_set,
                "feature_count": 0,
                "features_csv": str(features_csv),
            },
        )
        raise ValueError("no numeric betlas_* feature columns were found")

    labels = sorted(FOLD_LABELS)
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(df["fold_label_final"])
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    include = set(_included_models(benchmark_config))
    group_presence = _group_presence_mask(df, BENCHMARK_GROUP_COLUMNS)
    missing_group_count = int((~group_presence).sum())
    group_examples = _missing_group_examples(df, group_presence)
    groups = (
        _connected_group_series(df, BENCHMARK_GROUP_COLUMNS).to_numpy()
        if missing_group_count == 0
        else np.asarray([], dtype=object)
    )
    missing_rule_scores = [column for column in RULE_SCORE_COLUMNS if column not in df.columns]
    base_preflight = {
        "input_rows": int(len(df_all)),
        "label_filtered_rows": int(label_mask.sum()),
        "parse_ok_rows": int(len(df)),
        "dropped_rows": int(len(df_all) - len(df)),
        "label_filter": f"fold_label_final in {list(FOLD_LABELS)}",
        "parse_filter": "betlas_parse_ok == 1",
        "class_counts": {str(k): int(v) for k, v in df["fold_label_final"].value_counts().sort_index().items()},
        "group_columns_priority": list(BENCHMARK_GROUP_COLUMNS),
        "grouping_strategy": "connected_components_across_group_columns",
        "group_source_counts": _group_source_counts(df),
        "missing_group_row_count": missing_group_count,
        "missing_group_row_examples": group_examples,
        "group_count": int(pd.Series(groups).astype(str).nunique()) if missing_group_count == 0 else 0,
        "feature_set": feature_set,
        "feature_count": int(len(feature_cols)),
        "rule_score_columns_present": not missing_rule_scores,
        "missing_rule_score_columns": missing_rule_scores,
        "required_global_classes": list(FOLD_LABELS),
        "missing_global_classes": sorted(set(FOLD_LABELS) - set(df["fold_label_final"].astype(str))),
        "require_all_fold_labels": True,
        "features_csv": str(features_csv),
        "effective_split_strategy": "not_run",
        "effective_n_splits": 0,
        "folds": [],
        "model_dependency_status": {model: "requested" for model in sorted(include)},
        "allow_model_skip": bool(_cfg_get(benchmark_config, "models.allow_model_skip", False)),
    }
    if missing_group_count:
        write_json(
            out_dir / "benchmark_preflight.json",
            {
                **base_preflight,
                "status": "failed",
                "failure_stage": "group_coverage",
                "effective_split_strategy": "not_run_missing_group_identifiers",
                "error": (
                    "benchmark requires every retained row to have at least one non-empty "
                    f"group identifier in {', '.join(BENCHMARK_GROUP_COLUMNS)}; "
                    f"missing rows: {group_examples}"
                ),
            },
        )
        raise ValueError(
            "benchmark requires every retained row to have at least one non-empty "
            f"group identifier in {', '.join(BENCHMARK_GROUP_COLUMNS)}; "
            f"missing rows: {group_examples}"
        )
    if "grammar_rules" in include and missing_rule_scores:
        write_json(
            out_dir / "benchmark_preflight.json",
            {
                **base_preflight,
                "status": "failed",
                "failure_stage": "feature_schema",
                "effective_split_strategy": "not_run_missing_rule_scores",
                "error": (
                    "grammar_rules benchmark requires complete Betlas rule-score columns; "
                    f"missing: {missing_rule_scores}"
                ),
                "model_dependency_status": {"grammar_rules": "requested"},
            },
        )
        _validate_rule_score_values(df)
    missing_global_classes = sorted(set(FOLD_LABELS) - set(df["fold_label_final"].astype(str)))
    if missing_global_classes:
        write_json(
            out_dir / "benchmark_preflight.json",
            {
                **base_preflight,
                "status": "failed",
                "failure_stage": "class_coverage",
                "effective_split_strategy": "not_run_missing_global_classes",
                "error": (
                    "benchmark requires all Betlas fold labels before grouped CV; "
                    f"missing global classes: {missing_global_classes}"
                ),
            },
        )
        raise ValueError(
            "benchmark requires all Betlas fold labels before grouped CV; "
            f"missing global classes: {missing_global_classes}"
        )
    try:
        splits, split_strategy, n_splits = make_grouped_splits(
            X,
            y,
            groups,
            n_splits=n_splits,
            random_state=random_state,
        )
    except ValueError as exc:
        write_json(
            out_dir / "benchmark_preflight.json",
            {
                **base_preflight,
                "status": "failed",
                "failure_stage": "split_coverage",
                "effective_split_strategy": "split_failed",
                "error": str(exc),
                "split_error": str(exc),
            },
        )
        raise
    split_rows = _split_summary(splits, df["fold_label_final"], groups)
    allow_model_skip = bool(_cfg_get(benchmark_config, "models.allow_model_skip", False))
    model_status: dict[str, str] = {model: "requested" for model in sorted(include)}
    xgboost_available = _xgboost_available()
    xgboost_missing_error = ""
    if "xgboost_tuned" in include and not xgboost_available and bool(_cfg_get(benchmark_config, "xgboost.enabled", True)):
        model_status["xgboost_tuned"] = "unavailable: xgboost is not installed"
        if not allow_model_skip:
            xgboost_missing_error = (
                "benchmark config requested xgboost_tuned, but xgboost is not installed. "
                "Install betlas[ml] or set models.allow_model_skip=true."
            )
    preflight = {
        **base_preflight,
        "status": "failed" if xgboost_missing_error else "ok",
        "failure_stage": "dependency" if xgboost_missing_error else "",
        "effective_split_strategy": split_strategy,
        "effective_n_splits": int(n_splits),
        "folds": split_rows,
        "model_dependency_status": model_status,
    }
    write_json(out_dir / "benchmark_preflight.json", preflight)
    if xgboost_missing_error:
        raise ValueError(xgboost_missing_error)
    if "grammar_rules" in include:
        _validate_rule_score_values(df)
        X_rules = df[list(RULE_SCORE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    else:
        X_rules = pd.DataFrame(index=df.index)
    models = _make_models(random_state, tuple(encoder.classes_), benchmark_config)
    if (
        "xgboost_tuned" in include
        and bool(_cfg_get(benchmark_config, "xgboost.enabled", True))
        and xgboost_available
    ):
        models["xgboost_tuned"] = "TUNED_XGBOOST"
    if not models:
        error = "benchmark config did not enable any available models"
        write_json(
            out_dir / "benchmark_preflight.json",
            {
                **preflight,
                "status": "failed",
                "failure_stage": "dependency",
                "error": error,
                "model_dependency_status": {
                    **model_status,
                    **{model: "skipped_or_disabled" for model in sorted(include) if model not in model_status},
                },
            },
        )
        raise ValueError("benchmark config did not enable any available models")
    model_feature_columns = {
        model_name: list(RULE_SCORE_COLUMNS) if model_name == "grammar_rules" else list(feature_cols)
        for model_name in models
    }

    oof_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    fold_warnings: list[str] = []
    model_true_pred: dict[str, tuple[list[int], list[int]]] = {
        model_name: ([], []) for model_name in models
    }

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        for model_name, estimator in models.items():
            X_model = X_rules if model_name == "grammar_rules" else X
            if estimator == "TUNED_XGBOOST":
                model, trace = _inner_tune_xgb(
                    X_model.iloc[train_idx],
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
            model.fit(X_model.iloc[train_idx], y[train_idx])
            pred = np.asarray(model.predict(X_model.iloc[test_idx]), dtype=int)
            if hasattr(model, "predict_proba"):
                raw_proba = np.asarray(model.predict_proba(X_model.iloc[test_idx]))
                proba, alignment_warnings = _align_predict_proba(
                    raw_proba,
                    model,
                    labels=labels,
                    model_name=model_name,
                    fold=fold,
                )
                fold_warnings.extend(alignment_warnings)
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
                        "probability_source": "predict_proba"
                        if hasattr(model, "predict_proba")
                        else "predicted_label_one_hot",
                        "probability_calibration_status": "model_reported_uncalibrated"
                        if hasattr(model, "predict_proba")
                        else "not_applicable",
                        "probability_alignment_warnings": "|".join(alignment_warnings)
                        if hasattr(model, "predict_proba")
                        else "",
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
                "feature_count": len(model_feature_columns[model_name]),
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
    pd.DataFrame(
        [
            {"model": model_name, "feature_set": "rules_only" if model_name == "grammar_rules" else feature_set, "feature": feature}
            for model_name, columns in model_feature_columns.items()
            for feature in columns
        ]
    ).to_csv(out_dir / "feature_columns.csv", index=False)
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
            "benchmark_preflight": out_dir / "benchmark_preflight.json",
            "metrics_summary": out_dir / "metrics_summary.csv",
            "per_class_metrics": out_dir / "per_class_metrics.csv",
            "oof_predictions": out_dir / "oof_predictions.csv",
            "feature_columns": out_dir / "feature_columns.csv",
            "fold_assignments": out_dir / "fold_assignments.csv",
            "fold_label_counts": out_dir / "fold_label_counts.csv",
            "xgboost_tuning": out_dir / "xgboost_tuning.csv",
            "benchmark_config": out_dir / "benchmark_config_resolved.yaml",
            **{
                f"confusion_matrix_{model_name}": out_dir / f"confusion_matrix_{model_name}.csv"
                for model_name in confusion_matrices
            },
        },
        metrics={
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "best_macro_f1": float(metrics_summary.iloc[0]["macro_f1"]),
            "best_model": str(metrics_summary.iloc[0]["model"]),
        },
        extra={
            "labels": labels,
            "feature_columns_by_model": model_feature_columns,
            "split_strategy": split_strategy,
            "split_summary": split_rows,
            "fold_label_counts": fold_counts.to_dict(orient="records"),
            "benchmark_config": benchmark_config,
            "probability_alignment_warnings": fold_warnings,
            "model_dependency_status": model_status,
        },
    )
    write_json(out_dir / "benchmark_manifest.json", manifest)
    return BenchmarkResult(metrics_summary, per_class_metrics, confusion_matrices, oof_df, tuning_trace)
