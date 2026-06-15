from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..constants import FOLD_LABELS
from ..provenance import build_run_manifest, write_json
from .benchmark import numeric_feature_columns
from .splits import fold_label_counts, make_grouped_splits

CONFIG_RESOURCE = "conf/ablation_default.yaml"

AGGREGATE_FEATURE_PREFIXES = (
    "cz_rule_score_",
)
AGGREGATE_FEATURES = {
    "cz_rule_margin",
    "cz_parse_ok",
}


@dataclass(frozen=True)
class AblationResult:
    all_results: pd.DataFrame
    group_catalog: pd.DataFrame


def load_ablation_config(
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


def _cfg_str(config: Mapping[str, Any], dotted_key: str, default: str) -> str:
    return str(_cfg_get(config, dotted_key, default))


def raw_geometry_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in numeric_feature_columns(df):
        if column in AGGREGATE_FEATURES:
            continue
        if any(column.startswith(prefix) for prefix in AGGREGATE_FEATURE_PREFIXES):
            continue
        columns.append(column)
    return columns


def feature_group_for(column: str) -> str:
    if column.startswith(("cz_beta_run_", "cz_beta_segment_to_run_")):
        return "beta_run_topology"
    if column.startswith("cz_sheet_pair_"):
        return "sheet_pair_packing"
    if column.startswith(("cz_sheet_seq_", "cz_sheet_order_", "cz_top2_sheet_order_", "cz_jelly_roll_order_")):
        return "sheet_sequence_topology"
    if column.startswith("cz_contact"):
        return "contact_graph"
    if column.startswith(
        (
            "cz_angular_fft_k1",
            "cz_angular_fft_k2",
            "cz_angular_sector_",
            "cz_sandwich_lobe_guard",
        )
    ):
        return "angular_lobes"
    if column.startswith(("cz_axis_periodicity", "cz_angular_fft_k3_8", "cz_angular_fft_k")):
        return "axis_periodicity"
    if column.startswith(("cz_axis_best_", "cz_z_continuity", "cz_angular_gap", "cz_barrel_wall_")):
        return "axis_closure"
    if column.startswith(("cz_sheet_", "cz_largest_sheet", "cz_parallel_sheet", "cz_antiparallel_sheet")):
        return "sheet_global"
    if column.startswith(("cz_strand_order_", "cz_strand_direction_", "cz_strand_ntc_", "cz_strand_axis_")):
        return "strand_order"
    if column.startswith(("cz_alpha_shell_", "cz_helix_beta_", "cz_beta_alpha_")):
        return "alpha_shell"
    if column.startswith(("cz_helix_", "cz_residue_", "cz_beta_residue", "cz_beta_strand", "cz_strand_length", "cz_sse_count")):
        return "composition"
    if column.startswith(("cz_pca_",)):
        return "global_shape"
    return "misc"


def build_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in feature_cols:
        groups.setdefault(feature_group_for(column), []).append(column)
    return {group: sorted(columns) for group, columns in sorted(groups.items())}


def _make_model(random_state: int, config: Mapping[str, Any]) -> Any:
    model_name = _cfg_str(config, "model.name", "hist_gradient_boosting")
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("ablation model.name=xgboost requires the optional xgboost package") from exc
        return XGBClassifier(
            objective=_cfg_str(config, "model.objective", "multi:softprob"),
            eval_metric=_cfg_str(config, "model.eval_metric", "mlogloss"),
            num_class=len(FOLD_LABELS),
            n_estimators=_cfg_int(config, "model.n_estimators", 120),
            max_depth=_cfg_int(config, "model.max_depth", 4),
            learning_rate=_cfg_float(config, "model.learning_rate", 0.08),
            subsample=_cfg_float(config, "model.subsample", 0.90),
            colsample_bytree=_cfg_float(config, "model.colsample_bytree", 0.85),
            reg_lambda=_cfg_float(config, "model.reg_lambda", 3.0),
            tree_method=_cfg_str(config, "model.tree_method", "hist"),
            device=_cfg_str(config, "model.device", "cpu"),
            n_jobs=_cfg_int(config, "model.n_jobs", 1),
            verbosity=_cfg_int(config, "model.verbosity", 0),
            random_state=random_state,
        )
    if model_name != "hist_gradient_boosting":
        raise ValueError("ablation model.name must be 'hist_gradient_boosting' or 'xgboost'")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=_cfg_float(config, "model.learning_rate", 0.05),
                    max_leaf_nodes=_cfg_int(config, "model.max_leaf_nodes", 31),
                    l2_regularization=_cfg_float(config, "model.l2_regularization", 0.05),
                    max_iter=_cfg_int(config, "model.max_iter", 100),
                    random_state=random_state,
                ),
            ),
        ]
    )


def _groups(df: pd.DataFrame) -> np.ndarray:
    groups = pd.Series("", index=df.index, dtype=object)
    for column in ("cath_s35_cluster_id", "cath_homology_code", "pdb_id"):
        if column not in df:
            continue
        values = df[column].astype(str).str.strip()
        groups = groups.mask(groups.astype(str).str.strip() == "", values)
    return groups.to_numpy()


def _rate(true_labels: np.ndarray, pred_labels: np.ndarray, true_label: str, pred_label: str) -> float:
    mask = true_labels == true_label
    if not bool(np.any(mask)):
        return 0.0
    return float(np.mean(pred_labels[mask] == pred_label))


def _any_rate(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    true_label: str,
    pred_options: set[str],
) -> float:
    mask = true_labels == true_label
    if not bool(np.any(mask)):
        return 0.0
    return float(np.mean([pred in pred_options for pred in pred_labels[mask]]))


def evaluate_feature_subset(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    encoder: LabelEncoder,
    random_state: int,
    config: Mapping[str, Any],
    x_all: pd.DataFrame | None = None,
    y_encoded: np.ndarray | None = None,
) -> dict[str, float]:
    if not feature_cols:
        raise ValueError("cannot evaluate an empty feature subset")
    y = y_encoded if y_encoded is not None else encoder.transform(df["fold_label_final"])
    x = x_all.loc[:, feature_cols] if x_all is not None else df[feature_cols].apply(pd.to_numeric, errors="coerce")
    pred = np.empty_like(y)
    for train_idx, test_idx in splits:
        model = _make_model(random_state, config)
        model.fit(x.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x.iloc[test_idx])

    labels = list(encoder.classes_)
    true_labels = encoder.inverse_transform(y)
    pred_labels = encoder.inverse_transform(pred)
    boundary_labels = ["beta_barrel", "beta_sandwich", "jelly_roll"]
    boundary_indices = [encoder.transform([label])[0] for label in boundary_labels]
    sandwich_jelly_indices = [
        encoder.transform(["beta_sandwich"])[0],
        encoder.transform(["jelly_roll"])[0],
    ]
    sandwich_jelly_mask = np.isin(y, sandwich_jelly_indices)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "boundary_macro_f1": float(
            f1_score(y, pred, labels=boundary_indices, average="macro", zero_division=0)
        ),
        "sandwich_jelly_macro_f1": float(
            f1_score(y, pred, labels=sandwich_jelly_indices, average="macro", zero_division=0)
        ),
        "sandwich_jelly_accuracy": float(np.mean(y[sandwich_jelly_mask] == pred[sandwich_jelly_mask])),
        "sandwich_to_jelly_rate": _rate(true_labels, pred_labels, "beta_sandwich", "jelly_roll"),
        "jelly_to_sandwich_rate": _rate(true_labels, pred_labels, "jelly_roll", "beta_sandwich"),
        "sandwich_to_barrel_rate": _rate(true_labels, pred_labels, "beta_sandwich", "beta_barrel"),
        "jelly_to_barrel_rate": _rate(true_labels, pred_labels, "jelly_roll", "beta_barrel"),
        "barrel_to_sandwich_or_jelly_rate": _any_rate(
            true_labels,
            pred_labels,
            "beta_barrel",
            {"beta_sandwich", "jelly_roll"},
        ),
        "feature_count": float(len(feature_cols)),
        **{
            f"f1_{label}": float(
                f1_score(
                    y,
                    pred,
                    labels=[encoder.transform([label])[0]],
                    average="macro",
                    zero_division=0,
                )
            )
            for label in labels
        },
    }


def _row(
    *,
    ablation_type: str,
    name: str,
    removed_features: list[str],
    kept_features: list[str],
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None = None,
    fit_seconds: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ablation_type": ablation_type,
        "name": name,
        "removed_feature_count": len(removed_features),
        "kept_feature_count": len(kept_features),
        "removed_features": ";".join(removed_features),
        "kept_features": ";".join(kept_features),
    }
    if fit_seconds is not None:
        row["fit_seconds"] = float(fit_seconds)
    row.update(metrics)
    if baseline_metrics:
        for key, value in metrics.items():
            if key.endswith("count"):
                continue
            if isinstance(value, (float, int)) and key in baseline_metrics:
                row[f"delta_{key}"] = float(value) - float(baseline_metrics[key])
    return row


def _without(feature_cols: list[str], removed: set[str]) -> list[str]:
    return [column for column in feature_cols if column not in removed]


def run_ablation_suite(
    features_csv: Path,
    out_dir: Path,
    *,
    n_splits: int = 5,
    random_state: int = 13,
    max_group_combo_size: int = 3,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> AblationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_config = load_ablation_config(config_path, config)
    max_group_combo_size = int(
        _cfg_get(ablation_config, "ablation.max_group_combo_size", max_group_combo_size)
    )
    OmegaConf.save(
        config=OmegaConf.create(ablation_config),
        f=out_dir / "ablation_config_resolved.yaml",
    )
    df = pd.read_csv(features_csv, dtype=str, keep_default_na=False)
    df = df[df["fold_label_final"].isin(FOLD_LABELS)].copy()
    df = df[pd.to_numeric(df.get("cz_parse_ok", 0), errors="coerce").fillna(0).astype(int) == 1]
    feature_cols = raw_geometry_feature_columns(df)
    if not feature_cols:
        raise ValueError("no raw geometry feature columns found")
    x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    encoder = LabelEncoder()
    encoder.fit(sorted(FOLD_LABELS))
    groups = _groups(df)
    y = encoder.transform(df["fold_label_final"])
    splits, split_strategy, n_splits = make_grouped_splits(
        df[feature_cols],
        y,
        groups,
        n_splits=n_splits,
        random_state=random_state,
    )
    feature_groups = build_feature_groups(feature_cols)
    group_catalog = pd.DataFrame(
        [
            {"group": group, "feature_count": len(columns), "features": ";".join(columns)}
            for group, columns in feature_groups.items()
        ]
    )

    rows: list[dict[str, Any]] = []

    def write_partial() -> None:
        partial = pd.DataFrame(rows)
        if not partial.empty:
            partial.to_csv(out_dir / "ablation_summary.partial.csv", index=False)

    def evaluate_row(
        *,
        ablation_type: str,
        name: str,
        removed_features: list[str],
        kept_features: list[str],
        baseline_metrics: dict[str, float] | None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        start = time.perf_counter()
        metrics = evaluate_feature_subset(
            df,
            kept_features,
            splits=splits,
            encoder=encoder,
            random_state=random_state,
            config=ablation_config,
            x_all=x_all,
            y_encoded=y,
        )
        elapsed = time.perf_counter() - start
        row = _row(
            ablation_type=ablation_type,
            name=name,
            removed_features=removed_features,
            kept_features=kept_features,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            fit_seconds=elapsed,
        )
        row["ablation_index"] = len(rows)
        return row, metrics

    group_names = list(feature_groups)
    combo_limit = min(max_group_combo_size, max(1, len(group_names) - 1))
    total_evaluations = 1 + len(feature_groups) + len(feature_groups) + len(feature_cols) + len(feature_cols)
    for combo_size in range(2, combo_limit + 1):
        total_evaluations += sum(1 for _ in itertools.combinations(group_names, combo_size))

    row, baseline_metrics = evaluate_row(
        ablation_type="baseline",
        name="all_raw_geometry",
        removed_features=[],
        kept_features=feature_cols,
        baseline_metrics=None,
    )
    rows.append(row)
    write_partial()

    with tqdm(total=total_evaluations, initial=1, desc="Running full-scale ablations") as progress:
        for group, columns in feature_groups.items():
            kept = _without(feature_cols, set(columns))
            row, _ = evaluate_row(
                ablation_type="drop_group_single",
                name=group,
                removed_features=columns,
                kept_features=kept,
                baseline_metrics=baseline_metrics,
            )
            rows.append(row)
            progress.update(1)
            if len(rows) % 10 == 0:
                write_partial()

        for combo_size in range(2, combo_limit + 1):
            for combo in itertools.combinations(group_names, combo_size):
                removed = sorted({feature for group in combo for feature in feature_groups[group]})
                kept = _without(feature_cols, set(removed))
                if not kept:
                    continue
                row, _ = evaluate_row(
                    ablation_type=f"drop_group_combo_{combo_size}",
                    name="+".join(combo),
                    removed_features=removed,
                    kept_features=kept,
                    baseline_metrics=baseline_metrics,
                )
                rows.append(row)
                progress.update(1)
                if len(rows) % 10 == 0:
                    write_partial()

        for group, columns in feature_groups.items():
            row, _ = evaluate_row(
                ablation_type="only_group",
                name=group,
                removed_features=_without(feature_cols, set(columns)),
                kept_features=columns,
                baseline_metrics=baseline_metrics,
            )
            rows.append(row)
            progress.update(1)
            if len(rows) % 10 == 0:
                write_partial()

        for feature in feature_cols:
            kept = _without(feature_cols, {feature})
            row, _ = evaluate_row(
                ablation_type="drop_feature_single",
                name=feature,
                removed_features=[feature],
                kept_features=kept,
                baseline_metrics=baseline_metrics,
            )
            rows.append(row)
            progress.update(1)
            if len(rows) % 10 == 0:
                write_partial()

        for feature in feature_cols:
            row, _ = evaluate_row(
                ablation_type="only_feature_single",
                name=feature,
                removed_features=_without(feature_cols, {feature}),
                kept_features=[feature],
                baseline_metrics=baseline_metrics,
            )
            rows.append(row)
            progress.update(1)
            if len(rows) % 10 == 0:
                write_partial()

    write_partial()

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "ablation_summary.csv", index=False)
    group_catalog.to_csv(out_dir / "feature_group_catalog.csv", index=False)
    fold_counts = fold_label_counts(
        record_ids=df["record_id"].to_numpy(),
        labels=df["fold_label_final"].to_numpy(),
        groups=groups,
        splits=splits,
    )
    fold_counts.to_csv(out_dir / "fold_label_counts.csv", index=False)
    for ablation_type, part in results.groupby("ablation_type"):
        part.to_csv(out_dir / f"{ablation_type}.csv", index=False)

    aggregate_features_excluded = sorted(
        [
            column
            for column in numeric_feature_columns(df)
            if column not in feature_cols
            and (
                column in AGGREGATE_FEATURES
                or any(column.startswith(prefix) for prefix in AGGREGATE_FEATURE_PREFIXES)
            )
        ]
    )
    manifest = build_run_manifest(
        command="betlas ablate",
        parameters={
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "max_group_combo_size": int(max_group_combo_size),
            "model": _cfg_str(ablation_config, "model.name", "hist_gradient_boosting"),
            "split_strategy": split_strategy,
            "config_path": str(config_path) if config_path is not None else None,
        },
        inputs={"features_csv": features_csv},
        outputs={
            "ablation_summary": out_dir / "ablation_summary.csv",
            "feature_group_catalog": out_dir / "feature_group_catalog.csv",
            "fold_label_counts": out_dir / "fold_label_counts.csv",
            "ablation_config": out_dir / "ablation_config_resolved.yaml",
        },
        metrics={
            "n_rows": int(len(df)),
            "raw_feature_count": int(len(feature_cols)),
            "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
            "baseline_boundary_macro_f1": float(baseline_metrics["boundary_macro_f1"]),
        },
        extra={
            "aggregate_features_excluded": aggregate_features_excluded,
            "feature_groups": {group: len(columns) for group, columns in feature_groups.items()},
            "fold_label_counts": fold_counts.to_dict(orient="records"),
            "ablation_config": ablation_config,
        },
    )
    write_json(out_dir / "ablation_manifest.json", manifest)
    return AblationResult(results, group_catalog)
