#!/usr/bin/env python
"""Train a lightweight ML strand-count baseline from beta-barrel stave readout features.

The script joins an existing beta-barrel-staves CSV to the gold/pass records in the
structure dataset, generates out-of-fold ML predictions, and compares them with
the direct geometry `strand_count` on the same rows.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from .publication import require_publication_gold_provenance

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_GOLD_CSV = (
    REPO_ROOT
    / "data"
    / "readouts"
    / "beta_barrel_staves"
    / "processed"
    / "beta_barrel_publication_gold.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "readouts" / "beta_barrel_staves" / "ml_baseline"

SCORING_NUMERIC_FEATURES = [
    "strand_count",
    "confidence",
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
    "candidate_strands",
    "persistent_strands",
    "trajectory_candidate_strands",
    "trajectory_merge_count",
    "support_consensus_strands",
    "support_consensus_threshold",
    "layer_count_q75",
    "layer_count_q90",
    "layer_count_max",
    "selected_strand_support_mean",
    "selected_strand_support_min",
    "count_decision_score",
    "layer_count_max_plateau",
    "layer_count_largest_drop",
    "absent_from_max_strands",
    "terminal_strand_support",
    "terminal_consensus_strands",
    "terminal_support_fraction",
    "barrel_wall_graph_applied",
    "barrel_wall_graph_count",
    "barrel_wall_graph_edge_min_count",
    "barrel_wall_graph_support_ratio",
    "barrel_wall_graph_node_count",
    "barrel_wall_graph_edge_count",
    "barrel_wall_graph_component_edge_count",
    "barrel_wall_graph_component_avg_degree",
    "barrel_wall_graph_component_cycle_rank",
    "barrel_wall_graph_component_degree2_fraction",
    "barrel_wall_graph_component_branch_fraction",
    "barrel_wall_graph_component_radial_rank_mean",
    "barrel_wall_graph_component_radial_rank_min",
    "barrel_wall_graph_edge_density",
    "barrel_wall_graph_membership_score",
    "barrel_wall_graph_membership_block_count",
    "barrel_wall_graph_membership_largest_block_fraction",
    "barrel_wall_graph_membership_plateau_edges",
    "barrel_wall_graph_membership_next_drop_fraction",
    "barrel_wall_graph_membership_variant_agreement",
    "barrel_wall_graph_observed_layers",
    "axis_hypothesis_score",
    "axis_hypothesis_proxy_rank",
    "run_window_core_applied",
    "run_window_core_score",
    "run_window_core_start",
    "run_window_core_stop",
    "run_window_core_trim_left",
    "run_window_core_trim_right",
    "run_window_core_candidate_windows",
    "run_window_core_reanalyzed_windows",
    "run_window_core_base_strand_count",
    "run_window_core_base_support_consensus",
    "run_window_core_base_layer_q90",
    "run_window_core_base_layer_max",
    "run_window_core_layer_max_preservation",
    "run_window_core_layer_q90_preservation",
    "run_window_core_outside_support_fraction",
    "run_window_core_z_coverage_fraction",
    "run_window_core_max_z_gap",
    "run_window_core_centrality",
    "run_window_core_count_layer_improvement",
    "run_window_core_absent_improvement",
    "run_window_core_drop_improvement",
    "run_window_core_count_margin",
]

EXTRA_NUMERIC_FEATURES = [
    "count_threshold",
    "barrel_gate_score",
    "layer_count_median",
    "chain_residues",
    "sheet_residues",
    "informative_slices",
]

SCORING_CATEGORICAL_FEATURES = [
    "confidence_basis",
    "barrel_wall_graph_variant",
    "barrel_wall_graph_visibility_regime",
    "barrel_wall_graph_selection_strategy",
    "axis_hypothesis_name",
]

EXTRA_CATEGORICAL_FEATURES = [
    "result",
    "result_stage",
    "barrel_gate_enabled",
    "barrel_gate_passed",
    "barrel_gate_result",
    "reason",
]

DEFAULT_NUMERIC_STATUSES = ("COUNTED", "LOW_CONFIDENCE")
DEFAULT_GOLD_EVIDENCE_LEVELS = ("gold",)
DEFAULT_GOLD_QC_STATUSES = ("pass",)


@dataclass(frozen=True)
class JoinedRecord:
    record_id: str
    pdb_id: str
    group_id: str
    auth_chain_id: str
    asym_id: str
    gold: int
    original_pred: int
    betlas: dict[str, str]


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"NA", "nan", "None", "\\0", "\0"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text in {"NA", "nan", "None", "\\0", "\0"}:
        return float("nan")
    if text == "True":
        return 1.0
    if text == "False":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def betlas_row_key(row: dict[str, str]) -> tuple[str, str]:
    pdb_id = Path(row.get("filename", "")).stem.upper()
    chain_id = str(row.get("chain", "")).strip()
    return pdb_id, chain_id


def load_gold_rows(
    path: Path,
    *,
    evidence_levels: set[str] | None = None,
    qc_statuses: set[str] | None = None,
) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(path)
        if (evidence_levels is None or row.get("evidence_level") in evidence_levels)
        and (qc_statuses is None or row.get("qc_status") in qc_statuses)
    ]
    rows.sort(key=lambda row: row.get("record_id", ""))
    return rows


def load_gold_pass_rows(path: Path) -> list[dict[str, str]]:
    return load_gold_rows(
        path,
        evidence_levels=set(DEFAULT_GOLD_EVIDENCE_LEVELS),
        qc_statuses=set(DEFAULT_GOLD_QC_STATUSES),
    )


def join_gold_to_betlas(
    *,
    gold_csv: Path,
    betlas_csv: Path,
    numeric_statuses: set[str],
    evidence_levels: set[str] | None = None,
    qc_statuses: set[str] | None = None,
    allow_opm_derived_gold: bool = False,
    allow_internal_stress_test: bool = False,
) -> tuple[list[JoinedRecord], Counter[str]]:
    betlas_rows = read_csv(betlas_csv)
    betlas_by_key = {betlas_row_key(row): row for row in betlas_rows}
    joined: list[JoinedRecord] = []
    skipped: Counter[str] = Counter()

    if evidence_levels is None:
        evidence_levels = set(DEFAULT_GOLD_EVIDENCE_LEVELS)
    if qc_statuses is None:
        qc_statuses = set(DEFAULT_GOLD_QC_STATUSES)

    gold_rows = load_gold_rows(
        gold_csv,
        evidence_levels=evidence_levels,
        qc_statuses=qc_statuses,
    )
    require_publication_gold_provenance(
        gold_rows,
        dataset=gold_csv,
        allow_opm_derived_gold=allow_opm_derived_gold,
        allow_internal_stress_test=allow_internal_stress_test,
    )

    for gold in gold_rows:
        gold_count = parse_int(gold.get("strand_count_final"))
        if gold_count is None:
            skipped["missing_gold_count"] += 1
            continue

        pdb_id = str(gold.get("pdb_id", "")).strip().upper()
        structure_id = str(gold.get("structure_id", "")).strip().upper()
        structure_key = structure_id or pdb_id
        group_id = (
            str(gold.get("cv_group_id", "")).strip()
            or str(gold.get("uniprot_acc", "")).strip()
            or structure_key
        )
        auth_chain_id = str(gold.get("auth_chain_id", "")).strip()
        asym_id = str(gold.get("asym_id", "")).strip()
        record_id = str(gold.get("record_id", "")).strip()
        betlas = (
            betlas_by_key.get((structure_key, auth_chain_id))
            or betlas_by_key.get((structure_key, asym_id))
            or betlas_by_key.get((pdb_id, auth_chain_id))
            or betlas_by_key.get((pdb_id, asym_id))
            or betlas_by_key.get(tuple(record_id.rsplit("_", 1)))  # type: ignore[arg-type]
        )
        if betlas is None:
            skipped["missing_betlas_row"] += 1
            continue

        result = betlas.get("result", "")
        if result not in numeric_statuses:
            skipped[f"non_numeric_result:{result or 'blank'}"] += 1
            continue

        original_pred = parse_int(betlas.get("strand_count"))
        if original_pred is None:
            skipped["missing_original_prediction"] += 1
            continue

        joined.append(
            JoinedRecord(
                record_id=record_id,
                pdb_id=structure_key,
                group_id=group_id.upper(),
                auth_chain_id=auth_chain_id,
                asym_id=asym_id,
                gold=int(gold_count),
                original_pred=int(original_pred),
                betlas=betlas,
            )
        )

    return joined, skipped


def ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return float("nan")
    return float(numerator / denominator)


def build_feature_frame(
    records: list[JoinedRecord],
    *,
    include_original_count: bool,
    feature_set: str,
) -> Any:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This script requires pandas. Install with `pip install -e .[ml]`."
        ) from exc

    rows: list[dict[str, object]] = []
    numeric_features = list(SCORING_NUMERIC_FEATURES)
    categorical_features = list(SCORING_CATEGORICAL_FEATURES)
    if feature_set == "all":
        numeric_features += EXTRA_NUMERIC_FEATURES
        categorical_features += EXTRA_CATEGORICAL_FEATURES
    if not include_original_count:
        numeric_features.remove("strand_count")

    for record in records:
        betlas = record.betlas
        feature_row: dict[str, object] = {
            name: parse_float(betlas.get(name)) for name in numeric_features
        }
        strand_count = parse_float(betlas.get("strand_count"))
        layer_q90 = parse_float(betlas.get("layer_count_q90"))
        layer_max = parse_float(betlas.get("layer_count_max"))
        support = parse_float(betlas.get("support_consensus_strands"))
        candidates = parse_float(betlas.get("candidate_strands"))
        persistent = parse_float(betlas.get("persistent_strands"))
        trajectory = parse_float(betlas.get("trajectory_candidate_strands"))
        graph_count = parse_float(betlas.get("barrel_wall_graph_count"))
        selected_mean = parse_float(betlas.get("selected_strand_support_mean"))
        selected_min = parse_float(betlas.get("selected_strand_support_min"))

        feature_row.update(
            {
                "support_over_layer_q90": ratio(support, layer_q90),
                "candidate_over_layer_q90": ratio(candidates, layer_q90),
                "persistent_over_layer_q90": ratio(persistent, layer_q90),
                "trajectory_over_layer_q90": ratio(trajectory, layer_q90),
                "graph_count_minus_layer_q90": graph_count - layer_q90,
                "selected_min_over_mean": ratio(selected_min, selected_mean),
            }
        )
        if include_original_count:
            feature_row.update(
                {
                    "original_minus_layer_q90": strand_count - layer_q90,
                    "original_minus_layer_max": strand_count - layer_max,
                    "original_minus_support_consensus": strand_count - support,
                    "original_minus_candidate": strand_count - candidates,
                    "graph_count_minus_original": graph_count - strand_count,
                }
            )
        if feature_set == "all":
            chain_residues = parse_float(betlas.get("chain_residues"))
            sheet_residues = parse_float(betlas.get("sheet_residues"))
            feature_row["sheet_residue_fraction"] = ratio(sheet_residues, chain_residues)

        for name in categorical_features:
            value = str(betlas.get(name, "")).strip()
            feature_row[name] = value if value else "missing"
        rows.append(feature_row)

    frame = pd.DataFrame(rows)
    numeric_columns = [
        column
        for column in frame.columns
        if column not in categorical_features and frame[column].notna().any()
    ]
    categorical_columns = [column for column in categorical_features if column in frame.columns]
    return frame[numeric_columns + categorical_columns]


def one_hot_encoder() -> Any:
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_estimator(model_name: str, *, seed: int, jobs: int) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    def numeric_columns(frame: Any) -> list[str]:
        return list(frame.select_dtypes(include=[np.number]).columns)

    def categorical_columns(frame: Any) -> list[str]:
        numeric = set(numeric_columns(frame))
        return [column for column in frame.columns if column not in numeric]

    scale_numeric = model_name.startswith("svm")

    def preprocessor(frame: Any) -> ColumnTransformer:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(numeric_steps), numeric_columns(frame)),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("onehot", one_hot_encoder()),
                        ]
                    ),
                    categorical_columns(frame),
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    if model_name == "random_forest_classifier":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=jobs,
        )
    elif model_name == "random_forest_regressor":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=jobs,
        )
    elif model_name == "svm_classifier":
        model = SVC(C=10.0, gamma="scale", class_weight="balanced")
    elif model_name == "svm_regressor":
        model = SVR(C=10.0, gamma="scale", epsilon=0.5)
    elif model_name == "lightgbm_classifier":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "LightGBM is not installed. Install with `pip install -e .[gbdt]` "
                "or `pip install lightgbm`."
            ) from exc

        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        model = LGBMClassifier(
            objective="multiclass",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=8,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=jobs,
            verbosity=-1,
        )
    elif model_name == "catboost_classifier":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "CatBoost is not installed. Install with `pip install -e .[gbdt]` "
                "or `pip install catboost`."
            ) from exc

        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=jobs,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return lambda frame: Pipeline([("preprocess", preprocessor(frame)), ("model", model)])


def cv_splits(y: np.ndarray, *, folds: int, seed: int, groups: list[str] | None) -> Any:
    if groups:
        from sklearn.model_selection import GroupKFold

        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except ImportError:  # pragma: no cover - older sklearn fallback
            StratifiedGroupKFold = None  # type: ignore[assignment]

        unique_groups = len(set(groups))
        if unique_groups < 2:
            raise SystemExit("Need at least two CV groups for grouped CV.")
        min_class_count = min(Counter(y).values())
        if StratifiedGroupKFold is not None and min_class_count >= 2:
            return StratifiedGroupKFold(
                n_splits=min(folds, unique_groups, min_class_count),
                shuffle=True,
                random_state=seed,
            ).split(np.zeros(len(y)), y, groups=np.asarray(groups))
        return GroupKFold(n_splits=min(folds, unique_groups)).split(
            np.zeros(len(y)), y, groups=np.asarray(groups)
        )

    min_class_count = min(Counter(y).values())
    if min_class_count >= 2:
        from sklearn.model_selection import StratifiedKFold

        return StratifiedKFold(
            n_splits=min(folds, min_class_count),
            shuffle=True,
            random_state=seed,
        ).split(np.zeros(len(y)), y)

    from sklearn.model_selection import KFold

    return KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y
    )


def integer_predictions(raw_predictions: np.ndarray, *, label_min: int, label_max: int) -> np.ndarray:
    rounded = np.rint(np.asarray(raw_predictions, dtype=float)).astype(int)
    return np.clip(rounded, label_min, label_max)


def cross_validated_predictions(
    estimator_factory: Any,
    x_frame: Any,
    y: np.ndarray,
    *,
    model_name: str,
    folds: int,
    seed: int,
    groups: list[str] | None,
) -> tuple[np.ndarray, list[dict[str, float]], int]:
    from sklearn.base import clone

    predictions = np.zeros(len(y), dtype=int)
    importances: list[dict[str, float]] = []
    split_count = 0
    label_min = int(np.min(y))
    label_max = int(np.max(y))

    for train_index, test_index in cv_splits(y, folds=folds, seed=seed, groups=groups):
        split_count += 1
        estimator = clone(estimator_factory(x_frame))
        estimator.fit(x_frame.iloc[train_index], y[train_index])
        raw_predictions = np.asarray(estimator.predict(x_frame.iloc[test_index])).ravel()
        if model_name.endswith("regressor"):
            predictions[test_index] = integer_predictions(
                raw_predictions,
                label_min=label_min,
                label_max=label_max,
            )
        else:
            predictions[test_index] = raw_predictions.astype(int)

        model = estimator.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            names = estimator.named_steps["preprocess"].get_feature_names_out()
            importances.append(
                {
                    str(name): float(value)
                    for name, value in zip(names, model.feature_importances_, strict=False)
                }
            )

    return predictions, importances, split_count


def metrics_row(model: str, gold: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    errors = np.asarray(pred, dtype=int) - np.asarray(gold, dtype=int)
    abs_errors = np.abs(errors)
    n = int(len(gold))
    exact = int(np.sum(errors == 0))
    within_2 = int(np.sum(abs_errors <= 2))
    return {
        "model": model,
        "n": n,
        "exact": exact,
        "exact_rate": exact / n if n else 0.0,
        "within_2": within_2,
        "within_2_rate": within_2 / n if n else 0.0,
        "mae": float(np.mean(abs_errors)) if n else 0.0,
        "rmse": sqrt(float(np.mean(errors.astype(float) ** 2))) if n else 0.0,
        "bias": float(np.mean(errors)) if n else 0.0,
        "over": int(np.sum(errors > 0)),
        "under": int(np.sum(errors < 0)),
    }


def bucket_rows(gold: np.ndarray, original: np.ndarray, ml: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for count in sorted(set(int(value) for value in gold)):
        mask = gold == count
        original_errors = original[mask] - gold[mask]
        ml_errors = ml[mask] - gold[mask]
        rows.append(
            {
                "gold": count,
                "n": int(np.sum(mask)),
                "original_exact": int(np.sum(original_errors == 0)),
                "original_mae": float(np.mean(np.abs(original_errors))),
                "original_bias": float(np.mean(original_errors)),
                "ml_exact": int(np.sum(ml_errors == 0)),
                "ml_mae": float(np.mean(np.abs(ml_errors))),
                "ml_bias": float(np.mean(ml_errors)),
            }
        )
    return rows


def averaged_feature_importances(importances: list[dict[str, float]]) -> list[dict[str, object]]:
    if not importances:
        return []
    feature_names = sorted({name for fold in importances for name in fold})
    rows = []
    for name in feature_names:
        values = [fold.get(name, 0.0) for fold in importances]
        rows.append(
            {
                "feature": name,
                "importance_mean": float(np.mean(values)),
                "importance_std": float(np.std(values)),
            }
        )
    rows.sort(key=lambda row: float(row["importance_mean"]), reverse=True)
    return rows


def write_markdown(
    path: Path,
    *,
    summary_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    skipped: Counter[str],
    folds: int,
    feature_set: str,
    include_original_count: bool,
    grouped_by_pdb: bool,
) -> None:
    lines = [
        "# ML Strand Count Baseline",
        "",
        f"- Prediction mode: out-of-fold cross-validation ({folds} folds).",
        f"- Feature set: `{feature_set}`.",
        f"- Original `strand_count` feature: `{str(include_original_count).lower()}`.",
        f"- Grouped CV: `{str(grouped_by_pdb).lower()}` "
        "(`cv_group_id`, else UniProt, else structure/PDB).",
        f"- Skipped gold/pass rows: {sum(skipped.values())}",
        "",
        "| model | n | exact | within +/-2 | MAE | bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {n} | {exact} ({exact_rate:.3f}) | "
            "{within_2} ({within_2_rate:.3f}) | {mae:.3f} | {bias:+.3f} |".format(
                **row
            )
        )
    if skipped:
        lines.extend(["", "## Skipped Rows", ""])
        for reason, count in skipped.most_common():
            lines.append(f"- `{reason}`: {count}")
    if feature_rows:
        lines.extend(["", "## Top RF Feature Importances", ""])
        lines.append("| feature | mean importance |")
        lines.append("| --- | ---: |")
        for row in feature_rows[:25]:
            lines.append(f"| `{row['feature']}` | {float(row['importance_mean']):.5f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betlas-csv", type=Path, required=True)
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--model",
        choices=[
            "random_forest_classifier",
            "random_forest_regressor",
            "svm_classifier",
            "svm_regressor",
            "lightgbm_classifier",
            "catboost_classifier",
        ],
        default="random_forest_classifier",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--feature-set",
        choices=["scoring", "all"],
        default="scoring",
        help="`scoring` uses only scoring/report-energy features; `all` also includes metadata such as chain size and gate fields.",
    )
    parser.add_argument(
        "--numeric-status",
        action="append",
        default=None,
        help="Readout result statuses with numeric strand counts. Repeatable.",
    )
    parser.add_argument(
        "--counted-only",
        action="store_true",
        help="Shortcut for `--numeric-status COUNTED`.",
    )
    parser.add_argument(
        "--gold-evidence-level",
        action="append",
        default=None,
        help="Gold dataset evidence levels to include. Repeatable; defaults to gold.",
    )
    parser.add_argument(
        "--gold-qc-status",
        action="append",
        default=None,
        help="Gold dataset QC statuses to include. Repeatable; defaults to pass.",
    )
    parser.set_defaults(include_original_count_feature=False, group_by_pdb=True)
    parser.add_argument(
        "--include-original-count-feature",
        action="store_true",
        help="Expose the direct geometry strand_count itself as a post-processing feature.",
    )
    parser.add_argument(
        "--no-original-count-feature",
        dest="include_original_count_feature",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--row-level-cv",
        dest="group_by_pdb",
        action="store_false",
        help=(
            "Use row-level folds instead of the publication default GroupKFold "
            "by cv_group_id, UniProt, or structure/PDB fallback."
        ),
    )
    parser.add_argument(
        "--group-by-pdb",
        dest="group_by_pdb",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-opm-derived-gold",
        action="store_true",
        help="Allow legacy OPM-derived gold/pass labels without manual chain-count provenance for compatibility checks.",
    )
    parser.add_argument(
        "--allow-internal-stress-test",
        action="store_true",
        help="Allow AFDB/non-release stress-test rows for compatibility checks.",
    )
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This script requires scikit-learn. Install with `pip install -e .[ml]`."
        ) from exc

    statuses = {"COUNTED"} if args.counted_only else set(args.numeric_status or DEFAULT_NUMERIC_STATUSES)
    evidence_levels = set(args.gold_evidence_level or DEFAULT_GOLD_EVIDENCE_LEVELS)
    qc_statuses = set(args.gold_qc_status or DEFAULT_GOLD_QC_STATUSES)
    try:
        records, skipped = join_gold_to_betlas(
            gold_csv=args.gold_csv.expanduser().resolve(),
            betlas_csv=args.betlas_csv.expanduser().resolve(),
            numeric_statuses=statuses,
            evidence_levels=evidence_levels,
            qc_statuses=qc_statuses,
            allow_opm_derived_gold=bool(args.allow_opm_derived_gold),
            allow_internal_stress_test=bool(args.allow_internal_stress_test),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if len(records) < 10:
        raise SystemExit(f"Need at least 10 joined numeric gold rows; found {len(records)}.")

    x_frame = build_feature_frame(
        records,
        include_original_count=bool(args.include_original_count_feature),
        feature_set=args.feature_set,
    )
    y = np.asarray([record.gold for record in records], dtype=int)
    original = np.asarray([record.original_pred for record in records], dtype=int)
    groups = [record.group_id for record in records] if args.group_by_pdb else None

    estimator_factory = make_estimator(args.model, seed=int(args.seed), jobs=int(args.jobs))
    ml_predictions, fold_importances, actual_folds = cross_validated_predictions(
        estimator_factory,
        x_frame,
        y,
        model_name=args.model,
        folds=max(2, int(args.folds)),
        seed=int(args.seed),
        groups=groups,
    )

    summary_rows = [
        metrics_row("internal_direct_geometry_count", y, original),
        metrics_row(f"ml_oof_{args.model}", y, ml_predictions),
    ]
    per_record_rows = []
    for record, ml_pred in zip(records, ml_predictions, strict=True):
        original_error = record.original_pred - record.gold
        ml_error = int(ml_pred) - record.gold
        betlas = record.betlas
        per_record_rows.append(
            {
                "record_id": record.record_id,
                "pdb_id": record.pdb_id,
                "cv_group_id": record.group_id,
                "auth_chain_id": record.auth_chain_id,
                "asym_id": record.asym_id,
                "gold": record.gold,
                "original_pred": record.original_pred,
                "ml_pred": int(ml_pred),
                "original_err": original_error,
                "ml_err": ml_error,
                "ml_improved": int(abs(ml_error) < abs(original_error)),
                "ml_worsened": int(abs(ml_error) > abs(original_error)),
                "result": betlas.get("result", ""),
                "confidence_basis": betlas.get("confidence_basis", ""),
                "confidence": betlas.get("confidence", ""),
                "count_decision_score": betlas.get("count_decision_score", ""),
            }
        )

    out_dir = args.out_dir.expanduser().resolve()
    summary_path = out_dir / "summary.csv"
    per_record_path = out_dir / "per_record_predictions.csv"
    bucket_path = out_dir / "bucket_metrics.csv"
    importance_path = out_dir / "feature_importance.csv"
    markdown_path = out_dir / "summary.md"

    write_csv(
        summary_path,
        summary_rows,
        [
            "model",
            "n",
            "exact",
            "exact_rate",
            "within_2",
            "within_2_rate",
            "mae",
            "rmse",
            "bias",
            "over",
            "under",
        ],
    )
    write_csv(
        per_record_path,
        per_record_rows,
        [
            "record_id",
            "pdb_id",
            "cv_group_id",
            "auth_chain_id",
            "asym_id",
            "gold",
            "original_pred",
            "ml_pred",
            "original_err",
            "ml_err",
            "ml_improved",
            "ml_worsened",
            "result",
            "confidence_basis",
            "confidence",
            "count_decision_score",
        ],
    )
    write_csv(
        bucket_path,
        bucket_rows(y, original, ml_predictions),
        [
            "gold",
            "n",
            "original_exact",
            "original_mae",
            "original_bias",
            "ml_exact",
            "ml_mae",
            "ml_bias",
        ],
    )
    feature_rows = averaged_feature_importances(fold_importances)
    write_csv(
        importance_path,
        feature_rows,
        ["feature", "importance_mean", "importance_std"],
    )
    write_markdown(
        markdown_path,
        summary_rows=summary_rows,
        feature_rows=feature_rows,
        skipped=skipped,
        folds=actual_folds,
        feature_set=args.feature_set,
        include_original_count=bool(args.include_original_count_feature),
        grouped_by_pdb=bool(args.group_by_pdb),
    )

    print(f"Joined numeric gold rows: {len(records)}")
    for row in summary_rows:
        print(
            "{model}: exact={exact}/{n} ({exact_rate:.3f}), "
            "within2={within_2}/{n} ({within_2_rate:.3f}), "
            "mae={mae:.3f}, bias={bias:+.3f}".format(**row)
        )
    print(f"Wrote {display_path(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
