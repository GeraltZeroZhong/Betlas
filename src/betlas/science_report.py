from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .constants import FOLD_LABELS
from .provenance import build_run_manifest, write_json

CATH_METADATA_COLUMNS = (
    "record_id",
    "pdb_id",
    "chain_id",
    "domain_id",
    "residue_ranges",
    "fold_label_final",
    "evidence_level",
    "label_source_primary",
    "label_source_supporting",
    "cath_code",
    "cath_architecture_code",
    "cath_topology_code",
    "cath_homology_code",
    "cath_s35_cluster_id",
    "cath_s35_source",
    "cath_status",
    "cath_name",
    "assembly_id",
    "model_id",
    "qc_status",
    "allowed_for_publication_benchmark",
    "discovered_by_betlas",
    "label_conflict_notes",
)

EXPECTED_TEXT_DIAGNOSTIC_COLUMNS = {
    "cz_axis_best_name",
    "cz_error",
    "cz_fold_scores_json",
    "cz_top_fold",
    "cz_warnings",
    "source_mmcif_path",
    "source_mmcif_sha256",
}


@dataclass(frozen=True)
class ScienceReportResult:
    out_dir: Path
    manifest_path: Path
    outputs: dict[str, Path]


def _read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _num(series: pd.Series) -> pd.Series:
    values: list[float] = []
    for value in series:
        if value in {"", None}:
            values.append(float("nan"))
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            values.append(float("nan"))
            continue
        values.append(numeric if math.isfinite(numeric) else float("nan"))
    return pd.Series(values, index=series.index, dtype=float)


def _quantile(series: pd.Series, q: float) -> float:
    values = _num(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def repair_feature_table_metadata(
    features_csv: str | Path,
    labels_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Align label/CATH metadata and clamp tiny numerical artifacts in an existing feature table."""
    features = _read_table(features_csv)
    labels = _read_table(labels_csv)
    label_by_record = labels.set_index("record_id", drop=False)
    feature_index = features["record_id"].astype(str)
    for column in CATH_METADATA_COLUMNS:
        if column not in label_by_record.columns:
            continue
        values = feature_index.map(label_by_record[column]).fillna("")
        if column in features.columns:
            features[column] = values.where(values.astype(str) != "", features[column])
        else:
            features[column] = values

    entropy_columns = [column for column in features.columns if column.endswith("_entropy")]
    for column in entropy_columns:
        values = _num(features[column])
        tiny_negative = values.between(-1e-8, 0.0, inclusive="left")
        if tiny_negative.any():
            features.loc[tiny_negative, column] = "0.0"

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return features


def dataset_composition(features: pd.DataFrame) -> pd.DataFrame:
    parse_ok = _num(features.get("cz_parse_ok", pd.Series(index=features.index, dtype=str))).fillna(0)
    rows: list[dict[str, Any]] = []
    for label in FOLD_LABELS:
        part = features[features["fold_label_final"] == label]
        part_parse = parse_ok.loc[part.index] if len(part) else pd.Series(dtype=float)
        rows.append(
            {
                "fold_label": label,
                "n": int(len(part)),
                "parse_ok_n": int((part_parse == 1).sum()),
                "parse_ok_rate": float((part_parse == 1).mean()) if len(part_parse) else 0.0,
                "unique_pdb": int(part["pdb_id"].nunique()) if "pdb_id" in part else 0,
                "unique_cath_s35": int(part["cath_s35_cluster_id"].nunique())
                if "cath_s35_cluster_id" in part
                else 0,
                "unique_cath_homology": int(part["cath_homology_code"].nunique())
                if "cath_homology_code" in part
                else 0,
                "unique_cath_topology": int(part["cath_topology_code"].nunique())
                if "cath_topology_code" in part
                else 0,
                "s35_direct_rows": int((part.get("cath_s35_source", pd.Series(dtype=str)) == "cath_s35").sum())
                if "cath_s35_source" in part
                else 0,
                "s35_homology_fallback_rows": int(
                    (part.get("cath_s35_source", pd.Series(dtype=str)) == "homology_fallback").sum()
                )
                if "cath_s35_source" in part
                else 0,
                "max_rows_per_pdb": int(part["pdb_id"].value_counts().max()) if "pdb_id" in part and len(part) else 0,
                "max_rows_per_cath_s35": int(part["cath_s35_cluster_id"].value_counts().max())
                if "cath_s35_cluster_id" in part and len(part)
                else 0,
                "beta_residue_fraction_median": _quantile(part.get("cz_beta_residue_fraction", pd.Series(dtype=str)), 0.5),
                "beta_strand_count_median": _quantile(part.get("cz_beta_strand_count", pd.Series(dtype=str)), 0.5),
                "helix_count_median": _quantile(part.get("cz_helix_count", pd.Series(dtype=str)), 0.5),
            }
        )
    return pd.DataFrame(rows)


def label_rule_mapping_summary(labels: pd.DataFrame) -> pd.DataFrame:
    selectors = [
        (1, "tim_like_beta_alpha_barrel", "cath_topology_code in {3.20.20, 3.20.110}"),
        (2, "jelly_roll", "cath_topology_code == 2.60.120"),
        (3, "beta_propeller", "cath_architecture_code in {2.105, 2.110, 2.115, 2.120, 2.130, 2.140}"),
        (4, "beta_prism", "cath_architecture_code in {2.90, 2.100}"),
        (5, "beta_solenoid", "cath_architecture_code in {2.150, 2.160} or name regex beta-solenoid"),
        (6, "beta_barrel", "cath_architecture_code == 2.40"),
        (7, "beta_sandwich", "cath_architecture_code in {2.60, 2.70, 2.102}"),
    ]
    rows: list[dict[str, Any]] = []
    for priority, label, selector in selectors:
        part = labels[labels["fold_label_final"] == label] if "fold_label_final" in labels else pd.DataFrame()
        rows.append(
            {
                "priority": priority,
                "fold_label": label,
                "selector": selector,
                "n_rows": int(len(part)),
                "unique_topology": int(part["cath_topology_code"].nunique()) if "cath_topology_code" in part else 0,
                "unique_homology": int(part["cath_homology_code"].nunique()) if "cath_homology_code" in part else 0,
                "evidence_level": ";".join(sorted(part["evidence_level"].astype(str).unique()))
                if "evidence_level" in part and len(part)
                else "",
            }
        )
    return pd.DataFrame(rows)


def redundancy_audit(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, column in [
        ("pdb_id", "pdb_id"),
        ("cath_s35_cluster_id", "cath_s35_cluster_id"),
        ("cath_homology_code", "cath_homology_code"),
        ("cath_topology_code", "cath_topology_code"),
    ]:
        if column not in features:
            continue
        counts = features[column].astype(str).value_counts()
        rows.append(
            {
                "grouping": group_name,
                "unique_groups": int(len(counts)),
                "max_rows_per_group": int(counts.max()) if len(counts) else 0,
                "median_rows_per_group": float(counts.median()) if len(counts) else 0.0,
                "q90_rows_per_group": float(counts.quantile(0.90)) if len(counts) else 0.0,
                "q99_rows_per_group": float(counts.quantile(0.99)) if len(counts) else 0.0,
            }
        )
    if "cath_s35_source" in features:
        for source, part in features.groupby("cath_s35_source", dropna=False):
            rows.append(
                {
                    "grouping": f"cath_s35_source:{source}",
                    "unique_groups": int(part["cath_s35_cluster_id"].nunique())
                    if "cath_s35_cluster_id" in part
                    else 0,
                    "max_rows_per_group": int(len(part)),
                    "median_rows_per_group": float("nan"),
                    "q90_rows_per_group": float("nan"),
                    "q99_rows_per_group": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def cath_lineage_composition(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["fold_label_final", "cath_architecture_code", "cath_topology_code"]
    present = [key for key in keys if key in features.columns]
    if len(present) != len(keys):
        return pd.DataFrame(rows)
    grouped = features.groupby(keys, dropna=False).size().reset_index(name="n")
    return grouped.sort_values(["fold_label_final", "n"], ascending=[True, False])


def _is_bounded_feature(column: str) -> bool:
    if column.startswith("prob_") or column.startswith("cz_rule_score_"):
        return False
    bounded_tokens = (
        "fraction",
        "coverage",
        "entropy",
        "occupancy",
        "dominance",
        "balance",
        "flatness",
        "planarity",
        "density",
        "alignment",
        "abs_dot",
    )
    return any(token in column for token in bounded_tokens)


def feature_quality(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_columns = [
        column
        for column in features.columns
        if column.startswith("cz_") or column.startswith("source_mmcif_")
    ]
    candidate_columns = [column for column in candidate_columns if column not in EXPECTED_TEXT_DIAGNOSTIC_COLUMNS]
    for column in candidate_columns:
        values = _num(features[column]).replace([np.inf, -np.inf], np.nan)
        finite = values.notna()
        bounded = _is_bounded_feature(column)
        out_of_range = 0
        if bounded:
            out_of_range = int(((values < -1e-8) | (values > 1.0 + 1e-8)).sum())
        rows.append(
            {
                "feature": column,
                "n": int(len(values)),
                "finite_n": int(finite.sum()),
                "missing_or_non_numeric_n": int((~finite).sum()),
                "min": float(values.min(skipna=True)) if values.notna().any() else np.nan,
                "q25": _quantile(values, 0.25),
                "median": _quantile(values, 0.5),
                "q75": _quantile(values, 0.75),
                "max": float(values.max(skipna=True)) if values.notna().any() else np.nan,
                "bounded_0_1_expected": bool(bounded),
                "bounded_0_1_violations": out_of_range,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["bounded_0_1_violations", "missing_or_non_numeric_n", "feature"],
        ascending=[False, False, True],
    )


def source_feature_provenance_quality(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = int(len(features))
    sha = features.get("source_mmcif_sha256", pd.Series("", index=features.index, dtype=str)).astype(str)
    exists = _num(features.get("source_mmcif_exists", pd.Series(index=features.index, dtype=str))).fillna(0)
    size = _num(features.get("source_mmcif_size", pd.Series(index=features.index, dtype=str))).fillna(0)
    rows.append(
        {
            "check": "row_level_mmcif_exists",
            "n": n,
            "pass_n": int((exists == 1).sum()),
            "pass_rate": float((exists == 1).mean()) if n else 0.0,
        }
    )
    rows.append(
        {
            "check": "row_level_mmcif_size_positive",
            "n": n,
            "pass_n": int((size > 0).sum()),
            "pass_rate": float((size > 0).mean()) if n else 0.0,
        }
    )
    rows.append(
        {
            "check": "row_level_mmcif_sha256_present",
            "n": n,
            "pass_n": int((sha.str.len() == 64).sum()),
            "pass_rate": float((sha.str.len() == 64).mean()) if n else 0.0,
        }
    )
    rows.append(
        {
            "check": "row_level_mmcif_sha256_hex",
            "n": n,
            "pass_n": int(sha.str.fullmatch(r"[0-9a-f]{64}").fillna(False).sum()),
            "pass_rate": float(sha.str.fullmatch(r"[0-9a-f]{64}").fillna(False).mean()) if n else 0.0,
        }
    )
    return pd.DataFrame(rows)


def parse_failure_summary(features: pd.DataFrame) -> pd.DataFrame:
    parse_ok = _num(features.get("cz_parse_ok", pd.Series(index=features.index, dtype=str))).fillna(0)
    failed = features[parse_ok != 1].copy()
    if failed.empty:
        return pd.DataFrame(
            columns=[
                "fold_label",
                "cz_error",
                "n",
                "example_record_id",
                "example_pdb_id",
            ]
        )
    failed["cz_error"] = failed.get("cz_error", "").astype(str).replace("", "unspecified_parse_failure")
    grouped = (
        failed.groupby(["fold_label_final", "cz_error"], dropna=False)
        .agg(
            n=("record_id", "size"),
            example_record_id=("record_id", "first"),
            example_pdb_id=("pdb_id", "first"),
        )
        .reset_index()
        .rename(columns={"fold_label_final": "fold_label"})
        .sort_values(["n", "fold_label"], ascending=[False, True])
    )
    return grouped


def grammar_label_agreement(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "cz_top_fold" not in features.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for label in FOLD_LABELS:
        part = features[features["fold_label_final"] == label]
        if not len(part):
            continue
        top = part["cz_top_fold"].astype(str)
        rows.append(
            {
                "fold_label": label,
                "n": int(len(part)),
                "grammar_top1_match_n": int((top == label).sum()),
                "grammar_top1_match_rate": float((top == label).mean()),
                "rule_margin_median": _quantile(part.get("cz_rule_margin", pd.Series(dtype=str)), 0.5),
                "rule_margin_q10": _quantile(part.get("cz_rule_margin", pd.Series(dtype=str)), 0.1),
            }
        )
    confusion = pd.crosstab(
        features["fold_label_final"],
        features["cz_top_fold"],
        dropna=False,
    ).reset_index()
    return pd.DataFrame(rows), confusion


def benchmark_boundary_metrics(benchmark_dir: str | Path) -> pd.DataFrame:
    predictions = Path(benchmark_dir) / "oof_predictions.csv"
    if not predictions.exists():
        return pd.DataFrame()
    df = pd.read_csv(predictions, dtype=str, keep_default_na=False)
    if not len(df):
        return pd.DataFrame()
    focus = {"beta_barrel", "beta_sandwich", "jelly_roll"}
    rows: list[dict[str, Any]] = []
    for model, part in df.groupby("model", dropna=False):
        focus_part = part[part["true_label"].isin(focus)]
        if not len(focus_part):
            continue
        sandwich_jelly = focus_part[focus_part["true_label"].isin({"beta_sandwich", "jelly_roll"})]
        rows.append(
            {
                "model": model,
                "boundary_family_n": int(len(focus_part)),
                "boundary_family_accuracy": float((focus_part["true_label"] == focus_part["pred_label"]).mean()),
                "sandwich_jelly_n": int(len(sandwich_jelly)),
                "sandwich_jelly_accuracy": float((sandwich_jelly["true_label"] == sandwich_jelly["pred_label"]).mean())
                if len(sandwich_jelly)
                else float("nan"),
                "top2_margin_median": _quantile(focus_part.get("top2_margin", pd.Series(dtype=str)), 0.5),
                "top2_margin_q10": _quantile(focus_part.get("top2_margin", pd.Series(dtype=str)), 0.1),
            }
        )
    return pd.DataFrame(rows).sort_values("boundary_family_accuracy", ascending=False)


def model_performance_summary(benchmark_dir: str | Path | None) -> pd.DataFrame:
    if benchmark_dir is None:
        return pd.DataFrame()
    path = Path(benchmark_dir) / "metrics_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "macro_f1" in df:
        df["_macro_f1"] = _num(df["macro_f1"])
        df = df.sort_values("_macro_f1", ascending=False).drop(columns=["_macro_f1"])
    return df


def _best_model_name(benchmark_dir: str | Path | None) -> str:
    summary = model_performance_summary(benchmark_dir)
    if summary.empty or "model" not in summary:
        return ""
    return str(summary.iloc[0]["model"])


def calibration_summary(benchmark_dir: str | Path | None) -> pd.DataFrame:
    if benchmark_dir is None:
        return pd.DataFrame()
    path = Path(benchmark_dir) / "oof_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    label_prob_columns = {label: f"prob_{label}" for label in FOLD_LABELS}
    for model, part in df.groupby("model", dropna=False):
        if not all(column in part for column in label_prob_columns.values()):
            continue
        prob_matrix = np.vstack([_num(part[column]).fillna(0).to_numpy() for column in label_prob_columns.values()]).T
        true_indices = np.array([FOLD_LABELS.index(label) if label in FOLD_LABELS else -1 for label in part["true_label"]])
        valid = true_indices >= 0
        if not valid.any():
            continue
        prob_matrix = prob_matrix[valid]
        true_indices = true_indices[valid]
        pred = np.asarray(part.loc[valid, "pred_label"])
        true = np.asarray(part.loc[valid, "true_label"])
        confidence_column = "top1_probability" if "top1_probability" in part else "pred_probability"
        if confidence_column in part:
            confidence = _num(part.loc[valid, confidence_column]).to_numpy()
            fallback_confidence = prob_matrix.max(axis=1)
            confidence = np.where(np.isfinite(confidence), confidence, fallback_confidence)
        else:
            confidence = prob_matrix.max(axis=1)
        correct = (pred == true).astype(float)
        ece = 0.0
        for low in np.linspace(0.0, 0.9, 10):
            high = low + 0.1
            mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
            if mask.any():
                ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        one_hot = np.zeros_like(prob_matrix)
        one_hot[np.arange(len(true_indices)), true_indices] = 1.0
        clipped = np.clip(prob_matrix[np.arange(len(true_indices)), true_indices], 1e-12, 1.0)
        rows.append(
            {
                "model": model,
                "n": int(len(true_indices)),
                "accuracy": float(correct.mean()),
                "mean_top1_probability": float(confidence.mean()),
                "ece_10": ece,
                "multiclass_brier": float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1))),
                "negative_log_likelihood": float(-np.mean(np.log(clipped))),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def boundary_lineage_error_summary(
    features: pd.DataFrame,
    benchmark_dir: str | Path | None,
    *,
    model: str | None = None,
    min_n: int = 3,
) -> pd.DataFrame:
    if benchmark_dir is None:
        return pd.DataFrame()
    path = Path(benchmark_dir) / "oof_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    selected_model = model or _best_model_name(benchmark_dir)
    if not selected_model:
        return pd.DataFrame()
    predictions = pd.read_csv(path, dtype=str, keep_default_na=False)
    predictions = predictions[predictions["model"] == selected_model]
    focus = {"beta_barrel", "beta_sandwich", "jelly_roll"}
    predictions = predictions[predictions["true_label"].isin(focus)]
    metadata = features[
        [
            column
            for column in [
                "record_id",
                "pdb_id",
                "domain_id",
                "fold_label_final",
                "cath_architecture_code",
                "cath_topology_code",
                "cath_homology_code",
                "cath_name",
            ]
            if column in features.columns
        ]
    ]
    joined = predictions.merge(metadata, on="record_id", how="left")
    rows: list[dict[str, Any]] = []
    group_cols = ["true_label", "cath_topology_code", "cath_name"]
    for keys, part in joined.groupby(group_cols, dropna=False):
        if len(part) < min_n:
            continue
        pred_counts = part["pred_label"].value_counts().to_dict()
        example = part.iloc[0]
        example_pdb_id = (
            str(example.get("pdb_id", ""))
            or str(example.get("pdb_id_x", ""))
            or str(example.get("pdb_id_y", ""))
        )
        rows.append(
            {
                "model": selected_model,
                "true_label": keys[0],
                "cath_topology_code": keys[1],
                "cath_name": keys[2],
                "n": int(len(part)),
                "accuracy": float((part["true_label"] == part["pred_label"]).mean()),
                "top_predicted_label": str(max(pred_counts, key=pred_counts.get)) if pred_counts else "",
                "top_predicted_n": int(max(pred_counts.values())) if pred_counts else 0,
                "example_record_id": str(example.get("record_id", "")),
                "example_pdb_id": example_pdb_id,
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "n"], ascending=[True, False])


def topology_diagnostics_summary(topology_csv: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if topology_csv is None or not Path(topology_csv).exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(topology_csv, dtype=str, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for label in FOLD_LABELS:
        part = df[df["fold_label_final"] == label] if "fold_label_final" in df else pd.DataFrame()
        if not len(part):
            continue
        rows.append(
            {
                "fold_label": label,
                "n": int(len(part)),
                "ambiguity_mean": float(_num(part["cz_topology_ambiguity_score"]).mean()),
                "ambiguity_q90": _quantile(part["cz_topology_ambiguity_score"], 0.9),
                "boundary_region_rate": float(_num(part["cz_boundary_region_flag"]).fillna(0).mean()),
                "mixed_topology_rate": float(_num(part["cz_mixed_topology_flag"]).fillna(0).mean()),
                "jelly_rollness_median": _quantile(part["cz_jelly_rollness"], 0.5),
                "sandwichness_median": _quantile(part["cz_sandwichness"], 0.5),
                "barrel_likeness_median": _quantile(part["cz_barrel_likeness"], 0.5),
            }
        )
    type_rows: list[dict[str, Any]] = []
    if "cz_mixed_topology_types" in df:
        counts: dict[str, int] = {}
        for value in df["cz_mixed_topology_types"]:
            for item in str(value).split(";"):
                item = item.strip()
                if item:
                    counts[item] = counts.get(item, 0) + 1
        type_rows = [{"mixed_topology_type": key, "n": value} for key, value in sorted(counts.items())]
    return pd.DataFrame(rows), pd.DataFrame(type_rows)


def continuous_score_distribution(topology_csv: str | Path | None) -> pd.DataFrame:
    if topology_csv is None or not Path(topology_csv).exists():
        return pd.DataFrame()
    df = pd.read_csv(topology_csv, dtype=str, keep_default_na=False)
    score_columns = ["cz_jelly_rollness", "cz_sandwichness", "cz_barrel_likeness"]
    rows: list[dict[str, Any]] = []
    for label in FOLD_LABELS:
        part = df[df["fold_label_final"] == label] if "fold_label_final" in df else pd.DataFrame()
        for column in score_columns:
            if column not in part:
                continue
            rows.append(
                {
                    "fold_label": label,
                    "score": column.replace("cz_", ""),
                    "n": int(len(part)),
                    "q10": _quantile(part[column], 0.1),
                    "q25": _quantile(part[column], 0.25),
                    "median": _quantile(part[column], 0.5),
                    "q75": _quantile(part[column], 0.75),
                    "q90": _quantile(part[column], 0.9),
                }
            )
    return pd.DataFrame(rows)


def continuous_score_label_separation(topology_csv: str | Path | None) -> pd.DataFrame:
    if topology_csv is None or not Path(topology_csv).exists():
        return pd.DataFrame()
    df = pd.read_csv(topology_csv, dtype=str, keep_default_na=False)
    tasks = [
        ("jelly_rollness", "cz_jelly_rollness", "jelly_roll"),
        ("sandwichness", "cz_sandwichness", "beta_sandwich"),
        ("barrel_likeness", "cz_barrel_likeness", "beta_barrel"),
    ]
    rows: list[dict[str, Any]] = []
    for task_name, score_col, positive_label in tasks:
        if score_col not in df or "fold_label_final" not in df:
            continue
        y_true = (df["fold_label_final"] == positive_label).astype(int).to_numpy()
        scores = _num(df[score_col]).fillna(0.0).to_numpy()
        if len(np.unique(y_true)) < 2:
            roc_auc = float("nan")
            pr_auc = float("nan")
        else:
            roc_auc = float(roc_auc_score(y_true, scores))
            pr_auc = float(average_precision_score(y_true, scores))
        off_target = df["fold_label_final"] != positive_label
        rows.append(
            {
                "score": task_name,
                "positive_label": positive_label,
                "n": int(len(df)),
                "positive_n": int(y_true.sum()),
                "roc_auc": roc_auc,
                "average_precision": pr_auc,
                "off_target_high_rate_0_6": float(np.mean(scores[off_target.to_numpy()] >= 0.6))
                if bool(off_target.any())
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def high_ambiguity_records(topology_csv: str | Path | None, *, max_rows: int = 50) -> pd.DataFrame:
    if topology_csv is None or not Path(topology_csv).exists():
        return pd.DataFrame()
    df = pd.read_csv(topology_csv, dtype=str, keep_default_na=False)
    if "cz_topology_ambiguity_score" not in df:
        return pd.DataFrame()
    df["_ambiguity"] = _num(df["cz_topology_ambiguity_score"])
    columns = [
        column
        for column in [
            "record_id",
            "pdb_id",
            "domain_id",
            "fold_label_final",
            "cz_probability_top1_label",
            "cz_probability_top2_label",
            "cz_topology_ambiguity_score",
            "cz_boundary_region_flag",
            "cz_mixed_topology_flag",
            "cz_mixed_topology_types",
            "cz_jelly_rollness",
            "cz_sandwichness",
            "cz_barrel_likeness",
            "cz_probability_top2_margin",
            "cz_probability_entropy",
            "cz_rule_label_conflict",
            "cz_topology_ambiguity_reasons",
        ]
        if column in df.columns
    ]
    return df.sort_values("_ambiguity", ascending=False).head(max_rows)[columns]


def source_snapshot_summary(source_snapshot: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source_snapshot is None or not Path(source_snapshot).exists():
        return pd.DataFrame(), pd.DataFrame()
    payload = json.loads(Path(source_snapshot).read_text(encoding="utf-8"))
    extra = payload.get("extra", {})
    cath = extra.get("cath", {}).get("files", {})
    mmcif = extra.get("mmcif", {})
    cath_rows = [
        {
            "source": "CATH",
            "artifact": key,
            "url": value.get("url", ""),
            "exists": value.get("exists", False),
            "size": value.get("size", 0),
            "sha256": value.get("sha256", ""),
            "data_line_count": value.get("data_line_count", 0),
        }
        for key, value in cath.items()
    ]
    mmcif_rows = [
        {
            "source": "RCSB mmCIF",
            "requested_count": mmcif.get("requested_count", 0),
            "present_count": mmcif.get("present_count", 0),
            "missing_count": mmcif.get("missing_count", 0),
        }
    ]
    return pd.DataFrame(cath_rows), pd.DataFrame(mmcif_rows)


def staves_readouts_full_summary(staves_readouts_csv: str | Path | None) -> pd.DataFrame:
    if staves_readouts_csv is None or not Path(staves_readouts_csv).exists():
        return pd.DataFrame()
    df = pd.read_csv(staves_readouts_csv, dtype=str, keep_default_na=False)
    if not len(df):
        return pd.DataFrame()
    summary = df.groupby(["strand_count_final", "strand_count_source"], dropna=False).size().reset_index(name="n")
    summary = summary.rename(columns={"strand_count_final": "reference_strand_count"})
    summary["_reference_strand_count_sort"] = pd.to_numeric(
        summary["reference_strand_count"], errors="coerce"
    )
    return (
        summary.sort_values(["_reference_strand_count_sort", "strand_count_source", "reference_strand_count"])
        .drop(columns=["_reference_strand_count_sort"])
        .reset_index(drop=True)
    )


def _markdown_table(table: pd.DataFrame, *, max_rows: int = 20) -> list[str]:
    part = table.head(max_rows).astype(str)
    columns = list(part.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in part.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return lines


def write_markdown_summary(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Betlas Scientific Audit Report", ""]
    for name, table in tables.items():
        lines.extend([f"## {name.replace('_', ' ').title()}", ""])
        if table.empty:
            lines.extend(["No rows.", ""])
            continue
        lines.extend(_markdown_table(table))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_science_report(
    *,
    features_csv: str | Path,
    labels_csv: str | Path,
    out_dir: str | Path,
    topology_csv: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
    source_snapshot: str | Path | None = None,
    staves_readouts_csv: str | Path | None = None,
) -> ScienceReportResult:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _read_table(features_csv)
    labels = _read_table(labels_csv)

    tables: dict[str, pd.DataFrame] = {
        "dataset_composition": dataset_composition(features),
        "label_rule_mapping": label_rule_mapping_summary(labels),
        "redundancy_audit": redundancy_audit(features),
        "cath_lineage_composition": cath_lineage_composition(features),
        "feature_quality": feature_quality(features),
        "source_feature_provenance_quality": source_feature_provenance_quality(features),
        "parse_failure_summary": parse_failure_summary(features),
    }
    grammar_summary, grammar_confusion = grammar_label_agreement(features)
    tables["grammar_label_agreement"] = grammar_summary
    tables["grammar_confusion"] = grammar_confusion
    if benchmark_dir is not None:
        tables["model_performance_summary"] = model_performance_summary(benchmark_dir)
        tables["benchmark_boundary_metrics"] = benchmark_boundary_metrics(benchmark_dir)
        tables["calibration_summary"] = calibration_summary(benchmark_dir)
        tables["boundary_lineage_error_summary"] = boundary_lineage_error_summary(features, benchmark_dir)
    topology_summary, mixed_types = topology_diagnostics_summary(topology_csv)
    tables["topology_diagnostics_summary"] = topology_summary
    tables["mixed_topology_type_counts"] = mixed_types
    tables["continuous_score_distribution"] = continuous_score_distribution(topology_csv)
    tables["continuous_score_label_separation"] = continuous_score_label_separation(topology_csv)
    tables["high_ambiguity_records"] = high_ambiguity_records(topology_csv)
    cath_summary, mmcif_summary = source_snapshot_summary(source_snapshot)
    tables["source_cath_summary"] = cath_summary
    tables["source_mmcif_summary"] = mmcif_summary
    tables["staves_readouts_full_summary"] = staves_readouts_full_summary(staves_readouts_csv)

    outputs: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = path
    markdown_path = output_dir / "scientific_audit_report.md"
    write_markdown_summary(markdown_path, tables)
    outputs["markdown_summary"] = markdown_path

    manifest_path = output_dir / "scientific_audit_manifest.json"
    manifest = build_run_manifest(
        command="betlas science-report",
        parameters={
            "features_csv": str(features_csv),
            "labels_csv": str(labels_csv),
            "topology_csv": str(topology_csv or ""),
            "benchmark_dir": str(benchmark_dir or ""),
            "source_snapshot": str(source_snapshot or ""),
            "staves_readouts_csv": str(staves_readouts_csv or ""),
            "out_dir": str(out_dir),
        },
        inputs={
            "features_csv": features_csv,
            "labels_csv": labels_csv,
            **({"topology_csv": topology_csv} if topology_csv else {}),
            **({"source_snapshot": source_snapshot} if source_snapshot else {}),
            **({"staves_readouts_csv": staves_readouts_csv} if staves_readouts_csv else {}),
        },
        outputs=outputs,
        metrics={
            "feature_rows": int(len(features)),
            "label_rows": int(len(labels)),
            "feature_quality_flags": int(tables["feature_quality"]["bounded_0_1_violations"].sum())
            if "bounded_0_1_violations" in tables["feature_quality"]
            else 0,
        },
        extra={name: {"rows": int(len(table))} for name, table in tables.items()},
    )
    write_json(manifest_path, manifest)
    return ScienceReportResult(out_dir=output_dir, manifest_path=manifest_path, outputs=outputs)
