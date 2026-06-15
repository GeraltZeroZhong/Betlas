from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..constants import FOLD_LABELS
from ..ml.ablations import (
    _groups as ablation_groups,
)
from ..ml.ablations import (
    _make_model as make_ablation_model,
)
from ..ml.ablations import (
    _without,
    build_feature_groups,
    load_ablation_config,
    raw_geometry_feature_columns,
)
from ..ml.benchmark import run_grouped_benchmark
from ..ml.splits import make_grouped_splits
from ..provenance import build_run_manifest, file_state, write_json

BOUNDARY_LABELS = ("beta_barrel", "beta_sandwich", "jelly_roll")
SANDWICH_JELLY_LABELS = ("beta_sandwich", "jelly_roll")
SCORE_LABELS = {
    "cz_jelly_rollness": "jelly_roll",
    "cz_sandwichness": "beta_sandwich",
    "cz_barrel_likeness": "beta_barrel",
}


@dataclass(frozen=True)
class PublicationEvidenceResult:
    out_dir: Path
    subset_paths: dict[str, Path]
    outputs: dict[str, Path]
    manifest_path: Path


def _cfg_get(config: Mapping[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _cfg_bool(config: Mapping[str, Any], dotted_key: str, default: bool) -> bool:
    value = _cfg_get(config, dotted_key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _stable_seed_offset(*items: Any) -> int:
    text = "\0".join(str(item) for item in items)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100000


def _read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def _num(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _parse_ok(features: pd.DataFrame) -> pd.Series:
    if "cz_parse_ok" not in features:
        return pd.Series(False, index=features.index)
    return pd.to_numeric(features["cz_parse_ok"], errors="coerce").fillna(0).astype(int) == 1


def _first_nonempty_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    key = pd.Series("", index=df.index, dtype=object)
    for column in columns:
        if column not in df:
            continue
        values = df[column].astype(str)
        key = key.mask(key.astype(str).str.len() == 0, values)
    return key.mask(key.astype(str).str.len() == 0, df["record_id"].astype(str))


def _composite_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = []
    for column in columns:
        if column in df:
            values.append(df[column].astype(str))
        else:
            values.append(pd.Series("", index=df.index, dtype=object))
    if not values:
        return df["record_id"].astype(str)
    key = values[0]
    for value in values[1:]:
        key = key + "|" + value
    return key


def _representative_subset(features: pd.DataFrame, spec: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = features[
        _parse_ok(features) & features["fold_label_final"].isin(FOLD_LABELS)
    ].copy()
    for column, expected in dict(spec.get("filter", {})).items():
        if column not in work:
            work = work.iloc[0:0].copy()
            break
        work = work[work[column].astype(str) == str(expected)].copy()

    group_columns = [str(column) for column in spec.get("group_columns", ["cath_s35_cluster_id"])]
    key_strategy = str(spec.get("key_strategy", "first_nonempty"))
    if key_strategy == "composite":
        work["__publication_group_key"] = _composite_key(work, group_columns)
    else:
        work["__publication_group_key"] = _first_nonempty_key(work, group_columns)

    work["__source_priority"] = (work.get("cath_s35_source", "") == "cath_s35").astype(int)
    work["__beta_residue_count"] = _num(work.get("cz_beta_residue_count", pd.Series(index=work.index)), -1.0)
    work["__residue_count"] = _num(work.get("cz_residue_count", pd.Series(index=work.index)), -1.0)
    work["__strand_count"] = _num(work.get("cz_beta_strand_count", pd.Series(index=work.index)), -1.0)
    work = work.sort_values(
        [
            "__publication_group_key",
            "__source_priority",
            "__beta_residue_count",
            "__residue_count",
            "__strand_count",
            "record_id",
        ],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    )
    subset = work.drop_duplicates("__publication_group_key", keep="first").copy()
    group_key = subset["__publication_group_key"].copy()
    subset = subset.drop(
        columns=[
            "__publication_group_key",
            "__source_priority",
            "__beta_residue_count",
            "__residue_count",
            "__strand_count",
        ],
        errors="ignore",
    )
    summary_rows = []
    for label in FOLD_LABELS:
        part = subset[subset["fold_label_final"] == label]
        summary_rows.append(
            {
                "subset": str(spec["name"]),
                "fold_label": label,
                "rows": int(len(part)),
                "unique_pdb": int(part["pdb_id"].nunique()) if "pdb_id" in part else 0,
                "unique_group_keys": int(group_key.loc[part.index].nunique()) if len(part) else 0,
            }
        )
    return subset, pd.DataFrame(summary_rows)


def _write_subset_manifest(
    path: Path,
    *,
    source_features: Path,
    spec: Mapping[str, Any],
    subset: pd.DataFrame,
) -> Path:
    manifest_path = path.with_suffix(f"{path.suffix}.manifest.json")
    write_json(
        manifest_path,
        build_run_manifest(
            command="betlas publication-evidence subset",
            parameters=dict(spec),
            inputs={"features_csv": source_features},
            outputs={"subset_csv": path},
            metrics={
                "rows": int(len(subset)),
                "unique_pdb": int(subset["pdb_id"].nunique()) if "pdb_id" in subset else 0,
                "unique_cath_s35_cluster_id": int(subset["cath_s35_cluster_id"].nunique())
                if "cath_s35_cluster_id" in subset
                else 0,
            },
        ),
    )
    return manifest_path


def create_nonredundant_subsets(
    features_csv: Path,
    subset_dir: Path,
    specs: list[Mapping[str, Any]],
) -> tuple[dict[str, Path], pd.DataFrame]:
    features = _read_csv(features_csv)
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_paths: dict[str, Path] = {}
    summaries: list[pd.DataFrame] = []
    for spec in specs:
        name = str(spec["name"])
        subset, summary = _representative_subset(features, spec)
        out = subset_dir / f"{name}.csv"
        subset.to_csv(out, index=False)
        _write_subset_manifest(out, source_features=features_csv, spec=spec, subset=subset)
        subset_paths[name] = out
        summaries.append(summary)
    return subset_paths, pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def _metric_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=list(FOLD_LABELS), average="macro", zero_division=0))


def _metric_boundary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(BOUNDARY_LABELS), average="macro", zero_division=0)
    )


def _metric_sandwich_jelly_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(SANDWICH_JELLY_LABELS),
            average="macro",
            zero_division=0,
        )
    )


def _label_indices(labels: tuple[str, ...]) -> list[int]:
    label_to_index = {label: index for index, label in enumerate(FOLD_LABELS)}
    return [label_to_index[label] for label in labels]


def _f1_from_confusion(confusion: np.ndarray, label_indices: list[int]) -> float:
    scores: list[float] = []
    for index in label_indices:
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        denom = (2.0 * tp) + fp + fn
        scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def _confusion_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(FOLD_LABELS)}
    true_index = np.asarray([label_to_index[str(label)] for label in y_true], dtype=int)
    pred_index = np.asarray([label_to_index[str(label)] for label in y_pred], dtype=int)
    unique_groups, group_index = np.unique(groups.astype(str), return_inverse=True)
    confusion = np.zeros((len(unique_groups), len(FOLD_LABELS), len(FOLD_LABELS)), dtype=np.int64)
    np.add.at(confusion, (group_index, true_index, pred_index), 1)
    return unique_groups, confusion


def _bootstrap_confusion_ci(
    confusion_by_group: np.ndarray,
    label_indices: list[int],
    *,
    iterations: int,
    seed: int,
    ci: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    group_count = confusion_by_group.shape[0]
    for _ in range(iterations):
        sampled = rng.integers(0, group_count, size=group_count)
        confusion = confusion_by_group[sampled].sum(axis=0)
        values.append(_f1_from_confusion(confusion, label_indices))
    alpha = (1.0 - ci) / 2.0
    return {
        "estimate": _f1_from_confusion(confusion_by_group.sum(axis=0), label_indices),
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1.0 - alpha)),
        "bootstrap_n": int(len(values)),
    }


def _bootstrap_confusion_delta_ci(
    baseline_by_group: np.ndarray,
    candidate_by_group: np.ndarray,
    label_indices: list[int],
    *,
    iterations: int,
    seed: int,
    ci: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    group_count = baseline_by_group.shape[0]
    for _ in range(iterations):
        sampled = rng.integers(0, group_count, size=group_count)
        baseline = baseline_by_group[sampled].sum(axis=0)
        candidate = candidate_by_group[sampled].sum(axis=0)
        values.append(
            _f1_from_confusion(candidate, label_indices) - _f1_from_confusion(baseline, label_indices)
        )
    baseline_full = baseline_by_group.sum(axis=0)
    candidate_full = candidate_by_group.sum(axis=0)
    alpha = (1.0 - ci) / 2.0
    return {
        "baseline_estimate": _f1_from_confusion(baseline_full, label_indices),
        "ablation_estimate": _f1_from_confusion(candidate_full, label_indices),
        "delta_estimate": _f1_from_confusion(candidate_full, label_indices)
        - _f1_from_confusion(baseline_full, label_indices),
        "delta_ci_low": float(np.quantile(values, alpha)),
        "delta_ci_high": float(np.quantile(values, 1.0 - alpha)),
        "bootstrap_n": int(len(values)),
    }


def _group_index(groups: np.ndarray) -> list[np.ndarray]:
    frame = pd.DataFrame({"group": groups, "row": np.arange(len(groups))})
    return [part["row"].to_numpy(dtype=int) for _, part in frame.groupby("group", sort=False)]


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int,
    seed: int,
    ci: float,
) -> dict[str, float | int]:
    members = _group_index(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in sampled])
        values.append(metric(y_true[idx], y_pred[idx]))
    alpha = (1.0 - ci) / 2.0
    return {
        "estimate": metric(y_true, y_pred),
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1.0 - alpha)),
        "bootstrap_n": int(len(values)),
    }


def _bootstrap_score_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int,
    seed: int,
    ci: float,
) -> dict[str, float | int]:
    members = _group_index(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric(y_true[idx], score[idx])))
    alpha = (1.0 - ci) / 2.0
    return {
        "estimate": float(metric(y_true, score)),
        "ci_low": float(np.quantile(values, alpha)) if values else float("nan"),
        "ci_high": float(np.quantile(values, 1.0 - alpha)) if values else float("nan"),
        "bootstrap_n": int(len(values)),
    }


def benchmark_confidence_intervals(
    oof_csv: Path,
    out_csv: Path,
    *,
    iterations: int,
    seed: int,
    ci: float,
    dataset: str,
) -> pd.DataFrame:
    oof = _read_csv(oof_csv)
    rows: list[dict[str, Any]] = []
    for model, part in oof.groupby("model", sort=True):
        y_true = part["true_label"].astype(str).to_numpy()
        y_pred = part["pred_label"].astype(str).to_numpy()
        groups = part["group"].astype(str).to_numpy()
        _, confusion = _confusion_by_group(y_true, y_pred, groups)
        for metric_name, labels in [
            ("macro_f1", FOLD_LABELS),
            ("boundary_macro_f1", BOUNDARY_LABELS),
        ]:
            stats = _bootstrap_confusion_ci(
                confusion,
                _label_indices(tuple(labels)),
                iterations=iterations,
                seed=seed + _stable_seed_offset(dataset, model, metric_name),
                ci=ci,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric_name,
                    "unit": "group",
                    **stats,
                }
            )
    result = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result


def continuous_score_confidence_intervals(
    topology_csv: Path,
    features_csv: Path,
    out_csv: Path,
    *,
    iterations: int,
    seed: int,
    ci: float,
) -> pd.DataFrame:
    topology = _read_csv(topology_csv)
    feature_groups = _read_csv(
        features_csv,
        usecols=lambda column: column
        in {"record_id", "cath_s35_cluster_id", "cath_homology_code", "pdb_id"},
    )
    merged = topology.merge(feature_groups, on="record_id", how="left", suffixes=("", "_feature"))
    merged["__group"] = _first_nonempty_key(
        merged,
        ["cath_s35_cluster_id", "cath_homology_code", "pdb_id"],
    )
    rows: list[dict[str, Any]] = []
    for score_column, positive_label in SCORE_LABELS.items():
        if score_column not in merged:
            continue
        valid = pd.to_numeric(merged[score_column], errors="coerce").notna()
        part = merged[valid].copy()
        y_true = (part["fold_label_final"].astype(str) == positive_label).astype(int).to_numpy()
        score = pd.to_numeric(part[score_column], errors="coerce").to_numpy(dtype=float)
        groups = part["__group"].astype(str).to_numpy()
        for metric_name, metric in [
            ("roc_auc", roc_auc_score),
            ("average_precision", average_precision_score),
        ]:
            stats = _bootstrap_score_ci(
                y_true,
                score,
                groups,
                metric,
                iterations=iterations,
                seed=seed + _stable_seed_offset(score_column, metric_name),
                ci=ci,
            )
            rows.append(
                {
                    "score": score_column.replace("cz_", ""),
                    "positive_label": positive_label,
                    "metric": metric_name,
                    "unit": "group",
                    "n": int(len(part)),
                    "positive_n": int(y_true.sum()),
                    **stats,
                }
            )
    result = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result


def _resolve_ablation_features(
    spec: Mapping[str, Any],
    feature_cols: list[str],
    feature_groups: Mapping[str, list[str]],
) -> tuple[list[str], list[str]]:
    ablation_type = str(spec["ablation_type"])
    name = str(spec["name"])
    if ablation_type == "baseline":
        return [], list(feature_cols)
    if ablation_type.startswith("drop_group"):
        removed = sorted(
            {
                feature
                for group in name.split("+")
                for feature in feature_groups.get(group, [])
            }
        )
        return removed, _without(feature_cols, set(removed))
    if ablation_type == "drop_feature_single":
        return [name], _without(feature_cols, {name})
    raise ValueError(f"unsupported selected ablation: {ablation_type}:{name}")


def selected_ablation_oof_and_ci(
    features_csv: Path,
    out_dir: Path,
    *,
    specs: list[Mapping[str, Any]],
    config_path: str | Path,
    iterations: int,
    seed: int,
    ci: float,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_ablation_config(config_path)
    df = _read_csv(features_csv)
    df = df[df["fold_label_final"].isin(FOLD_LABELS)].copy()
    df = df[_parse_ok(df)].copy()
    feature_cols = raw_geometry_feature_columns(df)
    x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    encoder = LabelEncoder()
    encoder.fit(sorted(FOLD_LABELS))
    y = encoder.transform(df["fold_label_final"])
    groups = ablation_groups(df)
    splits, split_strategy, _ = make_grouped_splits(
        df[feature_cols],
        y,
        groups,
        n_splits=n_splits,
        random_state=seed,
    )
    feature_groups = build_feature_groups(feature_cols)
    resolved_specs = [{"ablation_type": "baseline", "name": "all_raw_geometry"}, *specs]

    oof_parts: list[pd.DataFrame] = []
    predictions: dict[str, np.ndarray] = {}
    for spec in tqdm(resolved_specs, desc="Selected ablation OOF"):
        removed, kept = _resolve_ablation_features(spec, feature_cols, feature_groups)
        pred = np.empty_like(y)
        for train_idx, test_idx in splits:
            model = make_ablation_model(seed, config)
            model.fit(x_all.loc[:, kept].iloc[train_idx], y[train_idx])
            pred[test_idx] = model.predict(x_all.loc[:, kept].iloc[test_idx])
        key = f"{spec['ablation_type']}:{spec['name']}"
        predictions[key] = encoder.inverse_transform(pred)
        oof_parts.append(
            pd.DataFrame(
                {
                    "ablation_type": str(spec["ablation_type"]),
                    "name": str(spec["name"]),
                    "record_id": df["record_id"].to_numpy(),
                    "group": groups,
                    "true_label": df["fold_label_final"].to_numpy(),
                    "pred_label": predictions[key],
                    "removed_feature_count": len(removed),
                    "kept_feature_count": len(kept),
                }
            )
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_path = out_dir / "selected_ablation_oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    y_true = df["fold_label_final"].astype(str).to_numpy()
    baseline_pred = predictions["baseline:all_raw_geometry"]
    baseline_groups, baseline_confusion = _confusion_by_group(y_true, baseline_pred, groups)
    rows: list[dict[str, Any]] = []
    for spec in resolved_specs[1:]:
        key = f"{spec['ablation_type']}:{spec['name']}"
        candidate_pred = predictions[key]
        candidate_groups, candidate_confusion = _confusion_by_group(y_true, candidate_pred, groups)
        if not np.array_equal(baseline_groups, candidate_groups):
            raise ValueError("baseline and candidate ablation group order differs")
        for metric_name, labels in [
            ("macro_f1", FOLD_LABELS),
            ("boundary_macro_f1", BOUNDARY_LABELS),
            ("sandwich_jelly_macro_f1", SANDWICH_JELLY_LABELS),
        ]:
            stats = _bootstrap_confusion_delta_ci(
                baseline_confusion,
                candidate_confusion,
                _label_indices(tuple(labels)),
                iterations=iterations,
                seed=seed + _stable_seed_offset(key, metric_name),
                ci=ci,
            )
            rows.append(
                {
                    "ablation_type": str(spec["ablation_type"]),
                    "name": str(spec["name"]),
                    "metric": metric_name,
                    "unit": "group",
                    **stats,
                }
            )
    delta_ci = pd.DataFrame(rows)
    delta_ci.to_csv(out_dir / "ablation_delta_confidence_intervals.csv", index=False)
    write_json(
        out_dir / "selected_ablation_manifest.json",
        build_run_manifest(
            command="betlas publication-evidence selected-ablation-oof",
            parameters={
                "n_splits": int(n_splits),
                "random_state": int(seed),
                "split_strategy": split_strategy,
                "iterations": int(iterations),
                "ci": float(ci),
                "config_path": str(config_path),
                "selected_ablations": [dict(spec) for spec in specs],
            },
            inputs={"features_csv": features_csv, "ablation_config": config_path},
            outputs={
                "selected_ablation_oof_predictions": out_dir / "selected_ablation_oof_predictions.csv",
                "ablation_delta_confidence_intervals": out_dir
                / "ablation_delta_confidence_intervals.csv",
            },
            metrics={"rows": int(len(df)), "raw_feature_count": int(len(feature_cols))},
        ),
    )
    return oof, delta_ci


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _case_row(row: pd.Series, case_type: str, focus: str) -> dict[str, Any]:
    pdb_id = str(row.get("pdb_id", "")).lower()
    return {
        "case_type": case_type,
        "record_id": row.get("record_id", ""),
        "pdb_id": pdb_id,
        "chain_id": row.get("chain_id", ""),
        "domain_id": row.get("domain_id", ""),
        "residue_ranges": row.get("residue_ranges", ""),
        "fold_label_final": row.get("fold_label_final", ""),
        "cath_code": row.get("cath_code", ""),
        "cath_topology_code": row.get("cath_topology_code", ""),
        "cath_name": row.get("cath_name", ""),
        "probability_top1_label": row.get("cz_probability_top1_label", ""),
        "probability_top2_label": row.get("cz_probability_top2_label", ""),
        "probability_top1": row.get("cz_probability_top1", ""),
        "probability_top2_margin": row.get("cz_probability_top2_margin", ""),
        "ambiguity_score": row.get("cz_topology_ambiguity_score", ""),
        "jelly_rollness": row.get("cz_jelly_rollness", ""),
        "sandwichness": row.get("cz_sandwichness", ""),
        "barrel_likeness": row.get("cz_barrel_likeness", ""),
        "mixed_topology_score": row.get("cz_mixed_topology_score", ""),
        "mixed_topology_types": row.get("cz_mixed_topology_types", ""),
        "ambiguity_reasons": row.get("cz_topology_ambiguity_reasons", ""),
        "review_focus": focus,
        "rcsb_url": f"https://www.rcsb.org/structure/{pdb_id.upper()}" if pdb_id else "",
    }


def _pick_first(part: pd.DataFrame, used: set[str]) -> pd.Series | None:
    for _, row in part.iterrows():
        record_id = str(row.get("record_id", ""))
        pdb_id = str(row.get("pdb_id", ""))
        if record_id and record_id not in used and pdb_id not in used:
            used.add(record_id)
            used.add(pdb_id)
            return row
    if len(part):
        row = part.iloc[0]
        used.add(str(row.get("record_id", "")))
        return row
    return None


def select_case_studies(
    topology_csv: Path,
    features_csv: Path,
    out_csv: Path,
    doc_path: Path,
) -> pd.DataFrame:
    topology = _read_csv(topology_csv)
    feature_cols = {
        "record_id",
        "source_mmcif_path",
        "source_mmcif_sha256",
        "source_mmcif_exists",
        "cz_beta_strand_count",
        "cz_sheet_count",
        "cz_sheet_pair_top2_fraction",
        "cz_axis_best_angular_coverage",
    }
    features = _read_csv(features_csv, usecols=lambda column: column in feature_cols)
    df = topology.merge(features, on="record_id", how="left")
    for column in [
        "cz_probability_top1",
        "cz_probability_top2_margin",
        "cz_topology_ambiguity_score",
        "cz_jelly_rollness",
        "cz_sandwichness",
        "cz_barrel_likeness",
        "cz_mixed_topology_score",
    ]:
        df[column] = pd.to_numeric(df.get(column, np.nan), errors="coerce")
    df["cz_mixed_topology_flag"] = pd.to_numeric(
        df.get("cz_mixed_topology_flag", 0), errors="coerce"
    ).fillna(0)

    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    relevance = {
        "beta_barrel": "cz_barrel_likeness",
        "jelly_roll": "cz_jelly_rollness",
        "beta_sandwich": "cz_sandwichness",
    }
    for label in FOLD_LABELS:
        part = df[
            (df["fold_label_final"] == label)
            & (df["cz_probability_top1_label"] == label)
            & (df["cz_mixed_topology_flag"] == 0)
        ].copy()
        rel_col = relevance.get(label)
        part["__case_score"] = part["cz_probability_top1"].fillna(0) - part[
            "cz_topology_ambiguity_score"
        ].fillna(0)
        if rel_col:
            part["__case_score"] += 0.25 * part[rel_col].fillna(0)
        part = part.sort_values(["__case_score", "record_id"], ascending=[False, True])
        row = _pick_first(part, used)
        if row is not None:
            rows.append(
                _case_row(
                    row,
                    f"canonical_{label}",
                    f"Canonical {label} example: correct primary model call, low ambiguity, and representative geometry.",
                )
            )

    sj = df[
        df["fold_label_final"].isin(SANDWICH_JELLY_LABELS)
        & (
            (
                df["cz_probability_top1_label"].isin(SANDWICH_JELLY_LABELS)
                & df["cz_probability_top2_label"].isin(SANDWICH_JELLY_LABELS)
            )
            | (df.get("cz_jelly_sandwich_overlap", "").astype(str) != "")
        )
    ].copy()
    sj["__case_score"] = (
        sj["cz_topology_ambiguity_score"].fillna(0)
        + sj["cz_jelly_rollness"].fillna(0)
        + sj["cz_sandwichness"].fillna(0)
        - sj["cz_probability_top2_margin"].fillna(1)
    )
    row = _pick_first(sj.sort_values(["__case_score", "record_id"], ascending=[False, True]), used)
    if row is not None:
        rows.append(
            _case_row(
                row,
                "sandwich_jelly_ambiguous",
                "Boundary case: both sandwichness and jelly-rollness are high, with low model margin.",
            )
        )

    off = df[
        (df["fold_label_final"] != "beta_barrel")
        & (
            (df["cz_barrel_likeness"] >= 0.60)
            | df.get("cz_mixed_topology_types", "").astype(str).str.contains("barrel_like", na=False)
        )
    ].copy()
    off["__case_score"] = (
        off["cz_barrel_likeness"].fillna(0) + off["cz_topology_ambiguity_score"].fillna(0)
    )
    row = _pick_first(off.sort_values(["__case_score", "record_id"], ascending=[False, True]), used)
    if row is not None:
        rows.append(
            _case_row(
                row,
                "barrel_like_off_target",
                "Off-target closure case: barrel-like geometry evidence appears outside a beta_barrel label.",
            )
        )

    mixed = df[df["cz_mixed_topology_flag"] == 1].copy()
    mixed["__case_score"] = (
        mixed["cz_mixed_topology_score"].fillna(0) + mixed["cz_topology_ambiguity_score"].fillna(0)
    )
    row = _pick_first(mixed.sort_values(["__case_score", "record_id"], ascending=[False, True]), used)
    if row is not None:
        rows.append(
            _case_row(
                row,
                "mixed_topology",
                "Hybrid-domain case: local secondary grammar is strong enough for manual topology audit.",
            )
        )

    cases = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out_csv, index=False)
    _write_case_study_doc(cases, doc_path)
    return cases


def select_manual_boundary_panel(topology_csv: Path, out_csv: Path, max_rows: int = 30) -> pd.DataFrame:
    df = _read_csv(topology_csv)
    for column in [
        "cz_topology_ambiguity_score",
        "cz_probability_top2_margin",
        "cz_jelly_rollness",
        "cz_sandwichness",
        "cz_barrel_likeness",
        "cz_mixed_topology_score",
    ]:
        df[column] = pd.to_numeric(df.get(column, np.nan), errors="coerce")
    df["cz_mixed_topology_flag"] = pd.to_numeric(
        df.get("cz_mixed_topology_flag", 0), errors="coerce"
    ).fillna(0)

    panels: list[pd.DataFrame] = []
    sj = df[df["fold_label_final"].isin(["beta_sandwich", "jelly_roll"])].copy()
    sj["review_category"] = "sandwich_jelly_boundary"
    sj["review_score"] = (
        sj["cz_topology_ambiguity_score"].fillna(0)
        + sj["cz_jelly_rollness"].fillna(0)
        + sj["cz_sandwichness"].fillna(0)
        - sj["cz_probability_top2_margin"].fillna(1)
    )
    panels.append(sj.nlargest(max(8, max_rows // 3), "review_score"))

    barrel = df[(df["fold_label_final"] != "beta_barrel") & (df["cz_barrel_likeness"] >= 0.55)].copy()
    barrel["review_category"] = "barrel_like_off_target"
    barrel["review_score"] = (
        barrel["cz_topology_ambiguity_score"].fillna(0) + barrel["cz_barrel_likeness"].fillna(0)
    )
    panels.append(barrel.nlargest(max(8, max_rows // 3), "review_score"))

    mixed = df[df["cz_mixed_topology_flag"] == 1].copy()
    mixed["review_category"] = "mixed_topology"
    mixed["review_score"] = (
        mixed["cz_topology_ambiguity_score"].fillna(0) + mixed["cz_mixed_topology_score"].fillna(0)
    )
    panels.append(mixed.nlargest(max(8, max_rows // 3), "review_score"))

    panel = pd.concat(panels, ignore_index=True).drop_duplicates("record_id").nlargest(
        max_rows, "review_score"
    )
    keep = [
        "review_category",
        "review_score",
        "record_id",
        "pdb_id",
        "chain_id",
        "domain_id",
        "residue_ranges",
        "fold_label_final",
        "cath_code",
        "cath_topology_code",
        "cath_name",
        "cz_probability_top1_label",
        "cz_probability_top2_label",
        "cz_probability_top1",
        "cz_probability_top2_margin",
        "cz_topology_ambiguity_score",
        "cz_topology_ambiguity_reasons",
        "cz_jelly_rollness",
        "cz_sandwichness",
        "cz_barrel_likeness",
        "cz_mixed_topology_score",
        "cz_mixed_topology_types",
    ]
    panel = panel[[column for column in keep if column in panel]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_csv, index=False)
    return panel


def _write_case_study_doc(cases: pd.DataFrame, doc_path: Path) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Case Study Entries",
        "",
        "This document lists fixed Betlas case-study entries for visualization.",
        "Entries are selected from external CATH-derived labels plus Betlas readouts;",
        "Betlas readouts are not used as ground-truth labels.",
        "",
        "Use the structure files recorded by the run manifest, or the linked RCSB",
        "entries, to render structures consistently.",
        "",
        "## Selected Entries",
        "",
        "| slot | record | PDB | chain | residues | label | CATH | model top1/top2 | ambiguity | JR | SW | barrel | focus |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in cases.itertuples(index=False):
        top = f"{row.probability_top1_label}/{row.probability_top2_label}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.case_type),
                    str(row.record_id),
                    str(row.pdb_id).upper(),
                    str(row.chain_id),
                    str(row.residue_ranges),
                    str(row.fold_label_final),
                    str(row.cath_code),
                    top,
                    _format_float(row.ambiguity_score),
                    _format_float(row.jelly_rollness),
                    _format_float(row.sandwichness),
                    _format_float(row.barrel_likeness),
                    str(row.review_focus),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Rendering Notes",
            "",
            "- Canonical slots cover the seven target fold classes.",
            "- Boundary slots cover sandwich/jelly ambiguity, barrel-like off-target closure, and mixed topology.",
            "- Keep residue ranges/domain ids visible in figure labels so reviewers can reproduce the exact domain crop.",
            "- For ambiguous cases, show the CATH label and Betlas continuous readouts together instead of forcing a hard relabel.",
            "",
            "## RCSB Links",
            "",
        ]
    )
    for row in cases.itertuples(index=False):
        lines.append(f"- `{row.record_id}`: {row.rcsb_url}")
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_nonredundant_benchmarks(
    subset_paths: Mapping[str, Path],
    out_root: Path,
    *,
    config_path: Path,
    n_splits: int,
    seed: int,
    reuse_existing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts: list[pd.DataFrame] = []
    per_class_parts: list[pd.DataFrame] = []
    for name, path in subset_paths.items():
        run_dir = out_root / name
        metrics_path = run_dir / "metrics_summary.csv"
        per_class_path = run_dir / "per_class_metrics.csv"
        if reuse_existing and metrics_path.exists() and per_class_path.exists():
            summary = pd.read_csv(metrics_path)
            per_class = pd.read_csv(per_class_path)
        else:
            result = run_grouped_benchmark(
                path,
                run_dir,
                n_splits=n_splits,
                random_state=seed,
                config_path=config_path,
            )
            summary = result.metrics_summary.copy()
            per_class = result.per_class_metrics.copy()
        summary.insert(0, "dataset", name)
        summary_parts.append(summary)
        per_class.insert(0, "dataset", name)
        per_class_parts.append(per_class)
    return pd.concat(summary_parts, ignore_index=True), pd.concat(per_class_parts, ignore_index=True)


def run_publication_evidence(config_path: str | Path) -> PublicationEvidenceResult:
    config = dict(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    out_dir = _as_path(_cfg_get(config, "outputs.out_dir", "runs/publication_evidence_full"))
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = _as_path(_cfg_get(config, "outputs.subset_dir", "data/processed/publication_evidence"))

    features_csv = _as_path(_cfg_get(config, "inputs.features", "data/processed/betlas_full_features.csv"))
    topology_csv = _as_path(
        _cfg_get(config, "inputs.topology", "runs/readouts/topology_diagnostics_full/topology_diagnostics.csv")
    )
    full_oof_csv = _as_path(
        _cfg_get(config, "inputs.benchmark_oof", "runs/betlas_full_benchmark/oof_predictions.csv")
    )
    benchmark_config = _as_path(_cfg_get(config, "benchmark.config", "configs/benchmark/full_scale.yaml"))
    ablation_config = _as_path(_cfg_get(config, "ablation_oof.config", "configs/benchmark/full_scale_ablation.yaml"))

    bootstrap_iterations = int(_cfg_get(config, "bootstrap.iterations", 1000))
    seed = int(_cfg_get(config, "bootstrap.seed", 13))
    ci = float(_cfg_get(config, "bootstrap.ci", 0.95))

    subset_specs = list(_cfg_get(config, "nonredundant_subsets", []))
    subset_paths, subset_summary = create_nonredundant_subsets(features_csv, subset_dir, subset_specs)
    outputs: dict[str, Path] = {}
    subset_summary_path = out_dir / "nonredundant_subset_summary.csv"
    subset_summary.to_csv(subset_summary_path, index=False)
    outputs["nonredundant_subset_summary"] = subset_summary_path

    if _cfg_bool(config, "benchmark.enabled", True):
        summary, per_class = _run_nonredundant_benchmarks(
            subset_paths,
            out_dir / "nonredundant_benchmarks",
            config_path=benchmark_config,
            n_splits=int(_cfg_get(config, "benchmark.splits", 5)),
            seed=int(_cfg_get(config, "benchmark.seed", seed)),
            reuse_existing=_cfg_bool(config, "benchmark.reuse_existing", False),
        )
        summary_path = out_dir / "nonredundant_benchmark_summary.csv"
        per_class_path = out_dir / "nonredundant_per_class_metrics.csv"
        summary.to_csv(summary_path, index=False)
        per_class.to_csv(per_class_path, index=False)
        outputs["nonredundant_benchmark_summary"] = summary_path
        outputs["nonredundant_per_class_metrics"] = per_class_path

    ci_parts = [
        benchmark_confidence_intervals(
            full_oof_csv,
            out_dir / "confidence_intervals_benchmark_full.csv",
            iterations=bootstrap_iterations,
            seed=seed,
            ci=ci,
            dataset="full_cath",
        )
    ]
    for name in subset_paths:
        oof_path = out_dir / "nonredundant_benchmarks" / name / "oof_predictions.csv"
        if oof_path.exists():
            ci_parts.append(
                benchmark_confidence_intervals(
                    oof_path,
                    out_dir / f"confidence_intervals_benchmark_{name}.csv",
                    iterations=bootstrap_iterations,
                    seed=seed,
                    ci=ci,
                    dataset=name,
                )
            )
    benchmark_ci = pd.concat(ci_parts, ignore_index=True)
    benchmark_ci_path = out_dir / "confidence_intervals_benchmark.csv"
    benchmark_ci.to_csv(benchmark_ci_path, index=False)
    outputs["confidence_intervals_benchmark"] = benchmark_ci_path

    continuous_ci = continuous_score_confidence_intervals(
        topology_csv,
        features_csv,
        out_dir / "confidence_intervals_continuous_scores.csv",
        iterations=bootstrap_iterations,
        seed=seed,
        ci=ci,
    )
    outputs["confidence_intervals_continuous_scores"] = out_dir / "confidence_intervals_continuous_scores.csv"

    if _cfg_bool(config, "ablation_oof.enabled", True):
        _, ablation_ci = selected_ablation_oof_and_ci(
            features_csv,
            out_dir / "selected_ablation_oof",
            specs=list(_cfg_get(config, "ablation_oof.selected", [])),
            config_path=ablation_config,
            iterations=bootstrap_iterations,
            seed=seed,
            ci=ci,
            n_splits=int(_cfg_get(config, "ablation_oof.splits", 5)),
        )
        outputs["selected_ablation_oof_predictions"] = out_dir / "selected_ablation_oof" / "selected_ablation_oof_predictions.csv"
        outputs["ablation_delta_confidence_intervals"] = out_dir / "selected_ablation_oof" / "ablation_delta_confidence_intervals.csv"
    else:
        ablation_ci = pd.DataFrame()

    case_csv = out_dir / "figure_ready_case_studies.csv"
    case_doc = _as_path(_cfg_get(config, "outputs.case_study_doc", "outputs/case_study_entries.md"))
    select_case_studies(topology_csv, features_csv, case_csv, case_doc)
    outputs["figure_ready_case_studies"] = case_csv
    outputs["figure_ready_case_study_doc"] = case_doc

    panel_csv = out_dir / "manual_boundary_review_panel.csv"
    select_manual_boundary_panel(
        topology_csv,
        panel_csv,
        max_rows=int(_cfg_get(config, "manual_boundary_panel.max_rows", 30)),
    )
    outputs["manual_boundary_review_panel"] = panel_csv

    manifest_path = out_dir / "publication_evidence_manifest.json"
    write_json(
        manifest_path,
        build_run_manifest(
            command="betlas publication-evidence",
            parameters=dict(config),
            inputs={
                "config": config_path,
                "features_csv": features_csv,
                "topology_csv": topology_csv,
                "full_benchmark_oof": full_oof_csv,
                "benchmark_config": benchmark_config,
                "ablation_config": ablation_config,
            },
            outputs=outputs,
            metrics={
                "nonredundant_subsets": int(len(subset_paths)),
                "benchmark_ci_rows": int(len(benchmark_ci)),
                "continuous_ci_rows": int(len(continuous_ci)),
                "ablation_ci_rows": int(len(ablation_ci)),
            },
            extra={
                "subset_files": {name: file_state(path) for name, path in subset_paths.items()},
                "bootstrap": {"iterations": bootstrap_iterations, "seed": seed, "ci": ci},
            },
        ),
    )
    outputs["manifest"] = manifest_path
    return PublicationEvidenceResult(
        out_dir=out_dir,
        subset_paths=dict(subset_paths),
        outputs=outputs,
        manifest_path=manifest_path,
    )
