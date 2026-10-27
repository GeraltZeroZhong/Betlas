"""Retrospective model overlays on an independently prepared geometry table.

Only source rows explicitly marked ``role=test`` supply predictions.  Raw saved
probabilities are summarized without refitting, calibration, prior correction,
or cross-model subtraction. Geometry overlap remains a label-conditioned input.
No reported association or disagreement is a causal decomposition.
"""

from __future__ import annotations

import json
from io import BytesIO
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_bytes, read_csv

TASK_LABELS = {
    "architecture": "cath_architecture_code",
    "topology": "cath_topology_code",
}
SYSTEMS = (
    "betlas--logistic--inverse",
    "betlas--logistic--none",
    "betlas--random_forest--none",
)
H_COLUMN = "cath_homology_code"
MARGIN_THRESHOLD = 0.2


def _geometry_table(source: pd.DataFrame | str | Path, overlap_column: str) -> pd.DataFrame:
    frame = source.copy() if isinstance(source, pd.DataFrame) else read_csv(source, dtype=str, keep_default_na=False)
    for short, canonical in (("A", TASK_LABELS["architecture"]), ("T", TASK_LABELS["topology"]), ("H", H_COLUMN)):
        if canonical not in frame and short in frame:
            frame = frame.rename(columns={short: canonical})
    required = {"record_id", H_COLUMN, *TASK_LABELS.values()}
    if missing := required - set(frame):
        raise ValueError(f"Geometry table lacks columns: {sorted(missing)}")
    for column in required:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Geometry identity field {column} contains missing values")
        frame[column] = frame[column].astype(str)
    if frame.record_id.duplicated().any():
        raise ValueError("Geometry table must have one row per record_id")
    if "role" in frame:
        if "geometry_role" in frame:
            raise ValueError("Use either role or geometry_role in the geometry table")
        frame = frame.rename(columns={"role": "geometry_role"})
    if overlap_column not in frame:
        frame[overlap_column] = pd.NA

    def overlap(value: Any) -> Any:
        if value is None or pd.isna(value) or str(value).strip().lower() in {"", "na", "nan", "none", "<na>"}:
            return pd.NA
        key = str(value).strip().lower()
        if key in {"true", "1", "1.0"}:
            return True
        if key in {"false", "0", "0.0"}:
            return False
        raise ValueError(f"{overlap_column} must be boolean or NA, not an overlap fraction")

    frame["geometry_label_overlap"] = pd.array(frame[overlap_column].map(overlap), dtype="boolean")
    return frame.reset_index(drop=True)


def _assessment_root(source: str | Path) -> Path:
    root = Path(source)
    for candidate in (root / "classification" / "assessment", root / "assessment", root):
        if all((candidate / task).is_dir() for task in TASK_LABELS):
            return candidate
    raise FileNotFoundError("Expected the saved architecture and topology assessment directories")


def _probability_summary(probability: np.ndarray, classes: np.ndarray, target: str) -> dict[str, Any]:
    """Summarize a valid simplex vector; do not renormalize saved probabilities."""
    order = np.argsort(-probability, kind="stable")
    first, second = int(order[0]), int(order[1])
    positive = probability > 0
    supported = target in classes
    result: dict[str, Any] = {
        "probability_available": True,
        "target_supported": supported,
        "probability_top_label": str(classes[first]),
        "probability_second_label": str(classes[second]),
        "top_probability": float(probability[first]),
        "second_probability": float(probability[second]),
        "top2_margin": float(probability[first] - probability[second]),
        "normalized_entropy": float(-np.sum(probability[positive] * np.log(probability[positive])) / np.log(len(classes))),
        "low_margin": bool(probability[first] - probability[second] <= MARGIN_THRESHOLD),
        "probability_sum": float(probability.sum()),
    }
    if supported:
        index = int(np.flatnonzero(classes == target)[0])
        result["target_probability"] = float(probability[index])
        result["log_loss"] = float(-np.log(max(probability[index], 1e-15)))
        residual = probability.copy()
        residual[index] -= 1
        result["brier"] = float(np.sum(residual ** 2))
    return result


def _topology_masses(probability: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    b = np.char.startswith(classes, "2.40.")
    j = classes == "2.60.120"
    other_s = np.char.startswith(classes, "2.60.") & ~j
    masks = {"B": b, "J": j, "S_other": other_s, "other": ~(b | j | other_s)}
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        result[f"mass_{name}"] = float(probability[mask].sum())
        result[f"support_{name}_classes"] = count
        result[f"support_{name}_status"] = "represented_in_fold_space" if count else "absent_from_fold_space"
    total = result["mass_B"] + result["mass_J"] + result["mass_S_other"]
    result["three_group_mass"] = total
    for name in ("B", "J", "S_other"):
        # Never turn an unsupported group into an apparent conditional probability.
        result[f"conditional_{name}"] = (
            result[f"mass_{name}"] / total if total > 0 and result[f"support_{name}_classes"] > 0 else np.nan
        )
    return result


def _load_system(
    directory: Path,
    geometry: pd.DataFrame,
    task: str,
    system: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    prediction_path = directory / "predictions.csv"
    predictions = read_csv(prediction_path, dtype=str, keep_default_na=False)
    required = {"record_id", "role", "split_id", "target_label", "prediction", "prediction_status"}
    if missing := required - set(predictions):
        raise ValueError(f"{prediction_path} lacks columns: {sorted(missing)}")
    test = predictions[predictions.role.eq("test")].copy()
    if test.record_id.duplicated().any():
        raise ValueError(f"{directory}: a record has multiple test predictions")
    lookup = test.set_index("record_id")
    output = geometry.copy()
    output["task"], output["system"] = task, system
    output["target_label"] = output[TASK_LABELS[task]]
    output["eligible_test"] = output.record_id.isin(lookup.index)
    output["oof_role"] = np.where(output.eligible_test, "test", "not_in_test_set")
    for source, destination in (
        ("split_id", "split_id"), ("prediction", "prediction"),
        ("prediction_status", "source_prediction_status"),
    ):
        output[destination] = output.record_id.map(lookup[source]).fillna("")
    for row in output.loc[output.eligible_test, ["record_id", "target_label"]].itertuples(index=False):
        if row.target_label != lookup.at[row.record_id, "target_label"]:
            raise ValueError(f"Reference label mismatch for {task}/{row.record_id}")
    output["prediction_available"] = (
        output.eligible_test & output.source_prediction_status.eq("ok") & output.prediction.ne("")
    )
    output["probability_available"] = False
    output["target_supported"] = pd.array([pd.NA] * len(output), dtype="boolean")
    output["low_margin"] = pd.array([pd.NA] * len(output), dtype="boolean")
    output["overlay_status"] = np.where(
        output.eligible_test,
        np.where(output.prediction_available, "missing_probability_record", "source_prediction_unavailable"),
        "not_in_test_set",
    )
    numeric = (
        "n_probability_classes", "top_probability", "second_probability", "top2_margin",
        "normalized_entropy", "probability_sum", "target_probability", "log_loss", "brier",
    )
    for column in numeric:
        output[column] = np.nan
    for column in ("probability_top_label", "probability_second_label", "probability_source", "probability_class_ids"):
        output[column] = ""
    if task == "topology":
        for group in ("B", "J", "S_other", "other"):
            output[f"mass_{group}"] = np.nan
            output[f"support_{group}_classes"] = np.nan
            output[f"support_{group}_status"] = "unavailable_probability_space"
        for column in ("three_group_mass", "conditional_B", "conditional_J", "conditional_S_other"):
            output[column] = np.nan

    row_indices = dict(zip(output.record_id, output.index, strict=True))
    consumed: set[tuple[str, str]] = set()
    support_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    ignored_non_test = 0
    for path in sorted(directory.glob("probabilities-*.npz")):
        with np.load(BytesIO(read_bytes(path)), allow_pickle=False) as archive:
            ids = archive["record_id"].astype(str)
            split_values = archive["split_id"].astype(str)
            classes = archive["class_ids"].astype(str)
            available = archive["prediction_available"].astype(bool)
            probabilities = np.asarray(archive["probabilities"], dtype=float)
        if len(split_values) != 1 or len(set(classes)) != len(classes) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid identities in {path}")
        split = str(split_values[0])
        if probabilities.shape != (len(ids), len(classes)) or available.shape != (len(ids),):
            raise ValueError(f"Probability dimensions disagree in {path}")
        relevant_test = test[test.split_id.eq(split)]
        targets = relevant_test.target_label.value_counts()
        for label in sorted(set(classes) | set(targets.index)):
            support_rows.append({
                "task": task, "system": system, "split_id": split, "class_label": label,
                "class_supported": label in classes, "test_n_source": int(targets.get(label, 0)),
                "n_probability_classes": len(classes), "probability_source": str(path),
            })
        source_records.append({"path": str(path), "split_id": split, "n_records": len(ids), "n_classes": len(classes)})
        for record_id, has_probability, probability in zip(ids, available, probabilities, strict=True):
            key = (split, record_id)
            if key in consumed:
                raise ValueError(f"Duplicate probability record {key}")
            consumed.add(key)
            if record_id not in lookup.index or lookup.at[record_id, "split_id"] != split:
                ignored_non_test += 1
                continue
            if record_id not in row_indices:
                continue
            index = row_indices[record_id]
            output.at[index, "probability_source"] = str(path)
            output.at[index, "n_probability_classes"] = len(classes)
            output.at[index, "probability_class_ids"] = ";".join(classes)
            output.at[index, "target_supported"] = output.at[index, "target_label"] in classes
            if not output.at[index, "prediction_available"]:
                continue
            if not has_probability:
                output.at[index, "overlay_status"] = "source_probability_unavailable"
                continue
            if len(classes) < 2:
                output.at[index, "overlay_status"] = "insufficient_probability_classes"
                continue
            if (
                not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1)
                or not np.isclose(probability.sum(), 1.0, atol=1e-7, rtol=1e-7)
            ):
                output.at[index, "overlay_status"] = "invalid_probability_vector"
                continue
            predicted = output.at[index, "prediction"]
            if predicted not in classes or probability[np.flatnonzero(classes == predicted)[0]] < probability.max() - 1e-12:
                raise ValueError(f"Native prediction and saved probability disagree: {task}/{record_id}")
            fields = _probability_summary(probability, classes, output.at[index, "target_label"])
            if task == "topology":
                fields.update(_topology_masses(probability, classes))
            for field, value in fields.items():
                output.at[index, field] = value
            output.at[index, "overlay_status"] = "ok" if fields["target_supported"] else "target_outside_probability_space"
    output["correct"] = np.where(
        output.prediction_available, output.prediction.eq(output.target_label).astype(float), np.nan,
    )
    output["correct_test_effectiveness"] = np.where(
        output.eligible_test, output.prediction_available & output.prediction.eq(output.target_label), np.nan,
    ).astype(float)
    output["overlap_stratum"] = output.geometry_label_overlap.map({True: "label_overlap", False: "no_label_overlap"}).fillna("overlap_unavailable")
    output["margin_stratum"] = output.low_margin.map({True: "margin_le_0.2", False: "margin_gt_0.2"}).fillna("margin_unavailable")
    provenance = {
        "task": task, "system": system, "predictions": str(prediction_path),
        "source_non_test_rows_ignored": int(predictions.role.ne("test").sum()),
        "npz_non_test_rows_ignored": ignored_non_test,
        "source_test_rows": len(test), "geometry_cohort_test_rows": int(output.eligible_test.sum()),
        "probability_files": source_records,
    }
    return output, support_rows, provenance


def _mean(series: pd.Series) -> float:
    available = pd.to_numeric(series, errors="coerce").dropna()
    return float(available.mean()) if len(available) else np.nan


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    cohort_n = len(frame)
    test_n = int(frame.eligible_test.sum())
    predicted_n = int(frame.prediction_available.sum())
    probability_n = int(frame.probability_available.sum())
    correct_n = int(frame.correct.fillna(0).sum())
    return {
        "cohort_n": cohort_n, "test_n": test_n, "predicted_n": predicted_n,
        "probability_n": probability_n, "loss_n": int(frame.log_loss.notna().sum()),
        "correct_n": correct_n,
        "prediction_coverage_cohort": predicted_n / cohort_n if cohort_n else np.nan,
        "prediction_coverage_test": predicted_n / test_n if test_n else np.nan,
        "probability_coverage_cohort": probability_n / cohort_n if cohort_n else np.nan,
        "probability_coverage_test": probability_n / test_n if test_n else np.nan,
        "correct_fraction_full_cohort": correct_n / cohort_n if cohort_n else np.nan,
        "accuracy_test_effectiveness": correct_n / test_n if test_n else np.nan,
        "accuracy_prediction_available": correct_n / predicted_n if predicted_n else np.nan,
        "log_loss_available": _mean(frame.log_loss), "brier_available": _mean(frame.brier),
        "normalized_entropy_available": _mean(frame.normalized_entropy),
        "margin_available": _mean(frame.top2_margin),
        "low_margin_fraction_available": _mean(frame.low_margin),
        "overlap_available_n": int(frame.geometry_label_overlap.notna().sum()),
    }


def _summaries(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries, per_class, per_h = [], [], []
    for (task, system), data in table.groupby(["task", "system"], sort=False):
        base = {"task": task, "system": system}
        class_parts = []
        h_parts = []
        for label, part in data.groupby("target_label", sort=True):
            metrics = _metrics(part)
            class_h_accuracy = part.groupby(H_COLUMN).correct_test_effectiveness.mean()
            # Missing test predictions contribute false negatives, not false positives.
            tp = int(part.correct.fillna(0).sum())
            guessed = int((data.prediction_available & data.prediction.eq(label)).sum())
            denominator = metrics["test_n"] + guessed
            metrics.update(
                target_label=label, recall_test_effectiveness=metrics["accuracy_test_effectiveness"],
                H_equal_recall_test_effectiveness=_mean(class_h_accuracy),
                n_H=part[H_COLUMN].nunique(),
                precision_available=tp / guessed if guessed else np.nan,
                f1_test_effectiveness=2 * tp / denominator if metrics["test_n"] and denominator else np.nan,
            )
            per_class.append(base | metrics)
            class_parts.append(metrics)
        for h, part in data.groupby(H_COLUMN, sort=True):
            metrics = _metrics(part)
            per_h.append(base | {H_COLUMN: h} | metrics)
            h_parts.append(metrics)
        summary = base | _metrics(data)
        hframe = pd.DataFrame(h_parts)
        cframe = pd.DataFrame(class_parts)
        summary.update(
            n_H=len(h_parts), n_H_test=int(hframe.test_n.gt(0).sum()),
            n_classes_test=int(cframe.test_n.gt(0).sum()),
            H_equal_accuracy_test_effectiveness=_mean(hframe.accuracy_test_effectiveness),
            H_equal_accuracy_prediction_available=_mean(hframe.accuracy_prediction_available),
            H_equal_log_loss_available=_mean(hframe.log_loss_available),
            H_equal_brier_available=_mean(hframe.brier_available),
            class_equal_recall_test_effectiveness=_mean(cframe.recall_test_effectiveness),
            class_H_equal_recall_test_effectiveness=_mean(cframe.H_equal_recall_test_effectiveness),
        )
        summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(per_class), pd.DataFrame(per_h)


def _calibration(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, system), cohort in table.groupby(["task", "system"], sort=False):
        data = cohort.loc[cohort.probability_available].copy()
        data["bin"] = np.minimum((data.top_probability * 10).astype(int), 9)
        for scope in ("all_valid_test_probabilities", "target_supported"):
            scope_data = data if scope == "all_valid_test_probabilities" else data.loc[data.target_supported.fillna(False)]
            for number in range(10):
                part = scope_data.loc[scope_data.bin.eq(number)]
                confidence, accuracy = _mean(part.top_probability), _mean(part.correct)
                h_means = part.groupby(H_COLUMN)[["top_probability", "correct"]].mean()
                rows.append({
                    "task": task, "system": system, "scope": scope, "bin": number,
                    "bin_left": number / 10, "bin_right": (number + 1) / 10,
                    "right_inclusive": number == 9, "n": len(part), "n_H": len(h_means),
                    "scope_n": len(scope_data), "cohort_n": len(cohort),
                    "mean_confidence": confidence, "accuracy": accuracy,
                    "calibration_gap": confidence - accuracy,
                    "H_equal_mean_confidence": _mean(h_means.top_probability),
                    "H_equal_accuracy": _mean(h_means.correct),
                })
    return pd.DataFrame(rows)


def _strata(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, system), data in table.groupby(["task", "system"], sort=False):
        for overlap in ("label_overlap", "no_label_overlap", "overlap_unavailable"):
            for margin in ("margin_le_0.2", "margin_gt_0.2", "margin_unavailable"):
                part = data.loc[data.overlap_stratum.eq(overlap) & data.margin_stratum.eq(margin)]
                h_accuracy = part.groupby(H_COLUMN).correct_test_effectiveness.mean()
                rows.append({
                    "task": task, "system": system, "overlap_stratum": overlap,
                    "margin_stratum": margin, "n_H": part[H_COLUMN].nunique(),
                    "H_equal_accuracy_test_effectiveness": _mean(h_accuracy),
                } | _metrics(part))
    return pd.DataFrame(rows)


def _disagreements(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons, summaries = [], []
    for task, task_data in table.groupby("task", sort=False):
        for left_name, right_name in combinations(SYSTEMS, 2):
            left = task_data.loc[task_data.system.eq(left_name)].set_index("record_id")
            right = task_data.loc[task_data.system.eq(right_name)].set_index("record_id").reindex(left.index)
            if not left.index.equals(right.index) or not left.target_label.equals(right.target_label):
                raise ValueError("Paired model overlay records or targets differ")
            both = left.prediction_available & right.prediction_available
            both_test = left.eligible_test & right.eligible_test
            comparison = pd.DataFrame({
                "record_id": left.index, "task": task,
                "left_system": left_name, "right_system": right_name,
                H_COLUMN: left[H_COLUMN].to_numpy(),
                "both_test": both_test.to_numpy(), "both_prediction_available": both.to_numpy(),
                "left_prediction": left.prediction.to_numpy(), "right_prediction": right.prediction.to_numpy(),
                "disagreement": np.where(both, left.prediction.ne(right.prediction).astype(float), np.nan),
                "margin_difference": (left.top2_margin - right.top2_margin).to_numpy(),
                "entropy_difference": (left.normalized_entropy - right.normalized_entropy).to_numpy(),
            })
            summaries.append({
                "task": task, "left_system": left_name, "right_system": right_name,
                "cohort_n": len(comparison), "both_test_n": int(both_test.sum()),
                "both_prediction_n": int(both.sum()),
                "disagreement_fraction_available": _mean(comparison.disagreement),
                "H_equal_disagreement_fraction_available": _mean(comparison.groupby(H_COLUMN).disagreement.mean()),
                "n_H_available": comparison.loc[comparison.both_prediction_available, H_COLUMN].nunique(),
            })
            comparisons.append(comparison)
    return pd.concat(comparisons, ignore_index=True), pd.DataFrame(summaries)


def run_model_overlay(
    source_runs: str | Path,
    geometry_domain_table: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    overlap_column: str = "label_overlap",
) -> dict[str, Any]:
    """Write six fixed-system retrospective overlays and descriptive summaries.

    Input identity columns are record_id and cath_architecture_code,
    cath_topology_code, cath_homology_code (A/T/H aliases are accepted). ``role``
    in the geometry table is renamed geometry_role and never determines whether
    a source prediction is OOF. The supplied overlap column is boolean/NA;
    fractional overlap can remain in another input column for continuous review.

    Full-cohort coverage and eligible-test effectiveness have separate
    denominators. Unavailable probabilities and unsupported target losses remain
    NA. No model, probability vector, class prior, or geometry input is changed.
    """
    geometry = _geometry_table(geometry_domain_table, overlap_column)
    if geometry.empty:
        raise ValueError("Geometry cohort is empty")
    assessment = _assessment_root(source_runs)
    tables, support, provenance = [], [], []
    for task in TASK_LABELS:
        for system in SYSTEMS:
            directory = assessment / task / system
            if not directory.is_dir():
                raise FileNotFoundError(f"Required frozen model directory is absent: {directory}")
            table, support_rows, source = _load_system(directory, geometry, task, system)
            tables.append(table)
            support.extend(support_rows)
            provenance.append(source)
    combined = pd.concat(tables, ignore_index=True)
    summary, per_class, per_h = _summaries(combined)
    disagreement, disagreement_summary = _disagreements(combined)
    artifacts = {
        "domain-summaries.csv": combined,
        "summary.csv": summary,
        "per-class.csv": per_class,
        "per-H.csv": per_h,
        "calibration.csv": _calibration(combined),
        "uncertainty-overlap-strata.csv": _strata(combined),
        "class-support.csv": pd.DataFrame(support),
        "head-disagreement.csv": disagreement,
        "head-disagreement-summary.csv": disagreement_summary,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in artifacts.items():
        table.to_csv(destination / name, index=False)
    protocol: dict[str, Any] = {
        "schema": "betlas.geometry-model-overlay.v1", "status": "complete",
        "tasks": list(TASK_LABELS), "systems": list(SYSTEMS),
        "n_geometry_records": len(geometry), "n_domain_system_rows": len(combined),
        "overlap_column": overlap_column,
        "overlap_interpretation": "Boolean reference-label-conditioned geometry relation; NA is not false.",
        "low_margin_threshold": MARGIN_THRESHOLD,
        "normalized_entropy": "Shannon entropy divided by log(number of classes in this fitted fold).",
        "log_loss": "-log(raw target probability clipped below at 1e-15), only supported targets.",
        "brier": "Multiclass sum of squared errors across supported fold classes, range [0,2].",
        "calibration": "Descriptive top-label reliability in fixed 10 equal-width bins; no calibration fitting.",
        "topology_masses": (
            "B=2.40.*, J=2.60.120, S_other=2.60.* except J, other=remaining modeled classes. "
            "These partition each fold's available class space, not the complete CATH catalog. "
            "Zero mass with zero supported classes does not mean zero biological probability. "
            "Conditional group values are NA for absent groups; always retain three_group_mass."
        ),
        "denominators": (
            "All geometry records are retained per system. Eligible means source role=test. "
            "Test effectiveness counts missing predictions as not correct; conditional metrics report availability. "
            "Full-cohort correct fraction is coverage plus performance, not accuracy on unsupported records."
        ),
        "scope": "Retrospective fixed selected-model OOF associations; not new independent validation or causal attribution.",
        "sources": provenance,
        "artifacts": {name: len(table) for name, table in artifacts.items()},
    }
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return protocol
