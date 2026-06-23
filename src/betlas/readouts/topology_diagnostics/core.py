from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ...constants import DEFAULT_FEATURES_CSV, DEFAULT_RUN_DIR, FOLD_LABELS
from ...ml.benchmark import numeric_feature_columns
from ...provenance import build_run_manifest, write_json
from ...schema import normalize_feature_columns
from .config import cfg_get, load_config

READOUT_NAME = "topology-diagnostics"
DEFAULT_PREDICTIONS_CSV = DEFAULT_RUN_DIR / "oof_predictions.csv"
DEFAULT_OUTPUT_CSV = Path("runs/readouts/topology_diagnostics/topology_diagnostics.csv")

TOPOLOGY_GEOMETRY_INPUT_COLUMNS = (
    "betlas_sheet_count",
    "betlas_sheet_face_count",
    "betlas_sheet_pair_top2_fraction",
    "betlas_sheet_pair_size_balance",
    "betlas_sheet_seq_top2_interleave_score",
    "betlas_sheet_seq_top2_order_displacement",
    "betlas_top2_sheet_order_nonlocal_fraction",
    "betlas_top2_sheet_order_inversion_fraction_mean",
    "betlas_sheet_seq_greek_key_proxy",
    "betlas_jelly_roll_order_nonlocal_score",
    "betlas_beta_run_eight_score",
    "betlas_sheet_pair_bilayer_score",
    "betlas_sandwich_lobe_guard_score",
    "betlas_sheet_pair_face_alignment",
    "betlas_sheet_pair_normal_abs_dot",
    "betlas_sheet_pair_cross_contact_density8",
    "betlas_barrel_wall_continuity_score",
    "betlas_axis_best_slice_coverage_median",
    "betlas_axis_best_angular_coverage",
    "betlas_axis_best_largest_gap_fraction",
    "betlas_axis_best_slice_largest_gap_fraction_mean",
    "betlas_contact8_cycle_rank_norm",
    "betlas_contact8_degree2_fraction",
    "betlas_angular_sector_occupancy12",
    "betlas_axis_best_slice_high_coverage_fraction",
    "betlas_angular_fft_k3_8_max",
    "betlas_angular_fft_k3_8_best_k",
    "betlas_axis_periodicity_score",
    "betlas_pca_elongation",
    "betlas_beta_alpha_alternation_fraction",
    "betlas_alpha_shell_radial_delta",
)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp") as handle:
            tmp_path = Path(handle.name)
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

ID_COLUMNS = [
    "record_id",
    "pdb_id",
    "chain_id",
    "domain_id",
    "residue_ranges",
    "source_mmcif_path",
    "source_mmcif_sha256",
    "source_path",
    "betlas_parse_ok",
    "betlas_error",
    "betlas_warnings",
    "fold_label_final",
    "cath_code",
    "cath_topology_code",
    "cath_name",
    "betlas_top_fold",
]

STATUS_COLUMNS = [
    "betlas_topology_status",
    "betlas_topology_error",
]

AMBIGUITY_COLUMNS = [
    "betlas_topology_ambiguity_score",
    "betlas_topology_ambiguity_level",
    "betlas_topology_ambiguity_reasons",
    "betlas_topology_ambiguity_basis_json",
    "betlas_boundary_region_flag",
    "betlas_probability_top1_label",
    "betlas_probability_top2_label",
    "betlas_probability_top1",
    "betlas_probability_top2",
    "betlas_probability_top2_margin",
    "betlas_probability_entropy",
    "betlas_probability_source",
    "betlas_probability_calibration_status",
    "betlas_rule_top1_label",
    "betlas_rule_top2_label",
    "betlas_rule_probability_margin",
    "betlas_rule_probability_entropy",
    "betlas_rule_label_conflict",
    "betlas_neighbor_label_entropy",
    "betlas_neighbor_disagreement_fraction",
    "betlas_boundary_neighbor_fraction",
    "betlas_boundary_neighbor_similarity",
    "betlas_boundary_neighbor_top_competitor",
]

CONTINUOUS_COLUMNS = [
    "betlas_jelly_rollness",
    "betlas_sandwichness",
    "betlas_barrel_likeness",
    "betlas_jelly_rollness_basis_json",
    "betlas_sandwichness_basis_json",
    "betlas_barrel_likeness_basis_json",
    "betlas_jelly_sandwich_overlap",
    "betlas_barrel_sandwich_overlap",
    "betlas_barrel_jelly_overlap",
]

MIXED_COLUMNS = [
    "betlas_mixed_topology_score",
    "betlas_mixed_topology_flag",
    "betlas_mixed_topology_types",
    "betlas_mixed_topology_basis_json",
    "betlas_secondary_topology_label",
    "betlas_secondary_topology_score",
    "betlas_manual_boundary_audit_priority",
]

ALL_READOUT_COLUMNS = STATUS_COLUMNS + AMBIGUITY_COLUMNS + CONTINUOUS_COLUMNS + MIXED_COLUMNS

MODE_COLUMNS = {
    "all": ALL_READOUT_COLUMNS,
    "ambiguity": STATUS_COLUMNS + AMBIGUITY_COLUMNS,
    "continuous": STATUS_COLUMNS + CONTINUOUS_COLUMNS,
    "mixed": STATUS_COLUMNS + MIXED_COLUMNS,
}

BOUNDARY_LABEL_PAIRS = {
    frozenset(("jelly_roll", "beta_sandwich")),
    frozenset(("beta_barrel", "beta_sandwich")),
    frozenset(("beta_barrel", "jelly_roll")),
    frozenset(("beta_propeller", "beta_prism")),
    frozenset(("beta_barrel", "tim_like_beta_alpha_barrel")),
}


@dataclass(frozen=True)
class TopologyDiagnosticsResult:
    diagnostics: pd.DataFrame
    output_csv: Path | None = None
    manifest_path: Path | None = None


def _range(config: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = cfg_get(config, f"scale_ranges.{key}", list(default))
    try:
        left, right = value
        return float(left), float(right)
    except (TypeError, ValueError):
        return default


def _weight(config: dict[str, Any], section: str, key: str, default: float) -> float:
    try:
        return float(cfg_get(config, f"{section}.{key}", default))
    except (TypeError, ValueError):
        return default


def _f(row: pd.Series | dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in {"", None}:
            return default
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    return value_f if math.isfinite(value_f) else default


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clip01((value - low) / (high - low))


def _bell(value: float, center: float, width: float) -> float:
    if width <= 0.0:
        return 0.0
    return _clip01(math.exp(-((value - center) / width) ** 2))


def _weighted_mean(components: dict[str, tuple[float, float]]) -> tuple[float, dict[str, float]]:
    total_weight = 0.0
    weighted_sum = 0.0
    clean: dict[str, float] = {}
    for name, (value, weight) in components.items():
        if weight <= 0.0:
            continue
        clean_value = _clip01(float(value))
        clean[name] = clean_value
        weighted_sum += clean_value * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0, clean
    return _clip01(weighted_sum / total_weight), clean


def _softmax(values: list[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    temp = max(float(temperature), 1e-6)
    arr = arr / temp
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = float(np.sum(exp))
    if denom <= 0.0:
        return np.full(len(arr), 1.0 / max(1, len(arr)), dtype=float)
    return exp / denom


def _entropy(probabilities: np.ndarray) -> float:
    probs = probabilities[np.isfinite(probabilities)]
    probs = probs[probs > 0.0]
    if len(probs) <= 1:
        return 0.0
    return _clip01(float(-np.sum(probs * np.log(probs)) / math.log(len(FOLD_LABELS))))


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _rule_scores_from_row(row: pd.Series) -> dict[str, float]:
    scores = {label: _f(row, f"betlas_rule_score_{label}") for label in FOLD_LABELS}
    if any(value != 0.0 for value in scores.values()):
        return scores
    raw_json = str(row.get("betlas_fold_scores_json", "") or "")
    if not raw_json:
        return scores
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return scores
    return {label: float(parsed.get(label, 0.0) or 0.0) for label in FOLD_LABELS}


def _has_finite_input(row: pd.Series, key: str) -> bool:
    if key not in row.index:
        return False
    value = row.get(key, "")
    if value in {"", None}:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _missing_topology_geometry_inputs(row: pd.Series) -> list[str]:
    return [column for column in TOPOLOGY_GEOMETRY_INPUT_COLUMNS if not _has_finite_input(row, column)]


def _topology_ineligible_reason(row: pd.Series) -> tuple[str, str] | None:
    parse_ok = int(_f(row, "betlas_parse_ok", default=1.0))
    if "betlas_parse_ok" in row.index and parse_ok != 1:
        return (
            "parse_failed",
            str(row.get("betlas_error", "") or row.get("betlas_warnings", "") or "betlas_parse_ok is not 1"),
        )

    score_status = str(row.get("betlas_score_status", "") or "").strip()
    if score_status and score_status != "ok":
        return (score_status, f"betlas_score_status is {score_status!r}")

    if "betlas_axis_best_slice_count" in row.index and _f(row, "betlas_axis_best_slice_count") <= 0.0:
        return ("no_informative_slices", "betlas_axis_best_slice_count is 0")

    scores = _rule_scores_from_row(row)
    if not any(math.isfinite(value) and value != 0.0 for value in scores.values()):
        return ("no_rule_score_signal", "finite nonzero Betlas rule-score signal is unavailable")
    missing_geometry = _missing_topology_geometry_inputs(row)
    if missing_geometry:
        shown = ", ".join(missing_geometry[:8])
        more = f"; plus {len(missing_geometry) - 8} more" if len(missing_geometry) > 8 else ""
        return (
            "missing_topology_geometry",
            "topology diagnostics require raw Betlas geometry columns from extract-features; "
            f"missing or non-finite columns: {shown}{more}",
        )
    return None


def _status_only_row(row: pd.Series, *, status: str, error: str) -> dict[str, Any]:
    out_row = {column: row.get(column, "") for column in ID_COLUMNS if column in row.index}
    out_row.update({column: "" for column in ALL_READOUT_COLUMNS})
    out_row.update(
        {
            "betlas_topology_status": status,
            "betlas_topology_error": error,
        }
    )
    return out_row


def _rule_probability_summary(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    scores = _rule_scores_from_row(row)
    temperature = float(cfg_get(config, "probability.rule_softmax_temperature", 1.0))
    probs = _softmax([scores[label] for label in FOLD_LABELS], temperature=temperature)
    order = np.argsort(probs)[::-1]
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1
    top1_label = FOLD_LABELS[top1]
    top2_label = FOLD_LABELS[top2]
    return {
        "scores": scores,
        "probabilities": {label: float(probs[index]) for index, label in enumerate(FOLD_LABELS)},
        "top1_label": top1_label,
        "top2_label": top2_label,
        "top1_probability": float(probs[top1]),
        "top2_probability": float(probs[top2]),
        "margin": float(probs[top1] - probs[top2]),
        "entropy": _entropy(probs),
    }


def _continuous_scores(row: pd.Series, rule_summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rule_probs: dict[str, float] = rule_summary["probabilities"]
    two_sheet_context = _weighted_mean(
        {
            "two_sheet_count": (_bell(_f(row, "betlas_sheet_count"), 2.0, 1.3), _weight(config, "continuous_weights.two_sheet_context", "two_sheet_count", 0.35)),
            "top2_sheet_fraction": (_f(row, "betlas_sheet_pair_top2_fraction"), _weight(config, "continuous_weights.two_sheet_context", "top2_sheet_fraction", 0.30)),
            "top2_sheet_balance": (_f(row, "betlas_sheet_pair_size_balance"), _weight(config, "continuous_weights.two_sheet_context", "top2_sheet_balance", 0.20)),
            "two_face_context": (_bell(_f(row, "betlas_sheet_face_count"), 2.0, 1.6), _weight(config, "continuous_weights.two_sheet_context", "two_face_context", 0.15)),
        }
    )
    greek_low, greek_high = _range(config, "greek_key_proxy", (0.05, 0.55))
    jelly_order_low, jelly_order_high = _range(config, "jelly_roll_order_nonlocal", (0.03, 0.45))
    jelly_score, jelly_basis = _weighted_mean(
        {
            "two_sheet_context": (two_sheet_context[0], _weight(config, "continuous_weights.jelly_rollness", "two_sheet_context", 0.18)),
            "top2_interleave": (_f(row, "betlas_sheet_seq_top2_interleave_score"), _weight(config, "continuous_weights.jelly_rollness", "top2_interleave", 0.17)),
            "order_displacement": (_f(row, "betlas_sheet_seq_top2_order_displacement"), _weight(config, "continuous_weights.jelly_rollness", "order_displacement", 0.15)),
            "nonlocal_sheet_order": (_f(row, "betlas_top2_sheet_order_nonlocal_fraction"), _weight(config, "continuous_weights.jelly_rollness", "nonlocal_sheet_order", 0.13)),
            "sheet_order_inversion": (_f(row, "betlas_top2_sheet_order_inversion_fraction_mean"), _weight(config, "continuous_weights.jelly_rollness", "sheet_order_inversion", 0.12)),
            "greek_key_proxy": (_scale(_f(row, "betlas_sheet_seq_greek_key_proxy"), greek_low, greek_high), _weight(config, "continuous_weights.jelly_rollness", "greek_key_proxy", 0.11)),
            "jelly_roll_order_nonlocal": (_scale(_f(row, "betlas_jelly_roll_order_nonlocal_score"), jelly_order_low, jelly_order_high), _weight(config, "continuous_weights.jelly_rollness", "jelly_roll_order_nonlocal", 0.08)),
            "eight_run_context": (_f(row, "betlas_beta_run_eight_score"), _weight(config, "continuous_weights.jelly_rollness", "eight_run_context", 0.04)),
            "grammar_jelly_probability": (rule_probs.get("jelly_roll", 0.0), _weight(config, "continuous_weights.jelly_rollness", "grammar_jelly_probability", 0.02)),
        }
    )

    bilayer_low, bilayer_high = _range(config, "bilayer_packing", (0.04, 0.36))
    lobe_low, lobe_high = _range(config, "lobe_guard", (0.04, 0.42))
    contact_low, contact_high = _range(config, "sheet_pair_cross_contact_density8", (0.05, 0.45))
    anti_low, anti_high = _range(config, "anti_closure_barrel_wall", (0.16, 0.52))
    sandwich_score, sandwich_basis = _weighted_mean(
        {
            "two_sheet_context": (two_sheet_context[0], _weight(config, "continuous_weights.sandwichness", "two_sheet_context", 0.17)),
            "top2_sheet_fraction": (_f(row, "betlas_sheet_pair_top2_fraction"), _weight(config, "continuous_weights.sandwichness", "top2_sheet_fraction", 0.12)),
            "top2_sheet_balance": (_f(row, "betlas_sheet_pair_size_balance"), _weight(config, "continuous_weights.sandwichness", "top2_sheet_balance", 0.12)),
            "bilayer_packing": (_scale(_f(row, "betlas_sheet_pair_bilayer_score"), bilayer_low, bilayer_high), _weight(config, "continuous_weights.sandwichness", "bilayer_packing", 0.16)),
            "lobe_guard": (_scale(_f(row, "betlas_sandwich_lobe_guard_score"), lobe_low, lobe_high), _weight(config, "continuous_weights.sandwichness", "lobe_guard", 0.13)),
            "face_alignment": (_f(row, "betlas_sheet_pair_face_alignment"), _weight(config, "continuous_weights.sandwichness", "face_alignment", 0.10)),
            "normal_alignment": (_f(row, "betlas_sheet_pair_normal_abs_dot"), _weight(config, "continuous_weights.sandwichness", "normal_alignment", 0.08)),
            "cross_sheet_contacts": (_scale(_f(row, "betlas_sheet_pair_cross_contact_density8"), contact_low, contact_high), _weight(config, "continuous_weights.sandwichness", "cross_sheet_contacts", 0.06)),
            "anti_closure": (1.0 - _scale(_f(row, "betlas_barrel_wall_continuity_score"), anti_low, anti_high), _weight(config, "continuous_weights.sandwichness", "anti_closure", 0.04)),
            "grammar_sandwich_probability": (rule_probs.get("beta_sandwich", 0.0), _weight(config, "continuous_weights.sandwichness", "grammar_sandwich_probability", 0.02)),
        }
    )

    closure_low, closure_high = _range(config, "slice_closure", (0.62, 0.88))
    coverage_low, coverage_high = _range(config, "angular_coverage", (0.64, 0.90))
    gap_low, gap_high = _range(config, "global_gap", (0.18, 0.52))
    slice_gap_low, slice_gap_high = _range(config, "slice_gap", (0.22, 0.58))
    wall_low, wall_high = _range(config, "barrel_wall_continuity", (0.08, 0.55))
    barrel_lobe_low, barrel_lobe_high = _range(config, "sandwich_lobe_guard_for_barrel", (0.18, 0.52))
    cycle_low, cycle_high = _range(config, "contact_cycle_rank", (0.03, 0.30))
    barrel_score, barrel_basis = _weighted_mean(
        {
            "slice_closure": (_scale(_f(row, "betlas_axis_best_slice_coverage_median"), closure_low, closure_high), _weight(config, "continuous_weights.barrel_likeness", "slice_closure", 0.14)),
            "angular_coverage": (_scale(_f(row, "betlas_axis_best_angular_coverage"), coverage_low, coverage_high), _weight(config, "continuous_weights.barrel_likeness", "angular_coverage", 0.12)),
            "small_global_gap": (1.0 - _scale(_f(row, "betlas_axis_best_largest_gap_fraction"), gap_low, gap_high), _weight(config, "continuous_weights.barrel_likeness", "small_global_gap", 0.10)),
            "slice_gap_continuity": (
                1.0 - _scale(_f(row, "betlas_axis_best_slice_largest_gap_fraction_mean"), slice_gap_low, slice_gap_high),
                _weight(config, "continuous_weights.barrel_likeness", "slice_gap_continuity", 0.08),
            ),
            "barrel_wall_continuity": (_scale(_f(row, "betlas_barrel_wall_continuity_score"), wall_low, wall_high), _weight(config, "continuous_weights.barrel_likeness", "barrel_wall_continuity", 0.14)),
            "contact_cycle_rank": (_scale(_f(row, "betlas_contact8_cycle_rank_norm"), cycle_low, cycle_high), _weight(config, "continuous_weights.barrel_likeness", "contact_cycle_rank", 0.10)),
            "degree2_ring": (_f(row, "betlas_contact8_degree2_fraction"), _weight(config, "continuous_weights.barrel_likeness", "degree2_ring", 0.10)),
            "sector_occupancy": (_f(row, "betlas_angular_sector_occupancy12"), _weight(config, "continuous_weights.barrel_likeness", "sector_occupancy", 0.08)),
            "high_coverage_slices": (_f(row, "betlas_axis_best_slice_high_coverage_fraction"), _weight(config, "continuous_weights.barrel_likeness", "high_coverage_slices", 0.08)),
            "non_bilayer_guard": (1.0 - _scale(_f(row, "betlas_sandwich_lobe_guard_score"), barrel_lobe_low, barrel_lobe_high), _weight(config, "continuous_weights.barrel_likeness", "non_bilayer_guard", 0.06)),
            "grammar_barrel_probability": (rule_probs.get("beta_barrel", 0.0), _weight(config, "continuous_weights.barrel_likeness", "grammar_barrel_probability", 0.04)),
        }
    )

    return {
        "betlas_jelly_rollness": jelly_score,
        "betlas_sandwichness": sandwich_score,
        "betlas_barrel_likeness": barrel_score,
        "betlas_jelly_rollness_basis_json": _json(jelly_basis),
        "betlas_sandwichness_basis_json": _json(sandwich_basis),
        "betlas_barrel_likeness_basis_json": _json(barrel_basis),
        "betlas_jelly_sandwich_overlap": _clip01(min(jelly_score, sandwich_score) * (1.0 - abs(jelly_score - sandwich_score))),
        "betlas_barrel_sandwich_overlap": _clip01(min(barrel_score, sandwich_score) * (1.0 - abs(barrel_score - sandwich_score))),
        "betlas_barrel_jelly_overlap": _clip01(min(barrel_score, jelly_score) * (1.0 - abs(barrel_score - jelly_score))),
    }


def _prediction_key_columns(df: pd.DataFrame) -> list[str]:
    for candidate in (["record_id"], ["domain_id"], ["pdb_id", "domain_id"]):
        if all(column in df.columns for column in candidate):
            return candidate
    return []


def _prepare_prediction_frame(
    predictions: pd.DataFrame | None,
    *,
    model: str | None,
) -> pd.DataFrame | None:
    if predictions is None or predictions.empty:
        return None
    pred = predictions.copy()
    if model and "model" in pred.columns:
        pred = pred[pred["model"].astype(str) == model].copy()
    if pred.empty:
        return None
    keys = _prediction_key_columns(pred)
    if not keys:
        return None
    pred = pred.drop_duplicates(keys, keep="first")
    return pred


def _prediction_frame_has_signal(predictions: pd.DataFrame) -> bool:
    prob_cols = [f"prob_{label}" for label in FOLD_LABELS]
    return all(column in predictions.columns for column in prob_cols) or "pred_probability" in predictions.columns


def _prediction_summary(row: pd.Series) -> dict[str, Any]:
    prob_cols = [f"prob_{label}" for label in FOLD_LABELS]
    if all(column in row.index for column in prob_cols):
        probs = np.asarray([_f(row, column) for column in prob_cols], dtype=float)
        total = float(np.sum(probs))
        if total > 0.0:
            probs = probs / total
            order = np.argsort(probs)[::-1]
            top1 = int(order[0])
            top2 = int(order[1]) if len(order) > 1 else top1
            return {
                "available": True,
                "top1_label": FOLD_LABELS[top1],
                "top2_label": FOLD_LABELS[top2],
                "top1_probability": float(probs[top1]),
                "top2_probability": float(probs[top2]),
                "margin": float(probs[top1] - probs[top2]),
                "entropy": _entropy(probs),
                "source": "prediction",
                "calibration_status": "model_reported_uncalibrated",
            }

    pred_label = str(row.get("pred_label", "") or "")
    top2_labels = [label for label in str(row.get("top2_labels", "") or "").split(";") if label]
    top1_probability = _f(row, "pred_probability", default=float("nan"))
    if not math.isfinite(top1_probability):
        return {"available": False}
    top1_label = pred_label or (top2_labels[0] if top2_labels else "")
    top2_label = top2_labels[1] if len(top2_labels) > 1 else ""
    top2_probability = float("nan")
    margin = float("nan")
    return {
        "available": True,
        "top1_label": top1_label,
        "top2_label": top2_label,
        "top1_probability": _clip01(top1_probability),
        "top2_probability": top2_probability,
        "margin": margin,
        "entropy": float("nan"),
        "source": "prediction",
        "calibration_status": "model_reported_uncalibrated",
    }


def _boundary_pair(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    return frozenset((left, right)) in BOUNDARY_LABEL_PAIRS


def _neighbor_readouts(features: pd.DataFrame, *, k_neighbors: int) -> pd.DataFrame:
    n = len(features)
    empty = pd.DataFrame(
        {
            "betlas_neighbor_label_entropy": np.zeros(n),
            "betlas_neighbor_disagreement_fraction": np.zeros(n),
            "betlas_boundary_neighbor_fraction": np.zeros(n),
            "betlas_boundary_neighbor_similarity": np.zeros(n),
            "betlas_boundary_neighbor_top_competitor": [""] * n,
        },
        index=features.index,
    )
    if n <= 2 or "fold_label_final" not in features.columns:
        return empty

    labels = features["fold_label_final"].astype(str)
    valid = labels.isin(FOLD_LABELS) & pd.to_numeric(features.get("betlas_parse_ok", 1), errors="coerce").fillna(0).astype(bool)
    if int(valid.sum()) <= 2:
        return empty

    feature_cols = numeric_feature_columns(features.loc[valid])
    feature_cols = [column for column in feature_cols if not column.startswith("betlas_rule_score_")]
    if not feature_cols:
        return empty

    matrix = features.loc[valid, feature_cols].apply(pd.to_numeric, errors="coerce")
    matrix = SimpleImputer(strategy="median").fit_transform(matrix)
    matrix = StandardScaler().fit_transform(matrix)
    n_valid = matrix.shape[0]
    neighbors = min(max(2, k_neighbors + 1), n_valid)
    model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)
    valid_indices = np.asarray(features.index[valid])
    valid_labels = labels.loc[valid].to_numpy()

    result = empty.copy()
    for local_i, row_index in enumerate(valid_indices):
        neighbor_locs = [idx for idx in indices[local_i].tolist() if idx != local_i][:k_neighbors]
        if not neighbor_locs:
            continue
        neighbor_labels = valid_labels[neighbor_locs]
        own_label = valid_labels[local_i]
        counts = pd.Series(neighbor_labels).value_counts()
        probs = counts.to_numpy(dtype=float) / max(1.0, float(counts.sum()))
        entropy = float(-np.sum(probs * np.log(probs)) / math.log(len(FOLD_LABELS))) if len(probs) > 1 else 0.0
        disagreement = float(np.mean(neighbor_labels != own_label))
        boundary_flags = np.asarray([_boundary_pair(own_label, label) for label in neighbor_labels], dtype=bool)
        neighbor_distances = np.asarray([distances[local_i][list(indices[local_i]).index(idx)] for idx in neighbor_locs], dtype=float)
        similarities = np.exp(-neighbor_distances)
        boundary_similarity = float(np.mean(similarities[boundary_flags])) if np.any(boundary_flags) else 0.0
        competitors = counts.drop(labels=[own_label], errors="ignore")
        competitor = str(competitors.index[0]) if len(competitors) else ""
        result.loc[row_index, "betlas_neighbor_label_entropy"] = _clip01(entropy)
        result.loc[row_index, "betlas_neighbor_disagreement_fraction"] = _clip01(disagreement)
        result.loc[row_index, "betlas_boundary_neighbor_fraction"] = _clip01(float(np.mean(boundary_flags)))
        result.loc[row_index, "betlas_boundary_neighbor_similarity"] = _clip01(boundary_similarity)
        result.loc[row_index, "betlas_boundary_neighbor_top_competitor"] = competitor
    return result


def _ambiguity_readouts(
    row: pd.Series,
    continuous: dict[str, Any],
    rule_summary: dict[str, Any],
    prediction_summary: dict[str, Any],
    neighbor: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    label = str(row.get("fold_label_final", "") or "")
    rule_top = str(rule_summary["top1_label"])
    rule_second = str(rule_summary["top2_label"])
    rule_label_conflict = bool(label in FOLD_LABELS and rule_top and label != rule_top)

    if prediction_summary.get("available") and math.isfinite(float(prediction_summary.get("margin", float("nan")))):
        prob_margin = float(prediction_summary["margin"])
        low, high = cfg_get(config, "probability.top2_margin_ambiguity_range", [0.03, 0.32])
        probability_margin_ambiguity = 1.0 - _scale(prob_margin, float(low), float(high))
        probability_entropy = float(prediction_summary.get("entropy", float("nan")))
    elif prediction_summary.get("available"):
        top1_probability = float(prediction_summary.get("top1_probability", float("nan")))
        prob_margin = float("nan")
        low, high = cfg_get(config, "probability.unavailable_margin_low_confidence_range", [0.55, 0.90])
        probability_margin_ambiguity = 1.0 - _scale(top1_probability, float(low), float(high))
        probability_entropy = float("nan")
    else:
        prob_margin = float(rule_summary["margin"])
        low, high = cfg_get(config, "probability.top2_margin_ambiguity_range", [0.03, 0.32])
        probability_margin_ambiguity = 1.0 - _scale(prob_margin, float(low), float(high))
        probability_entropy = float(rule_summary["entropy"])

    sandwich = float(continuous["betlas_sandwichness"])
    barrel = float(continuous["betlas_barrel_likeness"])
    jelly_sandwich_conflict = float(continuous["betlas_jelly_sandwich_overlap"])
    barrel_sandwich_conflict = float(continuous["betlas_barrel_sandwich_overlap"])
    barrel_vs_sequence_conflict = _clip01(
        barrel
        * sandwich
        * (
            0.5 * _f(row, "betlas_top2_sheet_order_nonlocal_fraction")
            + 0.5 * _f(row, "betlas_sheet_seq_top2_interleave_score")
        )
    )
    rule_boundary_pair = 1.0 if _boundary_pair(rule_top, rule_second) else 0.0
    label_boundary_pair = 1.0 if _boundary_pair(label, rule_top) else 0.0
    rule_conflict_strength = 0.0
    if rule_label_conflict:
        conflict_low, conflict_high = _range(config, "rule_conflict_margin", (0.25, 0.90))
        rule_conflict_strength = max(
            _scale(_f(row, "betlas_rule_margin"), conflict_low, conflict_high),
            0.75 * label_boundary_pair,
            0.50 * rule_boundary_pair,
        )
    model_label_conflict = 0.0
    if prediction_summary.get("available"):
        pred_label = str(prediction_summary.get("top1_label", "") or "")
        model_label_conflict = 1.0 if label in FOLD_LABELS and pred_label and pred_label != label else 0.0

    components = {
        "probability_margin": (probability_margin_ambiguity, _weight(config, "ambiguity_weights", "probability_margin", 0.22)),
        "probability_entropy": (_clip01(probability_entropy), _weight(config, "ambiguity_weights", "probability_entropy", 0.09)),
        "rule_margin": (
            1.0
            - _scale(
                float(rule_summary["margin"]),
                float(cfg_get(config, "probability.rule_margin_ambiguity_range", [0.03, 0.30])[0]),
                float(cfg_get(config, "probability.rule_margin_ambiguity_range", [0.03, 0.30])[1]),
            ),
            _weight(config, "ambiguity_weights", "rule_margin", 0.12),
        ),
        "jelly_sandwich_conflict": (jelly_sandwich_conflict, _weight(config, "ambiguity_weights", "jelly_sandwich_conflict", 0.14)),
        "barrel_sandwich_conflict": (barrel_sandwich_conflict, _weight(config, "ambiguity_weights", "barrel_sandwich_conflict", 0.10)),
        "barrel_sequence_conflict": (barrel_vs_sequence_conflict, _weight(config, "ambiguity_weights", "barrel_sequence_conflict", 0.08)),
        "boundary_neighbor": (_f(neighbor, "betlas_boundary_neighbor_fraction"), _weight(config, "ambiguity_weights", "boundary_neighbor", 0.09)),
        "neighbor_entropy": (_f(neighbor, "betlas_neighbor_label_entropy"), _weight(config, "ambiguity_weights", "neighbor_entropy", 0.07)),
        "label_rule_conflict": (rule_conflict_strength, _weight(config, "ambiguity_weights", "label_rule_conflict", 0.06)),
        "model_label_conflict": (model_label_conflict, _weight(config, "ambiguity_weights", "model_label_conflict", 0.03)),
    }
    score, basis = _weighted_mean(components)

    reasons: list[str] = []
    if basis["probability_margin"] >= 0.65:
        reasons.append("small_top2_margin" if math.isfinite(prob_margin) else "low_model_confidence")
    if jelly_sandwich_conflict >= 0.35:
        reasons.append("jelly_roll_sandwich_overlap")
    if barrel_sandwich_conflict >= 0.35:
        reasons.append("barrel_like_sandwich_overlap")
    if barrel_vs_sequence_conflict >= 0.30:
        reasons.append("closure_sequence_conflict")
    if _f(neighbor, "betlas_boundary_neighbor_fraction") >= 0.35:
        reasons.append("boundary_neighbors")
    label_conflict_threshold = float(cfg_get(config, "ambiguity.label_rule_conflict_threshold", 0.45))
    if rule_conflict_strength >= label_conflict_threshold:
        reasons.append("label_rule_conflict")
    if rule_boundary_pair > 0.0:
        reasons.append("top_rule_pair_is_boundary_pair")
    if label_boundary_pair > 0.0:
        reasons.append("label_rule_boundary_pair")

    high_threshold = float(cfg_get(config, "ambiguity.high_threshold", 0.55))
    moderate_threshold = float(cfg_get(config, "ambiguity.moderate_threshold", 0.35))
    if score >= high_threshold:
        level = "high"
    elif score >= moderate_threshold:
        level = "moderate"
    else:
        level = "low"

    return {
        "betlas_topology_ambiguity_score": score,
        "betlas_topology_ambiguity_level": level,
        "betlas_topology_ambiguity_reasons": ";".join(reasons),
        "betlas_topology_ambiguity_basis_json": _json(basis),
        "betlas_boundary_region_flag": int(
            score >= float(cfg_get(config, "ambiguity.boundary_score_threshold", 0.35))
            or (
                jelly_sandwich_conflict >= float(cfg_get(config, "ambiguity.jelly_sandwich_boundary_threshold", 0.40))
                and label in {"beta_sandwich", "jelly_roll", ""}
            )
            or (
                barrel_sandwich_conflict >= float(cfg_get(config, "ambiguity.barrel_sandwich_boundary_threshold", 0.48))
                and label in {"beta_barrel", "beta_sandwich", "jelly_roll", ""}
            )
            or _f(neighbor, "betlas_boundary_neighbor_fraction") >= float(cfg_get(config, "ambiguity.neighbor_boundary_threshold", 0.35))
            or rule_conflict_strength >= float(cfg_get(config, "ambiguity.strong_label_rule_conflict_threshold", 0.65))
        ),
        "betlas_probability_top1_label": str(prediction_summary.get("top1_label", rule_top) or rule_top),
        "betlas_probability_top2_label": str(prediction_summary.get("top2_label", rule_second) or rule_second),
        "betlas_probability_top1": float(prediction_summary.get("top1_probability", rule_summary["top1_probability"])),
        "betlas_probability_top2": float(prediction_summary.get("top2_probability", rule_summary["top2_probability"])),
        "betlas_probability_top2_margin": prob_margin,
        "betlas_probability_entropy": probability_entropy,
        "betlas_probability_source": str(prediction_summary.get("source", "rule_softmax") or "rule_softmax"),
        "betlas_probability_calibration_status": str(
            prediction_summary.get("calibration_status", "uncalibrated_rule_softmax") or "uncalibrated_rule_softmax"
        ),
        "betlas_rule_top1_label": rule_top,
        "betlas_rule_top2_label": rule_second,
        "betlas_rule_probability_margin": float(rule_summary["margin"]),
        "betlas_rule_probability_entropy": float(rule_summary["entropy"]),
        "betlas_rule_label_conflict": int(rule_conflict_strength >= label_conflict_threshold),
        "betlas_neighbor_label_entropy": _f(neighbor, "betlas_neighbor_label_entropy"),
        "betlas_neighbor_disagreement_fraction": _f(neighbor, "betlas_neighbor_disagreement_fraction"),
        "betlas_boundary_neighbor_fraction": _f(neighbor, "betlas_boundary_neighbor_fraction"),
        "betlas_boundary_neighbor_similarity": _f(neighbor, "betlas_boundary_neighbor_similarity"),
        "betlas_boundary_neighbor_top_competitor": str(neighbor.get("betlas_boundary_neighbor_top_competitor", "") or ""),
    }


def _mixed_topology_readouts(
    row: pd.Series,
    continuous: dict[str, Any],
    ambiguity: dict[str, Any],
    rule_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    label = str(row.get("fold_label_final", "") or "")
    jelly = float(continuous["betlas_jelly_rollness"])
    sandwich = float(continuous["betlas_sandwichness"])
    barrel = float(continuous["betlas_barrel_likeness"])
    rule_probs: dict[str, float] = rule_summary["probabilities"]

    propeller_repeat = _weighted_mean(
        {
            "propeller_rule": (rule_probs.get("beta_propeller", 0.0), 0.40),
            "periodic_fft": (
                _scale(_f(row, "betlas_angular_fft_k3_8_max"), *_range(config, "periodic_fft", (0.08, 0.24))),
                0.25,
            ),
            "repeat_k_context": (_bell(_f(row, "betlas_angular_fft_k3_8_best_k"), 6.0, 2.5), 0.20),
            "sector_occupancy": (_f(row, "betlas_angular_sector_occupancy12"), 0.15),
        }
    )[0]
    prism_repeat = _weighted_mean(
        {
            "prism_rule": (rule_probs.get("beta_prism", 0.0), 0.50),
            "three_face_context": (_bell(_f(row, "betlas_sheet_face_count"), 3.0, 1.0), 0.30),
            "k3_context": (_bell(_f(row, "betlas_angular_fft_k3_8_best_k"), 3.0, 1.2), 0.20),
        }
    )[0]
    solenoid_repeat = _weighted_mean(
        {
            "solenoid_rule": (rule_probs.get("beta_solenoid", 0.0), 0.45),
            "axis_periodicity": (_f(row, "betlas_axis_periodicity_score"), 0.35),
            "elongation": (
                _scale(_f(row, "betlas_pca_elongation"), *_range(config, "solenoid_elongation", (1.5, 4.5))),
                0.20,
            ),
        }
    )[0]
    tim_shell = _weighted_mean(
        {
            "tim_rule": (rule_probs.get("tim_like_beta_alpha_barrel", 0.0), 0.45),
            "beta_alpha_alternation": (_f(row, "betlas_beta_alpha_alternation_fraction"), 0.30),
            "alpha_shell": (
                _scale(_f(row, "betlas_alpha_shell_radial_delta"), *_range(config, "alpha_shell", (0.0, 8.0))),
                0.25,
            ),
        }
    )[0]

    local_scores = {
        "jelly_roll_like": jelly,
        "sandwich_like": sandwich,
        "barrel_like": barrel,
        "propeller_like_repeat": propeller_repeat,
        "prism_like_repeat": prism_repeat,
        "solenoid_like_axis": solenoid_repeat,
        "tim_like_alpha_shell": tim_shell,
    }
    ranked = sorted(local_scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ranked[0]
    secondary_name, secondary_score = ranked[1] if len(ranked) > 1 else ("", 0.0)

    hybrid_components = {
        "jelly_roll_inside_sandwich": (
            min(jelly, sandwich) if label in {"beta_sandwich", "jelly_roll", ""} else 0.75 * min(jelly, sandwich),
            _weight(config, "mixed_weights", "jelly_roll_inside_sandwich", 0.19),
        ),
        "barrel_closure_with_sandwich": (min(barrel, sandwich), _weight(config, "mixed_weights", "barrel_closure_with_sandwich", 0.17)),
        "barrel_closure_with_jelly_roll": (min(barrel, jelly), _weight(config, "mixed_weights", "barrel_closure_with_jelly_roll", 0.10)),
        "propeller_repeat_with_sandwich": (min(propeller_repeat, sandwich), _weight(config, "mixed_weights", "propeller_repeat_with_sandwich", 0.12)),
        "prism_repeat_with_sandwich": (min(prism_repeat, sandwich), _weight(config, "mixed_weights", "prism_repeat_with_sandwich", 0.08)),
        "solenoid_axis_with_sheet_packing": (min(solenoid_repeat, max(jelly, sandwich)), _weight(config, "mixed_weights", "solenoid_axis_with_sheet_packing", 0.08)),
        "tim_shell_with_barrel_closure": (min(tim_shell, barrel), _weight(config, "mixed_weights", "tim_shell_with_barrel_closure", 0.08)),
        "secondary_signal_strength": (secondary_score, _weight(config, "mixed_weights", "secondary_signal_strength", 0.10)),
        "ambiguity_support": (float(ambiguity["betlas_topology_ambiguity_score"]), _weight(config, "mixed_weights", "ambiguity_support", 0.08)),
    }
    score, basis = _weighted_mean(hybrid_components)

    types: list[str] = []
    thresholds = cfg_get(config, "mixed.type_thresholds", {})
    if basis["jelly_roll_inside_sandwich"] >= float(thresholds.get("jelly_roll_inside_sandwich", 0.50)):
        types.append("jelly_roll_like_sandwich_subtopology")
    if basis["barrel_closure_with_sandwich"] >= float(thresholds.get("barrel_closure_with_sandwich", 0.52)):
        types.append("sandwich_with_barrel_like_closure")
    if basis["barrel_closure_with_jelly_roll"] >= float(thresholds.get("barrel_closure_with_jelly_roll", 0.50)):
        types.append("jelly_roll_with_barrel_like_closure")
    if basis["propeller_repeat_with_sandwich"] >= float(thresholds.get("propeller_repeat_with_sandwich", 0.45)):
        types.append("propeller_like_repeat_with_sandwich_packing")
    if basis["prism_repeat_with_sandwich"] >= float(thresholds.get("prism_repeat_with_sandwich", 0.45)):
        types.append("prism_like_repeat_with_sandwich_packing")
    if basis["solenoid_axis_with_sheet_packing"] >= float(thresholds.get("solenoid_axis_with_sheet_packing", 0.45)):
        types.append("solenoid_like_axis_with_sheet_packing")
    if basis["tim_shell_with_barrel_closure"] >= float(thresholds.get("tim_shell_with_barrel_closure", 0.45)):
        types.append("tim_like_alpha_shell_with_barrel_closure")
    structural_type_count = len(types)
    label_grammar_conflict = bool(ambiguity["betlas_rule_label_conflict"])
    if label_grammar_conflict and (
        structural_type_count > 0 or float(ambiguity["betlas_topology_ambiguity_score"]) >= 0.55
    ):
        types.append("label_grammar_conflict")

    min_structural_types = int(cfg_get(config, "mixed.min_structural_type_count", 2))
    ambiguity_score = float(ambiguity["betlas_topology_ambiguity_score"])
    flag = int(
        score >= float(cfg_get(config, "mixed.flag_score_threshold", 0.48))
        or structural_type_count >= max(1, min_structural_types)
        or (
            structural_type_count > 0
            and label_grammar_conflict
            and ambiguity_score >= float(cfg_get(config, "mixed.moderate_ambiguity_threshold", 0.35))
        )
    )
    if score >= float(cfg_get(config, "mixed.high_score_threshold", 0.62)) or (
        flag and ambiguity_score >= float(cfg_get(config, "mixed.high_ambiguity_threshold", 0.55))
    ):
        priority = "high"
    elif flag or ambiguity_score >= float(cfg_get(config, "mixed.moderate_ambiguity_threshold", 0.35)):
        priority = "moderate"
    else:
        priority = "low"

    return {
        "betlas_mixed_topology_score": score,
        "betlas_mixed_topology_flag": flag,
        "betlas_mixed_topology_types": ";".join(types),
        "betlas_mixed_topology_basis_json": _json(basis | {"primary_signal": top_score}),
        "betlas_secondary_topology_label": secondary_name,
        "betlas_secondary_topology_score": float(secondary_score),
        "betlas_manual_boundary_audit_priority": priority,
    }


def compute_topology_diagnostics(
    features: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    model: str | None = "hist_gradient_boosting",
    k_neighbors: int = 12,
    config: dict[str, Any] | None = None,
    predictions_required: bool = False,
) -> pd.DataFrame:
    """Compute Betlas topology diagnostic readouts from a feature table."""
    config = config or load_config()
    if features.empty:
        return pd.DataFrame(columns=[*ID_COLUMNS, *ALL_READOUT_COLUMNS])

    df = features.copy()
    pred = _prepare_prediction_frame(predictions, model=model)
    prediction_marker = "__prediction_matched"
    if predictions is not None and predictions_required and pred is None:
        raise ValueError(
            "prediction CSV was provided but no usable prediction rows were found for "
            "the requested model/key columns; omit --predictions to use rule-softmax weights"
        )
    if pred is not None:
        if predictions_required and not _prediction_frame_has_signal(pred):
            raise ValueError(
                "prediction CSV matched the requested model/key columns but lacks usable probability "
                f"columns; expected either prob_<label> columns for {list(FOLD_LABELS)} or finite "
                "pred_probability values. Omit --predictions to use rule-softmax weights"
            )
        left_keys = _prediction_key_columns(df)
        right_keys = _prediction_key_columns(pred)
        if left_keys and right_keys and left_keys == right_keys:
            pred = pred.copy()
            pred[prediction_marker] = 1
            df = df.merge(pred, on=left_keys, how="left", suffixes=("", "_prediction"))
            if predictions_required:
                parse_ok = (
                    pd.to_numeric(df.get("betlas_parse_ok", 1), errors="coerce")
                    .fillna(0)
                    .astype(int)
                    == 1
                )
                expected = int(parse_ok.sum())
                marker = pd.to_numeric(df.get(prediction_marker, 0), errors="coerce").fillna(0)
                matched = int(marker.loc[parse_ok].sum())
                if matched < expected:
                    raise ValueError(
                        "prediction CSV did not match every parse-ok feature row "
                        f"for keys {left_keys}: matched {matched}/{expected}; "
                        "omit --predictions to use rule-softmax weights"
                    )
                usable = df.apply(lambda row: bool(_prediction_summary(row).get("available")), axis=1)
                usable_count = int(usable.loc[parse_ok].sum())
                if usable_count < expected:
                    raise ValueError(
                        "prediction CSV matched parse-ok feature rows but lacks usable probability "
                        f"signals for {expected - usable_count}/{expected} row(s); expected positive "
                        "prob_<label> totals or finite pred_probability values. Omit --predictions "
                        "to use rule-softmax weights"
                    )
            df = df.drop(columns=[prediction_marker], errors="ignore")
        elif predictions_required:
            raise ValueError(
                "prediction CSV key columns do not match feature CSV key columns; "
                "omit --predictions to use rule-softmax weights"
            )

    neighbors = _neighbor_readouts(df, k_neighbors=k_neighbors)
    rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        ineligible = _topology_ineligible_reason(row)
        if ineligible is not None:
            status, error = ineligible
            rows.append(_status_only_row(row, status=status, error=error))
            continue
        out_row = {column: row.get(column, "") for column in ID_COLUMNS if column in row.index}
        rule_summary = _rule_probability_summary(row, config)
        continuous = _continuous_scores(row, rule_summary, config)
        prediction_summary = _prediction_summary(row)
        ambiguity = _ambiguity_readouts(
            row,
            continuous,
            rule_summary,
            prediction_summary,
            neighbors.loc[index],
            config,
        )
        mixed = _mixed_topology_readouts(row, continuous, ambiguity, rule_summary, config)

        out_row.update(
            {
                "betlas_topology_status": "ok",
                "betlas_topology_error": "",
            }
        )
        out_row.update(ambiguity)
        out_row.update(continuous)
        out_row.update(mixed)
        rows.append(out_row)
    return pd.DataFrame(rows)


def select_mode_columns(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in MODE_COLUMNS:
        raise ValueError(f"unknown topology diagnostics mode {mode!r}; expected one of {sorted(MODE_COLUMNS)}")
    id_columns = [column for column in ID_COLUMNS if column in df.columns]
    columns = [column for column in MODE_COLUMNS[mode] if column in df.columns]
    return df[id_columns + columns].copy()


def run_topology_diagnostics(
    *,
    features_csv: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    model: str | None = None,
    out_csv: str | Path | None = None,
    mode: str = "all",
    k_neighbors: int | None = None,
    include_features: bool = False,
    config_path: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    write_manifest: bool = True,
    predictions_required: bool = False,
) -> TopologyDiagnosticsResult:
    config = load_config(config_path, overrides=config_overrides)
    out_csv_provided = out_csv is not None
    features_csv = features_csv or cfg_get(config, "io.features_csv", str(DEFAULT_FEATURES_CSV))
    if model is None:
        model = cfg_get(config, "probability.model", "hist_gradient_boosting")
    out_csv = out_csv or cfg_get(config, "io.output_csv", str(DEFAULT_OUTPUT_CSV))
    if k_neighbors is None:
        k_neighbors = int(cfg_get(config, "neighbors.k", 12))

    features_path = Path(features_csv)
    if not features_path.exists():
        hint = ""
        if features_path == DEFAULT_FEATURES_CSV:
            hint = (
                "; this is the default output path. Run "
                "`betlas extract-features --structure STRUCTURE.cif --chain CHAIN --out runs/features.csv` "
                "and pass `--features runs/features.csv`, or run the documented quickstart first"
            )
        raise FileNotFoundError(f"feature CSV does not exist: {features_path}{hint}")
    features = normalize_feature_columns(
        pd.read_csv(features_path, dtype=str, keep_default_na=False, low_memory=False)
    )
    predictions = None
    if predictions_csv is not None:
        predictions_path = Path(predictions_csv)
        if not predictions_path.exists():
            if not predictions_required:
                predictions_path = None
            else:
                raise FileNotFoundError(
                    f"prediction CSV does not exist: {predictions_path}; omit --predictions or pass --no-predictions to use rule-softmax weights"
                )
        if predictions_path is not None:
            predictions = pd.read_csv(predictions_path, dtype=str, keep_default_na=False, low_memory=False)
    diagnostics = compute_topology_diagnostics(
        features,
        predictions=predictions,
        model=model,
        k_neighbors=k_neighbors,
        config=config,
        predictions_required=predictions_required,
    )
    output = diagnostics if mode == "all" else select_mode_columns(diagnostics, mode)
    if include_features:
        merge_keys = [column for column in ("record_id", "domain_id", "pdb_id") if column in output.columns and column in features.columns]
        if merge_keys:
            output = features.merge(output, on=merge_keys, how="left", suffixes=("", "_readout"))
    output_path = Path(out_csv) if out_csv is not None else None
    if output_path is not None:
        _write_csv_atomic(output, output_path)

    written_manifest_path: Path | None = None
    if write_manifest:
        if manifest_path is None and output_path is not None and out_csv_provided:
            manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
        if manifest_path is None:
            manifest_path = cfg_get(config, "io.manifest_path", None)
        if manifest_path is None and output_path is not None:
            manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
        if manifest_path is not None:
            metrics = {
                "rows": int(len(output)),
                "mode": mode,
                "boundary_region_flags": int(pd.to_numeric(output.get("betlas_boundary_region_flag", 0), errors="coerce").fillna(0).sum())
                if "betlas_boundary_region_flag" in output
                else 0,
                "mixed_topology_flags": int(pd.to_numeric(output.get("betlas_mixed_topology_flag", 0), errors="coerce").fillna(0).sum())
                if "betlas_mixed_topology_flag" in output
                else 0,
            }
            manifest = build_run_manifest(
                command="betlas readout topology-diagnostics",
                parameters={
                    "mode": mode,
                    "model": model or "",
                    "k_neighbors": int(k_neighbors),
                    "include_features": bool(include_features),
                    "predictions_csv_requested": str(predictions_csv or ""),
                    "predictions_loaded": predictions is not None,
                },
                inputs={
                    "features_csv": features_csv,
                    **({"predictions_csv": predictions_csv} if predictions_csv is not None else {}),
                    **({"config_path": config_path} if config_path is not None else {}),
                },
                outputs={"diagnostics_csv": output_path} if output_path is not None else {},
                config=config,
                metrics=metrics,
                extra={
                    "readout": READOUT_NAME,
                    "cohort_level_fields": [
                        "betlas_neighbor_label_entropy",
                        "betlas_neighbor_disagreement_fraction",
                        "betlas_boundary_neighbor_fraction",
                        "betlas_boundary_neighbor_similarity",
                    ],
                    "label_aware_fields": [
                        "betlas_rule_label_conflict",
                        "betlas_neighbor_label_entropy",
                        "betlas_neighbor_disagreement_fraction",
                        "betlas_boundary_neighbor_fraction",
                    ],
                },
            )
            written_manifest_path = write_json(manifest_path, manifest)
    return TopologyDiagnosticsResult(
        diagnostics=output,
        output_csv=output_path,
        manifest_path=written_manifest_path,
    )
