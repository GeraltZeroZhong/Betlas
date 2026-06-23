from __future__ import annotations

import importlib.util
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from betlas.features.extract import assert_no_diagnostic_label_leakage, extract_feature_row
from betlas.ml.splits import make_grouped_splits
from betlas.models import DomainCandidate
from betlas.provenance import file_state, runtime_state, write_json
from betlas.schema import canonical_feature_name, normalize_feature_columns

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_BETLAS_BETA_ROOT = Path("data/external/betlas_beta")
DEFAULT_OUT_DIR = (
    REPO_ROOT / "data/readouts/beta_barrel_detection/betlas_151_layer_radial16_official"
)
DEFAULT_ESMC_NPZ = DEFAULT_OUT_DIR / "esmc_mean_embeddings_aligned.npz"

COMPARISON_CSV = (
    "eval_outputs/model_comparison_file_level_20260425_212649/"
    "file_level_model_comparison.csv"
)
PREDICTIONS_CSV = (
    "eval_outputs/model_comparison_file_level_20260425_212649/"
    "file_level_predictions_combined.csv"
)
BETLAS_BETA_CHAIN_CSV = (
    "eval_outputs/ablation_20260425_012308/"
    "eval_chain_results_20260425_012308_A11_production_full.csv"
)

LAYER_RADIAL16_RAW: list[str] = [
    "decision_score",
    "score_raw",
    "score_adjust",
    "valid_layers",
    "scored_layers",
    "total_layers",
    "valid_layer_frac",
    "scored_layer_frac",
    "junk_layers",
    "invalid_layers",
    "avg_radius",
    "chain_residues",
    "sheet_residues",
    "informative_slices",
    "pred_barrel_numeric",
    "filtered_or_error",
]


@dataclass(frozen=True)
class ReadoutPaths:
    betlas_beta_root: Path = DEFAULT_BETLAS_BETA_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    feature_columns: Path | None = None
    esmc_npz: Path = DEFAULT_ESMC_NPZ
    cohort_csv: Path | None = None
    features_csv: Path | None = None
    chain_results_csv: Path | None = None
    layer_values_csv: Path | None = None
    layer_manifest_csv: Path | None = None
    asset_id: str = ""
    asset_cache_dir: Path | None = None
    asset_manifest_verification: dict[str, Any] | None = None

    @property
    def positive_manifest(self) -> Path:
        return self.betlas_beta_root / "data/positive_manifest.csv"

    @property
    def negative_manifest(self) -> Path:
        return self.betlas_beta_root / "data/negative_manifest.csv"

    @property
    def file_predictions(self) -> Path:
        return self.betlas_beta_root / PREDICTIONS_CSV

    @property
    def file_comparison(self) -> Path:
        return self.betlas_beta_root / COMPARISON_CSV

    @property
    def betlas_beta_chain_results(self) -> Path:
        return self.betlas_beta_root / BETLAS_BETA_CHAIN_CSV

    @property
    def layer_radial_chain_results(self) -> Path:
        return self.chain_results_csv or self.betlas_beta_chain_results


@dataclass
class Evaluation:
    summary: dict[str, Any]
    fold_metrics: pd.DataFrame
    per_record: pd.DataFrame
    estimators: dict[int, Any]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dependency_status() -> dict[str, str]:
    return {"catboost": "available" if importlib.util.find_spec("catboost") else "missing"}


def write_dependency_preflight(out_dir: Path) -> dict[str, Any]:
    status = dependency_status()
    preflight = {
        "status": "ok" if status["catboost"] == "available" else "failed",
        "required_model_dependencies": ["catboost"],
        "model_dependency_status": status,
    }
    write_json(out_dir / "dependency_preflight.json", preflight)
    if status["catboost"] != "available":
        raise RuntimeError(
            "fixed-cohort beta-barrel detection readout requires CatBoost; "
            "install the reproducibility environment or run `python -m pip install catboost` "
            "before running this companion script"
        )
    return preflight


def layer_feature_name(raw: str) -> str:
    return f"layer_radial16__{raw}"


def read_feature_columns(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(f"Feature column file lacks `feature`: {path}")
    columns = [canonical_feature_name(column) for column in frame["feature"].astype(str).tolist()]
    if len(columns) != 151:
        raise ValueError(f"Expected Betlas 151 feature columns, found {len(columns)}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate Betlas feature column names in {path}")
    return columns


def _align_cached_frame(
    frame: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    path: Path,
    frame_name: str,
) -> pd.DataFrame:
    if "record_id" not in frame.columns:
        raise ValueError(f"{frame_name} lacks `record_id`: {path}")
    record_ids = cohort["record_id"].astype(str).tolist()
    cached_ids = frame["record_id"].astype(str)
    if cached_ids.duplicated().any():
        dupes = cached_ids.loc[cached_ids.duplicated()].head(8).tolist()
        raise ValueError(f"{frame_name} contains duplicate record_id values: {dupes}")
    missing = sorted(set(record_ids) - set(cached_ids))
    extra = sorted(set(cached_ids) - set(record_ids))
    if missing or extra:
        raise ValueError(
            f"{frame_name} record_id set does not match cohort "
            f"(missing={missing[:5]}, extra={extra[:5]}): {path}"
        )
    order = pd.Series(range(len(record_ids)), index=record_ids)
    aligned = frame.copy()
    aligned["_order"] = aligned["record_id"].astype(str).map(order)
    return aligned.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _safe_chain(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _safe_record_part(value: object) -> str:
    text = str(value).strip()
    out = []
    for char in text:
        out.append(char.lower() if char.isalnum() else "_")
    return "".join(out).strip("_") or "blank"


def _best_chain_by_file(chain_results: pd.DataFrame) -> dict[str, str]:
    if chain_results.empty:
        return {}
    frame = chain_results.copy()
    for column in ["decision_score", "score_adjust", "chain_residues", "sheet_residues"]:
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce").fillna(0.0)
    frame["_pred_barrel"] = frame["result"].astype(str).eq("BARREL").astype(int)
    frame = frame.sort_values(
        ["filename", "_pred_barrel", "decision_score", "score_adjust", "chain_residues", "chain"],
        ascending=[True, False, False, False, False, True],
    )
    return frame.drop_duplicates("filename").set_index("filename")["chain"].astype(str).to_dict()


def build_detection_cohort(paths: ReadoutPaths) -> pd.DataFrame:
    if paths.cohort_csv is not None:
        cohort = pd.read_csv(paths.cohort_csv)
        required = {
            "record_id",
            "filename",
            "pdb_id",
            "selected_chain_id",
            "label_chain_id",
            "fallback_chain_id",
            "manifest_source",
            "original_split",
            "corrected_split",
            "include_for_metrics",
            "y_true",
            "structure_path",
        }
        missing = sorted(required - set(cohort.columns))
        if missing:
            raise ValueError(f"Benchmark cohort is missing required columns: {missing}")
        cohort = cohort.copy()
        cohort["include_for_metrics"] = cohort["include_for_metrics"].astype(str).str.lower().isin(
            {"1", "true", "yes"}
        )
        cohort["y_true"] = pd.to_numeric(cohort["y_true"], errors="raise").astype(int)
        cohort["pdb_id"] = cohort["pdb_id"].astype(str).str.upper()
        cohort["structure_exists"] = cohort["structure_path"].astype(str).map(lambda item: Path(item).exists())
        if cohort["record_id"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate record_id values in benchmark cohort: {paths.cohort_csv}")
        if cohort["filename"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate filename values in benchmark cohort: {paths.cohort_csv}")
        return cohort

    predictions = pd.read_csv(paths.file_predictions)
    positives = pd.read_csv(paths.positive_manifest)
    negatives = pd.read_csv(paths.negative_manifest)
    chain_results_path = paths.layer_radial_chain_results
    if not chain_results_path.exists():
        raise FileNotFoundError(
            "LayerRadial16 chain-results CSV is required for this branch: "
            f"{display_path(chain_results_path)}"
        )
    chain_results = pd.read_csv(chain_results_path)
    best_chain = _best_chain_by_file(chain_results)

    pos_by_filename = positives.set_index("filename").to_dict(orient="index")
    neg_by_filename = negatives.set_index("filename").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for source in predictions.itertuples(index=False):
        filename = str(source.filename)
        original_split = str(source.original_split)
        if original_split == "positive":
            manifest_row = pos_by_filename[filename]
            structure_path = paths.betlas_beta_root / "data/positive" / filename
            selected_chain = _safe_chain(manifest_row.get("candidate_auth_asym_id", ""))
            label_chain = _safe_chain(manifest_row.get("candidate_label_asym_id", ""))
            manifest_source = "positive_manifest"
        elif original_split == "negative":
            manifest_row = neg_by_filename[filename]
            structure_path = paths.betlas_beta_root / "data/negative" / filename
            selected_chain = _safe_chain(manifest_row.get("chain", ""))
            label_chain = selected_chain
            manifest_source = "negative_manifest"
        else:
            raise ValueError(f"Unexpected original_split={original_split!r} for {filename}")

        corrected_split = str(source.corrected_split)
        include = bool(source.include_corrected) and corrected_split in {"positive", "negative"}
        y_true = 1 if corrected_split == "positive" else 0 if corrected_split == "negative" else -1
        fallback_chain = best_chain.get(filename, "")
        pdb_id = str(source.pdb_id).upper()
        rows.append(
            {
                "record_id": f"cbd_{_safe_record_part(Path(filename).stem)}_{_safe_record_part(selected_chain)}",
                "filename": filename,
                "pdb_id": pdb_id,
                "selected_chain_id": selected_chain,
                "label_chain_id": label_chain,
                "fallback_chain_id": fallback_chain,
                "manifest_source": manifest_source,
                "original_split": original_split,
                "corrected_split": corrected_split,
                "include_for_metrics": include,
                "y_true": y_true,
                "structure_path": str(structure_path),
                "structure_exists": structure_path.exists(),
                "correction_policy": str(source.correction_policy),
                "correction_reason": str(source.correction_reason),
            }
        )
    cohort = pd.DataFrame(rows)
    if cohort["filename"].duplicated().any():
        dupes = cohort.loc[cohort["filename"].duplicated(), "filename"].tolist()
        raise ValueError(f"Duplicate filenames in Betlas-Beta benchmark cohort: {dupes[:8]}")
    return cohort


def _domain_for(row: dict[str, Any], chain_id: str) -> DomainCandidate:
    return DomainCandidate(
        record_id=str(row["record_id"]),
        pdb_id=str(row["pdb_id"]).lower(),
        chain_id=str(chain_id),
        domain_id=str(row["record_id"]),
        residue_ranges="",
        fold_label_final="beta_barrel_detection_positive"
        if int(row.get("y_true", 0)) == 1
        else "beta_barrel_detection_negative",
        evidence_level="beta_barrel_detection_benchmark",
        label_source_primary="beta_barrel_detection_benchmark_manifest",
        qc_status="pass" if bool(row.get("include_for_metrics", False)) else "excluded",
        allowed_for_benchmark=bool(row.get("include_for_metrics", False)),
    )


def _candidate_chains_for_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    candidates = [
        ("selected_chain", _safe_chain(row.get("selected_chain_id", ""))),
        ("label_chain", _safe_chain(row.get("label_chain_id", ""))),
        ("fallback_chain", _safe_chain(row.get("fallback_chain_id", ""))),
    ]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, chain_id in candidates:
        if not chain_id or chain_id in seen:
            continue
        seen.add(chain_id)
        out.append((source, chain_id))
    return out


def _extract_one_feature_row(row: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    structure_path = Path(str(row["structure_path"]))
    best: dict[str, Any] | None = None
    best_source = ""
    best_chain = ""
    for chain_source, chain_id in _candidate_chains_for_row(row):
        domain = _domain_for(row, chain_id)
        extracted = extract_feature_row(domain, structure_path)
        raw_parse_ok = int(_numeric(extracted.get("betlas_parse_ok", 0)))
        beta_segment_count = _numeric(extracted.get("betlas_beta_strand_count", 0.0))
        extracted["betlas_geometry_parse_ok_raw"] = raw_parse_ok
        extracted["betlas_has_beta_segments"] = 1.0 if beta_segment_count > 0 else 0.0
        if not str(extracted.get("betlas_error", "") or "").strip():
            extracted["betlas_parse_ok"] = 1
        extracted["record_id"] = row["record_id"]
        extracted["filename"] = row["filename"]
        extracted["selected_chain_id"] = row["selected_chain_id"]
        extracted["extraction_chain_id"] = chain_id
        extracted["extraction_chain_source"] = chain_source
        extracted["structure_path"] = str(structure_path)
        extracted["structure_exists"] = int(structure_path.exists())
        if best is None or int(extracted.get("betlas_parse_ok", 0) or 0) > int(best.get("betlas_parse_ok", 0) or 0):
            best = extracted
            best_source = chain_source
            best_chain = chain_id
        if int(extracted.get("betlas_parse_ok", 0) or 0) == 1:
            break
    if best is None:
        domain = _domain_for(row, _safe_chain(row.get("selected_chain_id", "")))
        best = domain.to_dict()
        best["record_id"] = row["record_id"]
        best["filename"] = row["filename"]
        best["selected_chain_id"] = row["selected_chain_id"]
        best["extraction_chain_id"] = ""
        best["extraction_chain_source"] = "none"
        best["structure_path"] = str(structure_path)
        best["structure_exists"] = int(structure_path.exists())
        best["betlas_parse_ok"] = 0
        best["betlas_error"] = "no candidate chain id"
    else:
        best["extraction_chain_source"] = best_source
        best["extraction_chain_id"] = best_chain

    for feature in feature_columns:
        if feature not in best:
            best[feature] = 0.0
    assert_no_diagnostic_label_leakage(best)
    return best


def build_or_load_betlas_151(
    *,
    paths: ReadoutPaths,
    cohort: pd.DataFrame,
    feature_columns: list[str],
    workers: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    out_path = paths.out_dir / "betlas_151_chain_features.csv"
    if paths.features_csv is not None:
        frame = normalize_feature_columns(pd.read_csv(paths.features_csv, low_memory=False))
        frame = _align_cached_frame(
            frame,
            cohort,
            path=paths.features_csv,
            frame_name="Betlas 151 cached feature table",
        )
        missing_features = [column for column in feature_columns if column not in frame.columns]
        if missing_features:
            raise ValueError(
                "Betlas 151 cached feature table is missing required feature columns: "
                f"{missing_features[:8]}"
            )
        paths.out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        return frame
    if out_path.exists() and not force:
        frame = normalize_feature_columns(pd.read_csv(out_path))
        frame = _align_cached_frame(
            frame,
            cohort,
            path=out_path,
            frame_name="Betlas 151 generated feature cache",
        )
        missing_features = [column for column in feature_columns if column not in frame.columns]
        if missing_features:
            raise ValueError(
                "Betlas 151 generated feature cache is missing required feature columns: "
                f"{missing_features[:8]}"
            )
        frame.to_csv(out_path, index=False)
        return frame

    rows = cohort.to_dict(orient="records")
    feature_rows: list[dict[str, Any]] = []
    if workers <= 1:
        iterator = tqdm(rows, desc="Extracting Betlas 151", unit="chain")
        for row in iterator:
            feature_rows.append(_extract_one_feature_row(row, feature_columns))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_one_feature_row, row, feature_columns) for row in rows]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting Betlas 151",
                unit="chain",
            ):
                feature_rows.append(future.result())

    frame = pd.DataFrame(feature_rows)
    order = {record_id: index for index, record_id in enumerate(cohort["record_id"].astype(str))}
    frame["_order"] = frame["record_id"].astype(str).map(order)
    frame = frame.sort_values("_order").drop(columns=["_order"])
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def _numeric(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def build_or_load_layer_radial16(
    *,
    paths: ReadoutPaths,
    cohort: pd.DataFrame,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_path = paths.out_dir / "layer_radial16_feature_values.csv"
    manifest_path = paths.out_dir / "layer_radial16_feature_manifest.csv"
    if paths.layer_values_csv is not None and paths.layer_manifest_csv is not None:
        values = pd.read_csv(paths.layer_values_csv)
        manifest = pd.read_csv(paths.layer_manifest_csv)
        values = _align_cached_frame(
            values,
            cohort,
            path=paths.layer_values_csv,
            frame_name="LayerRadial16 cached feature table",
        )
        if "feature" not in manifest.columns:
            raise ValueError(f"LayerRadial16 manifest lacks `feature`: {paths.layer_manifest_csv}")
        missing_layer_columns = [
            column for column in manifest["feature"].astype(str).tolist() if column not in values.columns
        ]
        if missing_layer_columns:
            raise ValueError(
                "LayerRadial16 cached feature table is missing manifest columns: "
                f"{missing_layer_columns[:8]}"
            )
        paths.out_dir.mkdir(parents=True, exist_ok=True)
        values.to_csv(out_path, index=False)
        manifest.to_csv(manifest_path, index=False)
        return values, manifest
    if paths.layer_values_csv is not None or paths.layer_manifest_csv is not None:
        raise ValueError(
            "LayerRadial16 fixed inputs require both layer_values_csv and layer_manifest_csv."
        )
    if out_path.exists() and manifest_path.exists() and not force:
        values = pd.read_csv(out_path)
        manifest = pd.read_csv(manifest_path)
        values = _align_cached_frame(
            values,
            cohort,
            path=out_path,
            frame_name="LayerRadial16 generated feature cache",
        )
        if "feature" not in manifest.columns:
            raise ValueError(f"LayerRadial16 generated manifest lacks `feature`: {manifest_path}")
        missing_layer_columns = [
            column for column in manifest["feature"].astype(str).tolist() if column not in values.columns
        ]
        if missing_layer_columns:
            raise ValueError(
                "LayerRadial16 generated feature cache is missing manifest columns: "
                f"{missing_layer_columns[:8]}"
            )
        values.to_csv(out_path, index=False)
        return values, manifest

    chain_results = pd.read_csv(paths.layer_radial_chain_results)
    chain_results["filename"] = chain_results["filename"].astype(str)
    chain_results["chain"] = chain_results["chain"].astype(str)
    keyed = {
        (str(row.filename), str(row.chain)): row._asdict()
        for row in chain_results.itertuples(index=False)
    }
    best_by_file = chain_results.copy()
    for column in ["decision_score", "score_adjust", "chain_residues", "sheet_residues"]:
        best_by_file[column] = pd.to_numeric(best_by_file.get(column, 0), errors="coerce").fillna(0.0)
    best_by_file["_pred_barrel"] = best_by_file["result"].astype(str).eq("BARREL").astype(int)
    best_by_file = best_by_file.sort_values(
        ["filename", "_pred_barrel", "decision_score", "score_adjust", "chain_residues", "chain"],
        ascending=[True, False, False, False, False, True],
    ).drop_duplicates("filename")
    best_rows = {str(row.filename): row._asdict() for row in best_by_file.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    for item in cohort.itertuples(index=False):
        filename = str(item.filename)
        selected_chain = str(item.selected_chain_id)
        source_row = keyed.get((filename, selected_chain))
        source = "selected_chain"
        if source_row is None:
            source_row = best_rows.get(filename, {})
            source = "file_level_best_chain" if source_row else "missing"
        pred_barrel = str(source_row.get("pred_barrel", source_row.get("result", ""))).lower()
        filtered_or_error = str(source_row.get("result", "")).upper() in {
            "FILTERED_OUT",
            "ERROR",
            "SKIP",
        }
        values: dict[str, Any] = {
            "record_id": item.record_id,
            "filename": filename,
            "selected_chain_id": selected_chain,
            "layer_radial16_source": source,
            "layer_radial16_chain_id": source_row.get("chain", ""),
            layer_feature_name("pred_barrel_numeric"): 1.0
            if pred_barrel in {"true", "1", "barrel"}
            else 0.0,
            layer_feature_name("filtered_or_error"): 1.0 if filtered_or_error else 0.0,
        }
        for raw in LAYER_RADIAL16_RAW:
            feature = layer_feature_name(raw)
            if feature in values:
                continue
            values[feature] = _numeric(source_row.get(raw, 0.0))
        rows.append(values)

    values = pd.DataFrame(rows)
    manifest = pd.DataFrame(
        [
            {
                "feature": layer_feature_name(raw),
                "source_feature": raw,
                "feature_block": "LayerRadial16",
                "feature_family": "layer_radial_detection",
                "task": "beta_barrel_detection",
                "description": "Layer/radial detector diagnostic promoted as a Betlas auxiliary feature.",
            }
            for raw in LAYER_RADIAL16_RAW
        ]
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(out_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    return values, manifest


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    out = {
        "n_used": int(len(y_true)),
        "n_positive_files": int(np.sum(y_true == 1)),
        "n_negative_files": int(np.sum(y_true == 0)),
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
    }
    if y_score is not None:
        out["score_mean"] = float(np.mean(y_score)) if len(y_score) else 0.0
    return out


def make_catboost(*, iterations: int, seed: int, jobs: int):
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


def positive_class_probability(model: object, x_test: pd.DataFrame, *, fold: int) -> tuple[np.ndarray, str]:
    proba = np.asarray(model.predict_proba(x_test), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if proba.ndim != 2:
        return np.zeros(len(x_test), dtype=float), f"fold {fold}: predict_proba returned non-matrix output"
    if 1 in classes:
        return proba[:, classes.index(1)], ""
    if proba.shape[1] > 1 and not classes:
        return proba[:, 1], f"fold {fold}: classes_ unavailable; used column 1 as compatibility fallback"
    return np.zeros(proba.shape[0], dtype=float), f"fold {fold}: positive class 1 absent from estimator classes_"


def _coerce_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out.astype(float)


def load_esmc_embeddings(path: Path, record_ids: list[str]) -> tuple[pd.DataFrame, pd.Series] | None:
    if not path.exists():
        return None
    loaded = np.load(path, allow_pickle=False)
    id_key = "record_id" if "record_id" in loaded.files else "sample_id" if "sample_id" in loaded.files else ""
    if not id_key or "embeddings" not in loaded.files:
        raise ValueError(f"ESM-C cache must contain record_id/sample_id and embeddings arrays: {path}")
    ids = [str(value) for value in loaded[id_key].tolist()]
    embeddings = pd.DataFrame(loaded["embeddings"], index=ids)
    missing = sorted(set(record_ids) - set(ids))
    if missing:
        raise ValueError(f"ESM-C cache is missing {len(missing)} record ids; first={missing[:5]}")
    aligned = embeddings.loc[record_ids]
    if "esmc_available" in loaded.files:
        available_values = pd.Series(loaded["esmc_available"].astype(float), index=ids)
        available = available_values.loc[record_ids].rename("esmc_available")
    else:
        available = pd.Series(1.0, index=record_ids, name="esmc_available")
    return aligned, available


def _esmc_pca_by_fold(
    embeddings: pd.DataFrame,
    folds: np.ndarray,
    *,
    pca_dim: int,
    seed: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], int]:
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
    return train_pcs, test_pcs, actual_dim


def evaluate_catboost_readout(
    *,
    experiment: str,
    display_label: str,
    base_x: pd.DataFrame,
    layer_x: pd.DataFrame | None,
    cohort: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    include_esmc: bool,
    esmc_npz: Path,
    pca_dim: int,
    n_splits: int,
    iterations: int,
    seed: int,
    jobs: int,
) -> Evaluation | dict[str, Any]:
    start = time.perf_counter()
    feature_count = int(base_x.shape[1] + (0 if layer_x is None else layer_x.shape[1]))
    esmc_loaded = load_esmc_embeddings(esmc_npz, cohort["record_id"].astype(str).tolist()) if include_esmc else None
    if include_esmc and esmc_loaded is None:
        return {
            "experiment": experiment,
            "display_label": display_label,
            "status": "skipped",
            "skip_reason": f"ESM-C cache not found: {display_path(esmc_npz)}",
            "feature_count": feature_count,
            "include_esmc": True,
            "esmc_pca_dim": int(pca_dim),
            "runtime_seconds": round(time.perf_counter() - start, 3),
        }

    split_rows, split_strategy, effective_splits = make_grouped_splits(
        base_x,
        y,
        groups,
        n_splits=int(n_splits),
        random_state=int(seed),
    )
    fold_ids = np.zeros(len(y), dtype=int)
    for fold, (_train_index, test_index) in enumerate(split_rows):
        fold_ids[test_index] = fold

    train_pcs: dict[int, pd.DataFrame] = {}
    test_pcs: dict[int, pd.DataFrame] = {}
    esmc_available = pd.Series(np.zeros(len(y), dtype=float))
    actual_pca_dim = 0
    if include_esmc and esmc_loaded is not None:
        embeddings, available_by_record = esmc_loaded
        train_pcs, test_pcs, actual_pca_dim = _esmc_pca_by_fold(
            embeddings,
            fold_ids,
            pca_dim=int(pca_dim),
            seed=int(seed),
        )
        esmc_available = available_by_record.reset_index(drop=True)
        feature_count += actual_pca_dim + 1

    predictions = np.zeros(len(y), dtype=int)
    scores = np.zeros(len(y), dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    estimators: dict[int, Any] = {}
    probability_warnings: list[str] = []
    for fold, (train_index, test_index) in enumerate(split_rows):
        x_train_parts = [base_x.iloc[train_index].reset_index(drop=True)]
        x_test_parts = [base_x.iloc[test_index].reset_index(drop=True)]
        if layer_x is not None:
            x_train_parts.append(layer_x.iloc[train_index].reset_index(drop=True))
            x_test_parts.append(layer_x.iloc[test_index].reset_index(drop=True))
        if include_esmc:
            x_train_parts.append(train_pcs[fold].reset_index(drop=True))
            x_test_parts.append(test_pcs[fold].reset_index(drop=True))
            x_train_parts.append(pd.DataFrame({"esmc_available": esmc_available.iloc[train_index].to_numpy(float)}))
            x_test_parts.append(pd.DataFrame({"esmc_available": esmc_available.iloc[test_index].to_numpy(float)}))
        x_train = pd.concat(x_train_parts, axis=1)
        x_test = pd.concat(x_test_parts, axis=1)
        model = make_catboost(iterations=iterations, seed=seed + fold, jobs=jobs)
        model.fit(x_train, y[train_index])
        fold_scores, warning = positive_class_probability(model, x_test, fold=fold)
        if warning:
            probability_warnings.append(warning)
        fold_pred = (fold_scores >= 0.5).astype(int)
        predictions[test_index] = fold_pred
        scores[test_index] = fold_scores
        row = _metric_row(y[test_index], fold_pred, fold_scores)
        row.update(
            {
                "experiment": experiment,
                "display_label": display_label,
                "outer_fold": int(fold),
                "split_strategy": split_strategy,
                "effective_n_splits": int(effective_splits),
                "probability_alignment_warning": warning,
                "probability_calibration_status": "model_reported_uncalibrated",
            }
        )
        fold_metrics.append(row)
        estimators[fold] = model

    summary = _metric_row(y, predictions, scores)
    summary.update(
        {
            "experiment": experiment,
            "display_label": display_label,
            "status": "ok",
            "skip_reason": "",
            "feature_count": int(feature_count),
            "betlas_feature_count": int(base_x.shape[1]),
            "layer_radial16_feature_count": int(0 if layer_x is None else layer_x.shape[1]),
            "include_esmc": bool(include_esmc),
            "esmc_pca_dim": int(actual_pca_dim),
            "iterations": int(iterations),
            "split_strategy": split_strategy,
            "effective_n_splits": int(effective_splits),
            "probability_alignment_warnings": ";".join(probability_warnings),
            "probability_calibration_status": "model_reported_uncalibrated",
            "runtime_seconds": round(time.perf_counter() - start, 3),
        }
    )
    per_record = cohort.copy().reset_index(drop=True)
    per_record["experiment"] = experiment
    per_record["display_label"] = display_label
    per_record["outer_fold"] = fold_ids
    per_record["pred_barrel"] = predictions.astype(bool)
    per_record["pred_label"] = np.where(predictions == 1, "positive", "negative")
    per_record["prob_barrel"] = scores
    per_record["probability_source"] = "catboost_predict_proba"
    per_record["probability_calibration_status"] = "model_reported_uncalibrated"
    per_record["correct"] = predictions == y
    return Evaluation(summary, pd.DataFrame(fold_metrics), per_record, estimators)


def _external_corrected_summary(paths: ReadoutPaths) -> pd.DataFrame:
    columns = [
        "experiment",
        "display_label",
        "status",
        "feature_count",
        "n_used",
        "n_positive_files",
        "n_negative_files",
        "TP",
        "FP",
        "TN",
        "FN",
        "recall",
        "precision",
        "f1",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "notes",
    ]
    if paths.cohort_csv is not None:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(paths.file_comparison)
    keep = frame.loc[frame["scope"].astype(str) == "corrected"].copy()
    keep["experiment"] = keep["model_key"].astype(str)
    keep["display_label"] = keep["model"].astype(str)
    keep["status"] = "ok"
    keep["feature_count"] = np.nan
    return keep[columns].reset_index(drop=True)


def write_summary_markdown(path: Path, official: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Betlas 151 Beta-Barrel Detection Readout",
        "",
        "- Task: beta-barrel detection benchmark.",
        "- Classifier: grouped out-of-fold CatBoost binary classifier.",
        "- Probability columns are estimator-reported, uncalibrated CatBoost outputs.",
        "- Active Betlas feature progression: Betlas 151; Betlas 151 + LayerRadial16; Betlas 151 + LayerRadial16 + ESM-C when a matching cache is supplied.",
        "",
        "## Active Readouts",
        "",
        "| model | status | features | n | F1 | balanced accuracy | recall | precision | specificity | MCC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in official.itertuples(index=False):
        if row.status != "ok":
            lines.append(
                f"| `{row.display_label}` | skipped | {int(row.feature_count)} |  |  |  |  |  |  |  |"
            )
            continue
        lines.append(
            f"| `{row.display_label}` | ok | {int(row.feature_count)} | {int(row.n_used)} | "
            f"{float(row.f1):.3f} | {float(row.balanced_accuracy):.3f} | "
            f"{float(row.recall):.3f} | {float(row.precision):.3f} | "
            f"{float(row.specificity):.3f} | {float(row.mcc):.3f} |"
        )
    skipped = official.loc[official["status"] != "ok"]
    if not skipped.empty:
        lines.extend(["", "## Skipped Active Branches", ""])
        for row in skipped.itertuples(index=False):
            lines.append(f"- `{row.display_label}`: {row.skip_reason}")
    lines.extend(
        [
            "",
            "## Comparison With External Baselines",
            "",
            "| model | n | F1 | balanced accuracy | recall | precision | specificity | MCC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.itertuples(index=False):
        if row.status != "ok":
            continue
        lines.append(
            f"| `{row.display_label}` | {int(row.n_used)} | {float(row.f1):.3f} | "
            f"{float(row.balanced_accuracy):.3f} | {float(row.recall):.3f} | "
            f"{float(row.precision):.3f} | {float(row.specificity):.3f} | {float(row.mcc):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_official_detection_readout(
    *,
    paths: ReadoutPaths,
    workers: int = 1,
    force_extract: bool = False,
    n_splits: int = 5,
    iterations: int = 500,
    seed: int = 17,
    jobs: int = -1,
    pca_dim: int = 256,
) -> dict[str, Any]:
    run_start = time.perf_counter()
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    dependency_preflight = write_dependency_preflight(paths.out_dir)
    if paths.feature_columns is None:
        raise ValueError(
            "Betlas 151 detection readout requires --asset-id or an explicit "
            "--feature-columns CSV with exactly the 151 ordered Betlas feature names."
        )
    feature_columns = read_feature_columns(paths.feature_columns)
    cohort = build_detection_cohort(paths)
    cohort.to_csv(paths.out_dir / "benchmark_cohort.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(paths.out_dir / "feature_columns_151.csv", index=False)

    features = build_or_load_betlas_151(
        paths=paths,
        cohort=cohort,
        feature_columns=feature_columns,
        workers=int(workers),
        force=bool(force_extract),
    )
    layer_values, layer_manifest = build_or_load_layer_radial16(
        paths=paths,
        cohort=cohort,
        force=bool(force_extract),
    )

    metric_mask = cohort["include_for_metrics"].astype(bool).to_numpy()
    metric_cohort = cohort.loc[metric_mask].reset_index(drop=True)
    metric_features = features.loc[metric_mask].reset_index(drop=True)
    metric_layer_values = layer_values.loc[metric_mask].reset_index(drop=True)
    x151 = _coerce_numeric_frame(metric_features, feature_columns)
    layer_columns = layer_manifest["feature"].astype(str).tolist()
    layer_x = _coerce_numeric_frame(metric_layer_values, layer_columns)
    y = metric_cohort["y_true"].to_numpy(dtype=int)
    groups = metric_cohort["pdb_id"].astype(str).to_numpy()

    specs = [
        ("betlas_151_catboost", "Betlas 151", x151, None, False),
        ("betlas_151_layer_radial16_catboost", "Betlas 151 + LayerRadial16", x151, layer_x, False),
        (
            "betlas_151_layer_radial16_esmc_catboost",
            "Betlas 151 + LayerRadial16 + ESM-C",
            x151,
            layer_x,
            True,
        ),
    ]
    evaluations: list[Evaluation] = []
    summary_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for experiment, display_label, base_x, selected_layer_x, include_esmc in specs:
        result = evaluate_catboost_readout(
            experiment=experiment,
            display_label=display_label,
            base_x=base_x,
            layer_x=selected_layer_x,
            cohort=metric_cohort,
            y=y,
            groups=groups,
            include_esmc=include_esmc,
            esmc_npz=paths.esmc_npz,
            pca_dim=int(pca_dim),
            n_splits=int(n_splits),
            iterations=int(iterations),
            seed=int(seed),
            jobs=int(jobs),
        )
        if isinstance(result, Evaluation):
            evaluations.append(result)
            summary_rows.append(result.summary)
        else:
            skipped_rows.append(result)
            summary_rows.append(result)

    official = pd.DataFrame(summary_rows)
    official.to_csv(paths.out_dir / "official_summary.csv", index=False)
    if evaluations:
        pd.concat([item.fold_metrics for item in evaluations], ignore_index=True).to_csv(
            paths.out_dir / "official_fold_metrics.csv",
            index=False,
        )
        pd.concat([item.per_record for item in evaluations], ignore_index=True).to_csv(
            paths.out_dir / "official_per_record_predictions.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(paths.out_dir / "official_fold_metrics.csv", index=False)
        pd.DataFrame().to_csv(paths.out_dir / "official_per_record_predictions.csv", index=False)

    external = _external_corrected_summary(paths)
    official_for_comparison = official.loc[
        official["status"] == "ok",
        external.columns.intersection(official.columns),
    ].reindex(columns=external.columns)
    if external.empty:
        comparison = official_for_comparison.reset_index(drop=True)
    else:
        comparison = pd.concat([official_for_comparison, external], ignore_index=True)
    comparison = comparison.sort_values(
        ["f1", "balanced_accuracy", "mcc"],
        ascending=[False, False, False],
        na_position="last",
    )
    comparison.to_csv(paths.out_dir / "external_baseline_comparison.csv", index=False)
    write_summary_markdown(paths.out_dir / "summary.md", official=official, comparison=comparison)

    parse_ok = pd.to_numeric(features.get("betlas_parse_ok", 0), errors="coerce").fillna(0).astype(int)
    uses_cached_layer = paths.layer_values_csv is not None and paths.layer_manifest_csv is not None
    if paths.cohort_csv is not None:
        input_state = {
            "cohort_csv": file_state(paths.cohort_csv),
            **({"features_csv": file_state(paths.features_csv)} if paths.features_csv is not None else {}),
            **({"layer_values_csv": file_state(paths.layer_values_csv)} if paths.layer_values_csv is not None else {}),
            **({"layer_manifest_csv": file_state(paths.layer_manifest_csv)} if paths.layer_manifest_csv is not None else {}),
        }
        if not uses_cached_layer or paths.chain_results_csv is not None:
            input_state["layer_radial_chain_results"] = file_state(paths.layer_radial_chain_results)
        source_root = ""
    else:
        input_state = {
            "positive_manifest": file_state(paths.positive_manifest),
            "negative_manifest": file_state(paths.negative_manifest),
            "file_predictions": file_state(paths.file_predictions),
            "file_comparison": file_state(paths.file_comparison),
            "layer_radial_chain_results": file_state(paths.layer_radial_chain_results),
        }
        source_root = str(paths.betlas_beta_root)
    metadata = {
        "schema": "betlas.beta-barrel-detection.betlas-151-layer-radial16.v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohort_rows": int(len(cohort)),
        "metric_rows": int(metric_mask.sum()),
        "positive_metric_rows": int(np.sum(y == 1)),
        "negative_metric_rows": int(np.sum(y == 0)),
        "betlas_151_parse_ok": int(parse_ok.sum()),
        "betlas_151_parse_ok_rate": float(parse_ok.mean()) if len(parse_ok) else 0.0,
        "layer_radial16_features": layer_columns,
        "feature_columns": display_path(paths.feature_columns) if paths.feature_columns is not None else "",
        "source_root": source_root,
        "asset_id": paths.asset_id,
        "asset_cache_dir": display_path(paths.asset_cache_dir) if paths.asset_cache_dir is not None else "",
        "asset_manifest_verification": paths.asset_manifest_verification or {},
        "dependency_preflight": "dependency_preflight.json",
        "model_dependency_status": dependency_preflight["model_dependency_status"],
        "probability_calibration_status": "model_reported_uncalibrated",
        "cohort_csv": display_path(paths.cohort_csv) if paths.cohort_csv else "",
        "layer_radial_chain_results": (
            display_path(paths.layer_radial_chain_results)
            if (not uses_cached_layer or paths.chain_results_csv is not None)
            else ""
        ),
        "out_dir": display_path(paths.out_dir),
        "esmc_npz": display_path(paths.esmc_npz),
        "esmc_cache_exists": paths.esmc_npz.exists(),
        "skipped": skipped_rows,
        "parameters": {
            "workers": int(workers),
            "n_splits": int(n_splits),
            "iterations": int(iterations),
            "seed": int(seed),
            "jobs": int(jobs),
            "pca_dim": int(pca_dim),
        },
        "pipeline_runtime_seconds": round(time.perf_counter() - run_start, 3),
        "inputs": input_state,
        "runtime": runtime_state(),
    }
    write_json(paths.out_dir / "metadata.json", metadata)
    return metadata


__all__ = [
    "DEFAULT_OUT_DIR",
    "LAYER_RADIAL16_RAW",
    "ReadoutPaths",
    "build_detection_cohort",
    "build_or_load_betlas_151",
    "build_or_load_layer_radial16",
    "run_official_detection_readout",
]
