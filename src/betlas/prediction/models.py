"""Numerical inference for portable logistic and symmetric-tree checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, softmax

from .checkpoints import CheckpointError

CHECKPOINT_SCHEMA = "betlas.predictor.v1"


def load_model(path: str | Path, task: str) -> dict[str, Any]:
    """Load JSON weights without deserializing executable Python objects."""
    try:
        model = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"Could not read predictor JSON: {exc}") from exc
    if not isinstance(model, dict):
        raise CheckpointError("Checkpoint JSON must contain an object")
    if model.get("schema_version") != CHECKPOINT_SCHEMA or model.get("task") != task:
        raise CheckpointError("Checkpoint schema or prediction task does not match")
    columns = model.get("feature_columns", [])
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(c, str) and c for c in columns)
        or len(columns) != len(set(columns))
    ):
        raise CheckpointError("Checkpoint feature columns must be a nonempty unique list")
    classes = model.get("classes", [])
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(c, int) and not isinstance(c, bool) for c in classes)
        or len(set(classes)) != len(classes)
    ):
        raise CheckpointError("Checkpoint classes must be distinct")
    if not isinstance(model.get("model"), dict) or model["model"].get("kind") not in (
        "logistic",
        "catboost",
    ):
        raise CheckpointError(
            "Supported checkpoints use logistic or numerical symmetric-tree models"
        )
    return model


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise CheckpointError(f"Invalid checkpoint parameter: {name}")
    return values


def probabilities(
    checkpoint: Mapping[str, Any], features: Mapping[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """Evaluate one complete descriptor mapping in checkpoint column order."""
    columns = checkpoint["feature_columns"]
    missing = [column for column in columns if column not in features]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required descriptor columns: {', '.join(missing[:5])}"
        )
    try:
        values = np.asarray(
            [np.nan if features[c] is None else features[c] for c in columns], dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictor inputs must be numeric descriptor values") from exc
    unavailable = [c for c, v in zip(columns, values, strict=True) if not np.isfinite(v)]
    finite = np.isfinite(values)
    model = checkpoint["model"]
    width, count = len(columns), len(checkpoint["classes"])
    if model["kind"] == "logistic":
        imputer = _array(model["imputer_statistics"], (width,), "imputer_statistics")
        mean = _array(model["scaler_mean"], (width,), "scaler_mean")
        scale = _array(model["scaler_scale"], (width,), "scaler_scale")
        if np.any(scale <= 0):
            raise CheckpointError("Checkpoint scaling values must be positive")
        rows = 1 if count == 2 else count
        coef = _array(model["coef"], (rows, width), "coef")
        intercept = _array(model["intercept"], (rows,), "intercept")
        standardized = (np.where(finite, values, imputer) - mean) / scale
        scores = coef @ standardized + intercept
        result = (
            np.array([1 - expit(scores[0]), expit(scores[0])]) if rows == 1 else softmax(scores)
        )
    else:
        fill = float(model.get("fill_value", 0.0))
        if not np.isfinite(fill):
            raise CheckpointError("Tree-model fill value must be finite")
        values = np.where(finite, values, fill).astype(np.float32)
        tree_model = model["catboost"]
        infos = tree_model["features_info"]
        if infos.get("categorical_features") or infos.get("ctrs") or infos.get("text_features"):
            raise CheckpointError("Predictor supports numerical symmetric trees only")
        feature_map = {
            int(f["feature_index"]): int(f["flat_feature_index"]) for f in infos["float_features"]
        }
        scores = np.zeros(count, dtype=float)
        for tree in tree_model["oblivious_trees"]:
            index = 0
            splits = tree.get("splits") or []
            for bit, split in enumerate(splits):
                if split["split_type"] != "FloatFeature":
                    raise CheckpointError("Predictor encountered a nonnumeric tree split")
                feature = feature_map[int(split["float_feature_index"])]
                if not 0 <= feature < width:
                    raise CheckpointError(
                        "Tree feature index is outside the declared feature columns"
                    )
                index |= int(values[feature] > float(split["border"])) << bit
            leaves = _array(tree["leaf_values"], ((1 << len(splits)) * count,), "leaf_values")
            scores += leaves.reshape(-1, count)[index]
        scale, bias = tree_model["scale_and_bias"]
        scores = float(scale) * scores + _array(bias, (count,), "tree bias")
        result = softmax(scores)
    if not np.isfinite(result).all() or not np.isclose(result.sum(), 1.0):
        raise CheckpointError("Predictor produced invalid probabilities")
    return result, unavailable


def evaluate(checkpoint: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    schema = features.get("feature_schema")
    if schema is not None and schema != checkpoint["feature_schema"]:
        raise ValueError("Feature schema differs from the selected checkpoint")
    probs, unavailable = probabilities(checkpoint, features)
    if len(unavailable) == len(checkpoint["feature_columns"]):
        raise ValueError("All descriptor measurements are unavailable")
    task, classes = checkpoint["task"], checkpoint["classes"]
    if task in ("solenoid", "barrel"):
        if classes != [0, 1]:
            raise CheckpointError("Recognition checkpoints require ordered classes [0, 1]")
        threshold = float(checkpoint.get("threshold", 0.5))
        if not 0 <= threshold <= 1:
            raise CheckpointError("Recognition threshold must be between zero and one")
        label = int(probs[1] >= threshold)
    else:
        label = int(classes[int(np.argmax(probs))])
    result = {
        "task": task,
        "status": "ok",
        "prediction": label,
        "probabilities": {str(c): float(v) for c, v in zip(classes, probs, strict=True)},
        "feature_schema": checkpoint["feature_schema"],
        "checkpoint_release": checkpoint["release"],
        "imputed_feature_count": len(unavailable),
        "imputed_features": unavailable,
    }
    if task == "barrel-strands":
        result.update(strand_count=label, probability=float(np.max(probs)))
    else:
        result.update(
            {
                "is_solenoid" if task == "solenoid" else "is_barrel": bool(label),
                "probability": float(probs[1]),
                "threshold": threshold,
            }
        )
    return result
