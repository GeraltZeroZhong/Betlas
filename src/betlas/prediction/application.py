"""Checkpoint-backed solenoid, barrel and barrel-wall strand prediction."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .checkpoints import _task, resolve_checkpoint
from .models import evaluate, load_model


@lru_cache(maxsize=8)
def _cached_model(path: str, task: str, modified: int, size: int) -> dict[str, Any]:
    return load_model(path, task)


def _load(task: str, checkpoint, cache_dir, allow_download, base_url):
    path = resolve_checkpoint(
        task,
        checkpoint=checkpoint,
        cache_dir=cache_dir,
        allow_download=allow_download,
        base_url=base_url,
    ).resolve()
    stat = path.stat()
    return _cached_model(str(path), task, stat.st_mtime_ns, stat.st_size)


def _gate(features, *, barrel_checkpoint, cache_dir, allow_download, base_url):
    model = _load("barrel", barrel_checkpoint, cache_dir, allow_download, base_url)
    result = evaluate(model, features)
    if not result["is_barrel"]:
        return {
            "task": "barrel-strands",
            "status": "not_barrel",
            "prediction": None,
            "strand_count": None,
            "probability": None,
            "probabilities": {},
            "is_barrel": False,
            "barrel_probability": result["probability"],
            "feature_schema": result["feature_schema"],
            "checkpoint_release": result["checkpoint_release"],
            "imputed_feature_count": result["imputed_feature_count"],
            "imputed_features": result["imputed_features"],
        }
    return result


def predict_features(
    task: str,
    features: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    base_url: str | Path | None = None,
    assume_barrel: bool = False,
    barrel_checkpoint: str | Path | None = None,
    barrel_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict one descriptor mapping, automatically installing release weights.

    Missing measurements are NaN within present columns. Counting additionally
    needs the 16 layer/radial columns and a positive barrel call. Set
    ``assume_barrel`` to count directly on a caller-confirmed barrel.
    """
    _task(task)
    barrel = None
    if task == "barrel-strands" and not assume_barrel:
        if barrel_features is None:
            raise ValueError(
                "Counting from features requires assume_barrel=True or separately measured barrel_features"
            )
        barrel = _gate(
            barrel_features,
            barrel_checkpoint=barrel_checkpoint,
            cache_dir=cache_dir,
            allow_download=allow_download,
            base_url=base_url,
        )
        if barrel["status"] == "not_barrel":
            return barrel
    model = _load(task, checkpoint, cache_dir, allow_download, base_url)
    result = evaluate(model, features)
    if task == "barrel-strands":
        result["barrel_gate"] = "caller_confirmed" if assume_barrel else "predicted"
        if barrel is not None:
            result.update(is_barrel=True, barrel_probability=barrel["probability"])
    return result


def predict_structure(
    task: str,
    structure: str | Path,
    *,
    chain_id: str,
    model_id: int = 0,
    residue_ranges: str = "",
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    base_url: str | Path | None = None,
    dssp_binary: str | Path = "mkdssp",
    timeout: float = 300,
    assume_barrel: bool = False,
    barrel_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Measure a local structure and apply a released task model."""
    from .structure import layer_radial_features, measured_structure

    _task(task)
    if residue_ranges and task != "solenoid":
        raise ValueError(
            "Residue ranges are supported for solenoid domains; barrel tasks use the complete chain"
        )
    barrel = None
    if task == "barrel-strands" and not assume_barrel:
        barrel = predict_structure(
            "barrel",
            structure,
            chain_id=chain_id,
            model_id=model_id,
            checkpoint=barrel_checkpoint,
            cache_dir=cache_dir,
            allow_download=allow_download,
            base_url=base_url,
            dssp_binary=dssp_binary,
            timeout=timeout,
        )
        if not barrel["is_barrel"]:
            return {
                **barrel,
                "task": task,
                "status": "not_barrel",
                "prediction": None,
                "strand_count": None,
                "barrel_probability": barrel["probability"],
                "probability": None,
                "probabilities": {},
            }
    with measured_structure(
        structure,
        task=task,
        chain_id=chain_id,
        model_id=model_id,
        residue_ranges=residue_ranges,
        dssp_binary=dssp_binary,
        timeout=timeout,
    ) as (features, selected, metadata):
        model = _load(task, checkpoint, cache_dir, allow_download, base_url)
        if task == "barrel-strands":
            extra = [
                column
                for column in model["feature_columns"]
                if column.startswith("layer_radial16__")
            ]
            features.update(layer_radial_features(selected, chain_id, extra, dssp_binary))
        result = evaluate(model, features)
        if task == "barrel-strands":
            result["barrel_gate"] = "caller_confirmed" if assume_barrel else "predicted"
            if barrel is not None:
                result.update(is_barrel=True, barrel_probability=barrel["probability"])
        return {**result, **metadata}
