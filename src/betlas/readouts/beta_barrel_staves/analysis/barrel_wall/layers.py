from __future__ import annotations

from typing import Any

import numpy as np

from ...constants import EPSILON
from ...geometry.analysis_utils import collapse_points_by_strand


def radial_center(points_xy: np.ndarray, *, outer_min_points: int) -> np.ndarray:
    center = np.median(points_xy, axis=0)
    if points_xy.shape[0] < outer_min_points:
        return center

    radius = np.linalg.norm(points_xy - center, axis=1)
    outer_points = points_xy[radius >= np.quantile(radius, 0.50)]
    if outer_points.shape[0] >= max(4, outer_min_points // 2):
        return np.mean(outer_points, axis=0)
    return center


def quantile_outer_mask(radius: np.ndarray, outer_quantile: float) -> np.ndarray:
    threshold = np.quantile(radius, outer_quantile)
    return radius >= threshold


def radial_gap_outer_mask(
    radius: np.ndarray,
    *,
    min_layer_points: int,
    min_removed_points: int,
    min_gap_fraction: float,
) -> np.ndarray | None:
    if radius.shape[0] < min_layer_points + min_removed_points:
        return None

    order = np.argsort(radius, kind="mergesort")
    sorted_radius = radius[order]
    gaps = np.diff(sorted_radius)
    if gaps.size == 0:
        return None

    split_index = int(np.argmax(gaps))
    inner_count = split_index + 1
    outer_count = int(radius.shape[0] - inner_count)
    if inner_count < min_removed_points or outer_count < min_layer_points:
        return None

    radial_scale = max(float(np.median(sorted_radius)), EPSILON)
    if float(gaps[split_index]) < float(min_gap_fraction) * radial_scale:
        return None

    keep_mask = np.zeros(radius.shape[0], dtype=bool)
    keep_mask[order[split_index + 1 :]] = True
    return keep_mask


def ordered_layer_ids(
    points: np.ndarray,
    *,
    graph_config: Any,
    outer_filter_strategy: str,
) -> list[int]:
    return [
        strand_id
        for strand_id, _radial_rank in ordered_layer_observations(
            points,
            graph_config=graph_config,
            outer_filter_strategy=outer_filter_strategy,
        )
    ]


def ordered_layer_observations(
    points: np.ndarray,
    *,
    graph_config: Any,
    outer_filter_strategy: str,
) -> list[tuple[int, float]]:
    return [
        (strand_id, radial_rank)
        for strand_id, radial_rank, _angle in ordered_layer_angle_observations(
            points,
            graph_config=graph_config,
            outer_filter_strategy=outer_filter_strategy,
        )
    ]


def ordered_layer_angle_observations(
    points: np.ndarray,
    *,
    graph_config: Any,
    outer_filter_strategy: str,
) -> list[tuple[int, float, float]]:
    min_layer_points = int(getattr(graph_config, "min_layer_points", 4))
    collapsed = collapse_points_by_strand(np.asarray(points, dtype=float))
    if collapsed.ndim != 2 or collapsed.shape[0] < min_layer_points or collapsed.shape[1] < 4:
        return []

    outer_min_points = int(getattr(graph_config, "outer_filter_min_points", 8))
    outer_quantile = float(getattr(graph_config, "outer_filter_quantile", 0.25))
    outer_gap_min_fraction = float(getattr(graph_config, "outer_gap_min_fraction", 0.30))
    outer_gap_min_removed_points = int(
        getattr(graph_config, "outer_gap_min_removed_points", 2)
    )

    points_xy = collapsed[:, :2]
    center = radial_center(points_xy, outer_min_points=outer_min_points)
    delta = points_xy - center
    radius = np.linalg.norm(delta, axis=1)
    radial_rank = np.zeros(collapsed.shape[0], dtype=float)
    if collapsed.shape[0] > 1:
        radius_order = np.argsort(radius, kind="mergesort")
        radial_rank[radius_order] = np.linspace(
            0.0,
            1.0,
            num=collapsed.shape[0],
            dtype=float,
        )
    keep_mask = np.ones(collapsed.shape[0], dtype=bool)
    if outer_filter_strategy != "none" and collapsed.shape[0] >= outer_min_points:
        if outer_filter_strategy == "radial_gap":
            candidate_mask = radial_gap_outer_mask(
                radius,
                min_layer_points=min_layer_points,
                min_removed_points=outer_gap_min_removed_points,
                min_gap_fraction=outer_gap_min_fraction,
            )
        else:
            candidate_mask = quantile_outer_mask(radius, outer_quantile)

        if candidate_mask is not None and int(np.sum(candidate_mask)) >= min_layer_points:
            keep_mask = candidate_mask

    kept = collapsed[keep_mask]
    kept_rank = radial_rank[keep_mask]
    kept_delta = kept[:, :2] - center
    angles = np.arctan2(kept_delta[:, 1], kept_delta[:, 0])
    order = np.argsort(angles, kind="mergesort")
    return [
        (int(strand_id), float(rank), float(angle))
        for strand_id, rank, angle in zip(
            kept[order, 3],
            kept_rank[order],
            angles[order],
            strict=False,
        )
    ]
