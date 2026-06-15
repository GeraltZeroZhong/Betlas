from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from .alignment import PCAAligner
from .analysis_utils import (
    angular_gap_stats,
    collapse_points_by_strand,
    nearest_neighbor_spacing_stats,
    sequence_angle_order_stats,
)
from .slicer import ProteinSlicer


@dataclass(frozen=True)
class AxisSliceCandidate:
    name: str
    slices: dict[float, list[tuple[float, ...]]]
    score_key: tuple[int | float, ...]
    proxy_rank: int
    rotation_key: tuple[float, ...]


def axis_search_minimum_points(cfg: AppConfig) -> int:
    return max(
        int(cfg.analyzer.axis_search.min_points_per_scored_layer),
        int(cfg.analyzer.count.min_points_per_layer),
        int(cfg.analyzer.count.min_count),
    )


def axis_search_score_key(
    slices: dict[float, list[tuple[float, ...]]],
    minimum_points: int,
    cfg: AppConfig,
) -> tuple[int | float, ...]:
    """Rank candidate axis rotations using lightweight strand-count proxies."""
    rules_cfg = cfg.analyzer.rules
    collapsed_counts: list[int] = []
    full_pass_layers = 0
    gap_pass_layers = 0
    order_pass_layers = 0
    nn_pass_layers = 0
    gap_margins: list[float] = []
    order_local_fracs: list[float] = []
    order_mean_circ_dists: list[float] = []
    nn_inlier_fracs: list[float] = []
    duplicate_penalty = 0

    for points in slices.values():
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue

        collapsed = collapse_points_by_strand(pts)
        collapsed_count = int(collapsed.shape[0]) if collapsed.ndim == 2 else 0
        collapsed_counts.append(collapsed_count)

        if pts.shape[1] >= 4:
            unique_strands = int(np.unique(pts[:, 3].astype(int)).size)
        else:
            unique_strands = int(pts.shape[0])
        duplicate_penalty += int(pts.shape[0]) - unique_strands

        if collapsed_count < minimum_points:
            continue

        collapsed_points = np.asarray(collapsed, dtype=float)
        collapsed_xy = np.asarray(collapsed_points[:, :2], dtype=float)

        nn_ok = True
        nn_inlier_frac = 1.0
        if rules_cfg.nearest_neighbor.enabled:
            nn_stats = nearest_neighbor_spacing_stats(collapsed_xy)
            if nn_stats is not None:
                _, _, robust_cv, inlier_frac = nn_stats
                nn_inlier_frac = float(inlier_frac)
                nn_ok = (
                    float(robust_cv) <= float(rules_cfg.nearest_neighbor.max_robust_cv)
                    and nn_inlier_frac >= float(rules_cfg.nearest_neighbor.min_inlier_frac)
                )
        nn_inlier_fracs.append(nn_inlier_frac)
        if nn_ok:
            nn_pass_layers += 1

        gap_ok = True
        gap_margin = 0.0
        order_ok = True
        order_local_frac = 1.0
        order_mean_circ = 0.0
        if rules_cfg.angle.enabled:
            angle_stats = angular_gap_stats(collapsed_xy)
            if angle_stats is not None:
                max_gap_deg = float(angle_stats[0])
                gap_margin = float(rules_cfg.angle.max_gap_deg) - max_gap_deg
                gap_ok = max_gap_deg <= float(rules_cfg.angle.max_gap_deg)

            if rules_cfg.angle.order.enabled:
                order_stats = sequence_angle_order_stats(collapsed_points, rules_cfg.angle.order)
                if order_stats is not None:
                    order_local_frac = float(order_stats["order_local_frac"])
                    order_mean_circ = float(order_stats["order_mean_circ_dist_norm"])
                    order_ok = (
                        order_local_frac >= float(rules_cfg.angle.order.min_local_frac)
                        and order_mean_circ
                        <= float(rules_cfg.angle.order.max_mean_circ_dist_norm)
                    )

        gap_margins.append(gap_margin)
        order_local_fracs.append(order_local_frac)
        order_mean_circ_dists.append(order_mean_circ)

        if gap_ok:
            gap_pass_layers += 1
        if order_ok:
            order_pass_layers += 1
        if nn_ok and gap_ok and order_ok:
            full_pass_layers += 1

    if not collapsed_counts:
        return (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0)

    collapsed_array = np.asarray(collapsed_counts, dtype=float)
    layers_at_threshold = int(np.sum(collapsed_array >= float(minimum_points)))
    median_gap_margin = float(np.median(gap_margins)) if gap_margins else 0.0
    median_order_local = float(np.median(order_local_fracs)) if order_local_fracs else 0.0
    median_order_mean_circ = (
        float(np.median(order_mean_circ_dists)) if order_mean_circ_dists else 0.0
    )
    median_nn_inlier = float(np.median(nn_inlier_fracs)) if nn_inlier_fracs else 0.0
    geometry_terms = (
        full_pass_layers,
        gap_pass_layers,
        order_pass_layers,
        nn_pass_layers,
        median_gap_margin,
        median_order_local,
        -median_order_mean_circ,
        median_nn_inlier,
    )
    count_terms = (
        layers_at_threshold,
        float(np.median(collapsed_array)),
        float(np.mean(collapsed_array)),
        -int(duplicate_penalty),
    )
    if cfg.analyzer.axis_search.score_geometry_first:
        return (*geometry_terms, *count_terms)
    return (layers_at_threshold, *geometry_terms, *count_terms[1:])


def proper_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("Expected a 3x3 rotation matrix.")
    if float(np.linalg.det(rotation)) < 0.0:
        rotation = rotation.copy()
        rotation[:, 0] *= -1.0
    return rotation


def candidate_axis_rotations(rotation_matrix: np.ndarray) -> list[np.ndarray]:
    base_rotation = np.asarray(rotation_matrix, dtype=float)
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 2, 0),
    )
    return [
        proper_rotation_matrix(base_rotation[:, permutation])
        for permutation in permutations
    ]


def local_axis_rotation(angle_deg: float, axis: str) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    cosine = math.cos(theta)
    sine = math.sin(theta)

    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=float,
        )
    if axis == "y":
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=float,
        )
    raise ValueError(f"Unknown local rotation axis: {axis}")


def refinement_rotations(base_rotation: np.ndarray, angle_deg: float) -> list[np.ndarray]:
    if angle_deg <= 0.0:
        return [proper_rotation_matrix(base_rotation)]

    refined = [proper_rotation_matrix(base_rotation)]
    for axis in ("x", "y"):
        for sign in (-1.0, 1.0):
            refined.append(
                proper_rotation_matrix(
                    np.asarray(base_rotation, dtype=float)
                    @ local_axis_rotation(sign * angle_deg, axis)
                )
            )
    return refined


def aligned_coordinates_for_rotation(
    coordinates: np.ndarray,
    *,
    center: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    return np.dot(np.asarray(coordinates, dtype=float) - np.asarray(center, dtype=float), rotation_matrix)


def _slice_candidate_for_rotation(
    *,
    name: str,
    rotation_matrix: np.ndarray,
    coordinates: np.ndarray,
    residues_data: list[dict[str, object]],
    center: np.ndarray,
    slicer: ProteinSlicer,
    minimum_points: int,
    cfg: AppConfig,
) -> AxisSliceCandidate:
    aligned_coordinates = aligned_coordinates_for_rotation(
        coordinates,
        center=np.asarray(center, dtype=float),
        rotation_matrix=np.asarray(rotation_matrix, dtype=float),
    )
    slice_dict = slicer.slice_structure(aligned_coordinates, residues_data)
    score_key = axis_search_score_key(slice_dict, minimum_points, cfg)
    return AxisSliceCandidate(
        name=name,
        slices=slice_dict,
        score_key=score_key,
        proxy_rank=0,
        rotation_key=tuple(np.round(np.asarray(rotation_matrix, dtype=float).ravel(), 8)),
    )


def _with_proxy_ranks(candidates: list[AxisSliceCandidate]) -> list[AxisSliceCandidate]:
    ranked = sorted(candidates, key=lambda candidate: candidate.score_key, reverse=True)
    return [
        AxisSliceCandidate(
            name=candidate.name,
            slices=candidate.slices,
            score_key=candidate.score_key,
            proxy_rank=index + 1,
            rotation_key=candidate.rotation_key,
        )
        for index, candidate in enumerate(ranked)
    ]


def _dedupe_candidates(
    candidates: Iterable[AxisSliceCandidate],
) -> list[AxisSliceCandidate]:
    unique: list[AxisSliceCandidate] = []
    seen_names: set[str] = set()
    seen_rotations: set[tuple[float, ...]] = set()
    for candidate in candidates:
        if candidate.name in seen_names or candidate.rotation_key in seen_rotations:
            continue
        seen_names.add(candidate.name)
        seen_rotations.add(candidate.rotation_key)
        unique.append(candidate)
    return unique


def alignment_slice_candidates(
    aligner: PCAAligner,
    coordinates: np.ndarray,
    residues_data: list[dict[str, object]],
    slicer: ProteinSlicer,
    cfg: AppConfig,
) -> list[AxisSliceCandidate]:
    axis_cfg = cfg.analyzer.axis_search
    center = getattr(aligner, "center", None)
    rotation_matrix = getattr(aligner, "rotation_matrix", None)
    if center is None or rotation_matrix is None:
        raise RuntimeError("Aligner is not fitted. Call fit() before selecting a PCA axis.")

    minimum_points = axis_search_minimum_points(cfg)
    base_rotation = proper_rotation_matrix(np.asarray(rotation_matrix, dtype=float))
    if not axis_cfg.enabled:
        return _with_proxy_ranks(
            [
                _slice_candidate_for_rotation(
                    name="base_pca",
                    rotation_matrix=base_rotation,
                    coordinates=coordinates,
                    residues_data=residues_data,
                    center=np.asarray(center, dtype=float),
                    slicer=slicer,
                    minimum_points=minimum_points,
                    cfg=cfg,
                )
            ]
        )

    def best_slices_for_rotations(
        rotations: Iterable[tuple[str, np.ndarray]],
    ) -> tuple[np.ndarray, AxisSliceCandidate]:
        best_rotation: np.ndarray | None = None
        best_candidate: AxisSliceCandidate | None = None

        for name, candidate_rotation in rotations:
            candidate = _slice_candidate_for_rotation(
                name=name,
                rotation_matrix=candidate_rotation,
                coordinates=coordinates,
                residues_data=residues_data,
                center=np.asarray(center, dtype=float),
                slicer=slicer,
                minimum_points=minimum_points,
                cfg=cfg,
            )
            if best_candidate is None or candidate.score_key > best_candidate.score_key:
                best_rotation = np.asarray(candidate_rotation, dtype=float)
                best_candidate = candidate

        if best_rotation is None or best_candidate is None:
            raise RuntimeError("Axis search produced no candidate rotations.")
        return best_rotation, best_candidate

    base_candidate = _slice_candidate_for_rotation(
        name="base_pca",
        rotation_matrix=base_rotation,
        coordinates=coordinates,
        residues_data=residues_data,
        center=np.asarray(center, dtype=float),
        slicer=slicer,
        minimum_points=minimum_points,
        cfg=cfg,
    )
    candidates = [base_candidate]

    permutation_rotations = [
        (f"permutation_{index}", rotation)
        for index, rotation in enumerate(candidate_axis_rotations(base_rotation))
    ]
    best_rotation, best_candidate = best_slices_for_rotations(permutation_rotations)
    candidates.append(best_candidate)

    if axis_cfg.refine.enabled and float(axis_cfg.refine.angle_deg) > 0.0:
        refined_rotations = [
            (f"refine_{index}", rotation)
            for index, rotation in enumerate(
                refinement_rotations(best_rotation, float(axis_cfg.refine.angle_deg))
            )
        ]
        _best_refined_rotation, best_refined_candidate = best_slices_for_rotations(
            refined_rotations
        )
        candidates.append(best_refined_candidate)

    return _with_proxy_ranks(_dedupe_candidates(candidates))


def select_alignment_slices(
    aligner: PCAAligner,
    coordinates: np.ndarray,
    residues_data: list[dict[str, object]],
    slicer: ProteinSlicer,
    cfg: AppConfig,
) -> dict[float, list[tuple[float, ...]]]:
    candidates = alignment_slice_candidates(
        aligner,
        coordinates,
        residues_data,
        slicer,
        cfg,
    )
    if not candidates:
        raise RuntimeError("Axis search produced no candidate slices.")
    return max(candidates, key=lambda candidate: candidate.score_key).slices
