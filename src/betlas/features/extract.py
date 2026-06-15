from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import DIAGNOSTIC_PREFIX, FOLD_LABELS
from ..io.mmcif import build_structure_geometry
from ..models import DomainCandidate, GeometrySignature, StructureGeometry
from .geometry import (
    EPS,
    angular_coverage,
    beta_element_geometries,
    build_axis_hypotheses,
    circular_gaps,
    entropy,
    helix_element_geometries,
    pca,
    project_to_axis,
    stable_axis,
)
from .rules import grammar_rule_scores


def _nan_to_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _summary(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator + EPS)


def _bell(value: float, center: float, width: float) -> float:
    if width <= 0.0:
        return 0.0
    return float(math.exp(-((value - center) / width) ** 2))


def _min_pair_distance(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return 0.0
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(np.min(distances))


def _interval_overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> int:
    return max(0, min(a_stop, b_stop) - max(a_start, b_start) + 1)


def _beta_runs(geometry: StructureGeometry, *, max_gap: int = 2) -> list[dict[str, Any]]:
    """Merge overlapping/split SHEET ranges into sequence-level beta runs."""
    intervals: list[dict[str, Any]] = []
    for index, segment in enumerate(geometry.beta_segments):
        start = min(segment.start_auth_seq_id, segment.end_auth_seq_id)
        stop = max(segment.start_auth_seq_id, segment.end_auth_seq_id)
        intervals.append(
            {
                "start": start,
                "stop": stop,
                "segment_indices": [index],
                "sheet_ids": [segment.sheet_id or "unknown"],
            }
        )
    intervals.sort(key=lambda item: (int(item["start"]), int(item["stop"])))

    runs: list[dict[str, Any]] = []
    for interval in intervals:
        if not runs or int(interval["start"]) > int(runs[-1]["stop"]) + max_gap + 1:
            runs.append(dict(interval))
            continue
        runs[-1]["stop"] = max(int(runs[-1]["stop"]), int(interval["stop"]))
        runs[-1]["segment_indices"].extend(interval["segment_indices"])
        runs[-1]["sheet_ids"].extend(interval["sheet_ids"])

    for run_index, run in enumerate(runs):
        counts = Counter(str(sheet_id) for sheet_id in run["sheet_ids"])
        run["run_id"] = run_index
        run["dominant_sheet_id"] = counts.most_common(1)[0][0] if counts else "unknown"
        run["sheet_multiplicity"] = len(counts)
        run["length"] = int(run["stop"]) - int(run["start"]) + 1
    return runs


def _sequence_features(geometry: StructureGeometry) -> dict[str, float]:
    elements = [
        (segment.start_auth_seq_id, "B", segment.element_id)
        for segment in geometry.beta_segments
    ] + [
        (helix.start_auth_seq_id, "H", helix.element_id)
        for helix in geometry.helices
    ]
    elements.sort()
    if len(elements) < 2:
        return {
            "cz_beta_alpha_alternation_fraction": 0.0,
            "cz_beta_alpha_repeat_pairs": 0.0,
            "cz_sse_count": float(len(elements)),
        }
    transitions = 0
    repeat_pairs = 0
    for left, right in zip(elements, elements[1:], strict=False):
        if left[1] != right[1] and {left[1], right[1]} == {"B", "H"}:
            transitions += 1
        if left[1] == "B" and right[1] == "H":
            repeat_pairs += 1
    return {
        "cz_beta_alpha_alternation_fraction": transitions / max(1, len(elements) - 1),
        "cz_beta_alpha_repeat_pairs": float(repeat_pairs),
        "cz_sse_count": float(len(elements)),
    }


def _sheet_features(geometry: StructureGeometry, axis_origin: np.ndarray, axis_direction: np.ndarray) -> dict[str, float]:
    beta_geoms = beta_element_geometries(geometry)
    by_sheet: dict[str, list[Any]] = defaultdict(list)
    for item in beta_geoms:
        by_sheet[item.element.sheet_id or "unknown"].append(item)
    sheet_sizes = [len(items) for items in by_sheet.values()]
    features: dict[str, float] = {
        "cz_sheet_count": float(len(by_sheet)),
        "cz_sheet_face_count": float(sum(size >= 2 for size in sheet_sizes)),
        "cz_largest_sheet_fraction": max(sheet_sizes) / max(1, len(beta_geoms)) if sheet_sizes else 0.0,
        "cz_sheet_size_entropy": entropy(np.array([item.element.sheet_id for item in beta_geoms])),
    }
    features.update(_summary(sheet_sizes, "cz_sheet_strand_count"))

    sheet_spans: list[float] = []
    planarity: list[float] = []
    normal_changes: list[float] = []
    for items in by_sheet.values():
        if not items:
            continue
        midpoints = np.array([item.midpoint for item in items], dtype=float)
        xy, _z, _xyz = project_to_axis(midpoints, axis_origin, axis_direction)
        angles = np.arctan2(xy[:, 1], xy[:, 0])
        coverage, max_gap = angular_coverage(angles)
        sheet_spans.append(coverage)
        coords = np.concatenate([item.coords for item in items], axis=0)
        if len(coords) >= 3:
            _origin, _vectors, values = pca(coords)
            total = float(np.sum(values))
            planarity.append(float(values[-1] / total) if total > EPS else 0.0)
        if len(items) >= 2:
            axes = np.array([item.axis for item in items], dtype=float)
            dots = np.clip(np.abs(axes @ axes.T), 0.0, 1.0)
            upper = dots[np.triu_indices_from(dots, k=1)]
            normal_changes.append(float(np.mean(np.arccos(upper) / math.pi)) if len(upper) else 0.0)

    features.update(_summary(sheet_spans, "cz_sheet_angular_span"))
    features.update(_summary(planarity, "cz_sheet_planarity_residual"))
    features.update(_summary(normal_changes, "cz_sheet_axis_dispersion"))
    features["cz_sheet_angular_span_max"] = features.get("cz_sheet_angular_span_max", 0.0)
    return features


def _beta_run_features(
    geometry: StructureGeometry,
    segment_midpoints: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
) -> dict[str, float]:
    runs = _beta_runs(geometry)
    run_count = len(runs)
    segment_count = len(geometry.beta_segments)
    if not runs:
        return {
            "cz_beta_run_count": 0.0,
            "cz_beta_segment_to_run_ratio": 0.0,
            "cz_beta_run_eight_score": 0.0,
            "cz_beta_run_sheet_multiplicity_mean": 0.0,
            "cz_beta_run_sheet_multiplicity_max": 0.0,
            "cz_beta_run_multi_sheet_fraction": 0.0,
            "cz_beta_run_angular_coverage": 0.0,
            "cz_beta_run_largest_gap_fraction": 1.0,
        }

    run_midpoints: list[np.ndarray] = []
    for run in runs:
        indices = [int(index) for index in run["segment_indices"] if int(index) < len(segment_midpoints)]
        if indices:
            run_midpoints.append(np.mean(segment_midpoints[indices], axis=0))
    if run_midpoints:
        xy, _z, _xyz = project_to_axis(np.array(run_midpoints, dtype=float), origin, axis)
        coverage, largest_gap = angular_coverage(np.arctan2(xy[:, 1], xy[:, 0]))
    else:
        coverage, largest_gap = 0.0, 2.0 * math.pi

    multiplicities = [float(run["sheet_multiplicity"]) for run in runs]
    lengths = [float(run["length"]) for run in runs]
    features = {
        "cz_beta_run_count": float(run_count),
        "cz_beta_segment_to_run_ratio": _safe_ratio(segment_count, run_count),
        "cz_beta_run_eight_score": _bell(float(run_count), 8.0, 2.0),
        "cz_beta_run_sheet_multiplicity_mean": float(np.mean(multiplicities)),
        "cz_beta_run_sheet_multiplicity_max": float(np.max(multiplicities)),
        "cz_beta_run_multi_sheet_fraction": float(np.mean(np.asarray(multiplicities) > 1.0)),
        "cz_beta_run_angular_coverage": float(coverage),
        "cz_beta_run_largest_gap_fraction": float(largest_gap / (2.0 * math.pi)),
    }
    features.update(_summary(lengths, "cz_beta_run_length"))
    return features


def _sheet_plane(items: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.concatenate([item.coords for item in items], axis=0)
    origin, vectors, _values = pca(coords)
    normal = stable_axis(vectors[:, -1]) if len(coords) >= 3 else np.array([0.0, 0.0, 1.0])
    major = stable_axis(vectors[:, 0]) if len(coords) >= 3 else np.array([1.0, 0.0, 0.0])
    return origin, normal, major


def _top_sheet_items(geometry: StructureGeometry) -> tuple[list[tuple[str, list[Any]]], dict[str, list[Any]]]:
    by_sheet: dict[str, list[Any]] = defaultdict(list)
    for item in beta_element_geometries(geometry):
        by_sheet[item.element.sheet_id or "unknown"].append(item)
    ranked = sorted(by_sheet.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
    return ranked, by_sheet


def _sheet_pair_packing_features(
    geometry: StructureGeometry,
    origin: np.ndarray,
    axis: np.ndarray,
) -> dict[str, float]:
    ranked, _by_sheet = _top_sheet_items(geometry)
    total_segments = len(geometry.beta_segments)
    empty = {
        "cz_sheet_pair_top2_fraction": 0.0,
        "cz_sheet_pair_size_balance": 0.0,
        "cz_sheet_pair_centroid_distance": 0.0,
        "cz_sheet_pair_centroid_distance_norm": 0.0,
        "cz_sheet_pair_angular_separation_fraction": 0.0,
        "cz_sheet_pair_normal_abs_dot": 0.0,
        "cz_sheet_pair_face_alignment": 0.0,
        "cz_sheet_pair_cross_min_distance": 0.0,
        "cz_sheet_pair_cross_distance_mean": 0.0,
        "cz_sheet_pair_cross_contact_density8": 0.0,
        "cz_sheet_pair_cross_contact_density10": 0.0,
        "cz_sheet_pair_reciprocal_nearest_fraction8": 0.0,
        "cz_sheet_pair_bilayer_score": 0.0,
        "cz_sheet_pair_lobe_score": 0.0,
    }
    if len(ranked) < 2 or total_segments == 0:
        return empty

    (sheet_a, items_a), (sheet_b, items_b) = ranked[:2]
    if not items_a or not items_b:
        return empty
    size_a = len(items_a)
    size_b = len(items_b)
    centroid_a, normal_a, _major_a = _sheet_plane(items_a)
    centroid_b, normal_b, _major_b = _sheet_plane(items_b)
    center_delta = centroid_b - centroid_a
    center_distance = float(np.linalg.norm(center_delta))
    separation_axis = _unit(center_delta)
    normal_abs_dot = abs(float(np.dot(normal_a, normal_b)))
    face_alignment = max(abs(float(np.dot(separation_axis, normal_a))), abs(float(np.dot(separation_axis, normal_b))))

    centroids = np.array([centroid_a, centroid_b], dtype=float)
    xy_centroids, _z_centroids, _xyz = project_to_axis(centroids, origin, axis)
    centroid_angles = np.arctan2(xy_centroids[:, 1], xy_centroids[:, 0])
    angular_diff = abs(float(np.angle(np.exp(1j * (centroid_angles[1] - centroid_angles[0])))))
    angular_sep = angular_diff / math.pi

    all_midpoints = np.array(
        [item.midpoint for _sheet_id, items in ranked for item in items],
        dtype=float,
    )
    xy_all, _z_all, _ = project_to_axis(all_midpoints, origin, axis)
    radial_scale = float(np.median(np.linalg.norm(xy_all, axis=1))) if len(xy_all) else 0.0

    pair_distances: list[float] = []
    nearest_a: list[float] = []
    nearest_b: list[float] = []
    for item_a in items_a:
        distances = [_min_pair_distance(item_a.coords, item_b.coords) for item_b in items_b]
        pair_distances.extend(distances)
        if distances:
            nearest_a.append(float(np.min(distances)))
    for item_b in items_b:
        distances = [_min_pair_distance(item_b.coords, item_a.coords) for item_a in items_a]
        if distances:
            nearest_b.append(float(np.min(distances)))
    pair_arr = np.asarray(pair_distances, dtype=float)
    nearest = np.asarray([*nearest_a, *nearest_b], dtype=float)
    top2_fraction = (size_a + size_b) / max(1, total_segments)
    size_balance = min(size_a, size_b) / max(size_a, size_b)
    bilayer_score = (
        top2_fraction
        * size_balance
        * normal_abs_dot
        * face_alignment
        / (1.0 + _safe_ratio(center_distance, radial_scale + 1.0))
    )
    lobe_score = top2_fraction * size_balance * normal_abs_dot * angular_sep
    return {
        "cz_sheet_pair_top2_fraction": float(top2_fraction),
        "cz_sheet_pair_size_balance": float(size_balance),
        "cz_sheet_pair_centroid_distance": center_distance,
        "cz_sheet_pair_centroid_distance_norm": _safe_ratio(center_distance, radial_scale + 1.0),
        "cz_sheet_pair_angular_separation_fraction": float(angular_sep),
        "cz_sheet_pair_normal_abs_dot": float(normal_abs_dot),
        "cz_sheet_pair_face_alignment": float(face_alignment),
        "cz_sheet_pair_cross_min_distance": float(np.min(pair_arr)) if len(pair_arr) else 0.0,
        "cz_sheet_pair_cross_distance_mean": float(np.mean(pair_arr)) if len(pair_arr) else 0.0,
        "cz_sheet_pair_cross_contact_density8": float(np.mean(pair_arr <= 8.0)) if len(pair_arr) else 0.0,
        "cz_sheet_pair_cross_contact_density10": float(np.mean(pair_arr <= 10.0)) if len(pair_arr) else 0.0,
        "cz_sheet_pair_reciprocal_nearest_fraction8": float(np.mean(nearest <= 8.0)) if len(nearest) else 0.0,
        "cz_sheet_pair_bilayer_score": float(bilayer_score),
        "cz_sheet_pair_lobe_score": float(lobe_score),
        "cz_sheet_pair_top_sheet_ids_equal": 1.0 if sheet_a == sheet_b else 0.0,
    }


def _sheet_sequence_topology_features(geometry: StructureGeometry) -> dict[str, float]:
    runs = _beta_runs(geometry)
    ranked, _by_sheet = _top_sheet_items(geometry)
    if len(runs) < 2 or len(ranked) < 2:
        return {
            "cz_sheet_seq_top2_run_fraction": 0.0,
            "cz_sheet_seq_top2_transition_fraction": 0.0,
            "cz_sheet_seq_top2_block_count": 0.0,
            "cz_sheet_seq_top2_longest_block_fraction": 0.0,
            "cz_sheet_seq_top2_interleave_score": 0.0,
            "cz_sheet_seq_top2_order_displacement": 0.0,
            "cz_sheet_seq_greek_key_proxy": 0.0,
        }

    top2 = {ranked[0][0], ranked[1][0]}
    labels = [str(run["dominant_sheet_id"]) for run in runs]
    top2_mask = [label in top2 for label in labels]
    top2_labels = [label for label, keep in zip(labels, top2_mask, strict=False) if keep]
    if len(top2_labels) < 2:
        return {
            "cz_sheet_seq_top2_run_fraction": len(top2_labels) / max(1, len(runs)),
            "cz_sheet_seq_top2_transition_fraction": 0.0,
            "cz_sheet_seq_top2_block_count": float(len(top2_labels)),
            "cz_sheet_seq_top2_longest_block_fraction": 1.0 if top2_labels else 0.0,
            "cz_sheet_seq_top2_interleave_score": 0.0,
            "cz_sheet_seq_top2_order_displacement": 0.0,
            "cz_sheet_seq_greek_key_proxy": 0.0,
        }
    switches = [
        1 if left != right else 0
        for left, right in zip(top2_labels, top2_labels[1:], strict=False)
    ]
    block_lengths: list[int] = []
    current = 1
    for left, right in zip(top2_labels, top2_labels[1:], strict=False):
        if left == right:
            current += 1
        else:
            block_lengths.append(current)
            current = 1
    block_lengths.append(current)

    run_midpoints: list[np.ndarray] = []
    run_sequence_positions: list[int] = []
    for run_position, run in enumerate(runs):
        if str(run["dominant_sheet_id"]) not in top2:
            continue
        geoms = beta_element_geometries(geometry)
        indices = [int(index) for index in run["segment_indices"] if int(index) < len(geoms)]
        if not indices:
            continue
        run_midpoints.append(np.mean([geoms[index].midpoint for index in indices], axis=0))
        run_sequence_positions.append(run_position)
    if len(run_midpoints) >= 3:
        midpoints = np.array(run_midpoints, dtype=float)
        pc_origin, pc_vectors, _values = pca(midpoints)
        spatial_positions = (midpoints - pc_origin) @ pc_vectors[:, 0]
        spatial_order = np.argsort(spatial_positions)
        sequence_rank = np.empty(len(run_sequence_positions), dtype=int)
        for rank, idx in enumerate(np.argsort(run_sequence_positions)):
            sequence_rank[idx] = rank
        spatial_sequence_ranks = sequence_rank[spatial_order]
        displacement = float(
            np.mean(np.abs(np.diff(spatial_sequence_ranks))) / max(1.0, len(spatial_sequence_ranks) - 1)
        )
    else:
        displacement = 0.0

    transition_fraction = float(np.mean(switches))
    longest_block_fraction = max(block_lengths) / max(1, len(top2_labels))
    interleave_score = transition_fraction * (1.0 - longest_block_fraction)
    run_fraction = len(top2_labels) / max(1, len(runs))
    greek_key_proxy = (
        run_fraction
        * _bell(float(len(runs)), 8.0, 3.0)
        * (0.5 * transition_fraction + 0.5 * displacement)
    )
    return {
        "cz_sheet_seq_top2_run_fraction": float(run_fraction),
        "cz_sheet_seq_top2_transition_fraction": transition_fraction,
        "cz_sheet_seq_top2_block_count": float(len(block_lengths)),
        "cz_sheet_seq_top2_longest_block_fraction": float(longest_block_fraction),
        "cz_sheet_seq_top2_interleave_score": float(interleave_score),
        "cz_sheet_seq_top2_order_displacement": float(displacement),
        "cz_sheet_seq_greek_key_proxy": float(greek_key_proxy),
    }


def _sheet_order_key(segment: Any) -> tuple[int, int]:
    raw = str(segment.sheet_range_id or "")
    try:
        range_id = int(raw)
    except ValueError:
        range_id = 10_000
    return range_id, int(segment.start_auth_seq_id)


def _inversion_fraction(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    inversions = 0
    total = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += 1
            if values[i] > values[j]:
                inversions += 1
    return inversions / max(1, total)


def _sheet_order_topology_features(geometry: StructureGeometry) -> dict[str, float]:
    segments = list(geometry.beta_segments)
    if len(segments) < 3:
        return {
            "cz_sheet_order_adjacent_seq_step_mean": 0.0,
            "cz_sheet_order_adjacent_seq_step_max": 0.0,
            "cz_sheet_order_nonlocal_fraction": 0.0,
            "cz_sheet_order_inversion_fraction_mean": 0.0,
            "cz_top2_sheet_order_adjacent_seq_step_mean": 0.0,
            "cz_top2_sheet_order_nonlocal_fraction": 0.0,
            "cz_top2_sheet_order_inversion_fraction_mean": 0.0,
            "cz_jelly_roll_order_nonlocal_score": 0.0,
        }

    seq_order = sorted(range(len(segments)), key=lambda idx: segments[idx].start_auth_seq_id)
    seq_rank = {idx: rank for rank, idx in enumerate(seq_order)}
    by_sheet: dict[str, list[int]] = defaultdict(list)
    for idx, segment in enumerate(segments):
        by_sheet[segment.sheet_id or "unknown"].append(idx)
    top2_sheets = {
        sheet_id
        for sheet_id, _indices in sorted(
            by_sheet.items(),
            key=lambda item: (len(item[1]), item[0]),
            reverse=True,
        )[:2]
    }

    all_steps: list[float] = []
    all_nonlocal: list[float] = []
    all_inversions: list[float] = []
    top2_steps: list[float] = []
    top2_nonlocal: list[float] = []
    top2_inversions: list[float] = []
    for sheet_id, indices in by_sheet.items():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda idx: _sheet_order_key(segments[idx]))
        ranks = [seq_rank[idx] for idx in ordered]
        steps = [abs(right - left) for left, right in zip(ranks, ranks[1:], strict=False)]
        nonlocal_flags = [1.0 if step > 2 else 0.0 for step in steps]
        all_steps.extend(float(step) for step in steps)
        all_nonlocal.extend(nonlocal_flags)
        inv = _inversion_fraction(ranks)
        all_inversions.append(inv)
        if sheet_id in top2_sheets:
            top2_steps.extend(float(step) for step in steps)
            top2_nonlocal.extend(nonlocal_flags)
            top2_inversions.append(inv)

    runs = _beta_runs(geometry)
    top2_run_fraction = (
        sum(1 for run in runs if str(run["dominant_sheet_id"]) in top2_sheets) / max(1, len(runs))
    )
    run_eight = _bell(float(len(runs)), 8.0, 2.4)
    top2_nonlocal_mean = float(np.mean(top2_nonlocal)) if top2_nonlocal else 0.0
    top2_inv_mean = float(np.mean(top2_inversions)) if top2_inversions else 0.0
    return {
        "cz_sheet_order_adjacent_seq_step_mean": float(np.mean(all_steps)) if all_steps else 0.0,
        "cz_sheet_order_adjacent_seq_step_max": float(np.max(all_steps)) if all_steps else 0.0,
        "cz_sheet_order_nonlocal_fraction": float(np.mean(all_nonlocal)) if all_nonlocal else 0.0,
        "cz_sheet_order_inversion_fraction_mean": float(np.mean(all_inversions)) if all_inversions else 0.0,
        "cz_top2_sheet_order_adjacent_seq_step_mean": float(np.mean(top2_steps)) if top2_steps else 0.0,
        "cz_top2_sheet_order_nonlocal_fraction": top2_nonlocal_mean,
        "cz_top2_sheet_order_inversion_fraction_mean": top2_inv_mean,
        "cz_jelly_roll_order_nonlocal_score": float(
            top2_run_fraction * (0.5 * top2_nonlocal_mean + 0.5 * top2_inv_mean) * run_eight
        ),
    }


def _contact_graph_features(geometry: StructureGeometry) -> dict[str, float]:
    beta_geoms = beta_element_geometries(geometry)
    n = len(beta_geoms)
    if n < 2:
        return {
            "cz_contact8_edge_density": 0.0,
            "cz_contact10_edge_density": 0.0,
            "cz_contact8_cross_sheet_fraction": 0.0,
            "cz_contact8_component_count": float(n),
            "cz_contact8_largest_component_fraction": float(n),
            "cz_contact8_cycle_rank_norm": 0.0,
            "cz_contact8_degree2_fraction": 0.0,
        }
    edges8: list[tuple[int, int]] = []
    edges10: list[tuple[int, int]] = []
    cross8 = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = _min_pair_distance(beta_geoms[i].coords, beta_geoms[j].coords)
            if distance <= 10.0:
                edges10.append((i, j))
            if distance <= 8.0:
                edges8.append((i, j))
                if beta_geoms[i].element.sheet_id != beta_geoms[j].element.sheet_id:
                    cross8 += 1

    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i, j in edges8:
        adjacency[i].add(j)
        adjacency[j].add(i)
    seen: set[int] = set()
    component_sizes: list[int] = []
    for node in range(n):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        component_sizes.append(size)
    component_count = len(component_sizes)
    cycle_rank = max(0, len(edges8) - n + component_count)
    degrees = np.array([len(neighbors) for neighbors in adjacency], dtype=float)
    possible_edges = n * (n - 1) / 2
    return {
        "cz_contact8_edge_density": _safe_ratio(len(edges8), possible_edges),
        "cz_contact10_edge_density": _safe_ratio(len(edges10), possible_edges),
        "cz_contact8_cross_sheet_fraction": _safe_ratio(cross8, len(edges8)),
        "cz_contact8_component_count": float(component_count),
        "cz_contact8_largest_component_fraction": max(component_sizes) / max(1, n),
        "cz_contact8_cycle_rank_norm": _safe_ratio(cycle_rank, n),
        "cz_contact8_degree2_fraction": float(np.mean(degrees == 2.0)),
    }


def _contact_sequence_topology_features(geometry: StructureGeometry) -> dict[str, float]:
    beta_geoms = beta_element_geometries(geometry)
    n = len(beta_geoms)
    if n < 2:
        return {
            "cz_contact8_seq_gap_mean": 0.0,
            "cz_contact8_seq_gap_max": 0.0,
            "cz_contact8_nonlocal_fraction": 0.0,
            "cz_contact8_cross_sheet_nonlocal_fraction": 0.0,
            "cz_contact10_seq_gap_mean": 0.0,
            "cz_contact10_nonlocal_fraction": 0.0,
        }
    seq_order = sorted(range(n), key=lambda idx: beta_geoms[idx].element.start_auth_seq_id)
    seq_rank = {idx: rank for rank, idx in enumerate(seq_order)}
    gaps8: list[float] = []
    gaps10: list[float] = []
    cross_nonlocal8: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            distance = _min_pair_distance(beta_geoms[i].coords, beta_geoms[j].coords)
            gap = float(abs(seq_rank[i] - seq_rank[j]))
            if distance <= 10.0:
                gaps10.append(gap)
            if distance <= 8.0:
                gaps8.append(gap)
                if beta_geoms[i].element.sheet_id != beta_geoms[j].element.sheet_id:
                    cross_nonlocal8.append(1.0 if gap > 2.0 else 0.0)
    return {
        "cz_contact8_seq_gap_mean": float(np.mean(gaps8)) if gaps8 else 0.0,
        "cz_contact8_seq_gap_max": float(np.max(gaps8)) if gaps8 else 0.0,
        "cz_contact8_nonlocal_fraction": float(np.mean(np.asarray(gaps8) > 2.0)) if gaps8 else 0.0,
        "cz_contact8_cross_sheet_nonlocal_fraction": float(np.mean(cross_nonlocal8))
        if cross_nonlocal8
        else 0.0,
        "cz_contact10_seq_gap_mean": float(np.mean(gaps10)) if gaps10 else 0.0,
        "cz_contact10_nonlocal_fraction": float(np.mean(np.asarray(gaps10) > 2.0)) if gaps10 else 0.0,
    }


def _angular_lobe_features(
    segment_midpoints: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    base_features: dict[str, float | int | str],
) -> dict[str, float]:
    if len(segment_midpoints) < 3:
        return {
            "cz_angular_fft_k1": 0.0,
            "cz_angular_fft_k2": 0.0,
            "cz_angular_fft_k2_dominance": 0.0,
            "cz_angular_sector_occupancy12": 0.0,
            "cz_angular_sector_entropy12": 0.0,
            "cz_barrel_wall_continuity_score": 0.0,
            "cz_sandwich_lobe_guard_score": 0.0,
        }
    xy, _z, _xyz = project_to_axis(segment_midpoints, origin, axis)
    angles = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * math.pi)
    hist, _edges = np.histogram(angles, bins=36, range=(0.0, 2.0 * math.pi))
    hist = hist.astype(float)
    hist = (hist - np.mean(hist)) / (np.std(hist) + EPS)
    fft = np.abs(np.fft.rfft(hist))
    fft_norm = fft / (np.sum(fft[1:]) + EPS)
    sectors, _sector_edges = np.histogram(angles, bins=12, range=(0.0, 2.0 * math.pi))
    occupied = float(np.mean(sectors > 0))
    sector_entropy = entropy(np.repeat(np.arange(12), sectors)) if int(np.sum(sectors)) else 0.0
    k1 = float(fft_norm[1]) if len(fft_norm) > 1 else 0.0
    k2 = float(fft_norm[2]) if len(fft_norm) > 2 else 0.0
    kmax = max(k1, k2, float(base_features.get("cz_angular_fft_k3_8_max", 0.0)))
    closure = float(base_features.get("cz_axis_best_slice_coverage_median", 0.0))
    continuity = float(base_features.get("cz_z_continuity_fraction", 0.0))
    largest_sheet = float(base_features.get("cz_largest_sheet_fraction", 0.0))
    bilayer = float(base_features.get("cz_sheet_pair_bilayer_score", 0.0))
    lobe = float(base_features.get("cz_sheet_pair_lobe_score", 0.0))
    barrel_wall = closure * continuity * largest_sheet * (1.0 - min(1.0, k2 / (kmax + EPS)))
    sandwich_guard = k2 * bilayer + 0.5 * lobe
    return {
        "cz_angular_fft_k1": k1,
        "cz_angular_fft_k2": k2,
        "cz_angular_fft_k2_dominance": k2 / (kmax + EPS),
        "cz_angular_sector_occupancy12": occupied,
        "cz_angular_sector_entropy12": float(sector_entropy),
        "cz_barrel_wall_continuity_score": float(barrel_wall),
        "cz_sandwich_lobe_guard_score": float(sandwich_guard),
    }


def _axis_slice_features(
    beta_points: np.ndarray,
    segment_midpoints: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    min_points_per_slice: int = 4,
) -> dict[str, float]:
    xy_mid, z_mid, _xyz_mid = project_to_axis(segment_midpoints, origin, axis)
    angles_mid = np.arctan2(xy_mid[:, 1], xy_mid[:, 0]) if len(xy_mid) else np.array([])
    coverage_all, largest_gap = angular_coverage(angles_mid)
    radii_mid = np.linalg.norm(xy_mid, axis=1) if len(xy_mid) else np.array([])

    xy_points, z_points, _xyz_points = project_to_axis(beta_points, origin, axis)
    if len(z_points) == 0:
        return {
            "angular_coverage": 0.0,
            "largest_gap_fraction": 1.0,
            "radius_cv": 0.0,
            "z_range": 0.0,
            "z_continuity_fraction": 0.0,
            "slice_count": 0.0,
            "slice_coverage_mean": 0.0,
            "slice_coverage_median": 0.0,
            "slice_coverage_std": 0.0,
            "slice_high_coverage_fraction": 0.0,
            "slice_largest_gap_fraction_mean": 1.0,
        }

    z_min, z_max = float(np.min(z_points)), float(np.max(z_points))
    z_range = max(0.0, z_max - z_min)
    bins = max(4, min(24, int(math.ceil(z_range / 4.0)) if z_range > EPS else 4))
    edges = np.linspace(z_min, z_max + EPS, bins + 1)
    coverages: list[float] = []
    gap_fracs: list[float] = []
    occupied = 0
    point_angles = np.arctan2(xy_points[:, 1], xy_points[:, 0])
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        mask = (z_points >= left) & (z_points < right)
        if int(np.sum(mask)) < min_points_per_slice:
            continue
        occupied += 1
        cov, gap = angular_coverage(point_angles[mask])
        coverages.append(cov)
        gap_fracs.append(gap / (2.0 * math.pi))

    radius_mean = float(np.mean(radii_mid)) if len(radii_mid) else 0.0
    radius_std = float(np.std(radii_mid)) if len(radii_mid) else 0.0
    return {
        "angular_coverage": float(coverage_all),
        "largest_gap_fraction": float(largest_gap / (2.0 * math.pi)),
        "radius_cv": radius_std / (radius_mean + EPS),
        "z_range": z_range,
        "z_continuity_fraction": occupied / max(1, bins),
        "slice_count": float(occupied),
        "slice_coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "slice_coverage_median": float(np.median(coverages)) if coverages else 0.0,
        "slice_coverage_std": float(np.std(coverages)) if coverages else 0.0,
        "slice_high_coverage_fraction": float(np.mean(np.asarray(coverages) >= 0.75)) if coverages else 0.0,
        "slice_largest_gap_fraction_mean": float(np.mean(gap_fracs)) if gap_fracs else 1.0,
    }


def _axis_periodicity_features(
    segment_midpoints: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
) -> dict[str, float]:
    if len(segment_midpoints) < 4:
        return {
            "cz_axis_periodicity_score": 0.0,
            "cz_angular_fft_k3_8_max": 0.0,
            "cz_angular_fft_k3_8_best_k": 0.0,
            **{f"cz_angular_fft_k{k}": 0.0 for k in range(3, 9)},
        }
    xy, z, _xyz = project_to_axis(segment_midpoints, origin, axis)
    angles = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * math.pi)
    hist, _edges = np.histogram(angles, bins=36, range=(0.0, 2.0 * math.pi))
    hist = hist.astype(float)
    hist = (hist - np.mean(hist)) / (np.std(hist) + EPS)
    fft = np.abs(np.fft.rfft(hist))
    fft = fft / (np.sum(fft[1:]) + EPS)
    harmonics: dict[str, float] = {}
    best_k = 0
    best_val = 0.0
    for k in range(3, 9):
        value = float(fft[k]) if k < len(fft) else 0.0
        harmonics[f"cz_angular_fft_k{k}"] = value
        if value > best_val:
            best_val = value
            best_k = k

    order = np.argsort(z)
    angles_sorted = np.unwrap(angles[order])
    z_sorted = z[order]
    if len(np.unique(np.round(z_sorted, 3))) >= 4:
        coeff = np.polyfit(z_sorted, angles_sorted, deg=1)
        pred = np.polyval(coeff, z_sorted)
        ss_res = float(np.sum((angles_sorted - pred) ** 2))
        ss_tot = float(np.sum((angles_sorted - np.mean(angles_sorted)) ** 2))
        helical_r2 = max(0.0, 1.0 - ss_res / (ss_tot + EPS))
    else:
        helical_r2 = 0.0
    return {
        "cz_axis_periodicity_score": float(0.5 * helical_r2 + 0.5 * best_val),
        "cz_angular_fft_k3_8_max": best_val,
        "cz_angular_fft_k3_8_best_k": float(best_k),
        **harmonics,
    }


def _strand_order_features(
    segment_midpoints: np.ndarray,
    geometry: StructureGeometry,
    origin: np.ndarray,
    axis: np.ndarray,
) -> dict[str, float]:
    n = len(segment_midpoints)
    if n < 3:
        return {
            "cz_strand_order_displacement_mean": 0.0,
            "cz_strand_order_adjacent_seq_step_median": 0.0,
            "cz_strand_ntc_parallel_fraction": 0.0,
            "cz_strand_ntc_antiparallel_fraction": 0.0,
            "cz_strand_axis_abs_alignment_mean": 0.0,
        }
    xy, _z, _xyz = project_to_axis(segment_midpoints, origin, axis)
    angles = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * math.pi)
    angular_order = np.argsort(angles)
    seq_order = np.argsort([segment.start_auth_seq_id for segment in geometry.beta_segments])
    seq_rank = np.empty(n, dtype=int)
    for rank, idx in enumerate(seq_order):
        seq_rank[idx] = rank
    angular_ranks = seq_rank[angular_order]
    diffs = np.abs(np.diff(np.r_[angular_ranks, angular_ranks[0]]))
    cyclic_steps = np.minimum(diffs, n - diffs)
    displacement = float(np.mean(cyclic_steps) / max(1, n / 2.0))

    beta_geoms = beta_element_geometries(geometry)
    axes = np.array([item.directed_axis for item in beta_geoms], dtype=float)
    if len(axes) >= 2:
        dots = axes @ axes.T
        upper = dots[np.triu_indices_from(dots, k=1)]
        parallel_fraction = float(np.mean(upper > 0.0))
        antiparallel_fraction = float(np.mean(upper < 0.0))
        abs_alignment = float(np.mean(np.abs(upper)))
    else:
        parallel_fraction = 0.0
        antiparallel_fraction = 0.0
        abs_alignment = 0.0

    return {
        "cz_strand_order_displacement_mean": displacement,
        "cz_strand_order_adjacent_seq_step_median": float(np.median(cyclic_steps)),
        "cz_strand_ntc_parallel_fraction": parallel_fraction,
        "cz_strand_ntc_antiparallel_fraction": antiparallel_fraction,
        "cz_strand_axis_abs_alignment_mean": abs_alignment,
    }


def _alpha_shell_features(
    geometry: StructureGeometry,
    origin: np.ndarray,
    axis: np.ndarray,
) -> dict[str, float]:
    beta_geoms = beta_element_geometries(geometry)
    helix_geoms = helix_element_geometries(geometry)
    if not beta_geoms or not helix_geoms:
        return {
            "cz_alpha_shell_radial_delta": 0.0,
            "cz_helix_beta_radius_ratio": 0.0,
        }
    beta_mid = np.array([item.midpoint for item in beta_geoms], dtype=float)
    helix_mid = np.array([item.midpoint for item in helix_geoms], dtype=float)
    beta_xy, _bz, _ = project_to_axis(beta_mid, origin, axis)
    helix_xy, _hz, _ = project_to_axis(helix_mid, origin, axis)
    beta_radius = float(np.mean(np.linalg.norm(beta_xy, axis=1)))
    helix_radius = float(np.mean(np.linalg.norm(helix_xy, axis=1)))
    return {
        "cz_alpha_shell_radial_delta": helix_radius - beta_radius,
        "cz_helix_beta_radius_ratio": helix_radius / (beta_radius + EPS),
    }


def extract_signature(geometry: StructureGeometry) -> GeometrySignature:
    beta_geoms = beta_element_geometries(geometry)
    residue_count = len(geometry.residues)
    beta_residue_indices = sorted({idx for segment in geometry.beta_segments for idx in segment.residue_indices})
    helix_residue_indices = sorted({idx for helix in geometry.helices for idx in helix.residue_indices})
    beta_points = (
        np.concatenate([item.coords for item in beta_geoms], axis=0) if beta_geoms else np.empty((0, 3))
    )
    segment_midpoints = (
        np.array([item.midpoint for item in beta_geoms], dtype=float) if beta_geoms else np.empty((0, 3))
    )
    all_residue_points = np.array([residue.coord_ca for residue in geometry.residues], dtype=float)
    pca_points = beta_points if len(beta_points) else all_residue_points
    _pca_origin, _vectors, values = pca(pca_points) if len(pca_points) else (np.zeros(3), np.eye(3), np.zeros(3))
    pca_elongation = float(values[0] / (values[1] + EPS)) if len(values) >= 2 else 0.0
    pca_flatness = float(1.0 - values[-1] / (np.sum(values) + EPS)) if len(values) >= 3 else 0.0

    axes = build_axis_hypotheses(geometry)
    axis_rows: list[tuple[float, str, np.ndarray, np.ndarray, dict[str, float]]] = []
    for axis_hyp in axes:
        origin = np.array(axis_hyp.origin, dtype=float)
        direction = stable_axis(np.array(axis_hyp.direction, dtype=float))
        axis_features = _axis_slice_features(beta_points, segment_midpoints, origin, direction)
        score = (
            0.40 * axis_features["slice_coverage_median"]
            + 0.25 * axis_features["angular_coverage"]
            + 0.20 * axis_features["z_continuity_fraction"]
            + 0.15 * (1.0 - axis_features["largest_gap_fraction"])
        )
        axis_rows.append((score, axis_hyp.name, origin, direction, axis_features))
    if axis_rows:
        axis_rows.sort(key=lambda row: row[0], reverse=True)
        axis_score, axis_name, origin, direction, best_axis_features = axis_rows[0]
    else:
        axis_score, axis_name, origin, direction, best_axis_features = (
            0.0,
            "none",
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            _axis_slice_features(beta_points, segment_midpoints, np.zeros(3), np.array([0.0, 0.0, 1.0])),
        )

    strand_lengths = [float(len(segment.residue_indices)) for segment in geometry.beta_segments]
    helix_lengths = [float(len(helix.residue_indices)) for helix in geometry.helices]
    sheet_senses = [segment.sense_to_previous for segment in geometry.beta_segments if segment.sense_to_previous]
    sense_counts = Counter(sheet_senses)

    features: dict[str, float | int | str] = {
        "cz_parse_ok": 1 if geometry.residues and geometry.beta_segments else 0,
        "cz_residue_count": float(residue_count),
        "cz_beta_residue_count": float(len(beta_residue_indices)),
        "cz_helix_residue_count": float(len(helix_residue_indices)),
        "cz_beta_residue_fraction": len(beta_residue_indices) / max(1, residue_count),
        "cz_helix_residue_fraction": len(helix_residue_indices) / max(1, residue_count),
        "cz_beta_strand_count": float(len(geometry.beta_segments)),
        "cz_helix_count": float(len(geometry.helices)),
        "cz_pca_elongation": _nan_to_zero(pca_elongation),
        "cz_pca_flatness": _nan_to_zero(pca_flatness),
        "cz_axis_best_name": axis_name,
        "cz_axis_best_score": float(axis_score),
        "cz_axis_best_is_strand_axis": 1.0 if axis_name == "strand_axis" else 0.0,
        "cz_axis_best_is_point_pc1": 1.0 if axis_name == "point_pc1" else 0.0,
        "cz_axis_best_is_point_pc2": 1.0 if axis_name == "point_pc2" else 0.0,
        "cz_axis_best_is_point_pc3": 1.0 if axis_name == "point_pc3" else 0.0,
        "cz_z_continuity_fraction": best_axis_features["z_continuity_fraction"],
        "cz_parallel_sheet_order_fraction": sense_counts.get("parallel", 0) / max(1, len(sheet_senses)),
        "cz_antiparallel_sheet_order_fraction": sense_counts.get("anti-parallel", 0) / max(1, len(sheet_senses)),
    }
    for key, value in best_axis_features.items():
        features[f"cz_axis_best_{key}"] = float(value)
    features.update(_summary(strand_lengths, "cz_strand_length"))
    features.update(_summary(helix_lengths, "cz_helix_length"))
    features.update(_sequence_features(geometry))
    features.update(_sheet_features(geometry, origin, direction))
    features.update(_beta_run_features(geometry, segment_midpoints, origin, direction))
    features.update(_sheet_pair_packing_features(geometry, origin, direction))
    features.update(_sheet_sequence_topology_features(geometry))
    features.update(_sheet_order_topology_features(geometry))
    features.update(_contact_graph_features(geometry))
    features.update(_contact_sequence_topology_features(geometry))
    features.update(_axis_periodicity_features(segment_midpoints, origin, direction))
    features.update(_strand_order_features(segment_midpoints, geometry, origin, direction))
    features.update(_alpha_shell_features(geometry, origin, direction))
    features.update(_angular_lobe_features(segment_midpoints, origin, direction, features))

    if len(segment_midpoints) >= 2:
        xy, _z, _ = project_to_axis(segment_midpoints, origin, direction)
        gaps = circular_gaps(np.arctan2(xy[:, 1], xy[:, 0]))
        features["cz_angular_gap_cv"] = float(np.std(gaps) / (np.mean(gaps) + EPS))
    else:
        features["cz_angular_gap_cv"] = 0.0

    fold_scores = grammar_rule_scores(features)
    top_fold = max(fold_scores, key=fold_scores.get) if fold_scores else ""
    sorted_scores = sorted(fold_scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0
    features["cz_top_fold"] = top_fold
    features["cz_rule_margin"] = float(margin)
    features["cz_fold_scores_json"] = json.dumps(fold_scores, sort_keys=True)
    for label in FOLD_LABELS:
        features[f"cz_rule_score_{label}"] = float(fold_scores.get(label, 0.0))

    return GeometrySignature(
        domain=geometry.domain,
        features=features,
        fold_scores=fold_scores,
        warnings=geometry.warnings,
    )


def extract_feature_row(domain: DomainCandidate, mmcif_path: Path) -> dict[str, Any]:
    try:
        geometry = build_structure_geometry(domain, mmcif_path)
        signature = extract_signature(geometry)
        row = signature.to_row()
        row["cz_error"] = ""
        return row
    except Exception as exc:
        row = domain.to_dict()
        row.update(
            {
                "cz_parse_ok": 0,
                "cz_error": f"{type(exc).__name__}: {exc}",
                "cz_top_fold": "",
                "cz_fold_scores_json": "{}",
            }
        )
        return row


def assert_no_diagnostic_label_leakage(row: dict[str, Any]) -> None:
    forbidden_targets = {
        "fold_label_final",
        "fold_label_candidates",
        "evidence_level",
        "qc_status",
        "allowed_for_publication_benchmark",
    }
    for key in forbidden_targets:
        value = str(row.get(key, ""))
        if value.startswith(DIAGNOSTIC_PREFIX):
            raise ValueError(f"label field {key} was populated from diagnostic field {value!r}")
