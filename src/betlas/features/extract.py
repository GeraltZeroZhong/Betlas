from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import DIAGNOSTIC_PREFIX
from ..io.mmcif import available_auth_chain_ids, build_structure_geometry, is_mmcif_path
from ..models import DomainCandidate, GeometrySignature, StructureGeometry
from ..slicing import SliceConfig, _compute_axis_slice_summary
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
from .current import (
    FEATURE_COLUMNS, FEATURE_SCHEMA, CIRCULAR_MOMENT_COLUMNS, VALID_PLANE_COLUMNS,
    apply_applicability, circular_moments,
)


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


def _element_sequence_key(geometry: StructureGeometry, element: Any) -> tuple[str, int]:
    if geometry.secondary_structure_source == "dssp":
        first = geometry.residues[element.residue_indices[0]]
        return first.chain_id, int(first.label_seq_id)
    return element.chain_id, int(element.start_auth_seq_id)


def _beta_runs(geometry: StructureGeometry, *, max_gap: int = 2) -> list[dict[str, Any]]:
    """Merge overlapping/split SHEET ranges into sequence-level beta runs."""
    intervals: list[dict[str, Any]] = []
    for index, segment in enumerate(geometry.beta_segments):
        if geometry.secondary_structure_source == "dssp":
            start = int(geometry.residues[segment.residue_indices[0]].label_seq_id)
            stop = int(geometry.residues[segment.residue_indices[-1]].label_seq_id)
        else:
            start = min(segment.start_auth_seq_id, segment.end_auth_seq_id)
            stop = max(segment.start_auth_seq_id, segment.end_auth_seq_id)
        intervals.append(
            {
                "start": start,
                "stop": stop,
                "segment_indices": [index],
                "sheet_ids": [segment.sheet_id or "unknown"],
                "chain_id": segment.chain_id,
                "fragment_id": segment.fragment_id,
            }
        )
    intervals.sort(key=lambda item: (item["chain_id"], int(item["start"]), int(item["stop"])))

    runs: list[dict[str, Any]] = []
    for interval in intervals:
        if (not runs or interval["chain_id"] != runs[-1]["chain_id"]
                or interval["fragment_id"] != runs[-1]["fragment_id"]
                or int(interval["start"]) > int(runs[-1]["stop"]) + max_gap + 1):
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
        (_element_sequence_key(geometry, segment), "B", segment.element_id, segment.fragment_id)
        for segment in geometry.beta_segments
    ] + [
        (_element_sequence_key(geometry, helix), "H", helix.element_id, helix.fragment_id)
        for helix in geometry.helices
    ]
    elements.sort()
    if len(elements) < 2:
        return {
            "betlas_beta_alpha_alternation_fraction": 0.0,
            "betlas_beta_alpha_repeat_pairs": 0.0,
            "betlas_sse_count": float(len(elements)),
        }
    transitions = 0
    repeat_pairs = 0
    valid_pairs = 0
    for left, right in zip(elements, elements[1:], strict=False):
        if left[0][0] != right[0][0] or left[3] != right[3]:
            continue
        valid_pairs += 1
        if left[1] != right[1] and {left[1], right[1]} == {"B", "H"}:
            transitions += 1
        if left[1] == "B" and right[1] == "H":
            repeat_pairs += 1
    return {
        "betlas_beta_alpha_alternation_fraction": transitions / max(1, valid_pairs),
        "betlas_beta_alpha_repeat_pairs": float(repeat_pairs),
        "betlas_sse_count": float(len(elements)),
    }


def _sheet_features(geometry: StructureGeometry, axis_origin: np.ndarray, axis_direction: np.ndarray) -> dict[str, float]:
    beta_geoms = beta_element_geometries(geometry)
    by_sheet: dict[str, list[Any]] = defaultdict(list)
    for item in beta_geoms:
        by_sheet[item.element.sheet_id or "unknown"].append(item)
    sheet_sizes = [len(items) for items in by_sheet.values()]
    features: dict[str, float] = {
        "betlas_sheet_count": float(len(by_sheet)),
        "betlas_sheet_face_count": float(sum(size >= 2 for size in sheet_sizes)),
        "betlas_largest_sheet_fraction": max(sheet_sizes) / max(1, len(beta_geoms)) if sheet_sizes else 0.0,
        "betlas_sheet_size_entropy": entropy(np.array([item.element.sheet_id for item in beta_geoms])),
    }
    features.update(_summary(sheet_sizes, "betlas_sheet_strand_count"))

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

    features.update(_summary(sheet_spans, "betlas_sheet_angular_span"))
    features.update(_summary(planarity, "betlas_sheet_planarity_residual"))
    features.update(_summary(normal_changes, "betlas_sheet_axis_dispersion"))
    features["betlas_sheet_angular_span_max"] = features.get("betlas_sheet_angular_span_max", 0.0)
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
            "betlas_beta_run_count": 0.0,
            "betlas_beta_segment_to_run_ratio": 0.0,
            "betlas_beta_run_eight_score": 0.0,
            "betlas_beta_run_sheet_multiplicity_mean": 0.0,
            "betlas_beta_run_sheet_multiplicity_max": 0.0,
            "betlas_beta_run_multi_sheet_fraction": 0.0,
            "betlas_beta_run_angular_coverage": 0.0,
            "betlas_beta_run_largest_gap_fraction": 1.0,
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
        "betlas_beta_run_count": float(run_count),
        "betlas_beta_segment_to_run_ratio": _safe_ratio(segment_count, run_count),
        "betlas_beta_run_eight_score": _bell(float(run_count), 8.0, 2.0),
        "betlas_beta_run_sheet_multiplicity_mean": float(np.mean(multiplicities)),
        "betlas_beta_run_sheet_multiplicity_max": float(np.max(multiplicities)),
        "betlas_beta_run_multi_sheet_fraction": float(np.mean(np.asarray(multiplicities) > 1.0)),
        "betlas_beta_run_angular_coverage": float(coverage),
        "betlas_beta_run_largest_gap_fraction": float(largest_gap / (2.0 * math.pi)),
    }
    features.update(_summary(lengths, "betlas_beta_run_length"))
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
        "betlas_sheet_pair_top2_fraction": 0.0,
        "betlas_sheet_pair_size_balance": 0.0,
        "betlas_sheet_pair_centroid_distance": 0.0,
        "betlas_sheet_pair_centroid_distance_norm": 0.0,
        "betlas_sheet_pair_angular_separation_fraction": 0.0,
        "betlas_sheet_pair_valid_normal_abs_dot": 0.0,
        "betlas_sheet_pair_valid_face_alignment": 0.0,
        "betlas_sheet_pair_cross_min_distance": 0.0,
        "betlas_sheet_pair_cross_distance_mean": 0.0,
        "betlas_sheet_pair_cross_contact_density8": 0.0,
        "betlas_sheet_pair_cross_contact_density10": 0.0,
        "betlas_sheet_pair_reciprocal_nearest_fraction8": 0.0,
        "betlas_sheet_pair_valid_bilayer_score": 0.0,
        "betlas_sheet_pair_valid_lobe_score": 0.0,
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
        "betlas_sheet_pair_top2_fraction": float(top2_fraction),
        "betlas_sheet_pair_size_balance": float(size_balance),
        "betlas_sheet_pair_centroid_distance": center_distance,
        "betlas_sheet_pair_centroid_distance_norm": _safe_ratio(center_distance, radial_scale + 1.0),
        "betlas_sheet_pair_angular_separation_fraction": float(angular_sep),
        "betlas_sheet_pair_valid_normal_abs_dot": float(normal_abs_dot),
        "betlas_sheet_pair_valid_face_alignment": float(face_alignment),
        "betlas_sheet_pair_cross_min_distance": float(np.min(pair_arr)) if len(pair_arr) else 0.0,
        "betlas_sheet_pair_cross_distance_mean": float(np.mean(pair_arr)) if len(pair_arr) else 0.0,
        "betlas_sheet_pair_cross_contact_density8": float(np.mean(pair_arr <= 8.0)) if len(pair_arr) else 0.0,
        "betlas_sheet_pair_cross_contact_density10": float(np.mean(pair_arr <= 10.0)) if len(pair_arr) else 0.0,
        "betlas_sheet_pair_reciprocal_nearest_fraction8": float(np.mean(nearest <= 8.0)) if len(nearest) else 0.0,
        "betlas_sheet_pair_valid_bilayer_score": float(bilayer_score),
        "betlas_sheet_pair_valid_lobe_score": float(lobe_score),
        "betlas_sheet_pair_top_sheet_ids_equal": 1.0 if sheet_a == sheet_b else 0.0,
    }


def _sheet_sequence_topology_features(geometry: StructureGeometry) -> dict[str, float]:
    runs = _beta_runs(geometry)
    ranked, _by_sheet = _top_sheet_items(geometry)
    if len(runs) < 2 or len(ranked) < 2:
        return {
            "betlas_sheet_seq_top2_run_fraction": 0.0,
            "betlas_sheet_seq_top2_transition_fraction": 0.0,
            "betlas_sheet_seq_top2_block_count": 0.0,
            "betlas_sheet_seq_top2_longest_block_fraction": 0.0,
            "betlas_sheet_seq_top2_interleave_score": 0.0,
            "betlas_sheet_seq_top2_order_displacement": 0.0,
            "betlas_sheet_seq_greek_key_proxy": 0.0,
        }

    top2 = {ranked[0][0], ranked[1][0]}
    labels = [str(run["dominant_sheet_id"]) for run in runs]
    top2_mask = [label in top2 for label in labels]
    top2_labels = [label for label, keep in zip(labels, top2_mask, strict=False) if keep]
    if len(top2_labels) < 2:
        return {
            "betlas_sheet_seq_top2_run_fraction": len(top2_labels) / max(1, len(runs)),
            "betlas_sheet_seq_top2_transition_fraction": 0.0,
            "betlas_sheet_seq_top2_block_count": float(len(top2_labels)),
            "betlas_sheet_seq_top2_longest_block_fraction": 1.0 if top2_labels else 0.0,
            "betlas_sheet_seq_top2_interleave_score": 0.0,
            "betlas_sheet_seq_top2_order_displacement": 0.0,
            "betlas_sheet_seq_greek_key_proxy": 0.0,
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
        "betlas_sheet_seq_top2_run_fraction": float(run_fraction),
        "betlas_sheet_seq_top2_transition_fraction": transition_fraction,
        "betlas_sheet_seq_top2_block_count": float(len(block_lengths)),
        "betlas_sheet_seq_top2_longest_block_fraction": float(longest_block_fraction),
        "betlas_sheet_seq_top2_interleave_score": float(interleave_score),
        "betlas_sheet_seq_top2_order_displacement": float(displacement),
        "betlas_sheet_seq_greek_key_proxy": float(greek_key_proxy),
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
            "betlas_sheet_order_adjacent_seq_step_mean": 0.0,
            "betlas_sheet_order_adjacent_seq_step_max": 0.0,
            "betlas_sheet_order_nonlocal_fraction": 0.0,
            "betlas_sheet_order_inversion_fraction_mean": 0.0,
            "betlas_top2_sheet_order_adjacent_seq_step_mean": 0.0,
            "betlas_top2_sheet_order_nonlocal_fraction": 0.0,
            "betlas_top2_sheet_order_inversion_fraction_mean": 0.0,
            "betlas_jelly_roll_order_nonlocal_score": 0.0,
        }

    seq_order = sorted(range(len(segments)), key=lambda idx: _element_sequence_key(geometry, segments[idx]))
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
        "betlas_sheet_order_adjacent_seq_step_mean": float(np.mean(all_steps)) if all_steps else 0.0,
        "betlas_sheet_order_adjacent_seq_step_max": float(np.max(all_steps)) if all_steps else 0.0,
        "betlas_sheet_order_nonlocal_fraction": float(np.mean(all_nonlocal)) if all_nonlocal else 0.0,
        "betlas_sheet_order_inversion_fraction_mean": float(np.mean(all_inversions)) if all_inversions else 0.0,
        "betlas_top2_sheet_order_adjacent_seq_step_mean": float(np.mean(top2_steps)) if top2_steps else 0.0,
        "betlas_top2_sheet_order_nonlocal_fraction": top2_nonlocal_mean,
        "betlas_top2_sheet_order_inversion_fraction_mean": top2_inv_mean,
        "betlas_jelly_roll_order_nonlocal_score": float(
            top2_run_fraction * (0.5 * top2_nonlocal_mean + 0.5 * top2_inv_mean) * run_eight
        ),
    }


def _contact_graph_features(geometry: StructureGeometry) -> dict[str, float]:
    beta_geoms = beta_element_geometries(geometry)
    n = len(beta_geoms)
    if n < 2:
        return {
            "betlas_contact8_edge_density": 0.0,
            "betlas_contact10_edge_density": 0.0,
            "betlas_contact8_cross_sheet_fraction": 0.0,
            "betlas_contact8_component_count": float(n),
            "betlas_contact8_largest_component_fraction": float(n),
            "betlas_contact8_cycle_rank_norm": 0.0,
            "betlas_contact8_degree2_fraction": 0.0,
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
        "betlas_contact8_edge_density": _safe_ratio(len(edges8), possible_edges),
        "betlas_contact10_edge_density": _safe_ratio(len(edges10), possible_edges),
        "betlas_contact8_cross_sheet_fraction": _safe_ratio(cross8, len(edges8)),
        "betlas_contact8_component_count": float(component_count),
        "betlas_contact8_largest_component_fraction": max(component_sizes) / max(1, n),
        "betlas_contact8_cycle_rank_norm": _safe_ratio(cycle_rank, n),
        "betlas_contact8_degree2_fraction": float(np.mean(degrees == 2.0)),
    }


def _contact_sequence_topology_features(geometry: StructureGeometry) -> dict[str, float]:
    beta_geoms = beta_element_geometries(geometry)
    n = len(beta_geoms)
    if n < 2:
        return {
            "betlas_contact8_seq_gap_mean": 0.0,
            "betlas_contact8_seq_gap_max": 0.0,
            "betlas_contact8_nonlocal_fraction": 0.0,
            "betlas_contact8_cross_sheet_nonlocal_fraction": 0.0,
            "betlas_contact10_seq_gap_mean": 0.0,
            "betlas_contact10_nonlocal_fraction": 0.0,
        }
    seq_order = sorted(range(n), key=lambda idx: _element_sequence_key(geometry, beta_geoms[idx].element))
    seq_rank = {idx: rank for rank, idx in enumerate(seq_order)}
    gaps8: list[float] = []
    gaps10: list[float] = []
    cross_nonlocal8: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if beta_geoms[i].element.chain_id != beta_geoms[j].element.chain_id:
                continue
            distance = _min_pair_distance(beta_geoms[i].coords, beta_geoms[j].coords)
            gap = float(abs(seq_rank[i] - seq_rank[j]))
            if distance <= 10.0:
                gaps10.append(gap)
            if distance <= 8.0:
                gaps8.append(gap)
                if beta_geoms[i].element.sheet_id != beta_geoms[j].element.sheet_id:
                    cross_nonlocal8.append(1.0 if gap > 2.0 else 0.0)
    return {
        "betlas_contact8_seq_gap_mean": float(np.mean(gaps8)) if gaps8 else 0.0,
        "betlas_contact8_seq_gap_max": float(np.max(gaps8)) if gaps8 else 0.0,
        "betlas_contact8_nonlocal_fraction": float(np.mean(np.asarray(gaps8) > 2.0)) if gaps8 else 0.0,
        "betlas_contact8_cross_sheet_nonlocal_fraction": float(np.mean(cross_nonlocal8))
        if cross_nonlocal8
        else 0.0,
        "betlas_contact10_seq_gap_mean": float(np.mean(gaps10)) if gaps10 else 0.0,
        "betlas_contact10_nonlocal_fraction": float(np.mean(np.asarray(gaps10) > 2.0)) if gaps10 else 0.0,
    }


def _axis_slice_features(
    beta_points: np.ndarray,
    segment_midpoints: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    min_points_per_slice: int = 4,
) -> dict[str, float]:
    summary, _bins = _compute_axis_slice_summary(
        beta_points,
        segment_midpoints,
        origin,
        axis,
        config=SliceConfig(min_points_per_slice=min_points_per_slice),
    )
    return summary


def _strand_order_features(
    segment_midpoints: np.ndarray,
    geometry: StructureGeometry,
    origin: np.ndarray,
    axis: np.ndarray,
) -> dict[str, float]:
    n = len(segment_midpoints)
    if n < 3:
        return {
            "betlas_strand_order_displacement_mean": 0.0,
            "betlas_strand_order_adjacent_seq_step_median": 0.0,
            "betlas_strand_ntc_parallel_fraction": 0.0,
            "betlas_strand_ntc_antiparallel_fraction": 0.0,
            "betlas_strand_axis_abs_alignment_mean": 0.0,
        }
    xy, _z, _xyz = project_to_axis(segment_midpoints, origin, axis)
    angles = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * math.pi)
    angular_order = np.argsort(angles)
    seq_order = sorted(range(n), key=lambda idx: _element_sequence_key(geometry, geometry.beta_segments[idx]))
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
        "betlas_strand_order_displacement_mean": displacement,
        "betlas_strand_order_adjacent_seq_step_median": float(np.median(cyclic_steps)),
        "betlas_strand_ntc_parallel_fraction": parallel_fraction,
        "betlas_strand_ntc_antiparallel_fraction": antiparallel_fraction,
        "betlas_strand_axis_abs_alignment_mean": abs_alignment,
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
            "betlas_alpha_shell_radial_delta": 0.0,
            "betlas_helix_beta_radius_ratio": 0.0,
        }
    beta_mid = np.array([item.midpoint for item in beta_geoms], dtype=float)
    helix_mid = np.array([item.midpoint for item in helix_geoms], dtype=float)
    beta_xy, _bz, _ = project_to_axis(beta_mid, origin, axis)
    helix_xy, _hz, _ = project_to_axis(helix_mid, origin, axis)
    beta_radius = float(np.mean(np.linalg.norm(beta_xy, axis=1)))
    helix_radius = float(np.mean(np.linalg.norm(helix_xy, axis=1)))
    return {
        "betlas_alpha_shell_radial_delta": helix_radius - beta_radius,
        "betlas_helix_beta_radius_ratio": helix_radius / (beta_radius + EPS),
    }


def plane_applicability(geometry: StructureGeometry) -> dict[str, Any]:
    """Check whether both dominant sheet normals are uniquely defined."""
    ranked = _top_sheet_items(geometry)[0]
    if len(ranked) < 2:
        return {"plane_status": "no_pair", "plane_applicable": True}
    result: dict[str, Any] = {"plane_status": "available", "plane_applicable": True}
    for position, (sheet, items) in enumerate(ranked[:2], 1):
        coords = np.concatenate([item.coords for item in items])
        eigenvalues = np.linalg.eigvalsh(np.cov(coords.T)) if len(coords) >= 2 else np.zeros(3)
        threshold = max(1e-12, 1e-10 * max(float(eigenvalues[2]), 0.))
        valid = len(coords) >= 3 and eigenvalues[1] - eigenvalues[0] > threshold
        result.update({f"sheet{position}_id": sheet, f"sheet{position}_ca_n": len(coords),
                       f"sheet{position}_lambda0": float(eigenvalues[0]), f"sheet{position}_lambda1": float(eigenvalues[1]),
                       f"sheet{position}_lambda2": float(eigenvalues[2]), f"sheet{position}_normal_valid": bool(valid)})
        if not valid:
            result.update(plane_status="normal_unavailable", plane_applicable=False)
    return result


def extract_classifier_features(geometry: StructureGeometry) -> dict[str, Any]:
    """Extract the 136 ordered descriptors and their applicability metadata."""
    return extract_signature(geometry).features


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

    parse_ok = bool(geometry.residues and geometry.beta_segments)
    features: dict[str, float | int | str] = {
        "betlas_parse_ok": 1 if parse_ok else 0,
        "betlas_residue_count": float(residue_count),
        "betlas_beta_residue_count": float(len(beta_residue_indices)),
        "betlas_helix_residue_count": float(len(helix_residue_indices)),
        "betlas_beta_residue_fraction": len(beta_residue_indices) / max(1, residue_count),
        "betlas_helix_residue_fraction": len(helix_residue_indices) / max(1, residue_count),
        "betlas_beta_strand_count": float(len(geometry.beta_segments)),
        "betlas_helix_count": float(len(geometry.helices)),
        "betlas_pca_elongation": _nan_to_zero(pca_elongation),
        "betlas_pca_flatness": _nan_to_zero(pca_flatness),
        "betlas_axis_best_name": axis_name,
        "betlas_axis_best_score": float(axis_score),
        "betlas_axis_best_is_strand_axis": 1.0 if axis_name == "strand_axis" else 0.0,
        "betlas_axis_best_is_point_pc1": 1.0 if axis_name == "point_pc1" else 0.0,
        "betlas_axis_best_is_point_pc2": 1.0 if axis_name == "point_pc2" else 0.0,
        "betlas_axis_best_is_point_pc3": 1.0 if axis_name == "point_pc3" else 0.0,
        "betlas_z_continuity_fraction": best_axis_features["z_continuity_fraction"],
        "betlas_parallel_sheet_order_fraction": sense_counts.get("parallel", 0) / max(1, len(sheet_senses)),
        "betlas_antiparallel_sheet_order_fraction": sense_counts.get("anti-parallel", 0) / max(1, len(sheet_senses)),
    }
    for key, value in best_axis_features.items():
        features[f"betlas_axis_best_{key}"] = float(value)
    features.update(_summary(strand_lengths, "betlas_strand_length"))
    features.update(_summary(helix_lengths, "betlas_helix_length"))
    features.update(_sequence_features(geometry))
    features.update(_sheet_features(geometry, origin, direction))
    features.update(_beta_run_features(geometry, segment_midpoints, origin, direction))
    features.update(_sheet_pair_packing_features(geometry, origin, direction))
    features.update(_sheet_sequence_topology_features(geometry))
    features.update(_sheet_order_topology_features(geometry))
    features.update(_contact_graph_features(geometry))
    features.update(_contact_sequence_topology_features(geometry))
    features.update(_strand_order_features(segment_midpoints, geometry, origin, direction))
    features.update(_alpha_shell_features(geometry, origin, direction))

    if geometry.secondary_structure_source == "dssp":
        # Strand IDs are identifiers, not a spatial ordering. A ladder graph can
        # branch or cycle, so legacy linear-order statistics have no unique value.
        for name in _sheet_order_topology_features(geometry):
            features[name] = math.nan
        features["betlas_sheet_order_status"] = "unavailable_graph_has_no_declared_linear_order"
        edges = geometry.sheet_connections
        features["betlas_dssp_ladder_edge_count"] = float(len(edges))
        features["betlas_parallel_sheet_order_fraction"] = sum(edge[2] == "parallel" for edge in edges) / max(1, len(edges))
        features["betlas_antiparallel_sheet_order_fraction"] = sum(edge[2] == "anti-parallel" for edge in edges) / max(1, len(edges))
        features["betlas_secondary_structure_source"] = "dssp"
        features["betlas_domain_segment_count"] = float(geometry.domain_segment_count)
        features["betlas_ladder_pairing_status"] = (
            "partial_clipped_ladders" if "clipped_ladder_pairing_unavailable" in geometry.warnings
            else "available" if edges else "no_retained_ladders"
        )
        if len({residue.chain_id for residue in geometry.residues}) > 1:
            # Sequence topology across separate chains has no single ordering.
            for name in features:
                if name.startswith("betlas_sheet_seq_") or name in {
                    "betlas_strand_order_displacement_mean", "betlas_strand_order_adjacent_seq_step_median"
                }:
                    features[name] = math.nan
            features["betlas_sequence_order_status"] = "unavailable_across_chains"
        else:
            features["betlas_sequence_order_status"] = "polymer_label_order"

    if len(segment_midpoints) >= 2:
        xy, _z, _ = project_to_axis(segment_midpoints, origin, direction)
        gaps = circular_gaps(np.arctan2(xy[:, 1], xy[:, 0]))
        features["betlas_angular_gap_cv"] = float(np.std(gaps) / (np.mean(gaps) + EPS))
    else:
        features["betlas_angular_gap_cv"] = 0.0

    plane = plane_applicability(geometry)
    if not plane["plane_applicable"]:
        features.update(dict.fromkeys(VALID_PLANE_COLUMNS, math.nan))
    if not beta_geoms:
        moment_status = "no_beta_segments"
        moments = np.full(8, np.nan)
    else:
        xy, _z, _xyz = project_to_axis(segment_midpoints, origin, direction)
        radial = np.linalg.norm(xy, axis=1)
        moment_status = ("too_few_beta_nodes" if len(beta_geoms) < 4 else
                         "undefined_radial_angle" if (radial <= EPS).any() else "available")
        moments = circular_moments(np.arctan2(xy[:, 1], xy[:, 0])) if moment_status == "available" else np.full(8, np.nan)
    features.update(dict(zip(CIRCULAR_MOMENT_COLUMNS, map(float, moments), strict=True)))
    features.update(moment_status=moment_status, plane_status=plane["plane_status"])
    features = apply_applicability(features)
    features = {**{name: features.get(name, math.nan) for name in FEATURE_COLUMNS},
                **{name: value for name, value in features.items() if name not in FEATURE_COLUMNS}}

    return GeometrySignature(
        domain=geometry.domain,
        features=features,
        warnings=geometry.warnings,
    )


def extract_feature_row(domain: DomainCandidate, mmcif_path: Path) -> dict[str, Any]:
    try:
        geometry = build_structure_geometry(domain, mmcif_path)
        signature = extract_signature(geometry)
        row = signature.to_row()
        row["betlas_error"] = ""
        return row
    except Exception as exc:
        row = {**domain.to_dict(), **dict.fromkeys(FEATURE_COLUMNS, math.nan), "feature_schema": FEATURE_SCHEMA}
        row.update(
            {
                "betlas_parse_ok": 0,
                "betlas_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return row


def extract_structure_features(
    structure_path: str | Path,
    *,
    chain_id: str,
    residue_ranges: str = "",
    record_id: str | None = None,
    domain_id: str | None = None,
    pdb_id: str | None = None,
    model_id: int = 0,
) -> dict[str, Any]:
    """Extract Betlas features for one user-provided mmCIF chain."""

    path = Path(structure_path)
    if not is_mmcif_path(path):
        raise ValueError(
            "Betlas single-structure feature extraction currently accepts mmCIF files "
            "(.cif, .mmcif, .cif.gz, .mmcif.gz). Use readout commands for PDB inputs."
        )
    if not path.exists():
        raise FileNotFoundError(f"structure file does not exist: {path}")
    if not str(chain_id).strip():
        raise ValueError("chain id is required for single-structure feature extraction")
    if int(model_id) < 0:
        raise ValueError("model_id must be a zero-based non-negative integer")

    stem = path.name.split(".", 1)[0].lower()
    domain = DomainCandidate(
        record_id=record_id or f"{stem}_{chain_id}",
        pdb_id=(pdb_id or stem).lower(),
        chain_id=str(chain_id),
        domain_id=domain_id or f"{stem}_{chain_id}",
        residue_ranges=residue_ranges,
        model_id=int(model_id),
        fold_label_final="unlabeled",
        evidence_level="user_input",
        label_source_primary="user_input",
    )
    geometry = build_structure_geometry(domain, path)
    if not geometry.residues:
        chains = available_auth_chain_ids(path)
        available = ", ".join(chains) if chains else "<none>"
        range_hint = (
            f"; requested residue_ranges={residue_ranges!r}. If ranges include chain suffixes, "
            f"they must select residues from --chain {chain_id!r}"
            if str(residue_ranges).strip()
            else ""
        )
        raise ValueError(
            f"no residues were selected for chain {chain_id!r} in model_id={int(model_id)}{range_hint}; "
            f"available author chain ids: {available}"
        )
    if not geometry.beta_segments:
        warnings = "; ".join(geometry.warnings) if geometry.warnings else "no beta-sheet segments"
        raise ValueError(
            "Betlas feature extraction requires parsed beta-sheet segments for single-structure "
            f"mode; none were found for chain {chain_id!r}. Parser warnings: {warnings}. "
            "Use --write-failed-row if you need a status-only CSV row."
        )
    signature = extract_signature(geometry)
    row = signature.to_row()
    row["betlas_error"] = ""
    return row


def assert_no_diagnostic_label_leakage(row: dict[str, Any]) -> None:
    forbidden_targets = {
        "fold_label_final",
        "fold_label_candidates",
        "evidence_level",
        "qc_status",
        "allowed_for_benchmark",
    }
    for key in forbidden_targets:
        value = str(row.get(key, ""))
        if value.startswith(DIAGNOSTIC_PREFIX):
            raise ValueError(f"label field {key} was populated from diagnostic field {value!r}")
