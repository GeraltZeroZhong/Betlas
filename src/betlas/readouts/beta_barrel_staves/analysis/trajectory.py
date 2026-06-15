from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import atan2, degrees
from typing import Any

import numpy as np

from .numeric import angle_distance, circular_mean


@dataclass(frozen=True)
class TrajectoryMergeResult:
    support: dict[int, int]
    fragments: dict[int, list[int]]
    merge_count: int


class _DisjointSet:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


class TrajectoryMerger:
    """Merge DSSP beta-run trajectory fragments using slice-only geometry."""

    def __init__(self, count_config: Any):
        self.count_config = count_config

    def trajectory_stats(
        self,
        usable_layers: list[dict[str, object]],
    ) -> dict[int, dict[str, object]]:
        observations: dict[int, dict[str, list[float] | set[int] | dict[int, float]]] = defaultdict(
            lambda: {"z": [], "theta": [], "layers": set(), "layer_z": {}}
        )
        for layer_index, layer in enumerate(usable_layers):
            points = np.asarray(layer.get("_points", []), dtype=float)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 4:
                continue
            z_value = float(layer.get("z", layer_index) or layer_index)
            center = np.median(points[:, :2], axis=0)
            for point in points:
                strand_id = int(point[3])
                theta = float(atan2(float(point[1] - center[1]), float(point[0] - center[0])))
                observations[strand_id]["z"].append(z_value)
                observations[strand_id]["theta"].append(theta)
                layers = observations[strand_id]["layers"]
                if isinstance(layers, set):
                    layers.add(layer_index)
                layer_z = observations[strand_id]["layer_z"]
                if isinstance(layer_z, dict):
                    layer_z[layer_index] = z_value

        stats: dict[int, dict[str, object]] = {}
        for strand_id, values in observations.items():
            z_values = np.asarray(values["z"], dtype=float)
            theta_values = np.asarray(values["theta"], dtype=float)
            layer_set = set(values["layers"]) if isinstance(values["layers"], set) else set()
            layer_z = dict(values["layer_z"]) if isinstance(values["layer_z"], dict) else {}
            if z_values.size == 0:
                continue
            order = np.argsort(z_values, kind="mergesort")
            z_sorted = z_values[order]
            theta_unwrapped = np.unwrap(theta_values[order])
            if z_sorted.size >= 2 and float(np.ptp(z_sorted)) > 0.0:
                slope, intercept = np.polyfit(z_sorted, theta_unwrapped, 1)
            else:
                slope = 0.0
                intercept = float(theta_unwrapped[0])
            stats[strand_id] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "mean_angle": circular_mean(list(theta_values)),
                "layers": layer_set,
                "layer_z": layer_z,
                "support": int(len(layer_set)),
                "min_layer": int(min(layer_set)),
                "max_layer": int(max(layer_set)),
                "min_z": float(np.min(z_sorted)),
                "max_z": float(np.max(z_sorted)),
            }
        return stats

    def pair_compatible(
        self,
        first_id: int,
        second_id: int,
        first: dict[str, object],
        second: dict[str, object],
        *,
        max_run_id_gap: int | None = None,
        angle_tolerance_deg: float | None = None,
    ) -> bool:
        count_cfg = self.count_config
        run_id_gap_limit = (
            int(count_cfg.trajectory_merge_max_run_id_gap)
            if max_run_id_gap is None
            else int(max_run_id_gap)
        )
        if abs(first_id - second_id) > run_id_gap_limit:
            return False

        first_layers = set(first["layers"])
        second_layers = set(second["layers"])
        overlap = len(first_layers & second_layers)
        min_support = min(int(first["support"]), int(second["support"]))
        max_overlap = float(count_cfg.trajectory_merge_max_overlap_fraction) * max(1, min_support)
        if overlap > max_overlap:
            return False

        first_min = int(first["min_layer"])
        first_max = int(first["max_layer"])
        second_min = int(second["min_layer"])
        second_max = int(second["max_layer"])
        layer_gap = max(0, max(first_min, second_min) - min(first_max, second_max) - 1)
        if layer_gap > int(count_cfg.trajectory_merge_max_layer_gap):
            return False

        if overlap > 0:
            first_layer_z = dict(first.get("layer_z", {}))
            second_layer_z = dict(second.get("layer_z", {}))
            overlap_z = [
                0.5 * (float(first_layer_z[layer]) + float(second_layer_z[layer]))
                for layer in sorted(first_layers & second_layers)
                if layer in first_layer_z and layer in second_layer_z
            ]
            compare_z = float(np.median(np.asarray(overlap_z, dtype=float))) if overlap_z else 0.0
        else:
            first_min_z = float(first.get("min_z", float(first_min)))
            first_max_z = float(first.get("max_z", float(first_max)))
            second_min_z = float(second.get("min_z", float(second_min)))
            second_max_z = float(second.get("max_z", float(second_max)))
            compare_z = (min(first_max_z, second_max_z) + max(first_min_z, second_min_z)) / 2.0

        first_theta = float(first["slope"]) * compare_z + float(first["intercept"])
        second_theta = float(second["slope"]) * compare_z + float(second["intercept"])
        angle_delta = degrees(angle_distance(first_theta, second_theta))
        angle_tolerance = (
            float(count_cfg.trajectory_merge_angle_tolerance_deg)
            if angle_tolerance_deg is None
            else float(angle_tolerance_deg)
        )
        if angle_delta > angle_tolerance:
            return False

        first_span = max(1, first_max - first_min)
        second_span = max(1, second_max - second_min)
        if first_span >= 2 and second_span >= 2:
            slope_delta = degrees(abs(float(first["slope"]) - float(second["slope"])))
            if slope_delta > float(count_cfg.trajectory_merge_slope_tolerance_deg_per_layer):
                return False

        return True

    def merge(
        self,
        strand_support: dict[int, int],
        usable_layers: list[dict[str, object]],
        *,
        max_run_id_gap: int | None = None,
        angle_tolerance_deg: float | None = None,
    ) -> TrajectoryMergeResult:
        if not bool(self.count_config.trajectory_merge_enabled) or not strand_support:
            return TrajectoryMergeResult(
                dict(strand_support),
                {key: [key] for key in strand_support},
                0,
            )

        trajectory_stats = self.trajectory_stats(usable_layers)
        strand_ids = sorted(strand_support)
        if len(strand_ids) <= 1 or not trajectory_stats:
            return TrajectoryMergeResult(
                dict(strand_support),
                {key: [key] for key in strand_support},
                0,
            )

        groups = _DisjointSet(strand_ids)
        for index, first_id in enumerate(strand_ids):
            first = trajectory_stats.get(first_id)
            if first is None:
                continue
            for second_id in strand_ids[index + 1 :]:
                second = trajectory_stats.get(second_id)
                if second is None:
                    continue
                if self.pair_compatible(
                    first_id,
                    second_id,
                    first,
                    second,
                    max_run_id_gap=max_run_id_gap,
                    angle_tolerance_deg=angle_tolerance_deg,
                ):
                    groups.union(first_id, second_id)

        fragments_by_root: dict[int, list[int]] = defaultdict(list)
        for strand_id in strand_ids:
            fragments_by_root[groups.find(strand_id)].append(strand_id)

        compatible_fragment_groups: list[list[int]] = []
        for fragments in fragments_by_root.values():
            compatible_fragment_groups.extend(
                self._split_to_pairwise_compatible_groups(
                    sorted(fragments),
                    trajectory_stats,
                    max_run_id_gap=max_run_id_gap,
                    angle_tolerance_deg=angle_tolerance_deg,
                )
            )

        merged_support: dict[int, int] = {}
        merged_fragments: dict[int, list[int]] = {}
        for merged_id, fragments in enumerate(
            sorted(compatible_fragment_groups, key=lambda values: (min(values), len(values)))
        ):
            layer_union: set[int] = set()
            for fragment_id in fragments:
                fragment_stats = trajectory_stats.get(fragment_id)
                if fragment_stats is not None:
                    layer_union.update(set(fragment_stats["layers"]))
                else:
                    layer_union.update(range(int(strand_support[fragment_id])))
            merged_support[merged_id] = len(layer_union) if layer_union else max(
                strand_support[fragment_id] for fragment_id in fragments
            )
            merged_fragments[merged_id] = sorted(fragments)

        merge_count = sum(max(0, len(fragments) - 1) for fragments in merged_fragments.values())
        return TrajectoryMergeResult(merged_support, merged_fragments, int(merge_count))

    def _split_to_pairwise_compatible_groups(
        self,
        fragments: list[int],
        trajectory_stats: dict[int, dict[str, object]],
        *,
        max_run_id_gap: int | None,
        angle_tolerance_deg: float | None,
    ) -> list[list[int]]:
        if len(fragments) <= 2:
            return [fragments]

        groups: list[list[int]] = []
        for fragment_id in fragments:
            fragment_stats = trajectory_stats.get(fragment_id)
            if fragment_stats is None:
                groups.append([fragment_id])
                continue

            for group in groups:
                if all(
                    other_id in trajectory_stats
                    and
                    self.pair_compatible(
                        fragment_id,
                        other_id,
                        fragment_stats,
                        trajectory_stats[other_id],
                        max_run_id_gap=max_run_id_gap,
                        angle_tolerance_deg=angle_tolerance_deg,
                    )
                    for other_id in group
                ):
                    group.append(fragment_id)
                    break
            else:
                groups.append([fragment_id])
        return groups

    def should_use_extended_merge(
        self,
        *,
        candidate_strands: int,
        support_consensus_strands: int,
        layer_count_q90: int,
        layer_count_max: int,
    ) -> bool:
        count_cfg = self.count_config
        if not bool(count_cfg.trajectory_merge_extended_enabled):
            return False
        if candidate_strands < int(count_cfg.trajectory_merge_extended_min_candidate_strands):
            return False
        if layer_count_q90 < int(count_cfg.trajectory_merge_extended_min_layer_q90):
            return False
        support_gap = support_consensus_strands - layer_count_max
        return support_gap >= int(count_cfg.trajectory_merge_extended_min_support_gap)
