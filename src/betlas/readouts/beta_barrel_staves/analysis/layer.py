from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AnalyzerConfig
from ..constants import EPSILON, ROBUST_SIGMA_SCALE, THREE_SIGMA_MULTIPLIER, TOLERANCE
from ..geometry.analysis_utils import (
    angular_gap_stats,
    collapse_points_by_strand,
    nearest_neighbor_spacing_stats,
    sequence_angle_order_stats,
)


@dataclass(frozen=True)
class CandidatePoints:
    points: np.ndarray
    trim_left: int = 0
    trim_right: int = 0
    radial_outliers: int = 0


class LayerAnalyzer:
    """Evaluate and robustly clean one slice layer before strand-count aggregation."""

    def __init__(self, config: AnalyzerConfig):
        self.config = config

    @staticmethod
    def _empty_result(z_value: float, raw_count: int, reason: str) -> dict[str, object]:
        return {
            "z": float(z_value),
            "raw_points": int(raw_count),
            "collapsed_points": 0,
            "selected_points": 0,
            "layer_count": 0,
            "strand_ids": [],
            "usable": False,
            "geometric_pass": False,
            "reason": reason,
            "seq_core_trim_left": 0,
            "seq_core_trim_right": 0,
            "radial_outliers": 0,
        }

    def _point_matrix_for_rules(self, points: np.ndarray) -> np.ndarray:
        if points.shape[1] >= 4:
            return points[:, :4]
        return points[:, :3]

    def _strand_ids(self, points: np.ndarray) -> list[int]:
        if points.ndim != 2:
            return []
        if points.shape[1] >= 4:
            return sorted(int(value) for value in np.unique(points[:, 3].astype(int)))
        return list(range(int(points.shape[0])))

    def _radial_outlier_mask(self, points: np.ndarray) -> np.ndarray:
        outlier_cfg = self.config.rules.radial_outlier
        point_count = int(points.shape[0]) if points.ndim == 2 else 0
        if (
            not outlier_cfg.enabled
            or points.ndim != 2
            or points.shape[1] < 2
            or point_count < int(outlier_cfg.min_points)
        ):
            return np.ones(point_count, dtype=bool)

        points_xy = np.asarray(points[:, :2], dtype=float)
        center = np.median(points_xy, axis=0)
        radial_distance = np.sqrt(np.sum((points_xy - center) ** 2, axis=1))
        median_radius = float(np.median(radial_distance))
        mad_radius = float(np.median(np.abs(radial_distance - median_radius)))
        robust_sigma = float(ROBUST_SIGMA_SCALE * mad_radius)

        if median_radius <= TOLERANCE:
            return np.ones(point_count, dtype=bool)

        min_radius_ratio = float(getattr(outlier_cfg, "min_radius_ratio", 0.0))
        ratio_mask = radial_distance <= (float(outlier_cfg.max_radius_ratio) * median_radius)
        if min_radius_ratio > 0.0:
            ratio_mask = ratio_mask & (radial_distance >= (min_radius_ratio * median_radius))
        if robust_sigma > EPSILON:
            z_mask = np.abs(radial_distance - median_radius) <= (
                THREE_SIGMA_MULTIPLIER * robust_sigma
            )
            keep_mask = ratio_mask & z_mask
        else:
            keep_mask = ratio_mask

        return keep_mask

    def _apply_radial_filter(self, points: np.ndarray) -> tuple[np.ndarray, int]:
        keep_mask = self._radial_outlier_mask(points)
        if keep_mask.size == 0:
            return points, 0
        filtered = points[keep_mask]
        return filtered, int(points.shape[0] - filtered.shape[0])

    def _evaluate_nn_rule(
        self,
        points_xy: np.ndarray,
    ) -> tuple[tuple[float, float, float, float] | None, bool]:
        nn_cfg = self.config.rules.nearest_neighbor
        if not nn_cfg.enabled:
            return None, True

        stats = nearest_neighbor_spacing_stats(points_xy)
        if stats is None:
            return None, True

        _, _, robust_cv, inlier_fraction = stats
        is_valid = (robust_cv <= nn_cfg.max_robust_cv) and (
            inlier_fraction >= nn_cfg.min_inlier_frac
        )
        return stats, is_valid

    def _evaluate_angle_rule(
        self,
        points: np.ndarray,
        points_xy: np.ndarray,
    ) -> tuple[
        tuple[float, float, int, float, float] | None,
        dict[str, float | int] | None,
        bool,
    ]:
        angle_cfg = self.config.rules.angle
        if not angle_cfg.enabled:
            return None, None, True

        angle_stats = angular_gap_stats(points_xy)
        gap_ok = True
        if angle_stats is not None:
            gap_ok = float(angle_stats[0]) <= float(angle_cfg.max_gap_deg)

        order_stats = None
        order_ok = True
        if angle_cfg.order.enabled:
            order_stats = sequence_angle_order_stats(points, angle_cfg.order)
            if order_stats is not None:
                order_ok = (
                    float(order_stats["order_local_frac"])
                    >= float(angle_cfg.order.min_local_frac)
                    and float(order_stats["order_mean_circ_dist_norm"])
                    <= float(angle_cfg.order.max_mean_circ_dist_norm)
                )

        return angle_stats, order_stats, (gap_ok and order_ok)

    def _evaluate_candidate(self, candidate: CandidatePoints) -> dict[str, object]:
        count_cfg = self.config.count
        points = np.asarray(candidate.points, dtype=float)
        strand_ids = self._strand_ids(points)
        layer_count = int(len(strand_ids))
        count_ok = (
            layer_count >= int(count_cfg.min_points_per_layer)
            and int(count_cfg.min_count) <= layer_count <= int(count_cfg.max_count)
        )

        nn_stats = None
        angle_stats = None
        order_stats = None
        nn_ok = True
        angle_ok = True
        if layer_count >= int(self.config.layer_quality.min_geometry_points) and points.shape[1] >= 2:
            points_xy = np.asarray(points[:, :2], dtype=float)
            nn_stats, nn_ok = self._evaluate_nn_rule(points_xy)
            angle_stats, order_stats, angle_ok = self._evaluate_angle_rule(
                self._point_matrix_for_rules(points),
                points_xy,
            )

        geometric_pass = bool(nn_ok and angle_ok)
        usable = bool(count_ok and (geometric_pass or not count_cfg.require_geometric_consistency))
        if layer_count < int(count_cfg.min_points_per_layer):
            reason = f"Too few usable strand intersections ({layer_count} < {count_cfg.min_points_per_layer})"
        elif layer_count < int(count_cfg.min_count):
            reason = f"Layer count is below configured minimum ({layer_count} < {count_cfg.min_count})"
        elif layer_count > int(count_cfg.max_count):
            reason = f"Layer count exceeds configured maximum ({layer_count} > {count_cfg.max_count})"
        elif not geometric_pass:
            reason = "Geometric consistency is weak"
        else:
            reason = "OK"

        order_local_frac = float(order_stats.get("order_local_frac", 1.0)) if order_stats else 1.0
        order_mean_circ = (
            float(order_stats.get("order_mean_circ_dist_norm", 0.0)) if order_stats else 0.0
        )
        nn_inlier_frac = float(nn_stats[3]) if nn_stats else 1.0
        angle_margin = 0.0
        if angle_stats is not None:
            angle_margin = float(self.config.rules.angle.max_gap_deg) - float(angle_stats[0])

        quality_cfg = self.config.layer_quality
        capped_angle_margin = min(
            float(quality_cfg.angle_margin_cap),
            max(
                -float(quality_cfg.angle_margin_cap),
                angle_margin / float(quality_cfg.angle_margin_scale_deg),
            ),
        )
        quality = (
            (float(quality_cfg.usable_bonus) if usable else 0.0)
            + (float(quality_cfg.geometric_pass_bonus) if geometric_pass else 0.0)
            + nn_inlier_frac
            + order_local_frac
            - order_mean_circ
            + capped_angle_margin
        )

        return {
            "points": points,
            "selected_points": int(points.shape[0]),
            "layer_count": layer_count,
            "strand_ids": strand_ids,
            "usable": usable,
            "geometric_pass": geometric_pass,
            "reason": reason,
            "seq_core_trim_left": int(candidate.trim_left),
            "seq_core_trim_right": int(candidate.trim_right),
            "radial_outliers": int(candidate.radial_outliers),
            "quality": float(quality),
            "nn_median": nn_stats[0] if nn_stats else None,
            "nn_sigma": nn_stats[1] if nn_stats else None,
            "nn_cv": nn_stats[2] if nn_stats else None,
            "nn_inlier_frac": nn_stats[3] if nn_stats else None,
            "angle_max_gap_deg": angle_stats[0] if angle_stats else None,
            "angle_coverage_deg": angle_stats[1] if angle_stats else None,
            "angle_used_n": angle_stats[2] if angle_stats else None,
            "order_used_n": order_stats.get("order_used_n") if order_stats else None,
            "order_local_frac": order_stats.get("order_local_frac") if order_stats else None,
            "order_mean_circ_dist_norm": (
                order_stats.get("order_mean_circ_dist_norm") if order_stats else None
            ),
        }

    def _candidate_sort_key(self, evaluation: dict[str, object]) -> tuple[float, int, int, int]:
        return (
            float(evaluation.get("quality", 0.0)),
            -int(evaluation.get("seq_core_trim_left", 0))
            - int(evaluation.get("seq_core_trim_right", 0))
            - int(evaluation.get("radial_outliers", 0)),
            int(evaluation.get("layer_count", 0)),
            int(evaluation.get("selected_points", 0)),
        )

    def _material_core_improvement(
        self,
        full_evaluation: dict[str, object],
        candidate_evaluation: dict[str, object],
    ) -> bool:
        if bool(candidate_evaluation.get("usable")) and not bool(full_evaluation.get("usable")):
            return True
        if not bool(candidate_evaluation.get("usable")):
            return False

        quality_gain = float(candidate_evaluation.get("quality", 0.0)) - float(
            full_evaluation.get("quality", 0.0)
        )
        if quality_gain >= float(self.config.rules.sequence_core.min_quality_gain):
            return True

        candidate_count = int(candidate_evaluation.get("layer_count", 0))
        full_count = int(full_evaluation.get("layer_count", 0))
        trimmed_terminals = (
            int(candidate_evaluation.get("seq_core_trim_left", 0))
            + int(candidate_evaluation.get("seq_core_trim_right", 0))
        )
        full_order_local = float(full_evaluation.get("order_local_frac") or 1.0)
        candidate_order_local = float(candidate_evaluation.get("order_local_frac") or 1.0)
        full_order_global = float(full_evaluation.get("order_mean_circ_dist_norm") or 0.0)
        candidate_order_global = float(candidate_evaluation.get("order_mean_circ_dist_norm") or 0.0)
        full_nn_inlier = float(full_evaluation.get("nn_inlier_frac") or 1.0)
        candidate_nn_inlier = float(candidate_evaluation.get("nn_inlier_frac") or 1.0)
        quality_cfg = self.config.layer_quality
        order_or_spacing_improves = (
            (candidate_order_local - full_order_local)
            >= float(quality_cfg.terminal_order_local_gain)
            or (full_order_global - candidate_order_global)
            >= float(quality_cfg.terminal_order_global_gain)
            or (candidate_nn_inlier - full_nn_inlier)
            >= float(quality_cfg.terminal_nn_inlier_gain)
        )
        if (
            trimmed_terminals > 0
            and candidate_count < full_count
            and bool(candidate_evaluation.get("geometric_pass"))
            and order_or_spacing_improves
        ):
            return True

        return (
            candidate_count < full_count
            and int(candidate_evaluation.get("radial_outliers", 0)) > 0
            and bool(candidate_evaluation.get("geometric_pass"))
        )

    def _base_candidate(self, collapsed_points: np.ndarray) -> CandidatePoints:
        filtered_points, radial_outliers = self._apply_radial_filter(collapsed_points)
        return CandidatePoints(filtered_points, radial_outliers=radial_outliers)

    def _sequence_core_candidates(self, collapsed_points: np.ndarray) -> list[CandidatePoints]:
        core_cfg = self.config.rules.sequence_core
        if not core_cfg.enabled:
            return []

        if collapsed_points.ndim != 2 or collapsed_points.shape[0] == 0:
            return []
        if collapsed_points.shape[1] >= 4:
            ordered_points = collapsed_points[
                np.argsort(collapsed_points[:, 2], kind="mergesort")
            ]
        else:
            ordered_points = collapsed_points[np.argsort(collapsed_points[:, 2], kind="mergesort")]

        total_points = int(ordered_points.shape[0])
        min_points = int(self.config.count.min_points_per_layer)
        if total_points <= min_points:
            return []

        candidates: list[CandidatePoints] = []
        use_exhaustive_search = (
            bool(core_cfg.exhaustive_search)
            and total_points <= int(core_cfg.max_exhaustive_points)
        )
        if use_exhaustive_search:
            min_core_points = min_points
            trim_windows = (
                (start, total_points - stop)
                for start in range(total_points)
                for stop in range(start + min_core_points, total_points + 1)
                if not (start == 0 and stop == total_points)
            )
        else:
            max_trim = int(np.floor(total_points * float(core_cfg.max_terminal_trim_fraction)))
            min_core_points = max(
                min_points,
                int(np.ceil(total_points * float(core_cfg.min_core_fraction))),
            )
            trim_windows = (
                (trim_left, trim_right)
                for trim_left in range(max_trim + 1)
                for trim_right in range(max_trim + 1)
                if (trim_left != 0 or trim_right != 0)
                and trim_left + trim_right <= max_trim
            )

        for trim_left, trim_right in trim_windows:
            stop = total_points - trim_right
            if stop <= trim_left:
                continue
            candidate_points = ordered_points[trim_left:stop]
            if int(candidate_points.shape[0]) < min_core_points:
                continue
            filtered_points, radial_outliers = self._apply_radial_filter(candidate_points)
            candidates.append(
                CandidatePoints(
                    filtered_points,
                    trim_left=trim_left,
                    trim_right=trim_right,
                    radial_outliers=radial_outliers,
                )
            )
        return candidates

    def analyze_layer(self, z_value: float, raw_points: np.ndarray) -> dict[str, object]:
        raw_points = np.asarray(raw_points, dtype=float)
        raw_count = int(raw_points.shape[0]) if raw_points.ndim == 2 else 0
        if raw_points.ndim != 2 or raw_count == 0:
            return self._empty_result(z_value, raw_count, "No intersections")
        if raw_points.shape[1] < 3:
            return self._empty_result(
                z_value,
                raw_count,
                "Intersections require at least x, y, and sequence columns",
            )

        collapsed_points = collapse_points_by_strand(raw_points)
        if collapsed_points.ndim != 2:
            collapsed_points = raw_points

        base = self._base_candidate(collapsed_points)
        full_evaluation = self._evaluate_candidate(base)
        best_evaluation = full_evaluation

        for candidate in self._sequence_core_candidates(collapsed_points):
            evaluation = self._evaluate_candidate(candidate)
            if not self._material_core_improvement(full_evaluation, evaluation):
                continue
            if self._candidate_sort_key(evaluation) > self._candidate_sort_key(best_evaluation):
                best_evaluation = evaluation

        selected_points = np.asarray(best_evaluation.get("points", []), dtype=float)
        best_evaluation["_points"] = selected_points.tolist()
        best_evaluation.pop("points", None)
        best_evaluation.pop("quality", None)
        best_evaluation.update(
            {
                "z": float(z_value),
                "raw_points": raw_count,
                "collapsed_points": int(collapsed_points.shape[0]),
            }
        )
        return best_evaluation
