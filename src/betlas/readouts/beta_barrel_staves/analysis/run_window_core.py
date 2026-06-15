from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .numeric import quantile_count

AnalyzeWindow = Callable[[dict[float, list[tuple[float, ...]]]], dict[str, object]]


@dataclass(frozen=True)
class SliceProfileFeatures:
    layer_count_max: int
    layer_count_q90: int
    max_plateau_length: int
    largest_adjacent_drop: int
    absent_from_max: int
    terminal_support: int
    terminal_consensus: int
    total_support: int


@dataclass(frozen=True)
class WindowQuickScore:
    start_id: int
    stop_id: int
    score: float
    layer_count_q90: int
    layer_count_max: int
    z_coverage_fraction: float
    max_z_gap: int
    outside_support_fraction: float
    centrality: float
    trim_left: int
    trim_right: int


@dataclass(frozen=True)
class RunWindowCoreSelection:
    report: dict[str, object]
    start_id: int
    stop_id: int
    score: float
    feature_table: dict[str, int | float]


def _to_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _layer_counts(report: dict[str, object]) -> list[int]:
    return [
        _to_int(layer.get("layer_count", 0))
        for layer in list(report.get("layer_details", []) or [])
        if isinstance(layer, dict) and bool(layer.get("usable", False))
    ]


def _longest_plateau(values: list[int], target: int) -> int:
    longest = 0
    current = 0
    for value in values:
        if value == target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _largest_adjacent_drop(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    return max(0, max(first - second for first, second in zip(values, values[1:], strict=False)))


def _raw_strand_ids_from_slices(
    slices_dict: dict[float, list[tuple[float, ...]]],
) -> list[int]:
    ids: set[int] = set()
    for points in slices_dict.values():
        for point in points:
            if len(point) >= 4:
                ids.add(int(point[3]))
    return sorted(ids)


def _layer_strand_ids(layer: dict[str, object]) -> list[int]:
    return sorted(int(value) for value in list(layer.get("strand_ids", []) or []))


def filter_slices_by_run_window(
    slices_dict: dict[float, list[tuple[float, ...]]],
    start_id: int,
    stop_id: int,
) -> dict[float, list[tuple[float, ...]]]:
    filtered: dict[float, list[tuple[float, ...]]] = {}
    for z_value, points in slices_dict.items():
        kept = [
            point
            for point in points
            if len(point) >= 4 and start_id <= int(point[3]) <= stop_id
        ]
        if kept:
            filtered[float(z_value)] = kept
    return filtered


def _layer_boundary_spans(
    report: dict[str, object],
    *,
    min_layer_count: int,
) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    for layer in list(report.get("layer_details", []) or []):
        if not isinstance(layer, dict):
            continue
        if not bool(layer.get("usable", False)):
            continue
        if _to_int(layer.get("layer_count", 0)) < min_layer_count:
            continue
        strand_ids = _layer_strand_ids(layer)
        if strand_ids:
            spans.add((min(strand_ids), max(strand_ids)))
    return spans


def slice_profile_features(
    report: dict[str, object],
    *,
    terminal_band_size: int,
) -> SliceProfileFeatures:
    counts = _layer_counts(report)
    layer_count_max = max(counts) if counts else 0
    layer_count_q90 = quantile_count(counts, 0.90)
    max_layer_ids: set[int] = set()
    for layer in list(report.get("layer_details", []) or []):
        if not isinstance(layer, dict):
            continue
        if bool(layer.get("usable", False)) and _to_int(layer.get("layer_count")) == layer_count_max:
            max_layer_ids.update(_layer_strand_ids(layer))

    raw_support = {
        int(key): _to_int(value)
        for key, value in dict(report.get("strand_support", {}) or {}).items()
    }
    total_support = int(sum(raw_support.values()))
    raw_ids = sorted(raw_support)
    terminal_ids: set[int] = set()
    if raw_ids and terminal_band_size > 0:
        min_id = min(raw_ids)
        max_id = max(raw_ids)
        terminal_ids = {
            strand_id
            for strand_id in raw_ids
            if strand_id < min_id + terminal_band_size
            or strand_id > max_id - terminal_band_size
        }

    terminal_support = int(sum(raw_support.get(strand_id, 0) for strand_id in terminal_ids))
    trajectory_support = {
        int(key): _to_int(value)
        for key, value in dict(report.get("trajectory_strand_support", {}) or {}).items()
    }
    trajectory_fragments = {
        int(key): [int(value) for value in values]
        for key, values in dict(report.get("trajectory_fragments", {}) or {}).items()
    }
    consensus_threshold = _to_int(report.get("support_consensus_threshold", 0))
    consensus_ids = [
        strand_id
        for strand_id, support in trajectory_support.items()
        if support >= consensus_threshold
    ]
    absent_from_max = 0
    terminal_consensus = 0
    for strand_id in consensus_ids:
        fragments = trajectory_fragments.get(strand_id, [strand_id])
        if not set(fragments).intersection(max_layer_ids):
            absent_from_max += 1
        if fragments and all(fragment in terminal_ids for fragment in fragments):
            terminal_consensus += 1

    return SliceProfileFeatures(
        layer_count_max=int(layer_count_max),
        layer_count_q90=int(layer_count_q90),
        max_plateau_length=_longest_plateau(counts, layer_count_max),
        largest_adjacent_drop=_largest_adjacent_drop(counts),
        absent_from_max=int(absent_from_max),
        terminal_support=int(terminal_support),
        terminal_consensus=int(terminal_consensus),
        total_support=int(total_support),
    )


def report_feature_table(
    report: dict[str, object],
    *,
    terminal_band_size: int,
) -> dict[str, int | float]:
    features = slice_profile_features(report, terminal_band_size=terminal_band_size)
    terminal_support_fraction = (
        features.terminal_support / features.total_support if features.total_support else 0.0
    )
    return {
        "layer_count_max_plateau": int(features.max_plateau_length),
        "layer_count_largest_drop": int(features.largest_adjacent_drop),
        "absent_from_max_strands": int(features.absent_from_max),
        "terminal_strand_support": int(features.terminal_support),
        "terminal_consensus_strands": int(features.terminal_consensus),
        "terminal_support_fraction": float(terminal_support_fraction),
    }


class RunWindowCoreSelector:
    """Select a contiguous raw run-id window and re-count only that slice core."""

    def __init__(self, sequence_core_config: Any):
        self.config = sequence_core_config

    def select(
        self,
        slices_dict: dict[float, list[tuple[float, ...]]],
        base_report: dict[str, object],
        analyze_window: AnalyzeWindow,
    ) -> RunWindowCoreSelection | None:
        cfg = self.config
        if not bool(getattr(cfg, "run_window_enabled", False)):
            return None

        raw_ids = _raw_strand_ids_from_slices(slices_dict)
        if not raw_ids:
            return None
        min_raw_id = min(raw_ids)
        max_raw_id = max(raw_ids)
        raw_span_count = max_raw_id - min_raw_id + 1
        if raw_span_count < int(getattr(cfg, "run_window_min_candidate_strands", 0)):
            return None

        terminal_band_size = int(getattr(cfg, "run_window_terminal_band_size", 4))
        base_features = slice_profile_features(
            base_report,
            terminal_band_size=terminal_band_size,
        )
        base_count = _to_int(base_report.get("strand_count", 0))
        base_layer_gap = max(0, base_count - base_features.layer_count_max)
        base_support_gap = max(
            0,
            _to_int(base_report.get("support_consensus_strands", 0))
            - base_features.layer_count_max,
        )
        if base_count <= 0:
            return None
        if base_features.layer_count_max < int(
            getattr(cfg, "run_window_min_base_layer_max", 0)
        ):
            return None
        if base_support_gap < int(getattr(cfg, "run_window_min_support_gap", 0)):
            return None

        quick_scores = self._quick_scores(
            slices_dict=slices_dict,
            raw_ids=raw_ids,
            base_report=base_report,
            base_features=base_features,
        )
        if not quick_scores:
            return None

        max_scored = int(getattr(cfg, "run_window_max_scored_windows", 128))
        quick_scores = sorted(quick_scores, key=lambda item: item.score, reverse=True)[:max_scored]

        best_selection: RunWindowCoreSelection | None = None
        best_score = float("-inf")
        reanalyzed_windows = 0
        for quick in quick_scores:
            filtered = filter_slices_by_run_window(slices_dict, quick.start_id, quick.stop_id)
            if not filtered:
                continue
            reanalyzed_windows += 1
            window_report = analyze_window(filtered)
            window_count = _to_int(window_report.get("strand_count", 0))
            if not self._eligible_count_change(
                base_count=base_count,
                base_layer_gap=base_layer_gap,
                base_features=base_features,
                window_report=window_report,
            ):
                continue

            window_features = slice_profile_features(
                window_report,
                terminal_band_size=terminal_band_size,
            )
            if not self._passes_layer_preservation(
                base_features,
                layer_count_max=window_features.layer_count_max,
                layer_count_q90=window_features.layer_count_q90,
            ):
                continue
            count_layer_improvement = max(
                0,
                base_layer_gap - max(0, window_count - window_features.layer_count_max),
            )
            layer_preservation = self._layer_preservation(
                base_features,
                window_features,
            )
            absent_improvement = (
                base_features.absent_from_max - window_features.absent_from_max
            ) / max(1.0, float(base_features.absent_from_max))
            drop_improvement = (
                base_features.largest_adjacent_drop - window_features.largest_adjacent_drop
            ) / max(1.0, float(base_features.largest_adjacent_drop))
            disappearance_score = self._weighted_mean(
                (
                    (
                        absent_improvement,
                        float(getattr(cfg, "run_window_score_absent_improvement_weight", 0.0)),
                    ),
                    (
                        drop_improvement,
                        float(getattr(cfg, "run_window_score_drop_improvement_weight", 0.0)),
                    ),
                )
            )
            count_margin = min(
                1.0,
                max(0.0, window_count - window_features.layer_count_max)
                / max(1.0, float(base_layer_gap)),
            )
            score = (
                float(getattr(cfg, "run_window_score_count_improvement_weight", 1.0))
                * (count_layer_improvement / max(1.0, float(base_layer_gap)))
                + float(getattr(cfg, "run_window_score_layer_preservation_weight", 1.0))
                * layer_preservation
                + float(getattr(cfg, "run_window_score_z_coverage_weight", 0.0))
                * quick.z_coverage_fraction
                + float(getattr(cfg, "run_window_score_centrality_weight", 0.0))
                * quick.centrality
                + float(getattr(cfg, "run_window_score_disappearance_weight", 0.0))
                * disappearance_score
                + float(getattr(cfg, "run_window_score_count_margin_weight", 0.0))
                * count_margin
                - float(getattr(cfg, "run_window_score_outside_support_weight", 0.0))
                * quick.outside_support_fraction
                - float(getattr(cfg, "run_window_score_length_penalty_weight", 0.0))
                * ((quick.trim_left + quick.trim_right) / max(1.0, float(raw_span_count)))
            )
            if score < float(getattr(cfg, "run_window_min_score", 0.0)):
                continue
            if score <= best_score:
                continue

            feature_table = {
                "run_window_core_score": float(score),
                "run_window_core_start": int(quick.start_id),
                "run_window_core_stop": int(quick.stop_id),
                "run_window_core_trim_left": int(quick.trim_left),
                "run_window_core_trim_right": int(quick.trim_right),
                "run_window_core_layer_max_preservation": float(
                    window_features.layer_count_max
                    / max(1.0, float(base_features.layer_count_max))
                ),
                "run_window_core_layer_q90_preservation": float(
                    window_features.layer_count_q90
                    / max(1.0, float(base_features.layer_count_q90))
                ),
                "run_window_core_outside_support_fraction": float(
                    quick.outside_support_fraction
                ),
                "run_window_core_z_coverage_fraction": float(quick.z_coverage_fraction),
                "run_window_core_max_z_gap": int(quick.max_z_gap),
                "run_window_core_centrality": float(quick.centrality),
                "run_window_core_count_layer_improvement": int(count_layer_improvement),
                "run_window_core_absent_improvement": float(absent_improvement),
                "run_window_core_drop_improvement": float(drop_improvement),
                "run_window_core_count_margin": float(count_margin),
            }
            best_score = score
            best_selection = RunWindowCoreSelection(
                report=window_report,
                start_id=quick.start_id,
                stop_id=quick.stop_id,
                score=float(score),
                feature_table=feature_table,
            )

        if best_selection is not None:
            feature_table = dict(best_selection.feature_table)
            feature_table["run_window_core_candidate_windows"] = int(len(quick_scores))
            feature_table["run_window_core_reanalyzed_windows"] = int(reanalyzed_windows)
            best_selection = RunWindowCoreSelection(
                report=best_selection.report,
                start_id=best_selection.start_id,
                stop_id=best_selection.stop_id,
                score=best_selection.score,
                feature_table=feature_table,
            )
        return best_selection

    def _eligible_count_change(
        self,
        *,
        base_count: int,
        base_layer_gap: int,
        base_features: SliceProfileFeatures,
        window_report: dict[str, object],
    ) -> bool:
        cfg = self.config
        window_count = _to_int(window_report.get("strand_count", 0))
        if window_count <= 0:
            return False
        if window_count >= base_count:
            return False

        count_delta = base_count - window_count
        if count_delta < int(getattr(cfg, "run_window_min_prediction_gap", 1)):
            return False
        min_count_over_base_layer = int(
            getattr(cfg, "run_window_min_count_over_base_layer_max", 0)
        )
        if base_features.layer_count_max < int(
            getattr(cfg, "run_window_low_layer_max_threshold", 0)
        ):
            min_count_over_base_layer = max(
                min_count_over_base_layer,
                int(getattr(cfg, "run_window_low_layer_max_min_count_over_base_layer_max", 0)),
            )
        if window_count - base_features.layer_count_max < min_count_over_base_layer:
            return False
        max_count_over_base_layer = int(
            getattr(cfg, "run_window_max_count_over_base_layer_max", 0)
        )
        if (
            max_count_over_base_layer > 0
            and window_count - base_features.layer_count_max > max_count_over_base_layer
        ):
            return False
        if window_count - _to_int(window_report.get("layer_count_max", 0)) < int(
            getattr(cfg, "run_window_min_count_over_layer_max", 0)
        ):
            return False
        count_layer_improvement = base_layer_gap - max(
            0,
            window_count - _to_int(window_report.get("layer_count_max", 0)),
        )
        if count_layer_improvement < int(
            getattr(cfg, "run_window_min_count_layer_improvement", 0)
        ):
            return False
        if base_features.absent_from_max < int(
            getattr(cfg, "run_window_min_absent_from_max", 0)
        ):
            return False
        return True

    def _quick_scores(
        self,
        *,
        slices_dict: dict[float, list[tuple[float, ...]]],
        raw_ids: list[int],
        base_report: dict[str, object],
        base_features: SliceProfileFeatures,
    ) -> list[WindowQuickScore]:
        cfg = self.config
        min_raw_id = min(raw_ids)
        max_raw_id = max(raw_ids)
        raw_span_count = max_raw_id - min_raw_id + 1
        min_trimmed = int(getattr(cfg, "run_window_min_trimmed_runs", 0))
        min_window_span = max(
            int(getattr(cfg, "run_window_min_raw_span", 0)),
            int(getattr(cfg, "run_window_min_candidate_strands", 0)) // 2,
        )
        max_window_span = int(getattr(cfg, "run_window_max_raw_span", 0)) or raw_span_count
        max_window_span = min(max_window_span, raw_span_count)
        raw_support = {
            int(key): _to_int(value)
            for key, value in dict(base_report.get("strand_support", {}) or {}).items()
        }
        total_support = sum(raw_support.values())
        z_values = sorted(float(value) for value in slices_dict)
        z_index = {z_value: index for index, z_value in enumerate(z_values)}
        raw_support_prefix = self._prefix_counts(
            [
                raw_support.get(strand_id, 0)
                for strand_id in range(min_raw_id, max_raw_id + 1)
            ]
        )
        layer_prefix_counts: dict[float, list[int]] = {}
        for z_value in z_values:
            layer_ids = self._layer_id_set(slices_dict[z_value])
            layer_prefix_counts[z_value] = self._prefix_counts(
                [
                    1 if strand_id in layer_ids else 0
                    for strand_id in range(min_raw_id, max_raw_id + 1)
                ]
            )
        quick_scores: list[WindowQuickScore] = []

        bounds = self._candidate_window_bounds(
            raw_ids=raw_ids,
            base_report=base_report,
            base_features=base_features,
        )
        if not bounds:
            bounds = {
                (start_id, stop_id)
                for start_id in range(min_raw_id, max_raw_id + 1)
                for stop_id in range(start_id, max_raw_id + 1)
            }

        for start_id, stop_id in sorted(bounds):
            window_span = stop_id - start_id + 1
            if window_span < min_window_span or window_span > max_window_span:
                continue
            trim_left = start_id - min_raw_id
            trim_right = max_raw_id - stop_id
            if trim_left + trim_right < min_trimmed:
                continue
            counts: list[int] = []
            active_indices: list[int] = []
            for z_value in z_values:
                count = self._prefix_window_sum(
                    layer_prefix_counts[z_value],
                    start_id=start_id,
                    stop_id=stop_id,
                    min_id=min_raw_id,
                )
                if count:
                    counts.append(count)
                    active_indices.append(z_index[z_value])
            if not counts:
                continue
            layer_count_max = max(counts)
            layer_count_q90 = quantile_count(counts, 0.90)
            if not self._passes_layer_preservation(
                base_features,
                layer_count_max=layer_count_max,
                layer_count_q90=layer_count_q90,
            ):
                continue
            z_coverage_fraction, max_z_gap = self._z_coverage(active_indices)
            if z_coverage_fraction < float(
                getattr(cfg, "run_window_min_z_coverage_fraction", 0.0)
            ):
                continue
            inside_support = self._prefix_window_sum(
                raw_support_prefix,
                start_id=start_id,
                stop_id=stop_id,
                min_id=min_raw_id,
            )
            outside_support = total_support - inside_support
            outside_support_fraction = outside_support / max(1.0, float(total_support))
            layer_preservation = self._layer_preservation_values(
                base_features,
                layer_count_max=layer_count_max,
                layer_count_q90=layer_count_q90,
            )
            centrality = 1.0 - abs(
                ((start_id + stop_id) / 2.0) - ((min_raw_id + max_raw_id) / 2.0)
            ) / max(1.0, float(raw_span_count) / 2.0)
            trimmed_fraction = (trim_left + trim_right) / max(1.0, float(raw_span_count))
            quick_score = (
                float(getattr(cfg, "run_window_score_layer_preservation_weight", 1.0))
                * layer_preservation
                + float(getattr(cfg, "run_window_score_z_coverage_weight", 0.0))
                * z_coverage_fraction
                + float(getattr(cfg, "run_window_score_centrality_weight", 0.0))
                * centrality
                - float(getattr(cfg, "run_window_score_outside_support_weight", 0.0))
                * outside_support_fraction
                - float(getattr(cfg, "run_window_score_length_penalty_weight", 0.0))
                * trimmed_fraction
            )
            quick_scores.append(
                WindowQuickScore(
                    start_id=int(start_id),
                    stop_id=int(stop_id),
                    score=float(quick_score),
                    layer_count_q90=int(layer_count_q90),
                    layer_count_max=int(layer_count_max),
                    z_coverage_fraction=float(z_coverage_fraction),
                    max_z_gap=int(max_z_gap),
                    outside_support_fraction=float(outside_support_fraction),
                    centrality=float(centrality),
                    trim_left=int(trim_left),
                    trim_right=int(trim_right),
                )
            )
        return quick_scores

    @staticmethod
    def _layer_id_set(points: list[tuple[float, ...]]) -> set[int]:
        return {int(point[3]) for point in points if len(point) >= 4}

    @staticmethod
    def _prefix_counts(values: list[int]) -> list[int]:
        prefix = [0]
        running = 0
        for value in values:
            running += int(value)
            prefix.append(running)
        return prefix

    @staticmethod
    def _prefix_window_sum(
        prefix: list[int],
        *,
        start_id: int,
        stop_id: int,
        min_id: int,
    ) -> int:
        start_index = max(0, int(start_id) - int(min_id))
        stop_index = min(len(prefix) - 1, int(stop_id) - int(min_id) + 1)
        if stop_index <= start_index:
            return 0
        return int(prefix[stop_index] - prefix[start_index])

    def _candidate_window_bounds(
        self,
        *,
        raw_ids: list[int],
        base_report: dict[str, object],
        base_features: SliceProfileFeatures,
    ) -> set[tuple[int, int]]:
        cfg = self.config
        if not bool(getattr(cfg, "run_window_boundary_candidates_enabled", True)):
            return set()
        if not raw_ids:
            return set()

        min_raw_id = min(raw_ids)
        max_raw_id = max(raw_ids)
        layer_spans = _layer_boundary_spans(
            base_report,
            min_layer_count=max(1, base_features.layer_count_q90),
        )
        layer_spans.update(
            _layer_boundary_spans(
                base_report,
                min_layer_count=max(1, base_features.layer_count_max),
            )
        )
        if not layer_spans:
            return set()

        step = max(1, int(getattr(cfg, "run_window_boundary_padding_step", 1)))
        max_padding = max(0, int(getattr(cfg, "run_window_boundary_max_padding", 0)))
        paddings = list(range(0, max_padding + 1, step))
        if max_padding not in paddings:
            paddings.append(max_padding)

        bounds: set[tuple[int, int]] = set()
        for core_start, core_stop in layer_spans:
            for left_padding in paddings:
                start_id = max(min_raw_id, core_start - left_padding)
                for right_padding in paddings:
                    stop_id = min(max_raw_id, core_stop + right_padding)
                    if start_id <= stop_id:
                        bounds.add((int(start_id), int(stop_id)))
        return bounds

    def _passes_layer_preservation(
        self,
        base_features: SliceProfileFeatures,
        *,
        layer_count_max: int,
        layer_count_q90: int,
    ) -> bool:
        cfg = self.config
        min_layer_max = max(
            int(getattr(cfg, "run_window_min_layer_max_floor", 0)),
            int(round(base_features.layer_count_max * float(
                getattr(cfg, "run_window_min_layer_max_fraction", 0.0)
            ))),
        )
        min_layer_q90 = max(
            int(getattr(cfg, "run_window_min_layer_q90_floor", 0)),
            int(round(base_features.layer_count_q90 * float(
                getattr(cfg, "run_window_min_layer_q90_fraction", 0.0)
            ))),
        )
        return layer_count_max >= min_layer_max and layer_count_q90 >= min_layer_q90

    @staticmethod
    def _layer_preservation(
        base_features: SliceProfileFeatures,
        window_features: SliceProfileFeatures,
    ) -> float:
        return RunWindowCoreSelector._layer_preservation_values(
            base_features,
            layer_count_max=window_features.layer_count_max,
            layer_count_q90=window_features.layer_count_q90,
        )

    @staticmethod
    def _layer_preservation_values(
        base_features: SliceProfileFeatures,
        *,
        layer_count_max: int,
        layer_count_q90: int,
    ) -> float:
        return 0.5 * (
            min(1.0, layer_count_max / max(1.0, float(base_features.layer_count_max)))
            + min(1.0, layer_count_q90 / max(1.0, float(base_features.layer_count_q90)))
        )

    @staticmethod
    def _weighted_mean(values: tuple[tuple[float, float], ...]) -> float:
        weight_sum = sum(max(0.0, float(weight)) for _value, weight in values)
        if weight_sum <= 0.0:
            return 0.0
        return float(
            sum(float(value) * max(0.0, float(weight)) for value, weight in values)
            / weight_sum
        )

    @staticmethod
    def _z_coverage(active_indices: list[int]) -> tuple[float, int]:
        if not active_indices:
            return 0.0, 0
        sorted_indices = sorted(set(active_indices))
        span = sorted_indices[-1] - sorted_indices[0] + 1
        if span <= 0:
            return 0.0, 0
        max_gap = 0
        for first, second in zip(sorted_indices, sorted_indices[1:], strict=False):
            max_gap = max(max_gap, second - first - 1)
        return len(sorted_indices) / span, int(max_gap)
