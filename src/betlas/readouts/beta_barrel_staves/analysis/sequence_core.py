from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .numeric import quantile_count


@dataclass(frozen=True)
class GlobalSequenceCoreWindow:
    strand_count: int
    score: float
    selected_ids: list[int]
    trim_left: int
    trim_right: int
    layer_count_q90: int
    layer_count_max: int
    support_fraction: float
    centrality: float


class GlobalSequenceCoreSelector:
    """Select a contiguous barrel-core trajectory window from slice evidence."""

    def __init__(self, sequence_core_config: Any):
        self.config = sequence_core_config

    def select(
        self,
        *,
        trajectory_support: dict[int, int],
        trajectory_fragments: dict[int, list[int]],
        support_consensus_ids: list[int],
        usable_layers: list[dict[str, object]],
        current_count: int,
        layer_count_q90: int,
        layer_count_max: int,
        usable_layer_count: int,
    ) -> GlobalSequenceCoreWindow | None:
        core_cfg = self.config
        if not bool(core_cfg.global_enabled):
            return None
        if not trajectory_support or not trajectory_fragments or not usable_layers:
            return None

        trajectory_ids = sorted(int(strand_id) for strand_id in trajectory_support)
        trajectory_count = len(trajectory_ids)
        if trajectory_count < int(core_cfg.global_min_trajectory_strands):
            return None
        if layer_count_max < int(core_cfg.global_min_layer_count_max):
            return None
        if len(support_consensus_ids) - layer_count_max < int(core_cfg.global_min_support_gap):
            return None
        if current_count - layer_count_max < int(core_cfg.global_min_prediction_gap):
            return None

        max_window_count = min(
            trajectory_count,
            layer_count_max + int(core_cfg.global_layer_count_allowance),
        )
        min_window_count = min(layer_count_max, max_window_count)
        if min_window_count <= 0 or max_window_count <= 0:
            return None
        if trajectory_count - min_window_count < int(core_cfg.global_min_trimmed_trajectories):
            return None

        raw_fragments = {
            strand_id: set(int(value) for value in trajectory_fragments.get(strand_id, [strand_id]))
            for strand_id in trajectory_ids
        }
        layer_presence: list[list[bool]] = []
        for layer in usable_layers:
            raw_ids = set(int(value) for value in list(layer.get("strand_ids", []) or []))
            if not raw_ids:
                continue
            layer_presence.append(
                [bool(raw_ids.intersection(raw_fragments[strand_id])) for strand_id in trajectory_ids]
            )
        if not layer_presence:
            return None

        support_values = [int(trajectory_support[strand_id]) for strand_id in trajectory_ids]
        total_support = float(sum(support_values))
        best_window: GlobalSequenceCoreWindow | None = None
        best_score = float("-inf")

        min_centrality = float(core_cfg.global_min_centrality)
        min_support_fraction = float(core_cfg.global_min_window_support_fraction)
        min_layer_max_fraction = float(core_cfg.global_min_window_layer_max_fraction)
        min_layer_q90_fraction = float(core_cfg.global_min_window_q90_fraction)
        for window_count in range(min_window_count, max_window_count + 1):
            if current_count - window_count < int(core_cfg.global_min_prediction_gap):
                continue
            for start_index in range(0, trajectory_count - window_count + 1):
                stop_index = start_index + window_count
                selected_ids = trajectory_ids[start_index:stop_index]
                window_support = float(sum(support_values[start_index:stop_index]))
                support_fraction = window_support / max(1.0, float(window_count * usable_layer_count))
                if support_fraction < min_support_fraction:
                    continue

                midpoint = (float(start_index) + float(stop_index - 1)) / 2.0
                sequence_center = float(trajectory_count - 1) / 2.0
                centrality = 1.0 - abs(midpoint - sequence_center) / max(1.0, sequence_center)
                if centrality < min_centrality:
                    continue

                window_layer_counts = [
                    sum(1 for present in row[start_index:stop_index] if present)
                    for row in layer_presence
                ]
                window_q90 = quantile_count(window_layer_counts, 0.90)
                window_max = max(window_layer_counts) if window_layer_counts else 0
                if window_max < max(
                    int(core_cfg.global_min_window_layer_max_floor),
                    int(round(layer_count_max * min_layer_max_fraction)),
                ):
                    continue
                if window_q90 < max(
                    int(core_cfg.global_min_window_q90_floor),
                    int(round(layer_count_q90 * min_layer_q90_fraction)),
                ):
                    continue

                outside_count = trajectory_count - window_count
                outside_support_fraction = 0.0
                if outside_count > 0:
                    outside_support_fraction = (total_support - window_support) / max(
                        1.0,
                        float(outside_count * usable_layer_count),
                    )
                layer_preservation = 0.5 * (
                    min(1.0, window_q90 / max(1.0, float(layer_count_q90)))
                    + min(1.0, window_max / max(1.0, float(layer_count_max)))
                )
                trimmed_fraction = outside_count / max(1.0, float(trajectory_count))
                length_penalty = max(0, window_count - layer_count_max)
                score = (
                    float(core_cfg.global_score_support_weight) * support_fraction
                    - float(core_cfg.global_score_outside_support_weight) * outside_support_fraction
                    + float(core_cfg.global_score_centrality_weight) * centrality
                    + float(core_cfg.global_score_layer_preservation_weight) * layer_preservation
                    + float(core_cfg.global_score_trimmed_fraction_weight) * trimmed_fraction
                    - float(core_cfg.global_score_length_penalty_weight) * length_penalty
                )

                if score > best_score:
                    best_score = float(score)
                    best_window = GlobalSequenceCoreWindow(
                        strand_count=int(window_count),
                        score=float(score),
                        selected_ids=selected_ids,
                        trim_left=int(start_index),
                        trim_right=int(trajectory_count - stop_index),
                        layer_count_q90=int(window_q90),
                        layer_count_max=int(window_max),
                        support_fraction=float(support_fraction),
                        centrality=float(centrality),
                    )

        return best_window
