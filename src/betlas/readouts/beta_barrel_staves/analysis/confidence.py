from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .numeric import clamp_ratio, estimate_similarity


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    selected_support_mean: float
    selected_support_min: float
    layer_agreement: float


class ConfidenceEstimator:
    """Estimate report confidence from slice-derived agreement features."""

    def __init__(self, confidence_config: Any):
        self.config = confidence_config

    def selected_support_summary(
        self,
        selected_ids: list[int],
        trajectory_support: dict[int, int],
        usable_layer_count: int,
    ) -> tuple[float, float]:
        if not selected_ids or usable_layer_count <= 0:
            return 0.0, 0.0
        support_fractions = [
            trajectory_support.get(strand_id, 0) / usable_layer_count
            for strand_id in selected_ids
        ]
        return float(np.mean(support_fractions)), float(np.min(support_fractions))

    def layer_count_agreement(self, layer_counts: list[int], strand_count: int) -> float:
        if not layer_counts:
            return 0.0
        cfg = self.config
        tolerance = max(
            float(cfg.layer_agreement_min_tolerance),
            float(strand_count) * float(cfg.layer_agreement_count_fraction),
        )
        return float(
            np.mean(
                [
                    max(0.0, 1.0 - abs(layer_count - strand_count) / tolerance)
                    for layer_count in layer_counts
                ]
            )
        )

    def supporting_layer_count(
        self,
        *,
        layer_counts: list[int],
        strand_count: int,
        selected_ids: list[int],
        trajectory_fragments: dict[int, list[int]],
        usable_layers: list[dict[str, object]],
        min_layer_coverage_fraction: float,
    ) -> int:
        if selected_ids and strand_count > 0:
            return sum(
                1
                for layer in usable_layers
                if self._selected_layer_coverage(
                    layer,
                    selected_ids,
                    trajectory_fragments,
                    strand_count,
                )
                >= min_layer_coverage_fraction
            )
        tolerance = int(self.config.layer_count_support_tolerance)
        return sum(1 for layer_count in layer_counts if abs(layer_count - strand_count) <= tolerance)

    def estimate(
        self,
        *,
        strand_count: int,
        selected_ids: list[int],
        trajectory_support: dict[int, int],
        usable_layer_count: int,
        layer_counts: list[int],
        support_consensus_count: int,
        trajectory_candidate_count: int,
        layer_count_q90: int,
        count_decision_score: float,
        layer_support_fraction: float,
        usable_layer_fraction: float,
        geometric_pass_fraction: float,
        radial_outlier_layers: int,
        trimmed_layers: int,
    ) -> ConfidenceResult:
        if strand_count <= 0 or usable_layer_count <= 0:
            return ConfidenceResult(0.0, 0.0, 0.0, 0.0)

        cfg = self.config
        selected_mean, selected_min = self.selected_support_summary(
            selected_ids,
            trajectory_support,
            usable_layer_count,
        )
        layer_agreement = self.layer_count_agreement(layer_counts, strand_count)
        decision_agreement = clamp_ratio(
            count_decision_score / float(cfg.decision_score_normalizer)
        )
        feature_agreement = clamp_ratio(
            float(cfg.feature_support_consensus_weight)
            * estimate_similarity(
                strand_count,
                support_consensus_count,
                scale_fraction=float(cfg.feature_support_consensus_scale),
            )
            + float(cfg.feature_trajectory_candidate_weight)
            * estimate_similarity(
                strand_count,
                trajectory_candidate_count,
                scale_fraction=float(cfg.feature_trajectory_candidate_scale),
            )
            + float(cfg.feature_layer_q90_weight)
            * estimate_similarity(
                strand_count,
                layer_count_q90,
                scale_fraction=float(cfg.feature_layer_q90_scale),
            )
        )
        outlier_penalty = min(
            float(cfg.max_outlier_penalty),
            float(cfg.radial_outlier_layer_penalty) * radial_outlier_layers
            + float(cfg.trimmed_layer_penalty) * trimmed_layers,
        )
        confidence = (
            float(cfg.decision_agreement_weight) * decision_agreement
            + float(cfg.feature_agreement_weight) * feature_agreement
            + float(cfg.selected_support_weight) * selected_mean
            + float(cfg.layer_support_fraction_weight) * layer_support_fraction
            + float(cfg.layer_agreement_weight) * layer_agreement
            + float(cfg.usable_layer_fraction_weight) * usable_layer_fraction
            + float(cfg.geometric_pass_fraction_weight) * geometric_pass_fraction
            - outlier_penalty
        )
        return ConfidenceResult(
            confidence=clamp_ratio(float(confidence)),
            selected_support_mean=float(selected_mean),
            selected_support_min=float(selected_min),
            layer_agreement=float(layer_agreement),
        )

    @staticmethod
    def _selected_layer_coverage(
        layer: dict[str, object],
        selected_ids: list[int],
        trajectory_fragments: dict[int, list[int]],
        strand_count: int,
    ) -> float:
        raw_ids = set(layer.get("strand_ids", []) or [])
        return (
            sum(
                1
                for selected_id in selected_ids
                if set(trajectory_fragments.get(selected_id, [selected_id])).intersection(raw_ids)
            )
            / strand_count
        )
