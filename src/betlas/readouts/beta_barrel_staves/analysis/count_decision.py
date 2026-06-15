from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .numeric import estimate_similarity


@dataclass(frozen=True)
class CountDecision:
    strand_count: int
    confidence_basis: str
    decision_score: float
    selected_ids: list[int]

    def to_report(self) -> dict[str, object]:
        return {
            "strand_count": int(self.strand_count),
            "confidence_basis": self.confidence_basis,
            "decision_score": float(self.decision_score),
            "selected_ids": [int(value) for value in self.selected_ids],
        }


class CountDecisionEngine:
    """Select a strand count from slice-derived support and layer-count evidence."""

    def __init__(self, count_config: Any):
        self.config = count_config

    def _score_count_candidate(
        self,
        count: int,
        *,
        supports_desc: list[int],
        usable_layer_count: int,
        consensus_threshold: int,
        support_consensus_strands: int,
        persistent_strands: int,
        candidate_strands: int,
        layer_counts: list[int],
        layer_count_q75: int,
        layer_count_q90: int,
        layer_count_max: int,
    ) -> float:
        cfg = self.config
        if count <= 0:
            return -1.0

        score = 0.0
        score += float(cfg.score_support_consensus_weight) * estimate_similarity(
            count,
            support_consensus_strands,
            scale_fraction=float(cfg.score_support_consensus_scale),
        )
        score += float(cfg.score_layer_q90_weight) * estimate_similarity(
            count,
            layer_count_q90,
            scale_fraction=float(cfg.score_layer_q90_scale),
        )
        score += float(cfg.score_layer_q75_weight) * estimate_similarity(
            count,
            layer_count_q75,
            scale_fraction=float(cfg.score_layer_q75_scale),
        )
        score += float(cfg.score_layer_max_weight) * estimate_similarity(
            count,
            layer_count_max,
            scale_fraction=float(cfg.score_layer_max_scale),
        )
        score += float(cfg.score_candidate_weight) * estimate_similarity(
            count,
            candidate_strands,
            scale_fraction=float(cfg.score_candidate_scale),
        )
        score += float(cfg.score_persistent_weight) * estimate_similarity(
            count,
            persistent_strands,
            scale_fraction=float(cfg.score_persistent_scale),
        )
        if support_consensus_strands and candidate_strands:
            score += float(cfg.score_joint_support_candidate_weight) * (
                estimate_similarity(
                    count,
                    support_consensus_strands,
                    scale_fraction=float(cfg.score_joint_support_scale),
                )
                * estimate_similarity(
                    count,
                    candidate_strands,
                    scale_fraction=float(cfg.score_joint_candidate_scale),
                )
            )
            if support_consensus_strands == candidate_strands and count == support_consensus_strands:
                score += float(cfg.score_exact_support_candidate_bonus)

        if layer_counts:
            tolerance = max(
                float(cfg.score_layer_agreement_min_tolerance),
                float(count) * float(cfg.score_layer_agreement_count_fraction),
            )
            layer_agreement = float(
                np.mean([max(0.0, 1.0 - abs(layer_count - count) / tolerance) for layer_count in layer_counts])
            )
            score += float(cfg.score_layer_agreement_weight) * layer_agreement

        if supports_desc and usable_layer_count > 0:
            if count <= len(supports_desc):
                selected = np.asarray(supports_desc[:count], dtype=float) / usable_layer_count
                score += float(cfg.score_selected_mean_support_weight) * float(np.mean(selected))
                score += float(cfg.score_selected_min_support_weight) * float(np.min(selected))
            if count < len(supports_desc):
                next_support = supports_desc[count]
                if next_support >= consensus_threshold:
                    score -= float(cfg.score_next_consensus_penalty)
                score -= float(cfg.score_next_support_penalty) * min(
                    1.0,
                    next_support / max(1, usable_layer_count),
                )

        if candidate_strands and count > candidate_strands:
            score -= float(cfg.score_over_candidate_base_penalty)
            score -= float(cfg.score_over_candidate_per_count_penalty) * (count - candidate_strands)
        if persistent_strands and count < persistent_strands:
            score -= float(cfg.score_under_persistent_per_count_penalty) * (
                persistent_strands - count
            )
        if support_consensus_strands and count < support_consensus_strands - 2:
            score -= min(
                float(cfg.score_under_support_gap_max_penalty),
                float(cfg.score_under_support_gap_per_count_penalty)
                * (support_consensus_strands - count - 2),
            )
        if (
            support_consensus_strands
            and count > support_consensus_strands + int(cfg.score_over_support_gap_min_delta)
            and layer_count_q90 < count - 2
        ):
            score -= float(cfg.score_over_support_gap_per_count_penalty) * (
                count - support_consensus_strands - int(cfg.score_over_support_gap_min_delta)
            )

        return float(score)

    def select(
        self,
        *,
        strand_support: dict[int, int],
        support_consensus_ids: list[int],
        persistent_ids: list[int],
        consensus_threshold: int,
        layer_counts: list[int],
        layer_count_mode: int,
        layer_count_q75: int,
        layer_count_q90: int,
        layer_count_max: int,
        usable_layer_count: int,
    ) -> CountDecision:
        cfg = self.config
        min_count = int(cfg.min_count)
        max_count = int(cfg.max_count)
        candidate_strands = len(strand_support)
        support_consensus_strands = len(support_consensus_ids)
        persistent_strands = len(persistent_ids)

        baseline_count = layer_count_mode or support_consensus_strands or persistent_strands
        if candidate_strands <= 0 and baseline_count <= 0:
            return CountDecision(0, "no_strand_support", 0.0, [])

        upper_bound = max(candidate_strands, layer_count_q90, layer_count_max, baseline_count)
        upper_bound = max(min_count, min(max_count, upper_bound))
        supports_desc = sorted(strand_support.values(), reverse=True)

        scores = {
            count: self._score_count_candidate(
                count,
                supports_desc=supports_desc,
                usable_layer_count=usable_layer_count,
                consensus_threshold=consensus_threshold,
                support_consensus_strands=support_consensus_strands,
                persistent_strands=persistent_strands,
                candidate_strands=candidate_strands,
                layer_counts=layer_counts,
                layer_count_q75=layer_count_q75,
                layer_count_q90=layer_count_q90,
                layer_count_max=layer_count_max,
            )
            for count in range(min_count, upper_bound + 1)
        }
        best_count, best_score = max(scores.items(), key=lambda item: (item[1], -abs(item[0] - layer_count_q90)))
        basis = "support_layer_consensus"

        selected_ids = [
            strand_id
            for strand_id, _support in sorted(
                strand_support.items(),
                key=lambda item: (-item[1], item[0]),
            )[:best_count]
        ]
        return CountDecision(int(best_count), basis, float(best_score), selected_ids)
