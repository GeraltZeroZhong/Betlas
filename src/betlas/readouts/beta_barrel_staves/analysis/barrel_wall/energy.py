from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from .curve import (
    largest_block_fraction,
    next_drop_fraction,
    plateau_edges,
    same_edge_variant_agreement,
    variant_agreement,
)
from .types import GraphCoreCandidate


@dataclass(frozen=True)
class MembershipScore:
    score: float
    block_count: int
    largest_block_fraction: float
    plateau_edges: int
    next_drop_fraction: float
    variant_agreement: int


def _near_value_score(value: float, target: float, *, scale: float) -> float:
    return exp(-abs(float(value) - float(target)) / max(1e-9, float(scale)))


def _bounded_fraction(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def score_membership_candidate(
    candidate: GraphCoreCandidate,
    *,
    all_candidates: list[GraphCoreCandidate],
    by_variant: dict[str, list[GraphCoreCandidate]],
    support_consensus: int,
    layer_q90: int,
    config: Any,
) -> MembershipScore:
    block_count, block_largest_fraction = largest_block_fraction(candidate.selected_ids)
    agreement_tolerance = int(
        getattr(config, "sparse_membership_variant_agreement_tolerance", 2)
    )
    plateau_tolerance = int(getattr(config, "sparse_membership_plateau_tolerance", 0))
    plateau_count = plateau_edges(
        candidate,
        by_variant,
        tolerance=plateau_tolerance,
    )
    drop_fraction = next_drop_fraction(candidate, by_variant)
    variant_count = variant_agreement(
        candidate,
        all_candidates,
        tolerance=agreement_tolerance,
    )
    same_edge_count = same_edge_variant_agreement(
        candidate,
        all_candidates,
        tolerance=int(getattr(config, "sparse_membership_same_edge_tolerance", 0)),
    )

    support_fit = 0.0
    if support_consensus > 0 and candidate.core_count > 0:
        support_fit = min(
            float(candidate.core_count) / float(support_consensus),
            float(support_consensus) / float(candidate.core_count),
        )
    layer_lift = 0.0
    if layer_q90 > 0 and candidate.core_count > 0:
        layer_lift = max(0.0, float(candidate.core_count - layer_q90)) / float(
            candidate.core_count
        )

    max_edge = max((int(item.edge_min_count) for item in all_candidates), default=1)
    edge_score = float(candidate.edge_min_count) / max(1.0, float(max_edge))
    block_fragmentation = float(block_count) / max(1.0, float(candidate.core_count))
    variant_fraction = float(variant_count) / max(
        1.0,
        float(len({item.graph_variant for item in all_candidates})),
    )
    same_edge_fraction = float(same_edge_count) / max(
        1.0,
        float(len({item.graph_variant for item in all_candidates})),
    )
    plateau_score = min(
        1.0,
        float(plateau_count)
        / max(1.0, float(getattr(config, "sparse_membership_plateau_scale", 2.0))),
    )
    drop_score = min(
        1.0,
        drop_fraction
        / max(1e-9, float(getattr(config, "sparse_membership_drop_scale", 0.30))),
    )

    degree_score = _near_value_score(
        candidate.component_avg_degree,
        2.0,
        scale=float(getattr(config, "sparse_membership_degree_scale", 0.90)),
    )
    cycle_rank_score = _near_value_score(
        candidate.component_cycle_rank,
        1.0,
        scale=float(getattr(config, "sparse_membership_cycle_rank_scale", 2.0)),
    )
    degree2_fraction = _bounded_fraction(candidate.component_degree2_fraction)
    branch_fraction = _bounded_fraction(candidate.component_branch_fraction)
    radial_rank_score = _bounded_fraction(candidate.component_radial_rank_mean)

    score = (
        float(getattr(config, "sparse_membership_degree_weight", 1.00)) * degree_score
        + float(getattr(config, "sparse_membership_cycle_rank_weight", 0.0))
        * cycle_rank_score
        + float(getattr(config, "sparse_membership_degree2_fraction_weight", 0.0))
        * degree2_fraction
        + float(getattr(config, "sparse_membership_drop_weight", 0.75)) * drop_score
        + float(getattr(config, "sparse_membership_plateau_weight", 0.65))
        * plateau_score
        + float(getattr(config, "sparse_membership_variant_agreement_weight", 0.45))
        * variant_fraction
        + float(getattr(config, "sparse_membership_same_edge_agreement_weight", 0.55))
        * same_edge_fraction
        + float(getattr(config, "sparse_membership_support_fit_weight", 0.20))
        * support_fit
        + float(getattr(config, "sparse_membership_layer_lift_weight", 0.20))
        * layer_lift
        + float(getattr(config, "sparse_membership_edge_weight", 0.10)) * edge_score
        + float(getattr(config, "sparse_membership_radial_rank_weight", 0.0))
        * radial_rank_score
        - float(getattr(config, "sparse_membership_branch_penalty_weight", 0.0))
        * branch_fraction
        - float(getattr(config, "sparse_membership_block_penalty_weight", 0.35))
        * block_fragmentation
    )
    if candidate.graph_variant == "outer":
        score += float(getattr(config, "sparse_membership_outer_bonus", 0.08))
    elif candidate.graph_variant == "outer_gap":
        score += float(getattr(config, "sparse_membership_outer_gap_bonus", 0.04))

    return MembershipScore(
        score=float(score),
        block_count=int(block_count),
        largest_block_fraction=float(block_largest_fraction),
        plateau_edges=int(plateau_count),
        next_drop_fraction=float(drop_fraction),
        variant_agreement=int(variant_count),
    )
