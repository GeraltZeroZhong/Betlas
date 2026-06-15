from __future__ import annotations

from dataclasses import replace
from typing import Any

from .curve import (
    candidates_by_variant,
    plateau_edges,
    same_edge_variant_agreement,
)
from .energy import MembershipScore, score_membership_candidate
from .types import GraphCoreCandidate


def select_sparse_wall_membership_candidate(
    *,
    raw_candidates: list[GraphCoreCandidate],
    outer_candidates: list[GraphCoreCandidate],
    outer_gap_candidates: list[GraphCoreCandidate],
    support_consensus: int,
    layer_q90: int,
    config: Any,
) -> GraphCoreCandidate | None:
    """
    Select a sparse-visibility barrel-wall hypothesis from slice adjacency.

    This is a model selection step, not a post-hoc correction. Each candidate is a
    concrete 2-core component observed in the same slice-neighbor graph. The
    score prefers a near-cycle topology, stable edge-threshold plateaus,
    agreement across raw/outer/radial-gap variants, and compatibility with
    layer/support evidence.
    """
    if not bool(getattr(config, "sparse_membership_enabled", False)):
        return None
    all_candidates = [
        candidate
        for candidate in raw_candidates + outer_candidates + outer_gap_candidates
        if candidate.core_count > 0
    ]
    if not all_candidates:
        return None

    min_count = int(getattr(config, "sparse_membership_min_selected_count", 18))
    max_count = int(getattr(config, "sparse_membership_max_selected_count", 64))
    min_edge = int(getattr(config, "sparse_membership_min_edge_min_count", 2))
    max_avg_degree = float(getattr(config, "sparse_membership_max_component_avg_degree", 4.0))
    same_edge_avg_degree_limit = float(
        getattr(config, "sparse_membership_same_edge_max_component_avg_degree", max_avg_degree)
    )
    same_edge_min_variants_for_degree_relaxation = int(
        getattr(config, "sparse_membership_same_edge_min_variants_for_degree_relaxation", 2)
    )
    max_selected_over_support = int(
        getattr(config, "sparse_membership_max_selected_over_support", 2)
    )
    max_reduction_without_plateau = int(
        getattr(config, "sparse_membership_max_support_reduction_without_plateau", 10)
    )
    min_same_edge_for_selection = int(
        getattr(config, "sparse_membership_min_same_edge_variants_for_selection", 3)
    )
    min_plateau_for_selection = int(
        getattr(config, "sparse_membership_min_plateau_for_selection", 2)
    )
    max_layer_q90 = int(getattr(config, "sparse_membership_max_layer_q90", 0))
    if max_layer_q90 > 0 and int(layer_q90) > max_layer_q90:
        return None

    by_variant = candidates_by_variant(all_candidates)
    same_edge_tolerance = int(getattr(config, "sparse_membership_same_edge_tolerance", 0))
    plateau_tolerance = int(getattr(config, "sparse_membership_plateau_tolerance", 0))
    eligible = []
    for candidate in all_candidates:
        if not min_count <= int(candidate.core_count) <= max_count:
            continue
        if int(candidate.edge_min_count) < min_edge:
            continue
        if int(candidate.core_count) > int(support_consensus) + max_selected_over_support:
            continue

        same_edge_agreement = same_edge_variant_agreement(
            candidate,
            all_candidates,
            tolerance=same_edge_tolerance,
        )
        avg_degree_limit = (
            same_edge_avg_degree_limit
            if same_edge_agreement >= same_edge_min_variants_for_degree_relaxation
            else max_avg_degree
        )
        if float(candidate.component_avg_degree) > avg_degree_limit:
            continue

        plateau_count = plateau_edges(
            candidate,
            by_variant,
            tolerance=plateau_tolerance,
        )
        strong_same_edge_agreement = same_edge_agreement >= min_same_edge_for_selection
        strong_plateau = plateau_count >= min_plateau_for_selection
        if not strong_same_edge_agreement and not strong_plateau:
            continue

        support_reduction = int(support_consensus) - int(candidate.core_count)
        if (
            support_reduction > max_reduction_without_plateau
            and plateau_count
            < int(getattr(config, "sparse_membership_min_plateau_for_large_reduction", 2))
        ):
            continue
        eligible.append(candidate)
    if not eligible:
        return None

    scored: list[tuple[MembershipScore, GraphCoreCandidate]] = []
    for candidate in eligible:
        membership = score_membership_candidate(
            candidate,
            all_candidates=all_candidates,
            by_variant=by_variant,
            support_consensus=int(support_consensus),
            layer_q90=int(layer_q90),
            config=config,
        )
        if membership.score < float(getattr(config, "sparse_membership_min_score", 0.0)):
            continue
        scored.append((membership, candidate))
    if not scored:
        return None

    membership, candidate = max(
        scored,
        key=lambda item: (
            item[0].score,
            item[0].variant_agreement,
            item[0].plateau_edges,
            item[1].component_degree2_fraction,
            -item[1].component_branch_fraction,
            item[1].edge_min_count,
            -abs(float(item[1].component_avg_degree) - 2.0),
            item[1].graph_variant == "outer",
            item[1].graph_variant == "outer_gap",
            item[1].core_count,
        ),
    )
    return replace(
        candidate,
        selection_strategy="sparse_wall_membership",
        membership_score=float(membership.score),
        membership_block_count=int(membership.block_count),
        membership_largest_block_fraction=float(membership.largest_block_fraction),
        membership_plateau_edges=int(membership.plateau_edges),
        membership_next_drop_fraction=float(membership.next_drop_fraction),
        membership_variant_agreement=int(membership.variant_agreement),
    )
