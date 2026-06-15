from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .types import GraphCoreCandidate, to_int


def complete_layer_cohort_candidate(
    config: Any,
    base_report: dict[str, object],
    *,
    layer_q90: int,
) -> GraphCoreCandidate | None:
    if not bool(getattr(config, "complete_layer_cohort_enabled", True)):
        return None

    cohort_counts: Counter[tuple[int, ...]] = Counter()
    eligible_layers = 0
    for layer in list(base_report.get("layer_details", []) or []):
        if not isinstance(layer, dict) or not bool(layer.get("usable", False)):
            continue
        if to_int(layer.get("layer_count", 0)) != layer_q90:
            continue
        strand_ids = tuple(
            sorted(int(value) for value in list(layer.get("strand_ids", []) or []))
        )
        if len(strand_ids) != layer_q90:
            continue
        eligible_layers += 1
        cohort_counts[strand_ids] += 1

    if not cohort_counts or eligible_layers <= 0:
        return None

    selected_ids, selected_layers = cohort_counts.most_common(1)[0]
    min_layers = int(getattr(config, "complete_layer_cohort_min_layers", 3))
    min_fraction = float(getattr(config, "complete_layer_cohort_min_fraction", 0.80))
    if selected_layers < min_layers:
        return None
    if selected_layers / max(1.0, float(eligible_layers)) < min_fraction:
        return None

    cohort_union = {strand_id for cohort in cohort_counts for strand_id in cohort}
    return GraphCoreCandidate(
        graph_variant="layer_cohort",
        edge_min_count=0,
        core_count=int(len(selected_ids)),
        selected_ids=list(selected_ids),
        node_count=int(len(cohort_union)),
        edge_count=0,
        observed_layers=int(selected_layers),
    )


def select_complete_visibility_candidate(
    config: Any,
    *,
    raw_candidates: list[GraphCoreCandidate],
    outer_candidates: list[GraphCoreCandidate],
    outer_gap_candidates: list[GraphCoreCandidate],
    layer_cohort_candidate: GraphCoreCandidate | None,
    layer_q90: int,
    layer_max: int,
    support_ratio: float,
    q90_spread: int,
) -> GraphCoreCandidate | None:
    outer_gap_reduction = _select_complete_outer_gap_reduction(
        config,
        outer_gap_candidates,
        layer_q90=layer_q90,
        support_ratio=support_ratio,
        q90_spread=q90_spread,
    )
    if outer_gap_reduction is not None:
        return outer_gap_reduction

    layer_max_candidate = _select_complete_layer_max_candidate(
        config,
        raw_candidates=raw_candidates,
        outer_gap_candidates=outer_gap_candidates,
        layer_q90=layer_q90,
        layer_max=layer_max,
    )
    if layer_max_candidate is not None:
        return layer_max_candidate

    candidates = raw_candidates + outer_candidates + outer_gap_candidates
    candidates = [candidate for candidate in candidates if candidate.core_count > 0]
    if not candidates:
        return None

    exact = [candidate for candidate in candidates if candidate.core_count == layer_q90]
    if exact:
        return max(
            exact,
            key=lambda candidate: (
                candidate.edge_min_count,
                candidate.graph_variant == "raw",
                candidate.node_count,
            ),
        )
    return layer_cohort_candidate


def _select_complete_layer_max_candidate(
    config: Any,
    *,
    raw_candidates: list[GraphCoreCandidate],
    outer_gap_candidates: list[GraphCoreCandidate],
    layer_q90: int,
    layer_max: int,
) -> GraphCoreCandidate | None:
    if not bool(getattr(config, "complete_layer_max_candidate_enabled", True)):
        return None
    if layer_max <= layer_q90:
        return None

    max_delta = int(getattr(config, "complete_layer_max_candidate_max_q90_delta", 1))
    if max_delta > 0 and layer_max - layer_q90 > max_delta:
        return None

    eligible = [
        candidate
        for candidate in raw_candidates + outer_gap_candidates
        if candidate.core_count == layer_max
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            candidate.edge_min_count,
            candidate.graph_variant == "raw",
            candidate.node_count,
        ),
    )


def _select_complete_outer_gap_reduction(
    config: Any,
    candidates: list[GraphCoreCandidate],
    *,
    layer_q90: int,
    support_ratio: float,
    q90_spread: int,
) -> GraphCoreCandidate | None:
    if not bool(getattr(config, "complete_outer_gap_reduction_enabled", True)):
        return None

    min_layer_q90 = int(getattr(config, "complete_outer_gap_min_layer_q90", 12))
    if layer_q90 < min_layer_q90:
        return None

    max_support_ratio = float(config.complete_outer_gap_max_support_ratio)
    if support_ratio > max_support_ratio:
        return None

    max_q90_spread = int(config.complete_outer_gap_max_q90_spread)
    if q90_spread > max_q90_spread:
        return None

    min_reduction = int(getattr(config, "complete_outer_gap_min_reduction", 2))
    min_count_fraction = float(
        getattr(config, "complete_outer_gap_min_count_fraction", 0.60)
    )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.core_count > 0
        and candidate.core_count <= layer_q90 - min_reduction
        and candidate.core_count >= int(np.ceil(layer_q90 * min_count_fraction))
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            abs(candidate.core_count - layer_q90),
            -candidate.edge_min_count,
        ),
    )
