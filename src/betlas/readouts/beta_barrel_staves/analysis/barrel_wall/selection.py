from __future__ import annotations

from typing import Any

from .complete import complete_layer_cohort_candidate, select_complete_visibility_candidate
from .features import graph_selection_feature_table
from .graph import candidate_curve
from .membership import select_sparse_wall_membership_candidate
from .types import (
    BarrelWallGraphSelection,
    GraphCoreCandidate,
    to_int,
)


def _graph_decision_score(candidate: GraphCoreCandidate, config: Any) -> float:
    edge_fraction = float(candidate.edge_min_count) / max(1.0, float(candidate.observed_layers))
    if candidate.membership_score <= 0.0:
        return min(1.0, edge_fraction)
    membership_scale = max(
        1e-9,
        float(getattr(config, "sparse_membership_min_score", 1.0)),
    )
    membership_fraction = float(candidate.membership_score) / membership_scale
    return min(1.0, max(edge_fraction, membership_fraction))


class BarrelWallGraphCoreSelector:
    """
    Count barrel-wall staves from cross-slice angular-adjacency evidence.

    Each slice contributes a cyclic neighbor observation among collapsed
    beta-run intersections. Barrel-wall runs should remain in a stable graph
    core across z, while plug/loop beta fragments tend to be intermittent,
    radially inner, or unstable neighbors.
    """

    def __init__(self, graph_config: Any):
        self.config = graph_config

    def select(
        self,
        slices_dict: dict[float, list[tuple[float, ...]]],
        base_report: dict[str, object],
    ) -> BarrelWallGraphSelection | None:
        cfg = self.config
        if not bool(getattr(cfg, "enabled", False)):
            return None

        layer_q90 = to_int(base_report.get("layer_count_q90", 0))
        layer_max = to_int(base_report.get("layer_count_max", 0))
        support_consensus = to_int(base_report.get("support_consensus_strands", 0))
        current_count = to_int(base_report.get("strand_count", 0))
        if layer_q90 <= 0 or current_count <= 0:
            return None

        support_ratio = support_consensus / max(1.0, float(layer_q90))
        q90_spread = max(0, layer_max - layer_q90)
        complete_visibility = (
            layer_q90 >= int(getattr(cfg, "complete_min_layer_q90", 0))
            and support_ratio <= float(getattr(cfg, "complete_max_support_ratio", 0.0))
            and q90_spread <= int(getattr(cfg, "complete_max_q90_spread", 0))
        )
        sparse_visibility = support_ratio >= float(
            getattr(cfg, "sparse_min_support_ratio", 0.0)
        )
        if not complete_visibility and not sparse_visibility:
            return None

        raw_candidates = (
            candidate_curve(
                slices_dict,
                graph_variant="raw",
                edge_min_counts=list(getattr(cfg, "edge_min_counts", [])),
                outer_filter_strategy="none",
                graph_config=cfg,
                include_near_cycle=bool(
                    getattr(cfg, "near_cycle_variant_enabled", False)
                ),
            )
            if bool(getattr(cfg, "raw_variant_enabled", True))
            else []
        )
        outer_candidates = (
            candidate_curve(
                slices_dict,
                graph_variant="outer",
                edge_min_counts=list(getattr(cfg, "outer_edge_min_counts", [])),
                outer_filter_strategy="quantile",
                graph_config=cfg,
                include_near_cycle=bool(
                    getattr(cfg, "near_cycle_variant_enabled", False)
                ),
            )
            if bool(getattr(cfg, "outer_variant_enabled", True))
            else []
        )
        outer_gap_candidates = (
            candidate_curve(
                slices_dict,
                graph_variant="outer_gap",
                edge_min_counts=list(getattr(cfg, "outer_edge_min_counts", [])),
                outer_filter_strategy="radial_gap",
                graph_config=cfg,
                include_near_cycle=bool(
                    getattr(cfg, "near_cycle_variant_enabled", False)
                ),
            )
            if bool(getattr(cfg, "outer_gap_variant_enabled", True))
            else []
        )
        if not raw_candidates and not outer_candidates and not outer_gap_candidates:
            return None

        selected = None
        visibility_regime = ""
        if complete_visibility:
            layer_cohort_candidate = complete_layer_cohort_candidate(
                cfg,
                base_report,
                layer_q90=layer_q90,
            )
            selected = select_complete_visibility_candidate(
                cfg,
                raw_candidates=raw_candidates,
                outer_candidates=outer_candidates,
                outer_gap_candidates=outer_gap_candidates,
                layer_cohort_candidate=layer_cohort_candidate,
                layer_q90=layer_q90,
                layer_max=layer_max,
                support_ratio=support_ratio,
                q90_spread=q90_spread,
            )
            if selected is not None:
                visibility_regime = "complete"

        if selected is None and sparse_visibility:
            selected = self._select_sparse_visibility(
                outer_candidates=outer_candidates,
                outer_gap_candidates=outer_gap_candidates,
                raw_candidates=raw_candidates,
                support_consensus=support_consensus,
                layer_q90=layer_q90,
            )
            visibility_regime = "sparse"

        if selected is None:
            return None
        min_count = int(getattr(cfg, "min_selected_count", 1))
        max_count = int(getattr(cfg, "max_selected_count", 64))
        if not min_count <= selected.core_count <= max_count:
            return None

        feature_table = graph_selection_feature_table(
            selected=selected,
            current_count=current_count,
            support_ratio=support_ratio,
            visibility_regime=visibility_regime,
            raw_candidates=raw_candidates,
            outer_candidates=outer_candidates,
            outer_gap_candidates=outer_gap_candidates,
        )
        return BarrelWallGraphSelection(
            strand_count=int(selected.core_count),
            selected_ids=list(selected.selected_ids),
            confidence_basis="barrel_wall_graph_core",
            graph_variant=selected.graph_variant,
            edge_min_count=int(selected.edge_min_count),
            visibility_regime=visibility_regime,
            support_ratio=float(support_ratio),
            score=_graph_decision_score(selected, cfg),
            feature_table=feature_table,
        )

    def _select_sparse_visibility(
        self,
        *,
        outer_candidates: list[GraphCoreCandidate],
        outer_gap_candidates: list[GraphCoreCandidate],
        raw_candidates: list[GraphCoreCandidate],
        support_consensus: int,
        layer_q90: int,
    ) -> GraphCoreCandidate | None:
        wall_membership = select_sparse_wall_membership_candidate(
            raw_candidates=raw_candidates,
            outer_candidates=outer_candidates,
            outer_gap_candidates=outer_gap_candidates,
            support_consensus=support_consensus,
            layer_q90=layer_q90,
            config=self.config,
        )
        if wall_membership is not None:
            return wall_membership
        return None
