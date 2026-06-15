from __future__ import annotations

from .types import GraphCoreCandidate, curve_counts


def graph_selection_feature_table(
    *,
    selected: GraphCoreCandidate,
    current_count: int,
    support_ratio: float,
    visibility_regime: str,
    raw_candidates: list[GraphCoreCandidate],
    outer_candidates: list[GraphCoreCandidate],
    outer_gap_candidates: list[GraphCoreCandidate],
) -> dict[str, object]:
    return {
        "barrel_wall_graph_applied": int(selected.core_count != current_count),
        "barrel_wall_graph_count": int(selected.core_count),
        "barrel_wall_graph_variant": selected.graph_variant,
        "barrel_wall_graph_edge_min_count": int(selected.edge_min_count),
        "barrel_wall_graph_visibility_regime": visibility_regime,
        "barrel_wall_graph_selection_strategy": selected.selection_strategy,
        "barrel_wall_graph_support_ratio": float(support_ratio),
        "barrel_wall_graph_node_count": int(selected.node_count),
        "barrel_wall_graph_edge_count": int(selected.edge_count),
        "barrel_wall_graph_component_edge_count": int(selected.component_edge_count),
        "barrel_wall_graph_component_avg_degree": float(selected.component_avg_degree),
        "barrel_wall_graph_component_cycle_rank": float(selected.component_cycle_rank),
        "barrel_wall_graph_component_degree2_fraction": float(
            selected.component_degree2_fraction
        ),
        "barrel_wall_graph_component_branch_fraction": float(
            selected.component_branch_fraction
        ),
        "barrel_wall_graph_component_radial_rank_mean": float(
            selected.component_radial_rank_mean
        ),
        "barrel_wall_graph_component_radial_rank_min": float(
            selected.component_radial_rank_min
        ),
        "barrel_wall_graph_edge_density": (
            float(selected.edge_count) / max(1.0, float(selected.core_count))
        ),
        "barrel_wall_graph_membership_score": float(selected.membership_score),
        "barrel_wall_graph_membership_block_count": int(
            selected.membership_block_count
        ),
        "barrel_wall_graph_membership_largest_block_fraction": float(
            selected.membership_largest_block_fraction
        ),
        "barrel_wall_graph_membership_plateau_edges": int(
            selected.membership_plateau_edges
        ),
        "barrel_wall_graph_membership_next_drop_fraction": float(
            selected.membership_next_drop_fraction
        ),
        "barrel_wall_graph_membership_variant_agreement": int(
            selected.membership_variant_agreement
        ),
        "barrel_wall_graph_observed_layers": int(selected.observed_layers),
        "barrel_wall_graph_selected_ids": ";".join(
            str(value) for value in selected.selected_ids
        ),
        "barrel_wall_graph_raw_curve": curve_counts(raw_candidates),
        "barrel_wall_graph_outer_curve": curve_counts(outer_candidates),
        "barrel_wall_graph_outer_gap_curve": curve_counts(outer_gap_candidates),
    }
