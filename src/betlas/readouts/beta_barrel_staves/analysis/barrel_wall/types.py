from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphEvidence:
    graph_variant: str
    edge_support: dict[tuple[int, int], int]
    node_support: dict[int, int]
    node_radial_rank_sum: dict[int, float]
    node_radial_rank_count: dict[int, int]
    observed_layers: int


@dataclass(frozen=True)
class GraphCoreCandidate:
    graph_variant: str
    edge_min_count: int
    core_count: int
    selected_ids: list[int]
    node_count: int
    edge_count: int
    observed_layers: int
    component_edge_count: int = 0
    component_avg_degree: float = 0.0
    component_cycle_rank: float = 0.0
    component_degree2_fraction: float = 0.0
    component_branch_fraction: float = 0.0
    component_radial_rank_mean: float = 0.0
    component_radial_rank_min: float = 0.0
    selection_strategy: str = ""
    membership_score: float = 0.0
    membership_block_count: int = 0
    membership_largest_block_fraction: float = 0.0
    membership_plateau_edges: int = 0
    membership_next_drop_fraction: float = 0.0
    membership_variant_agreement: int = 0


@dataclass(frozen=True)
class BarrelWallGraphSelection:
    strand_count: int
    selected_ids: list[int]
    confidence_basis: str
    graph_variant: str
    edge_min_count: int
    visibility_regime: str
    support_ratio: float
    score: float
    feature_table: dict[str, object]


def to_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def curve_counts(candidates: list[GraphCoreCandidate]) -> str:
    return ";".join(
        f"{candidate.graph_variant}:e{candidate.edge_min_count}={candidate.core_count}"
        for candidate in candidates
    )


def empty_barrel_wall_feature_table() -> dict[str, object]:
    return {
        "barrel_wall_graph_applied": 0,
        "barrel_wall_graph_count": 0,
        "barrel_wall_graph_variant": "",
        "barrel_wall_graph_edge_min_count": 0,
        "barrel_wall_graph_visibility_regime": "",
        "barrel_wall_graph_selection_strategy": "",
        "barrel_wall_graph_support_ratio": 0.0,
        "barrel_wall_graph_node_count": 0,
        "barrel_wall_graph_edge_count": 0,
        "barrel_wall_graph_component_edge_count": 0,
        "barrel_wall_graph_component_avg_degree": 0.0,
        "barrel_wall_graph_component_cycle_rank": 0.0,
        "barrel_wall_graph_component_degree2_fraction": 0.0,
        "barrel_wall_graph_component_branch_fraction": 0.0,
        "barrel_wall_graph_component_radial_rank_mean": 0.0,
        "barrel_wall_graph_component_radial_rank_min": 0.0,
        "barrel_wall_graph_edge_density": 0.0,
        "barrel_wall_graph_membership_score": 0.0,
        "barrel_wall_graph_membership_block_count": 0,
        "barrel_wall_graph_membership_largest_block_fraction": 0.0,
        "barrel_wall_graph_membership_plateau_edges": 0,
        "barrel_wall_graph_membership_next_drop_fraction": 0.0,
        "barrel_wall_graph_membership_variant_agreement": 0,
        "barrel_wall_graph_observed_layers": 0,
        "barrel_wall_graph_selected_ids": "",
        "barrel_wall_graph_raw_curve": "",
        "barrel_wall_graph_outer_curve": "",
        "barrel_wall_graph_outer_gap_curve": "",
    }
