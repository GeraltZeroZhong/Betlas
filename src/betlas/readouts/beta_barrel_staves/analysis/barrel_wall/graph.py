from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from ...constants import EPSILON
from .layers import ordered_layer_angle_observations
from .types import GraphCoreCandidate, GraphEvidence


def largest_connected_component(
    nodes: set[int],
    adjacency: dict[int, set[int]],
) -> list[int]:
    seen: set[int] = set()
    best: list[int] = []
    for start in sorted(nodes):
        if start in seen:
            continue
        component: list[int] = []
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node] & nodes):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        if len(component) > len(best):
            best = component
    return sorted(best)


def two_core(nodes: set[int], adjacency: dict[int, set[int]]) -> set[int]:
    core = set(nodes)
    changed = True
    while changed:
        changed = False
        to_remove = [node for node in core if len(adjacency[node] & core) < 2]
        if to_remove:
            core.difference_update(to_remove)
            changed = True
    return core


def edge_key(first_id: int, second_id: int) -> tuple[int, int]:
    return (first_id, second_id) if first_id < second_id else (second_id, first_id)


def _layer_neighbor_edges(
    ordered_observations: list[tuple[int, float, float]],
    *,
    graph_config: Any,
) -> list[tuple[int, int]]:
    layer_count = len(ordered_observations)
    if layer_count < 2:
        return []

    angles = np.asarray([angle for _strand_id, _rank, angle in ordered_observations], dtype=float)
    wrapped_angles = np.concatenate([angles, [angles[0] + (2.0 * np.pi)]])
    angular_gaps = np.diff(wrapped_angles)
    max_gap_index = int(np.argmax(angular_gaps))
    max_gap = float(angular_gaps[max_gap_index])
    median_gap = float(np.median(angular_gaps))
    coverage_fraction = max(0.0, (2.0 * np.pi) - max_gap) / (2.0 * np.pi)

    close_layer = (
        max_gap
        <= float(getattr(graph_config, "partial_layer_close_max_gap_ratio", 3.0))
        * max(median_gap, EPSILON)
        or coverage_fraction
        >= float(getattr(graph_config, "partial_layer_min_closed_coverage_fraction", 0.75))
    )
    edges: list[tuple[int, int]] = []
    for index, (first_id, _rank, _angle) in enumerate(ordered_observations):
        if not close_layer and index == max_gap_index:
            continue
        second_id = ordered_observations[(index + 1) % layer_count][0]
        if first_id != second_id:
            edges.append(edge_key(int(first_id), int(second_id)))
    return edges


def restricted_adjacency(
    nodes: set[int],
    adjacency: dict[int, set[int]],
) -> dict[int, set[int]]:
    return {node: set(adjacency[node] & nodes) for node in nodes}


def prune_to_near_cycle(
    nodes: set[int],
    adjacency: dict[int, set[int]],
    edge_support: dict[tuple[int, int], int],
) -> list[int]:
    """
    Derive a simple-cycle-like hypothesis from a supported slice-neighbor graph.

    The raw 2-core is intentionally permissive: intermittent plug/loop beta
    fragments can enter as branches if they are adjacent to barrel-wall points in
    several z layers. A physical barrel wall should instead look like one
    connected cycle in the aggregated annular adjacency graph. This routine
    removes the weakest edges incident to branch nodes, repeatedly recomputing
    the 2-core, and returns the largest remaining cycle-like component.
    """
    selected = set(largest_connected_component(two_core(nodes, adjacency), adjacency))
    if not selected:
        return []

    working = restricted_adjacency(selected, adjacency)
    while selected:
        degrees = {node: len(working[node] & selected) for node in selected}
        branch_nodes = [node for node, degree in degrees.items() if degree > 2]
        if not branch_nodes:
            break

        removable_edges: list[tuple[int, int, int, int, int]] = []
        for node in branch_nodes:
            for neighbor in sorted(working[node] & selected):
                first_id, second_id = edge_key(node, neighbor)
                removable_edges.append(
                    (
                        int(edge_support.get((first_id, second_id), 0)),
                        int(degrees.get(neighbor, 0)),
                        first_id,
                        second_id,
                        node,
                    )
                )
        if not removable_edges:
            break

        _support, _neighbor_degree, first_id, second_id, _branch_node = min(
            removable_edges
        )
        working[first_id].discard(second_id)
        working[second_id].discard(first_id)

        core_nodes = two_core(selected, working)
        selected = set(largest_connected_component(core_nodes, working))
        working = restricted_adjacency(selected, working)

    return sorted(selected)


def build_graph_evidence(
    slices_dict: dict[float, list[tuple[float, ...]]],
    *,
    graph_variant: str,
    outer_filter_strategy: str,
    graph_config: Any,
) -> GraphEvidence:
    edge_support: dict[tuple[int, int], int] = defaultdict(int)
    node_support: dict[int, int] = defaultdict(int)
    node_radial_rank_sum: dict[int, float] = defaultdict(float)
    node_radial_rank_count: dict[int, int] = defaultdict(int)
    observed_layers = 0
    min_layer_points = int(getattr(graph_config, "min_layer_points", 4))

    for _z_value, points in sorted(slices_dict.items()):
        ordered_observations = ordered_layer_angle_observations(
            np.asarray(points, dtype=float),
            graph_config=graph_config,
            outer_filter_strategy=outer_filter_strategy,
        )
        ordered_ids = [strand_id for strand_id, _rank, _angle in ordered_observations]
        if len(ordered_ids) < min_layer_points:
            continue

        observed_layers += 1
        for strand_id, radial_rank, _angle in ordered_observations:
            node_support[int(strand_id)] += 1
            node_radial_rank_sum[int(strand_id)] += float(radial_rank)
            node_radial_rank_count[int(strand_id)] += 1

        for first_id, second_id in _layer_neighbor_edges(
            ordered_observations,
            graph_config=graph_config,
        ):
            edge_support[(first_id, second_id)] += 1

    return GraphEvidence(
        graph_variant=graph_variant,
        edge_support=dict(edge_support),
        node_support=dict(node_support),
        node_radial_rank_sum=dict(node_radial_rank_sum),
        node_radial_rank_count=dict(node_radial_rank_count),
        observed_layers=int(observed_layers),
    )


def candidate_from_evidence(
    evidence: GraphEvidence,
    *,
    edge_min_count: int,
    min_node_support: int,
    selection_mode: str = "two_core",
) -> GraphCoreCandidate:
    supported_nodes = {
        strand_id
        for strand_id, support in evidence.node_support.items()
        if support >= min_node_support
    }
    adjacency: dict[int, set[int]] = defaultdict(set)
    kept_edges = 0
    for (first_id, second_id), support in evidence.edge_support.items():
        if support < edge_min_count:
            continue
        if first_id not in supported_nodes or second_id not in supported_nodes:
            continue
        adjacency[first_id].add(second_id)
        adjacency[second_id].add(first_id)
        kept_edges += 1

    core_nodes = two_core(supported_nodes, adjacency)
    selected_ids = largest_connected_component(core_nodes, adjacency)
    graph_variant = evidence.graph_variant
    if selection_mode == "near_cycle":
        selected_ids = prune_to_near_cycle(core_nodes, adjacency, evidence.edge_support)
        graph_variant = f"{evidence.graph_variant}_cycle"
    selected_set = set(selected_ids)
    component_edge_count = 0
    for first_id in selected_set:
        for second_id in adjacency[first_id] & selected_set:
            if first_id < second_id:
                component_edge_count += 1
    component_degrees = [
        len(adjacency[strand_id] & selected_set) for strand_id in selected_ids
    ]
    component_avg_degree = (
        2.0 * float(component_edge_count) / float(len(selected_ids))
        if selected_ids
        else 0.0
    )
    component_cycle_rank = (
        float(component_edge_count) - float(len(selected_ids)) + 1.0
        if selected_ids
        else 0.0
    )
    component_degree2_fraction = (
        sum(1 for degree in component_degrees if degree == 2)
        / max(1.0, float(len(component_degrees)))
        if component_degrees
        else 0.0
    )
    component_branch_fraction = (
        sum(1 for degree in component_degrees if degree > 2)
        / max(1.0, float(len(component_degrees)))
        if component_degrees
        else 0.0
    )
    component_radial_ranks = [
        float(evidence.node_radial_rank_sum[strand_id])
        / max(1.0, float(evidence.node_radial_rank_count[strand_id]))
        for strand_id in selected_ids
        if evidence.node_radial_rank_count.get(strand_id, 0) > 0
    ]
    component_radial_rank_mean = (
        float(np.mean(np.asarray(component_radial_ranks, dtype=float)))
        if component_radial_ranks
        else 0.0
    )
    component_radial_rank_min = (
        float(min(component_radial_ranks)) if component_radial_ranks else 0.0
    )
    return GraphCoreCandidate(
        graph_variant=graph_variant,
        edge_min_count=int(edge_min_count),
        core_count=int(len(selected_ids)),
        selected_ids=selected_ids,
        node_count=int(len(supported_nodes)),
        edge_count=int(kept_edges),
        observed_layers=int(evidence.observed_layers),
        component_edge_count=int(component_edge_count),
        component_avg_degree=float(component_avg_degree),
        component_cycle_rank=float(component_cycle_rank),
        component_degree2_fraction=float(component_degree2_fraction),
        component_branch_fraction=float(component_branch_fraction),
        component_radial_rank_mean=float(component_radial_rank_mean),
        component_radial_rank_min=float(component_radial_rank_min),
    )


def candidate_curve(
    slices_dict: dict[float, list[tuple[float, ...]]],
    *,
    graph_variant: str,
    edge_min_counts: list[int],
    outer_filter_strategy: str,
    graph_config: Any,
    include_near_cycle: bool = False,
) -> list[GraphCoreCandidate]:
    evidence = build_graph_evidence(
        slices_dict,
        graph_variant=graph_variant,
        outer_filter_strategy=outer_filter_strategy,
        graph_config=graph_config,
    )
    min_node_support = int(getattr(graph_config, "min_node_support_layers", 2))
    candidates: list[GraphCoreCandidate] = []
    for edge_min_count in sorted(set(int(value) for value in edge_min_counts)):
        if int(edge_min_count) <= 0:
            continue
        candidates.append(
            candidate_from_evidence(
                evidence,
                edge_min_count=int(edge_min_count),
                min_node_support=min_node_support,
            )
        )
        if include_near_cycle:
            candidates.append(
                candidate_from_evidence(
                    evidence,
                    edge_min_count=int(edge_min_count),
                    min_node_support=min_node_support,
                    selection_mode="near_cycle",
                )
            )
    return candidates
