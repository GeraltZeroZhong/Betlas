from __future__ import annotations

from collections import defaultdict

from .types import GraphCoreCandidate


def sequence_blocks(selected_ids: list[int]) -> list[list[int]]:
    blocks: list[list[int]] = []
    for strand_id in sorted(int(value) for value in selected_ids):
        if not blocks or strand_id != blocks[-1][-1] + 1:
            blocks.append([strand_id])
        else:
            blocks[-1].append(strand_id)
    return blocks


def largest_block_fraction(selected_ids: list[int]) -> tuple[int, float]:
    if not selected_ids:
        return 0, 0.0
    blocks = sequence_blocks(selected_ids)
    largest = max((len(block) for block in blocks), default=0)
    return len(blocks), float(largest) / max(1.0, float(len(selected_ids)))


def candidates_by_variant(
    candidates: list[GraphCoreCandidate],
) -> dict[str, list[GraphCoreCandidate]]:
    grouped: dict[str, list[GraphCoreCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.graph_variant].append(candidate)
    return {
        variant: sorted(items, key=lambda candidate: candidate.edge_min_count)
        for variant, items in grouped.items()
    }


def next_drop_fraction(
    candidate: GraphCoreCandidate,
    by_variant: dict[str, list[GraphCoreCandidate]],
) -> float:
    same_variant = by_variant.get(candidate.graph_variant, [])
    stronger = [
        item
        for item in same_variant
        if int(item.edge_min_count) > int(candidate.edge_min_count)
    ]
    if not stronger or candidate.core_count <= 0:
        return 0.0
    next_candidate = min(stronger, key=lambda item: item.edge_min_count)
    drop = max(0, int(candidate.core_count) - int(next_candidate.core_count))
    return float(drop) / max(1.0, float(candidate.core_count))


def plateau_edges(
    candidate: GraphCoreCandidate,
    by_variant: dict[str, list[GraphCoreCandidate]],
    *,
    tolerance: int,
) -> int:
    same_variant = by_variant.get(candidate.graph_variant, [])
    return sum(
        1
        for item in same_variant
        if int(item.edge_min_count) >= int(candidate.edge_min_count)
        and abs(int(item.core_count) - int(candidate.core_count)) <= tolerance
    )


def variant_agreement(
    candidate: GraphCoreCandidate,
    candidates: list[GraphCoreCandidate],
    *,
    tolerance: int,
) -> int:
    agreeing_variants = {
        item.graph_variant
        for item in candidates
        if item.core_count > 0
        and abs(int(item.core_count) - int(candidate.core_count)) <= tolerance
    }
    return len(agreeing_variants)


def same_edge_variant_agreement(
    candidate: GraphCoreCandidate,
    candidates: list[GraphCoreCandidate],
    *,
    tolerance: int,
) -> int:
    agreeing_variants = {
        item.graph_variant
        for item in candidates
        if item.core_count > 0
        and int(item.edge_min_count) == int(candidate.edge_min_count)
        and abs(int(item.core_count) - int(candidate.core_count)) <= tolerance
    }
    return len(agreeing_variants)
