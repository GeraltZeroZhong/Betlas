"""Pairing, spatial contact and polymer order are distinct structure relations.

No graph statistic here is a barrel detector or a unique topology identifier.
Summaries describe the retained graph, including explicit annotation limitations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from betlas.models import StructureGeometry

SCHEMA = "betlas.geometry-connectivity.v1"


def _graph_statistics(nodes: list[str], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    adjacency = {node: set() for node in nodes}
    for left, right in pairs:
        if left == right or left not in adjacency or right not in adjacency:
            raise ValueError("Pairing edges must join two distinct retained nodes")
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(nodes)
    components = []
    while unseen:
        stack = [unseen.pop()]
        component = set(stack)
        while stack:
            for neighbor in adjacency[stack.pop()] & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(component)
    degrees = [len(adjacency[node]) for node in nodes]
    edge_count = sum(degrees) // 2
    cycle_rank = edge_count - len(nodes) + len(components)
    # A degree-two connected component is a simple cycle. Branching graphs may
    # also contain cycles; cycle rank measures those without choosing a basis.
    simple_cycle_nodes = sum(len(c) for c in components
                             if len(c) >= 3 and all(len(adjacency[n]) == 2 for n in c))
    return {
        "node_count": len(nodes), "edge_count": edge_count,
        "component_count": len(components),
        "largest_component_fraction": max(map(len, components), default=0) / len(nodes) if nodes else None,
        "degree_mean": float(np.mean(degrees)) if degrees else None,
        "degree_max": max(degrees, default=0),
        "branch_fraction": sum(d > 2 for d in degrees) / len(nodes) if nodes else None,
        "degree_two_fraction": sum(d == 2 for d in degrees) / len(nodes) if nodes else None,
        "isolated_fraction": sum(d == 0 for d in degrees) / len(nodes) if nodes else None,
        "cycle_rank": cycle_rank,
        "cycle_rank_per_node": cycle_rank / len(nodes) if nodes else None,
        "simple_cycle_component_node_fraction": simple_cycle_nodes / len(nodes) if nodes else None,
    }


def summarize_connectivity(
    geometry: StructureGeometry, *, include_contacts: bool = False,
) -> dict[str, Any]:
    """Export auditable node/edge records and retained pairing graph statistics.

    Sequence ranks are computed within each polymer chain using label_seq_id,
    never author numbering, node IDs or a purported spatial sheet order.
    Cross-fragment order is retained but is not an observed connecting loop.
    """
    nodes = []
    for element in geometry.beta_segments:
        residues = [geometry.residues[i] for i in element.residue_indices]
        if not residues or any(r.label_seq_id is None for r in residues):
            raise ValueError("Connectivity requires retained polymer label positions")
        if len({r.chain_id for r in residues}) != 1:
            raise ValueError("A beta node cannot span multiple polymer chains")
        positions = sorted(int(r.label_seq_id) for r in residues)
        points = np.array([r.coord_ca for r in residues], dtype=float)
        direction = points[-1] - points[0]
        norm = float(np.linalg.norm(direction))
        nodes.append({
            "node_id": element.element_id, "chain_id": residues[0].chain_id,
            "fragment_id": element.fragment_id, "sheet_id": element.sheet_id,
            "sheet_range_id": element.sheet_range_id,
            "label_start": positions[0], "label_end": positions[-1],
            "label_positions": positions,
            "residue_uids": [r.residue_uid for r in residues],
            "residue_count": len(residues),
            "ntc_direction": (direction / norm).tolist() if norm else None,
        })
    nodes.sort(key=lambda n: (n["chain_id"], n["label_start"], n["label_end"]))
    by_id = {node["node_id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Beta node identifiers must be unique")
    by_chain: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        by_chain[node["chain_id"]].append(node)
    for members in by_chain.values():
        for rank, node in enumerate(members):
            node["polymer_strand_rank"] = rank

    observed: dict[str, set[int]] = defaultdict(set)
    for residue in geometry.residues:
        if residue.label_seq_id is not None:
            observed[residue.chain_id].add(int(residue.label_seq_id))

    def order_context(left: dict, right: dict) -> dict:
        if left["chain_id"] != right["chain_id"]:
            return {"sequence_context": "cross_chain", "strand_rank_gap": None,
                    "polymer_residue_gap": None, "observed_chain_path": False}
        first, last = sorted((left, right), key=lambda n: n["label_start"])
        if first["label_end"] >= last["label_start"]:
            return {"sequence_context": "overlapping_polymer_ranges", "strand_rank_gap": None,
                    "polymer_residue_gap": None, "observed_chain_path": False}
        rank_gap = abs(left["polymer_strand_rank"] - right["polymer_strand_rank"])
        residue_gap = max(0, last["label_start"] - first["label_end"] - 1)
        same_fragment = bool(first["fragment_id"] and first["fragment_id"] == last["fragment_id"])
        path_present = set(range(first["label_end"], last["label_start"] + 1)) <= observed[first["chain_id"]]
        return {"sequence_context": "same_fragment" if same_fragment else "cross_fragment",
                "strand_rank_gap": rank_gap, "polymer_residue_gap": residue_gap,
                "observed_chain_path": same_fragment and path_present}

    senses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for left, right, sense in geometry.sheet_connections:
        if left not in by_id or right not in by_id or left == right:
            raise ValueError("Pairing edges must join two distinct retained nodes")
        senses[tuple(sorted((left, right)))].add(sense)
    pairing_edges = []
    for (left, right), values in sorted(senses.items()):
        pairing_edges.append({"source": left, "target": right,
                              "relation": "annotation_pairing", "senses": sorted(values),
                              "sense": next(iter(values)) if len(values) == 1 else "conflicting",
                              **order_context(by_id[left], by_id[right])})
    sequence_relations = []
    for members in by_chain.values():
        for left, right in zip(members, members[1:], strict=False):
            sequence_relations.append({"source": left["node_id"], "target": right["node_id"],
                                       "relation": "polymer_order", **order_context(left, right)})

    warnings = list(geometry.warnings)
    if any(edge["sense"] == "conflicting" for edge in pairing_edges):
        warnings.append("conflicting_pairing_senses")
    if not nodes:
        status = "no_beta_segments"
    elif "missing_deposited_sheet_order" in warnings:
        status = "annotation_pairing_missing"
    elif any(w in warnings for w in ("clipped_ladder_pairing_unavailable", "clipped_deposited_pairing_unavailable")):
        status = "partial_clipped_pairings"
    elif not pairing_edges:
        status = "no_retained_pairings"
    else:
        status = "available"
    stats = _graph_statistics(list(by_id), list(senses))
    known = [edge for edge in pairing_edges if edge["sense"] in {"parallel", "anti-parallel"}]
    same_chain = [edge for edge in pairing_edges if edge["strand_rank_gap"] is not None]
    gaps = [edge["strand_rank_gap"] for edge in same_chain]
    stats.update({
        "known_sense_fraction": len(known) / len(pairing_edges) if pairing_edges else None,
        "parallel_fraction_known": sum(e["sense"] == "parallel" for e in known) / len(known) if known else None,
        "antiparallel_fraction_known": sum(e["sense"] == "anti-parallel" for e in known) / len(known) if known else None,
        "cross_chain_fraction": sum(e["sequence_context"] == "cross_chain" for e in pairing_edges) / len(pairing_edges) if pairing_edges else None,
        "cross_fragment_fraction": sum(e["sequence_context"] == "cross_fragment" for e in pairing_edges) / len(pairing_edges) if pairing_edges else None,
        "sequence_span_available_fraction": len(same_chain) / len(pairing_edges) if pairing_edges else None,
        "strand_rank_gap_mean": float(np.mean(gaps)) if gaps else None,
        "strand_rank_gap_max": max(gaps) if gaps else None,
        "nonlocal_fraction_same_chain": sum(gap > 2 for gap in gaps) / len(gaps) if gaps else None,
        "observed_path_fraction_same_chain": sum(e["observed_chain_path"] for e in same_chain) / len(same_chain) if same_chain else None,
    })
    result = {"schema": SCHEMA, "record_id": geometry.domain.record_id,
              "annotation_source": geometry.secondary_structure_source,
              "pairing_status": status, "warnings": warnings,
              "metrics": {f"pairing_{key}": value for key, value in stats.items()},
              "nodes": nodes, "pairing_edges": pairing_edges,
              "sequence_relations": sequence_relations,
              "interpretation": "Retained graph summaries do not establish a barrel or unique fold."}
    if include_contacts:
        elements = {element.element_id: element for element in geometry.beta_segments}
        contacts = []
        for i, left in enumerate(nodes):
            left_points = np.array([geometry.residues[j].coord_ca for j in elements[left["node_id"]].residue_indices])
            for right in nodes[i + 1:]:
                right_points = np.array([geometry.residues[j].coord_ca for j in elements[right["node_id"]].residue_indices])
                distance = float(np.min(np.linalg.norm(left_points[:, None] - right_points[None, :], axis=2)))
                if distance <= 10.0:
                    contacts.append({"source": left["node_id"], "target": right["node_id"],
                                     "relation": "ca_spatial_contact", "min_ca_distance_angstrom": distance,
                                     "within_8_angstrom": distance <= 8.0,
                                     **order_context(left, right)})
        result["contact_edges"] = contacts
    return result
