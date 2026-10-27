"""Descriptive label overlays on a frozen, label-independent geometry graph.

Neighbor shares are label-conditioned summaries, not physical similarity or
independent observations. Isolated and geometrically unavailable domains retain
undefined shares. Target-family weights use all nodes in the supplied phase.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any


def _records(table: Any) -> list[dict[str, Any]]:
    if hasattr(table, "to_dict") and not isinstance(table, Mapping):
        return table.to_dict(orient="records")
    return [dict(row) for row in table]


def _text(value: Any) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _boolean(value: Any) -> bool:
    text = _text(value).lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _mean(values) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def summarize_relationships(nodes: Any, edges: Any, label_column: str,
                            all_labels=None, *, pair_tables=True) -> dict[str, Any]:
    """Overlay labels on short mutual edges and return three descriptive tables.

    Nodes require ``record_id, cath_homology_code, packing_eligible`` and the
    requested label column; ``pdb_id`` is optional and missing PDB support is
    recorded. Edges require ``source_id, target_id, distance, mutual, short_edge``.
    Reciprocal rows are deduplicated into one undirected domain pair. No further
    H/PDB exclusions are introduced here: upstream graph construction owns them.

    The returned ``directed`` table contains all ordered label pairs, including
    diagonal pairs. ``undirected`` contains all unordered label pairs, including
    diagonals. ``per_domain`` retains every input node. These are lists suitable
    for DataFrame construction; ``summary`` states denominators and conventions.
    Set ``pair_tables=False`` for domain-only overlays with many topology labels.
    """
    node_rows, edge_rows = _records(nodes), _records(edges)
    indexed = {}
    for row in node_rows:
        record_id = _text(row.get("record_id"))
        label, homology = _text(row.get(label_column)), _text(row.get("cath_homology_code"))
        if not record_id or not label or not homology:
            raise ValueError("Every phase node requires record_id, label, and cath_homology_code")
        if record_id in indexed:
            raise ValueError(f"Duplicate node record_id: {record_id}")
        indexed[record_id] = {"record_id": record_id, "label": label, "homology": homology,
                              "pdb_id": _text(row.get("pdb_id")),
                              "packing_eligible": _boolean(row.get("packing_eligible"))}
    observed_labels = {row["label"] for row in indexed.values()}
    labels = sorted(observed_labels) if all_labels is None else list(dict.fromkeys(map(str, all_labels)))
    if observed_labels.difference(labels):
        raise ValueError("all_labels omits labels present in phase nodes")
    family_counts = Counter(row["homology"] for row in indexed.values())
    unique_edges = {}
    edge_filter = Counter()
    for edge in edge_rows:
        if not _boolean(edge.get("mutual")) or not _boolean(edge.get("short_edge")):
            edge_filter["not_mutual_and_short"] += 1
            continue
        source, target = str(edge["source_id"]), str(edge["target_id"])
        if source not in indexed or target not in indexed:
            raise ValueError(f"Selected edge references a missing phase node: {source}, {target}")
        if source == target:
            raise ValueError(f"Self edge cannot define a neighborhood: {source}")
        if not indexed[source]["packing_eligible"] or not indexed[target]["packing_eligible"]:
            edge_filter["geometrically_unavailable_endpoint"] += 1
            continue
        distance = float(edge["distance"])
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Selected graph distances must be finite and nonnegative")
        key = tuple(sorted((source, target)))
        if key in unique_edges:
            if not math.isclose(unique_edges[key], distance, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"Reciprocal edge distances disagree: {key}")
            edge_filter["duplicate_undirected_pair"] += 1
        unique_edges[key] = distance

    neighbors = defaultdict(dict)
    for (source, target), distance in unique_edges.items():
        neighbors[source][target] = distance
        neighbors[target][source] = distance
    per_domain, domain_lookup = [], {}
    for record_id, node in indexed.items():
        adjacent = neighbors[record_id]
        status = ("geometry_unavailable" if not node["packing_eligible"] else
                  "measured" if adjacent else "no_neighbors")
        fractions = weighted_fractions = None
        overlap = mixing = weighted_mixing = None
        if adjacent:
            counts = Counter(indexed[target]["label"] for target in adjacent)
            fractions = {label: count / len(adjacent) for label, count in sorted(counts.items())}
            weighted = defaultdict(float)
            for target in adjacent:
                neighbor = indexed[target]
                weighted[neighbor["label"]] += 1 / family_counts[neighbor["homology"]]
            total_weight = sum(weighted.values())
            weighted_fractions = {label: weight / total_weight for label, weight in sorted(weighted.items())}
            overlap = any(label != node["label"] for label in counts)
            mixing = 1 - fractions.get(node["label"], 0.0)
            weighted_mixing = 1 - weighted_fractions.get(node["label"], 0.0)
        item = dict(node)
        item.update(neighborhood_status=status,
                    n_neighbors=len(adjacent) if node["packing_eligible"] else None,
                    n_neighbor_h=len({indexed[target]["homology"] for target in adjacent}) if node["packing_eligible"] else None,
                    n_neighbor_labels=len(fractions) if fractions is not None else None,
                    label_overlap=overlap, cross_label_fraction=mixing,
                    target_h_reweighted_cross_label_fraction=weighted_mixing,
                    neighbor_ids=sorted(adjacent) if node["packing_eligible"] else None,
                    neighbor_label_fractions=fractions,
                    target_h_reweighted_neighbor_label_fractions=weighted_fractions,
                    neighbor_distance_median=statistics.median(adjacent.values()) if adjacent else None,
                    neighbor_distance_min=min(adjacent.values()) if adjacent else None,
                    neighbor_distance_max=max(adjacent.values()) if adjacent else None,
                    h_phase_node_count=family_counts[node["homology"]])
        per_domain.append(item)
        domain_lookup[record_id] = item

    if not pair_tables:
        return {"per_domain": per_domain, "summary": {
            "label_column": label_column, "n_phase_nodes": len(indexed),
            "n_unique_short_mutual_edges": len(unique_edges), "pair_tables": False}}

    by_label = {label: [row for row in per_domain if row["label"] == label] for label in labels}
    eligible_by_label = {label: [row for row in rows if row["packing_eligible"]] for label, rows in by_label.items()}
    active_by_label = {label: [row for row in rows if row["neighborhood_status"] == "measured"]
                       for label, rows in by_label.items()}
    n_eligible = sum(row["packing_eligible"] for row in per_domain)
    n_active = sum(row["neighborhood_status"] == "measured" for row in per_domain)
    active_h_by_label = {}
    source_h_fractions = {}
    source_h_weighted_fractions = {}
    natural_fractions = {}
    source_largest_h = {}
    for label in labels:
        groups = defaultdict(list)
        for row in active_by_label[label]:
            groups[row["homology"]].append(row)
        active_h_by_label[label] = groups
        ordinary, reweighted = {}, {}
        for homology, rows in groups.items():
            ordinary[homology], reweighted[homology] = defaultdict(float), defaultdict(float)
            for row in rows:
                for target, fraction in row["neighbor_label_fractions"].items():
                    ordinary[homology][target] += fraction / len(rows)
                for target, fraction in row["target_h_reweighted_neighbor_label_fractions"].items():
                    reweighted[homology][target] += fraction / len(rows)
        source_h_fractions[label], source_h_weighted_fractions[label] = ordinary, reweighted
        natural_fractions[label] = defaultdict(float)
        for row in active_by_label[label]:
            for target, fraction in row["neighbor_label_fractions"].items():
                natural_fractions[label][target] += fraction / len(active_by_label[label])
        counts = Counter(row["homology"] for row in by_label[label])
        source_largest_h[label] = min(counts, key=lambda h: (-counts[h], h)) if counts else None

    directed_support = defaultdict(list)
    undirected_support = defaultdict(list)
    label_order = {label: index for index, label in enumerate(labels)}
    for (source, target), distance in unique_edges.items():
        left, right = indexed[source]["label"], indexed[target]["label"]
        directed_support[(left, right)].append((source, target, distance))
        directed_support[(right, left)].append((target, source, distance))
        key = (left, right) if label_order[left] <= label_order[right] else (right, left)
        undirected_support[key].append((source, target, distance))

    def directed_status(source_label, target_label):
        if not by_label[source_label]:
            return "source_label_absent"
        if not by_label[target_label]:
            return "target_label_absent"
        if not eligible_by_label[source_label]:
            return "source_geometry_unavailable"
        if not eligible_by_label[target_label]:
            return "target_geometry_unavailable"
        if not active_by_label[source_label]:
            return "source_no_neighbors"
        return "measured"

    directed = []
    for source_label in labels:
        source_nodes, source_eligible = by_label[source_label], eligible_by_label[source_label]
        source_active = active_by_label[source_label]
        source_h = {row["homology"] for row in source_nodes}
        source_eligible_h = {row["homology"] for row in source_eligible}
        ordinary = source_h_fractions[source_label]
        weighted = source_h_weighted_fractions[source_label]
        largest_h = source_largest_h[source_label]
        for target_label in labels:
            target_nodes, target_eligible = by_label[target_label], eligible_by_label[target_label]
            status = directed_status(source_label, target_label)
            measured = status == "measured"
            supported = directed_support[(source_label, target_label)]
            sources = {source for source, _, _ in supported}
            targets = {target for _, target, _ in supported}
            source_counts = Counter(indexed[source]["homology"] for source, _, _ in supported)
            target_counts = Counter(indexed[target]["homology"] for _, target, _ in supported)
            h_equal = _mean(values.get(target_label, 0.0) for values in ordinary.values()) if measured else None
            after_drop = _mean(values.get(target_label, 0.0) for h, values in ordinary.items()
                               if h != largest_h) if measured else None
            row = {
                "source_label": source_label, "target_label": target_label, "status": status,
                "n_source_domains": len(source_nodes), "n_source_eligible": len(source_eligible),
                "n_source_measured": len(source_active), "n_source_h": len(source_h),
                "n_source_eligible_h": len(source_eligible_h), "n_source_measured_h": len(ordinary),
                "n_source_isolated": len(source_eligible) - len(source_active),
                "n_target_domains": len(target_nodes), "n_target_eligible": len(target_eligible),
                "n_target_h": len({item["homology"] for item in target_nodes}),
                "n_target_measured": len(active_by_label[target_label]),
                "source_domain_coverage": len(source_active) / len(source_eligible) if source_eligible else None,
                "source_h_coverage": len(ordinary) / len(source_eligible_h) if source_eligible_h else None,
                "target_prevalence": len(target_nodes) / len(indexed) if indexed else None,
                "target_eligible_prevalence": len(target_eligible) / n_eligible if n_eligible else None,
                "domain_equal_fraction": natural_fractions[source_label].get(target_label, 0.0) if measured else None,
                "source_h_equal_fraction": h_equal,
                "target_h_reweighted_source_h_equal_fraction": _mean(values.get(target_label, 0.0)
                    for values in weighted.values()) if measured else None,
                "n_directed_edges": len(supported) if measured else None,
                "n_support_source_domains": len(sources) if measured else None,
                "n_support_target_domains": len(targets) if measured else None,
                "n_support_source_h": len(source_counts) if measured else None,
                "n_support_target_h": len(target_counts) if measured else None,
                "n_support_source_pdb": len({indexed[node]["pdb_id"] for node in sources if indexed[node]["pdb_id"]}) if measured else None,
                "n_support_target_pdb": len({indexed[node]["pdb_id"] for node in targets if indexed[node]["pdb_id"]}) if measured else None,
                "n_support_source_missing_pdb": sum(not indexed[node]["pdb_id"] for node in sources) if measured else None,
                "n_support_target_missing_pdb": sum(not indexed[node]["pdb_id"] for node in targets) if measured else None,
                "max_source_h_edge_fraction": max(source_counts.values()) / len(supported) if supported else None,
                "max_target_h_edge_fraction": max(target_counts.values()) / len(supported) if supported else None,
                "largest_source_h": largest_h,
                "drop_largest_source_h_fraction": after_drop,
                "drop_largest_source_h_delta": after_drop - h_equal if after_drop is not None and h_equal is not None else None,
                "distance_median": statistics.median(distance for _, _, distance in supported) if supported else None,
            }
            directed.append(row)

    undirected = []
    for i, left in enumerate(labels):
        for right in labels[i:]:
            supported = undirected_support[(left, right)]
            support_nodes = {node for source, target, _ in supported for node in (source, target)}
            h_edge_counts = Counter()
            for source, target, _ in supported:
                # One edge counts once per incident H, even when both nodes share H.
                h_edge_counts.update({indexed[source]["homology"], indexed[target]["homology"]})
            available = bool(eligible_by_label[left] and eligible_by_label[right])
            status = ("label_absent" if not by_label[left] or not by_label[right] else
                      "geometry_unavailable" if not available else
                      "observed_edges" if supported else "no_observed_edges")
            undirected.append({
                "label_1": left, "label_2": right, "status": status,
                "n_edges": len(supported) if available else None,
                "n_support_domains": len(support_nodes) if available else None,
                "n_support_h": len(h_edge_counts) if available else None,
                "n_support_pdb": len({indexed[node]["pdb_id"] for node in support_nodes if indexed[node]["pdb_id"]}) if available else None,
                "n_support_missing_pdb": sum(not indexed[node]["pdb_id"] for node in support_nodes) if available else None,
                "max_h_edge_fraction": max(h_edge_counts.values()) / len(supported) if supported else None,
                "distance_median": statistics.median(distance for _, _, distance in supported) if supported else None,
                "distance_min": min((distance for _, _, distance in supported), default=None),
                "distance_max": max((distance for _, _, distance in supported), default=None),
            })
    return {
        "directed": directed, "per_domain": per_domain, "undirected": undirected,
        "summary": {
            "schema_version": "geometry-relationship-overlay-v1", "label_column": label_column,
            "labels": labels, "n_phase_nodes": len(indexed), "n_phase_h": len(family_counts),
            "n_eligible_nodes": n_eligible, "n_measured_nodes": n_active,
            "n_isolated_eligible_nodes": n_eligible - n_active,
            "n_unavailable_nodes": len(indexed) - n_eligible,
            "n_input_edge_rows": len(edge_rows), "n_unique_short_mutual_edges": len(unique_edges),
            "edge_filter_counts": dict(edge_filter),
            "neighbor_share": "Each measured source domain contributes a normalized neighbor-label distribution",
            "source_h_equal": "Mean within each measured source H, then equal mean over measured source H",
            "target_h_weight": "Each neighbor has weight 1 / its H node count among ALL supplied phase nodes",
            "missing_policy": "Unavailable geometry and no-neighbor shares are NA; unsupported source/target labels are NA",
            "zero_policy": "Supported targets absent from a measurable source neighborhood have observed zero share",
            "prevalence_denominators": "All phase nodes and all geometrically eligible phase nodes are reported separately",
            "largest_source_h": "Largest phase source-label H by domain count; lexical tie break; delete it from H-equal mean",
            "support_concentration": "Maximum H contribution is an edge fraction, not an independent-sample fraction",
            "undirected_h_edge_fraction": "A domain pair counts once for each distinct H incident to it",
            "inference_scope": "Descriptive label-conditioned graph overlay; no independent-pair CI or p-values",
            "weight_semantics": "Sampling-balance weights; not physical similarity",
        },
    }
