"""Exact pairing/order patterns and observed shape ranges at fixed connectivity.

K_pair preserves strand pairing and orientation but ignores polymer order.
K_chain additionally preserves polymer direction, chain/fragment membership and
observed path status. Pattern identity is established by attributed isomorphism,
never by a graph hash alone. These graphs are not full protein topology labels.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import isomorphism

from .cohorts import A, H, T
from .io import read_bytes, read_csv, read_json, write_json
from .packing import PACKING_BLOCKS, prepare_packing_features

P_COLUMNS = tuple(column for columns in PACKING_BLOCKS.values() for column in columns)
KNOWN_SENSES = {"parallel", "anti-parallel"}
COMPLETE_STATUSES = {"available", "no_retained_pairings"}
EDGE_FIELDS = ("kind", "pair_sense", "polymer_order", "sequence_context", "observed_path")
SELECTION_RULE = ("At most three K_chain patterns, using only graph-primary domains with all 18 applicable P values finite; "
                  ">=2 H and >=6 domains in that joint-eligible subset, ordered by H support then domain support then pattern ID. "
                  "P values, variance, A/T labels and alignment outcomes do not rank patterns.")


@dataclass
class GraphForms:
    pair: nx.Graph
    chain: nx.DiGraph
    status: dict[str, Any]


def load_graph_records(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read existing compressed graph rows with bounded mounted-drive reads."""
    content = read_bytes(path)
    stream = gzip.GzipFile(fileobj=io.BytesIO(content)) if content.startswith(b"\x1f\x8b") else io.BytesIO(content)
    rows = {}
    with io.TextIOWrapper(stream, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row["record_id"])
            if key in rows:
                raise ValueError(f"Duplicate graph record: {key}")
            rows[key] = row
    return rows


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("graph", record)


def _edge_attributes(**updates: Any) -> dict[str, Any]:
    return {"kind": "strand_relation", "pair_sense": "none", "polymer_order": False,
            "sequence_context": "none", "observed_path": "none"} | updates


def build_graph_forms(record: Mapping[str, Any]) -> GraphForms:
    """Build ID-independent graph representations, retaining unknown states."""
    raw = _payload(record)
    nodes = list(raw.get("nodes", []))
    node_ids = [str(node["node_id"]) for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Strand identifiers must be unique within a domain")
    pair, chain = nx.Graph(), nx.DiGraph()
    for node_id in node_ids:
        pair.add_node(node_id, kind="strand")
        chain.add_node(node_id, kind="strand")
    unknown_senses = 0
    for edge in raw.get("pairing_edges", []):
        left, right = str(edge["source"]), str(edge["target"])
        if left == right or left not in pair or right not in pair:
            raise ValueError("Pairing edge has invalid endpoints")
        sense = str(edge.get("sense", "unknown"))
        if pair.has_edge(left, right):
            raise ValueError("Expected one collapsed pairing edge per strand pair")
        unknown_senses += sense not in KNOWN_SENSES
        pair.add_edge(left, right, sense=sense)
        for source, target in ((left, right), (right, left)):
            chain.add_edge(source, target, **_edge_attributes(pair_sense=sense))

    metadata_complete = True
    chain_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        node_id = str(node["node_id"])
        chain_id, fragment = str(node.get("chain_id", "")), str(node.get("fragment_id", ""))
        metadata_complete &= bool(chain_id and fragment and node.get("polymer_strand_rank") is not None)
        if chain_id:
            auxiliary = ("chain", chain_id)
            chain.add_node(auxiliary, kind="chain")
            chain.add_edge(auxiliary, node_id, **_edge_attributes(kind="chain_member"))
            chain_members[chain_id].append(node)
        if chain_id and fragment:
            auxiliary = ("fragment", chain_id, fragment)
            chain.add_node(auxiliary, kind="fragment")
            chain.add_edge(auxiliary, node_id, **_edge_attributes(kind="fragment_member"))

    expected_order = set()
    for members in chain_members.values():
        if any(node.get("polymer_strand_rank") is None for node in members):
            metadata_complete = False
            continue
        ordered = sorted(members, key=lambda node: int(node["polymer_strand_rank"]))
        ranks = [int(node["polymer_strand_rank"]) for node in ordered]
        metadata_complete &= ranks == list(range(len(ordered)))
        expected_order.update((str(left["node_id"]), str(right["node_id"]))
                              for left, right in zip(ordered, ordered[1:], strict=False))
    actual_order = set()
    order_attributes_known = True
    for edge in raw.get("sequence_relations", []):
        left, right = str(edge["source"]), str(edge["target"])
        if left == right or left not in pair or right not in pair or (left, right) in actual_order:
            raise ValueError("Polymer-order edge has invalid or duplicate endpoints")
        actual_order.add((left, right))
        context = str(edge.get("sequence_context", "unknown"))
        observed = edge.get("observed_chain_path")
        known = isinstance(observed, bool) and context in {"same_fragment", "cross_fragment"}
        order_attributes_known &= known
        attributes = dict(chain.get_edge_data(left, right) or _edge_attributes())
        attributes.update(polymer_order=True, sequence_context=context,
                          observed_path=str(observed).lower() if isinstance(observed, bool) else "unknown")
        chain.add_edge(left, right, **attributes)
    order_complete = bool(metadata_complete and order_attributes_known and actual_order == expected_order)
    source_status = str(record.get("status", "ok"))
    pairing_status = str(raw.get("pairing_status", "unavailable"))
    complete = source_status == "ok" and pairing_status in COMPLETE_STATUSES and bool(nodes)
    status = {
        "source_status": source_status, "pairing_status": pairing_status,
        "n_strands": len(nodes), "n_pairing_edges": pair.number_of_edges(),
        "n_unknown_pairing_senses": unknown_senses, "polymer_order_complete": order_complete,
        "pair_primary_eligible": bool(complete and unknown_senses == 0),
        "chain_primary_eligible": bool(complete and unknown_senses == 0 and order_complete),
        "partial_pairing": pairing_status == "partial_clipped_pairings",
    }
    return GraphForms(pair, chain, status)


def _matcher(left: nx.Graph, right: nx.Graph):
    node_match = isomorphism.categorical_node_match(["kind", "mapping_anchor"], ["", None])
    if left.is_directed() != right.is_directed():
        raise ValueError("Cannot compare pair and chain graph kinds")
    if left.is_directed():
        return isomorphism.DiGraphMatcher(left, right, node_match=node_match,
                                         edge_match=isomorphism.categorical_edge_match(EDGE_FIELDS, [None] * len(EDGE_FIELDS)))
    return isomorphism.GraphMatcher(left, right, node_match=node_match,
                                    edge_match=isomorphism.categorical_edge_match("sense", None))


def _bucket(graph: nx.Graph) -> tuple[Any, ...]:
    """Use WL only to avoid unnecessary expensive exact comparisons."""
    encoded = graph.copy()
    for _node, attributes in encoded.nodes(data=True):
        attributes["iso_node"] = attributes["kind"]
    for _left, _right, attributes in encoded.edges(data=True):
        attributes["iso_edge"] = repr(tuple(attributes.get(field) for field in EDGE_FIELDS)) if graph.is_directed() else attributes["sense"]
    return (graph.number_of_nodes(), graph.number_of_edges(),
            nx.weisfeiler_lehman_graph_hash(encoded, node_attr="iso_node", edge_attr="iso_edge", iterations=3))


def compare_graph_mapping(
    left_graph: Mapping[str, Any], right_graph: Mapping[str, Any], strand_mapping: Mapping[str, str],
) -> dict[str, Any]:
    """Check the supplied spatial strand mapping, without substituting an isomorphism.

    Partial one-to-one mappings receive induced-subgraph summaries, but full
    compatibility remains unknown. Unknown pairing/chain states also remain NA.
    Multiple source strands mapped to one target are explicitly ambiguous.
    """
    left, right = build_graph_forms(left_graph), build_graph_forms(right_graph)
    mapping = {str(key): str(value) for key, value in strand_mapping.items()}
    if not set(mapping) <= set(left.pair) or not set(mapping.values()) <= set(right.pair):
        raise ValueError("Spatial strand mapping refers to absent graph nodes")
    result: dict[str, Any] = {
        "abstract_pair_isomorphic": _matcher(left.pair, right.pair).is_isomorphic() if left.pair and right.pair else None,
        "abstract_chain_isomorphic": _matcher(left.chain, right.chain).is_isomorphic() if left.pair and right.pair else None,
        "query_strands": len(left.pair), "target_strands": len(right.pair),
        "mapped_query_strands": len(mapping), "mapped_target_strands": len(set(mapping.values())),
        "mapping_complete": False, "pair_mapping_compatible": None, "chain_mapping_compatible": None,
        "mapped_pair_edge_precision": None, "mapped_pair_edge_recall": None,
        "mapped_pair_sense_consistent": None, "status": "partial_mapping",
        "query_pair_primary_eligible": left.status["pair_primary_eligible"],
        "target_pair_primary_eligible": right.status["pair_primary_eligible"],
        "query_chain_primary_eligible": left.status["chain_primary_eligible"],
        "target_chain_primary_eligible": right.status["chain_primary_eligible"],
    }
    if len(set(mapping.values())) != len(mapping):
        result["status"] = "ambiguous_many_to_one_mapping"
        return result
    complete = bool(left.pair and set(mapping) == set(left.pair) and set(mapping.values()) == set(right.pair))
    result["mapping_complete"] = complete
    query_edges = {(mapping[a], mapping[b]): attributes["sense"]
                   for a, b, attributes in left.pair.subgraph(mapping).edges(data=True)}
    target = right.pair.subgraph(set(mapping.values()))
    preserved = sum(target.has_edge(a, b) for a, b in query_edges)
    checked_senses = [(sense, target[a][b]["sense"]) for (a, b), sense in query_edges.items() if target.has_edge(a, b)]
    known = all(a in KNOWN_SENSES and b in KNOWN_SENSES for a, b in checked_senses)
    result.update(
        mapped_query_pair_edges=len(query_edges), mapped_target_pair_edges=target.number_of_edges(),
        mapped_preserved_pair_edges=preserved,
        mapped_pair_edge_precision=preserved / len(query_edges) if query_edges else None,
        mapped_pair_edge_recall=preserved / target.number_of_edges() if target.number_of_edges() else None,
        mapped_pair_sense_consistent=all(a == b for a, b in checked_senses) if checked_senses and known else None,
    )
    if complete:
        result["status"] = "complete_mapping"
        if left.status["pair_primary_eligible"] and right.status["pair_primary_eligible"]:
            result["pair_mapping_compatible"] = bool(preserved == len(query_edges) == target.number_of_edges()
                                                     and all(a == b for a, b in checked_senses))
        else:
            result["status"] = "complete_mapping_pairing_unavailable_or_partial"
        if left.status["chain_primary_eligible"] and right.status["chain_primary_eligible"]:
            first, second = left.chain.copy(), right.chain.copy()
            for index, (source, destination) in enumerate(sorted(mapping.items())):
                first.nodes[source]["mapping_anchor"] = index
                second.nodes[destination]["mapping_anchor"] = index
            result["chain_mapping_compatible"] = _matcher(first, second).is_isomorphic()
    return result


def _assign_patterns(forms: dict[str, GraphForms], kind: str) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[Any, ...], list[tuple[str, str, nx.Graph]]] = defaultdict(list)
    assignments, prototypes, mappings = {}, [], []
    for record_id in sorted(forms):
        graph = getattr(forms[record_id], kind)
        if not forms[record_id].pair:
            assignments[record_id] = ""
            continue
        bucket = _bucket(graph)
        matched = None
        for pattern_id, reference_id, representative in buckets[bucket]:
            matcher = _matcher(graph, representative)
            if matcher.is_isomorphic():
                matched = (pattern_id, reference_id, matcher.mapping)
                break
        if matched is None:
            pattern_id = f"K{kind}-p{len(prototypes) + 1:05d}"
            buckets[bucket].append((pattern_id, record_id, graph))
            prototypes.append({"pattern_id": pattern_id, "kind": f"K_{kind}", "representative_id": record_id})
            matched = (pattern_id, record_id, {node: node for node in graph})
        pattern_id, reference_id, correspondence = matched
        assignments[record_id] = pattern_id
        mappings.append({"record_id": record_id, "kind": f"K_{kind}", "pattern_id": pattern_id,
                         "representative_id": reference_id,
                         "strand_mapping_to_representative": {str(source): str(target) for source, target in correspondence.items()
                                                              if graph.nodes[source]["kind"] == "strand"}})
    return assignments, prototypes, mappings


def _h_balanced_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.sort_values("record_id")
    rng = np.random.default_rng(seed)
    groups = {h: list(rng.permutation(part.sort_values("record_id").index.to_numpy())) for h, part in frame.groupby(H, sort=True)}
    families = list(rng.permutation(sorted(groups)))
    selected = []
    while len(selected) < maximum:
        progressed = False
        for family in families:
            if groups[family] and len(selected) < maximum:
                selected.append(groups[family].pop())
                progressed = True
        if not progressed:
            break
    return frame.loc[selected].sort_values("record_id")


def _pattern_distributions(data: pd.DataFrame, fields: tuple[str, ...], group_column: str = "chain_pattern") -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, h_rows = [], []
    for pattern, members in data.groupby(group_column, sort=True):
        for field in fields:
            numeric = pd.to_numeric(members[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
            measured = pd.DataFrame({H: members[H], "value": numeric})
            families = measured.groupby(H).agg(n_total=("value", "size"), n_observed=("value", "count"), mean=("value", "mean"))
            observed = numeric.dropna()
            summaries.append({"pattern_id": pattern, "measurement": field, "n_total": len(members),
                              "n_observed": len(observed), "n_missing": len(members) - len(observed),
                              "H_total": len(families), "H_observed": int(families.n_observed.gt(0).sum()),
                              "median": observed.median(), "q25": observed.quantile(.25), "q75": observed.quantile(.75),
                              "minimum": observed.min(), "maximum": observed.max(), "H_equal_mean": families["mean"].mean()})
            h_rows.extend({"pattern_id": pattern, "measurement": field, H: h, **row}
                          for h, row in families.to_dict("index").items())
    return pd.DataFrame(summaries), pd.DataFrame(h_rows)


def _shape_selection(domains: pd.DataFrame, summary: pd.DataFrame, seed: int, maximum_patterns: int) -> dict[str, pd.DataFrame]:
    domains = domains.copy()
    domains["packing_complete_eligible"] = np.isfinite(domains[list(P_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    eligible = domains.loc[domains.chain_primary_eligible & domains.packing_complete_eligible]
    support = eligible.groupby("chain_pattern").agg(primary_complete_P_domains=("record_id", "size"), primary_complete_P_H=(H, "nunique")).reset_index()
    candidates = summary.loc[summary.kind.eq("K_chain")].merge(support, left_on="pattern_id", right_on="chain_pattern", how="left")
    selected = candidates.loc[candidates.primary_complete_P_H.ge(2) & candidates.primary_complete_P_domains.ge(6)].sort_values(
        ["primary_complete_P_H", "primary_complete_P_domains", "pattern_id"], ascending=[False, False, True]
    ).head(maximum_patterns).copy()
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    population = eligible.loc[eligible.chain_pattern.isin(selected.pattern_id)].copy()
    requests, series = [], []
    request_fields = [column for column in ("record_id", "pdb_id", A, T, H, "role", "chain_pattern") if column in domains]
    for pattern, members in population.groupby("chain_pattern", sort=True):
        sampled = _h_balanced_sample(members, 128, seed) if len(members) > 1000 else members.sort_values("record_id")
        for row in sampled[request_fields].to_dict("records"):
            requests.append(row | {"pattern_population_n": len(members), "pattern_requested_n": len(sampled),
                                   "sampling": "H_balanced_max128" if len(members) > 1000 else "all_joint_eligible_pattern_domains"})
        axis = "betlas_pca_elongation"
        available = sampled.loc[sampled[axis].notna()].copy()
        used = set()
        for quantile in (0., .25, .5, .75, 1.):
            if len(used) == len(available):
                break
            target = available[axis].quantile(quantile)
            candidates = available.loc[~available.record_id.isin(used)].copy()
            candidates["delta"] = (candidates[axis] - target).abs()
            row = candidates.sort_values(["delta", "record_id"]).iloc[0]
            used.add(row.record_id)
            series.append({"pattern_id": pattern, "record_id": row.record_id, H: row[H], "axis": axis,
                           "axis_value": row[axis], "target_quantile": quantile,
                           "interpretation": "Observed structures ordered by a fixed shape coordinate; not a physical trajectory."})
    p_summary, p_h = _pattern_distributions(population, P_COLUMNS)
    return {"selected-patterns.csv": selected, "selected-pattern-population.csv": population,
            "coordinate-requests.csv": pd.DataFrame(requests, columns=request_fields + ["pattern_population_n", "pattern_requested_n", "sampling"]),
            "series-representatives.csv": pd.DataFrame(series, columns=["pattern_id", "record_id", H, "axis", "axis_value", "target_quantile", "interpretation"]),
            "pattern-P-distributions.csv": p_summary, "pattern-P-per-H.csv": p_h}


def run_connection_patterns(
    cohort_csv: str | Path | pd.DataFrame, graphs_jsonl_gz: str | Path,
    output_dir: str | Path, *, seed: int = 13, maximum_patterns: int = 3,
) -> dict[str, Any]:
    """Build the full catalogue and select bounded geometry requests by support."""
    cohort = cohort_csv.copy() if isinstance(cohort_csv, pd.DataFrame) else read_csv(cohort_csv, dtype=str, keep_default_na=False)
    if cohort.record_id.duplicated().any() or cohort.record_id.isna().any():
        raise ValueError("Cohort requires unique, nonmissing record_id")
    raw = load_graph_records(graphs_jsonl_gz)
    if set(cohort.record_id) != set(raw):
        raise ValueError("Raw graph identities must equal the complete cohort")
    forms = {record_id: build_graph_forms(record) for record_id, record in raw.items()}
    pair, pair_prototypes, pair_mappings = _assign_patterns(forms, "pair")
    chain, chain_prototypes, chain_mappings = _assign_patterns(forms, "chain")
    metadata = [column for column in ("record_id", "pdb_id", A, T, H, "role", "feature_status") if column in cohort]
    domains = cohort[metadata].copy()
    for key in next(iter(forms.values())).status:
        domains[key] = domains.record_id.map({record_id: value.status[key] for record_id, value in forms.items()})
    domains["pair_pattern"] = domains.record_id.map(pair)
    domains["chain_pattern"] = domains.record_id.map(chain)
    physical = prepare_packing_features(cohort)
    for column in P_COLUMNS:
        domains[column] = physical[column].to_numpy()
    summaries = []
    for kind in ("pair", "chain"):
        for pattern, part in domains.loc[domains[f"{kind}_pattern"].ne("")].groupby(f"{kind}_pattern", sort=True):
            primary = part.loc[part[f"{kind}_primary_eligible"]]
            summaries.append({"pattern_id": pattern, "kind": f"K_{kind}", "retained_domains": len(part),
                              "primary_domains": len(primary), "primary_H": primary[H].nunique(),
                              "primary_A": primary[A].nunique(), "primary_T": primary[T].nunique(),
                              "primary_PDB": primary.pdb_id.nunique() if "pdb_id" in primary else None,
                              "n_strands": part.n_strands.iloc[0], "n_pairing_edges": part.n_pairing_edges.iloc[0],
                              "partial_domains": int(part.partial_pairing.sum()),
                              "unknown_sense_domains": int(part.n_unknown_pairing_senses.gt(0).sum())})
    summary = pd.DataFrame(summaries)
    selection_tables = _shape_selection(domains, summary, seed, maximum_patterns)
    selected = selection_tables["selected-patterns.csv"]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "domain-patterns.csv").exists():
        raise FileExistsError("Pattern catalogue already exists; use a new output directory")
    tables = {"domain-patterns.csv": domains, "pattern-summary.csv": summary} | selection_tables
    for name, table in tables.items():
        table.to_csv(destination / name, index=False)
    with gzip.open(destination / "pattern-mappings.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in pair_mappings + chain_mappings:
            handle.write(json.dumps(row) + "\n")
    with gzip.open(destination / "pattern-prototypes.jsonl.gz", "wt", encoding="utf-8") as handle:
        for prototype in pair_prototypes + chain_prototypes:
            handle.write(json.dumps(prototype | {"raw_graph": _payload(raw[prototype["representative_id"]])}) + "\n")
    protocol = {
        "schema": "betlas.connection-patterns.v1", "status": "catalogue_complete", "domains": len(domains),
        "exact_comparison": "Attributed VF2 isomorphism; WL hashes only prefilter candidates, never establish equality.",
        "K_pair": "Undirected strand graph with parallel/anti-parallel pairing attributes; no polymer order, geometry, size or CATH labels.",
        "K_chain": "Directed multiplex strand graph plus anonymous chain/fragment membership nodes; N-to-C adjacency, fragment crossing and observed path retained. IDs and loop lengths are not equality attributes.",
        "primary": "Source status ok, nonempty graph, complete retained pairing, all pairing senses known; K_chain additionally needs complete chain/fragment/order metadata.",
        "retained": "Partial or unknown graphs remain in the catalogue with status and are excluded from primary pattern support and shape selection.",
        "selection": SELECTION_RULE,
        "shape_sampling": "All joint graph/P-eligible domains in selected patterns up to 1000 per pattern; above 1000, round-robin H-balanced sample of at most128, seed fixed. All-joint-eligible-member P distributions remain available.",
        "series": "Five observed representatives at fixed elongation quantiles of the requested geometry sample, with unique IDs; no interpolation and no replacement for pattern distributions.",
        "mapping": "Abstract isomorphism and compatibility with a supplied spatial strand mapping are separate. Partial spatial maps do not establish full-graph compatibility.",
        "seed": seed, "networkx_version": nx.__version__, "selected_patterns": selected.pattern_id.tolist(),
        "artifacts": {name: len(table) for name, table in tables.items()},
    }
    write_json(destination / "catalogue-protocol.json", protocol)
    return protocol


def select_connection_series(output_dir: str | Path, *, seed: int = 13) -> dict[str, Any]:
    """Apply joint P/K eligibility while preserving the earlier graph-only attempt."""
    destination = Path(output_dir)
    domains = read_csv(destination / "domain-patterns.csv", dtype={A: str, T: str, H: str})
    summary = read_csv(destination / "pattern-summary.csv")
    tables = _shape_selection(domains, summary, seed, 3)
    archive = destination / "attempt01-graph-only"
    archive.mkdir(exist_ok=False)
    for name in (*tables, "catalogue-protocol.json"):
        if (destination / name).exists():
            shutil.copy2(destination / name, archive / name)
    for name, table in tables.items():
        table.to_csv(destination / name, index=False)
    protocol = read_json(destination / "catalogue-protocol.json")
    protocol.update(selection=SELECTION_RULE, selection_revision="joint_P18_K_eligibility",
                    selection_revision_reason="Two/three-strand graph-only patterns lack a complete angular block; joint physical applicability is required before shape study, without ranking by values or variation.",
                    previous_selection_archive=str(archive), selected_patterns=tables["selected-patterns.csv"].pattern_id.tolist())
    protocol["artifacts"].update({name: len(table) for name, table in tables.items()})
    write_json(destination / "catalogue-protocol.json", protocol)
    return protocol


def refresh_pattern_features(
    cohort_csv: str | Path | pd.DataFrame, output_dir: str | Path, *, variant: str,
    blocks: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Describe revised physical measurements on unchanged pattern membership.

    This never changes graph patterns, selected members, coordinate requests or
    series representatives. Original feature tables remain intact.
    """
    if not variant or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in variant):
        raise ValueError("A simple nonempty measurement variant name is required")
    destination = Path(output_dir)
    cohort = cohort_csv.copy() if isinstance(cohort_csv, pd.DataFrame) else read_csv(cohort_csv, dtype=str, keep_default_na=False)
    domains = read_csv(destination / "domain-patterns.csv", dtype={A: str, T: str, H: str})
    if cohort.record_id.duplicated().any() or set(cohort.record_id) != set(domains.record_id):
        raise ValueError("Revised features must retain all original cohort identities")
    blocks = dict(PACKING_BLOCKS if blocks is None else blocks)
    fields = tuple(field for columns in blocks.values() for field in columns)
    physical = prepare_packing_features(cohort, blocks)
    physical["record_id"] = cohort.record_id.to_numpy()
    revised = domains.drop(columns=list(P_COLUMNS)).merge(physical, on="record_id", validate="one_to_one")
    original_population = read_csv(destination / "selected-pattern-population.csv", dtype={H: str})
    population = revised.loc[revised.record_id.isin(original_population.record_id)]
    summary, by_h = _pattern_distributions(population, fields)
    artifacts = {f"domain-pattern-features-{variant}.csv": revised,
                 f"pattern-P-distributions-{variant}.csv": summary, f"pattern-P-per-H-{variant}.csv": by_h}
    for name, table in artifacts.items():
        if (destination / name).exists():
            raise FileExistsError(f"Measurement variant already exists: {name}")
        table.to_csv(destination / name, index=False)
    protocol = {"measurement_variant": variant, "domains": len(revised), "selected_population_n": len(population),
                "pattern_membership_changed": False, "original_tables_preserved": True,
                "measurement_blocks": blocks,
                "artifacts": {name: len(table) for name, table in artifacts.items()}}
    write_json(destination / f"feature-protocol-{variant}.json", protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("catalogue", "selection", "geometry"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--coordinate-dir")
    args = parser.parse_args()
    config = read_json(args.config)
    output = Path(config["outputs"]) / "connection-patterns"
    source = Path(config["previous_derived"])
    if args.stage == "catalogue":
        result = run_connection_patterns(source / "cohort.csv", source / "graphs.jsonl.gz", output, seed=int(config.get("seed", 13)))
    elif args.stage == "selection":
        result = select_connection_series(output, seed=int(config.get("seed", 13)))
    else:
        if not args.coordinate_dir:
            parser.error("geometry requires --coordinate-dir")
        from .connection_patterns_geometry import run_pattern_geometry
        result = run_pattern_geometry(source / "graphs.jsonl.gz", args.coordinate_dir, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
