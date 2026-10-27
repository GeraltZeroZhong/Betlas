"""Summarize frozen packing overlays without changing graphs or candidates.

Ranks describe observed graphs. They are neither tests nor independent evidence.
The component sensitivity is an induced subgraph: no neighbor list is refitted.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json
from .relationships import summarize_relationships

PHASES = ("discovery", "confirmation", "all")
VIEWS = ("rms", "l1", "drop_global_shape", "drop_sheet_shape",
         "drop_axis_organization", "drop_angular_repetition")
LABEL_DTYPES = {A: str, H: str, T: str, "mother_h_pdb_component": str,
                "a": str, "b": str, "support_h": str,
                "source_label": str, "target_label": str, "label_1": str,
                "label_2": str, "largest_source_h": str, "homology": str}


def read_table(path):
    return read_csv(path, dtype=LABEL_DTYPES, low_memory=False)


def pair_key(a, b):
    return "--".join(sorted((str(a), str(b))))


def contexts():
    for phase in PHASES:
        for view in VIEWS:
            yield phase, view, 30, "q75"
        for k, cut in ((10, "q75"), (60, "q75"), (30, "q50"), (30, "q90")):
            yield phase, "rms", k, cut


def context_columns(phase, view, k, cut):
    return {"phase": phase, "view": view, "k": k, "cut": cut}


def relation_directory(root, view, k, cut):
    suffix = "" if cut == "q75" else "-" + cut
    return root / view / f"relations-k{k}{suffix}"


def support_status(row):
    """Apply the frozen discovery edge/H floor, without a significance claim."""
    if row.get("status") != "measured":
        return "unavailable"
    if row["n_directed_edges"] == 0:
        return "no_observed_edges"
    if min(row["n_support_source_h"], row["n_support_target_h"]) < 2:
        return "insufficient_support_h"
    if row["n_directed_edges"] < 10:
        return "insufficient_edges"
    return "meets_edge_h_floor"


def enrich_directions(directions, nodes, label_column):
    result = directions.copy()
    eligible = nodes.loc[nodes.packing_eligible.eq(True)]
    prevalence = eligible.groupby(label_column)[H].nunique() / eligible[H].nunique()
    result["pair_id"] = [pair_key(a, b) for a, b in zip(result.source_label, result.target_label, strict=True)]
    result["target_eligible_h_prevalence"] = result.target_label.map(prevalence)
    result["descriptive_target_h_lift"] = (
        result.target_h_reweighted_source_h_equal_fraction / result.target_eligible_h_prevalence)
    result["support_status"] = [support_status(row) for row in result.to_dict("records")]
    adequate = result.support_status.eq("meets_edge_h_floor")
    for column in ("source_h_equal_fraction", "target_h_reweighted_source_h_equal_fraction",
                   "descriptive_target_h_lift"):
        result["supported_" + column] = result[column].where(adequate)
    return result


def descriptive_ranking(directions, labels):
    """Rank only unordered pairs from the same label level; preserve absent pairs."""
    indexed = directions.set_index(["source_label", "target_label"])
    records = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            # Match the frozen lexical pair ID so side-specific support is unambiguous.
            a, b = sorted((left, right))
            ab = indexed.loc[(a, b)] if (a, b) in indexed.index else None
            ba = indexed.loc[(b, a)] if (b, a) in indexed.index else None
            scores = [x.descriptive_target_h_lift if x is not None else np.nan for x in (ab, ba)]
            score = min(scores) if np.isfinite(scores).all() else np.nan
            row = {"a": a, "b": b, "pair_id": pair_key(a, b),
                   "descriptive_min_lift": score,
                   "n_edges": ab.n_directed_edges if ab is not None else np.nan,
                   "support_h_a": ab.n_support_source_h if ab is not None else np.nan,
                   "support_h_b": ab.n_support_target_h if ab is not None else np.nan,
                   "support_status": ab.support_status if ab is not None else "unavailable",
                   "a_to_b_h_equal": ab.source_h_equal_fraction if ab is not None else np.nan,
                   "b_to_a_h_equal": ba.source_h_equal_fraction if ba is not None else np.nan,
                   "a_to_b_target_h_weighted": ab.target_h_reweighted_source_h_equal_fraction if ab is not None else np.nan,
                   "b_to_a_target_h_weighted": ba.target_h_reweighted_source_h_equal_fraction if ba is not None else np.nan,
                   "a_to_b_lift": scores[0], "b_to_a_lift": scores[1]}
            records.append(row)
    table = pd.DataFrame(records)
    # Shared scores share ranks. Edge counts are display tie breaks only.
    table["descriptive_lift_rank"] = table.descriptive_min_lift.rank(ascending=False, method="min")
    sufficient = table.support_status.eq("meets_edge_h_floor")
    table["supported_lift_rank"] = table.descriptive_min_lift.where(sufficient).rank(ascending=False, method="min")
    return table.sort_values(["descriptive_min_lift", "n_edges", "pair_id"], ascending=[False, False, True])


def coverage_table(directions, nodes, labels):
    columns = ["n_source_domains", "n_source_eligible", "n_source_measured", "n_source_h",
               "n_source_eligible_h", "n_source_measured_h", "n_source_isolated",
               "source_domain_coverage", "source_h_coverage"]
    indexed = directions.drop_duplicates("source_label").set_index("source_label")
    rows = []
    for label in labels:
        row = {A: label}
        row.update(indexed.loc[label, columns].to_dict())
        row["n_geometry_unavailable"] = row["n_source_domains"] - row["n_source_eligible"]
        row["n_pdb"] = nodes.loc[nodes[A].eq(label), "pdb_id"].nunique()
        row["coverage_status"] = ("absent" if row["n_source_domains"] == 0 else
                                  "no_geometry" if row["n_source_eligible"] == 0 else
                                  "no_short_mutual_neighbors" if row["n_source_measured"] == 0 else
                                  "single_measured_h" if row["n_source_measured_h"] < 2 else "measured")
        rows.append(row)
    return pd.DataFrame(rows)


def selected_edges(edges, cut=None):
    keep = edges.mutual.eq(True)
    keep &= edges.short_edge.eq(True) if cut is None else edges.distance.le(cut)
    return edges.loc[keep].copy()


def major_h_contributions(nodes, edges, label_column, pair_ids):
    """Identify all contributing H; counts describe edge incidence, not replication."""
    meta = nodes.set_index("record_id")
    selected = selected_edges(edges)
    # Domain-pair rows can be reciprocal; one oriented neighbor per source is used.
    pairs = set(tuple(sorted((row.source_id, row.target_id))) for row in selected.itertuples())
    oriented = []
    for left, right in pairs:
        oriented.extend(((left, right), (right, left)))
    neighbor_count = Counter(source for source, _ in oriented)
    active = meta.loc[list(neighbor_count)] if neighbor_count else meta.iloc[:0]
    active_per_h = active.groupby(H).size().to_dict()
    active_h_per_label = active.groupby(label_column)[H].nunique().to_dict()
    h_count = meta.groupby(H).size().to_dict()
    target_weight_total = Counter()
    for source, target in oriented:
        target_weight_total[source] += 1 / h_count[meta.at[target, H]]
    accum = {}
    for source, target in oriented:
        a, b = meta.at[source, label_column], meta.at[target, label_column]
        if a == b or pair_key(a, b) not in pair_ids:
            continue
        sh, th = meta.at[source, H], meta.at[target, H]
        for side, h, record in (("source", sh, source), ("target", th, target)):
            key = a, b, side, h
            if key not in accum:
                accum[key] = {"edges": 0, "domains": set(), "h_equal_contribution": 0., "target_h_weighted_contribution": 0.}
            value = accum[key]
            value["edges"] += 1
            value["domains"].add(record)
            normalizer = active_per_h[sh] * active_h_per_label[a]
            value["h_equal_contribution"] += 1 / neighbor_count[source] / normalizer
            value["target_h_weighted_contribution"] += (1 / h_count[th]) / target_weight_total[source] / normalizer
    rows = []
    totals = Counter()
    for (a, b, side, _h), item in accum.items():
        totals[a, b, side] += item["edges"]
    for (a, b, side, h), item in accum.items():
        rows.append({"source_label": a, "target_label": b, "pair_id": pair_key(a, b),
                     "side": side, "support_h": h, "n_edges": item["edges"],
                     "edge_fraction": item["edges"] / totals[a, b, side],
                     "n_support_domains": len(item["domains"]),
                     "h_equal_contribution": item["h_equal_contribution"],
                     "target_h_weighted_contribution": item["target_h_weighted_contribution"]})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["edge_contribution_rank"] = result.groupby(["source_label", "target_label", "side"]).n_edges.rank(method="min", ascending=False)
    return result


def induced_component_deletion(nodes, edges):
    """Delete the largest observed full-mother H/PDB component and its edges."""
    counts = nodes.mother_h_pdb_component.value_counts()
    largest = sorted(counts.index, key=lambda component: (-counts[component], str(component)))[0]
    remaining = nodes.loc[~nodes.mother_h_pdb_component.eq(largest)].copy()
    ids = set(remaining.record_id)
    retained = edges.loc[edges.source_id.isin(ids) & edges.target_id.isin(ids)].copy()
    metadata = {"component": str(largest), "selection": "Largest S35 projection of full-mother H/PDB components; lexical tie break",
                "removed_nodes": int(counts[largest]), "original_nodes": len(nodes),
                "remaining_nodes": len(remaining), "original_edge_rows": len(edges),
                "remaining_edge_rows": len(retained),
                "estimand": "Induced subgraph of the original all-phase RMS k30 q75 graph; no kNN reconstruction or replacement edges",
                "weighting": "Recomputed H node-count weights and source-H means within remaining supplied nodes",
                "limitation": "Deletion sensitivity does not establish independence or correct other shared-component dependence"}
    return remaining, retained, metadata


def _markdown_table(frame):
    def fmt(value):
        if pd.isna(value):
            return "NA"
        return f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def run_summaries(config_path):
    started = time.time()
    cfg = read_json(config_path)
    derived, outputs = Path(cfg["derived"]), Path(cfg["outputs"])
    target = outputs / "atlas-analysis"
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    cohort = read_table(derived / "cohort.csv")
    labels = sorted(cohort[A].unique(), key=lambda value: tuple(map(int, value.split("."))))
    frozen = read_json(derived / "frozen-candidates.json")
    cuts = read_json(derived / "short-distance-cuts.json")
    candidate_ids = {pair_key(row["a"], row["b"]) for row in frozen["new_architecture_pairs"]}
    historical_id = pair_key("2.40", "2.60")
    pair_ids = candidate_ids | {historical_id}
    coverage, directions, candidates, ranks, contributions, strict_legacy, t_ranks = [], [], [], [], [], [], []
    raw_directions, node_cache, edge_cache = {}, {}, {}
    for phase, view, k, cut in contexts():
        folder = outputs / "atlas" / phase
        context = context_columns(phase, view, k, cut)
        if (phase, view) not in node_cache:
            node_cache[phase, view] = read_table(folder / view / "nodes.csv")
        nodes = node_cache[phase, view]
        rel = read_table(relation_directory(folder, view, k, cut) / "architecture-directed.csv")
        rel = enrich_directions(rel, nodes, A)
        raw_directions[phase, view, k, cut] = rel
        coverage.append(coverage_table(rel, nodes, labels).assign(**context))
        directions.append(rel.assign(**context))
        chosen = rel.loc[rel.pair_id.isin(pair_ids)].assign(**context)
        candidates.append(chosen)
        ranks.append(descriptive_ranking(rel, labels).assign(**context))
        if (phase, view, k) not in edge_cache:
            edge_cache[phase, view, k] = read_table(folder / view / f"edges-k{k}.csv.gz")
        edges = edge_cache[phase, view, k].copy()
        if cut != "q75":
            edges["short_edge"] = edges.distance.le(cuts[view][cut])
        contributions.append(major_h_contributions(nodes, edges, A, pair_ids).assign(**context))
        if view == "rms" and k == 30 and cut == "q75":
            for labelset in ("strict", "legacy"):
                rel = read_table(folder / view / f"{labelset}-directed.csv")
                strict_legacy.append(enrich_directions(rel, nodes, labelset + "_label").assign(phase=phase, labelset=labelset))
            rel = read_table(folder / view / "within-2.60-topology.csv")
            within = nodes.loc[nodes[A].eq("2.60")]
            rel = enrich_directions(rel, within, T)
            topologies = sorted(cohort.loc[cohort[A].eq("2.60"), T].unique(), key=lambda value: tuple(map(int, value.split("."))))
            t_ranks.append(descriptive_ranking(rel, topologies).assign(phase=phase, architecture="2.60", label_level="topology"))

    coverage = pd.concat(coverage, ignore_index=True)
    directions = pd.concat(directions, ignore_index=True)
    candidates = pd.concat(candidates, ignore_index=True)
    ranks = pd.concat(ranks, ignore_index=True)
    contributions = pd.concat(contributions, ignore_index=True)
    primary_ranks = []
    for phase in PHASES:
        primary = ranks.loc[ranks.phase.eq(phase) & ranks.view.eq("rms") & ranks.k.eq(30) & ranks.cut.eq("q75")].copy()
        stability = Counter()
        for view in VIEWS:
            rel = raw_directions[phase, view, 30, "q75"]
            supported = rel.loc[rel.n_support_source_h.ge(2) & rel.n_support_target_h.ge(2)]
            stability.update(set(supported.pair_id))
        primary["fixed_views_with_two_h_per_side"] = primary.pair_id.map(stability).fillna(0).astype(int)
        primary["frozen_selected_new_pair"] = primary.pair_id.isin(candidate_ids)
        primary["historical_architecture_pair"] = primary.pair_id.eq(historical_id)
        primary["meets_frozen_support_rule"] = primary.support_status.eq("meets_edge_h_floor") & primary.fixed_views_with_two_h_per_side.ge(3)
        primary = primary.sort_values(["fixed_views_with_two_h_per_side", "descriptive_min_lift", "n_edges", "pair_id"], ascending=[False, False, False, True])
        primary["phase_descriptive_rank"] = np.arange(1, len(primary) + 1)
        primary.loc[primary.descriptive_min_lift.isna(), "phase_descriptive_rank"] = np.nan
        primary_ranks.append(primary)
    primary_ranks = pd.concat(primary_ranks, ignore_index=True)
    artifacts = {"coverage-all-contexts": coverage, "architecture-directed-all-contexts": directions,
                 "fixed-pairs-all-contexts": candidates, "fixed-pairs-h-contributions": contributions,
                 "architecture-ranks-all-contexts": ranks, "architecture-primary-phase-ranking": primary_ranks,
                 "strict-legacy-directed": pd.concat(strict_legacy, ignore_index=True),
                 "within-2.60-topology-ranks": pd.concat(t_ranks, ignore_index=True)}
    artifacts["coverage-primary"] = coverage.loc[coverage.view.eq("rms") & coverage.k.eq(30) & coverage.cut.eq("q75")]
    artifacts["fixed-pairs-primary"] = primary_ranks.loc[primary_ranks.pair_id.isin(pair_ids)]
    artifacts["jelly-roll-within-2.60-topology-ranks"] = artifacts["within-2.60-topology-ranks"].loc[
        artifacts["within-2.60-topology-ranks"].a.eq("2.60.120") | artifacts["within-2.60-topology-ranks"].b.eq("2.60.120")]
    mapping = artifacts["strict-legacy-directed"]
    mapping_comparison = mapping.loc[mapping.labelset.eq("strict")].merge(
        mapping.loc[mapping.labelset.eq("legacy")],
        on=["phase", "source_label", "target_label", "pair_id"],
        suffixes=("_strict", "_legacy"), validate="one_to_one")
    for name in ("source_h_equal_fraction", "target_h_reweighted_source_h_equal_fraction"):
        mapping_comparison[name + "_legacy_minus_strict"] = (
            mapping_comparison[name + "_legacy"] - mapping_comparison[name + "_strict"])
    artifacts["strict-versus-legacy-comparison"] = mapping_comparison
    for name, table in artifacts.items():
        table.to_csv(target / f"{name}.csv", index=False)

    # The primary all-graph is filtered exactly once; this is not a new kNN graph.
    nodes, edges = node_cache["all", "rms"], edge_cache["all", "rms", 30]
    remaining, retained, deletion = induced_component_deletion(nodes, edges)
    sensitivity = target / "drop-largest-mother-component"
    sensitivity.mkdir()
    remaining.to_csv(sensitivity / "nodes.csv", index=False)
    retained.to_csv(sensitivity / "edges-k30.csv.gz", index=False)
    result = summarize_relationships(remaining, retained, A, all_labels=labels)
    for name in ("directed", "undirected", "per_domain"):
        pd.DataFrame(result[name]).to_csv(sensitivity / f"architecture-{name}.csv", index=False)
    write_json(sensitivity / "summary.json", result["summary"])
    changed = enrich_directions(pd.DataFrame(result["directed"]), remaining, A)
    changed.to_csv(sensitivity / "architecture-directed-with-support.csv", index=False)
    after_coverage = coverage_table(changed, remaining, labels)
    after_coverage.to_csv(sensitivity / "coverage.csv", index=False)
    before = raw_directions["all", "rms", 30, "q75"]
    comparison = before.merge(changed, on=["source_label", "target_label", "pair_id"], suffixes=("_full", "_after_deletion"), validate="one_to_one")
    for name in ("source_h_equal_fraction", "target_h_reweighted_source_h_equal_fraction", "descriptive_target_h_lift"):
        comparison[name + "_delta"] = comparison[name + "_after_deletion"] - comparison[name + "_full"]
        adequate = comparison.support_status_full.eq("meets_edge_h_floor") & comparison.support_status_after_deletion.eq("meets_edge_h_floor")
        comparison["supported_" + name + "_delta"] = comparison[name + "_delta"].where(adequate)
    comparison.to_csv(sensitivity / "full-versus-deletion.csv", index=False)
    comparison.loc[comparison.pair_id.isin(pair_ids)].to_csv(sensitivity / "fixed-pairs-full-versus-deletion.csv", index=False)
    before_coverage = coverage_table(before, nodes, labels)
    before_coverage.merge(after_coverage, on=A, suffixes=("_full", "_after_deletion"), validate="one_to_one").to_csv(sensitivity / "coverage-comparison.csv", index=False)
    removed = nodes.loc[nodes.mother_h_pdb_component.eq(deletion["component"])]
    deletion["removed_h"] = int(removed[H].nunique())
    deletion["removed_pdb"] = int(removed.pdb_id.nunique())
    deletion["removed_fraction"] = len(removed) / len(nodes)
    deletion["removed_mother_domain_weight"] = int(pd.to_numeric(removed.mother_s35_domain_count).sum())
    deletion["remaining_unique_short_mutual_edges"] = result["summary"]["n_unique_short_mutual_edges"]
    deletion["fixed_pairs_after_deletion"] = changed.loc[changed.pair_id.isin(pair_ids), ["source_label", "target_label", "support_status", "n_directed_edges", "n_support_source_h", "n_support_target_h"]].to_dict("records")
    write_json(sensitivity / "protocol.json", deletion)

    # Verify H contribution decompositions against their source relation tables.
    summed = contributions.groupby(["phase", "view", "k", "cut", "source_label", "target_label", "side"])[["h_equal_contribution", "target_h_weighted_contribution"]].sum().reset_index()
    checks = summed.merge(candidates, on=["phase", "view", "k", "cut", "source_label", "target_label"], validate="many_to_one")
    if not np.allclose(checks.h_equal_contribution, checks.source_h_equal_fraction, atol=1e-10, rtol=1e-10):
        raise AssertionError("Source-H contribution decomposition differs from frozen summary")
    if not np.allclose(checks.target_h_weighted_contribution, checks.target_h_reweighted_source_h_equal_fraction, atol=1e-10, rtol=1e-10):
        raise AssertionError("Target-H contribution decomposition differs from frozen summary")
    if len(ranks) != 30 * 210 or len(coverage) != 30 * 21:
        raise AssertionError("Incomplete context/label coverage")
    original = read_table(derived / "discovery-candidate-ranking.csv")
    original_pairs = [pair_key(row.a, row.b) for row in original.itertuples() if row.selected]
    reconstructed = primary_ranks.loc[primary_ranks.phase.eq("discovery") & primary_ranks.meets_frozen_support_rule & ~primary_ranks.historical_architecture_pair].head(3).pair_id.tolist()
    if original_pairs != reconstructed:
        raise AssertionError("Discovery selection ordering differs from the frozen candidate file")

    methodology = {
        "schema": "betlas.geometry-atlas-summary.v1", "contexts": 30,
        "architectures": labels, "fixed_pair_ids": sorted(pair_ids),
        "ranking": "Within each context, rank all 210 same-level unordered A pairs by minimum bidirectional target-H-reweighted source-H-equal neighbor fraction divided by target eligible-H prevalence; equal scores share rank. No null model or significance interpretation.",
        "primary_phase_ranking": "Reproduce the frozen discovery ordering in every phase: six-view support count (>=2 supporting H on both sides), minimum bidirectional descriptive lift, edge count. Final lexical pair tie break; unmeasurable scores retain NA rank. Phase ranks are descriptive and never reselect candidates.",
        "supported_fields": "The frozen floor is >=10 cross-pair undirected domain edges, >=2 supporting H on each side. Supported values are NA below this floor; raw measured fractions and observed zeroes remain separately available. Primary phase support additionally requires >=3/6 views.",
        "directed_edges": "For unequal labels, n_directed_edges in each orientation equals the number of undirected cross-label domain pairs. Diagonal values count both orientations.",
        "h_contributions": "All contributing source and target H are exported. Each side separately decomposes the total H-equal fraction and target-H-weighted fraction. Edge fractions describe concentration, not effective independent samples.",
        "strict": "barrel=A2.40; jelly_roll=T2.60.120; other_sandwich=A2.60 excluding T2.60.120",
        "legacy": "Adds A2.70 and A2.102 to other_sandwich; exact same packing nodes and edges",
        "topology_ranking": "All 946 unordered T pairs among the 44 full-cohort T within A2.60; absent phase topologies retain NA. Within-A induced edges and within-A source/target denominators. J comparisons are not ranked against A pairs.",
        "dependence": "All phases retain historical development exposure. H holdout and direct-PDB purging do not remove full-mother H/PDB component dependence. No independent-domain or independent-edge p-values/CIs are calculated.",
        "contribution_checks": len(checks), "frozen_candidate_order_reproduced": True,
        "elapsed_seconds": time.time() - started,
        "component_deletion": deletion,
    }
    write_json(target / "methodology-and-validation.json", methodology)
    summary = artifacts["fixed-pairs-primary"]
    show = summary[["phase", "pair_id", "n_edges", "support_h_a", "support_h_b", "fixed_views_with_two_h_per_side", "descriptive_min_lift", "phase_descriptive_rank", "meets_frozen_support_rule"]]
    strict = artifacts["strict-legacy-directed"]
    strict = strict.loc[strict.source_label.ne(strict.target_label) & strict.source_label.ne("other_architecture") & strict.target_label.ne("other_architecture"), ["phase", "labelset", "source_label", "target_label", "n_directed_edges", "source_h_equal_fraction", "target_h_reweighted_source_h_equal_fraction", "n_support_source_h", "n_support_target_h"]]
    (target / "ANALYSIS.md").write_text(
        "# Frozen packing-atlas analysis\n\n"
        "These are descriptive, label-conditioned overlays of label-independent packing distances. "
        "No candidate, scaling, distance cut or neighbor list was selected again after opening confirmation. "
        "An edge supports local packing resemblance; it does not establish a continuous transition, connectivity equivalence or an annotation-independent physical boundary.\n\n"
        "## Fixed architecture pairs\n\n" + _markdown_table(show) + "\n\n"
        "The lift is a descriptive opportunity adjustment, not a null-model test. Differences between directions can reflect unequal target availability. "
        "The support rule reproduces the frozen discovery floor and view-stability requirement; satisfying it is descriptive retention, not independent statistical confirmation. "
        "All 210 architecture pairs and every contributing H are exported, including unsupported and zero-edge pairs.\n\n"
        "## Historical label mapping\n\n" + _markdown_table(strict) + "\n\n"
        "Jelly-roll is topology 2.60.120 inside architecture 2.60. The 2.40–2.60 architecture pair and the 946 within-2.60 topology pairs are ranked separately. "
        "Strict and legacy historical summaries use identical graph edges; legacy expands other_sandwich to architectures 2.70 and 2.102.\n\n"
        "## Largest-component deletion\n\n"
        f"Removing full-mother component {deletion['component']} deletes {deletion['removed_nodes']:,}/{deletion['original_nodes']:,} S35 nodes "
        f"({deletion['removed_fraction']:.2%}) and leaves {deletion['remaining_nodes']:,} nodes and "
        f"{deletion['remaining_unique_short_mutual_edges']:,} short mutual edges. The deleted representatives map to "
        f"{deletion['removed_mother_domain_weight']:,} mother-domain records. "
        "This is the original graph's induced subgraph. No farther neighbors replace deleted edges. H weights and coverage are recomputed on the remaining nodes. "
        "Support-limited effects/differences are NA in the supported columns, while raw descriptive measurements are preserved. "
        "This sensitivity does not remove remaining dependence or create external validation.\n\n"
        "## Reading limits\n\n"
        "Rare or phase-absent classes remain in all 21 architecture rows. Missing coverage is not evidence of a sharp boundary. "
        "The main analysis is the S35 representative cohort, not 158,943 independent domains. Full-mother component membership is used only to expose dependency. "
        "The component deletion, H contribution decomposition and historical remapping can weaken generality even when a pair has many edges.\n",
        encoding="utf-8")
    return {"output": str(target), "contexts": 30, "coverage_rows": len(coverage), "rank_rows": len(ranks),
            "contribution_checks": len(checks), "largest_component": deletion,
            "elapsed_seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_summaries(args.config), indent=2), flush=True)


if __name__ == "__main__":
    main()
