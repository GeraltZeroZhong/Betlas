"""Test induced-graph and sparse-support semantics using synthetic labels."""
import pandas as pd

from betlas.geometry_relations.cohorts import A, H
from betlas.geometry_relations.relationships import summarize_relationships
from betlas.geometry_relations.summaries import (
    descriptive_ranking,
    enrich_directions,
    induced_component_deletion,
    major_h_contributions,
)


def fixture():
    nodes = pd.DataFrame([
        {"record_id": name, A: label, H: h, "pdb_id": name,
         "packing_eligible": True, "mother_h_pdb_component": component}
        for name, label, h, component in [
            ("a", "2.10", "h1", "big"), ("b", "2.10", "h2", "small"),
            ("c", "2.40", "h3", "big"), ("d", "2.40", "h4", "small2"),
            ("e", "2.60", "h5", "big")]])
    edges = pd.DataFrame([
        {"source_id": source, "target_id": target, "distance": .2,
         "mutual": True, "short_edge": True}
        for source, target in [("a", "c"), ("c", "a"), ("b", "d"), ("d", "b"), ("c", "e")]])
    return nodes, edges


def test_component_deletion_only_retains_existing_edges_and_absent_labels():
    nodes, edges = fixture()
    remaining, retained, meta = induced_component_deletion(nodes, edges)
    assert set(remaining.record_id) == {"b", "d"}
    assert set(zip(retained.source_id, retained.target_id, strict=True)) == {("b", "d"), ("d", "b")}
    assert meta["removed_nodes"] == 3
    rel = pd.DataFrame(summarize_relationships(remaining, retained, A, all_labels=["2.10", "2.40", "2.60"])["directed"])
    enriched = enrich_directions(rel, remaining, A)
    pair = enriched.loc[enriched.source_label.eq("2.10") & enriched.target_label.eq("2.40")].iloc[0]
    assert pair.source_h_equal_fraction == 1
    assert pair.support_status == "insufficient_support_h"
    assert pd.isna(pair.supported_source_h_equal_fraction)
    absent = enriched.loc[enriched.target_label.eq("2.60")]
    assert absent.source_h_equal_fraction.isna().all()


def test_h_contributions_decompose_fractions_and_deduplicate_edges():
    nodes, edges = fixture()
    rel = pd.DataFrame(summarize_relationships(nodes, edges, A)["directed"])
    contributions = major_h_contributions(nodes, edges, A, {"2.10--2.40"})
    summed = contributions.groupby(["source_label", "target_label", "side"])["h_equal_contribution"].sum()
    assert summed.loc["2.10", "2.40", "source"] == 1
    assert summed.loc["2.10", "2.40", "target"] == 1
    assert contributions.loc[contributions.source_label.eq("2.10") & contributions.side.eq("source"), "n_edges"].sum() == 2
    actual = rel.loc[rel.source_label.eq("2.40") & rel.target_label.eq("2.10"), "source_h_equal_fraction"].iloc[0]
    assert summed.loc["2.40", "2.10", "source"] == actual


def test_rank_retains_same_level_absent_pairs_with_na():
    nodes, edges = fixture()
    rel = pd.DataFrame(summarize_relationships(nodes, edges, A)["directed"])
    rank = descriptive_ranking(enrich_directions(rel, nodes, A), ["2.10", "2.40", "2.60", "2.170"])
    assert len(rank) == 6
    absent = rank.loc[rank.a.eq("2.170") | rank.b.eq("2.170")]
    assert absent.descriptive_lift_rank.isna().all()
    assert rank.supported_lift_rank.isna().all()
    assert (rank.a + "--" + rank.b).equals(rank.pair_id)
