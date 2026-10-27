import numpy as np
import pandas as pd

from betlas.geometry_relations.relationships import summarize_relationships
from betlas.geometry_relations.scaffold_neighborhoods import pattern_neighbor_stats


def test_partial_graphs_do_not_enter_pattern_fraction():
    nodes = pd.DataFrame({"record_id": ["q", "same", "different", "partial"],
                          "chain_pattern": ["K1", "K1", "K2", "K1"],
                          "chain_primary_eligible": [True, True, True, False]})
    edges = pd.DataFrame({"source_id": ["q", "q", "q", "partial"],
                          "target_id": ["same", "different", "partial", "q"],
                          "mutual": [True]*4, "short_edge": [True]*4})
    result = pattern_neighbor_stats(nodes, edges).set_index("record_id")
    assert result.loc["q", "chain_pattern_neighbor_count"] == 2
    assert result.loc["q", "same_chain_pattern_fraction"] == .5
    assert result.loc["partial", "chain_pattern_neighbor_count"] == 0
    assert np.isnan(result.loc["partial", "same_chain_pattern_fraction"])


def test_domain_only_overlay_uses_requested_label_resolution():
    nodes = pd.DataFrame({"record_id": ["a", "b", "c"], "packing_eligible": [True]*3,
                          "cath_homology_code": ["H1", "H2", "H3"],
                          "A": ["A1", "A1", "A2"], "T": ["T1", "T2", "T3"]})
    edges = pd.DataFrame({"source_id": ["a", "a"], "target_id": ["b", "c"],
                          "mutual": [True]*2, "short_edge": [True]*2, "distance": [.1, .2]})
    full = summarize_relationships(nodes, edges, "T")["per_domain"]
    limited = summarize_relationships(nodes, edges, "T", pair_tables=False)["per_domain"]
    assert full == limited
    assert limited[0]["cross_label_fraction"] == 1.
    assert summarize_relationships(nodes, edges, "A", pair_tables=False)["per_domain"][0]["cross_label_fraction"] == .5
