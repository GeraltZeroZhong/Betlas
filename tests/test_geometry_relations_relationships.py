from __future__ import annotations

import pytest

from betlas.geometry_relations.relationships import summarize_relationships


def _node(record_id, label, homology, eligible=True):
    return {"record_id": record_id, "architecture": label, "cath_homology_code": homology,
            "pdb_id": f"pdb-{record_id}", "packing_eligible": eligible}


def _edge(source, target, distance=.2, mutual=True, short=True):
    return {"source_id": source, "target_id": target, "distance": distance,
            "mutual": mutual, "short_edge": short}


def _fixture():
    nodes = [_node("a1", "A", "ha1"), _node("a2", "A", "ha1"),
             _node("a3", "A", "ha2"), _node("a4", "A", "ha3"),
             _node("a5", "A", "ha4", False), _node("b1", "B", "hb1"),
             _node("b2", "B", "hb2"), _node("b3", "B", "hb2", False),
             _node("c1", "C", "hc1")]
    edges = [_edge("a1", "b1"), _edge("b1", "a1"), _edge("a1", "b2"),
             _edge("a1", "c1"), _edge("a2", "b1"), _edge("a3", "c1"),
             _edge("a4", "b1", mutual=False), _edge("a5", "b1")]
    return nodes, edges


def _directed(result, source, target):
    return next(row for row in result["directed"] if row["source_label"] == source and row["target_label"] == target)


def test_domain_normalization_then_source_h_balance_and_target_h_reweighting():
    nodes, edges = _fixture()
    result = summarize_relationships(nodes, edges, "architecture", ["A", "B", "C", "D"])
    row = _directed(result, "A", "B")
    assert row["domain_equal_fraction"] == pytest.approx((2 / 3 + 1) / 3)
    assert row["source_h_equal_fraction"] == pytest.approx(((2 / 3 + 1) / 2) / 2)
    # hb2 has two phase nodes, including unavailable b3; its neighbor receives weight 1/2.
    assert row["target_h_reweighted_source_h_equal_fraction"] == pytest.approx(.4)
    assert row["source_domain_coverage"] == .75
    assert row["source_h_coverage"] == pytest.approx(2 / 3)
    assert row["target_prevalence"] == pytest.approx(3 / 9)
    assert row["target_eligible_prevalence"] == pytest.approx(2 / 7)
    assert row["largest_source_h"] == "ha1"
    assert row["drop_largest_source_h_fraction"] == 0
    assert row["drop_largest_source_h_delta"] == pytest.approx(-5 / 12)


def test_mutual_short_undirected_pairs_are_deduplicated_and_unavailable_edges_excluded():
    nodes, edges = _fixture()
    result = summarize_relationships(nodes, edges, "architecture")
    assert result["summary"]["n_unique_short_mutual_edges"] == 5
    assert result["summary"]["edge_filter_counts"] == {
        "duplicate_undirected_pair": 1, "not_mutual_and_short": 1,
        "geometrically_unavailable_endpoint": 1,
    }
    row = _directed(result, "A", "B")
    assert row["n_directed_edges"] == 3
    assert row["n_support_source_domains"] == 2
    assert row["n_support_target_domains"] == 2
    assert row["n_support_source_h"] == 1
    assert row["max_source_h_edge_fraction"] == 1
    assert row["max_target_h_edge_fraction"] == pytest.approx(2 / 3)
    undirected = next(row for row in result["undirected"] if (row["label_1"], row["label_2"]) == ("A", "B"))
    assert undirected["n_edges"] == 3
    assert undirected["n_support_domains"] == 4
    assert undirected["n_support_h"] == 3
    assert undirected["max_h_edge_fraction"] == 1


def test_isolation_missing_geometry_and_absent_labels_never_become_mixing_zero():
    nodes, edges = _fixture()
    result = summarize_relationships(nodes, edges, "architecture", ["A", "B", "C", "D"])
    assert len(result["directed"]) == 16
    assert len(result["undirected"]) == 10
    lookup = {row["record_id"]: row for row in result["per_domain"]}
    for record_id in ("a4", "a5", "b3"):
        assert lookup[record_id]["label_overlap"] is None
        assert lookup[record_id]["cross_label_fraction"] is None
        assert lookup[record_id]["neighbor_label_fractions"] is None
    assert lookup["a4"]["n_neighbors"] == 0
    assert lookup["a5"]["n_neighbors"] is None
    assert _directed(result, "A", "D")["source_h_equal_fraction"] is None
    assert _directed(result, "D", "A")["source_h_equal_fraction"] is None
    # C is supported but B has no C neighbors: this is an observed zero.
    assert _directed(result, "B", "C")["source_h_equal_fraction"] == 0


def test_within_label_edges_have_zero_mixing_but_complete_self_label_share():
    nodes = [_node("a", "A", "h1"), _node("b", "A", "h2")]
    result = summarize_relationships(nodes, [_edge("a", "b")], "architecture")
    assert all(row["cross_label_fraction"] == 0 for row in result["per_domain"])
    assert all(row["label_overlap"] is False for row in result["per_domain"])
    assert result["directed"][0]["source_h_equal_fraction"] == 1
    assert result["directed"][0]["n_directed_edges"] == 2
    assert result["undirected"][0]["n_edges"] == 1


def test_all_unavailable_or_no_edges_keep_label_coverage_but_no_fraction():
    nodes = [_node("a", "A", "h1", "False"), _node("b", "B", "h2", "true")]
    result = summarize_relationships(nodes, [], "architecture", ["A", "B", "C"])
    assert result["summary"]["n_eligible_nodes"] == 1
    assert _directed(result, "B", "A")["status"] == "target_geometry_unavailable"
    assert _directed(result, "B", "B")["status"] == "source_no_neighbors"
    assert all(row["source_h_equal_fraction"] is None for row in result["directed"])
    assert all(row["label_overlap"] is None for row in result["per_domain"])


def test_reordered_inputs_do_not_change_family_balanced_shares():
    nodes, edges = _fixture()
    before = summarize_relationships(nodes, edges, "architecture")
    after = summarize_relationships(nodes[::-1], edges[::-1], "architecture")
    for left, right in zip(before["directed"], after["directed"], strict=True):
        for column in ("source_h_equal_fraction", "target_h_reweighted_source_h_equal_fraction", "drop_largest_source_h_fraction"):
            if left[column] is None:
                assert right[column] is None
            else:
                assert left[column] == pytest.approx(right[column])


@pytest.mark.parametrize("defect", ["duplicate_node", "missing_node", "self_edge", "distance_disagreement", "omitted_label"])
def test_invalid_graph_identity_or_declared_label_support_is_rejected(defect):
    nodes = [_node("a", "A", "h1"), _node("b", "B", "h2")]
    edges, labels = [_edge("a", "b")], None
    if defect == "duplicate_node":
        nodes.append(nodes[0])
    elif defect == "missing_node":
        edges.append(_edge("a", "missing"))
    elif defect == "self_edge":
        edges.append(_edge("a", "a"))
    elif defect == "distance_disagreement":
        edges.append(_edge("b", "a", .9))
    else:
        labels = ["A"]
    with pytest.raises(ValueError):
        summarize_relationships(nodes, edges, "architecture", labels)
