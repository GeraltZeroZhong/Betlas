from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from betlas.geometry_relations.cohorts import A, H, T
from betlas.geometry_relations.scaffold_summary import (
    common_pair_readouts,
    correlation_tables,
    graph_relation,
    join_alignments,
    legacy_sq_comparison,
    load_alignment_source,
    run_scaffold_summary,
    summarize_differences,
    summarize_metrics,
    tier_random_differences,
)


def _cohort():
    return pd.DataFrame([
        {"record_id": "q", A: "2.10", T: "2.10.20", H: "2.10.20.10", "pdb_id": "aaaa"},
        {"record_id": "t", A: "2.60", T: "2.60.40", H: "2.60.40.10", "pdb_id": "bbbb"},
    ])


def _pairs(panel="legacy", pair_id="legacy_pair_001"):
    return pd.DataFrame([{"pair_id": pair_id, "query_id": "q", "target_id": "t", "panel": panel,
                          "packing_distance": .2, "packing_distance_legacy": .8,
                          "role": "candidate", "relation_kind": "test_relation"}])


def _raw(pair_id="legacy_pair_001", mode="sNS"):
    return pd.DataFrame([{"pair_id": pair_id, "query_id": "q", "target_id": "t", "mode": mode,
                          "status": "ok", "alignment_status": "ok", "min_tm_score": .7,
                          "measurement_scope": "actual_correspondences", "min_beta_coverage_3p5": .5,
                          "graph_abstract_chain_isomorphic": "True", "graph_mapping_complete": "False"}])


def test_join_preserves_requested_denominators_orientation_and_labels():
    requests = _pairs()[["pair_id", "query_id", "target_id"]].assign(mode="sNS")
    requests = pd.concat([requests, requests.assign(mode="fNS")], ignore_index=True)
    result = join_alignments(_pairs(), requests, _cohort(), [_raw()])
    assert len(result) == 3  # inherited SQ request retained even if absent
    assert set(result.query_A) == {"2.10"}
    missing = result.loc[result["mode"].eq("fNS")].iloc[0]
    assert missing.status == "not_run"
    assert pd.isna(missing.min_beta_coverage_3p5)
    assert missing.packing_distance == .2 and missing.packing_distance_legacy == .8
    assert missing.graph_relation == "spatial_mapping_unavailable"
    with pytest.raises(ValueError, match="Multiple raw"):
        join_alignments(_pairs(), requests, _cohort(), [_raw(), _raw()])
    with pytest.raises(ValueError, match="planned orientation"):
        join_alignments(_pairs(), requests, _cohort(), [_raw().assign(query_id="t")])


def test_source_pointers_and_scalar_only_SQ_do_not_gain_beta_measurements(tmp_path):
    path = tmp_path / "pair-alignments.csv"
    _raw(pair_id="pair_001").assign(stdout_path="raw/out.txt", min_beta_coverage_3p5=.99).to_csv(path, index=False)
    source = load_alignment_source(path, legacy=True, scalar_only=True)
    assert source.pair_id.iloc[0] == "legacy_pair_001"
    assert source.source_pair_id_alignment.iloc[0] == "pair_001"
    assert source.stdout_path.iloc[0] == str(tmp_path / "raw/out.txt")
    assert source.source_stdout_path.iloc[0] == "raw/out.txt"
    assert "min_beta_coverage_3p5" not in source
    requests = _pairs()[["pair_id", "query_id", "target_id"]].assign(mode="sNS")
    joined = join_alignments(_pairs(), requests, _cohort(), [_raw(), source])
    sq = joined.loc[joined["mode"].eq("SQ")].iloc[0]
    assert sq.min_tm_score == .7
    assert pd.isna(sq.min_beta_coverage_3p5)
    assert sq.graph_relation == "not_measured_scalar_only"
    comparison = legacy_sq_comparison(joined)
    assert comparison.minus_SQ_min_tm_score.iloc[0] == 0
    assert not any("beta_coverage" in c for c in comparison)


def test_graph_relation_requires_complete_actual_mapping():
    row = {"graph_abstract_pair_isomorphic": True, "graph_abstract_chain_isomorphic": True,
           "graph_mapping_complete": False, "graph_pair_mapping_compatible": True,
           "graph_chain_mapping_compatible": True}
    assert graph_relation(row) == "partial_spatial_mapping"
    row["graph_mapping_complete"] = True
    assert graph_relation(row) == "connection_preserved"
    row["graph_chain_mapping_compatible"] = False
    assert graph_relation(row) == "pairing_preserved_chain_changed"
    row["graph_pair_mapping_compatible"] = False
    assert graph_relation(row) == "pairing_changed"
    row["graph_pair_mapping_compatible"] = None
    assert graph_relation(row) == "complete_mapping_pair_graph_unavailable"
    row["graph_pair_mapping_compatible"] = True
    row["graph_chain_mapping_compatible"] = None
    assert graph_relation(row) == "complete_mapping_chain_graph_unavailable"


def test_anchor_and_H_means_do_not_weight_repeated_pairs_as_independent():
    data = pd.DataFrame({"panel": ["atlas"] * 6, "mode": ["sNS"] * 6,
                         "unit_anchor": ["a", "a", "a", "b", "c", "d"],
                         "query_H": ["h1", "h1", "h1", "h1", "h2", "h2"],
                         "score": [0., 0., 0., 1., 1., np.nan],
                         "native_available": [True] * 6, "correspondence_available": [True] * 6})
    summary, anchors, hs = summarize_metrics(data, ["panel", "mode"], ["score"])
    row = summary.iloc[0]
    assert row["mean"] == pytest.approx(.4)
    assert row.anchor_equal_mean == pytest.approx(2 / 3)
    assert row.query_H_equal_anchor_mean == pytest.approx(.75)
    assert row.n_requested == 6 and row.n_observed == 5 and row.n_missing == 1
    assert row.n_anchors_requested == 4 and row.n_anchors_observed == 3
    assert len(anchors) == 3 and len(hs) == 2


def _atlas():
    return pd.DataFrame([
        {"pair_id": "a1", "unit_anchor": "a", "query_H": "h1", "query_A": "2.10", "mode": "sNS",
         "distance_tier": "nearest", "actual_topology_relation": "different_T", "min_tm_score": .8,
         "packing_distance": .1},
        {"pair_id": "a2", "unit_anchor": "a", "query_H": "h1", "query_A": "2.10", "mode": "sNS",
         "distance_tier": "random", "actual_topology_relation": "different_T", "min_tm_score": .2,
         "packing_distance": .9},
        {"pair_id": "b1", "unit_anchor": "b", "query_H": "h2", "query_A": "2.60", "mode": "sNS",
         "distance_tier": "nearest", "actual_topology_relation": "different_T", "min_tm_score": .9,
         "packing_distance": .3},
    ])


def test_tier_difference_uses_same_anchor_random_and_retains_missing():
    differences = tier_random_differences(_atlas(), ("min_tm_score",))
    assert differences.loc[differences.unit_anchor.eq("a"), "tier_minus_random"].iloc[0] == pytest.approx(.6)
    assert differences.loc[differences.unit_anchor.eq("b"), "status"].iloc[0] == "no_random_target"
    summary = summarize_differences(differences).iloc[0]
    assert summary.n_requested == 2 and summary.n_observed == 1
    assert summary.query_H_equal_mean == pytest.approx(.6)
    invalid = _atlas()
    invalid.loc[1, "actual_topology_relation"] = "same_T"
    with pytest.raises(ValueError, match="topology stratum"):
        tier_random_differences(invalid)


def test_common_readouts_require_identical_pair_set():
    pair_id = "matched_002"
    pairs = _pairs("common_matched", pair_id)
    requests = pairs[["pair_id", "query_id", "target_id"]].assign(mode="sNS")
    data = join_alignments(pairs, requests, _cohort(), [_raw(pair_id)], include_legacy_sq=False)
    readouts = pd.DataFrame([{"pair_id": pair_id, "field": "P_feature", "query_value": 2,
                              "target_value": 1, "query_minus_target": 1}])
    joint = common_pair_readouts(data, readouts)
    assert joint.min_beta_coverage_3p5.iloc[0] == .5
    assert joint.query_minus_target.iloc[0] == 1
    with pytest.raises(ValueError, match="different pairs"):
        common_pair_readouts(data, readouts.assign(pair_id="unselected_pair"))


def test_correlations_remain_descriptive_and_within_anchor():
    data = pd.DataFrame({"panel": ["atlas"] * 6, "mode": ["sNS"] * 6,
                         "actual_topology_relation": ["different_T"] * 6,
                         "unit_anchor": ["a"] * 3 + ["b"] * 3, "query_H": ["h1"] * 3 + ["h2"] * 3,
                         "packing_distance": [1, 2, 3, 1, 2, 3], "min_tm_score": [3, 2, 1, 1, 2, 3]})
    for threshold in ("2", "3p5", "5"):
        data[f"min_beta_coverage_{threshold}"] = data.min_tm_score
    pooled, within, per_h = correlation_tables(data)
    tm = within.loc[within.metric.eq("min_tm_score")]
    assert sorted(tm.spearman_rho) == [-1, 1]
    assert len(per_h) == 8
    assert not any("p_value" in c or "confidence" in c for table in (pooled, within, per_h) for c in table)


def test_runner_writes_joined_sources_and_explicit_denominators(tmp_path):
    derived, outputs, previous = (tmp_path / name for name in ("derived", "outputs", "previous"))
    derived.mkdir()
    (outputs / "scaffold-legacy-smoke").mkdir(parents=True)
    (previous / "structure-validation").mkdir(parents=True)
    _pairs().to_csv(derived / "study-pairs.csv", index=False)
    _pairs()[["pair_id", "query_id", "target_id"]].assign(mode="sNS").to_csv(derived / "alignment-requests.csv", index=False)
    _cohort().to_csv(derived / "cohort.csv", index=False)
    _raw("pair_001").drop(columns="measurement_scope").to_csv(outputs / "scaffold-legacy-smoke/scaffold-alignments.csv", index=False)
    _raw("pair_001").drop(columns=["measurement_scope", "mode"]).to_csv(previous / "structure-validation/pair-alignments.csv", index=False)
    config = {"derived": str(derived), "outputs": str(outputs), "previous_outputs": str(previous),
              "measurement_cohort": str(derived / "cohort.csv")}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    destination = run_scaffold_summary(cfg)
    joined = pd.read_csv(destination / "alignments-joined.csv.gz", dtype=str)
    assert len(joined) == 2 and joined.query_A.iloc[0] == "2.10"
    protocol = json.loads((destination / "summary-protocol.json").read_text())
    assert protocol["n_requested_new_alignments"] == 1
    assert protocol["n_rows_including_inherited_SQ"] == 2
    assert (destination / "graph-relations.csv").exists()
    with pytest.raises(FileExistsError):
        run_scaffold_summary(cfg)
