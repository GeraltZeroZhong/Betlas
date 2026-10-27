from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from betlas.geometry_relations import connection_patterns as module
from betlas.geometry_relations.cohorts import A, H, T


def _record(record_id="x", *, count=4, pairs=None, status="available", sense="anti-parallel"):
    if pairs is None:
        pairs = [(i, i + 1) for i in range(count - 1)]
    nodes = [{"node_id": f"s{i}", "chain_id": "A", "fragment_id": "F1", "sheet_id": "S",
              "polymer_strand_rank": i, "residue_uids": [f"A:{i * 5 + j}:" for j in range(5)],
              "residue_count": 5} for i in range(count)]
    edges = [{"source": f"s{i}", "target": f"s{j}", "sense": sense} for i, j in pairs]
    sequence = [{"source": f"s{i}", "target": f"s{i + 1}", "sequence_context": "same_fragment",
                 "observed_chain_path": True} for i in range(count - 1)]
    return {"record_id": record_id, "status": "ok", "graph": {
        "record_id": record_id, "pairing_status": status, "nodes": nodes,
        "pairing_edges": edges, "sequence_relations": sequence,
    }}


def _rename(record):
    changed = deepcopy(record)
    for node in changed["graph"]["nodes"]:
        node["node_id"] = "renamed-" + node["node_id"]
        node["chain_id"], node["fragment_id"] = "other-chain", "other-fragment"
    for column in ("pairing_edges", "sequence_relations"):
        for edge in changed["graph"][column]:
            edge["source"] = "renamed-" + edge["source"]
            edge["target"] = "renamed-" + edge["target"]
    return changed


def test_abstract_isomorphism_is_not_spatial_mapping_compatibility():
    left = _record()
    right = _record(pairs=[(0, 2), (2, 1), (1, 3)])
    result = module.compare_graph_mapping(left, right, {f"s{i}": f"s{i}" for i in range(4)})
    assert result["abstract_pair_isomorphic"]
    assert not result["abstract_chain_isomorphic"]
    assert result["mapping_complete"]
    assert not result["pair_mapping_compatible"]
    assert not result["chain_mapping_compatible"]
    graph_only_permutation = module.compare_graph_mapping(left, right, {"s0": "s0", "s1": "s2", "s2": "s1", "s3": "s3"})
    assert graph_only_permutation["pair_mapping_compatible"]
    assert not graph_only_permutation["chain_mapping_compatible"]


def test_identifiers_are_irrelevant_but_direction_and_observed_paths_are_preserved():
    left = _record()
    right = _rename(left)
    mapping = {f"s{i}": f"renamed-s{i}" for i in range(4)}
    equal = module.compare_graph_mapping(left, right, mapping)
    assert equal["pair_mapping_compatible"] and equal["chain_mapping_compatible"]
    right["graph"]["sequence_relations"][1]["observed_chain_path"] = False
    changed_path = module.compare_graph_mapping(left, right, mapping)
    assert changed_path["pair_mapping_compatible"] and not changed_path["chain_mapping_compatible"]
    right["graph"]["pairing_edges"][0]["sense"] = "parallel"
    changed_sense = module.compare_graph_mapping(left, right, mapping)
    assert not changed_sense["abstract_pair_isomorphic"]
    assert not changed_sense["pair_mapping_compatible"]


def test_partial_mappings_and_partial_annotations_do_not_become_full_compatibility():
    left = _record()
    result = module.compare_graph_mapping(left, left, {"s0": "s0", "s1": "s1"})
    assert result["abstract_pair_isomorphic"]
    assert result["mapped_pair_edge_precision"] == 1
    assert result["pair_mapping_compatible"] is None
    assert result["chain_mapping_compatible"] is None
    ambiguous = module.compare_graph_mapping(left, left, {"s0": "s0", "s1": "s0"})
    assert ambiguous["status"] == "ambiguous_many_to_one_mapping"
    right = _record(status="partial_clipped_pairings")
    partial = module.compare_graph_mapping(left, right, {f"s{i}": f"s{i}" for i in range(4)})
    assert partial["abstract_pair_isomorphic"] and partial["mapping_complete"]
    assert partial["pair_mapping_compatible"] is None
    assert partial["chain_mapping_compatible"] is None
    unknown = module.build_graph_forms(_record(sense="unknown"))
    assert unknown.status["n_unknown_pairing_senses"] == 3
    assert not unknown.status["pair_primary_eligible"]


def test_fragment_partition_is_not_replaced_by_fragment_names():
    left, right = _record(), _record()
    for node in right["graph"]["nodes"][2:]:
        node["fragment_id"] = "F2"
    right["graph"]["sequence_relations"][1].update(sequence_context="cross_fragment", observed_chain_path=False)
    comparison = module.compare_graph_mapping(left, right, {f"s{i}": f"s{i}" for i in range(4)})
    assert comparison["pair_mapping_compatible"]
    assert not comparison["abstract_chain_isomorphic"]
    assert not comparison["chain_mapping_compatible"]


def test_hash_collision_cannot_merge_nonisomorphic_patterns(monkeypatch):
    cycle = _record(count=6, pairs=[(i, (i + 1) % 6) for i in range(6)])
    triangles = _record(count=6, pairs=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    monkeypatch.setattr(module, "_bucket", lambda graph: ("forced_collision",))
    assignment, prototypes, _mappings = module._assign_patterns(
        {"cycle": module.build_graph_forms(cycle), "triangles": module.build_graph_forms(triangles)}, "pair"
    )
    assert len(prototypes) == 2
    assert assignment["cycle"] != assignment["triangles"]


def test_catalogue_selects_by_support_and_keeps_all_statuses(tmp_path: Path):
    records = [_record(f"r{i}") for i in range(6)] + [_record("partial", status="partial_clipped_pairings"), _record("unknown", sense="unknown")]
    records.append({"record_id": "failed", "status": "geometry_unavailable"})
    graph_path = tmp_path / "graphs.jsonl.gz"
    with gzip.open(graph_path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    cohort = pd.DataFrame({"record_id": [record["record_id"] for record in records], "pdb_id": [f"p{i}" for i in range(9)],
                           A: ["2.10"] * 9, T: ["2.10.10"] * 9, H: ["h1", "h2", "h3"] * 3,
                           "betlas_parse_ok": [1] * 8 + [0], "betlas_beta_strand_count": [4] * 9,
                           "betlas_beta_residue_count": [20] * 9, "betlas_sheet_face_count": [1] * 9,
                           "betlas_axis_best_slice_count": [2] * 9})
    for column in module.P_COLUMNS:
        cohort[column] = 0.5
    cohort["betlas_pca_elongation"] = list(range(9))
    protocol = module.run_connection_patterns(cohort, graph_path, tmp_path / "out")
    assert protocol["domains"] == 9
    selected = pd.read_csv(tmp_path / "out/selected-patterns.csv")
    assert len(selected) == 1 and selected.primary_domains.iloc[0] == 6
    request = pd.read_csv(tmp_path / "out/coordinate-requests.csv")
    assert set(request.record_id) == {f"r{i}" for i in range(6)}
    domain = pd.read_csv(tmp_path / "out/domain-patterns.csv").set_index("record_id")
    assert not domain.loc["partial", "chain_primary_eligible"]
    assert not domain.loc["unknown", "pair_primary_eligible"]
    assert pd.isna(domain.loc["failed", "pair_pattern"])
    representatives = pd.read_csv(tmp_path / "out/series-representatives.csv")
    assert len(representatives) == 5 and representatives.record_id.nunique() == 5


def test_shape_selection_requires_joint_P_availability_not_graph_support_only():
    rows = []
    for pattern, count in (("many_incomplete", 12), ("eligible", 6)):
        for index in range(count):
            row = {"record_id": f"{pattern}-{index}", "chain_pattern": pattern, "chain_primary_eligible": True,
                   H: f"h{index}", A: "2.60", T: "2.60.40"}
            row.update({column: 1.0 for column in module.P_COLUMNS})
            if pattern == "many_incomplete":
                row["betlas_angular_circular_moment_k8"] = float("nan")
            rows.append(row)
    summary = pd.DataFrame({"pattern_id": ["many_incomplete", "eligible"], "kind": ["K_chain"] * 2,
                            "primary_H": [12, 6], "primary_domains": [12, 6]})
    output = module._shape_selection(pd.DataFrame(rows), summary, 13, 3)
    assert output["selected-patterns.csv"].pattern_id.tolist() == ["eligible"]
    assert len(output["coordinate-requests.csv"]) == 6


def test_H_balanced_sampling_preserves_small_families_and_no_replacement():
    population = pd.DataFrame({"record_id": [f"r{i:04d}" for i in range(1100)], H: ["large"] * 1080 + [f"small{i}" for i in range(20)]})
    sample = module._h_balanced_sample(population, 128, 13)
    assert len(sample) == 128 and sample.record_id.nunique() == 128
    assert sample[H].nunique() == 21
    pd.testing.assert_frame_equal(sample, module._h_balanced_sample(population, 128, 13))


def test_revised_P_values_do_not_reselect_patterns_or_display_series(tmp_path: Path):
    rows = []
    for index in range(8):
        row = {"record_id": f"r{index}", "chain_pattern": "selected" if index < 6 else "not_selected",
               "chain_primary_eligible": True, H: f"h{index % 2}", A: "2.60", T: "2.60.40"}
        row.update({column: 0.5 for column in module.P_COLUMNS})
        rows.append(row)
    domains = pd.DataFrame(rows)
    domains.to_csv(tmp_path / "domain-patterns.csv", index=False)
    domains.iloc[:6].to_csv(tmp_path / "selected-pattern-population.csv", index=False)
    series_text = "record_id\nr0\nr5\n"
    (tmp_path / "series-representatives.csv").write_text(series_text)
    cohort = domains.copy()
    cohort["betlas_parse_ok"] = 1
    cohort["betlas_beta_strand_count"] = 4
    cohort["betlas_beta_residue_count"] = 20
    cohort["betlas_sheet_face_count"] = 1
    cohort["betlas_axis_best_slice_count"] = 2
    moment_fields = tuple(f"betlas_angular_circular_moment_k{i}" for i in range(1, 9))
    revised_blocks = dict(module.PACKING_BLOCKS)
    revised_blocks["angular_repetition"] = moment_fields
    for field in moment_fields:
        cohort[field] = 0.25
    original = (tmp_path / "domain-patterns.csv").read_bytes()
    module.refresh_pattern_features(cohort, tmp_path, variant="current")
    result = module.refresh_pattern_features(cohort, tmp_path, variant="synthetic-v2", blocks=revised_blocks)
    assert result["selected_population_n"] == 6
    assert not result["pattern_membership_changed"]
    assert (tmp_path / "domain-patterns.csv").read_bytes() == original
    assert (tmp_path / "series-representatives.csv").read_text() == series_text
    summary = pd.read_csv(tmp_path / "pattern-P-distributions-synthetic-v2.csv").set_index("measurement")
    assert summary.loc[moment_fields[0], "median"] == 0.25
    assert summary.loc[moment_fields[0], "n_total"] == 6
    assert "betlas_angular_circular_moment_k1" in summary.index
