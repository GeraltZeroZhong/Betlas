from __future__ import annotations

import json

import numpy as np
import pytest

from betlas.geometry_relations.scaffold_cases import (
    CASE_PROTOCOL,
    freeze_case_protocol,
    project_pair,
    select_cases,
    trace_segments,
)


def _row(pair="p1", **kwargs):
    return {"pair_id": pair, "query_id": "q"+pair, "target_id": "t"+pair, "mode": "sNS",
            "status": "ok", "alignment_directory": "/tmp/fixture",
            "min_beta_coverage_3p5": ".8", "min_strand_coverage_3p5": ".75",
            "graph_mapping_complete": "True", "graph_pair_mapping_compatible": "False",
            "graph_chain_mapping_compatible": "False", **kwargs}


def test_csv_false_and_missing_graph_states_do_not_become_compatible_evidence():
    selected, ledger = select_cases([
        _row("negative", graph_mapping_complete="False"),
        _row("unknown", graph_pair_mapping_compatible="", graph_chain_mapping_compatible=""),
        _row("positive"),
    ])
    assert len(selected) == 1
    assert selected[0]["alignment"]["pair_id"] == "positive"
    assert sum(r["eligible"] for r in ledger) == 1


def test_case_cap_and_cross_kind_pair_deduplication():
    rows = [_row(str(i), one_to_one_sheet_pair_count="2", whole_sheet_bijection="True",
                 one_to_one_sheet_independent_fit_rmsd="1", one_to_one_sheet_assembly_rmsd_gap="2")
            for i in range(7)]
    selected, _ = select_cases(rows)
    assert len([c for c in selected if c["case_kind"] == "shape_shared_graph_different"]) == 4
    assert len([c for c in selected if c["case_kind"] == "sheet_placement_difference"]) == 3
    pairs = [tuple(sorted((c["alignment"]["query_id"], c["alignment"]["target_id"]))) for c in selected]
    assert len(set(pairs)) == len(pairs)


def test_freer_alignment_requires_both_coverage_gain_and_fragmentation():
    reference = _row("p", min_beta_coverage_3p5=".3", query_split_strand_fraction="0",
                     target_split_strand_fraction="0", strand_correspondence_longest_run_fraction=".8",
                     graph_mapping_complete="False")
    improved = {**reference, "mode": "fNS", "min_beta_coverage_3p5": ".6"}
    assert not select_cases([reference, improved])[0]
    improved["query_split_strand_fraction"] = ".4"
    selected, _ = select_cases([reference, improved])
    assert len(selected) == 1
    assert selected[0]["case_kind"] == "freer_alignment_more_fragmented"
    assert selected[0]["reference_alignment"] == reference


def test_freeze_does_not_silently_overwrite_a_different_protocol(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_case_protocol(path)
    assert json.loads(path.read_text()) == CASE_PROTOCOL
    path.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        freeze_case_protocol(path)


def test_physical_projection_preserves_distances_and_breaks_unobserved_gaps():
    xyz = np.array([[0,0,0], [2,0,0], [0,2,0], [0,0,3]], dtype=float)
    residues = [{"coordinate": p.tolist(), "residue_uid": f"A:{i}:", "label_seq_id": label,
                 "chain_id": "A"} for i, (p,label) in enumerate(zip(xyz,[1,2,8,9],strict=True))]
    graph = {"nodes": [{"node_id": "B1", "sheet_id": "S", "chain_id": "A",
                        "residue_uids": [r["residue_uid"] for r in residues], "label_positions": [1,2,8,9]}]}
    transform = {"rotation": [[0,-1,0],[1,0,0],[0,0,1]], "translation": [5,7,9]}
    qp, tp, camera = project_pair(residues, residues, graph, transform)
    assert np.linalg.norm(qp[1]-qp[0]) == pytest.approx(2)
    assert np.linalg.norm(tp[3]-tp[0]) == pytest.approx(3)
    assert np.linalg.det(camera["camera_basis_rows"]) == pytest.approx(1)
    assert trace_segments(residues) == [[0,1],[2,3]]
