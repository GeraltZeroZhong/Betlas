"""Synthetic reference-isolation and score-semantics checks."""

import json
import subprocess
from pathlib import Path

import pytest
from Bio.PDB.PDBExceptions import PDBConstructionException

from scripts.external_baselines import structure_retrieval as retrieval
from scripts.external_baselines.beta_barrel_detection.foldseek.runner import FoldseekHit


def record(identifier, target=None):
    row = dict(sample_id=identifier, structure_path=f"/{identifier}.cif", chain_id="A",
               pdb_id=identifier, group_id=identifier)
    if target is not None:
        row["target"] = target
    return row


def hit(query, reference, score, coverage=1):
    return FoldseekHit(query.encode().hex() + "_A", reference.encode().hex() + "_A",
                       20, 20, 20, coverage, coverage, score, score, score, 1e-5, 100)


def test_both_reference_classes_and_same_group_or_pdb_are_enforced():
    q, refs = [record("q")], [record("p", 1), record("n", 0)]
    retrieval.validate_collections(q, refs)
    for field in ("pdb_id", "group_id", "sample_id"):
        with pytest.raises(ValueError, match="overlap"):
            retrieval.validate_collections([dict(q[0], **{field: refs[0][field]})], refs)
    with pytest.raises(ValueError, match="both binary"):
        retrieval.validate_collections(q, refs[:1])
    with pytest.raises(ValueError, match="source overlap"):
        retrieval.validate_collections([dict(q[0], structure_path=refs[0]["structure_path"])], refs)


def test_margin_uses_both_classes_and_native_tie_is_negative():
    queries, refs = [record("q")], [record("p", 1), record("n", 0)]
    rows = retrieval.summarize_hits([hit("q", "p", .8), hit("q", "n", .6)], queries, refs)
    assert rows[0]["score"] == pytest.approx(.2)
    assert rows[0]["prediction"] == 1
    tied = retrieval.summarize_hits([hit("q", "p", .8), hit("q", "n", .8)], queries, refs)
    assert tied[0]["prediction"] == 0 and tied[0]["score"] == 0


def test_missing_one_class_is_zero_evidence_both_missing_is_unavailable():
    queries, refs = [record("q")], [record("p", 1), record("n", 0)]
    row = retrieval.summarize_hits([hit("q", "p", .8)], queries, refs)[0]
    assert row["score"] == .8 and row["positive_hit_available"]
    assert not row["negative_hit_available"] and row["best_negative_score"] is None
    unavailable = retrieval.summarize_hits([hit("q", "p", .8, .2)], queries, refs)[0]
    assert unavailable["status"] == "unavailable"
    assert unavailable["score"] is None and unavailable["prediction"] is None


def test_outside_reference_hit_and_invalid_tm_score_fail():
    queries, refs = [record("q")], [record("p", 1), record("n", 0)]
    with pytest.raises(ValueError, match="outside"):
        retrieval.summarize_hits([hit("q", "outside", .8)], queries, refs)
    with pytest.raises(ValueError, match="finite"):
        retrieval.summarize_hits([hit("q", "p", float("nan"))], queries, refs)


def test_corrupt_coordinate_is_recorded_as_unavailable(tmp_path, monkeypatch):
    def corrupt(path):
        raise PDBConstructionException("Invalid coordinate at line 1")
    monkeypatch.setattr(retrieval, "_parse_structure", corrupt)
    ready, inspection = retrieval._prepare([record("q")], tmp_path / "inputs")
    assert ready == [] and inspection[0]["status"] == "unavailable"
    assert "PDBConstructionException" in inspection[0]["reason"]


@pytest.mark.parametrize("fail", [False, True, "malformed"])
@pytest.mark.parametrize("alignment_type", [1, 2])
def test_native_invocation_preserves_query_denominator_and_linux_scratch(tmp_path, monkeypatch, fail, alignment_type):
    queries, refs = [record("q"), record("missing")], [record("p", 1), record("n", 0)]
    binary = tmp_path / "foldseek"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    def prepare(rows, folder):
        folder.mkdir()
        return [r for r in rows if r["sample_id"] != "missing"], [
            dict(r, status="unavailable" if r["sample_id"] == "missing" else "ok",
                 reason="missing coordinates" if r["sample_id"] == "missing" else "") for r in rows]

    def backend(command, **kwargs):
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "10.synthetic", "")
        assert command[command.index("--gpu") + 1] == "0"
        assert command[command.index("--alignment-type") + 1] == str(alignment_type)
        assert command[command.index("--max-seqs") + 1] == "2"
        assert Path(command[5]).parent == tmp_path
        if fail is True:
            raise subprocess.TimeoutExpired(command, 1)
        Path(command[4]).write_text("71_A\t70_A\t20\t20\t20\t1\t1\t.8\t.7\t.7\t1e-5\t100\n"
                                    if fail != "malformed" else "outside\tbad\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(retrieval, "_prepare", prepare)
    monkeypatch.setattr(subprocess, "run", backend)
    summary = retrieval.run(queries, refs, tmp_path / "out", foldseek=str(binary),
                            task="beta_solenoid", scratch_dir=tmp_path, exhaustive=True,
                            alignment_type=alignment_type)
    assert summary["alignment_type"] == alignment_type
    assert summary["n_queries"] == 2
    result = json.loads((tmp_path / "out/result-00000.json").read_text())
    assert result["status"] == ("unavailable" if fail else "ok")
    missing = json.loads((tmp_path / "out/result-00001.json").read_text())
    assert missing["status"] == "unavailable" and missing["reason"] == "missing coordinates"
    assert not list(tmp_path.glob("betlas-specialist-foldseek-*"))
