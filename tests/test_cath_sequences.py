"""Synthetic tests for annotation-independent official CATH sequences."""

import json

import pandas as pd
import pytest

from betlas.benchmarks.sequences import build_sequence_table, main


def manifest(ids=("9aaaA00", "9aaaa00")):
    return pd.DataFrame({"record_id": ids, "domain_id": ids, "cath_version": "4.4.0",
                         "cath_domain_length": 4, "feature_status": "annotation_failed"})


def test_exact_case_join_preserves_all_rows_and_raw_boundary_header(tmp_path):
    fasta = tmp_path / "domains.fa"
    fasta.write_text(
        ">cath|4_4_0|9aaaa00/10A-13B\nWYVV\n"
        ">cath|4_4_0|9aaaA00/-2-1\nACDE\n"
    )
    result = build_sequence_table(manifest(), fasta)
    assert result.record_id.tolist() == ["9aaaA00", "9aaaa00"]
    assert result.domain_sequence.tolist() == ["ACDE", "WYVV"]
    assert result.sequence_header.tolist() == ["cath|4_4_0|9aaaA00/-2-1", "cath|4_4_0|9aaaa00/10A-13B"]
    assert result.sequence_boundary_provenance.tolist() == ["-2-1", "10A-13B"]
    assert result.sequence_length_matches_cath_domain_length.all()
    assert set(result.sequence_status) == {"ok"}  # DSSP failure is irrelevant.


def test_missing_and_length_disagreement_remain_visible(tmp_path):
    fasta = tmp_path / "domains.fa"
    fasta.write_text(">cath|4_4_0|9aaaA00/1-5\nACDEF\n")
    result = build_sequence_table(manifest(), fasta).set_index("record_id")
    assert result.loc["9aaaA00", "sequence_status"] == "ok"
    assert result.loc["9aaaA00", "sequence_length_difference"] == 1
    assert not result.loc["9aaaA00", "sequence_length_matches_cath_domain_length"]
    assert result.loc["9aaaa00", "sequence_status"] == "missing_sequence"
    assert result.loc["9aaaa00", "domain_sequence"] == ""
    assert pd.isna(result.loc["9aaaa00", "cath_sequence_length"])


@pytest.mark.parametrize("text, match", [
    (">cath|4_3_0|9aaaA00/1-4\nACDE\n", "FASTA release"),
    (">9aaaA00\nACDE\n", "Invalid fixed-release"),
    (">cath|4_4_0|9bbbA00/1-4\nACDE\n>cath|4_4_0|9bbbA00/1-4\nACDE\n", "Duplicate FASTA"),
])
def test_header_and_duplicate_conflicts_fail_even_for_unselected_records(tmp_path, text, match):
    fasta = tmp_path / "domains.fa"
    fasta.write_text(text)
    with pytest.raises(ValueError, match=match):
        build_sequence_table(manifest(), fasta)


def test_multichain_and_invalid_sequence_do_not_become_valid_embeddings(tmp_path):
    fasta = tmp_path / "domains.fa"
    fasta.write_text(">cath|4_4_0|9aaaA00/1-4\nACDE\n>cath|4_4_0|9aaaa00/1-4\nAC*E\n")
    frame = manifest()
    frame["segments_json"] = [json.dumps([{"chain_id": "A"}, {"chain_id": "B"}]), "[]"]
    result = build_sequence_table(frame, fasta)
    assert result.sequence_status.tolist() == ["multichain_requires_explicit_pooling", "invalid_sequence"]
    assert result.domain_sequence.tolist() == ["", ""]
    assert result.cath_sequence_length.tolist() == [4, 4]


def test_empty_manifest_and_cli_keep_empty_schema(tmp_path):
    empty = manifest().iloc[:0]
    manifest_path, fasta, out = (tmp_path / name for name in ["manifest.csv", "domains.fa", "out.csv"])
    empty.to_csv(manifest_path, index=False)
    fasta.write_text(">cath|4_4_0|9aaaA00/1-4\nACDE\n")
    assert main(["--manifest", str(manifest_path), "--fasta", str(fasta), "--out-csv", str(out)]) == 0
    result = pd.read_csv(out)
    assert result.empty
    assert {"record_id", "domain_sequence", "sequence_status", "sequence_header"} <= set(result)


def test_duplicate_manifest_ids_fail(tmp_path):
    fasta = tmp_path / "domains.fa"
    fasta.write_text(">cath|4_4_0|9aaaA00/1-4\nACDE\n")
    with pytest.raises(ValueError, match="unique"):
        build_sequence_table(manifest(("9aaaA00", "9aaaA00")), fasta)
