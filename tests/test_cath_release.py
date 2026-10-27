"""Synthetic checks for fixed-release hierarchy and domain-boundary contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from betlas.io.cath_release import (
    build_release_manifest,
    read_cath_names,
    read_domain_list,
    read_domall,
    select_s35_representatives,
    summarize_support,
)


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    domains = tmp_path / "domains.txt"
    domains.write_text(
        "# Synthetic CLF 2.0, ordered representatives first\n"
        "9aaaA00 2 10 20 30 1 1 1 1 1 40 2.0\n"
        "9aabA01 2 10 20 30 1 1 1 1 2 41 1.5\n"
        "9aacA01 2 10 20 30 2 1 1 1 1 20 2.1\n"
        "9aad001 2 20 10 10 1 1 1 1 1 12 999.0\n"
        "9aaeA01 2 20 10 10 2 1 1 1 1 12 2.0\n"
        "9aafA00 1 10 10 10 1 1 1 1 1 12 1.0\n"
    )
    boundaries = tmp_path / "boundaries.txt"
    boundaries.write_text(
        "9aaaA D01 F00 1 A -2 - A 37 -\n"
        "9aabA D01 F00 1 A 1 - A 41 -\n"
        "9aacA D02 F01 2 A 1 A A 10 B B 30 - B 39 - 1 A 11 - A 29 - A 40 - A 40 - (1)\n"
        "9aad0 D01 F00 1 0 1 - 0 12 -\n"
    )
    names = tmp_path / "names.txt"
    names.write_text(
        "2 9aaaA00 :Mainly Beta\n"
        "2.10 9aaaA00 :Synthetic architecture one\n"
        "2.10.20 9aaaA00 :Synthetic topology one\n"
        "2.10.20.30 9aaaA00 :\n"
        "2.20 9aad001 :Synthetic architecture two\n"
        "2.20.10 9aad001 :Synthetic topology two\n"
        "2.20.10.10 9aad001 :Synthetic superfamily two\n"
    )
    return domains, boundaries, names


def test_full_hierarchy_and_boundary_mapping(tmp_path: Path) -> None:
    frame = build_release_manifest(*_files(tmp_path))
    assert len(frame) == 5
    rows = frame.set_index("record_id")
    assert rows.loc["9aaaA00", "cath_s35_cluster_id"] == "2.10.20.30.1"
    assert rows.loc["9aacA01", "cath_s60_cluster_id"] == "2.10.20.30.2.1"
    assert rows.loc["9aacA01", "cath_s95_cluster_id"] == "2.10.20.30.2.1.1"
    assert rows.loc["9aacA01", "cath_s100_cluster_id"] == "2.10.20.30.2.1.1.1"
    assert rows.loc["9aad001", "cath_s35_cluster_id"] != rows.loc["9aaaA00", "cath_s35_cluster_id"]
    assert rows.loc["9aaaA00", "boundary_status"] == "ok"
    assert rows.loc["9aabA01", "boundary_status"] == "ok"  # D01/F00 need not mean ID 00.
    assert json.loads(rows.loc["9aaaA00", "segments_json"])[0]["start"] == -2
    assert rows.loc["9aad001", "chain_id"] == ""
    assert rows.loc["9aad001", "cath_chain_id"] == "0"
    segments = json.loads(rows.loc["9aacA01", "segments_json"])
    assert len(segments) == 2
    assert segments[0] == {"chain_id": "A", "start": 1, "start_ins": "A", "end": 10, "end_ins": "B"}
    assert segments[1]["chain_id"] == "B"
    assert rows.loc["9aaaA00", "cath_homology_name"] == ""
    assert set(frame.cath_version) == {"4.4.0"}


def test_missing_boundary_keeps_domain_without_whole_chain_fallback(tmp_path: Path) -> None:
    frame = build_release_manifest(*_files(tmp_path)).set_index("record_id")
    missing = frame.loc["9aaeA01"]
    assert missing.boundary_status == "unmapped"
    assert missing.boundary_reason == "missing_domall_chain"
    assert json.loads(missing.segments_json) == []


@pytest.mark.parametrize("domain_id, expected", [
    ("9aaaA00", "whole_chain_id_conflicts_with_domall"),
    ("9aaaA03", "domain_ordinal_absent_from_domall"),
])
def test_conflicting_boundary_keeps_explicit_mapping_failure(tmp_path: Path, domain_id: str, expected: str) -> None:
    domains, boundaries, names = _files(tmp_path)
    domains.write_text(f"{domain_id} 2 10 20 30 1 1 1 1 1 40 2.0\n")
    boundaries.write_text("9aaaA D02 F00 1 A 1 - A 20 - 1 A 21 - A 40 -\n")
    frame = build_release_manifest(domains, boundaries, names)
    assert frame.iloc[0].boundary_reason == expected
    assert frame.iloc[0].segments_json == "[]"


@pytest.mark.parametrize("row", [
    "9aaaA00 2 10 20 30 1 1 1 1 1 40",  # wrong column count
    "9aaaA00 2 10 20 30 0 1 1 1 1 40 2.0",  # invalid hierarchy
    "9aaaA00 2 10 20 30 1 1 1 1 1 40 nan",  # non-finite resolution
    "9aaaA00 1 10 20 30 1 1 1 1 1 0 2.0",  # filtered classes are still validated
])
def test_clf_rejects_malformed_rows(tmp_path: Path, row: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(row + "\n")
    with pytest.raises(ValueError):
        read_domain_list(path, class_id=2)


@pytest.mark.parametrize("second", [
    "9aaaA00 2 10 20 30 1 1 1 1 2 40 2.0",  # duplicate ID
    "9aabA00 2 10 20 30 1 1 1 1 1 40 2.0",  # duplicate SOLID code
])
def test_clf_duplicate_conflicts_fail(tmp_path: Path, second: str) -> None:
    path = tmp_path / "duplicate.txt"
    path.write_text("9aaaA00 2 10 20 30 1 1 1 1 1 40 2.0\n" + second + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_domain_list(path)


@pytest.mark.parametrize("row", [
    "9aaaA D01 F00 1 A 1 - A 10",  # truncated
    "9aaaA D01 F00 1 A 1 - B 10 -",  # invalid cross-chain segment
    "9aaaA D01 F01 1 A 1 - A 10 - A 11 - A 12 -",  # missing fragment count
    "9aaaA D01 F00 1 A 1 - A 10 - trailing",  # extra fields
])
def test_domall_rejects_invalid_format(tmp_path: Path, row: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(row + "\n")
    with pytest.raises(ValueError):
        read_domall(path)


def test_domall_fragments_and_case_are_preserved(tmp_path: Path) -> None:
    _, boundaries, _ = _files(tmp_path)
    with boundaries.open("a") as handle:
        handle.write("9aaaa D01 F00 1 a 1 - a 4 -\n")
    entries = read_domall(boundaries)
    assert set(entries) >= {"9aaaA", "9aaaa"}
    assert entries["9aacA"].fragments[0]["residue_count"] == 1
    assert entries["9aacA"].fragments[0]["start"] == 40
    assert entries["9aaaa"].domains[0][0]["chain_id"] == "a"


@pytest.mark.parametrize("segments, reason", [
    ("1 A 10 - A 1 -", "nonmonotonic_author_numbering"),
    ("2 A 1 - A 10 - A 10 - A 20 -", "overlapping_author_ranges"),
])
def test_author_numbering_anomalies_are_preserved_and_flagged(
    tmp_path: Path, segments: str, reason: str,
) -> None:
    domains, boundaries, names = _files(tmp_path)
    boundaries.write_text(f"9aaaA D01 F00 {segments}\n")
    row = build_release_manifest(domains, boundaries, names).set_index("record_id").loc["9aaaA00"]
    assert row.boundary_status == "requires_coordinate_order"
    assert row.boundary_reason == reason
    assert json.loads(row.segments_json)[0]["start"] == (10 if "nonmonotonic" in reason else 1)


def test_fixed_release_version_conflict_fails(tmp_path: Path) -> None:
    domains, boundaries, names = _files(tmp_path)
    domains.write_text("# CATH VERSION: v4.3.0\n" + domains.read_text())
    with pytest.raises(ValueError, match="declared CATH version"):
        build_release_manifest(domains, boundaries, names)


def test_names_are_not_misparsed_as_representative_ids(tmp_path: Path) -> None:
    _, _, names = _files(tmp_path)
    with names.open("a") as handle:
        handle.write("2.10.20.40 ??????? :\n")
    assert read_cath_names(names)["2.10"] == "Synthetic architecture one"
    assert read_cath_names(names)["2.10.20.40"] == ""
    with names.open("a") as handle:
        handle.write("2.10 9aaaA00 :Duplicated\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_cath_names(names)


def test_support_counts_preserve_unmapped_records(tmp_path: Path) -> None:
    frame = build_release_manifest(*_files(tmp_path))
    support = summarize_support(frame).set_index("cath_architecture_code")
    assert support.loc["2.10", "domain_count"] == 3
    assert support.loc["2.10", "s35_count"] == 2
    assert support.loc["2.20", "domain_count"] == 2
    assert support.loc["2.20", "boundary_ok_count"] == 1
    assert len(summarize_support(frame, level="topology")) == 2


def test_representative_selection_only_uses_explicit_availability(tmp_path: Path) -> None:
    frame = build_release_manifest(*_files(tmp_path))
    representatives = select_s35_representatives(frame)
    assert set(representatives.record_id) == {"9aaaA00", "9aacA01", "9aad001", "9aaeA01"}
    # A lower-resolution number alone does not displace the official representative.
    assert "9aabA01" not in set(representatives.record_id)
    available = pd.Series(True, index=frame.record_id)
    available.loc["9aaaA00"] = False
    available.loc["9aaeA01"] = False
    changed = select_s35_representatives(frame, availability=available).set_index("record_id")
    assert changed.loc["9aabA01", "representative_source"] == "availability_fallback"
    assert changed.loc["9aabA01", "official_representative_id"] == "9aaaA00"
    assert changed.loc["9aaeA01", "representative_status"] == "unavailable"
    with pytest.raises(ValueError, match="complete Boolean"):
        select_s35_representatives(frame, availability=available.iloc[:1])


def test_official_representative_file_is_validated(tmp_path: Path) -> None:
    domains, boundaries, names = _files(tmp_path)
    frame = build_release_manifest(domains, boundaries, names)
    official = tmp_path / "s35.txt"
    official.write_text("\n".join(line for line in domains.read_text().splitlines()
                                  if not line.startswith("9aab")) + "\n")
    chosen = select_s35_representatives(frame, official)
    assert set(chosen.representative_source) == {"official"}
    with official.open("a") as handle:
        handle.write("9aabA01 2 10 20 30 1 1 1 1 2 41 1.5\n")
    with pytest.raises(ValueError, match="exactly once"):
        select_s35_representatives(frame, official)
