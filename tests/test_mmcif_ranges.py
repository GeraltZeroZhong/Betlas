from __future__ import annotations

from betlas.io.mmcif import parse_residue_ranges


def test_parse_cath_chopping_ranges() -> None:
    assert parse_residue_ranges("2-78:A,187-208:A") == [("A", 2, 78), ("A", 187, 208)]
    assert parse_residue_ranges("-1-56:A") == [("A", -1, 56)]
    assert parse_residue_ranges("153-0:B") == [("B", 0, 153)]
