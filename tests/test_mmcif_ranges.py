from __future__ import annotations

import pytest

from betlas.io.mmcif import build_structure_geometry, parse_residue_ranges
from betlas.models import DomainCandidate


def test_parse_cath_chopping_ranges() -> None:
    assert parse_residue_ranges("2-78:A,187-208:A") == [("A", 2, 78), ("A", 187, 208)]
    assert parse_residue_ranges("A:2-78") == [("A", 2, 78)]
    assert parse_residue_ranges("2-78", fallback_chain_id="A") == [("A", 2, 78)]
    assert parse_residue_ranges("-1-56:A") == [("A", -1, 56)]
    assert parse_residue_ranges("153-0:B") == [("B", 0, 153)]


def test_parse_residue_ranges_rejects_unparsed_input() -> None:
    with pytest.raises(ValueError, match="invalid residue range"):
        parse_residue_ranges("A/2-78")


def test_parse_residue_ranges_rejects_conflicting_chain_ids() -> None:
    with pytest.raises(ValueError, match="inconsistent chain ids"):
        parse_residue_ranges("A:10-20:B")


def test_parse_residue_ranges_rejects_insertion_code_ranges() -> None:
    with pytest.raises(ValueError, match="numeric author residue ranges only"):
        parse_residue_ranges("10A-20:A")
    with pytest.raises(ValueError, match="numeric author residue ranges only"):
        parse_residue_ranges("A:10-20B")


def test_build_structure_geometry_rejects_selected_insertion_code_residues(tmp_path) -> None:
    structure = tmp_path / "insertion.cif"
    structure.write_text(
        "\n".join(
            [
                "data_insertion",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.auth_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.label_asym_id",
                "_atom_site.auth_seq_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "ATOM 1 C CA CA . ALA A A 1 1 ? 0.000 0.000 0.000 1.00",
                "ATOM 2 C CA CA . ALA A A 2 2 A 1.000 0.000 0.000 1.00",
                "ATOM 3 C CA CA . ALA A A 3 3 ? 2.000 0.000 0.000 1.00",
                "loop_",
                "_struct_sheet_range.sheet_id",
                "_struct_sheet_range.id",
                "_struct_sheet_range.beg_auth_asym_id",
                "_struct_sheet_range.end_auth_asym_id",
                "_struct_sheet_range.beg_auth_seq_id",
                "_struct_sheet_range.end_auth_seq_id",
                "S1 1 A A 1 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    domain = DomainCandidate(
        record_id="r",
        pdb_id="test",
        chain_id="A",
        domain_id="d",
        residue_ranges="",
        fold_label_final="unlabeled",
        evidence_level="unit",
        label_source_primary="unit",
    )

    with pytest.raises(ValueError, match="insertion-coded residue"):
        build_structure_geometry(domain, structure)
