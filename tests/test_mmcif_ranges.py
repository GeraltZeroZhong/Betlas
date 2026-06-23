from __future__ import annotations

import pytest

from betlas.io.mmcif import build_structure_geometry, inspect_mmcif_chains, parse_residue_ranges
from betlas.models import DomainCandidate


def _domain(*, residue_ranges: str = "") -> DomainCandidate:
    return DomainCandidate(
        record_id="r",
        pdb_id="test",
        chain_id="A",
        domain_id="d",
        residue_ranges=residue_ranges,
        fold_label_final="unlabeled",
        evidence_level="unit",
        label_source_primary="unit",
    )


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
    domain = _domain()

    with pytest.raises(ValueError, match="insertion-coded residue"):
        build_structure_geometry(domain, structure)


def test_build_structure_geometry_keeps_protein_like_hetatm_mse(tmp_path) -> None:
    structure = tmp_path / "mse.cif"
    structure.write_text(
        "\n".join(
            [
                "data_mse",
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
                "HETATM 1 C CA CA . MSE A A 1 1 ? 0.000 0.000 0.000 1.00",
                "HETATM 2 C CA CA . MSE A A 2 2 ? 1.000 0.000 0.000 1.00",
                "HETATM 3 C CA CA . MSE A A 3 3 ? 2.000 0.000 0.000 1.00",
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

    geometry = build_structure_geometry(_domain(), structure)

    assert [residue.residue_name for residue in geometry.residues] == ["MSE", "MSE", "MSE"]
    assert len(geometry.beta_segments) == 1


def test_inspect_counts_protein_like_hetatm_ca_residues_not_atoms(tmp_path) -> None:
    structure = tmp_path / "mse_atoms.cif"
    structure.write_text(
        "\n".join(
            [
                "data_mse_atoms",
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
                "HETATM 1 N N N . MSE A A 1 1 ? 0.000 0.000 0.000 1.00",
                "HETATM 2 C CA CA . MSE A A 1 1 ? 1.000 0.000 0.000 1.00",
                "HETATM 3 C C C . MSE A A 1 1 ? 2.000 0.000 0.000 1.00",
                "HETATM 4 O O O . MSE A A 1 1 ? 3.000 0.000 0.000 1.00",
                "HETATM 5 SE SE SE . MSE A A 1 1 ? 4.000 0.000 0.000 1.00",
                "",
            ]
        ),
        encoding="utf-8",
    )

    row = inspect_mmcif_chains(structure)[0]

    assert row["standard_ca_residue_count"] == 1
    assert row["protein_like_hetatm_ca_count"] == 1


def test_build_structure_geometry_rejects_nonnumeric_selected_author_residue_id(tmp_path) -> None:
    structure = tmp_path / "nonnumeric.cif"
    structure.write_text(
        "\n".join(
            [
                "data_nonnumeric",
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
                "ATOM 2 C CA CA . ALA A A 2A 2 ? 1.000 0.000 0.000 1.00",
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

    with pytest.raises(ValueError, match="numeric author residue IDs only"):
        build_structure_geometry(_domain(), structure)


def test_build_structure_geometry_rejects_nonnumeric_author_residue_id_inside_selected_range(tmp_path) -> None:
    structure = tmp_path / "range_nonnumeric.cif"
    structure.write_text(
        "\n".join(
            [
                "data_range_nonnumeric",
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
                "ATOM 2 C CA CA . ALA A A 2A 2 ? 1.000 0.000 0.000 1.00",
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

    with pytest.raises(ValueError, match="numeric author residue IDs only"):
        build_structure_geometry(_domain(residue_ranges="1-3:A"), structure)


def test_out_of_range_insertion_code_sheet_boundary_does_not_block_selected_range(tmp_path) -> None:
    structure = tmp_path / "out_of_range_insertion.cif"
    structure.write_text(
        "\n".join(
            [
                "data_out_of_range_insertion",
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
                "ATOM 2 C CA CA . ALA A A 2 2 ? 1.000 0.000 0.000 1.00",
                "ATOM 3 C CA CA . ALA A A 3 3 ? 2.000 0.000 0.000 1.00",
                "loop_",
                "_struct_sheet_range.sheet_id",
                "_struct_sheet_range.id",
                "_struct_sheet_range.beg_auth_asym_id",
                "_struct_sheet_range.end_auth_asym_id",
                "_struct_sheet_range.beg_auth_seq_id",
                "_struct_sheet_range.end_auth_seq_id",
                "_struct_sheet_range.pdbx_beg_PDB_ins_code",
                "_struct_sheet_range.pdbx_end_PDB_ins_code",
                "S1 1 A A 1 3 ? ?",
                "S1 2 A A 10 12 A ?",
                "",
            ]
        ),
        encoding="utf-8",
    )

    geometry = build_structure_geometry(_domain(residue_ranges="1-3:A"), structure)

    assert len(geometry.beta_segments) == 1


def test_build_structure_geometry_respects_mmcif_model_id(tmp_path) -> None:
    structure = tmp_path / "two_models.cif"
    atom_rows = []
    atom_id = 1
    for model_num, offset in [(1, 0.0), (2, 100.0)]:
        for seq_id in [1, 2, 3]:
            atom_rows.append(
                f"ATOM {atom_id} {model_num} C CA CA . ALA A A {seq_id} {seq_id} ? "
                f"{offset + seq_id:.3f} 0.000 0.000 1.00"
            )
            atom_id += 1
    structure.write_text(
        "\n".join(
            [
                "data_two_models",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.pdbx_PDB_model_num",
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
                *atom_rows,
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

    model_1 = build_structure_geometry(_domain(), structure)
    model_2 = build_structure_geometry(
        DomainCandidate.from_mapping({**_domain().__dict__, "model_id": 1}),
        structure,
    )
    inspect_row = inspect_mmcif_chains(structure)[0]

    assert [residue.coord_ca[0] for residue in model_1.residues] == [1.0, 2.0, 3.0]
    assert [residue.coord_ca[0] for residue in model_2.residues] == [101.0, 102.0, 103.0]
    assert inspect_row["model_ids"] == ["0", "1"]


def test_range_selection_rejects_nonnumeric_author_residue_id_on_selected_chain(tmp_path) -> None:
    structure = tmp_path / "range_skips_nonnumeric.cif"
    structure.write_text(
        "\n".join(
            [
                "data_range_skips_nonnumeric",
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
                "ATOM 2 C CA CA . ALA A A 2 2 ? 1.000 0.000 0.000 1.00",
                "ATOM 3 C CA CA . ALA A A 3 3 ? 2.000 0.000 0.000 1.00",
                "ATOM 4 C CA CA . ALA A A X99 99 ? 99.000 0.000 0.000 1.00",
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

    with pytest.raises(ValueError, match="numeric author residue IDs only"):
        build_structure_geometry(_domain(residue_ranges="1-3:A"), structure)


def test_reversed_sheet_range_overlaps_narrow_selected_range(tmp_path) -> None:
    structure = tmp_path / "reversed_sheet.cif"
    structure.write_text(
        "\n".join(
            [
                "data_reversed_sheet",
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
                "ATOM 2 C CA CA . ALA A A 2 2 ? 1.000 0.000 0.000 1.00",
                "ATOM 3 C CA CA . ALA A A 3 3 ? 2.000 0.000 0.000 1.00",
                "loop_",
                "_struct_sheet_range.sheet_id",
                "_struct_sheet_range.id",
                "_struct_sheet_range.beg_auth_asym_id",
                "_struct_sheet_range.end_auth_asym_id",
                "_struct_sheet_range.beg_auth_seq_id",
                "_struct_sheet_range.end_auth_seq_id",
                "S1 1 A A 3 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    geometry = build_structure_geometry(_domain(residue_ranges="2-3:A"), structure)

    assert len(geometry.beta_segments) == 1
    assert geometry.beta_segments[0].start_auth_seq_id == 2
    assert geometry.beta_segments[0].end_auth_seq_id == 3
