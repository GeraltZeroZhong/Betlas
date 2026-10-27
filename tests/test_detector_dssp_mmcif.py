"""DSSP input and residue identities for supported multi-character chains."""
import os

from Bio.PDB import MMCIFIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from betlas.readouts.beta_barrel_detection import loader as detector_loader


def test_multichar_export_keeps_polymer_metadata_and_selected_model(tmp_path):
    data = {
        "data_": "synthetic", "_entity_poly.entity_id": ["1"],
        "_entity_poly.type": ["polypeptide(L)"],
        "_entity_poly.pdbx_seq_one_letter_code_can": ["A"],
        "_atom_site.group_PDB": ["ATOM", "ATOM"],
        "_atom_site.id": ["1", "2"], "_atom_site.type_symbol": ["C", "C"],
        "_atom_site.label_atom_id": ["CA", "CA"], "_atom_site.label_alt_id": [".", "."],
        "_atom_site.label_comp_id": ["ALA", "ALA"],
        "_atom_site.label_asym_id": ["A", "A"], "_atom_site.label_entity_id": ["1", "1"],
        "_atom_site.label_seq_id": ["1", "1"], "_atom_site.pdbx_PDB_ins_code": ["?", "?"],
        "_atom_site.Cartn_x": ["1.0", "2.0"], "_atom_site.Cartn_y": ["0.0", "0.0"],
        "_atom_site.Cartn_z": ["0.0", "0.0"], "_atom_site.occupancy": ["1.0", "1.0"],
        "_atom_site.B_iso_or_equiv": ["10.0", "10.0"],
        "_atom_site.auth_seq_id": ["10", "10"], "_atom_site.auth_asym_id": ["AAA", "AAA"],
        "_atom_site.pdbx_PDB_model_num": ["1", "2"],
    }
    source = tmp_path / "models.cif"
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(source))
    loader = detector_loader.ProteinLoader(str(source), model_id=1)
    exported = loader._export_protein_only_structure()
    try:
        actual = MMCIF2Dict(exported)
        assert actual["_entity_poly.pdbx_seq_one_letter_code_can"] == ["A"]
        assert actual["_atom_site.pdbx_PDB_model_num"] == ["2"]
        assert actual["_atom_site.Cartn_x"] == ["2.0"]
        assert actual["_atom_site.auth_asym_id"] == ["AAA"]
    finally:
        os.unlink(exported)


def test_multichar_dssp_uses_mmcif_residue_summary(monkeypatch):
    loader = object.__new__(detector_loader.ProteinLoader)
    loader.dssp_bin = "mock-dssp"
    monkeypatch.setattr(loader, "_has_multichar_chain_ids", lambda: True)
    monkeypatch.setattr(detector_loader, "require_dssp_binary", lambda value: value)
    expected = {("AAA", (" ", 10, " ")): "E"}
    monkeypatch.setattr(loader, "_run_dssp_mmcif_output", lambda path, binary: expected)
    monkeypatch.setattr(detector_loader, "DSSP", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Classic DSSP cannot carry the author chain identifier")))
    assert loader._run_dssp("source.cif", file_type="MMCIF") == expected
