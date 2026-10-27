"""Synthetic input contracts; these tests do not claim native biological inference."""
import csv
import json

import pytest
from Bio.PDB import MMCIFIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from scripts.external_baselines.beta_barrel_detection.cooper_beta.inputs import select_structure
from scripts.external_baselines.beta_barrel_detection.cooper_beta.learned import (
    FEATURES,
    frozen_selection,
)
from scripts.external_baselines.beta_barrel_detection.cooper_beta.runner import normalize_row
from scripts.external_baselines.beta_barrel_detection.isitabarrel.runner import load_results_tsv
from scripts.external_baselines.beta_barrel_detection.pred_tmbb2.sequences import (
    declared_polymer_sequence,
)
from scripts.external_baselines.beta_barrel_detection.pred_tmbb2.specialist import run_sample


def synthetic_cif(path):
    rows = [(1, "A", 1, "ALA", 1), (2, "B", 1, "GLY", 2),
            (3, "A", 2, "ALA", 1)]
    data = {"data_": "synthetic", "_entity_poly.entity_id": ["1", "2"],
            "_entity_poly.pdbx_seq_one_letter_code_can": ["AGST", "GG"],
            "_atom_site.group_PDB": ["ATOM"] * 3,
            "_atom_site.id": [str(r[0]) for r in rows],
            "_atom_site.type_symbol": ["C"] * 3,
            "_atom_site.label_atom_id": ["CA"] * 3,
            "_atom_site.label_alt_id": ["."] * 3,
            "_atom_site.label_comp_id": [r[3] for r in rows],
            "_atom_site.label_asym_id": [r[1] for r in rows],
            "_atom_site.label_entity_id": [str(r[4]) for r in rows],
            "_atom_site.label_seq_id": ["1"] * 3,
            "_atom_site.pdbx_PDB_ins_code": ["?"] * 3,
            "_atom_site.Cartn_x": ["1.0", "2.0", "3.0"],
            "_atom_site.Cartn_y": ["0.0"] * 3,
            "_atom_site.Cartn_z": ["0.0"] * 3,
            "_atom_site.occupancy": ["1.0"] * 3,
            "_atom_site.B_iso_or_equiv": ["10.0"] * 3,
            "_atom_site.auth_seq_id": ["10"] * 3,
            "_atom_site.auth_asym_id": [r[1] for r in rows],
            "_atom_site.pdbx_PDB_model_num": [str(r[2]) for r in rows]}
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    return path


def test_selected_model_chain_preserves_coordinates_and_polymer(tmp_path):
    source = synthetic_cif(tmp_path / "source.cif")
    selected, metadata = select_structure(source, tmp_path / "selected", chain_id="A", model_index=1)
    result = MMCIF2Dict(str(selected))
    assert result["_atom_site.id"] == ["3"]
    assert result["_atom_site.Cartn_x"] == ["3.0"]
    assert result["_atom_site.auth_seq_id"] == ["10"]
    assert result["_entity_poly.pdbx_seq_one_letter_code_can"] == ["AGST", "GG"]
    assert metadata["model_index"] == 1


def test_declared_sequence_contains_unobserved_positions(tmp_path):
    source = synthetic_cif(tmp_path / "source.cif")
    record = declared_polymer_sequence(source, chain_id="A")
    assert record["sequence"] == "AGST"
    assert record["sequence_source"] == "mmcif_entity_poly_canonical"


def test_declared_sequence_ignores_ligand_entity_sharing_author_chain(tmp_path):
    path = synthetic_cif(tmp_path / "ligand.cif")
    data = MMCIF2Dict(str(path))
    for key, values in data.items():
        if key.startswith("_atom_site."):
            values.append(values[0])
    data["_atom_site.id"][-1] = "4"
    data["_atom_site.label_entity_id"][-1] = "3"
    data["_atom_site.group_PDB"][-1] = "HETATM"
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    assert declared_polymer_sequence(path, chain_id="A")["sequence"] == "AGST"


def test_absent_declared_sequence_never_becomes_observed_fragment(tmp_path):
    path = tmp_path / "source.pdb"
    path.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C\nEND\n")
    with pytest.raises(ValueError, match="SEQRES is absent"):
        declared_polymer_sequence(path, chain_id="A")


def test_declared_monomer_table_supported_without_canonical_string(tmp_path):
    path = synthetic_cif(tmp_path / "source.cif")
    data = MMCIF2Dict(str(path))
    data.pop("_entity_poly.pdbx_seq_one_letter_code_can")
    data.update({"_entity_poly_seq.entity_id": ["1", "1", "1"],
                 "_entity_poly_seq.num": ["1", "2", "3"],
                 "_entity_poly_seq.mon_id": ["ALA", "GLY", "SER"]})
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    result = declared_polymer_sequence(path, chain_id="A")
    assert result["sequence"] == "AGS"
    assert result["sequence_source"] == "mmcif_entity_poly_seq"


@pytest.mark.parametrize("native,prediction,status", [("BARREL", 1, "ok"),
                                                       ("NON_BARREL", 0, "ok"),
                                                       ("ERROR", None, "unavailable")])
def test_cooper_native_decisions_are_not_continuous_scores(native, prediction, status):
    result = normalize_row({"result": native, "degraded": "false"}, sample_id="toy")
    assert result["score"] is None
    assert result["prediction"] == prediction
    assert result["status"] == status


def test_cooper_degraded_output_is_unavailable():
    result = normalize_row({"result": "NON_BARREL", "degraded": "true"}, sample_id="toy")
    assert result["status"] == "unavailable"
    assert result["prediction"] is None


def test_isitabarrel_nonfinite_decision_is_not_negative(tmp_path):
    path = tmp_path / "native.tsv"
    path.write_text("MAP_NAME\tCC2_TO_H4\ntoy\tnan\n")
    with pytest.raises(ValueError, match="Missing score"):
        load_results_tsv(path)


def test_pred_ambiguous_sequence_is_unavailable_without_execution(tmp_path):
    result = run_sample("ACX", sample_id="toy", sequence_source="synthetic",
                        output_dir=tmp_path / "run", juchmme_dir=tmp_path / "missing")
    assert result["status"] == "unavailable"
    assert result["prediction"] is None
    assert "canonical" in result["reason"]
    assert json.loads((tmp_path / "run" / "result.json").read_text())["score"] is None


def test_pred_topology_length_mismatch_is_unavailable(tmp_path, monkeypatch):
    from scripts.external_baselines.beta_barrel_detection.pred_tmbb2 import specialist
    from scripts.external_baselines.beta_barrel_detection.pred_tmbb2.runner import PredTmbb2Result
    row = PredTmbb2Result("toy", "NON_BARREL", 0, 0, "LP_tm_strands>=3", "LP", "ii")
    monkeypatch.setattr(specialist, "run_baseline", lambda *args, **kwargs: [row])
    result = run_sample("ACDE", sample_id="toy", sequence_source="synthetic",
                        output_dir=tmp_path / "run", juchmme_dir=tmp_path)
    assert result["status"] == "unavailable"
    assert result["prediction"] is None
    assert "length" in result["reason"]


def test_isitabarrel_hybrid_native_value_kept_only_as_auxiliary(tmp_path):
    path = tmp_path / "native.tsv"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, ["MAP_NAME", "CC2_TO_H4"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"MAP_NAME": "toy", "CC2_TO_H4": 5})
    native = load_results_tsv(path)[0]
    assert native.result == "BARREL"
    assert native.score == 5


def test_frozen_learned_threshold_uses_declared_selection():
    # Arbitrary synthetic thresholds, independent of fitted research artifacts.
    metadata = {"feature_columns": FEATURES, "machine_learning": {
        "deployment_threshold": 0.37,
        "outer_selections": [{"outer_fold": 2, "threshold": 0.63}]}}
    assert frozen_selection(metadata, "deployment") == 0.37
    assert frozen_selection(metadata, "outer_2") == 0.63
    with pytest.raises(ValueError, match="absent"):
        frozen_selection(metadata, "outer_3")


def test_frozen_learned_feature_order_must_match():
    metadata = {"feature_columns": list(reversed(FEATURES))}
    with pytest.raises(ValueError, match="feature order"):
        frozen_selection(metadata, "deployment")


def test_corrupt_supported_structure_is_unavailable(tmp_path):
    from scripts.external_baselines.beta_barrel_detection.cooper_beta.runner import (
        run_sample as run_cooper,
    )
    path = tmp_path / "corrupt.pdb"
    path.write_text("ATOM      1  CA  ALA A   1       broken  broken  broken  1.00 10.00           C\nEND\n")
    result = run_cooper(path, sample_id="corrupt", chain_id="A", output_dir=tmp_path / "run")
    assert result["status"] == "unavailable"
    assert result["prediction"] is None
