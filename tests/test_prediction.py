"""Synthetic predictor math, checkpoint installation, and annotation contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFIO

from betlas.io.mmcif import read_mmcif_dict
from betlas.prediction import CheckpointError, checkpoints, install_checkpoint, predict_features
from betlas.prediction.models import evaluate, probabilities


def logistic(task="solenoid", intercept=0):
    return {
        "schema_version": "betlas.predictor.v1",
        "release": "test",
        "task": task,
        "feature_schema": "synthetic",
        "feature_columns": ["x", "y"],
        "classes": [0, 1],
        "threshold": 0.5,
        "model": {
            "kind": "logistic",
            "imputer_statistics": [2.0, 1.0],
            "scaler_mean": [1.0, 1.0],
            "scaler_scale": [2.0, 1.0],
            "coef": [[2.0, -1.0]],
            "intercept": [intercept],
        },
    }


def save_model(tmp_path, model):
    path = tmp_path / (model["task"] + ".json")
    path.write_text(json.dumps(model))
    return path


def test_logistic_imputation_matches_hand_calculated_logit():
    model = logistic(intercept=0.5)
    result = evaluate(model, {"x": np.nan, "y": 2.0})
    assert result["probability"] == pytest.approx(1 / (1 + np.exp(-0.5)))
    assert result["is_solenoid"] and result["imputed_features"] == ["x"]
    with pytest.raises(ValueError, match="required descriptor"):
        evaluate(model, {"x": 1.0})
    with pytest.raises(ValueError, match="schema"):
        evaluate(model, {"x": 1.0, "y": 1.0, "feature_schema": "old"})
    with pytest.raises(ValueError, match="All descriptor"):
        evaluate(model, {"x": np.nan, "y": np.nan})


def test_no_beta_parse_flag_does_not_invalidate_finite_recognition_measurements():
    result = evaluate(logistic(), {"x": 0.0, "y": 0.0, "betlas_parse_ok": 0})
    assert result["status"] == "ok"


def test_numerical_symmetric_tree_float32_and_split_order():
    model = {
        "feature_columns": ["x", "y"],
        "classes": [8, 10, 12],
        "model": {
            "kind": "catboost",
            "fill_value": 0.0,
            "catboost": {
                "features_info": {
                    "float_features": [
                        {"feature_index": 0, "flat_feature_index": 0},
                        {"feature_index": 1, "flat_feature_index": 1},
                    ]
                },
                "scale_and_bias": [2.0, [0.0, 0.1, -0.1]],
                "oblivious_trees": [
                    {
                        "splits": [
                            {"split_type": "FloatFeature", "float_feature_index": 1, "border": 0.0},
                            {"split_type": "FloatFeature", "float_feature_index": 0, "border": 1.0},
                        ],
                        "leaf_values": [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            -1.0,
                            2.0,
                            0.0,
                            -2.0,
                            3.0,
                            0.0,
                            -3.0,
                        ],
                    }
                ],
            },
        },
    }
    result, missing = probabilities(model, {"x": 1.0 + 1e-9, "y": 1.0})
    logits = np.array([2.0, 0.1, -2.1])
    expected = np.exp(logits) / np.exp(logits).sum()
    np.testing.assert_allclose(result, expected)
    assert missing == []
    _, missing = probabilities(model, {"x": None, "y": 1.0})
    assert missing == ["x"]


def test_local_install_reuse_corruption_and_offline(tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    source = save_model(mirror, logistic())
    entry = {
        "filename": source.name,
        "byte_size": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(checkpoints, "_manifest", lambda: {"solenoid": entry})
    cache = tmp_path / "cache"
    installed = install_checkpoint(
        "solenoid", cache_dir=cache, base_url=mirror, allow_download=False
    )
    assert installed.read_bytes() == source.read_bytes()
    source.unlink()
    assert install_checkpoint("solenoid", cache_dir=cache, allow_download=False) == installed
    installed.write_text("corrupt")
    with pytest.raises(CheckpointError, match="not installed"):
        install_checkpoint("solenoid", cache_dir=cache, allow_download=False)
    source.write_text("unverified")
    with pytest.raises(CheckpointError, match="SHA-256"):
        install_checkpoint("solenoid", cache_dir=cache, base_url=mirror, allow_download=False)
    assert installed.read_text() == "corrupt"
    assert not list(installed.parent.glob(".checkpoint-*"))


def test_feature_counting_gate_uses_separate_recognition_measurements(tmp_path):
    barrel = save_model(tmp_path, logistic("barrel", -10))
    with pytest.raises(ValueError, match="assume_barrel"):
        predict_features("barrel-strands", {"x": 1.0, "y": 1.0}, allow_download=False)
    result = predict_features(
        "barrel-strands",
        {},
        barrel_features={"x": 1.0, "y": 1.0},
        barrel_checkpoint=barrel,
        allow_download=False,
    )
    assert result["status"] == "not_barrel" and result["strand_count"] is None


def _two_chain_fixture(path):
    data = read_mmcif_dict(Path(__file__).resolve().parents[1] / "examples/mini.cif")
    data = {
        key: copy.deepcopy(value)
        for key, value in data.items()
        if key == "data_" or key.startswith("_atom_site.")
    }
    n = len(data["_atom_site.id"])
    for key in list(data):
        if not key.startswith("_atom_site."):
            continue
        first = data[key]
        second = first.copy()
        if key in ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"):
            second = ["B"] * n
        if key == "_atom_site.id":
            second = [str(n + i + 1) for i in range(n)]
        data[key] = first + second
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))


@pytest.mark.parametrize(
    "task,ranges,expected_chains,expected_count",
    [
        ("solenoid", "1-14:A", {"A"}, 8),
        ("barrel", "", {"A"}, 16),
        ("barrel-strands", "", {"A", "B"}, 32),
    ],
)
def test_annotation_context_matches_task_training(
    tmp_path, monkeypatch, task, ranges, expected_chains, expected_count
):
    from betlas.prediction import structure

    source = tmp_path / "toy.cif"
    _two_chain_fixture(source)
    inspected = []

    def annotate(source, target, **kwargs):
        data = read_mmcif_dict(source)
        inspected.append((set(data["_atom_site.auth_asym_id"]), len(data["_atom_site.id"])))
        data["_dssp_struct_summary.label_seq_id"] = ["1"]
        writer = MMCIFIO()
        writer.set_dict(data)
        writer.save(str(target))
        return {"dssp_version": "synthetic annotation stand-in"}

    monkeypatch.setattr(structure, "annotate_structure", annotate)
    with structure.measured_structure(source, task=task, chain_id="A", residue_ranges=ranges) as (
        features,
        _,
        meta,
    ):
        assert features["betlas_beta_strand_count"] == 0
        assert meta["chain_id"] == "A"
    assert inspected == [(expected_chains, expected_count)]


def test_pdb_model_selection_preserves_author_chain_and_numbering(tmp_path):
    from betlas.prediction.structure import _model_input

    source = tmp_path / "toy.pdb"
    lines = []
    for model, x in [(1, 1.0), (2, 8.0)]:
        lines.extend(
            [
                f"MODEL     {model:4d}\n",
                f"ATOM      1  CA  ALA Q  11    {x:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00  0.00           C  \n",
                "ENDMDL\n",
            ]
        )
    source.write_text("".join(lines) + "END\n")
    selected = _model_input(source, tmp_path / "selected.cif", model_id=1)
    data = read_mmcif_dict(selected)
    assert data["_atom_site.auth_asym_id"] == ["Q"]
    assert data["_atom_site.auth_seq_id"] == ["11"]
    assert data["_atom_site.pdbx_PDB_model_num"] == ["1"]
    assert float(data["_atom_site.Cartn_x"][0]) == 8.0


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        1,
        "model",
        {"schema_version": "betlas.predictor.v1", "task": "solenoid", "feature_columns": [[]]},
    ],
)
def test_supplied_checkpoint_reports_invalid_json_shape(tmp_path, payload):
    from betlas.prediction.models import load_model

    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(CheckpointError):
        load_model(path, "solenoid")


@pytest.mark.parametrize(
    "data",
    [
        {"_struct_sheet.id": ["AA", "AA"]},
        {"_struct_sheet_range.sheet_id": ["A", "A"], "_struct_sheet_range.id": ["r", "r"]},
        {"_dssp_struct_ladder.id": ["AA", "AA"]},
    ],
)
def test_duplicate_dssp_writer_identifiers_are_rejected(data):
    from betlas.prediction.structure import _check_annotation_ids

    with pytest.raises(ValueError, match="DSSP 4.6.1"):
        _check_annotation_ids(data)
