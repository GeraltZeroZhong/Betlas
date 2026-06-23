from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.reproducibility.readout_benchmarks.beta_barrel_detection.run_feature_block_ablation import (
    _align_record_frame as align_feature_block_records,
)
from scripts.reproducibility.readout_benchmarks.beta_barrel_detection.run_feature_block_ablation import (
    write_split_preflight as write_feature_block_split_preflight,
)
from scripts.reproducibility.readout_benchmarks.beta_barrel_staves.run_betlas151_layer_radial16_official import (
    _align_record_frame as align_staves_records,
)
from scripts.reproducibility.readout_benchmarks.beta_barrel_staves.run_betlas151_layer_radial16_official import (
    build_fold_preflight,
    write_fold_preflight,
)
from scripts.reproducibility.readouts.beta_barrel_detection.betlas_readout import (
    write_split_preflight,
)


def test_staves_fixed_cohort_preflight_fails_on_pdb_fold_leakage(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {"record_id": "r1", "pdb_id": "LEAK", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 0},
            {"record_id": "r2", "pdb_id": "LEAK", "auth_chain_id": "B", "reference_count": 10, "outer_fold": 1},
            {"record_id": "r3", "pdb_id": "SAFE1", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 1},
            {"record_id": "r4", "pdb_id": "SAFE2", "auth_chain_id": "A", "reference_count": 10, "outer_fold": 0},
        ]
    )

    preflight = build_fold_preflight(rows)

    assert preflight["status"] == "failed"
    assert preflight["failure_stage"] == "fold_contract"
    assert preflight["pdb_fold_leakage"][0]["pdb_id"] == "LEAK"
    with pytest.raises(ValueError, match="pdb_id appears in more than one outer_fold"):
        write_fold_preflight(rows, tmp_path)
    written = json.loads((tmp_path / "fold_preflight.json").read_text(encoding="utf-8"))
    assert written["status"] == "failed"


def test_staves_fixed_cohort_preflight_detects_case_insensitive_pdb_fold_leakage() -> None:
    rows = pd.DataFrame(
        [
            {"record_id": "r1", "pdb_id": "Leak", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 0},
            {"record_id": "r2", "pdb_id": "leak", "auth_chain_id": "B", "reference_count": 10, "outer_fold": 1},
            {"record_id": "r3", "pdb_id": "SAFE1", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 1},
            {"record_id": "r4", "pdb_id": "SAFE2", "auth_chain_id": "A", "reference_count": 10, "outer_fold": 0},
        ]
    )

    preflight = build_fold_preflight(rows)

    assert preflight["status"] == "failed"
    assert preflight["pdb_fold_leakage"][0]["pdb_id"] == "LEAK"


def test_staves_fixed_cohort_preflight_fails_when_test_fold_lacks_reference_class() -> None:
    rows = pd.DataFrame(
        [
            {"record_id": "r1", "pdb_id": "P1", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 0},
            {"record_id": "r2", "pdb_id": "P2", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 0},
            {"record_id": "r3", "pdb_id": "P3", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 1},
            {"record_id": "r4", "pdb_id": "P4", "auth_chain_id": "A", "reference_count": 10, "outer_fold": 1},
        ]
    )

    preflight = build_fold_preflight(rows)

    assert preflight["status"] == "failed"
    assert "test partition lacks reference_count classes" in preflight["error"]


def test_staves_fixed_cohort_preflight_records_fold_count_coverage() -> None:
    rows = pd.DataFrame(
        [
            {"record_id": "r1", "pdb_id": "P1", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 0},
            {"record_id": "r2", "pdb_id": "P2", "auth_chain_id": "A", "reference_count": 10, "outer_fold": 0},
            {"record_id": "r3", "pdb_id": "P3", "auth_chain_id": "A", "reference_count": 8, "outer_fold": 1},
            {"record_id": "r4", "pdb_id": "P4", "auth_chain_id": "A", "reference_count": 10, "outer_fold": 1},
        ]
    )

    preflight = build_fold_preflight(rows)

    assert preflight["status"] == "ok"
    assert preflight["reference_count_classes"] == [8, 10]
    assert preflight["folds"][0]["train_reference_count_classes"] == [8, 10]
    assert preflight["folds"][0]["test_reference_count_classes"] == [8, 10]


def test_detection_fixed_cohort_split_preflight_writes_ok_and_failed_states(tmp_path) -> None:
    ok = write_split_preflight(
        out_dir=tmp_path / "ok",
        y=np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
        groups=np.asarray(["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]),
        n_splits=2,
        seed=13,
    )
    assert ok["status"] == "ok"
    assert ok["effective_n_splits"] == 2
    assert (tmp_path / "ok" / "fixed_cohort_split_preflight.json").exists()

    with pytest.raises(ValueError, match="requires both binary classes"):
        write_split_preflight(
            out_dir=tmp_path / "failed",
            y=np.asarray([1, 1, 1, 1]),
            groups=np.asarray(["g0", "g1", "g2", "g3"]),
            n_splits=2,
            seed=13,
        )
    failed = json.loads((tmp_path / "failed" / "fixed_cohort_split_preflight.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["failure_stage"] == "class_coverage"


def test_detection_fixed_cohort_split_preflight_canonicalizes_group_case_and_space(tmp_path) -> None:
    ok = write_split_preflight(
        out_dir=tmp_path / "canonical",
        y=np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
        groups=np.asarray(["g0", " G0 ", "g1", "G1", "g2", "G2", "g3", "G3"]),
        n_splits=2,
        seed=13,
    )

    assert ok["status"] == "ok"
    assert ok["group_count"] == 4


def test_fixed_cohort_record_alignment_rejects_duplicate_source_ids(tmp_path) -> None:
    rows = pd.DataFrame({"record_id": ["r1", "r2"]})
    duplicate_source = pd.DataFrame(
        [
            {"record_id": "r1", "feature": 1.0},
            {"record_id": "r1", "feature": 2.0},
            {"record_id": "r2", "feature": 3.0},
        ]
    )

    with pytest.raises(ValueError, match="duplicate record_id"):
        align_feature_block_records(
            duplicate_source,
            ["r1", "r2"],
            path=tmp_path / "features.csv",
            frame_name="feature table",
        )
    with pytest.raises(ValueError, match="duplicate record_id"):
        align_staves_records(
            duplicate_source,
            rows,
            path=tmp_path / "features.csv",
            frame_name="feature table",
        )


def test_feature_block_ablation_split_preflight_fails_before_model_for_one_class(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires both binary classes"):
        write_feature_block_split_preflight(
            out_dir=tmp_path / "feature_block",
            y=np.asarray([1, 1, 1, 1]),
            groups=np.asarray(["g0", "g1", "g2", "g3"]),
            n_splits=2,
            seed=13,
        )

    written = json.loads(
        (tmp_path / "feature_block" / "fixed_cohort_split_preflight.json").read_text(encoding="utf-8")
    )
    assert written["status"] == "failed"
    assert written["failure_stage"] == "class_coverage"
