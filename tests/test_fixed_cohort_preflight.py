from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

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
