from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from betlas.constants import FOLD_LABELS
from betlas.publication.evidence import (
    _confusion_by_group,
    _f1_from_confusion,
    _label_indices,
    _representative_subset,
)


def test_confusion_f1_matches_sklearn() -> None:
    y_true = np.array(
        [
            "beta_barrel",
            "beta_barrel",
            "beta_sandwich",
            "jelly_roll",
            "jelly_roll",
            "beta_propeller",
        ]
    )
    y_pred = np.array(
        [
            "beta_barrel",
            "beta_sandwich",
            "beta_sandwich",
            "jelly_roll",
            "beta_sandwich",
            "beta_propeller",
        ]
    )
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    _, confusion = _confusion_by_group(y_true, y_pred, groups)

    observed = _f1_from_confusion(confusion.sum(axis=0), _label_indices(tuple(FOLD_LABELS)))
    expected = f1_score(y_true, y_pred, labels=list(FOLD_LABELS), average="macro", zero_division=0)

    assert observed == expected


def test_representative_subset_prefers_direct_fuller_domain() -> None:
    rows = pd.DataFrame(
        [
            {
                "record_id": "a1",
                "pdb_id": "1aaa",
                "fold_label_final": "beta_barrel",
                "cath_s35_cluster_id": "g1",
                "cath_homology_code": "h1",
                "cath_s35_source": "homology_fallback",
                "cz_parse_ok": "1",
                "cz_beta_residue_count": "20",
                "cz_residue_count": "100",
                "cz_beta_strand_count": "4",
            },
            {
                "record_id": "a2",
                "pdb_id": "2aaa",
                "fold_label_final": "beta_barrel",
                "cath_s35_cluster_id": "g1",
                "cath_homology_code": "h1",
                "cath_s35_source": "cath_s35",
                "cz_parse_ok": "1",
                "cz_beta_residue_count": "10",
                "cz_residue_count": "80",
                "cz_beta_strand_count": "3",
            },
            {
                "record_id": "b1",
                "pdb_id": "3bbb",
                "fold_label_final": "jelly_roll",
                "cath_s35_cluster_id": "g2",
                "cath_homology_code": "h2",
                "cath_s35_source": "homology_fallback",
                "cz_parse_ok": "0",
                "cz_beta_residue_count": "50",
                "cz_residue_count": "120",
                "cz_beta_strand_count": "6",
            },
            {
                "record_id": "b2",
                "pdb_id": "4bbb",
                "fold_label_final": "jelly_roll",
                "cath_s35_cluster_id": "g2",
                "cath_homology_code": "h2",
                "cath_s35_source": "homology_fallback",
                "cz_parse_ok": "1",
                "cz_beta_residue_count": "40",
                "cz_residue_count": "110",
                "cz_beta_strand_count": "5",
            },
        ]
    )

    subset, summary = _representative_subset(
        rows,
        {
            "name": "test",
            "key_strategy": "first_nonempty",
            "group_columns": ["cath_s35_cluster_id", "cath_homology_code", "pdb_id"],
        },
    )

    assert subset["record_id"].tolist() == ["a2", "b2"]
    assert int(summary["rows"].sum()) == 2
