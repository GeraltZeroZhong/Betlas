from __future__ import annotations

import numpy as np
import pandas as pd

from betlas.external_baselines.context import _metric_rows, _normalise_embeddings


def test_normalise_embeddings_handles_zero_rows() -> None:
    embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalised = _normalise_embeddings(embeddings)

    assert np.allclose(normalised[0], [0.6, 0.8])
    assert np.allclose(normalised[1], [0.0, 0.0])


def test_metric_rows_reports_coverage_and_top2_accuracy() -> None:
    predictions = pd.DataFrame(
        [
            {
                "true_label": "beta_barrel",
                "pred_label": "beta_barrel",
                "top2_labels": "beta_barrel;beta_sandwich",
            },
            {
                "true_label": "jelly_roll",
                "pred_label": "beta_sandwich",
                "top2_labels": "beta_sandwich;jelly_roll",
            },
            {
                "true_label": "beta_propeller",
                "pred_label": "",
                "top2_labels": "",
            },
        ]
    )

    rows = _metric_rows(predictions, model="example", dataset="unit")
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["accuracy"]["n"] == 2
    assert by_metric["accuracy"]["coverage"] == 2 / 3
    assert by_metric["top2_accuracy"]["value"] == 1.0
