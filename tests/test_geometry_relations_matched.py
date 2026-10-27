from __future__ import annotations

import pandas as pd

from betlas.geometry_relations.cohorts import H
from betlas.geometry_relations.matched import match_metadata


def test_matching_uses_covariates_and_excludes_shared_identities():
    def frame(rows):
        return pd.DataFrame(
            rows,
            columns=[
                "observed_length",
                "beta_fraction",
                "strand_count",
                "missing_fraction",
                H,
                "pdb_id",
                "outcome",
            ],
        )

    left = frame([[100, 0.5, 8, 0, "h1", "p1", 1000], [120, 0.6, 10, 0, "h2", "p2", -1000]])
    right = frame(
        [
            [100, 0.5, 8, 0, "h1", "p3", 1000],
            [100, 0.5, 8, 0, "h3", "p1", 1000],
            [105, 0.5, 8, 0, "h4", "p4", -1000],
            [120, 0.6, 10, 0, "h5", "p5", 1000],
            [1000, 0.5, 8, 0, "h6", "p6", 1000],
        ]
    )
    first = match_metadata(left, right)
    assert {(i, j) for i, j, _ in first} == {(0, 2), (1, 3)}
    left["outcome"] *= -1
    right["outcome"] *= -1
    assert match_metadata(left, right) == first
    right.loc[:, "strand_count"] = 99
    assert match_metadata(left, right) == []
