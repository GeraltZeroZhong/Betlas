from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betlas.geometry_relations.cohorts import A, H, T
from betlas.geometry_relations.packing import PACKING_BLOCKS
from betlas.geometry_relations.phenotypes import run_phenotype_analysis


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = ["j1", "j2", "j3", "s1", "s2", "b1", "f1", "o1"]
    cohort = pd.DataFrame({
        "record_id": ids, A: ["2.60"] * 5 + ["2.40", "2.60", "2.70"],
        T: ["2.60.120"] * 3 + ["2.60.40", "2.60.40", "2.40.10", "2.60.50", "2.70.10"],
        H: ["hj1", "hj1", "hj2", "hs1", "hs2", "hb", "hf", "ho"],
        "role": ["discovery", "discovery", "confirmation", "confirmation", "discovery", "descriptive_single_h", "confirmation", "descriptive_single_h"],
        "feature_status": ["ok"] * 6 + ["annotation_failure", "ok"],
        "betlas_parse_ok": [1] * 6 + [0, 1],
        "betlas_residue_count": [100, 101, 201, 200, 300, 120, 0, 80],
        "betlas_beta_residue_fraction": [0.4, 0.5, 0.6, 0.7, 0.2, 0.5, 0, 0.3],
        "betlas_beta_residue_count": [40] * 8,
        "betlas_beta_strand_count": [8, 9, 10, 8, 7, 6, 0, 5],
        "betlas_sheet_count": [2, 2, 2, 3, 2, 1, 0, 2],
        "betlas_sheet_face_count": [2, 1, 2, 2, 2, 1, 0, 2],
        "betlas_axis_best_slice_count": [5] * 8,
        "betlas_sheet_pair_centroid_distance_norm": [0.4, 0.4, 0.6, 0.8, 0.5, 0.0, 0.0, 0.7],
    })
    for columns in PACKING_BLOCKS.values():
        for column in columns:
            cohort[column] = [0.0, 0.0, 1.0, 0.6, 0.7, 0.9, 0.0, 0.3]
    connectivity = pd.DataFrame({
        "record_id": ids,
        "status": ["ok"] * 6 + ["annotation_unavailable", "ok"],
        "pairing_status": ["available", "available", "partial_clipped_pairings", "no_retained_pairings", "annotation_pairing_missing", "available", "", "available"],
        "pairing_node_count": [8, 9, 10, 8, 7, 6, np.nan, 5],
        "pairing_edge_count": [7, 8, 16, 0, 99, 6, np.nan, 4],
        "pairing_cycle_rank": [0, 0, 7, 0, 99, 1, np.nan, 0],
        "pairing_cycle_rank_per_node": [0, 0, 0.7, 0, 99, 1 / 6, np.nan, 0],
        "pairing_branch_fraction": [0, 0, 0.5, 0, 99, 0, np.nan, 0],
        "pairing_strand_rank_gap_mean": [2, 3, 4, np.nan, 99, 1.5, np.nan, 2],
    })
    geometry = pd.DataFrame({
        "record_id": ids, "near_barrel": [True, pd.NA, False, False, True, False, pd.NA, True],
        "label_overlap": [True, pd.NA, False, False, True, False, pd.NA, True],
    })
    return cohort, connectivity, geometry


def _run(tmp_path: Path) -> dict[str, pd.DataFrame]:
    protocol = run_phenotype_analysis(*_inputs(), tmp_path)
    assert protocol["domains"] == 8
    return {name: pd.read_csv(tmp_path / name) for name in protocol["artifacts"]}


def test_H1_keeps_domain_median_and_H_equal_means_distinct(tmp_path: Path) -> None:
    tables = _run(tmp_path)
    summary = tables["H1-distributions.csv"]
    row = summary.loc[
        summary.scope.eq("all") & summary.group.eq("J") & summary.view.eq("P_applicable")
        & summary.measurement.eq("betlas_axis_best_angular_coverage")
    ].iloc[0]
    assert row.n_total == 3 and row.n_observed == 3 and row.H_observed == 2
    assert row.domain_median == 0
    assert row.domain_mean == pytest.approx(1 / 3)
    assert row.H_equal_mean == 0.5
    assert "2.70.10" not in set(summary.group)
    assert {"J", "2.60.40", "2.60.50"} == set(summary.group)
    confirmation = summary.loc[
        summary.scope.eq("role:confirmation") & summary.group.eq("J") & summary.view.eq("P_applicable")
        & summary.measurement.eq("betlas_axis_best_angular_coverage")
    ].iloc[0]
    assert confirmation.n_total == 1 and confirmation.H_equal_mean == 1


def test_partial_graphs_are_separate_and_observed_zero_is_not_missing(tmp_path: Path) -> None:
    tables = _run(tmp_path)
    summary = tables["H1-distributions.csv"]
    selected = summary.loc[summary.scope.eq("all") & summary.group.eq("J") & summary.measurement.eq("pairing_cycle_rank")].set_index("view")
    assert selected.loc["K_complete", "n_observed"] == 2
    assert selected.loc["K_complete", "n_missing"] == 1
    assert selected.loc["K_complete", "H_equal_mean"] == 0
    assert selected.loc["K_complete", "pairing_partial_n"] == 1
    assert selected.loc["K_retained_sensitivity", "n_observed"] == 3
    assert selected.loc["K_retained_sensitivity", "H_equal_mean"] == 3.5
    assert selected.loc["K_partial_retained_only", "H_equal_mean"] == 7
    other = summary.loc[summary.scope.eq("all") & summary.group.eq("2.60.40") & summary.view.eq("K_complete")].set_index("measurement")
    assert other.loc["pairing_cycle_rank", "n_observed"] == 1
    assert other.loc["pairing_cycle_rank", "domain_median"] == 0
    assert other.loc["pairing_strand_rank_gap_mean", "n_observed"] == 0
    assert np.isnan(other.loc["pairing_strand_rank_gap_mean", "domain_median"])


def test_two_sheet_measurements_require_two_multistrand_sheets(tmp_path: Path) -> None:
    domain = _run(tmp_path)["domain-phenotypes.csv"].set_index("record_id")
    assert domain.loc["j1", "two_sheet_applicable"]
    assert domain.loc["j1", "betlas_sheet_pair_centroid_distance_norm"] == 0.4
    assert not domain.loc["j2", "two_sheet_applicable"]
    assert np.isnan(domain.loc["j2", "betlas_sheet_pair_centroid_distance_norm"])
    assert not domain.loc["b1", "two_sheet_applicable"]
    assert np.isnan(domain.loc["b1", "betlas_sheet_pair_centroid_distance_norm"])
    assert np.isnan(domain.loc["f1", "betlas_axis_best_angular_coverage"])


def test_near_barrel_unknowns_and_fixed_bin_boundaries_are_preserved(tmp_path: Path) -> None:
    tables = _run(tmp_path)
    near = tables["H3-near-barrel.csv"]
    row = near.loc[near.scope.eq("all") & near.group.eq("J") & near.stratifier.eq("overall")].iloc[0]
    assert row.n_total == 3 and row.near_barrel_available_n == 2 and row.near_barrel_unavailable_n == 1
    assert row.near_barrel_n == 1 and row.near_barrel_fraction_available == 0.5
    assert row.H_equal_near_barrel_fraction_available == 0.5
    domain = tables["domain-phenotypes.csv"].set_index("record_id")
    assert domain.loc["j1", "length_bin"] == "le100"
    assert domain.loc["j2", "length_bin"] == "101_200"
    assert domain.loc["j3", "length_bin"] == "gt200"
    assert domain.loc["s1", "length_bin"] == "101_200"
    assert domain.loc["j2", "beta_fraction_bin"] == "le0.5"
    assert domain.loc["j1", "strand_count_bin"] == "le8"
    assert domain.loc["j2", "strand_count_bin"] == "gt8"
    assert domain.loc["f1", "length_bin"] == "unavailable"
    missing = near.loc[near.scope.eq("all") & near.group.eq("2.60.50") & near.stratifier.eq("overall")].iloc[0]
    assert missing.near_barrel_available_n == 0
    assert np.isnan(missing.near_barrel_fraction_available)


def test_inputs_are_not_mutated_and_duplicate_or_fractional_relations_fail(tmp_path: Path) -> None:
    cohort, connectivity, geometry = _inputs()
    originals = [frame.copy(deep=True) for frame in (cohort, connectivity, geometry)]
    run_phenotype_analysis(cohort, connectivity, geometry, tmp_path / "valid")
    for frame, original in zip((cohort, connectivity, geometry), originals, strict=True):
        pd.testing.assert_frame_equal(frame, original)
    duplicate = pd.concat([connectivity, connectivity.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate record_id"):
        run_phenotype_analysis(cohort, duplicate, geometry, tmp_path / "duplicate")
    geometry["near_barrel"] = [0.7] * len(geometry)
    with pytest.raises(ValueError, match="boolean or NA"):
        run_phenotype_analysis(cohort, connectivity, geometry, tmp_path / "fraction")


def test_confirmation_uses_only_its_phase_neighborhood(tmp_path: Path) -> None:
    cohort, connectivity, geometry = _inputs()
    geometry["near_barrel_confirmation"] = pd.array([pd.NA, pd.NA, True, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="boolean")
    geometry["near_barrel_discovery"] = pd.array([False, False, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="boolean")
    run_phenotype_analysis(cohort, connectivity, geometry, tmp_path)
    near = pd.read_csv(tmp_path / "H3-near-barrel.csv")
    overall = near.loc[near.stratifier.eq("overall")].set_index(["scope", "group"])
    assert overall.loc[("role:confirmation", "J"), "near_barrel_fraction_available"] == 1
    assert overall.loc[("role:discovery", "J"), "near_barrel_fraction_available"] == 0
    assert overall.loc[("all", "J"), "near_barrel_fraction_available"] == 0.5
    # The all-graph relation of s1 is measured false, but its phase relation is unknown.
    assert overall.loc[("role:confirmation", "2.60.40"), "near_barrel_available_n"] == 0
    assert pd.isna(overall.loc[("role:confirmation", "2.60.40"), "near_barrel_fraction_available"])
