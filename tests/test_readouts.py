from __future__ import annotations

import math

import pandas as pd

from betlas.ml.benchmark import numeric_feature_columns
from betlas.readouts import get_readout, list_readouts
from betlas.readouts.beta_barrel_staves import StrandCountAnalyzer, build_config
from betlas.readouts.beta_barrel_staves.config import AnalyzerConfig
from betlas.readouts.topology_diagnostics import compute_topology_diagnostics


def _ring_layer(count: int, *, missing: set[int] | None = None) -> list[tuple[float, float, float, float]]:
    missing = missing or set()
    points = []
    for strand_id in range(count):
        if strand_id in missing:
            continue
        theta = (2.0 * math.pi * strand_id) / count
        points.append(
            (
                10.0 * math.cos(theta),
                10.0 * math.sin(theta),
                float(strand_id),
                float(strand_id),
            )
        )
    return points


def _ring_layer_with_terminal_outliers(count: int) -> list[tuple[float, float, float, float]]:
    return [
        (10.5, 0.0, -10.0, -1.0),
        *_ring_layer(count),
        (-10.5, 0.0, float(count + 10), float(count)),
    ]


def _ring_layer_with_inner_plug(count: int) -> list[tuple[float, float, float, float]]:
    points = _ring_layer(count)
    for offset in range(5):
        theta = (2.0 * math.pi * offset) / 5.0
        points.append(
            (
                2.0 * math.cos(theta),
                2.0 * math.sin(theta),
                float(count + offset),
                float(count + offset),
            )
        )
    return points


def test_readout_registry_exposes_beta_barrel_staves():
    names = [spec.name for spec in list_readouts()]

    assert names == [
        "beta-barrel-detection",
        "beta-barrel-staves",
        "bfvd-viral-beta-fold-scan",
        "fold-continuous-scores",
        "mixed-topology",
        "topology-ambiguity",
        "topology-diagnostics",
    ]
    assert get_readout("beta-barrel-detection").summary.startswith("Betlas native")
    assert get_readout("beta-barrel-staves").summary.startswith("Secondary readout")
    assert get_readout("topology-ambiguity").summary.startswith("Boundary-region ambiguity")


def test_beta_barrel_staves_config_defaults_are_betlas_owned():
    cfg = build_config()

    assert cfg.output.csv_path == "beta_barrel_staves_results.csv"
    assert cfg.barrel_gate.enabled is False
    assert cfg.analyzer.rules.barrel_wall_graph.enabled is True


def test_beta_barrel_staves_readout_counts_persistent_ring():
    slices = {float(index): _ring_layer(8) for index in range(6)}

    report = StrandCountAnalyzer().analyze(slices)

    assert report["strand_count"] == 8
    assert report["persistent_strands"] == 8
    assert report["candidate_strands"] == 8
    assert report["support_consensus_strands"] == 8
    assert report["run_window_core_applied"] == 0


def test_beta_barrel_staves_readout_trims_terminal_outliers():
    slices = {float(index): _ring_layer_with_terminal_outliers(8) for index in range(6)}

    report = StrandCountAnalyzer().analyze(slices)

    assert report["strand_count"] == 8
    assert report["trimmed_layers"] == 6


def test_beta_barrel_staves_readout_filters_inner_plug_points():
    slices = {float(index): _ring_layer_with_inner_plug(16) for index in range(6)}
    cfg = AnalyzerConfig()
    cfg.rules.radial_outlier.min_radius_ratio = 0.55

    report = StrandCountAnalyzer(cfg).analyze(slices)

    assert report["strand_count"] == 16
    assert report["radial_outlier_points"] == 30


def _topology_row(record_id: str, label: str, **updates: float | str) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "record_id": record_id,
        "pdb_id": record_id[:4],
        "domain_id": record_id,
        "fold_label_final": label,
        "cz_parse_ok": 1,
        "cz_top_fold": label,
        "cz_sheet_count": 2.0,
        "cz_sheet_face_count": 2.0,
        "cz_sheet_pair_top2_fraction": 0.9,
        "cz_sheet_pair_size_balance": 0.9,
        "cz_sheet_pair_bilayer_score": 0.15,
        "cz_sandwich_lobe_guard_score": 0.16,
        "cz_sheet_pair_face_alignment": 0.8,
        "cz_sheet_pair_normal_abs_dot": 0.8,
        "cz_sheet_pair_cross_contact_density8": 0.25,
        "cz_axis_best_slice_coverage_median": 0.72,
        "cz_axis_best_angular_coverage": 0.76,
        "cz_axis_best_largest_gap_fraction": 0.24,
        "cz_axis_best_slice_largest_gap_fraction_mean": 0.32,
        "cz_barrel_wall_continuity_score": 0.12,
        "cz_angular_sector_occupancy12": 0.65,
        "cz_axis_best_slice_high_coverage_fraction": 0.4,
        "cz_sheet_seq_top2_interleave_score": 0.1,
        "cz_sheet_seq_top2_order_displacement": 0.2,
        "cz_top2_sheet_order_nonlocal_fraction": 0.2,
        "cz_top2_sheet_order_inversion_fraction_mean": 0.1,
        "cz_sheet_seq_greek_key_proxy": 0.05,
        "cz_jelly_roll_order_nonlocal_score": 0.03,
        "cz_beta_run_eight_score": 0.4,
        "cz_angular_fft_k3_8_max": 0.1,
        "cz_angular_fft_k3_8_best_k": 6.0,
        "cz_axis_periodicity_score": 0.0,
        "cz_pca_elongation": 1.2,
        "cz_beta_alpha_alternation_fraction": 0.1,
        "cz_alpha_shell_radial_delta": 0.0,
    }
    for fold_label in (
        "beta_barrel",
        "beta_prism",
        "beta_propeller",
        "jelly_roll",
        "beta_solenoid",
        "beta_sandwich",
        "tim_like_beta_alpha_barrel",
    ):
        row[f"cz_rule_score_{fold_label}"] = 0.0
    row[f"cz_rule_score_{label}"] = 2.0
    row.update(updates)
    return row


def test_topology_diagnostics_scores_continuous_and_mixed_signals():
    rows = [
        _topology_row(
            "ambiguousA",
            "beta_sandwich",
            cz_top_fold="jelly_roll",
            cz_rule_score_jelly_roll=3.4,
            cz_rule_score_beta_sandwich=3.2,
            cz_sheet_seq_top2_interleave_score=0.8,
            cz_sheet_seq_top2_order_displacement=0.7,
            cz_top2_sheet_order_nonlocal_fraction=0.9,
            cz_top2_sheet_order_inversion_fraction_mean=0.6,
            cz_sheet_seq_greek_key_proxy=0.6,
            cz_jelly_roll_order_nonlocal_score=0.5,
            cz_beta_run_eight_score=1.0,
        ),
        _topology_row(
            "barrelB",
            "beta_barrel",
            cz_rule_score_beta_barrel=3.0,
            cz_axis_best_slice_coverage_median=0.9,
            cz_axis_best_angular_coverage=0.9,
            cz_axis_best_largest_gap_fraction=0.1,
            cz_axis_best_slice_largest_gap_fraction_mean=0.18,
            cz_barrel_wall_continuity_score=0.55,
            cz_angular_sector_occupancy12=1.0,
            cz_axis_best_slice_high_coverage_fraction=0.9,
            cz_sandwich_lobe_guard_score=0.04,
        ),
        _topology_row("sandC", "beta_sandwich", cz_rule_score_beta_sandwich=3.0),
    ]

    diagnostics = compute_topology_diagnostics(pd.DataFrame(rows), k_neighbors=2)
    ambiguous = diagnostics[diagnostics["record_id"] == "ambiguousA"].iloc[0]
    barrel = diagnostics[diagnostics["record_id"] == "barrelB"].iloc[0]

    assert ambiguous["cz_jelly_rollness"] > 0.55
    assert ambiguous["cz_sandwichness"] > 0.45
    assert ambiguous["cz_topology_ambiguity_score"] > 0.4
    assert ambiguous["cz_mixed_topology_flag"] == 1
    assert "jelly_roll_like_sandwich_subtopology" in ambiguous["cz_mixed_topology_types"]
    assert barrel["cz_barrel_likeness"] > 0.7


def test_topology_diagnostics_columns_are_excluded_from_ml_features():
    df = pd.DataFrame(
        [
            {
                "record_id": "x",
                "fold_label_final": "beta_barrel",
                "cz_parse_ok": 1,
                "cz_axis_best_angular_coverage": 0.8,
                "cz_topology_ambiguity_score": 0.9,
                "cz_boundary_region_flag": 1,
                "cz_jelly_rollness": 0.7,
                "cz_mixed_topology_score": 0.6,
                "cz_neighbor_label_entropy": 0.5,
            }
        ]
    )

    columns = numeric_feature_columns(df)

    assert "cz_axis_best_angular_coverage" in columns
    assert "cz_topology_ambiguity_score" not in columns
    assert "cz_boundary_region_flag" not in columns
    assert "cz_jelly_rollness" not in columns
    assert "cz_mixed_topology_score" not in columns
    assert "cz_neighbor_label_entropy" not in columns
