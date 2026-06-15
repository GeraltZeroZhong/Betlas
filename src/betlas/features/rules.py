from __future__ import annotations

import math

from ..constants import FOLD_LABELS


def _f(features: dict[str, float | int | str], key: str, default: float = 0.0) -> float:
    try:
        value = features.get(key, default)
        if value in {"", None}:
            return default
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except (TypeError, ValueError):
        return default


def _bell(value: float, center: float, width: float) -> float:
    if width <= 0:
        return 0.0
    return math.exp(-((value - center) / width) ** 2)


def grammar_rule_scores(features: dict[str, float | int | str]) -> dict[str, float]:
    closure = _f(features, "cz_axis_best_slice_coverage_median")
    all_coverage = _f(features, "cz_axis_best_angular_coverage")
    gap = _f(features, "cz_axis_best_largest_gap_fraction")
    elongation = _f(features, "cz_pca_elongation")
    flatness = _f(features, "cz_pca_flatness")
    sheet_count = _f(features, "cz_sheet_count")
    face_count = _f(features, "cz_sheet_face_count")
    strand_count = _f(features, "cz_beta_strand_count")
    run_count = _f(features, "cz_beta_run_count", strand_count)
    helix_count = _f(features, "cz_helix_count")
    alpha_shell = _f(features, "cz_alpha_shell_radial_delta")
    alternation = _f(features, "cz_beta_alpha_alternation_fraction")
    prop_fft = _f(features, "cz_angular_fft_k3_8_max")
    best_k = _f(features, "cz_angular_fft_k3_8_best_k")
    solenoid = _f(features, "cz_axis_periodicity_score")
    sheet_span = _f(features, "cz_sheet_angular_span_max")
    barrel_wall = _f(features, "cz_barrel_wall_continuity_score")
    bilayer = _f(features, "cz_sheet_pair_bilayer_score")
    lobe_guard = _f(features, "cz_sandwich_lobe_guard_score")
    k2_dominance = _f(features, "cz_angular_fft_k2_dominance")
    top2_fraction = _f(features, "cz_sheet_pair_top2_fraction")
    top2_balance = _f(features, "cz_sheet_pair_size_balance")
    greek_proxy = _f(features, "cz_sheet_seq_greek_key_proxy")
    nonlocal_order = _f(features, "cz_jelly_roll_order_nonlocal_score")
    interleave = _f(features, "cz_sheet_seq_top2_interleave_score")
    order_displacement = _f(features, "cz_sheet_seq_top2_order_displacement")
    two_sheet = _bell(sheet_count, 2.0, 1.2)
    three_face = _bell(face_count, 3.0, 1.0)
    eight_strands = _bell(strand_count, 8.0, 2.2)
    eight_runs = _bell(run_count, 8.0, 2.4)

    scores = {
        "beta_barrel": 1.1 * closure
        + 1.0 * all_coverage
        + 0.7 * barrel_wall
        + 0.4 * (1.0 - gap)
        + 0.2 * sheet_span
        - 0.35 * lobe_guard
        - 0.20 * k2_dominance,
        "beta_prism": 1.2 * three_face + 0.7 * _bell(best_k, 3.0, 1.0) + 0.4 * flatness + 0.3 * all_coverage,
        "beta_propeller": 1.1 * prop_fft + 0.9 * _bell(best_k, 6.0, 2.5) + 0.5 * all_coverage + 0.3 * flatness,
        "jelly_roll": 0.8 * eight_strands
        + 0.9 * eight_runs
        + 0.6 * two_sheet
        + 0.5 * top2_fraction
        + 0.6 * greek_proxy
        + 0.7 * nonlocal_order
        + 0.5 * interleave
        + 0.4 * order_displacement
        + 0.3 * (1.0 - closure),
        "beta_solenoid": 1.3 * solenoid + 1.0 * min(1.5, elongation / 3.0) + 0.5 * _f(features, "cz_z_continuity_fraction"),
        "beta_sandwich": 0.8 * two_sheet
        + 1.0 * bilayer
        + 0.7 * lobe_guard
        + 0.4 * top2_balance
        + 0.5 * (1.0 - closure)
        + 0.4 * flatness
        + 0.2 * min(1.0, face_count / 3.0),
        "tim_like_beta_alpha_barrel": 1.0 * _bell(strand_count, 8.0, 2.5)
        + 0.9 * min(1.0, helix_count / 8.0)
        + 0.8 * alternation
        + 0.5 * max(0.0, alpha_shell / 6.0)
        + 0.4 * closure,
    }
    return {label: float(scores.get(label, 0.0)) for label in FOLD_LABELS}
