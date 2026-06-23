from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .constants import (
    DEFAULT_ALLOWED_SUFFIXES,
    DEFAULT_FILL_SHEET_HOLE_LENGTH,
    DEFAULT_INPUT_PATH,
    DEFAULT_MIN_CHAIN_RESIDUES,
    DEFAULT_MIN_INFORMATIVE_SLICES,
    DEFAULT_MIN_SHEET_RESIDUES,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SLICE_STEP_SIZE,
)
from .exceptions import ConfigValidationError


@dataclass
class RuntimeConfig:
    workers: int | None = None
    prepare_workers: int | None = None
    prepare_batch_size: int = 16
    analysis_batch_size: int = 64
    cpu_reserve: int = 1
    dssp_bin_path: str | None = None
    fail_on_dssp_error: bool = True
    prepare_cache_enabled: bool = True
    prepare_cache_dir: str | None = None
    check_env: bool = False


@dataclass
class InputConfig:
    path: str = DEFAULT_INPUT_PATH
    allowed_suffixes: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_SUFFIXES))
    chain_id: str = ""
    min_chain_residues: int = DEFAULT_MIN_CHAIN_RESIDUES
    min_sheet_residues: int = DEFAULT_MIN_SHEET_RESIDUES
    min_informative_slices: int = DEFAULT_MIN_INFORMATIVE_SLICES


@dataclass
class OutputConfig:
    csv_path: str = DEFAULT_OUTPUT_CSV
    metadata_path: str | None = None
    summary_limit: int = 50


@dataclass
class BarrelGateConfig:
    enabled: bool = True
    pass_results: list[str] = field(default_factory=lambda: ["BARREL"])
    fail_on_error: bool = True
    overrides: list[str] = field(default_factory=list)


@dataclass
class SlicerConfig:
    step_size: float = DEFAULT_SLICE_STEP_SIZE
    fill_sheet_hole_length: int = DEFAULT_FILL_SHEET_HOLE_LENGTH


@dataclass
class LeastSquaresConfig:
    method: str = "trf"
    loss: str = "soft_l1"
    f_scale: float = 1.0


@dataclass
class StrandCountConfig:
    min_points_per_layer: int = 4
    min_usable_layers: int = 3
    min_strand_support_layers: int = 2
    min_strand_support_fraction: float = 0.20
    consensus_min_strand_support_layers: int = 2
    consensus_min_strand_support_fraction: float = 0.03
    trajectory_merge_enabled: bool = True
    trajectory_merge_max_run_id_gap: int = 2
    trajectory_merge_max_layer_gap: int = 12
    trajectory_merge_max_overlap_fraction: float = 0.15
    trajectory_merge_angle_tolerance_deg: float = 24.0
    trajectory_merge_slope_tolerance_deg_per_layer: float = 8.0
    trajectory_merge_extended_enabled: bool = True
    trajectory_merge_extended_max_run_id_gap: int = 16
    trajectory_merge_extended_angle_tolerance_deg: float = 12.0
    trajectory_merge_extended_min_candidate_strands: int = 25
    trajectory_merge_extended_min_layer_q90: int = 8
    trajectory_merge_extended_min_support_gap: int = 16
    min_layer_coverage_fraction: float = 0.50
    min_confidence: float = 0.45
    min_count: int = 1
    max_count: int = 64
    require_geometric_consistency: bool = False
    score_support_consensus_weight: float = 0.34
    score_layer_q90_weight: float = 0.34
    score_layer_q75_weight: float = 0.12
    score_layer_max_weight: float = 0.08
    score_candidate_weight: float = 0.07
    score_persistent_weight: float = 0.05
    score_joint_support_candidate_weight: float = 0.22
    score_exact_support_candidate_bonus: float = 0.18
    score_layer_agreement_weight: float = 0.18
    score_selected_mean_support_weight: float = 0.12
    score_selected_min_support_weight: float = 0.06
    score_next_consensus_penalty: float = 0.12
    score_next_support_penalty: float = 0.05
    score_over_candidate_base_penalty: float = 0.50
    score_over_candidate_per_count_penalty: float = 0.08
    score_under_persistent_per_count_penalty: float = 0.10
    score_under_support_gap_per_count_penalty: float = 0.09
    score_under_support_gap_max_penalty: float = 1.0
    score_over_support_gap_min_delta: int = 4
    score_over_support_gap_per_count_penalty: float = 0.08
    score_layer_agreement_min_tolerance: float = 2.0
    score_layer_agreement_count_fraction: float = 0.15
    score_support_consensus_scale: float = 0.24
    score_layer_q90_scale: float = 0.18
    score_layer_q75_scale: float = 0.20
    score_layer_max_scale: float = 0.16
    score_candidate_scale: float = 0.22
    score_persistent_scale: float = 0.30
    score_joint_support_scale: float = 0.20
    score_joint_candidate_scale: float = 0.20


@dataclass
class AxisSearchRefineConfig:
    enabled: bool = True
    angle_deg: float = 5.0


@dataclass
class AxisSearchConfig:
    enabled: bool = True
    min_points_per_scored_layer: int = 0
    score_geometry_first: bool = False
    report_scoring_enabled: bool = True
    report_candidate_limit: int = 1
    report_confidence_weight: float = 1.00
    report_decision_weight: float = 0.20
    refine: AxisSearchRefineConfig = field(default_factory=AxisSearchRefineConfig)


@dataclass
class NearestNeighborRuleConfig:
    enabled: bool = True
    max_robust_cv: float = 0.55
    min_inlier_frac: float = 0.60


@dataclass
class AngleOrderRuleConfig:
    enabled: bool = True
    local_step_max: int = 2
    min_local_frac: float = 0.70
    max_mean_circ_dist_norm: float = 0.35


@dataclass
class AngleRuleConfig:
    enabled: bool = True
    max_gap_deg: float = 200.0
    order: AngleOrderRuleConfig = field(default_factory=AngleOrderRuleConfig)


@dataclass
class SequenceCoreRuleConfig:
    enabled: bool = True
    exhaustive_search: bool = True
    max_exhaustive_points: int = 32
    max_terminal_trim_fraction: float = 0.35
    min_core_fraction: float = 0.60
    min_quality_gain: float = 0.25
    global_enabled: bool = True
    global_min_layer_count_max: int = 18
    global_min_trajectory_strands: int = 22
    global_min_support_gap: int = 4
    global_min_prediction_gap: int = 3
    global_layer_count_allowance: int = 2
    global_min_trimmed_trajectories: int = 3
    global_min_centrality: float = 0.40
    global_min_window_support_fraction: float = 0.15
    global_min_window_layer_max_floor: int = 8
    global_min_window_q90_floor: int = 6
    global_min_window_layer_max_fraction: float = 0.70
    global_min_window_q90_fraction: float = 0.65
    global_score_support_weight: float = 1.20
    global_score_outside_support_weight: float = 0.45
    global_score_centrality_weight: float = 0.35
    global_score_layer_preservation_weight: float = 0.25
    global_score_trimmed_fraction_weight: float = 0.08
    global_score_length_penalty_weight: float = 0.08
    run_window_enabled: bool = True
    run_window_min_candidate_strands: int = 12
    run_window_min_base_layer_max: int = 9
    run_window_min_support_gap: int = 4
    run_window_min_prediction_gap: int = 2
    run_window_min_count_over_base_layer_max: int = 2
    run_window_low_layer_max_threshold: int = 12
    run_window_low_layer_max_min_count_over_base_layer_max: int = 3
    run_window_max_count_over_base_layer_max: int = 8
    run_window_min_count_over_layer_max: int = 2
    run_window_min_count_layer_improvement: int = 5
    run_window_min_absent_from_max: int = 10
    run_window_min_trimmed_runs: int = 2
    run_window_terminal_band_size: int = 4
    run_window_boundary_candidates_enabled: bool = True
    run_window_boundary_padding_step: int = 2
    run_window_boundary_max_padding: int = 8
    run_window_min_raw_span: int = 16
    run_window_max_raw_span: int = 0
    run_window_max_scored_windows: int = 40
    run_window_min_layer_max_floor: int = 6
    run_window_min_layer_q90_floor: int = 4
    run_window_min_layer_max_fraction: float = 1.00
    run_window_min_layer_q90_fraction: float = 0.70
    run_window_min_z_coverage_fraction: float = 0.60
    run_window_min_score: float = 1.15
    run_window_score_count_improvement_weight: float = 0.75
    run_window_score_layer_preservation_weight: float = 0.35
    run_window_score_outside_support_weight: float = 0.10
    run_window_score_z_coverage_weight: float = 0.25
    run_window_score_centrality_weight: float = 0.0
    run_window_score_disappearance_weight: float = 1.00
    run_window_score_absent_improvement_weight: float = 0.50
    run_window_score_drop_improvement_weight: float = 0.50
    run_window_score_count_margin_weight: float = 0.25
    run_window_score_length_penalty_weight: float = 0.45


@dataclass
class BarrelWallGraphRuleConfig:
    enabled: bool = True
    min_layer_points: int = 4
    min_node_support_layers: int = 2
    min_selected_count: int = 4
    max_selected_count: int = 64
    raw_variant_enabled: bool = True
    outer_variant_enabled: bool = True
    outer_gap_variant_enabled: bool = True
    near_cycle_variant_enabled: bool = True
    edge_min_counts: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10])
    outer_edge_min_counts: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    outer_filter_min_points: int = 8
    outer_filter_quantile: float = 0.25
    outer_gap_min_fraction: float = 0.30
    outer_gap_min_removed_points: int = 2
    complete_min_layer_q90: int = 8
    complete_max_support_ratio: float = 1.75
    complete_max_q90_spread: int = 1
    complete_layer_max_candidate_enabled: bool = True
    complete_layer_max_candidate_max_q90_delta: int = 1
    complete_outer_gap_reduction_enabled: bool = True
    complete_outer_gap_min_layer_q90: int = 12
    complete_outer_gap_max_support_ratio: float = 1.25
    complete_outer_gap_max_q90_spread: int = 0
    complete_outer_gap_min_reduction: int = 2
    complete_outer_gap_min_count_fraction: float = 0.60
    complete_layer_cohort_enabled: bool = True
    complete_layer_cohort_min_layers: int = 3
    complete_layer_cohort_min_fraction: float = 0.80
    sparse_min_support_ratio: float = 1.60
    sparse_membership_enabled: bool = True
    sparse_membership_min_selected_count: int = 16
    sparse_membership_max_selected_count: int = 64
    sparse_membership_min_edge_min_count: int = 2
    sparse_membership_max_component_avg_degree: float = 4.0
    sparse_membership_same_edge_max_component_avg_degree: float = 4.0
    sparse_membership_same_edge_min_variants_for_degree_relaxation: int = 2
    sparse_membership_max_selected_over_support: int = 2
    sparse_membership_max_support_reduction_without_plateau: int = 10
    sparse_membership_min_plateau_for_large_reduction: int = 2
    sparse_membership_max_layer_q90: int = 0
    sparse_membership_min_score: float = 1.50
    sparse_membership_variant_agreement_tolerance: int = 2
    sparse_membership_same_edge_tolerance: int = 0
    sparse_membership_plateau_tolerance: int = 0
    sparse_membership_min_same_edge_variants_for_selection: int = 2
    sparse_membership_min_plateau_for_selection: int = 2
    sparse_membership_plateau_scale: float = 2.0
    sparse_membership_drop_scale: float = 0.30
    sparse_membership_degree_weight: float = 1.60
    sparse_membership_degree_scale: float = 0.90
    sparse_membership_cycle_rank_weight: float = 0.35
    sparse_membership_cycle_rank_scale: float = 2.0
    sparse_membership_degree2_fraction_weight: float = 0.65
    sparse_membership_drop_weight: float = 0.75
    sparse_membership_plateau_weight: float = 1.00
    sparse_membership_variant_agreement_weight: float = 0.45
    sparse_membership_same_edge_agreement_weight: float = 1.35
    sparse_membership_support_fit_weight: float = 0.05
    sparse_membership_layer_lift_weight: float = 0.20
    sparse_membership_edge_weight: float = 0.10
    sparse_membership_radial_rank_weight: float = 0.50
    sparse_membership_branch_penalty_weight: float = 0.95
    sparse_membership_block_penalty_weight: float = 0.55
    sparse_membership_outer_bonus: float = 0.30
    sparse_membership_outer_gap_bonus: float = 0.05
    partial_layer_close_max_gap_ratio: float = 3.0
    partial_layer_min_closed_coverage_fraction: float = 0.75


@dataclass
class RadialOutlierRuleConfig:
    enabled: bool = True
    min_points: int = 5
    min_radius_ratio: float = 0.0
    max_radius_ratio: float = 1.8


@dataclass
class ConfidenceConfig:
    layer_count_support_tolerance: int = 2
    layer_agreement_min_tolerance: float = 2.0
    layer_agreement_count_fraction: float = 0.15
    decision_score_normalizer: float = 1.20
    feature_support_consensus_weight: float = 0.55
    feature_trajectory_candidate_weight: float = 0.30
    feature_layer_q90_weight: float = 0.15
    feature_support_consensus_scale: float = 0.20
    feature_trajectory_candidate_scale: float = 0.25
    feature_layer_q90_scale: float = 0.18
    decision_agreement_weight: float = 0.42
    feature_agreement_weight: float = 0.20
    selected_support_weight: float = 0.12
    layer_support_fraction_weight: float = 0.08
    layer_agreement_weight: float = 0.06
    usable_layer_fraction_weight: float = 0.07
    geometric_pass_fraction_weight: float = 0.03
    max_outlier_penalty: float = 0.05
    radial_outlier_layer_penalty: float = 0.002
    trimmed_layer_penalty: float = 0.001


@dataclass
class LayerQualityConfig:
    min_geometry_points: int = 3
    usable_bonus: float = 4.0
    geometric_pass_bonus: float = 2.0
    angle_margin_scale_deg: float = 180.0
    angle_margin_cap: float = 1.0
    terminal_order_local_gain: float = 0.08
    terminal_order_global_gain: float = 0.05
    terminal_nn_inlier_gain: float = 0.05


@dataclass
class AnalyzerRulesConfig:
    nearest_neighbor: NearestNeighborRuleConfig = field(default_factory=NearestNeighborRuleConfig)
    angle: AngleRuleConfig = field(default_factory=AngleRuleConfig)
    sequence_core: SequenceCoreRuleConfig = field(default_factory=SequenceCoreRuleConfig)
    barrel_wall_graph: BarrelWallGraphRuleConfig = field(default_factory=BarrelWallGraphRuleConfig)
    radial_outlier: RadialOutlierRuleConfig = field(default_factory=RadialOutlierRuleConfig)


@dataclass
class AnalyzerConfig:
    count: StrandCountConfig = field(default_factory=StrandCountConfig)
    axis_search: AxisSearchConfig = field(default_factory=AxisSearchConfig)
    rules: AnalyzerRulesConfig = field(default_factory=AnalyzerRulesConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    layer_quality: LayerQualityConfig = field(default_factory=LayerQualityConfig)


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    barrel_gate: BarrelGateConfig = field(default_factory=BarrelGateConfig)
    slicer: SlicerConfig = field(default_factory=SlicerConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)


COMPAT_OVERRIDE_PATHS = {
    "DSSP_BIN_PATH": "runtime.dssp_bin_path",
    "PREPARE_BATCH_SIZE": "runtime.prepare_batch_size",
    "ANALYSIS_BATCH_SIZE": "runtime.analysis_batch_size",
    "PREPARE_CACHE_ENABLED": "runtime.prepare_cache_enabled",
    "PREPARE_CACHE_DIR": "runtime.prepare_cache_dir",
    "BARREL_GATE_ENABLED": "barrel_gate.enabled",
    "BARREL_GATE_FAIL_ON_ERROR": "barrel_gate.fail_on_error",
    "SLICE_STEP_SIZE": "slicer.step_size",
    "MIN_POINTS_PER_SLICE": "analyzer.count.min_points_per_layer",
    "MIN_POINTS_PER_LAYER": "analyzer.count.min_points_per_layer",
    "MIN_USABLE_LAYERS": "analyzer.count.min_usable_layers",
    "MIN_STRAND_SUPPORT_LAYERS": "analyzer.count.min_strand_support_layers",
    "MIN_STRAND_SUPPORT_FRACTION": "analyzer.count.min_strand_support_fraction",
    "TRAJECTORY_MERGE_ENABLED": "analyzer.count.trajectory_merge_enabled",
    "TRAJECTORY_MERGE_MAX_RUN_ID_GAP": "analyzer.count.trajectory_merge_max_run_id_gap",
    "TRAJECTORY_MERGE_MAX_LAYER_GAP": "analyzer.count.trajectory_merge_max_layer_gap",
    "TRAJECTORY_MERGE_MAX_OVERLAP_FRACTION": (
        "analyzer.count.trajectory_merge_max_overlap_fraction"
    ),
    "TRAJECTORY_MERGE_ANGLE_TOLERANCE_DEG": (
        "analyzer.count.trajectory_merge_angle_tolerance_deg"
    ),
    "TRAJECTORY_MERGE_SLOPE_TOLERANCE_DEG_PER_LAYER": (
        "analyzer.count.trajectory_merge_slope_tolerance_deg_per_layer"
    ),
    "TRAJECTORY_MERGE_EXTENDED_ENABLED": "analyzer.count.trajectory_merge_extended_enabled",
    "TRAJECTORY_MERGE_EXTENDED_MAX_RUN_ID_GAP": (
        "analyzer.count.trajectory_merge_extended_max_run_id_gap"
    ),
    "TRAJECTORY_MERGE_EXTENDED_ANGLE_TOLERANCE_DEG": (
        "analyzer.count.trajectory_merge_extended_angle_tolerance_deg"
    ),
    "TRAJECTORY_MERGE_EXTENDED_MIN_CANDIDATE_STRANDS": (
        "analyzer.count.trajectory_merge_extended_min_candidate_strands"
    ),
    "TRAJECTORY_MERGE_EXTENDED_MIN_LAYER_Q90": (
        "analyzer.count.trajectory_merge_extended_min_layer_q90"
    ),
    "TRAJECTORY_MERGE_EXTENDED_MIN_SUPPORT_GAP": (
        "analyzer.count.trajectory_merge_extended_min_support_gap"
    ),
    "MIN_LAYER_COVERAGE_FRACTION": "analyzer.count.min_layer_coverage_fraction",
    "MIN_CONFIDENCE": "analyzer.count.min_confidence",
    "MIN_COUNT": "analyzer.count.min_count",
    "MAX_COUNT": "analyzer.count.max_count",
    "REQUIRE_GEOMETRIC_CONSISTENCY": "analyzer.count.require_geometric_consistency",
    "AXIS_SEARCH_ENABLED": "analyzer.axis_search.enabled",
    "AXIS_SEARCH_MIN_POINTS_PER_SCORED_LAYER": (
        "analyzer.axis_search.min_points_per_scored_layer"
    ),
    "AXIS_SEARCH_SCORE_GEOMETRY_FIRST": "analyzer.axis_search.score_geometry_first",
    "AXIS_SEARCH_REPORT_SCORING_ENABLED": (
        "analyzer.axis_search.report_scoring_enabled"
    ),
    "AXIS_SEARCH_REPORT_CANDIDATE_LIMIT": (
        "analyzer.axis_search.report_candidate_limit"
    ),
    "AXIS_SEARCH_REFINE_ENABLED": "analyzer.axis_search.refine.enabled",
    "AXIS_SEARCH_REFINE_ANGLE_DEG": "analyzer.axis_search.refine.angle_deg",
    "NN_RULE_ENABLED": "analyzer.rules.nearest_neighbor.enabled",
    "NN_MAX_ROBUST_CV": "analyzer.rules.nearest_neighbor.max_robust_cv",
    "NN_MIN_INLIER_FRAC": "analyzer.rules.nearest_neighbor.min_inlier_frac",
    "ANGLE_RULE_ENABLED": "analyzer.rules.angle.enabled",
    "ANGLE_MAX_GAP_DEG": "analyzer.rules.angle.max_gap_deg",
    "ANGLE_ORDER_RULE_ENABLED": "analyzer.rules.angle.order.enabled",
    "ANGLE_ORDER_LOCAL_STEP_MAX": "analyzer.rules.angle.order.local_step_max",
    "ANGLE_ORDER_MIN_LOCAL_FRAC": "analyzer.rules.angle.order.min_local_frac",
    "ANGLE_ORDER_MAX_MEAN_CIRC_DIST_NORM": "analyzer.rules.angle.order.max_mean_circ_dist_norm",
    "SEQUENCE_CORE_RULE_ENABLED": "analyzer.rules.sequence_core.enabled",
    "SEQUENCE_CORE_EXHAUSTIVE_SEARCH": "analyzer.rules.sequence_core.exhaustive_search",
    "SEQUENCE_CORE_MAX_EXHAUSTIVE_POINTS": "analyzer.rules.sequence_core.max_exhaustive_points",
    "SEQUENCE_CORE_MAX_TERMINAL_TRIM_FRACTION": "analyzer.rules.sequence_core.max_terminal_trim_fraction",
    "SEQUENCE_CORE_MIN_CORE_FRACTION": "analyzer.rules.sequence_core.min_core_fraction",
    "SEQUENCE_CORE_MIN_QUALITY_GAIN": "analyzer.rules.sequence_core.min_quality_gain",
    "SEQUENCE_CORE_GLOBAL_ENABLED": "analyzer.rules.sequence_core.global_enabled",
    "SEQUENCE_CORE_GLOBAL_MIN_LAYER_COUNT_MAX": (
        "analyzer.rules.sequence_core.global_min_layer_count_max"
    ),
    "SEQUENCE_CORE_GLOBAL_MIN_TRAJECTORY_STRANDS": (
        "analyzer.rules.sequence_core.global_min_trajectory_strands"
    ),
    "SEQUENCE_CORE_GLOBAL_MIN_SUPPORT_GAP": "analyzer.rules.sequence_core.global_min_support_gap",
    "SEQUENCE_CORE_GLOBAL_MIN_PREDICTION_GAP": (
        "analyzer.rules.sequence_core.global_min_prediction_gap"
    ),
    "SEQUENCE_CORE_GLOBAL_LAYER_COUNT_ALLOWANCE": (
        "analyzer.rules.sequence_core.global_layer_count_allowance"
    ),
    "SEQUENCE_CORE_GLOBAL_MIN_TRIMMED_TRAJECTORIES": (
        "analyzer.rules.sequence_core.global_min_trimmed_trajectories"
    ),
    "SEQUENCE_CORE_GLOBAL_MIN_CENTRALITY": "analyzer.rules.sequence_core.global_min_centrality",
    "SEQUENCE_CORE_GLOBAL_MIN_WINDOW_SUPPORT_FRACTION": (
        "analyzer.rules.sequence_core.global_min_window_support_fraction"
    ),
    "SEQUENCE_CORE_GLOBAL_MIN_WINDOW_LAYER_MAX_FLOOR": (
        "analyzer.rules.sequence_core.global_min_window_layer_max_floor"
    ),
    "SEQUENCE_CORE_GLOBAL_MIN_WINDOW_Q90_FLOOR": (
        "analyzer.rules.sequence_core.global_min_window_q90_floor"
    ),
    "RADIAL_OUTLIER_RULE_ENABLED": "analyzer.rules.radial_outlier.enabled",
    "RADIAL_OUTLIER_MIN_POINTS": "analyzer.rules.radial_outlier.min_points",
    "RADIAL_OUTLIER_MIN_RADIUS_RATIO": "analyzer.rules.radial_outlier.min_radius_ratio",
    "RADIAL_OUTLIER_MAX_RADIUS_RATIO": "analyzer.rules.radial_outlier.max_radius_ratio",
    "CONFIDENCE_LAYER_COUNT_SUPPORT_TOLERANCE": (
        "analyzer.confidence.layer_count_support_tolerance"
    ),
    "CONFIDENCE_DECISION_SCORE_NORMALIZER": "analyzer.confidence.decision_score_normalizer",
    "LAYER_QUALITY_MIN_GEOMETRY_POINTS": "analyzer.layer_quality.min_geometry_points",
    "LAYER_QUALITY_USABLE_BONUS": "analyzer.layer_quality.usable_bonus",
    "LAYER_QUALITY_GEOMETRIC_PASS_BONUS": "analyzer.layer_quality.geometric_pass_bonus",
    "MIN_CHAIN_RESIDUES": "input.min_chain_residues",
    "MIN_SHEET_RESIDUES": "input.min_sheet_residues",
    "MIN_INFORMATIVE_SLICES": "input.min_informative_slices",
    "SUMMARY_LIMIT": "output.summary_limit",
}


class Config:
    """Flat configuration shim kept for compatibility."""


def _register_schema() -> None:
    cs = ConfigStore.instance()
    if getattr(_register_schema, "_done", False):
        return
    cs.store(name="beta_barrel_staves_schema", node=AppConfig)
    _register_schema._done = True


def _to_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_to_override_value(v) for v in value)
        return f"[{inner}]"
    return json.dumps(str(value))


def normalize_overrides(overrides: Mapping[str, Any] | list[str] | None = None) -> list[str]:
    if overrides is None:
        return []
    if isinstance(overrides, list):
        return list(overrides)

    normalized: list[str] = []
    for key, value in overrides.items():
        target_key = COMPAT_OVERRIDE_PATHS.get(key, key)
        normalized.append(f"{target_key}={_to_override_value(value)}")
    return normalized


def compose_config(
    overrides: Mapping[str, Any] | list[str] | None = None,
    *,
    config_name: str = "config",
) -> DictConfig:
    _register_schema()
    with initialize_config_module(config_module="betlas.readouts.beta_barrel_staves.conf", version_base=None):
        file_cfg = compose(config_name=config_name, overrides=normalize_overrides(overrides))
    structured_cfg = OmegaConf.structured(AppConfig)
    return OmegaConf.merge(structured_cfg, file_cfg)


def build_config(
    overrides: Mapping[str, Any] | list[str] | None = None,
    *,
    config_name: str = "config",
) -> AppConfig:
    cfg = compose_config(overrides, config_name=config_name)
    app_cfg = OmegaConf.to_object(cfg)
    if not isinstance(app_cfg, AppConfig):
        raise TypeError("Hydra returned an unexpected configuration object.")
    validate_config(app_cfg)
    sync_compat_config(app_cfg)
    return app_cfg


def _require_positive(name: str, value: int | float) -> None:
    if not isfinite(float(value)):
        raise ConfigValidationError(f"`{name}` must be finite.")
    if value <= 0:
        raise ConfigValidationError(f"`{name}` must be greater than 0.")


def _require_non_negative(name: str, value: int | float) -> None:
    if not isfinite(float(value)):
        raise ConfigValidationError(f"`{name}` must be finite.")
    if value < 0:
        raise ConfigValidationError(f"`{name}` must be greater than or equal to 0.")


def _require_ratio(name: str, value: float) -> None:
    if not isfinite(float(value)):
        raise ConfigValidationError(f"`{name}` must be finite.")
    if not 0.0 <= value <= 1.0:
        raise ConfigValidationError(f"`{name}` must be between 0 and 1.")


def validate_config(cfg: AppConfig) -> None:
    """Validate user-editable configuration values before running analysis."""
    if cfg.runtime.workers is not None:
        _require_positive("runtime.workers", int(cfg.runtime.workers))
    if cfg.runtime.prepare_workers is not None:
        _require_positive("runtime.prepare_workers", int(cfg.runtime.prepare_workers))
    _require_positive("runtime.prepare_batch_size", int(cfg.runtime.prepare_batch_size))
    _require_positive("runtime.analysis_batch_size", int(cfg.runtime.analysis_batch_size))
    _require_non_negative("runtime.cpu_reserve", int(cfg.runtime.cpu_reserve))

    if not cfg.input.allowed_suffixes:
        raise ConfigValidationError("`input.allowed_suffixes` must contain at least one suffix.")
    for suffix in cfg.input.allowed_suffixes:
        if not str(suffix).startswith("."):
            raise ConfigValidationError("Each `input.allowed_suffixes` value must start with '.'.")
    _require_non_negative("input.min_chain_residues", int(cfg.input.min_chain_residues))
    _require_non_negative("input.min_sheet_residues", int(cfg.input.min_sheet_residues))
    _require_non_negative("input.min_informative_slices", int(cfg.input.min_informative_slices))

    if not cfg.barrel_gate.pass_results:
        raise ConfigValidationError("`barrel_gate.pass_results` must contain at least one result label.")
    for result_label in cfg.barrel_gate.pass_results:
        if not str(result_label).strip():
            raise ConfigValidationError("`barrel_gate.pass_results` values must be non-empty strings.")
    for override in cfg.barrel_gate.overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ConfigValidationError("Each `barrel_gate.overrides` value must be a Hydra KEY=VALUE string.")

    _require_positive("slicer.step_size", float(cfg.slicer.step_size))
    _require_non_negative("slicer.fill_sheet_hole_length", int(cfg.slicer.fill_sheet_hole_length))

    count = cfg.analyzer.count
    _require_positive("analyzer.count.min_points_per_layer", int(count.min_points_per_layer))
    _require_non_negative("analyzer.count.min_usable_layers", int(count.min_usable_layers))
    _require_non_negative(
        "analyzer.count.min_strand_support_layers",
        int(count.min_strand_support_layers),
    )
    _require_ratio(
        "analyzer.count.min_strand_support_fraction",
        float(count.min_strand_support_fraction),
    )
    _require_non_negative(
        "analyzer.count.consensus_min_strand_support_layers",
        int(count.consensus_min_strand_support_layers),
    )
    _require_ratio(
        "analyzer.count.consensus_min_strand_support_fraction",
        float(count.consensus_min_strand_support_fraction),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_max_run_id_gap",
        int(count.trajectory_merge_max_run_id_gap),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_max_layer_gap",
        int(count.trajectory_merge_max_layer_gap),
    )
    _require_ratio(
        "analyzer.count.trajectory_merge_max_overlap_fraction",
        float(count.trajectory_merge_max_overlap_fraction),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_angle_tolerance_deg",
        float(count.trajectory_merge_angle_tolerance_deg),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_slope_tolerance_deg_per_layer",
        float(count.trajectory_merge_slope_tolerance_deg_per_layer),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_extended_max_run_id_gap",
        int(count.trajectory_merge_extended_max_run_id_gap),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_extended_angle_tolerance_deg",
        float(count.trajectory_merge_extended_angle_tolerance_deg),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_extended_min_candidate_strands",
        int(count.trajectory_merge_extended_min_candidate_strands),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_extended_min_layer_q90",
        int(count.trajectory_merge_extended_min_layer_q90),
    )
    _require_non_negative(
        "analyzer.count.trajectory_merge_extended_min_support_gap",
        int(count.trajectory_merge_extended_min_support_gap),
    )
    _require_ratio(
        "analyzer.count.min_layer_coverage_fraction",
        float(count.min_layer_coverage_fraction),
    )
    _require_ratio("analyzer.count.min_confidence", float(count.min_confidence))
    _require_positive("analyzer.count.min_count", int(count.min_count))
    _require_positive("analyzer.count.max_count", int(count.max_count))
    if int(count.max_count) < int(count.min_count):
        raise ConfigValidationError("`analyzer.count.max_count` must be >= `analyzer.count.min_count`.")
    for field_name in (
        "score_support_consensus_weight",
        "score_layer_q90_weight",
        "score_layer_q75_weight",
        "score_layer_max_weight",
        "score_candidate_weight",
        "score_persistent_weight",
        "score_joint_support_candidate_weight",
        "score_exact_support_candidate_bonus",
        "score_layer_agreement_weight",
        "score_selected_mean_support_weight",
        "score_selected_min_support_weight",
        "score_next_consensus_penalty",
        "score_next_support_penalty",
        "score_over_candidate_base_penalty",
        "score_over_candidate_per_count_penalty",
        "score_under_persistent_per_count_penalty",
        "score_under_support_gap_per_count_penalty",
        "score_under_support_gap_max_penalty",
        "score_over_support_gap_per_count_penalty",
    ):
        _require_non_negative(f"analyzer.count.{field_name}", float(getattr(count, field_name)))
    _require_non_negative(
        "analyzer.count.score_over_support_gap_min_delta",
        int(count.score_over_support_gap_min_delta),
    )
    _require_positive(
        "analyzer.count.score_layer_agreement_min_tolerance",
        float(count.score_layer_agreement_min_tolerance),
    )
    _require_ratio(
        "analyzer.count.score_layer_agreement_count_fraction",
        float(count.score_layer_agreement_count_fraction),
    )
    for field_name in (
        "score_support_consensus_scale",
        "score_layer_q90_scale",
        "score_layer_q75_scale",
        "score_layer_max_scale",
        "score_candidate_scale",
        "score_persistent_scale",
        "score_joint_support_scale",
        "score_joint_candidate_scale",
    ):
        _require_positive(f"analyzer.count.{field_name}", float(getattr(count, field_name)))

    refine_angle = float(cfg.analyzer.axis_search.refine.angle_deg)
    _require_non_negative(
        "analyzer.axis_search.min_points_per_scored_layer",
        int(cfg.analyzer.axis_search.min_points_per_scored_layer),
    )
    _require_positive(
        "analyzer.axis_search.report_candidate_limit",
        int(cfg.analyzer.axis_search.report_candidate_limit),
    )
    for field_name in (
        "report_confidence_weight",
        "report_decision_weight",
    ):
        _require_non_negative(
            f"analyzer.axis_search.{field_name}",
            float(getattr(cfg.analyzer.axis_search, field_name)),
        )
    if not 0.0 <= refine_angle <= 180.0:
        raise ConfigValidationError("`analyzer.axis_search.refine.angle_deg` must be between 0 and 180.")

    nn = cfg.analyzer.rules.nearest_neighbor
    _require_non_negative("analyzer.rules.nearest_neighbor.max_robust_cv", float(nn.max_robust_cv))
    _require_ratio("analyzer.rules.nearest_neighbor.min_inlier_frac", float(nn.min_inlier_frac))
    angle = cfg.analyzer.rules.angle
    if not 0.0 <= float(angle.max_gap_deg) <= 360.0:
        raise ConfigValidationError("`analyzer.rules.angle.max_gap_deg` must be between 0 and 360.")
    _require_non_negative("analyzer.rules.angle.order.local_step_max", int(angle.order.local_step_max))
    _require_ratio("analyzer.rules.angle.order.min_local_frac", float(angle.order.min_local_frac))
    _require_ratio(
        "analyzer.rules.angle.order.max_mean_circ_dist_norm",
        float(angle.order.max_mean_circ_dist_norm),
    )

    sequence_core = cfg.analyzer.rules.sequence_core
    _require_positive(
        "analyzer.rules.sequence_core.max_exhaustive_points",
        int(sequence_core.max_exhaustive_points),
    )
    _require_ratio(
        "analyzer.rules.sequence_core.max_terminal_trim_fraction",
        float(sequence_core.max_terminal_trim_fraction),
    )
    _require_ratio(
        "analyzer.rules.sequence_core.min_core_fraction",
        float(sequence_core.min_core_fraction),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.min_quality_gain",
        float(sequence_core.min_quality_gain),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_min_layer_count_max",
        int(sequence_core.global_min_layer_count_max),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_min_trajectory_strands",
        int(sequence_core.global_min_trajectory_strands),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_min_support_gap",
        int(sequence_core.global_min_support_gap),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_min_prediction_gap",
        int(sequence_core.global_min_prediction_gap),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_layer_count_allowance",
        int(sequence_core.global_layer_count_allowance),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_min_trimmed_trajectories",
        int(sequence_core.global_min_trimmed_trajectories),
    )
    _require_ratio(
        "analyzer.rules.sequence_core.global_min_centrality",
        float(sequence_core.global_min_centrality),
    )
    _require_ratio(
        "analyzer.rules.sequence_core.global_min_window_support_fraction",
        float(sequence_core.global_min_window_support_fraction),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_min_window_layer_max_floor",
        int(sequence_core.global_min_window_layer_max_floor),
    )
    _require_non_negative(
        "analyzer.rules.sequence_core.global_min_window_q90_floor",
        int(sequence_core.global_min_window_q90_floor),
    )
    _require_ratio(
        "analyzer.rules.sequence_core.global_min_window_layer_max_fraction",
        float(sequence_core.global_min_window_layer_max_fraction),
    )
    _require_ratio(
        "analyzer.rules.sequence_core.global_min_window_q90_fraction",
        float(sequence_core.global_min_window_q90_fraction),
    )
    for field_name in (
        "global_score_support_weight",
        "global_score_outside_support_weight",
        "global_score_centrality_weight",
        "global_score_layer_preservation_weight",
        "global_score_trimmed_fraction_weight",
        "global_score_length_penalty_weight",
        "run_window_min_score",
        "run_window_score_count_improvement_weight",
        "run_window_score_layer_preservation_weight",
        "run_window_score_outside_support_weight",
        "run_window_score_z_coverage_weight",
        "run_window_score_centrality_weight",
        "run_window_score_disappearance_weight",
        "run_window_score_absent_improvement_weight",
        "run_window_score_drop_improvement_weight",
        "run_window_score_count_margin_weight",
        "run_window_score_length_penalty_weight",
    ):
        _require_non_negative(
            f"analyzer.rules.sequence_core.{field_name}",
            float(getattr(sequence_core, field_name)),
        )
    for field_name in (
        "run_window_min_candidate_strands",
        "run_window_min_base_layer_max",
        "run_window_min_support_gap",
        "run_window_min_prediction_gap",
        "run_window_min_count_over_base_layer_max",
        "run_window_low_layer_max_threshold",
        "run_window_low_layer_max_min_count_over_base_layer_max",
        "run_window_max_count_over_base_layer_max",
        "run_window_min_count_over_layer_max",
        "run_window_min_count_layer_improvement",
        "run_window_min_absent_from_max",
        "run_window_min_trimmed_runs",
        "run_window_terminal_band_size",
        "run_window_boundary_padding_step",
        "run_window_boundary_max_padding",
        "run_window_min_raw_span",
        "run_window_max_raw_span",
        "run_window_max_scored_windows",
        "run_window_min_layer_max_floor",
        "run_window_min_layer_q90_floor",
    ):
        _require_non_negative(
            f"analyzer.rules.sequence_core.{field_name}",
            int(getattr(sequence_core, field_name)),
        )
    for field_name in (
        "run_window_min_layer_max_fraction",
        "run_window_min_layer_q90_fraction",
        "run_window_min_z_coverage_fraction",
    ):
        _require_ratio(
            f"analyzer.rules.sequence_core.{field_name}",
            float(getattr(sequence_core, field_name)),
        )

    graph = cfg.analyzer.rules.barrel_wall_graph
    for field_name in (
        "min_layer_points",
        "min_node_support_layers",
        "min_selected_count",
        "max_selected_count",
        "outer_filter_min_points",
        "outer_gap_min_removed_points",
        "complete_min_layer_q90",
        "complete_max_q90_spread",
        "complete_layer_max_candidate_max_q90_delta",
        "complete_outer_gap_min_layer_q90",
        "complete_outer_gap_max_q90_spread",
        "complete_outer_gap_min_reduction",
        "complete_layer_cohort_min_layers",
        "sparse_membership_min_selected_count",
        "sparse_membership_max_selected_count",
        "sparse_membership_min_edge_min_count",
        "sparse_membership_max_selected_over_support",
        "sparse_membership_max_support_reduction_without_plateau",
        "sparse_membership_min_plateau_for_large_reduction",
        "sparse_membership_max_layer_q90",
        "sparse_membership_variant_agreement_tolerance",
        "sparse_membership_same_edge_tolerance",
        "sparse_membership_plateau_tolerance",
        "sparse_membership_min_same_edge_variants_for_selection",
        "sparse_membership_min_plateau_for_selection",
    ):
        _require_non_negative(
            f"analyzer.rules.barrel_wall_graph.{field_name}",
            int(getattr(graph, field_name)),
        )
    if int(graph.max_selected_count) < int(graph.min_selected_count):
        raise ConfigValidationError(
            "`analyzer.rules.barrel_wall_graph.max_selected_count` must be >= "
            "`analyzer.rules.barrel_wall_graph.min_selected_count`."
        )
    for field_name in ("edge_min_counts", "outer_edge_min_counts"):
        values = list(getattr(graph, field_name))
        if not values:
            raise ConfigValidationError(
                f"`analyzer.rules.barrel_wall_graph.{field_name}` must not be empty."
            )
        for value in values:
            _require_positive(
                f"analyzer.rules.barrel_wall_graph.{field_name}[]",
                int(value),
            )
    _require_ratio(
        "analyzer.rules.barrel_wall_graph.outer_filter_quantile",
        float(graph.outer_filter_quantile),
    )
    _require_ratio(
        "analyzer.rules.barrel_wall_graph.outer_gap_min_fraction",
        float(graph.outer_gap_min_fraction),
    )
    _require_ratio(
        "analyzer.rules.barrel_wall_graph.complete_outer_gap_min_count_fraction",
        float(graph.complete_outer_gap_min_count_fraction),
    )
    _require_ratio(
        "analyzer.rules.barrel_wall_graph.complete_layer_cohort_min_fraction",
        float(graph.complete_layer_cohort_min_fraction),
    )
    _require_positive(
        "analyzer.rules.barrel_wall_graph.complete_max_support_ratio",
        float(graph.complete_max_support_ratio),
    )
    _require_positive(
        "analyzer.rules.barrel_wall_graph.complete_outer_gap_max_support_ratio",
        float(graph.complete_outer_gap_max_support_ratio),
    )
    _require_positive(
        "analyzer.rules.barrel_wall_graph.sparse_min_support_ratio",
        float(graph.sparse_min_support_ratio),
    )
    for field_name in (
        "sparse_membership_max_component_avg_degree",
        "sparse_membership_same_edge_max_component_avg_degree",
        "sparse_membership_min_score",
        "sparse_membership_plateau_scale",
        "sparse_membership_drop_scale",
        "sparse_membership_degree_weight",
        "sparse_membership_degree_scale",
        "sparse_membership_cycle_rank_weight",
        "sparse_membership_cycle_rank_scale",
        "sparse_membership_degree2_fraction_weight",
        "sparse_membership_drop_weight",
        "sparse_membership_plateau_weight",
        "sparse_membership_variant_agreement_weight",
        "sparse_membership_same_edge_agreement_weight",
        "sparse_membership_support_fit_weight",
        "sparse_membership_layer_lift_weight",
        "sparse_membership_edge_weight",
        "sparse_membership_radial_rank_weight",
        "sparse_membership_branch_penalty_weight",
        "sparse_membership_block_penalty_weight",
        "sparse_membership_outer_bonus",
        "sparse_membership_outer_gap_bonus",
        "partial_layer_close_max_gap_ratio",
    ):
        _require_non_negative(
            f"analyzer.rules.barrel_wall_graph.{field_name}",
            float(getattr(graph, field_name)),
        )
    _require_ratio(
        "analyzer.rules.barrel_wall_graph.partial_layer_min_closed_coverage_fraction",
        float(graph.partial_layer_min_closed_coverage_fraction),
    )
    if int(graph.sparse_membership_max_selected_count) < int(
        graph.sparse_membership_min_selected_count
    ):
        raise ConfigValidationError(
            "`analyzer.rules.barrel_wall_graph.sparse_membership_max_selected_count` "
            "must be >= `sparse_membership_min_selected_count`."
        )

    radial = cfg.analyzer.rules.radial_outlier
    _require_positive("analyzer.rules.radial_outlier.min_points", int(radial.min_points))
    _require_ratio(
        "analyzer.rules.radial_outlier.min_radius_ratio",
        float(radial.min_radius_ratio),
    )
    _require_positive(
        "analyzer.rules.radial_outlier.max_radius_ratio",
        float(radial.max_radius_ratio),
    )
    if float(radial.max_radius_ratio) < float(radial.min_radius_ratio):
        raise ConfigValidationError(
            "`analyzer.rules.radial_outlier.max_radius_ratio` must be >= "
            "`min_radius_ratio`."
        )

    confidence = cfg.analyzer.confidence
    _require_non_negative(
        "analyzer.confidence.layer_count_support_tolerance",
        int(confidence.layer_count_support_tolerance),
    )
    _require_positive(
        "analyzer.confidence.layer_agreement_min_tolerance",
        float(confidence.layer_agreement_min_tolerance),
    )
    _require_ratio(
        "analyzer.confidence.layer_agreement_count_fraction",
        float(confidence.layer_agreement_count_fraction),
    )
    _require_positive(
        "analyzer.confidence.decision_score_normalizer",
        float(confidence.decision_score_normalizer),
    )
    for field_name in (
        "feature_support_consensus_weight",
        "feature_trajectory_candidate_weight",
        "feature_layer_q90_weight",
        "decision_agreement_weight",
        "feature_agreement_weight",
        "selected_support_weight",
        "layer_support_fraction_weight",
        "layer_agreement_weight",
        "usable_layer_fraction_weight",
        "geometric_pass_fraction_weight",
        "max_outlier_penalty",
        "radial_outlier_layer_penalty",
        "trimmed_layer_penalty",
    ):
        _require_non_negative(
            f"analyzer.confidence.{field_name}",
            float(getattr(confidence, field_name)),
        )
    for field_name in (
        "feature_support_consensus_scale",
        "feature_trajectory_candidate_scale",
        "feature_layer_q90_scale",
    ):
        _require_positive(
            f"analyzer.confidence.{field_name}",
            float(getattr(confidence, field_name)),
        )

    layer_quality = cfg.analyzer.layer_quality
    _require_non_negative(
        "analyzer.layer_quality.min_geometry_points",
        int(layer_quality.min_geometry_points),
    )
    _require_non_negative("analyzer.layer_quality.usable_bonus", float(layer_quality.usable_bonus))
    _require_non_negative(
        "analyzer.layer_quality.geometric_pass_bonus",
        float(layer_quality.geometric_pass_bonus),
    )
    _require_positive(
        "analyzer.layer_quality.angle_margin_scale_deg",
        float(layer_quality.angle_margin_scale_deg),
    )
    _require_non_negative(
        "analyzer.layer_quality.angle_margin_cap",
        float(layer_quality.angle_margin_cap),
    )
    for field_name in (
        "terminal_order_local_gain",
        "terminal_order_global_gain",
        "terminal_nn_inlier_gain",
    ):
        _require_non_negative(
            f"analyzer.layer_quality.{field_name}",
            float(getattr(layer_quality, field_name)),
        )


def sync_compat_config(cfg: AppConfig) -> None:
    for compat_name, path in COMPAT_OVERRIDE_PATHS.items():
        target: object = cfg
        for part in path.split("."):
            target = getattr(target, part)
        setattr(Config, compat_name, target)


sync_compat_config(AppConfig())
