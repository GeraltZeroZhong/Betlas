from __future__ import annotations

from math import pi

DEFAULT_INPUT_PATH = ""
DEFAULT_OUTPUT_CSV = "beta_barrel_staves_results.csv"
DEFAULT_ALLOWED_SUFFIXES = (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz")
DEFAULT_SLICE_STEP_SIZE = 1.0
DEFAULT_FILL_SHEET_HOLE_LENGTH = 1

DEFAULT_MIN_CHAIN_RESIDUES = 20
DEFAULT_MIN_SHEET_RESIDUES = 10
DEFAULT_MIN_INFORMATIVE_SLICES = 3
MIN_NEAREST_NEIGHBOR_POINTS = 3
MIN_ANGULAR_GAP_POINTS = 5
MIN_SEQUENCE_ANGLE_ORDER_POINTS = 6
SMALL_NEAREST_NEIGHBOR_MATRIX_POINTS = 64

ROBUST_SIGMA_SCALE = 1.4826
THREE_SIGMA_MULTIPLIER = 3.0
RAD_TO_DEG = 180.0 / pi
FULL_ROTATION_DEG = 360.0

EPSILON = 1e-12
TOLERANCE = 1e-9
COVARIANCE_FLOOR = 1e-12

RESULT_COUNTED = "COUNTED"
RESULT_LOW_CONFIDENCE = "LOW_CONFIDENCE"
RESULT_FILTERED_OUT = "FILTERED_OUT"
RESULT_ERROR = "ERROR"

DEFAULT_RESULT_COLUMNS = (
    "filename",
    "source_path",
    "chain",
    "result",
    "result_stage",
    "strand_count",
    "score_type",
    "calibration_status",
    "config_profile",
    "confidence",
    "confidence_basis",
    "count_threshold",
    "barrel_gate_enabled",
    "barrel_gate_passed",
    "barrel_gate_result",
    "barrel_gate_score",
    "barrel_gate_reason",
    "layer_count_mode",
    "layer_count_median",
    "supporting_layers",
    "usable_layers",
    "total_layers",
    "layer_support_fraction",
    "usable_layer_fraction",
    "geometric_pass_layers",
    "geometric_pass_fraction",
    "trimmed_layers",
    "radial_outlier_layers",
    "radial_outlier_points",
    "candidate_strands",
    "persistent_strands",
    "trajectory_candidate_strands",
    "trajectory_merge_count",
    "support_consensus_strands",
    "support_consensus_threshold",
    "layer_count_q75",
    "layer_count_q90",
    "layer_count_max",
    "selected_strand_support_mean",
    "selected_strand_support_min",
    "count_decision_score",
    "layer_count_max_plateau",
    "layer_count_largest_drop",
    "absent_from_max_strands",
    "terminal_strand_support",
    "terminal_consensus_strands",
    "terminal_support_fraction",
    "barrel_wall_graph_applied",
    "barrel_wall_graph_count",
    "barrel_wall_graph_variant",
    "barrel_wall_graph_edge_min_count",
    "barrel_wall_graph_visibility_regime",
    "barrel_wall_graph_selection_strategy",
    "barrel_wall_graph_support_ratio",
    "barrel_wall_graph_node_count",
    "barrel_wall_graph_edge_count",
    "barrel_wall_graph_component_edge_count",
    "barrel_wall_graph_component_avg_degree",
    "barrel_wall_graph_component_cycle_rank",
    "barrel_wall_graph_component_degree2_fraction",
    "barrel_wall_graph_component_branch_fraction",
    "barrel_wall_graph_component_radial_rank_mean",
    "barrel_wall_graph_component_radial_rank_min",
    "barrel_wall_graph_edge_density",
    "barrel_wall_graph_membership_score",
    "barrel_wall_graph_membership_block_count",
    "barrel_wall_graph_membership_largest_block_fraction",
    "barrel_wall_graph_membership_plateau_edges",
    "barrel_wall_graph_membership_next_drop_fraction",
    "barrel_wall_graph_membership_variant_agreement",
    "barrel_wall_graph_observed_layers",
    "barrel_wall_graph_selected_ids",
    "barrel_wall_graph_raw_curve",
    "barrel_wall_graph_outer_curve",
    "barrel_wall_graph_outer_gap_curve",
    "axis_hypothesis_name",
    "axis_hypothesis_score",
    "axis_hypothesis_proxy_rank",
    "axis_hypothesis_candidates",
    "run_window_core_applied",
    "run_window_core_score",
    "run_window_core_start",
    "run_window_core_stop",
    "run_window_core_trim_left",
    "run_window_core_trim_right",
    "run_window_core_candidate_windows",
    "run_window_core_reanalyzed_windows",
    "run_window_core_base_strand_count",
    "run_window_core_base_support_consensus",
    "run_window_core_base_layer_q90",
    "run_window_core_base_layer_max",
    "run_window_core_layer_max_preservation",
    "run_window_core_layer_q90_preservation",
    "run_window_core_outside_support_fraction",
    "run_window_core_z_coverage_fraction",
    "run_window_core_max_z_gap",
    "run_window_core_centrality",
    "run_window_core_count_layer_improvement",
    "run_window_core_absent_improvement",
    "run_window_core_drop_improvement",
    "run_window_core_count_margin",
    "chain_residues",
    "sheet_residues",
    "informative_slices",
    "reason",
)

DEFAULT_SUMMARY_COLUMNS = (
    "filename",
    "chain",
    "result",
    "strand_count",
    "confidence",
    "layer_counts",
    "candidate_strands",
    "reason",
)

SUMMARY_DISPLAY_NAMES = {
    "filename": "Filename",
    "chain": "Chain",
    "result": "Result",
    "strand_count": "Strands",
    "confidence": "Conf.",
    "layer_counts": "Consensus/Usable/Total Layers",
    "candidate_strands": "Candidates",
    "reason": "Reason",
}

SUMMARY_COLUMN_WIDTHS = {
    "filename": 20,
    "chain": 5,
    "result": 15,
    "strand_count": 7,
    "confidence": 7,
    "layer_counts": 22,
    "candidate_strands": 10,
    "reason": 64,
}

THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
