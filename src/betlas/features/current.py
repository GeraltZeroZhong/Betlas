"""Current descriptor contract. Feature order is independent of input values."""
from __future__ import annotations

import numpy as np

FEATURE_SCHEMA = "betlas.classifier.circular-moments-valid-planes.v3"
FEATURE_COLUMNS = (
    'betlas_residue_count',
    'betlas_beta_residue_count',
    'betlas_helix_residue_count',
    'betlas_beta_residue_fraction',
    'betlas_helix_residue_fraction',
    'betlas_beta_strand_count',
    'betlas_helix_count',
    'betlas_pca_elongation',
    'betlas_pca_flatness',
    'betlas_axis_best_score',
    'betlas_axis_best_is_strand_axis',
    'betlas_axis_best_is_point_pc1',
    'betlas_axis_best_is_point_pc2',
    'betlas_axis_best_is_point_pc3',
    'betlas_z_continuity_fraction',
    'betlas_parallel_sheet_order_fraction',
    'betlas_antiparallel_sheet_order_fraction',
    'betlas_axis_best_angular_coverage',
    'betlas_axis_best_largest_gap_fraction',
    'betlas_axis_best_radius_cv',
    'betlas_axis_best_z_range',
    'betlas_axis_best_z_continuity_fraction',
    'betlas_axis_best_slice_count',
    'betlas_axis_best_slice_coverage_mean',
    'betlas_axis_best_slice_coverage_median',
    'betlas_axis_best_slice_coverage_std',
    'betlas_axis_best_slice_high_coverage_fraction',
    'betlas_axis_best_slice_largest_gap_fraction_mean',
    'betlas_strand_length_mean',
    'betlas_strand_length_std',
    'betlas_strand_length_min',
    'betlas_strand_length_median',
    'betlas_strand_length_max',
    'betlas_helix_length_mean',
    'betlas_helix_length_std',
    'betlas_helix_length_min',
    'betlas_helix_length_median',
    'betlas_helix_length_max',
    'betlas_beta_alpha_alternation_fraction',
    'betlas_beta_alpha_repeat_pairs',
    'betlas_sse_count',
    'betlas_sheet_count',
    'betlas_sheet_face_count',
    'betlas_largest_sheet_fraction',
    'betlas_sheet_size_entropy',
    'betlas_sheet_strand_count_mean',
    'betlas_sheet_strand_count_std',
    'betlas_sheet_strand_count_min',
    'betlas_sheet_strand_count_median',
    'betlas_sheet_strand_count_max',
    'betlas_sheet_angular_span_mean',
    'betlas_sheet_angular_span_std',
    'betlas_sheet_angular_span_min',
    'betlas_sheet_angular_span_median',
    'betlas_sheet_angular_span_max',
    'betlas_sheet_planarity_residual_mean',
    'betlas_sheet_planarity_residual_std',
    'betlas_sheet_planarity_residual_min',
    'betlas_sheet_planarity_residual_median',
    'betlas_sheet_planarity_residual_max',
    'betlas_sheet_axis_dispersion_mean',
    'betlas_sheet_axis_dispersion_std',
    'betlas_sheet_axis_dispersion_min',
    'betlas_sheet_axis_dispersion_median',
    'betlas_sheet_axis_dispersion_max',
    'betlas_beta_run_count',
    'betlas_beta_segment_to_run_ratio',
    'betlas_beta_run_eight_score',
    'betlas_beta_run_sheet_multiplicity_mean',
    'betlas_beta_run_sheet_multiplicity_max',
    'betlas_beta_run_multi_sheet_fraction',
    'betlas_beta_run_angular_coverage',
    'betlas_beta_run_largest_gap_fraction',
    'betlas_beta_run_length_mean',
    'betlas_beta_run_length_std',
    'betlas_beta_run_length_min',
    'betlas_beta_run_length_median',
    'betlas_beta_run_length_max',
    'betlas_sheet_pair_top2_fraction',
    'betlas_sheet_pair_size_balance',
    'betlas_sheet_pair_centroid_distance',
    'betlas_sheet_pair_centroid_distance_norm',
    'betlas_sheet_pair_angular_separation_fraction',
    'betlas_sheet_pair_cross_min_distance',
    'betlas_sheet_pair_cross_distance_mean',
    'betlas_sheet_pair_cross_contact_density8',
    'betlas_sheet_pair_cross_contact_density10',
    'betlas_sheet_pair_reciprocal_nearest_fraction8',
    'betlas_sheet_seq_top2_run_fraction',
    'betlas_sheet_seq_top2_transition_fraction',
    'betlas_sheet_seq_top2_block_count',
    'betlas_sheet_seq_top2_longest_block_fraction',
    'betlas_sheet_seq_top2_interleave_score',
    'betlas_sheet_seq_top2_order_displacement',
    'betlas_sheet_seq_greek_key_proxy',
    'betlas_sheet_order_adjacent_seq_step_mean',
    'betlas_sheet_order_adjacent_seq_step_max',
    'betlas_sheet_order_nonlocal_fraction',
    'betlas_sheet_order_inversion_fraction_mean',
    'betlas_top2_sheet_order_adjacent_seq_step_mean',
    'betlas_top2_sheet_order_nonlocal_fraction',
    'betlas_top2_sheet_order_inversion_fraction_mean',
    'betlas_jelly_roll_order_nonlocal_score',
    'betlas_contact8_edge_density',
    'betlas_contact10_edge_density',
    'betlas_contact8_cross_sheet_fraction',
    'betlas_contact8_component_count',
    'betlas_contact8_largest_component_fraction',
    'betlas_contact8_cycle_rank_norm',
    'betlas_contact8_degree2_fraction',
    'betlas_contact8_seq_gap_mean',
    'betlas_contact8_seq_gap_max',
    'betlas_contact8_nonlocal_fraction',
    'betlas_contact8_cross_sheet_nonlocal_fraction',
    'betlas_contact10_seq_gap_mean',
    'betlas_contact10_nonlocal_fraction',
    'betlas_strand_order_displacement_mean',
    'betlas_strand_order_adjacent_seq_step_median',
    'betlas_strand_ntc_parallel_fraction',
    'betlas_strand_ntc_antiparallel_fraction',
    'betlas_strand_axis_abs_alignment_mean',
    'betlas_alpha_shell_radial_delta',
    'betlas_helix_beta_radius_ratio',
    'betlas_angular_gap_cv',
    'betlas_angular_circular_moment_k1',
    'betlas_angular_circular_moment_k2',
    'betlas_angular_circular_moment_k3',
    'betlas_angular_circular_moment_k4',
    'betlas_angular_circular_moment_k5',
    'betlas_angular_circular_moment_k6',
    'betlas_angular_circular_moment_k7',
    'betlas_angular_circular_moment_k8',
    'betlas_sheet_pair_valid_normal_abs_dot',
    'betlas_sheet_pair_valid_face_alignment',
    'betlas_sheet_pair_valid_bilayer_score',
    'betlas_sheet_pair_valid_lobe_score',
)

FEATURE_FAMILIES = {'composition': ('betlas_residue_count', 'betlas_beta_residue_count', 'betlas_helix_residue_count', 'betlas_beta_residue_fraction', 'betlas_helix_residue_fraction', 'betlas_beta_strand_count', 'betlas_helix_count', 'betlas_strand_length_mean', 'betlas_strand_length_std', 'betlas_strand_length_min', 'betlas_strand_length_median', 'betlas_strand_length_max', 'betlas_helix_length_mean', 'betlas_helix_length_std', 'betlas_helix_length_min', 'betlas_helix_length_median', 'betlas_helix_length_max', 'betlas_sse_count', 'betlas_helix_beta_radius_ratio'), 'global_shape': ('betlas_pca_elongation', 'betlas_pca_flatness'), 'sheet_global': ('betlas_parallel_sheet_order_fraction', 'betlas_antiparallel_sheet_order_fraction', 'betlas_sheet_count', 'betlas_sheet_face_count', 'betlas_largest_sheet_fraction', 'betlas_sheet_size_entropy', 'betlas_sheet_strand_count_mean', 'betlas_sheet_strand_count_std', 'betlas_sheet_strand_count_min', 'betlas_sheet_strand_count_median', 'betlas_sheet_strand_count_max', 'betlas_sheet_angular_span_mean', 'betlas_sheet_angular_span_std', 'betlas_sheet_angular_span_min', 'betlas_sheet_angular_span_median', 'betlas_sheet_angular_span_max', 'betlas_sheet_planarity_residual_mean', 'betlas_sheet_planarity_residual_std', 'betlas_sheet_planarity_residual_min', 'betlas_sheet_planarity_residual_median', 'betlas_sheet_planarity_residual_max', 'betlas_sheet_axis_dispersion_mean', 'betlas_sheet_axis_dispersion_std', 'betlas_sheet_axis_dispersion_min', 'betlas_sheet_axis_dispersion_median', 'betlas_sheet_axis_dispersion_max'), 'beta_run_topology': ('betlas_beta_run_count', 'betlas_beta_segment_to_run_ratio', 'betlas_beta_run_eight_score', 'betlas_beta_run_sheet_multiplicity_mean', 'betlas_beta_run_sheet_multiplicity_max', 'betlas_beta_run_multi_sheet_fraction', 'betlas_beta_run_angular_coverage', 'betlas_beta_run_largest_gap_fraction', 'betlas_beta_run_length_mean', 'betlas_beta_run_length_std', 'betlas_beta_run_length_min', 'betlas_beta_run_length_median', 'betlas_beta_run_length_max'), 'sheet_pair_packing': ('betlas_sheet_pair_top2_fraction', 'betlas_sheet_pair_size_balance', 'betlas_sheet_pair_centroid_distance', 'betlas_sheet_pair_centroid_distance_norm', 'betlas_sheet_pair_angular_separation_fraction', 'betlas_sheet_pair_cross_min_distance', 'betlas_sheet_pair_cross_distance_mean', 'betlas_sheet_pair_cross_contact_density8', 'betlas_sheet_pair_cross_contact_density10', 'betlas_sheet_pair_reciprocal_nearest_fraction8', 'betlas_sheet_pair_valid_normal_abs_dot', 'betlas_sheet_pair_valid_face_alignment', 'betlas_sheet_pair_valid_bilayer_score', 'betlas_sheet_pair_valid_lobe_score'), 'sheet_sequence_topology': ('betlas_sheet_seq_top2_run_fraction', 'betlas_sheet_seq_top2_transition_fraction', 'betlas_sheet_seq_top2_block_count', 'betlas_sheet_seq_top2_longest_block_fraction', 'betlas_sheet_seq_top2_interleave_score', 'betlas_sheet_seq_top2_order_displacement', 'betlas_sheet_seq_greek_key_proxy', 'betlas_sheet_order_adjacent_seq_step_mean', 'betlas_sheet_order_adjacent_seq_step_max', 'betlas_sheet_order_nonlocal_fraction', 'betlas_sheet_order_inversion_fraction_mean', 'betlas_top2_sheet_order_adjacent_seq_step_mean', 'betlas_top2_sheet_order_nonlocal_fraction', 'betlas_top2_sheet_order_inversion_fraction_mean', 'betlas_jelly_roll_order_nonlocal_score'), 'contact_graph': ('betlas_contact8_edge_density', 'betlas_contact10_edge_density', 'betlas_contact8_cross_sheet_fraction', 'betlas_contact8_component_count', 'betlas_contact8_largest_component_fraction', 'betlas_contact8_cycle_rank_norm', 'betlas_contact8_degree2_fraction', 'betlas_contact8_seq_gap_mean', 'betlas_contact8_seq_gap_max', 'betlas_contact8_nonlocal_fraction', 'betlas_contact8_cross_sheet_nonlocal_fraction', 'betlas_contact10_seq_gap_mean', 'betlas_contact10_nonlocal_fraction'), 'axis_closure': ('betlas_axis_best_score', 'betlas_axis_best_is_strand_axis', 'betlas_axis_best_is_point_pc1', 'betlas_axis_best_is_point_pc2', 'betlas_axis_best_is_point_pc3', 'betlas_z_continuity_fraction', 'betlas_axis_best_angular_coverage', 'betlas_axis_best_largest_gap_fraction', 'betlas_axis_best_radius_cv', 'betlas_axis_best_z_range', 'betlas_axis_best_z_continuity_fraction', 'betlas_axis_best_slice_count', 'betlas_axis_best_slice_coverage_mean', 'betlas_axis_best_slice_coverage_median', 'betlas_axis_best_slice_coverage_std', 'betlas_axis_best_slice_high_coverage_fraction', 'betlas_axis_best_slice_largest_gap_fraction_mean', 'betlas_angular_gap_cv'), 'axis_periodicity': ('betlas_angular_circular_moment_k3', 'betlas_angular_circular_moment_k4', 'betlas_angular_circular_moment_k5', 'betlas_angular_circular_moment_k6', 'betlas_angular_circular_moment_k7', 'betlas_angular_circular_moment_k8'), 'strand_order': ('betlas_strand_order_displacement_mean', 'betlas_strand_order_adjacent_seq_step_median', 'betlas_strand_ntc_parallel_fraction', 'betlas_strand_ntc_antiparallel_fraction', 'betlas_strand_axis_abs_alignment_mean'), 'alpha_shell': ('betlas_beta_alpha_alternation_fraction', 'betlas_beta_alpha_repeat_pairs', 'betlas_alpha_shell_radial_delta'), 'angular_lobes': ('betlas_angular_circular_moment_k1', 'betlas_angular_circular_moment_k2')}

FAMILY_DEFINITIONS = {'composition': 'Residue/SSE counts, strand and helix lengths, beta/alpha abundance and domain segmentation', 'global_shape': 'PCA eigenvalue ratios describing spatial extent and anisotropy', 'sheet_global': 'Sheet counts/sizes, angular spans, planarity, strand-axis dispersion and sheet parallelism', 'beta_run_topology': 'Sequence-contiguous beta runs, run lengths, multiplicity and angular arrangement', 'sheet_pair_packing': 'Top-two sheet separation, valid-plane alignment, cross contacts and packing', 'sheet_sequence_topology': 'Sheet sequence interleaving and pairing-derived strand-order/nonlocality', 'contact_graph': 'Beta-strand contact graph density, components, cycles, degree and sequence locality', 'axis_closure': 'Axis-conditioned angular closure, slice occupancy/continuity and wall support', 'axis_periodicity': 'Higher-order angular repetition: continuous circular-moment magnitudes k=3..8', 'strand_order': 'Strand direction/order relative to spatial axes and neighbors', 'alpha_shell': 'Helix context relative to beta geometry and sequence alternation', 'angular_lobes': 'Low-order angular organization: continuous circular-moment magnitudes k=1,2'}

CIRCULAR_MOMENT_COLUMNS = tuple(f"betlas_angular_circular_moment_k{k}" for k in range(1, 9))
VALID_PLANE_COLUMNS = tuple("betlas_sheet_pair_valid_" + suffix for suffix in (
    "normal_abs_dot", "face_alignment", "bilayer_score", "lobe_score"))
AXIS_DEPENDENT_COLUMNS = (
    "betlas_z_continuity_fraction", "betlas_beta_run_angular_coverage", "betlas_beta_run_largest_gap_fraction",
    "betlas_sheet_pair_centroid_distance_norm", "betlas_sheet_pair_angular_separation_fraction",
    "betlas_sheet_pair_valid_bilayer_score", "betlas_sheet_pair_valid_lobe_score",
    "betlas_strand_order_displacement_mean", "betlas_strand_order_adjacent_seq_step_median",
    "betlas_alpha_shell_radial_delta", "betlas_helix_beta_radius_ratio", "betlas_angular_gap_cv",
    *CIRCULAR_MOMENT_COLUMNS,
)


def circular_moments(angles, maximum_order=8):
    """Return magnitudes of continuous circular Fourier moments."""
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        return np.full(maximum_order, np.nan)
    return np.abs(np.exp(1j * angles[:, None] * np.arange(1, maximum_order + 1)).mean(axis=0))


def apply_applicability(features):
    """Retain sparse structures while marking undefined measurements as NaN."""
    output = dict(features)
    sparse = float(output["betlas_beta_strand_count"]) < 4
    if sparse:
        for field in output:
            if field.startswith(("betlas_axis_best_", "betlas_sheet_angular_span_")) or field in AXIS_DEPENDENT_COLUMNS:
                output[field] = np.nan
    if float(output["betlas_beta_residue_count"]) < 3:
        output["betlas_pca_elongation"] = np.nan
    output["axis_applicability"] = "too_few_beta_nodes" if sparse else "available"
    output["elongation_applicability"] = "too_few_beta_points" if float(output["betlas_beta_residue_count"]) < 3 else "available"
    output["feature_schema"] = FEATURE_SCHEMA
    return output
