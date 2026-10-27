"""Column definitions for the current geometry representation and task results."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .features.current import FEATURE_COLUMNS
from .models import DomainCandidate


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    dtype: str
    units: str
    value_range: str
    definition: str
    formula: str
    missing_value: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    table: str
    role: str
    dtype: str
    required: bool
    definition: str
    units: str
    value_range: str
    missing_value: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DESCRIPTOR_DETAILS = {'betlas_residue_count': {'family': 'composition',
                          'dtype': 'numeric',
                          'units': 'dimensionless unless the definition names Angstrom distances',
                          'value_range': 'non-negative',
                          'definition': 'Betlas composition feature emitted by the deterministic '
                                        'geometry extraction pipeline.',
                          'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                     'betlas.slicing, or the matching readout module.',
                          'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                          'source': 'betlas.features.extract'},
 'betlas_beta_residue_count': {'family': 'composition',
                               'dtype': 'numeric',
                               'units': 'dimensionless unless the definition names Angstrom distances',
                               'value_range': 'non-negative',
                               'definition': 'Betlas composition feature emitted by the deterministic '
                                             'geometry extraction pipeline.',
                               'formula': 'Deterministic formula implemented in '
                                          'betlas.features.extract, betlas.slicing, or the matching '
                                          'readout module.',
                               'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                               'source': 'betlas.features.extract'},
 'betlas_helix_residue_count': {'family': 'composition',
                                'dtype': 'numeric',
                                'units': 'dimensionless unless the definition names Angstrom distances',
                                'value_range': 'non-negative',
                                'definition': 'Betlas composition feature emitted by the deterministic '
                                              'geometry extraction pipeline.',
                                'formula': 'Deterministic formula implemented in '
                                           'betlas.features.extract, betlas.slicing, or the matching '
                                           'readout module.',
                                'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                'source': 'betlas.features.extract'},
 'betlas_beta_residue_fraction': {'family': 'composition',
                                  'dtype': 'numeric',
                                  'units': 'dimensionless unless the definition names Angstrom '
                                           'distances',
                                  'value_range': '[0, 1]',
                                  'definition': 'Betlas composition feature emitted by the '
                                                'deterministic geometry extraction pipeline.',
                                  'formula': 'Deterministic formula implemented in '
                                             'betlas.features.extract, betlas.slicing, or the matching '
                                             'readout module.',
                                  'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                   'geometry',
                                  'source': 'betlas.features.extract'},
 'betlas_helix_residue_fraction': {'family': 'composition',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': '[0, 1]',
                                   'definition': 'Betlas composition feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_beta_strand_count': {'family': 'composition',
                              'dtype': 'numeric',
                              'units': 'dimensionless unless the definition names Angstrom distances',
                              'value_range': 'non-negative',
                              'definition': 'Betlas composition feature emitted by the deterministic '
                                            'geometry extraction pipeline.',
                              'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                         'betlas.slicing, or the matching readout module.',
                              'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                              'source': 'betlas.features.extract'},
 'betlas_helix_count': {'family': 'composition',
                        'dtype': 'numeric',
                        'units': 'dimensionless unless the definition names Angstrom distances',
                        'value_range': 'non-negative',
                        'definition': 'Betlas composition feature emitted by the deterministic geometry '
                                      'extraction pipeline.',
                        'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                   'betlas.slicing, or the matching readout module.',
                        'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                        'source': 'betlas.features.extract'},
 'betlas_pca_elongation': {'family': 'global_shape',
                           'dtype': 'numeric',
                           'units': 'dimensionless unless the definition names Angstrom distances',
                           'value_range': 'real-valued',
                           'definition': 'Betlas composition feature emitted by the deterministic '
                                         'geometry extraction pipeline.',
                           'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                      'betlas.slicing, or the matching readout module.',
                           'missing_value': 'Zero for empty counts; NaN for inapplicable geometry; NaN '
                                            'when beta_residue_count < 3',
                           'source': 'betlas.features.extract'},
 'betlas_pca_flatness': {'family': 'global_shape',
                         'dtype': 'numeric',
                         'units': 'dimensionless unless the definition names Angstrom distances',
                         'value_range': 'real-valued',
                         'definition': 'Betlas composition feature emitted by the deterministic '
                                       'geometry extraction pipeline.',
                         'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                    'betlas.slicing, or the matching readout module.',
                         'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                         'source': 'betlas.features.extract'},
 'betlas_axis_best_score': {'family': 'axis_closure',
                            'dtype': 'numeric',
                            'units': 'dimensionless unless the definition names Angstrom distances',
                            'value_range': 'real-valued',
                            'definition': 'Best-axis closure or z-slice descriptor computed from '
                                          'projected beta-structure coordinates.',
                            'formula': '0.40 * slice_coverage_median + 0.25 * angular_coverage + 0.20 * '
                                       'z_continuity_fraction + 0.15 * (1 - largest_gap_fraction)',
                            'missing_value': 'Zero for empty counts; NaN for inapplicable geometry; NaN '
                                             'when beta_strand_count < 4',
                            'source': 'betlas.features.extract'},
 'betlas_axis_best_is_strand_axis': {'family': 'axis_closure',
                                     'dtype': 'numeric',
                                     'units': 'dimensionless unless the definition names Angstrom '
                                              'distances',
                                     'value_range': '{0, 1}',
                                     'definition': 'Best-axis closure or z-slice descriptor computed '
                                                   'from projected beta-structure coordinates.',
                                     'formula': 'Deterministic formula implemented in '
                                                'betlas.features.extract, betlas.slicing, or the '
                                                'matching readout module.',
                                     'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                      'geometry; NaN when beta_strand_count < 4',
                                     'source': 'betlas.features.extract'},
 'betlas_axis_best_is_point_pc1': {'family': 'axis_closure',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': '{0, 1}',
                                   'definition': 'Best-axis closure or z-slice descriptor computed from '
                                                 'projected beta-structure coordinates.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry; NaN when beta_strand_count < 4',
                                   'source': 'betlas.features.extract'},
 'betlas_axis_best_is_point_pc2': {'family': 'axis_closure',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': '{0, 1}',
                                   'definition': 'Best-axis closure or z-slice descriptor computed from '
                                                 'projected beta-structure coordinates.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry; NaN when beta_strand_count < 4',
                                   'source': 'betlas.features.extract'},
 'betlas_axis_best_is_point_pc3': {'family': 'axis_closure',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': '{0, 1}',
                                   'definition': 'Best-axis closure or z-slice descriptor computed from '
                                                 'projected beta-structure coordinates.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry; NaN when beta_strand_count < 4',
                                   'source': 'betlas.features.extract'},
 'betlas_z_continuity_fraction': {'family': 'axis_closure',
                                  'dtype': 'numeric',
                                  'units': 'dimensionless unless the definition names Angstrom '
                                           'distances',
                                  'value_range': '[0, 1]',
                                  'definition': 'Best-axis closure or z-slice descriptor computed from '
                                                'projected beta-structure coordinates.',
                                  'formula': 'Deterministic formula implemented in '
                                             'betlas.features.extract, betlas.slicing, or the matching '
                                             'readout module.',
                                  'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                   'geometry; NaN when beta_strand_count < 4',
                                  'source': 'betlas.features.extract'},
 'betlas_parallel_sheet_order_fraction': {'family': 'sheet_global',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': '[0, 1]',
                                          'definition': 'Betlas sheet_geometry feature emitted by the '
                                                        'deterministic geometry extraction pipeline.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry',
                                          'source': 'betlas.features.extract'},
 'betlas_antiparallel_sheet_order_fraction': {'family': 'sheet_global',
                                              'dtype': 'numeric',
                                              'units': 'dimensionless unless the definition names '
                                                       'Angstrom distances',
                                              'value_range': '[0, 1]',
                                              'definition': 'Betlas sheet_geometry feature emitted by '
                                                            'the deterministic geometry extraction '
                                                            'pipeline.',
                                              'formula': 'Deterministic formula implemented in '
                                                         'betlas.features.extract, betlas.slicing, or '
                                                         'the matching readout module.',
                                              'missing_value': 'Zero for empty counts; NaN for '
                                                               'inapplicable geometry',
                                              'source': 'betlas.features.extract'},
 'betlas_axis_best_angular_coverage': {'family': 'axis_closure',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless unless the definition names Angstrom '
                                                'distances',
                                       'value_range': '[0, 1]',
                                       'definition': 'Best-axis closure or z-slice descriptor computed '
                                                     'from projected beta-structure coordinates.',
                                       'formula': 'Deterministic formula implemented in '
                                                  'betlas.features.extract, betlas.slicing, or the '
                                                  'matching readout module.',
                                       'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                        'geometry; NaN when beta_strand_count < 4',
                                       'source': 'betlas.features.extract'},
 'betlas_axis_best_largest_gap_fraction': {'family': 'axis_closure',
                                           'dtype': 'numeric',
                                           'units': 'dimensionless unless the definition names Angstrom '
                                                    'distances',
                                           'value_range': '[0, 1]',
                                           'definition': 'Best-axis closure or z-slice descriptor '
                                                         'computed from projected beta-structure '
                                                         'coordinates.',
                                           'formula': 'Deterministic formula implemented in '
                                                      'betlas.features.extract, betlas.slicing, or the '
                                                      'matching readout module.',
                                           'missing_value': 'Zero for empty counts; NaN for '
                                                            'inapplicable geometry; NaN when '
                                                            'beta_strand_count < 4',
                                           'source': 'betlas.features.extract'},
 'betlas_axis_best_radius_cv': {'family': 'axis_closure',
                                'dtype': 'numeric',
                                'units': 'dimensionless unless the definition names Angstrom distances',
                                'value_range': 'real-valued',
                                'definition': 'Best-axis closure or z-slice descriptor computed from '
                                              'projected beta-structure coordinates.',
                                'formula': 'Deterministic formula implemented in '
                                           'betlas.features.extract, betlas.slicing, or the matching '
                                           'readout module.',
                                'missing_value': 'Zero for empty counts; NaN for inapplicable geometry; '
                                                 'NaN when beta_strand_count < 4',
                                'source': 'betlas.features.extract'},
 'betlas_axis_best_z_range': {'family': 'axis_closure',
                              'dtype': 'numeric',
                              'units': 'dimensionless unless the definition names Angstrom distances',
                              'value_range': 'real-valued',
                              'definition': 'Best-axis closure or z-slice descriptor computed from '
                                            'projected beta-structure coordinates.',
                              'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                         'betlas.slicing, or the matching readout module.',
                              'missing_value': 'Zero for empty counts; NaN for inapplicable geometry; '
                                               'NaN when beta_strand_count < 4',
                              'source': 'betlas.features.extract'},
 'betlas_axis_best_z_continuity_fraction': {'family': 'axis_closure',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': '[0, 1]',
                                            'definition': 'Best-axis closure or z-slice descriptor '
                                                          'computed from projected beta-structure '
                                                          'coordinates.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'Zero for empty counts; NaN for '
                                                             'inapplicable geometry; NaN when '
                                                             'beta_strand_count < 4',
                                            'source': 'betlas.features.extract'},
 'betlas_axis_best_slice_count': {'family': 'axis_closure',
                                  'dtype': 'numeric',
                                  'units': 'dimensionless unless the definition names Angstrom '
                                           'distances',
                                  'value_range': 'non-negative',
                                  'definition': 'Best-axis closure or z-slice descriptor computed from '
                                                'projected beta-structure coordinates.',
                                  'formula': 'Deterministic formula implemented in '
                                             'betlas.features.extract, betlas.slicing, or the matching '
                                             'readout module.',
                                  'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                   'geometry; NaN when beta_strand_count < 4',
                                  'source': 'betlas.features.extract'},
 'betlas_axis_best_slice_coverage_mean': {'family': 'axis_closure',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': 'real-valued',
                                          'definition': 'Best-axis closure or z-slice descriptor '
                                                        'computed from projected beta-structure '
                                                        'coordinates.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry; NaN when beta_strand_count < 4',
                                          'source': 'betlas.features.extract'},
 'betlas_axis_best_slice_coverage_median': {'family': 'axis_closure',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': 'real-valued',
                                            'definition': 'Best-axis closure or z-slice descriptor '
                                                          'computed from projected beta-structure '
                                                          'coordinates.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'Zero for empty counts; NaN for '
                                                             'inapplicable geometry; NaN when '
                                                             'beta_strand_count < 4',
                                            'source': 'betlas.features.extract'},
 'betlas_axis_best_slice_coverage_std': {'family': 'axis_closure',
                                         'dtype': 'numeric',
                                         'units': 'dimensionless unless the definition names Angstrom '
                                                  'distances',
                                         'value_range': 'real-valued',
                                         'definition': 'Best-axis closure or z-slice descriptor '
                                                       'computed from projected beta-structure '
                                                       'coordinates.',
                                         'formula': 'Deterministic formula implemented in '
                                                    'betlas.features.extract, betlas.slicing, or the '
                                                    'matching readout module.',
                                         'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                          'geometry; NaN when beta_strand_count < 4',
                                         'source': 'betlas.features.extract'},
 'betlas_axis_best_slice_high_coverage_fraction': {'family': 'axis_closure',
                                                   'dtype': 'numeric',
                                                   'units': 'dimensionless unless the definition names '
                                                            'Angstrom distances',
                                                   'value_range': '[0, 1]',
                                                   'definition': 'Best-axis closure or z-slice '
                                                                 'descriptor computed from projected '
                                                                 'beta-structure coordinates.',
                                                   'formula': 'Deterministic formula implemented in '
                                                              'betlas.features.extract, betlas.slicing, '
                                                              'or the matching readout module.',
                                                   'missing_value': 'Zero for empty counts; NaN for '
                                                                    'inapplicable geometry; NaN when '
                                                                    'beta_strand_count < 4',
                                                   'source': 'betlas.features.extract'},
 'betlas_axis_best_slice_largest_gap_fraction_mean': {'family': 'axis_closure',
                                                      'dtype': 'numeric',
                                                      'units': 'dimensionless unless the definition '
                                                               'names Angstrom distances',
                                                      'value_range': 'real-valued',
                                                      'definition': 'Best-axis closure or z-slice '
                                                                    'descriptor computed from projected '
                                                                    'beta-structure coordinates.',
                                                      'formula': 'Deterministic formula implemented in '
                                                                 'betlas.features.extract, '
                                                                 'betlas.slicing, or the matching '
                                                                 'readout module.',
                                                      'missing_value': 'Zero for empty counts; NaN for '
                                                                       'inapplicable geometry; NaN when '
                                                                       'beta_strand_count < 4',
                                                      'source': 'betlas.features.extract'},
 'betlas_strand_length_mean': {'family': 'composition',
                               'dtype': 'numeric',
                               'units': 'dimensionless unless the definition names Angstrom distances',
                               'value_range': 'real-valued',
                               'definition': 'Betlas composition feature emitted by the deterministic '
                                             'geometry extraction pipeline.',
                               'formula': 'Deterministic formula implemented in '
                                          'betlas.features.extract, betlas.slicing, or the matching '
                                          'readout module.',
                               'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                               'source': 'betlas.features.extract'},
 'betlas_strand_length_std': {'family': 'composition',
                              'dtype': 'numeric',
                              'units': 'dimensionless unless the definition names Angstrom distances',
                              'value_range': 'real-valued',
                              'definition': 'Betlas composition feature emitted by the deterministic '
                                            'geometry extraction pipeline.',
                              'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                         'betlas.slicing, or the matching readout module.',
                              'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                              'source': 'betlas.features.extract'},
 'betlas_strand_length_min': {'family': 'composition',
                              'dtype': 'numeric',
                              'units': 'dimensionless unless the definition names Angstrom distances',
                              'value_range': 'real-valued',
                              'definition': 'Betlas composition feature emitted by the deterministic '
                                            'geometry extraction pipeline.',
                              'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                         'betlas.slicing, or the matching readout module.',
                              'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                              'source': 'betlas.features.extract'},
 'betlas_strand_length_median': {'family': 'composition',
                                 'dtype': 'numeric',
                                 'units': 'dimensionless unless the definition names Angstrom distances',
                                 'value_range': 'real-valued',
                                 'definition': 'Betlas composition feature emitted by the deterministic '
                                               'geometry extraction pipeline.',
                                 'formula': 'Deterministic formula implemented in '
                                            'betlas.features.extract, betlas.slicing, or the matching '
                                            'readout module.',
                                 'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                 'source': 'betlas.features.extract'},
 'betlas_strand_length_max': {'family': 'composition',
                              'dtype': 'numeric',
                              'units': 'dimensionless unless the definition names Angstrom distances',
                              'value_range': 'real-valued',
                              'definition': 'Betlas composition feature emitted by the deterministic '
                                            'geometry extraction pipeline.',
                              'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                         'betlas.slicing, or the matching readout module.',
                              'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                              'source': 'betlas.features.extract'},
 'betlas_helix_length_mean': {'family': 'composition',
                              'dtype': 'numeric',
                              'units': 'dimensionless unless the definition names Angstrom distances',
                              'value_range': 'real-valued',
                              'definition': 'Betlas composition feature emitted by the deterministic '
                                            'geometry extraction pipeline.',
                              'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                         'betlas.slicing, or the matching readout module.',
                              'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                              'source': 'betlas.features.extract'},
 'betlas_helix_length_std': {'family': 'composition',
                             'dtype': 'numeric',
                             'units': 'dimensionless unless the definition names Angstrom distances',
                             'value_range': 'real-valued',
                             'definition': 'Betlas composition feature emitted by the deterministic '
                                           'geometry extraction pipeline.',
                             'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                        'betlas.slicing, or the matching readout module.',
                             'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                             'source': 'betlas.features.extract'},
 'betlas_helix_length_min': {'family': 'composition',
                             'dtype': 'numeric',
                             'units': 'dimensionless unless the definition names Angstrom distances',
                             'value_range': 'real-valued',
                             'definition': 'Betlas composition feature emitted by the deterministic '
                                           'geometry extraction pipeline.',
                             'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                        'betlas.slicing, or the matching readout module.',
                             'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                             'source': 'betlas.features.extract'},
 'betlas_helix_length_median': {'family': 'composition',
                                'dtype': 'numeric',
                                'units': 'dimensionless unless the definition names Angstrom distances',
                                'value_range': 'real-valued',
                                'definition': 'Betlas composition feature emitted by the deterministic '
                                              'geometry extraction pipeline.',
                                'formula': 'Deterministic formula implemented in '
                                           'betlas.features.extract, betlas.slicing, or the matching '
                                           'readout module.',
                                'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                'source': 'betlas.features.extract'},
 'betlas_helix_length_max': {'family': 'composition',
                             'dtype': 'numeric',
                             'units': 'dimensionless unless the definition names Angstrom distances',
                             'value_range': 'real-valued',
                             'definition': 'Betlas composition feature emitted by the deterministic '
                                           'geometry extraction pipeline.',
                             'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                        'betlas.slicing, or the matching readout module.',
                             'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                             'source': 'betlas.features.extract'},
 'betlas_beta_alpha_alternation_fraction': {'family': 'alpha_shell',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': '[0, 1]',
                                            'definition': 'Betlas alpha_shell feature emitted by the '
                                                          'deterministic geometry extraction pipeline.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'Zero for empty counts; NaN for '
                                                             'inapplicable geometry',
                                            'source': 'betlas.features.extract'},
 'betlas_beta_alpha_repeat_pairs': {'family': 'alpha_shell',
                                    'dtype': 'numeric',
                                    'units': 'dimensionless unless the definition names Angstrom '
                                             'distances',
                                    'value_range': 'real-valued',
                                    'definition': 'Betlas alpha_shell feature emitted by the '
                                                  'deterministic geometry extraction pipeline.',
                                    'formula': 'Deterministic formula implemented in '
                                               'betlas.features.extract, betlas.slicing, or the '
                                               'matching readout module.',
                                    'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                     'geometry',
                                    'source': 'betlas.features.extract'},
 'betlas_sse_count': {'family': 'composition',
                      'dtype': 'numeric',
                      'units': 'dimensionless unless the definition names Angstrom distances',
                      'value_range': 'non-negative',
                      'definition': 'Betlas composition feature emitted by the deterministic geometry '
                                    'extraction pipeline.',
                      'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                 'betlas.slicing, or the matching readout module.',
                      'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                      'source': 'betlas.features.extract'},
 'betlas_sheet_count': {'family': 'sheet_global',
                        'dtype': 'numeric',
                        'units': 'dimensionless unless the definition names Angstrom distances',
                        'value_range': 'non-negative',
                        'definition': 'Betlas sheet_geometry feature emitted by the deterministic '
                                      'geometry extraction pipeline.',
                        'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                   'betlas.slicing, or the matching readout module.',
                        'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                        'source': 'betlas.features.extract'},
 'betlas_sheet_face_count': {'family': 'sheet_global',
                             'dtype': 'numeric',
                             'units': 'dimensionless unless the definition names Angstrom distances',
                             'value_range': 'non-negative',
                             'definition': 'Betlas sheet_geometry feature emitted by the deterministic '
                                           'geometry extraction pipeline.',
                             'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                        'betlas.slicing, or the matching readout module.',
                             'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                             'source': 'betlas.features.extract'},
 'betlas_largest_sheet_fraction': {'family': 'sheet_global',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': '[0, 1]',
                                   'definition': 'Betlas sheet_geometry feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_sheet_size_entropy': {'family': 'sheet_global',
                               'dtype': 'numeric',
                               'units': 'dimensionless unless the definition names Angstrom distances',
                               'value_range': 'real-valued',
                               'definition': 'Betlas sheet_geometry feature emitted by the '
                                             'deterministic geometry extraction pipeline.',
                               'formula': 'Deterministic formula implemented in '
                                          'betlas.features.extract, betlas.slicing, or the matching '
                                          'readout module.',
                               'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                               'source': 'betlas.features.extract'},
 'betlas_sheet_strand_count_mean': {'family': 'sheet_global',
                                    'dtype': 'numeric',
                                    'units': 'dimensionless unless the definition names Angstrom '
                                             'distances',
                                    'value_range': 'non-negative',
                                    'definition': 'Betlas sheet_geometry feature emitted by the '
                                                  'deterministic geometry extraction pipeline.',
                                    'formula': 'Deterministic formula implemented in '
                                               'betlas.features.extract, betlas.slicing, or the '
                                               'matching readout module.',
                                    'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                     'geometry',
                                    'source': 'betlas.features.extract'},
 'betlas_sheet_strand_count_std': {'family': 'sheet_global',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'non-negative',
                                   'definition': 'Betlas sheet_geometry feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_sheet_strand_count_min': {'family': 'sheet_global',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'non-negative',
                                   'definition': 'Betlas sheet_geometry feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_sheet_strand_count_median': {'family': 'sheet_global',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': 'non-negative',
                                      'definition': 'Betlas sheet_geometry feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry',
                                      'source': 'betlas.features.extract'},
 'betlas_sheet_strand_count_max': {'family': 'sheet_global',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'non-negative',
                                   'definition': 'Betlas sheet_geometry feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_sheet_angular_span_mean': {'family': 'sheet_global',
                                    'dtype': 'numeric',
                                    'units': 'dimensionless unless the definition names Angstrom '
                                             'distances',
                                    'value_range': 'real-valued',
                                    'definition': 'Betlas sheet_geometry feature emitted by the '
                                                  'deterministic geometry extraction pipeline.',
                                    'formula': 'Deterministic formula implemented in '
                                               'betlas.features.extract, betlas.slicing, or the '
                                               'matching readout module.',
                                    'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                     'geometry; NaN when beta_strand_count < 4',
                                    'source': 'betlas.features.extract'},
 'betlas_sheet_angular_span_std': {'family': 'sheet_global',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'real-valued',
                                   'definition': 'Betlas sheet_geometry feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry; NaN when beta_strand_count < 4',
                                   'source': 'betlas.features.extract'},
 'betlas_sheet_angular_span_min': {'family': 'sheet_global',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'real-valued',
                                   'definition': 'Betlas sheet_geometry feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry; NaN when beta_strand_count < 4',
                                   'source': 'betlas.features.extract'},
 'betlas_sheet_angular_span_median': {'family': 'sheet_global',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': 'real-valued',
                                      'definition': 'Betlas sheet_geometry feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry; NaN when beta_strand_count < 4',
                                      'source': 'betlas.features.extract'},
 'betlas_sheet_angular_span_max': {'family': 'sheet_global',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'real-valued',
                                   'definition': 'Betlas sheet_geometry feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry; NaN when beta_strand_count < 4',
                                   'source': 'betlas.features.extract'},
 'betlas_sheet_planarity_residual_mean': {'family': 'sheet_global',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': 'real-valued',
                                          'definition': 'Betlas sheet_geometry feature emitted by the '
                                                        'deterministic geometry extraction pipeline.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry',
                                          'source': 'betlas.features.extract'},
 'betlas_sheet_planarity_residual_std': {'family': 'sheet_global',
                                         'dtype': 'numeric',
                                         'units': 'dimensionless unless the definition names Angstrom '
                                                  'distances',
                                         'value_range': 'real-valued',
                                         'definition': 'Betlas sheet_geometry feature emitted by the '
                                                       'deterministic geometry extraction pipeline.',
                                         'formula': 'Deterministic formula implemented in '
                                                    'betlas.features.extract, betlas.slicing, or the '
                                                    'matching readout module.',
                                         'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                          'geometry',
                                         'source': 'betlas.features.extract'},
 'betlas_sheet_planarity_residual_min': {'family': 'sheet_global',
                                         'dtype': 'numeric',
                                         'units': 'dimensionless unless the definition names Angstrom '
                                                  'distances',
                                         'value_range': 'real-valued',
                                         'definition': 'Betlas sheet_geometry feature emitted by the '
                                                       'deterministic geometry extraction pipeline.',
                                         'formula': 'Deterministic formula implemented in '
                                                    'betlas.features.extract, betlas.slicing, or the '
                                                    'matching readout module.',
                                         'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                          'geometry',
                                         'source': 'betlas.features.extract'},
 'betlas_sheet_planarity_residual_median': {'family': 'sheet_global',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': 'real-valued',
                                            'definition': 'Betlas sheet_geometry feature emitted by the '
                                                          'deterministic geometry extraction pipeline.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'Zero for empty counts; NaN for '
                                                             'inapplicable geometry',
                                            'source': 'betlas.features.extract'},
 'betlas_sheet_planarity_residual_max': {'family': 'sheet_global',
                                         'dtype': 'numeric',
                                         'units': 'dimensionless unless the definition names Angstrom '
                                                  'distances',
                                         'value_range': 'real-valued',
                                         'definition': 'Betlas sheet_geometry feature emitted by the '
                                                       'deterministic geometry extraction pipeline.',
                                         'formula': 'Deterministic formula implemented in '
                                                    'betlas.features.extract, betlas.slicing, or the '
                                                    'matching readout module.',
                                         'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                          'geometry',
                                         'source': 'betlas.features.extract'},
 'betlas_sheet_axis_dispersion_mean': {'family': 'sheet_global',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless unless the definition names Angstrom '
                                                'distances',
                                       'value_range': 'real-valued',
                                       'definition': 'Betlas sheet_geometry feature emitted by the '
                                                     'deterministic geometry extraction pipeline.',
                                       'formula': 'Deterministic formula implemented in '
                                                  'betlas.features.extract, betlas.slicing, or the '
                                                  'matching readout module.',
                                       'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                        'geometry',
                                       'source': 'betlas.features.extract'},
 'betlas_sheet_axis_dispersion_std': {'family': 'sheet_global',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': 'real-valued',
                                      'definition': 'Betlas sheet_geometry feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry',
                                      'source': 'betlas.features.extract'},
 'betlas_sheet_axis_dispersion_min': {'family': 'sheet_global',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': 'real-valued',
                                      'definition': 'Betlas sheet_geometry feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry',
                                      'source': 'betlas.features.extract'},
 'betlas_sheet_axis_dispersion_median': {'family': 'sheet_global',
                                         'dtype': 'numeric',
                                         'units': 'dimensionless unless the definition names Angstrom '
                                                  'distances',
                                         'value_range': 'real-valued',
                                         'definition': 'Betlas sheet_geometry feature emitted by the '
                                                       'deterministic geometry extraction pipeline.',
                                         'formula': 'Deterministic formula implemented in '
                                                    'betlas.features.extract, betlas.slicing, or the '
                                                    'matching readout module.',
                                         'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                          'geometry',
                                         'source': 'betlas.features.extract'},
 'betlas_sheet_axis_dispersion_max': {'family': 'sheet_global',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': 'real-valued',
                                      'definition': 'Betlas sheet_geometry feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry',
                                      'source': 'betlas.features.extract'},
 'betlas_beta_run_count': {'family': 'beta_run_topology',
                           'dtype': 'numeric',
                           'units': 'dimensionless unless the definition names Angstrom distances',
                           'value_range': 'non-negative',
                           'definition': 'Betlas beta_run_topology feature emitted by the deterministic '
                                         'geometry extraction pipeline.',
                           'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                      'betlas.slicing, or the matching readout module.',
                           'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                           'source': 'betlas.features.extract'},
 'betlas_beta_segment_to_run_ratio': {'family': 'beta_run_topology',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': 'real-valued',
                                      'definition': 'Betlas beta_run_topology feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry',
                                      'source': 'betlas.features.extract'},
 'betlas_beta_run_eight_score': {'family': 'beta_run_topology',
                                 'dtype': 'numeric',
                                 'units': 'dimensionless unless the definition names Angstrom distances',
                                 'value_range': 'real-valued',
                                 'definition': 'Betlas beta_run_topology feature emitted by the '
                                               'deterministic geometry extraction pipeline.',
                                 'formula': 'Deterministic formula implemented in '
                                            'betlas.features.extract, betlas.slicing, or the matching '
                                            'readout module.',
                                 'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                 'source': 'betlas.features.extract'},
 'betlas_beta_run_sheet_multiplicity_mean': {'family': 'beta_run_topology',
                                             'dtype': 'numeric',
                                             'units': 'dimensionless unless the definition names '
                                                      'Angstrom distances',
                                             'value_range': 'real-valued',
                                             'definition': 'Betlas beta_run_topology feature emitted by '
                                                           'the deterministic geometry extraction '
                                                           'pipeline.',
                                             'formula': 'Deterministic formula implemented in '
                                                        'betlas.features.extract, betlas.slicing, or '
                                                        'the matching readout module.',
                                             'missing_value': 'Zero for empty counts; NaN for '
                                                              'inapplicable geometry',
                                             'source': 'betlas.features.extract'},
 'betlas_beta_run_sheet_multiplicity_max': {'family': 'beta_run_topology',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': 'real-valued',
                                            'definition': 'Betlas beta_run_topology feature emitted by '
                                                          'the deterministic geometry extraction '
                                                          'pipeline.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'Zero for empty counts; NaN for '
                                                             'inapplicable geometry',
                                            'source': 'betlas.features.extract'},
 'betlas_beta_run_multi_sheet_fraction': {'family': 'beta_run_topology',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': '[0, 1]',
                                          'definition': 'Betlas beta_run_topology feature emitted by '
                                                        'the deterministic geometry extraction '
                                                        'pipeline.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry',
                                          'source': 'betlas.features.extract'},
 'betlas_beta_run_angular_coverage': {'family': 'beta_run_topology',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': '[0, 1]',
                                      'definition': 'Betlas beta_run_topology feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry; NaN when beta_strand_count < 4',
                                      'source': 'betlas.features.extract'},
 'betlas_beta_run_largest_gap_fraction': {'family': 'beta_run_topology',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': '[0, 1]',
                                          'definition': 'Betlas beta_run_topology feature emitted by '
                                                        'the deterministic geometry extraction '
                                                        'pipeline.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry; NaN when beta_strand_count < 4',
                                          'source': 'betlas.features.extract'},
 'betlas_beta_run_length_mean': {'family': 'beta_run_topology',
                                 'dtype': 'numeric',
                                 'units': 'dimensionless unless the definition names Angstrom distances',
                                 'value_range': 'real-valued',
                                 'definition': 'Betlas beta_run_topology feature emitted by the '
                                               'deterministic geometry extraction pipeline.',
                                 'formula': 'Deterministic formula implemented in '
                                            'betlas.features.extract, betlas.slicing, or the matching '
                                            'readout module.',
                                 'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                 'source': 'betlas.features.extract'},
 'betlas_beta_run_length_std': {'family': 'beta_run_topology',
                                'dtype': 'numeric',
                                'units': 'dimensionless unless the definition names Angstrom distances',
                                'value_range': 'real-valued',
                                'definition': 'Betlas beta_run_topology feature emitted by the '
                                              'deterministic geometry extraction pipeline.',
                                'formula': 'Deterministic formula implemented in '
                                           'betlas.features.extract, betlas.slicing, or the matching '
                                           'readout module.',
                                'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                'source': 'betlas.features.extract'},
 'betlas_beta_run_length_min': {'family': 'beta_run_topology',
                                'dtype': 'numeric',
                                'units': 'dimensionless unless the definition names Angstrom distances',
                                'value_range': 'real-valued',
                                'definition': 'Betlas beta_run_topology feature emitted by the '
                                              'deterministic geometry extraction pipeline.',
                                'formula': 'Deterministic formula implemented in '
                                           'betlas.features.extract, betlas.slicing, or the matching '
                                           'readout module.',
                                'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                'source': 'betlas.features.extract'},
 'betlas_beta_run_length_median': {'family': 'beta_run_topology',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'real-valued',
                                   'definition': 'Betlas beta_run_topology feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_beta_run_length_max': {'family': 'beta_run_topology',
                                'dtype': 'numeric',
                                'units': 'dimensionless unless the definition names Angstrom distances',
                                'value_range': 'real-valued',
                                'definition': 'Betlas beta_run_topology feature emitted by the '
                                              'deterministic geometry extraction pipeline.',
                                'formula': 'Deterministic formula implemented in '
                                           'betlas.features.extract, betlas.slicing, or the matching '
                                           'readout module.',
                                'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                'source': 'betlas.features.extract'},
 'betlas_sheet_pair_top2_fraction': {'family': 'sheet_pair_packing',
                                     'dtype': 'numeric',
                                     'units': 'dimensionless unless the definition names Angstrom '
                                              'distances',
                                     'value_range': '[0, 1]',
                                     'definition': 'Betlas sheet_pair_packing feature emitted by the '
                                                   'deterministic geometry extraction pipeline.',
                                     'formula': 'Deterministic formula implemented in '
                                                'betlas.features.extract, betlas.slicing, or the '
                                                'matching readout module.',
                                     'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                      'geometry',
                                     'source': 'betlas.features.extract'},
 'betlas_sheet_pair_size_balance': {'family': 'sheet_pair_packing',
                                    'dtype': 'numeric',
                                    'units': 'dimensionless unless the definition names Angstrom '
                                             'distances',
                                    'value_range': 'real-valued',
                                    'definition': 'Betlas sheet_pair_packing feature emitted by the '
                                                  'deterministic geometry extraction pipeline.',
                                    'formula': 'Deterministic formula implemented in '
                                               'betlas.features.extract, betlas.slicing, or the '
                                               'matching readout module.',
                                    'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                     'geometry',
                                    'source': 'betlas.features.extract'},
 'betlas_sheet_pair_centroid_distance': {'family': 'sheet_pair_packing',
                                         'dtype': 'numeric',
                                         'units': 'dimensionless unless the definition names Angstrom '
                                                  'distances',
                                         'value_range': 'real-valued',
                                         'definition': 'Betlas sheet_pair_packing feature emitted by '
                                                       'the deterministic geometry extraction pipeline.',
                                         'formula': 'Deterministic formula implemented in '
                                                    'betlas.features.extract, betlas.slicing, or the '
                                                    'matching readout module.',
                                         'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                          'geometry',
                                         'source': 'betlas.features.extract'},
 'betlas_sheet_pair_centroid_distance_norm': {'family': 'sheet_pair_packing',
                                              'dtype': 'numeric',
                                              'units': 'dimensionless unless the definition names '
                                                       'Angstrom distances',
                                              'value_range': 'real-valued',
                                              'definition': 'Betlas sheet_pair_packing feature emitted '
                                                            'by the deterministic geometry extraction '
                                                            'pipeline.',
                                              'formula': 'Deterministic formula implemented in '
                                                         'betlas.features.extract, betlas.slicing, or '
                                                         'the matching readout module.',
                                              'missing_value': 'Zero for empty counts; NaN for '
                                                               'inapplicable geometry; NaN when '
                                                               'beta_strand_count < 4',
                                              'source': 'betlas.features.extract'},
 'betlas_sheet_pair_angular_separation_fraction': {'family': 'sheet_pair_packing',
                                                   'dtype': 'numeric',
                                                   'units': 'dimensionless unless the definition names '
                                                            'Angstrom distances',
                                                   'value_range': '[0, 1]',
                                                   'definition': 'Betlas sheet_pair_packing feature '
                                                                 'emitted by the deterministic geometry '
                                                                 'extraction pipeline.',
                                                   'formula': 'Deterministic formula implemented in '
                                                              'betlas.features.extract, betlas.slicing, '
                                                              'or the matching readout module.',
                                                   'missing_value': 'Zero for empty counts; NaN for '
                                                                    'inapplicable geometry; NaN when '
                                                                    'beta_strand_count < 4',
                                                   'source': 'betlas.features.extract'},
 'betlas_sheet_pair_cross_min_distance': {'family': 'sheet_pair_packing',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': 'real-valued',
                                          'definition': 'Betlas sheet_pair_packing feature emitted by '
                                                        'the deterministic geometry extraction '
                                                        'pipeline.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry',
                                          'source': 'betlas.features.extract'},
 'betlas_sheet_pair_cross_distance_mean': {'family': 'sheet_pair_packing',
                                           'dtype': 'numeric',
                                           'units': 'dimensionless unless the definition names Angstrom '
                                                    'distances',
                                           'value_range': 'real-valued',
                                           'definition': 'Betlas sheet_pair_packing feature emitted by '
                                                         'the deterministic geometry extraction '
                                                         'pipeline.',
                                           'formula': 'Deterministic formula implemented in '
                                                      'betlas.features.extract, betlas.slicing, or the '
                                                      'matching readout module.',
                                           'missing_value': 'Zero for empty counts; NaN for '
                                                            'inapplicable geometry',
                                           'source': 'betlas.features.extract'},
 'betlas_sheet_pair_cross_contact_density8': {'family': 'sheet_pair_packing',
                                              'dtype': 'numeric',
                                              'units': 'dimensionless unless the definition names '
                                                       'Angstrom distances',
                                              'value_range': 'real-valued',
                                              'definition': 'Betlas sheet_pair_packing feature emitted '
                                                            'by the deterministic geometry extraction '
                                                            'pipeline.',
                                              'formula': 'Deterministic formula implemented in '
                                                         'betlas.features.extract, betlas.slicing, or '
                                                         'the matching readout module.',
                                              'missing_value': 'Zero for empty counts; NaN for '
                                                               'inapplicable geometry',
                                              'source': 'betlas.features.extract'},
 'betlas_sheet_pair_cross_contact_density10': {'family': 'sheet_pair_packing',
                                               'dtype': 'numeric',
                                               'units': 'dimensionless unless the definition names '
                                                        'Angstrom distances',
                                               'value_range': 'real-valued',
                                               'definition': 'Betlas sheet_pair_packing feature emitted '
                                                             'by the deterministic geometry extraction '
                                                             'pipeline.',
                                               'formula': 'Deterministic formula implemented in '
                                                          'betlas.features.extract, betlas.slicing, or '
                                                          'the matching readout module.',
                                               'missing_value': 'Zero for empty counts; NaN for '
                                                                'inapplicable geometry',
                                               'source': 'betlas.features.extract'},
 'betlas_sheet_pair_reciprocal_nearest_fraction8': {'family': 'sheet_pair_packing',
                                                    'dtype': 'numeric',
                                                    'units': 'dimensionless unless the definition names '
                                                             'Angstrom distances',
                                                    'value_range': 'real-valued',
                                                    'definition': 'Betlas sheet_pair_packing feature '
                                                                  'emitted by the deterministic '
                                                                  'geometry extraction pipeline.',
                                                    'formula': 'Deterministic formula implemented in '
                                                               'betlas.features.extract, '
                                                               'betlas.slicing, or the matching readout '
                                                               'module.',
                                                    'missing_value': 'Zero for empty counts; NaN for '
                                                                     'inapplicable geometry',
                                                    'source': 'betlas.features.extract'},
 'betlas_sheet_seq_top2_run_fraction': {'family': 'sheet_sequence_topology',
                                        'dtype': 'numeric',
                                        'units': 'dimensionless unless the definition names Angstrom '
                                                 'distances',
                                        'value_range': '[0, 1]',
                                        'definition': 'Betlas sheet_sequence_topology feature emitted '
                                                      'by the deterministic geometry extraction '
                                                      'pipeline.',
                                        'formula': 'Deterministic formula implemented in '
                                                   'betlas.features.extract, betlas.slicing, or the '
                                                   'matching readout module.',
                                        'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                         'geometry',
                                        'source': 'betlas.features.extract'},
 'betlas_sheet_seq_top2_transition_fraction': {'family': 'sheet_sequence_topology',
                                               'dtype': 'numeric',
                                               'units': 'dimensionless unless the definition names '
                                                        'Angstrom distances',
                                               'value_range': '[0, 1]',
                                               'definition': 'Betlas sheet_sequence_topology feature '
                                                             'emitted by the deterministic geometry '
                                                             'extraction pipeline.',
                                               'formula': 'Deterministic formula implemented in '
                                                          'betlas.features.extract, betlas.slicing, or '
                                                          'the matching readout module.',
                                               'missing_value': 'Zero for empty counts; NaN for '
                                                                'inapplicable geometry',
                                               'source': 'betlas.features.extract'},
 'betlas_sheet_seq_top2_block_count': {'family': 'sheet_sequence_topology',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless unless the definition names Angstrom '
                                                'distances',
                                       'value_range': 'non-negative',
                                       'definition': 'Betlas sheet_sequence_topology feature emitted by '
                                                     'the deterministic geometry extraction pipeline.',
                                       'formula': 'Deterministic formula implemented in '
                                                  'betlas.features.extract, betlas.slicing, or the '
                                                  'matching readout module.',
                                       'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                        'geometry',
                                       'source': 'betlas.features.extract'},
 'betlas_sheet_seq_top2_longest_block_fraction': {'family': 'sheet_sequence_topology',
                                                  'dtype': 'numeric',
                                                  'units': 'dimensionless unless the definition names '
                                                           'Angstrom distances',
                                                  'value_range': '[0, 1]',
                                                  'definition': 'Betlas sheet_sequence_topology feature '
                                                                'emitted by the deterministic geometry '
                                                                'extraction pipeline.',
                                                  'formula': 'Deterministic formula implemented in '
                                                             'betlas.features.extract, betlas.slicing, '
                                                             'or the matching readout module.',
                                                  'missing_value': 'Zero for empty counts; NaN for '
                                                                   'inapplicable geometry',
                                                  'source': 'betlas.features.extract'},
 'betlas_sheet_seq_top2_interleave_score': {'family': 'sheet_sequence_topology',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': 'real-valued',
                                            'definition': 'Betlas sheet_sequence_topology feature '
                                                          'emitted by the deterministic geometry '
                                                          'extraction pipeline.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'Zero for empty counts; NaN for '
                                                             'inapplicable geometry',
                                            'source': 'betlas.features.extract'},
 'betlas_sheet_seq_top2_order_displacement': {'family': 'sheet_sequence_topology',
                                              'dtype': 'numeric',
                                              'units': 'dimensionless unless the definition names '
                                                       'Angstrom distances',
                                              'value_range': 'real-valued',
                                              'definition': 'Betlas sheet_sequence_topology feature '
                                                            'emitted by the deterministic geometry '
                                                            'extraction pipeline.',
                                              'formula': 'Deterministic formula implemented in '
                                                         'betlas.features.extract, betlas.slicing, or '
                                                         'the matching readout module.',
                                              'missing_value': 'Zero for empty counts; NaN for '
                                                               'inapplicable geometry',
                                              'source': 'betlas.features.extract'},
 'betlas_sheet_seq_greek_key_proxy': {'family': 'sheet_sequence_topology',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': 'real-valued',
                                      'definition': 'Betlas sheet_sequence_topology feature emitted by '
                                                    'the deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry',
                                      'source': 'betlas.features.extract'},
 'betlas_sheet_order_adjacent_seq_step_mean': {'family': 'sheet_sequence_topology',
                                               'dtype': 'numeric',
                                               'units': 'dimensionless unless the definition names '
                                                        'Angstrom distances',
                                               'value_range': 'real-valued',
                                               'definition': 'Betlas sheet_order_topology feature '
                                                             'emitted by the deterministic geometry '
                                                             'extraction pipeline.',
                                               'formula': 'Deterministic formula implemented in '
                                                          'betlas.features.extract, betlas.slicing, or '
                                                          'the matching readout module.',
                                               'missing_value': 'Zero for empty counts; NaN for '
                                                                'inapplicable geometry',
                                               'source': 'betlas.features.extract'},
 'betlas_sheet_order_adjacent_seq_step_max': {'family': 'sheet_sequence_topology',
                                              'dtype': 'numeric',
                                              'units': 'dimensionless unless the definition names '
                                                       'Angstrom distances',
                                              'value_range': 'real-valued',
                                              'definition': 'Betlas sheet_order_topology feature '
                                                            'emitted by the deterministic geometry '
                                                            'extraction pipeline.',
                                              'formula': 'Deterministic formula implemented in '
                                                         'betlas.features.extract, betlas.slicing, or '
                                                         'the matching readout module.',
                                              'missing_value': 'Zero for empty counts; NaN for '
                                                               'inapplicable geometry',
                                              'source': 'betlas.features.extract'},
 'betlas_sheet_order_nonlocal_fraction': {'family': 'sheet_sequence_topology',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': '[0, 1]',
                                          'definition': 'Betlas sheet_order_topology feature emitted by '
                                                        'the deterministic geometry extraction '
                                                        'pipeline.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry',
                                          'source': 'betlas.features.extract'},
 'betlas_sheet_order_inversion_fraction_mean': {'family': 'sheet_sequence_topology',
                                                'dtype': 'numeric',
                                                'units': 'dimensionless unless the definition names '
                                                         'Angstrom distances',
                                                'value_range': 'real-valued',
                                                'definition': 'Betlas sheet_order_topology feature '
                                                              'emitted by the deterministic geometry '
                                                              'extraction pipeline.',
                                                'formula': 'Deterministic formula implemented in '
                                                           'betlas.features.extract, betlas.slicing, or '
                                                           'the matching readout module.',
                                                'missing_value': 'Zero for empty counts; NaN for '
                                                                 'inapplicable geometry',
                                                'source': 'betlas.features.extract'},
 'betlas_top2_sheet_order_adjacent_seq_step_mean': {'family': 'sheet_sequence_topology',
                                                    'dtype': 'numeric',
                                                    'units': 'dimensionless unless the definition names '
                                                             'Angstrom distances',
                                                    'value_range': 'real-valued',
                                                    'definition': 'Betlas sheet_order_topology feature '
                                                                  'emitted by the deterministic '
                                                                  'geometry extraction pipeline.',
                                                    'formula': 'Deterministic formula implemented in '
                                                               'betlas.features.extract, '
                                                               'betlas.slicing, or the matching readout '
                                                               'module.',
                                                    'missing_value': 'Zero for empty counts; NaN for '
                                                                     'inapplicable geometry',
                                                    'source': 'betlas.features.extract'},
 'betlas_top2_sheet_order_nonlocal_fraction': {'family': 'sheet_sequence_topology',
                                               'dtype': 'numeric',
                                               'units': 'dimensionless unless the definition names '
                                                        'Angstrom distances',
                                               'value_range': '[0, 1]',
                                               'definition': 'Betlas sheet_order_topology feature '
                                                             'emitted by the deterministic geometry '
                                                             'extraction pipeline.',
                                               'formula': 'Deterministic formula implemented in '
                                                          'betlas.features.extract, betlas.slicing, or '
                                                          'the matching readout module.',
                                               'missing_value': 'Zero for empty counts; NaN for '
                                                                'inapplicable geometry',
                                               'source': 'betlas.features.extract'},
 'betlas_top2_sheet_order_inversion_fraction_mean': {'family': 'sheet_sequence_topology',
                                                     'dtype': 'numeric',
                                                     'units': 'dimensionless unless the definition '
                                                              'names Angstrom distances',
                                                     'value_range': 'real-valued',
                                                     'definition': 'Betlas sheet_order_topology feature '
                                                                   'emitted by the deterministic '
                                                                   'geometry extraction pipeline.',
                                                     'formula': 'Deterministic formula implemented in '
                                                                'betlas.features.extract, '
                                                                'betlas.slicing, or the matching '
                                                                'readout module.',
                                                     'missing_value': 'Zero for empty counts; NaN for '
                                                                      'inapplicable geometry',
                                                     'source': 'betlas.features.extract'},
 'betlas_jelly_roll_order_nonlocal_score': {'family': 'sheet_sequence_topology',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': 'real-valued',
                                            'definition': 'Betlas sheet_order_topology feature emitted '
                                                          'by the deterministic geometry extraction '
                                                          'pipeline.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'Zero for empty counts; NaN for '
                                                             'inapplicable geometry',
                                            'source': 'betlas.features.extract'},
 'betlas_contact8_edge_density': {'family': 'contact_graph',
                                  'dtype': 'numeric',
                                  'units': 'dimensionless unless the definition names Angstrom '
                                           'distances',
                                  'value_range': 'real-valued',
                                  'definition': 'Betlas contact_graph feature emitted by the '
                                                'deterministic geometry extraction pipeline.',
                                  'formula': 'Deterministic formula implemented in '
                                             'betlas.features.extract, betlas.slicing, or the matching '
                                             'readout module.',
                                  'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                   'geometry',
                                  'source': 'betlas.features.extract'},
 'betlas_contact10_edge_density': {'family': 'contact_graph',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'real-valued',
                                   'definition': 'Betlas contact_graph feature emitted by the '
                                                 'deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_contact8_cross_sheet_fraction': {'family': 'contact_graph',
                                          'dtype': 'numeric',
                                          'units': 'dimensionless unless the definition names Angstrom '
                                                   'distances',
                                          'value_range': '[0, 1]',
                                          'definition': 'Betlas contact_graph feature emitted by the '
                                                        'deterministic geometry extraction pipeline.',
                                          'formula': 'Deterministic formula implemented in '
                                                     'betlas.features.extract, betlas.slicing, or the '
                                                     'matching readout module.',
                                          'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                           'geometry',
                                          'source': 'betlas.features.extract'},
 'betlas_contact8_component_count': {'family': 'contact_graph',
                                     'dtype': 'numeric',
                                     'units': 'dimensionless unless the definition names Angstrom '
                                              'distances',
                                     'value_range': 'non-negative',
                                     'definition': 'Betlas contact_graph feature emitted by the '
                                                   'deterministic geometry extraction pipeline.',
                                     'formula': 'Deterministic formula implemented in '
                                                'betlas.features.extract, betlas.slicing, or the '
                                                'matching readout module.',
                                     'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                      'geometry',
                                     'source': 'betlas.features.extract'},
 'betlas_contact8_largest_component_fraction': {'family': 'contact_graph',
                                                'dtype': 'numeric',
                                                'units': 'dimensionless unless the definition names '
                                                         'Angstrom distances',
                                                'value_range': '[0, 1]',
                                                'definition': 'Betlas contact_graph feature emitted by '
                                                              'the deterministic geometry extraction '
                                                              'pipeline.',
                                                'formula': 'Deterministic formula implemented in '
                                                           'betlas.features.extract, betlas.slicing, or '
                                                           'the matching readout module.',
                                                'missing_value': 'Zero for empty counts; NaN for '
                                                                 'inapplicable geometry',
                                                'source': 'betlas.features.extract'},
 'betlas_contact8_cycle_rank_norm': {'family': 'contact_graph',
                                     'dtype': 'numeric',
                                     'units': 'dimensionless unless the definition names Angstrom '
                                              'distances',
                                     'value_range': 'real-valued',
                                     'definition': 'Betlas contact_graph feature emitted by the '
                                                   'deterministic geometry extraction pipeline.',
                                     'formula': 'Deterministic formula implemented in '
                                                'betlas.features.extract, betlas.slicing, or the '
                                                'matching readout module.',
                                     'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                      'geometry',
                                     'source': 'betlas.features.extract'},
 'betlas_contact8_degree2_fraction': {'family': 'contact_graph',
                                      'dtype': 'numeric',
                                      'units': 'dimensionless unless the definition names Angstrom '
                                               'distances',
                                      'value_range': '[0, 1]',
                                      'definition': 'Betlas contact_graph feature emitted by the '
                                                    'deterministic geometry extraction pipeline.',
                                      'formula': 'Deterministic formula implemented in '
                                                 'betlas.features.extract, betlas.slicing, or the '
                                                 'matching readout module.',
                                      'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                       'geometry',
                                      'source': 'betlas.features.extract'},
 'betlas_contact8_seq_gap_mean': {'family': 'contact_graph',
                                  'dtype': 'numeric',
                                  'units': 'dimensionless unless the definition names Angstrom '
                                           'distances',
                                  'value_range': 'real-valued',
                                  'definition': 'Betlas contact_sequence_topology feature emitted by '
                                                'the deterministic geometry extraction pipeline.',
                                  'formula': 'Deterministic formula implemented in '
                                             'betlas.features.extract, betlas.slicing, or the matching '
                                             'readout module.',
                                  'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                   'geometry',
                                  'source': 'betlas.features.extract'},
 'betlas_contact8_seq_gap_max': {'family': 'contact_graph',
                                 'dtype': 'numeric',
                                 'units': 'dimensionless unless the definition names Angstrom distances',
                                 'value_range': 'real-valued',
                                 'definition': 'Betlas contact_sequence_topology feature emitted by the '
                                               'deterministic geometry extraction pipeline.',
                                 'formula': 'Deterministic formula implemented in '
                                            'betlas.features.extract, betlas.slicing, or the matching '
                                            'readout module.',
                                 'missing_value': 'Zero for empty counts; NaN for inapplicable geometry',
                                 'source': 'betlas.features.extract'},
 'betlas_contact8_nonlocal_fraction': {'family': 'contact_graph',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless unless the definition names Angstrom '
                                                'distances',
                                       'value_range': '[0, 1]',
                                       'definition': 'Betlas contact_sequence_topology feature emitted '
                                                     'by the deterministic geometry extraction '
                                                     'pipeline.',
                                       'formula': 'Deterministic formula implemented in '
                                                  'betlas.features.extract, betlas.slicing, or the '
                                                  'matching readout module.',
                                       'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                        'geometry',
                                       'source': 'betlas.features.extract'},
 'betlas_contact8_cross_sheet_nonlocal_fraction': {'family': 'contact_graph',
                                                   'dtype': 'numeric',
                                                   'units': 'dimensionless unless the definition names '
                                                            'Angstrom distances',
                                                   'value_range': '[0, 1]',
                                                   'definition': 'Betlas contact_sequence_topology '
                                                                 'feature emitted by the deterministic '
                                                                 'geometry extraction pipeline.',
                                                   'formula': 'Deterministic formula implemented in '
                                                              'betlas.features.extract, betlas.slicing, '
                                                              'or the matching readout module.',
                                                   'missing_value': 'Zero for empty counts; NaN for '
                                                                    'inapplicable geometry',
                                                   'source': 'betlas.features.extract'},
 'betlas_contact10_seq_gap_mean': {'family': 'contact_graph',
                                   'dtype': 'numeric',
                                   'units': 'dimensionless unless the definition names Angstrom '
                                            'distances',
                                   'value_range': 'real-valued',
                                   'definition': 'Betlas contact_sequence_topology feature emitted by '
                                                 'the deterministic geometry extraction pipeline.',
                                   'formula': 'Deterministic formula implemented in '
                                              'betlas.features.extract, betlas.slicing, or the matching '
                                              'readout module.',
                                   'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                    'geometry',
                                   'source': 'betlas.features.extract'},
 'betlas_contact10_nonlocal_fraction': {'family': 'contact_graph',
                                        'dtype': 'numeric',
                                        'units': 'dimensionless unless the definition names Angstrom '
                                                 'distances',
                                        'value_range': '[0, 1]',
                                        'definition': 'Betlas contact_sequence_topology feature emitted '
                                                      'by the deterministic geometry extraction '
                                                      'pipeline.',
                                        'formula': 'Deterministic formula implemented in '
                                                   'betlas.features.extract, betlas.slicing, or the '
                                                   'matching readout module.',
                                        'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                         'geometry',
                                        'source': 'betlas.features.extract'},
 'betlas_strand_order_displacement_mean': {'family': 'strand_order',
                                           'dtype': 'numeric',
                                           'units': 'dimensionless unless the definition names Angstrom '
                                                    'distances',
                                           'value_range': 'real-valued',
                                           'definition': 'Betlas strand_order feature emitted by the '
                                                         'deterministic geometry extraction pipeline.',
                                           'formula': 'Deterministic formula implemented in '
                                                      'betlas.features.extract, betlas.slicing, or the '
                                                      'matching readout module.',
                                           'missing_value': 'Zero for empty counts; NaN for '
                                                            'inapplicable geometry; NaN when '
                                                            'beta_strand_count < 4',
                                           'source': 'betlas.features.extract'},
 'betlas_strand_order_adjacent_seq_step_median': {'family': 'strand_order',
                                                  'dtype': 'numeric',
                                                  'units': 'dimensionless unless the definition names '
                                                           'Angstrom distances',
                                                  'value_range': 'real-valued',
                                                  'definition': 'Betlas strand_order feature emitted by '
                                                                'the deterministic geometry extraction '
                                                                'pipeline.',
                                                  'formula': 'Deterministic formula implemented in '
                                                             'betlas.features.extract, betlas.slicing, '
                                                             'or the matching readout module.',
                                                  'missing_value': 'Zero for empty counts; NaN for '
                                                                   'inapplicable geometry; NaN when '
                                                                   'beta_strand_count < 4',
                                                  'source': 'betlas.features.extract'},
 'betlas_strand_ntc_parallel_fraction': {'family': 'strand_order',
                                         'dtype': 'numeric',
                                         'units': 'dimensionless unless the definition names Angstrom '
                                                  'distances',
                                         'value_range': '[0, 1]',
                                         'definition': 'Betlas strand_order feature emitted by the '
                                                       'deterministic geometry extraction pipeline.',
                                         'formula': 'Deterministic formula implemented in '
                                                    'betlas.features.extract, betlas.slicing, or the '
                                                    'matching readout module.',
                                         'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                          'geometry',
                                         'source': 'betlas.features.extract'},
 'betlas_strand_ntc_antiparallel_fraction': {'family': 'strand_order',
                                             'dtype': 'numeric',
                                             'units': 'dimensionless unless the definition names '
                                                      'Angstrom distances',
                                             'value_range': '[0, 1]',
                                             'definition': 'Betlas strand_order feature emitted by the '
                                                           'deterministic geometry extraction pipeline.',
                                             'formula': 'Deterministic formula implemented in '
                                                        'betlas.features.extract, betlas.slicing, or '
                                                        'the matching readout module.',
                                             'missing_value': 'Zero for empty counts; NaN for '
                                                              'inapplicable geometry',
                                             'source': 'betlas.features.extract'},
 'betlas_strand_axis_abs_alignment_mean': {'family': 'strand_order',
                                           'dtype': 'numeric',
                                           'units': 'dimensionless unless the definition names Angstrom '
                                                    'distances',
                                           'value_range': 'real-valued',
                                           'definition': 'Betlas strand_order feature emitted by the '
                                                         'deterministic geometry extraction pipeline.',
                                           'formula': 'Deterministic formula implemented in '
                                                      'betlas.features.extract, betlas.slicing, or the '
                                                      'matching readout module.',
                                           'missing_value': 'Zero for empty counts; NaN for '
                                                            'inapplicable geometry',
                                           'source': 'betlas.features.extract'},
 'betlas_alpha_shell_radial_delta': {'family': 'alpha_shell',
                                     'dtype': 'numeric',
                                     'units': 'dimensionless unless the definition names Angstrom '
                                              'distances',
                                     'value_range': 'real-valued',
                                     'definition': 'Betlas alpha_shell feature emitted by the '
                                                   'deterministic geometry extraction pipeline.',
                                     'formula': 'Deterministic formula implemented in '
                                                'betlas.features.extract, betlas.slicing, or the '
                                                'matching readout module.',
                                     'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                      'geometry; NaN when beta_strand_count < 4',
                                     'source': 'betlas.features.extract'},
 'betlas_helix_beta_radius_ratio': {'family': 'composition',
                                    'dtype': 'numeric',
                                    'units': 'dimensionless unless the definition names Angstrom '
                                             'distances',
                                    'value_range': 'real-valued',
                                    'definition': 'Betlas composition feature emitted by the '
                                                  'deterministic geometry extraction pipeline.',
                                    'formula': 'Deterministic formula implemented in '
                                               'betlas.features.extract, betlas.slicing, or the '
                                               'matching readout module.',
                                    'missing_value': 'Zero for empty counts; NaN for inapplicable '
                                                     'geometry; NaN when beta_strand_count < 4',
                                    'source': 'betlas.features.extract'},
 'betlas_angular_gap_cv': {'family': 'axis_closure',
                           'dtype': 'numeric',
                           'units': 'dimensionless unless the definition names Angstrom distances',
                           'value_range': 'real-valued',
                           'definition': 'Betlas axis_closure feature emitted by the deterministic '
                                         'geometry extraction pipeline.',
                           'formula': 'Deterministic formula implemented in betlas.features.extract, '
                                      'betlas.slicing, or the matching readout module.',
                           'missing_value': 'Zero for empty counts; NaN for inapplicable geometry; NaN '
                                            'when beta_strand_count < 4',
                           'source': 'betlas.features.extract'},
 'betlas_angular_circular_moment_k1': {'family': 'angular_lobes',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-1 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*1*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_angular_circular_moment_k2': {'family': 'angular_lobes',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-2 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*2*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_angular_circular_moment_k3': {'family': 'axis_periodicity',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-3 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*3*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_angular_circular_moment_k4': {'family': 'axis_periodicity',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-4 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*4*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_angular_circular_moment_k5': {'family': 'axis_periodicity',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-5 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*5*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_angular_circular_moment_k6': {'family': 'axis_periodicity',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-6 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*6*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_angular_circular_moment_k7': {'family': 'axis_periodicity',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-7 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*7*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_angular_circular_moment_k8': {'family': 'axis_periodicity',
                                       'dtype': 'numeric',
                                       'units': 'dimensionless',
                                       'value_range': '[0,1]',
                                       'definition': 'Magnitude of the order-8 continuous circular '
                                                     'moment of beta-node midpoint angles around the '
                                                     'selected axis.',
                                       'formula': 'abs(mean_j(exp(i*8*theta_j)))',
                                       'missing_value': 'NaN for unavailable angular geometry or fewer '
                                                        'than four beta strands; NaN when '
                                                        'beta_strand_count < 4',
                                       'source': 'betlas.features.current.circular_moments'},
 'betlas_sheet_pair_valid_normal_abs_dot': {'family': 'sheet_pair_packing',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': 'real-valued',
                                            'definition': 'Betlas sheet_pair_packing feature emitted by '
                                                          'the deterministic geometry extraction '
                                                          'pipeline. Defined when both dominant sheet '
                                                          'planes have unique normals; zero when no '
                                                          'sheet pair exists.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'NaN if either dominant plane is '
                                                             'unavailable; zero if there is no sheet '
                                                             'pair',
                                            'source': 'betlas.features.extract'},
 'betlas_sheet_pair_valid_face_alignment': {'family': 'sheet_pair_packing',
                                            'dtype': 'numeric',
                                            'units': 'dimensionless unless the definition names '
                                                     'Angstrom distances',
                                            'value_range': 'real-valued',
                                            'definition': 'Betlas sheet_pair_packing feature emitted by '
                                                          'the deterministic geometry extraction '
                                                          'pipeline. Defined when both dominant sheet '
                                                          'planes have unique normals; zero when no '
                                                          'sheet pair exists.',
                                            'formula': 'Deterministic formula implemented in '
                                                       'betlas.features.extract, betlas.slicing, or the '
                                                       'matching readout module.',
                                            'missing_value': 'NaN if either dominant plane is '
                                                             'unavailable; zero if there is no sheet '
                                                             'pair',
                                            'source': 'betlas.features.extract'},
 'betlas_sheet_pair_valid_bilayer_score': {'family': 'sheet_pair_packing',
                                           'dtype': 'numeric',
                                           'units': 'dimensionless unless the definition names Angstrom '
                                                    'distances',
                                           'value_range': 'real-valued',
                                           'definition': 'Betlas sheet_pair_packing feature emitted by '
                                                         'the deterministic geometry extraction '
                                                         'pipeline. Defined when both dominant sheet '
                                                         'planes have unique normals; zero when no '
                                                         'sheet pair exists.',
                                           'formula': 'Deterministic formula implemented in '
                                                      'betlas.features.extract, betlas.slicing, or the '
                                                      'matching readout module.',
                                           'missing_value': 'NaN if either dominant plane is '
                                                            'unavailable; zero if there is no sheet '
                                                            'pair; NaN when beta_strand_count < 4',
                                           'source': 'betlas.features.extract'},
 'betlas_sheet_pair_valid_lobe_score': {'family': 'sheet_pair_packing',
                                        'dtype': 'numeric',
                                        'units': 'dimensionless unless the definition names Angstrom '
                                                 'distances',
                                        'value_range': 'real-valued',
                                        'definition': 'Betlas sheet_pair_packing feature emitted by the '
                                                      'deterministic geometry extraction pipeline. '
                                                      'Defined when both dominant sheet planes have '
                                                      'unique normals; zero when no sheet pair exists.',
                                        'formula': 'Deterministic formula implemented in '
                                                   'betlas.features.extract, betlas.slicing, or the '
                                                   'matching readout module.',
                                        'missing_value': 'NaN if either dominant plane is unavailable; '
                                                         'zero if there is no sheet pair; NaN when '
                                                         'beta_strand_count < 4',
                                        'source': 'betlas.features.extract'}}

_METADATA = {
    "feature_schema": "Descriptor schema identifier used by task checkpoints.",
    "moment_status": "Applicability of the circular-moment measurements.",
    "plane_status": "Availability of both dominant sheet normals.",
    "axis_applicability": "Whether the structure has at least four beta strands.",
    "elongation_applicability": "Whether at least three beta C-alpha points are present.",
    "betlas_parse_ok": "One when residues and beta segments were parsed successfully.",
    "betlas_error": "Extraction error, or an empty string after successful extraction.",
    "betlas_warnings": "Nonfatal structure parsing or measurement warnings.",
    "betlas_axis_best_name": "Name of the selected axis hypothesis.",
    "betlas_sheet_pair_top_sheet_ids_equal": "Identifiers of the two dominant sheets.",
    "betlas_secondary_structure_source": "Source of secondary-structure annotations.",
    "betlas_sheet_order_status": "Availability of a declared sheet strand ordering.",
    "betlas_sequence_order_status": "Availability of a within-chain residue ordering.",
    "betlas_ladder_pairing_status": "Availability of retained DSSP pairing ladders.",
    "betlas_domain_segment_count": "Number of selected domain segments.",
    "betlas_dssp_ladder_edge_count": "Number of retained DSSP strand-pairing edges.",
}


def list_feature_specs(columns=None) -> tuple[FeatureSpec, ...]:
    """Return the fixed descriptor dictionary, preserving model input order."""
    selected = FEATURE_COLUMNS if columns is None else tuple(columns)
    result = []
    for name in selected:
        if name in _DESCRIPTOR_DETAILS:
            result.append(FeatureSpec(name=name, **_DESCRIPTOR_DETAILS[name]))
        elif name in _METADATA:
            result.append(FeatureSpec(name, "schema", "numeric" if name in {
                "betlas_parse_ok", "betlas_domain_segment_count", "betlas_dssp_ladder_edge_count"
            } else "string", "not applicable", "status or metadata", _METADATA[name],
                "Metadata accompanying the measured descriptors.", "empty string when absent", "betlas.features.extract"))
        else:
            raise KeyError(f"Unknown current Betlas descriptor or metadata column {name!r}")
    return tuple(result)


def describe_feature(name: str) -> FeatureSpec:
    return list_feature_specs((name,))[0]


def _feature_column(spec: FeatureSpec) -> ColumnSpec:
    return ColumnSpec(spec.name, "feature_csv", "feature" if spec.name in FEATURE_COLUMNS else "metadata",
                      spec.dtype, spec.name in FEATURE_COLUMNS, spec.definition, spec.units, spec.value_range,
                      spec.missing_value, spec.source)


def list_readout_column_specs(readout: str | None = None) -> tuple[ColumnSpec, ...]:
    """Describe learned task results or independent native geometry readouts."""
    from .readouts.beta_barrel_detection.constants import DEFAULT_RESULT_COLUMNS as DETECTION
    from .readouts.beta_barrel_staves.constants import DEFAULT_RESULT_COLUMNS as STAVES
    geometry = {"beta-barrel-detection": DETECTION, "beta-barrel-staves": STAVES}
    requested = readout.replace("_", "-") if readout else None
    tasks = ("solenoid", "barrel", "strand-count")
    if requested is not None and requested not in {*geometry, *tasks}:
        raise KeyError(f"Unknown Betlas readout {readout!r}")
    task_columns = {
        "task": ("string", "Requested task name."),
        "prediction": ("integer", "Binary recognition decision or barrel-wall strand count."),
        "probability": ("numeric", "Model probability of the positive class or selected strand-count class."),
        "probabilities": ("mapping", "Model probabilities indexed by class."),
        "status": ("string", "Prediction availability or failure status."),
        "feature_schema": ("string", "Descriptor contract required by the checkpoint."),
        "checkpoint_release": ("string", "Release identifier of the loaded checkpoint."),
        "imputed_feature_count": ("integer", "Number of unavailable descriptors replaced by stored imputation values."),
        "imputed_features": ("list", "Names of unavailable descriptors replaced by stored imputation values."),
        "is_solenoid": ("boolean", "Solenoid recognition decision."),
        "is_barrel": ("boolean", "Geometric barrel recognition decision."),
        "threshold": ("numeric", "Stored binary classification threshold."),
        "strand_count": ("integer", "Predicted number of beta strands in the barrel wall."),
        "barrel_probability": ("numeric", "Barrel recognition probability used by the count gate."),
        "chain_id": ("string", "Selected author chain identifier."),
        "model_id": ("integer", "Selected zero-based coordinate model index."),
        "residue_count": ("integer", "Selected observed C-alpha residue count."),
        "beta_strand_count": ("integer", "Number of DSSP-derived beta strands."),
        "secondary_structure_source": ("string", "Source of secondary-structure annotations."),
        "dssp_version": ("string", "Version of DSSP used to annotate the input."),
        "warnings": ("list", "Nonfatal parsing or measurement warnings."),
    }
    result = []
    if requested is None or requested in tasks:
        for name, (dtype, definition) in task_columns.items():
            result.append(ColumnSpec(name, "prediction", "prediction", dtype, name in {
                "task", "prediction", "status", "feature_schema", "checkpoint_release"
            }, definition, "dimensionless", "task dependent", "null when unavailable", "betlas.prediction"))
    definitions = {
        "score_type": "Score interpretation declared by the native geometry readout.",
        "calibration_status": "Calibration status of the native geometry score.",
        "config_profile": "Configuration profile used for the native geometry readout.",
        "decision_score": "Heuristic positive-decision support from the geometry detector; not a calibrated probability.",
        "confidence": "Heuristic stave-candidate support and agreement; not a calibrated probability.",
        "strand_count": "Candidate strand count estimated from slices.",
    }
    for name, columns in geometry.items():
        if requested is not None and requested != name:
            continue
        for column in columns:
            numeric = any(token in column for token in ("count", "score", "confidence", "fraction"))
            dtype = "string" if column in {"score_type", "confidence_basis", "decision_basis"} else "numeric" if numeric else "string"
            result.append(ColumnSpec(column, f"{name}_csv", "measurement", dtype, True,
                                     definitions.get(column, f"Native geometry readout field: {column}."),
                                     "as defined by the readout", "readout dependent", "empty or NaN when unavailable",
                                     f"betlas.readouts.{name.replace('-', '_')}"))
    return tuple(result)


def describe_readout_column(name: str, readout: str | None = None) -> ColumnSpec:
    for spec in list_readout_column_specs(readout):
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown prediction column {name!r}")


def list_column_specs(table: str | None = None) -> tuple[ColumnSpec, ...]:
    specs = [_feature_column(spec) for spec in list_feature_specs((*FEATURE_COLUMNS, *_METADATA))]
    specs.extend(ColumnSpec(name, "feature_csv", "metadata", "string", name == "record_id",
                            f"Input structure identifier or annotation: {name}.", "not applicable", "string",
                            "empty string when absent", "betlas.models.DomainCandidate")
                 for name in DomainCandidate.__dataclass_fields__)
    specs.extend(list_readout_column_specs())
    return tuple(spec for spec in specs if table is None or spec.table == table)


def describe_column(name: str, table: str | None = None) -> ColumnSpec:
    for spec in list_column_specs(table):
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown Betlas column {name!r}")
