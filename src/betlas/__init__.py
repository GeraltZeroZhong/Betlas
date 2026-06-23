from __future__ import annotations

from .constants import FOLD_LABELS
from .examples import copy_example, list_examples
from .features.extract import extract_feature_row, extract_structure_features
from .grammars import (
    compute_grammar_features,
    explain_fold_grammar,
    get_grammar,
    list_grammars,
    score_fold_grammar,
)
from .models import DomainCandidate, GeometrySignature, StructureGeometry
from .slicing import (
    SliceBin,
    SliceBundle,
    SliceConfig,
    SlicePoint,
    list_slice_axes,
    slice_feature_row,
    slice_mmcif,
    slice_structure,
    summarize_slices,
)
from .specs import (
    ColumnSpec,
    FeatureSpec,
    RuleSpec,
    describe_column,
    describe_feature,
    describe_readout_column,
    list_column_specs,
    list_feature_specs,
    list_readout_column_specs,
    list_rule_specs,
)


def __getattr__(name: str):
    if name == "count_beta_barrel_staves":
        from .readouts.beta_barrel_staves import count_beta_barrel_staves

        return count_beta_barrel_staves
    if name in {
        "compute_topology_diagnostics",
        "detect_mixed_topology",
        "score_fold_continuous_signals",
        "score_topology_ambiguity",
    }:
        from .readouts import topology_diagnostics

        return getattr(topology_diagnostics, name)
    raise AttributeError(f"module 'betlas' has no attribute {name!r}")


__all__ = [
    "compute_topology_diagnostics",
    "compute_grammar_features",
    "count_beta_barrel_staves",
    "detect_mixed_topology",
    "DomainCandidate",
    "describe_column",
    "describe_feature",
    "describe_readout_column",
    "ColumnSpec",
    "copy_example",
    "FeatureSpec",
    "FOLD_LABELS",
    "GeometrySignature",
    "list_slice_axes",
    "explain_fold_grammar",
    "get_grammar",
    "list_grammars",
    "list_feature_specs",
    "list_column_specs",
    "list_readout_column_specs",
    "list_examples",
    "list_rule_specs",
    "score_fold_continuous_signals",
    "score_fold_grammar",
    "score_topology_ambiguity",
    "SliceBin",
    "SliceBundle",
    "SliceConfig",
    "SlicePoint",
    "RuleSpec",
    "slice_feature_row",
    "slice_mmcif",
    "slice_structure",
    "summarize_slices",
    "StructureGeometry",
    "extract_feature_row",
    "extract_structure_features",
]
