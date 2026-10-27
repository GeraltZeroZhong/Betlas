from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .constants import FOLD_LABELS
from .examples import copy_example, list_examples
from .features.extract import extract_classifier_features, extract_feature_row, extract_structure_features
from .grammars import (
    compute_grammar_features,
    get_grammar,
    list_grammars,
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
    describe_column,
    describe_feature,
    describe_readout_column,
    list_column_specs,
    list_feature_specs,
    list_readout_column_specs,
)

try:
    __version__ = version("betlas")
except PackageNotFoundError:  # pragma: no cover - source-tree fallback without package metadata
    __version__ = "0.0.0"


def __getattr__(name: str):
    if name in {"predict_structure", "predict_features", "list_checkpoints", "install_checkpoint"}:
        from . import prediction

        return getattr(prediction, name)
    raise AttributeError(f"module 'betlas' has no attribute {name!r}")


__all__ = [
    "__version__",
    "predict_structure",
    "predict_features",
    "list_checkpoints",
    "install_checkpoint",
    "compute_grammar_features",
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
    "get_grammar",
    "list_grammars",
    "list_feature_specs",
    "list_column_specs",
    "list_readout_column_specs",
    "list_examples",
    "SliceBin",
    "SliceBundle",
    "SliceConfig",
    "SlicePoint",
    "slice_feature_row",
    "slice_mmcif",
    "slice_structure",
    "summarize_slices",
    "StructureGeometry",
    "extract_classifier_features",
    "extract_feature_row",
    "extract_structure_features",
]
