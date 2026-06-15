from __future__ import annotations

from .constants import FOLD_LABELS
from .features.extract import extract_feature_row
from .models import DomainCandidate, GeometrySignature, StructureGeometry


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
    "count_beta_barrel_staves",
    "detect_mixed_topology",
    "DomainCandidate",
    "FOLD_LABELS",
    "GeometrySignature",
    "score_fold_continuous_signals",
    "score_topology_ambiguity",
    "StructureGeometry",
    "extract_feature_row",
]
