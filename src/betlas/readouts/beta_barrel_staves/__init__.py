"""
betlas.readouts.beta_barrel_staves

A small toolkit/pipeline to count beta-strand staves from PDB/mmCIF structures.
"""
# ruff: noqa: E402
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .bootstrap import configure_thread_environment

configure_thread_environment()

try:
    __version__ = version("betlas")
except PackageNotFoundError:  # pragma: no cover - editable tree before metadata exists
    __version__ = "0.0.0"

__all__ = [
    "AppConfig",
    "AnalysisReport",
    "Config",
    "ConfigValidationError",
    "ChainNotFoundError",
    "BetaBarrelStavesReadoutError",
    "DetectionResult",
    "DsspError",
    "DsspNotFoundError",
    "InputValidationError",
    "LayerDiagnostic",
    "PipelineRunResult",
    "PreparedChainPayload",
    "ProteinLoader",
    "PCAAligner",
    "ProteinSlicer",
    "ResidueRecord",
    "StructureParseError",
    "StrandCountAnalyzer",
    "BarrelAnalyzer",
    "StrandCountReport",
    "StrandCountResult",
    "build_config",
    "count_beta_barrel_staves",
    "count_strands",
    "detect",
    "find_dssp_binary",
    "require_dssp_binary",
    "main",
    "__version__",
]

from .analysis.analyzer import BarrelAnalyzer, StrandCountAnalyzer
from .config import AppConfig, Config, build_config
from .exceptions import (
    BetaBarrelStavesReadoutError,
    ChainNotFoundError,
    ConfigValidationError,
    DsspError,
    DsspNotFoundError,
    InputValidationError,
    StructureParseError,
)
from .geometry.alignment import PCAAligner
from .geometry.slicer import ProteinSlicer
from .io.loader import ProteinLoader
from .models import (
    AnalysisReport,
    DetectionResult,
    LayerDiagnostic,
    PipelineRunResult,
    PreparedChainPayload,
    ResidueRecord,
    StrandCountReport,
    StrandCountResult,
)
from .pipeline import count_strands, detect, main
from .readout import count_beta_barrel_staves
from .runtime import find_dssp_binary, require_dssp_binary
