"""
betlas.readouts.beta_barrel_detection

A small toolkit/pipeline to detect beta-barrel-like protein chains from PDB/mmCIF structures.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("betlas")
except PackageNotFoundError:  # pragma: no cover - editable tree before metadata exists
    __version__ = "0.0.0"

__all__ = [
    "AppConfig",
    "AnalysisReport",
    "Config",
    "ConfigValidationError",
    "ChainSliceBundle",
    "ChainNotFoundError",
    "BetlasBetaError",
    "DetectionResult",
    "DsspError",
    "DsspNotFoundError",
    "InputValidationError",
    "PipelineRunResult",
    "StructureParseError",
    "build_config",
    "detect",
    "find_dssp_binary",
    "require_dssp_binary",
    "main",
    "__version__",
]

from .config import AppConfig, Config, build_config
from .exceptions import (
    BetlasBetaError,
    ChainNotFoundError,
    ConfigValidationError,
    DsspError,
    DsspNotFoundError,
    InputValidationError,
    StructureParseError,
)
from .models import (
    AnalysisReport,
    ChainSliceBundle,
    DetectionResult,
    PipelineRunResult,
)
from .pipeline import detect, main
from .runtime import find_dssp_binary, require_dssp_binary
