from __future__ import annotations


class BetlasBetaError(Exception):
    """Base exception for Betlas beta-barrel detection user-facing errors."""


class ConfigValidationError(BetlasBetaError, ValueError):
    """Raised when a Betlas beta-barrel detection configuration contains invalid values."""


class InputValidationError(BetlasBetaError, ValueError):
    """Raised when an input path or file set cannot be used for analysis."""


class DsspNotFoundError(BetlasBetaError, RuntimeError):
    """Raised when the DSSP executable cannot be found."""


class DsspError(BetlasBetaError, RuntimeError):
    """Raised when DSSP fails while preparing a structure."""


class StructureParseError(BetlasBetaError, ValueError):
    """Raised when a PDB/mmCIF structure cannot be parsed."""


class ChainNotFoundError(BetlasBetaError, KeyError):
    """Raised when a requested chain is not present in a structure."""
