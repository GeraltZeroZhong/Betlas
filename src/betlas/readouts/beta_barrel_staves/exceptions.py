from __future__ import annotations


class BetaBarrelStavesReadoutError(Exception):
    """Base exception for Betlas beta-barrel-staves readout user-facing errors."""


class ConfigValidationError(BetaBarrelStavesReadoutError, ValueError):
    """Raised when a Betlas beta-barrel-staves readout configuration contains invalid values."""


class InputValidationError(BetaBarrelStavesReadoutError, ValueError):
    """Raised when an input path or file set cannot be used for analysis."""


class DsspNotFoundError(BetaBarrelStavesReadoutError, RuntimeError):
    """Raised when the DSSP executable cannot be found."""


class DsspError(BetaBarrelStavesReadoutError, RuntimeError):
    """Raised when DSSP fails while preparing a structure."""


class StructureParseError(BetaBarrelStavesReadoutError, ValueError):
    """Raised when a PDB/mmCIF structure cannot be parsed."""


class ChainNotFoundError(BetaBarrelStavesReadoutError, KeyError):
    """Raised when a requested chain is not present in a structure."""
