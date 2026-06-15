"""Input preparation, cache, and result-output interfaces."""

from __future__ import annotations

from .loader import ProteinLoader
from .metadata import (
    RUN_METADATA_SCHEMA,
    build_run_metadata,
    default_metadata_path,
    write_run_metadata,
)
from .preparation import PrepareFailure, prepare_file_batch, prepare_one_file
from .results import ResultCsvWriter, print_results_summary, write_results_csv

__all__ = [
    "PrepareFailure",
    "ProteinLoader",
    "RUN_METADATA_SCHEMA",
    "ResultCsvWriter",
    "build_run_metadata",
    "default_metadata_path",
    "prepare_file_batch",
    "prepare_one_file",
    "print_results_summary",
    "write_run_metadata",
    "write_results_csv",
]
