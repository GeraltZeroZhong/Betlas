from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import build_config
from .models import PipelineRunResult
from .pipeline import apply_runtime_overrides, run_pipeline_result

READOUT_NAME = "beta-barrel-staves"


def count_beta_barrel_staves(
    path: str | Path,
    *,
    output: str | Path | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    overrides: Mapping[str, Any] | list[str] | None = None,
    write_csv: bool = True,
    print_summary: bool = True,
) -> PipelineRunResult:
    """Run the Betlas beta-barrel stave-count readout."""
    cfg = build_config(overrides)
    cfg = apply_runtime_overrides(
        cfg,
        input_path=str(path),
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=str(output) if output is not None else None,
    )
    return run_pipeline_result(
        cfg,
        write_csv=write_csv,
        print_summary=print_summary,
        strict_input=True,
        raise_on_all_prepare_failures=True,
    )
