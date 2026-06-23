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
    barrel_decisions: str | Path | None = None,
    allow_ungated: bool = False,
    workers: int | None = None,
    prepare_workers: int | None = None,
    overrides: Mapping[str, Any] | list[str] | None = None,
    write_csv: bool | None = None,
    print_summary: bool = False,
) -> PipelineRunResult:
    """Run the Betlas beta-barrel stave-count readout."""
    if barrel_decisions is None and not allow_ungated:
        raise ValueError(
            "count_beta_barrel_staves requires barrel_decisions from beta-barrel-detection "
            "or explicit allow_ungated=True for exploratory counting"
        )
    cfg = build_config(overrides)
    cfg = apply_runtime_overrides(
        cfg,
        input_path=str(path),
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=str(output) if output is not None else None,
    )
    should_write_csv = bool(output) if write_csv is None else bool(write_csv)
    if barrel_decisions is None:
        return run_pipeline_result(
            cfg,
            write_csv=should_write_csv,
            print_summary=print_summary,
            strict_input=True,
            raise_on_all_prepare_failures=True,
        )

    from .cli import _apply_barrel_decisions, _load_barrel_decisions
    from .io.results import print_results_summary, write_results_csv

    decisions = _load_barrel_decisions(str(barrel_decisions))
    result = run_pipeline_result(
        cfg,
        write_csv=False,
        print_summary=False,
        strict_input=True,
        raise_on_all_prepare_failures=False,
    )
    rows = _apply_barrel_decisions(result.raw_rows(), decisions)
    output_path = cfg.output.csv_path if should_write_csv else None
    if should_write_csv:
        write_results_csv(rows, cfg.output.csv_path)
    if print_summary:
        print_results_summary(
            rows,
            cfg.output.csv_path,
            summary_limit=cfg.output.summary_limit,
            write_csv=False,
            output_written=should_write_csv,
        )
    return PipelineRunResult.from_rows(
        rows,
        input_files=result.input_files,
        output_path=output_path,
        config=cfg,
    )
