from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from ..bootstrap import configure_thread_environment
from ..config import AppConfig, build_config, sync_compat_config, validate_config
from ..exceptions import InputValidationError
from ..io.metadata import build_run_metadata, default_metadata_path, write_run_metadata
from ..io.results import print_results_summary, write_results_csv
from ..models import PipelineRunResult
from ..runtime import require_dssp_binary
from .execution import iter_prepared_payload_batches, run_analysis_stream

_INTERNAL_RESULT_KEYS = ("_source_path", "_chain_index")
_RESULT_STAGE_ORDER = {
    "prefilter": 0,
    "barrel_gate": 1,
    "count": 2,
    "prepare": 3,
    "error": 4,
}


def _has_allowed_suffix(path: Path, allowed_suffixes: tuple[str, ...]) -> bool:
    if not allowed_suffixes:
        return True
    filename = path.name.lower()
    return any(filename.endswith(suffix) for suffix in allowed_suffixes)


def discover_input_files(
    input_path: str,
    allowed_suffixes: list[str],
    *,
    strict: bool = False,
) -> list[str]:
    """Resolve a directory or single structure file into an explicit file list."""
    if not str(input_path).strip():
        raise InputValidationError("Input path is required.")
    path = Path(input_path).expanduser()
    normalized_suffixes = tuple(str(suffix).lower() for suffix in allowed_suffixes)
    if not path.exists():
        if strict:
            raise InputValidationError(f"Input path does not exist: {path}")
        return [str(path)]

    if path.is_dir():
        files = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and _has_allowed_suffix(file_path, normalized_suffixes)
        )
        if strict and not files:
            allowed = ", ".join(allowed_suffixes)
            raise InputValidationError(f"No structure files ({allowed}) were found in: {path}")
        return [str(file_path) for file_path in sorted(files)]

    if strict and not _has_allowed_suffix(path, normalized_suffixes):
        allowed = ", ".join(allowed_suffixes)
        raise InputValidationError(
            f"Input file has unsupported suffix {path.suffix!r}. Expected one of: {allowed}."
        )
    return [str(path)]


def os_cpu_count() -> int:
    import os

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return os.cpu_count() or 1


def resolve_analysis_worker_count(configured_workers: int | None, cpu_reserve: int) -> int:
    """Choose a sensible default analysis worker count from available CPUs."""
    if configured_workers is not None:
        return max(1, int(configured_workers))

    available_cpus = os_cpu_count()
    return max(1, available_cpus - max(0, cpu_reserve))


def resolve_prepare_worker_count(configured_workers: int | None, analysis_workers: int) -> int:
    if configured_workers is not None:
        return max(1, int(configured_workers))
    return max(1, analysis_workers)


def apply_runtime_overrides(
    cfg: AppConfig,
    *,
    input_path: str | None = None,
    chain_id: str | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    out_csv: str | None = None,
) -> AppConfig:
    updated = deepcopy(cfg)
    if input_path is not None:
        updated.input.path = str(input_path)
    if chain_id is not None:
        updated.input.chain_id = str(chain_id)
    if workers is not None:
        updated.runtime.workers = int(workers)
    if prepare_workers is not None:
        updated.runtime.prepare_workers = int(prepare_workers)
    if out_csv is not None:
        updated.output.csv_path = str(out_csv)
    sync_compat_config(updated)
    return updated


def _safe_sort_int(value: object, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _resolved_path_key(path_value: object) -> str:
    try:
        return str(Path(str(path_value)).expanduser().resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return str(path_value)


def _input_order_maps(input_files: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    exact = {str(path): index for index, path in enumerate(input_files)}
    resolved = {_resolved_path_key(path): index for index, path in enumerate(input_files)}
    return exact, resolved


def _unique_basename_order(input_files: list[str]) -> dict[str, int]:
    basenames = [Path(path).name for path in input_files]
    counts = Counter(basenames)
    return {
        Path(path).name: index
        for index, path in enumerate(input_files)
        if counts[Path(path).name] == 1
    }


def _row_input_index(
    row: dict[str, object],
    *,
    exact_order: dict[str, int],
    resolved_order: dict[str, int],
    basename_order: dict[str, int],
    fallback: int,
) -> int:
    source_path = str(row.get("_source_path", "") or "")
    if source_path in exact_order:
        return exact_order[source_path]

    resolved_source = _resolved_path_key(source_path) if source_path else ""
    if resolved_source in resolved_order:
        return resolved_order[resolved_source]

    filename = str(row.get("filename", "") or "")
    return basename_order.get(Path(filename).name, fallback)


def _sort_and_sanitize_result_rows(
    rows: list[dict[str, object]],
    input_files: list[str],
) -> list[dict[str, object]]:
    exact_order, resolved_order = _input_order_maps(input_files)
    basename_order = _unique_basename_order(input_files)
    fallback_index = len(input_files)

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _row_input_index(
                row,
                exact_order=exact_order,
                resolved_order=resolved_order,
                basename_order=basename_order,
                fallback=fallback_index,
            ),
            _safe_sort_int(row.get("_chain_index", 0)),
            str(row.get("chain", "") or ""),
            _RESULT_STAGE_ORDER.get(str(row.get("result_stage", "") or ""), 99),
            str(row.get("filename", "") or ""),
            str(row.get("reason", "") or ""),
        ),
    )
    return [
        {key: value for key, value in row.items() if key not in _INTERNAL_RESULT_KEYS}
        for row in sorted_rows
    ]


def _failure_reason_summary(rows: list[dict[str, object]], *, limit: int = 3) -> str:
    reasons = [str(row.get("reason", "")).strip() for row in rows if str(row.get("reason", "")).strip()]
    if not reasons:
        return ""
    shown = "; ".join(reasons[:limit])
    more = f"; plus {len(reasons) - limit} more failure(s)" if len(reasons) > limit else ""
    return f" Reasons: {shown}{more}"


def _prepare_error_rows(errors: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for error in errors:
        source_path, _, detail = error.partition(":")
        filename = Path(source_path.strip()).name if source_path.strip() else ""
        rows.append(
            {
                "filename": filename,
                "source_path": source_path.strip(),
                "chain": "",
                "result": "ERROR",
                "result_stage": "prepare",
                "score_type": "not_applicable",
                "calibration_status": "not_applicable",
                "config_profile": "native",
                "reason": detail.strip() or error,
            }
        )
    return rows


def _write_metadata_if_requested(
    cfg: AppConfig,
    *,
    input_files: list[str],
    rows: list[dict[str, object]],
    output_path: str | None,
    write_csv: bool,
) -> None:
    if not write_csv or not output_path:
        return
    metadata_path = cfg.output.metadata_path or default_metadata_path(output_path)
    metadata = build_run_metadata(
        cfg,
        input_files=input_files,
        output_csv=output_path,
        row_count=len(rows),
    )
    write_run_metadata(metadata, metadata_path)


def run_pipeline_result(
    cfg: AppConfig,
    *,
    write_csv: bool = True,
    print_summary: bool = True,
    strict_input: bool = False,
    show_progress: bool = True,
    raise_on_all_prepare_failures: bool = False,
) -> PipelineRunResult:
    """Run the full beta-strand stave counting pipeline and return structured results."""
    configure_thread_environment()
    cfg = deepcopy(cfg)
    validate_config(cfg)

    files = discover_input_files(cfg.input.path, cfg.input.allowed_suffixes, strict=strict_input)
    if Path(cfg.input.path).expanduser().is_dir() and not files:
        allowed = "/".join(cfg.input.allowed_suffixes)
        if print_summary:
            print(f"No {allowed} files found in: {cfg.input.path}")
        if write_csv:
            write_results_csv([], cfg.output.csv_path)
            _write_metadata_if_requested(
                cfg,
                input_files=files,
                rows=[],
                output_path=cfg.output.csv_path,
                write_csv=write_csv,
            )
        output_path = cfg.output.csv_path if write_csv else None
        return PipelineRunResult.from_rows([], input_files=files, output_path=output_path, config=cfg)

    cfg.runtime.dssp_bin_path = require_dssp_binary(cfg.runtime.dssp_bin_path)
    sync_compat_config(cfg)

    analysis_workers = resolve_analysis_worker_count(cfg.runtime.workers, cfg.runtime.cpu_reserve)
    prepare_workers = resolve_prepare_worker_count(cfg.runtime.prepare_workers, analysis_workers)
    if write_csv and Path(cfg.output.csv_path).expanduser().is_dir():
        raise InputValidationError(
            f"Output CSV path points to a directory: {cfg.output.csv_path}"
        )

    if print_summary:
        print(
            f"\nRunning streaming pipeline with {prepare_workers} prepare worker(s) "
            f"and {analysis_workers} analysis worker(s)..."
        )
    prepare_errors: list[str] = []

    def record_prepare_errors(errors: list[str]) -> None:
        prepare_errors.extend(errors)

    payload_batches = iter_prepared_payload_batches(
        files,
        cfg,
        prepare_workers,
        on_errors=record_prepare_errors,
        show_progress=show_progress,
    )
    results = run_analysis_stream(
        payload_batches,
        cfg,
        analysis_workers,
        show_progress=show_progress,
    )
    prepare_rows = _prepare_error_rows(prepare_errors)

    all_results = _sort_and_sanitize_result_rows([*results, *prepare_rows], files)
    if write_csv:
        write_results_csv(all_results, cfg.output.csv_path)
        _write_metadata_if_requested(
            cfg,
            input_files=files,
            rows=all_results,
            output_path=cfg.output.csv_path,
            write_csv=write_csv,
        )
    if prepare_rows and not results:
        if print_summary:
            print_results_summary(
                all_results,
                cfg.output.csv_path,
                summary_limit=cfg.output.summary_limit,
                write_csv=False,
                output_written=write_csv,
            )
        output_path = cfg.output.csv_path if write_csv else None
        run_result = PipelineRunResult.from_rows(
            all_results,
            input_files=files,
            output_path=output_path,
            config=cfg,
        )
        if raise_on_all_prepare_failures:
            raise InputValidationError(
                f"All {len(files)} input file(s) failed during preparation."
                f"{_failure_reason_summary(all_results)}"
            )
        return run_result

    if not all_results:
        if print_summary:
            print("No analyzable chain payloads were produced.")
            if write_csv:
                print(f"\nResults written to: {cfg.output.csv_path}")
        output_path = cfg.output.csv_path if write_csv else None
        return PipelineRunResult.from_rows([], input_files=files, output_path=output_path, config=cfg)

    if print_summary:
        print_results_summary(
            all_results,
            cfg.output.csv_path,
            summary_limit=cfg.output.summary_limit,
            write_csv=False,
            output_written=write_csv,
        )
    output_path = cfg.output.csv_path if write_csv else None
    return PipelineRunResult.from_rows(
        all_results,
        input_files=files,
        output_path=output_path,
        config=cfg,
    )


def run_pipeline(cfg: AppConfig) -> list[dict[str, object]]:
    """Run the full beta-strand stave counting pipeline from a resolved config."""
    return run_pipeline_result(cfg).raw_rows()


def count_strands(
    input_path: str,
    *,
    config: AppConfig | None = None,
    cfg: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    output: str | None = None,
    write_csv: bool | None = None,
    print_summary: bool = False,
    show_progress: bool | None = None,
    strict_input: bool = True,
    allow_ungated: bool = False,
) -> PipelineRunResult:
    """
    Public Python API for counting beta-strand staves with structured results.

    CSV output is written only when ``output`` is provided or ``write_csv=True``.
    """
    if not allow_ungated:
        raise ValueError(
            "count_strands requires explicit allow_ungated=True. For biologically gated "
            "counting, use count_beta_barrel_staves(..., barrel_decisions=...)."
        )
    if config is not None and cfg is not None:
        raise TypeError("Pass only one of `config` or `cfg`.")
    if overrides is not None and (config is not None or cfg is not None):
        raise TypeError("Pass `overrides` only when Betlas beta-barrel-staves readout builds the config for you.")
    resolved_cfg = config or cfg or build_config(overrides)
    if workers is None and resolved_cfg.runtime.workers is None:
        workers = 1
    if prepare_workers is None and resolved_cfg.runtime.prepare_workers is None:
        prepare_workers = 1
    resolved_cfg = apply_runtime_overrides(
        resolved_cfg,
        input_path=input_path,
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=output,
    )
    should_write_csv = bool(output) if write_csv is None else bool(write_csv)
    should_show_progress = bool(print_summary) if show_progress is None else bool(show_progress)
    return run_pipeline_result(
        resolved_cfg,
        write_csv=should_write_csv,
        print_summary=print_summary,
        show_progress=should_show_progress,
        strict_input=strict_input,
    )


def detect(
    input_path: str,
    *,
    config: AppConfig | None = None,
    cfg: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    output: str | None = None,
    write_csv: bool | None = None,
    print_summary: bool = False,
    show_progress: bool | None = None,
    strict_input: bool = True,
    allow_ungated: bool = False,
) -> PipelineRunResult:
    """Compatibility alias for :func:`count_strands`."""
    return count_strands(
        input_path,
        config=config,
        cfg=cfg,
        overrides=overrides,
        workers=workers,
        prepare_workers=prepare_workers,
        output=output,
        write_csv=write_csv,
        print_summary=print_summary,
        show_progress=show_progress,
        strict_input=strict_input,
        allow_ungated=allow_ungated,
    )


def main(
    input_path: str | None = None,
    *,
    workers: int | None = None,
    prepare_workers: int | None = None,
    out_csv: str | None = None,
    cfg: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
    allow_ungated: bool = False,
) -> list[dict[str, object]]:
    """
    Backward-compatible entry point with optional Hydra overrides.
    """
    if not allow_ungated:
        raise ValueError(
            "beta-barrel-staves pipeline.main requires explicit allow_ungated=True. "
            "For biologically gated counting, use count_beta_barrel_staves(..., barrel_decisions=...)."
        )
    if cfg is not None and overrides is not None:
        raise TypeError("Pass `overrides` only when Betlas beta-barrel-staves readout builds the config for you.")
    resolved_cfg = cfg or build_config(overrides)
    resolved_cfg = apply_runtime_overrides(
        resolved_cfg,
        input_path=input_path,
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=out_csv,
    )
    return run_pipeline_result(
        resolved_cfg,
        write_csv=True,
        print_summary=True,
        strict_input=True,
        show_progress=True,
        raise_on_all_prepare_failures=True,
    ).raw_rows()
