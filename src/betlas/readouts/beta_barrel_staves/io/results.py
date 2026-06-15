from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from ..constants import (
    DEFAULT_RESULT_COLUMNS,
    DEFAULT_SUMMARY_COLUMNS,
    SUMMARY_COLUMN_WIDTHS,
    SUMMARY_DISPLAY_NAMES,
)


def _result_fieldnames(
    rows: list[dict[str, object]],
    *,
    include_extra: bool = False,
) -> list[str]:
    ordered_keys: list[str] = list(DEFAULT_RESULT_COLUMNS)
    seen_keys: set[str] = set(ordered_keys)

    if not include_extra:
        return ordered_keys

    for row in rows:
        for key in row:
            if key not in seen_keys:
                seen_keys.add(key)
                ordered_keys.append(key)
    return ordered_keys


def _row_for_fieldnames(row: dict[str, object], fieldnames: list[str]) -> dict[str, object]:
    return {key: row.get(key, "") for key in fieldnames}


def _ensure_output_parent(output_path: str) -> None:
    parent = Path(output_path).expanduser().parent
    if str(parent) and parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


class ResultCsvWriter:
    """Incrementally write result rows using the stable Betlas beta-barrel-staves readout schema."""

    def __init__(self, output_path: str, fieldnames: Iterable[str] | None = None):
        self.output_path = output_path
        self.fieldnames = list(fieldnames or DEFAULT_RESULT_COLUMNS)
        self._handle = None
        self._writer = None
        self._tmp_path: str | None = None

    def __enter__(self) -> ResultCsvWriter:
        _ensure_output_parent(self.output_path)
        output = Path(self.output_path).expanduser()
        fd, self._tmp_path = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=str(output.parent),
        )
        self._handle = os.fdopen(fd, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc, traceback
        self.close(commit=exc_type is None)

    def write_rows(self, rows: Iterable[dict[str, object]]) -> None:
        if self._writer is None:
            raise RuntimeError("ResultCsvWriter must be opened before writing rows.")
        for row in rows:
            self._writer.writerow(_row_for_fieldnames(row, self.fieldnames))

    def close(self, *, commit: bool = True) -> None:
        tmp_path = self._tmp_path
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None
        self._tmp_path = None
        if not tmp_path:
            return
        if commit:
            os.replace(tmp_path, Path(self.output_path).expanduser())
            return
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def write_results_csv(
    rows: list[dict[str, object]],
    output_path: str,
    *,
    include_extra: bool = False,
) -> None:
    """Write result rows without requiring pandas."""
    ordered_keys = _result_fieldnames(rows, include_extra=include_extra)
    _ensure_output_parent(output_path)
    output = Path(output_path).expanduser()
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=str(output.parent),
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(_row_for_fieldnames(row, ordered_keys))
        os.replace(tmp_path, output)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _layer_counts(row: dict[str, object]) -> str:
    supporting_layers = _safe_int(row.get("supporting_layers", 0))
    usable_layers = _safe_int(row.get("usable_layers", 0))
    total_layers = _safe_int(row.get("total_layers", 0))
    return f"{supporting_layers}/{usable_layers}/{total_layers}"


def _summary_rows(results: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in results:
        confidence = _safe_float(row.get("confidence", 0.0))
        rows.append(
            {
                "filename": str(row.get("filename", "")),
                "chain": str(row.get("chain", "")),
                "result": str(row.get("result", "")),
                "strand_count": str(row.get("strand_count", "")),
                "confidence": f"{confidence:.2f}" if confidence else "",
                "layer_counts": _layer_counts(row),
                "candidate_strands": str(row.get("candidate_strands", "")),
                "reason": str(row.get("reason", "")),
            }
        )
    return rows


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common())


def _summary_limit_value(summary_limit: int | None) -> int | None:
    if summary_limit is None:
        return 50
    try:
        value = int(summary_limit)
    except (TypeError, ValueError):
        return 50
    if value < 0:
        return None
    return value


def _limited_summary_rows(
    results: list[dict[str, object]],
    summary_limit: int | None,
) -> tuple[list[dict[str, object]], int | None]:
    limit = _summary_limit_value(summary_limit)
    if limit is None:
        return _summary_rows(results), None
    return _summary_rows(results[:limit]), limit


def print_results_summary(
    results: list[dict[str, object]],
    output_path: str,
    *,
    summary_limit: int | None = 50,
    write_csv: bool = True,
    output_written: bool | None = None,
) -> None:
    """Print a human-readable summary and persist the CSV."""
    if output_written is None:
        output_written = write_csv
    summary_rows, resolved_limit = _limited_summary_rows(results, summary_limit)
    result_counts = Counter(str(row.get("result", "") or "<blank>") for row in results)
    stage_counts = Counter(str(row.get("result_stage", "") or "<blank>") for row in results)

    print("\n=== Summary ===")
    print(f"Rows: {len(results)}")
    print(f"Results: {_format_counter(result_counts)}")
    print(f"Stages: {_format_counter(stage_counts)}")

    if pd is not None:
        dataframe = pd.DataFrame(summary_rows)
        if not dataframe.empty:
            display_frame = dataframe[list(DEFAULT_SUMMARY_COLUMNS)].rename(
                columns=SUMMARY_DISPLAY_NAMES
            )
            print()
            print(display_frame.to_string(index=False))
        if resolved_limit is not None and len(results) > resolved_limit:
            omitted = len(results) - resolved_limit
            print(f"\n... omitted {omitted} row(s) from console summary.")
        if write_csv:
            write_results_csv(results, output_path)
        if output_written or write_csv:
            print(f"\nResults written to: {output_path}")
        return

    widths = SUMMARY_COLUMN_WIDTHS
    header = (
        f"{SUMMARY_DISPLAY_NAMES['filename']:<{widths['filename']}} | "
        f"{SUMMARY_DISPLAY_NAMES['chain']:<{widths['chain']}} | "
        f"{SUMMARY_DISPLAY_NAMES['result']:<{widths['result']}} | "
        f"{SUMMARY_DISPLAY_NAMES['strand_count']:<{widths['strand_count']}} | "
        f"{SUMMARY_DISPLAY_NAMES['confidence']:<{widths['confidence']}} | "
        f"{SUMMARY_DISPLAY_NAMES['layer_counts']:<{widths['layer_counts']}} | "
        f"{SUMMARY_DISPLAY_NAMES['candidate_strands']:<{widths['candidate_strands']}} | "
        f"{SUMMARY_DISPLAY_NAMES['reason']:<{widths['reason']}}"
    )
    if summary_rows:
        print()
        print(header)
        print("-" * len(header))
    for row in summary_rows:
        print(
            f"{str(row.get('filename', '')):<{widths['filename']}} | "
            f"{str(row.get('chain', '')):<{widths['chain']}} | "
            f"{str(row.get('result', '')):<{widths['result']}} | "
            f"{str(row.get('strand_count', '')):<{widths['strand_count']}} | "
            f"{str(row.get('confidence', '')):<{widths['confidence']}} | "
            f"{str(row.get('layer_counts', '')):<{widths['layer_counts']}} | "
            f"{str(row.get('candidate_strands', '')):<{widths['candidate_strands']}} | "
            f"{str(row.get('reason', '')):<{widths['reason']}}"
        )

    if resolved_limit is not None and len(results) > resolved_limit:
        omitted = len(results) - resolved_limit
        print(f"\n... omitted {omitted} row(s) from console summary.")
    if write_csv:
        write_results_csv(results, output_path)
    if output_written or write_csv:
        print(f"\nResults written to: {output_path}")
