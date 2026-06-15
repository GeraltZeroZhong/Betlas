from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

from ..config import AppConfig
from ..constants import RESULT_ERROR
from ..io.preparation import (
    iter_file_batches,
    prepare_file_batch,
    resolve_prepare_batch_size,
)
from .chain import analyze_chain_payload

DEFAULT_ANALYSIS_BATCH_SIZE = 64
PREPARE_IN_FLIGHT_MULTIPLIER = 2
ANALYSIS_IN_FLIGHT_MULTIPLIER = 2


try:
    from tqdm.auto import tqdm

    tqdm_write = tqdm.write
except Exception:  # pragma: no cover

    class _NullProgress:
        def __init__(self, iterable=None, **kwargs):
            del kwargs
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

        def update(self, value: int = 1) -> None:
            del value

    def tqdm(iterable=None, **kwargs):
        return _NullProgress(iterable, **kwargs)

    def tqdm_write(message: str) -> None:
        print(message)


def resolve_analysis_batch_size(cfg: AppConfig) -> int:
    batch_size = getattr(cfg.runtime, "analysis_batch_size", DEFAULT_ANALYSIS_BATCH_SIZE)
    try:
        return max(1, int(batch_size))
    except (TypeError, ValueError):
        return DEFAULT_ANALYSIS_BATCH_SIZE


def iter_payload_batches(
    payloads: list[dict[str, object]],
    batch_size: int,
) -> Iterable[list[dict[str, object]]]:
    for index in range(0, len(payloads), batch_size):
        yield payloads[index : index + batch_size]


def iter_prepared_payload_batches(
    files: Iterable[str],
    cfg: AppConfig,
    prepare_workers: int,
    *,
    on_errors: Callable[[list[str]], None] | None = None,
    show_progress: bool = True,
) -> Iterable[list[dict[str, object]]]:
    """Run preparation and yield chain-level payload batches as they complete."""
    file_list = list(files)

    if prepare_workers <= 1:
        batch_size = resolve_prepare_batch_size(cfg)
        with tqdm(
            total=len(file_list),
            desc="Preparing",
            unit="file",
            disable=not show_progress,
        ) as progress_bar:
            for file_batch in iter_file_batches(file_list, batch_size):
                result = prepare_file_batch(file_batch, cfg)
                errors = [str(error) for error in result.get("errors", [])]
                if errors:
                    if on_errors is not None:
                        on_errors(errors)
                    if show_progress:
                        for error in errors:
                            tqdm_write(f"  [X] Load failed: {error}")
                progress_bar.update(int(result.get("processed", len(file_batch))))
                payload_batch = result.get("payloads", [])
                if payload_batch:
                    yield payload_batch
        return

    batch_size = resolve_prepare_batch_size(cfg)
    batch_iterator = iter(iter_file_batches(file_list, batch_size))
    max_in_flight = max(1, int(prepare_workers) * PREPARE_IN_FLIGHT_MULTIPLIER)
    pending: set[Future] = set()
    future_sizes: dict[Future, int] = {}
    future_batches: dict[Future, list[str]] = {}

    with ProcessPoolExecutor(max_workers=prepare_workers) as executor:

        def submit_next_batch() -> bool:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                return False
            future = executor.submit(prepare_file_batch, batch, cfg)
            pending.add(future)
            future_sizes[future] = len(batch)
            future_batches[future] = list(batch)
            return True

        for _ in range(max_in_flight):
            if not submit_next_batch():
                break

        with tqdm(
            total=len(file_list),
            desc="Preparing",
            unit="file",
            disable=not show_progress,
        ) as progress_bar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed_count = future_sizes.pop(future, 0)
                    submitted_batch = future_batches.pop(future, [])
                    payload_batch: list[dict[str, object]] = []
                    try:
                        result = future.result()
                    except Exception as exc:
                        errors = [
                            f"{Path(path).name}: prepare worker failed for {path}: {exc}"
                            for path in submitted_batch
                        ] or [f"Prepare worker failed: {exc}"]
                        if on_errors is not None:
                            on_errors(errors)
                        if show_progress:
                            for error in errors:
                                tqdm_write(f"  [X] {error}")
                    else:
                        errors = [str(error) for error in result.get("errors", [])]
                        if errors:
                            if on_errors is not None:
                                on_errors(errors)
                            if show_progress:
                                for error in errors:
                                    tqdm_write(f"  [X] Load failed: {error}")
                        completed_count = int(result.get("processed", completed_count))
                        payload_batch = result.get("payloads", [])
                    finally:
                        progress_bar.update(completed_count)

                    if payload_batch:
                        yield payload_batch

                    while len(pending) < max_in_flight and submit_next_batch():
                        pass


def collect_payloads(
    files: Iterable[str],
    cfg: AppConfig,
    prepare_workers: int,
    *,
    show_progress: bool = True,
) -> list[dict[str, object]]:
    """Run the preparation phase and collect chain-level payloads."""
    payloads: list[dict[str, object]] = []
    for payload_batch in iter_prepared_payload_batches(
        files,
        cfg,
        prepare_workers,
        show_progress=show_progress,
    ):
        payloads.extend(payload_batch)
    return payloads


def analyze_payload_batch(
    payloads: list[dict[str, object]],
    cfg: AppConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        try:
            rows.append(analyze_chain_payload(payload, cfg))
        except Exception as exc:
            rows.append(
                {
                    "filename": str(payload.get("filename", "")),
                    "chain": str(payload.get("chain", "")),
                    "_source_path": str(payload.get("source_path", "")),
                    "_chain_index": payload.get("_chain_index", 0),
                    "result": RESULT_ERROR,
                    "result_stage": "error",
                    "reason": f"Analysis failed: {type(exc).__name__}: {exc}",
                }
            )
    return rows


def run_analysis_stream(
    payload_batches: Iterable[list[dict[str, object]]],
    cfg: AppConfig,
    workers: int,
    *,
    on_results: Callable[[list[dict[str, object]]], None] | None = None,
    show_progress: bool = True,
) -> list[dict[str, object]]:
    """Analyze payload batches with bounded in-flight work."""
    results: list[dict[str, object]] = []
    batch_size = resolve_analysis_batch_size(cfg)

    def handle_rows(rows: list[dict[str, object]], progress_bar) -> None:
        if on_results is not None:
            on_results(rows)
        results.extend(rows)
        progress_bar.update(len(rows))

    if workers <= 1:
        with tqdm(desc="Analyzing", unit="chain", disable=not show_progress) as progress_bar:
            for payload_group in payload_batches:
                for payload_batch in iter_payload_batches(payload_group, batch_size):
                    handle_rows(analyze_payload_batch(payload_batch, cfg), progress_bar)
        return results

    max_in_flight = max(1, int(workers) * ANALYSIS_IN_FLIGHT_MULTIPLIER)
    pending: set[Future] = set()
    future_sizes: dict[Future, int] = {}
    future_payloads: dict[Future, list[dict[str, object]]] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(desc="Analyzing", unit="chain", disable=not show_progress) as progress_bar:

            def drain_completed(done: set[Future]) -> None:
                for future in done:
                    future_sizes.pop(future, 0)
                    submitted_payloads = future_payloads.pop(future, [])
                    try:
                        rows = future.result()
                    except Exception as exc:
                        rows = [
                            {
                                "filename": str(payload.get("filename", "")),
                                "chain": str(payload.get("chain", "")),
                                "_source_path": str(payload.get("source_path", "")),
                                "_chain_index": payload.get("_chain_index", 0),
                                "result": RESULT_ERROR,
                                "result_stage": "error",
                                "reason": f"Worker crashed: {exc}",
                            }
                            for payload in submitted_payloads
                        ]
                        if not rows:
                            rows = [
                                {
                                    "filename": "",
                                    "chain": "",
                                    "result": RESULT_ERROR,
                                    "result_stage": "error",
                                    "reason": f"Worker crashed: {exc}",
                                }
                            ]
                    handle_rows(rows, progress_bar)

            for payload_group in payload_batches:
                for payload_batch in iter_payload_batches(payload_group, batch_size):
                    future = executor.submit(analyze_payload_batch, payload_batch, cfg)
                    pending.add(future)
                    future_sizes[future] = len(payload_batch)
                    future_payloads[future] = list(payload_batch)
                    while len(pending) >= max_in_flight:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        drain_completed(done)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                drain_completed(done)

    return results


def run_analysis(
    payloads: list[dict[str, object]],
    cfg: AppConfig,
    workers: int,
) -> list[dict[str, object]]:
    """Run the CPU-heavy chain analysis stage."""
    return run_analysis_stream([payloads], cfg, workers)
