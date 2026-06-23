from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException

from ...provenance import file_state
from .bootstrap import configure_thread_environment
from .config import build_config
from .constants import DEFAULT_RESULT_COLUMNS, RESULT_ERROR, RESULT_FILTERED_OUT
from .exceptions import BetaBarrelStavesReadoutError
from .io.metadata import build_run_metadata, default_metadata_path, write_run_metadata
from .io.results import print_results_summary, write_results_csv
from .pipeline import apply_runtime_overrides, run_pipeline_result
from .runtime import runtime_summary


def _looks_like_hydra_override(token: str) -> bool:
    return token.startswith(("+", "~")) or ("=" in token)


def _package_version() -> str:
    try:
        return version("betlas")
    except PackageNotFoundError:
        return "0.0.0"


def _has_override(overrides: list[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(override.startswith(prefixes) for override in overrides)


def _has_true_override(overrides: list[str], key: str) -> bool:
    for override in overrides:
        if not override.startswith((f"{key}=", f"+{key}=", f"++{key}=")):
            continue
        value = override.split("=", 1)[1].strip().lower()
        return value in {"1", "true", "yes", "on"}
    return False


def _reject_unknown_options(parser: argparse.ArgumentParser, tokens: list[str]) -> None:
    for token in tokens:
        if token.startswith("-"):
            parser.error(f"unrecognized argument: {token}")


def _resolved_source_key(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve(strict=False))


def _recover_positional_path(
    parser: argparse.ArgumentParser,
    path: str | None,
    tokens: list[str],
) -> tuple[str | None, list[str]]:
    remaining: list[str] = []
    recovered_path = path
    for token in tokens:
        if not _looks_like_hydra_override(token) and not token.startswith("-"):
            if recovered_path is None:
                recovered_path = token
                continue
            parser.error(f"unexpected extra input path: {token}")
        remaining.append(token)
    return recovered_path, remaining


def _load_barrel_decisions(path: str) -> dict[tuple[str, str], dict[str, str]]:
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"barrel decisions CSV does not exist: {csv_path}")
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    basename_sources: dict[tuple[str, str], set[str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            chain = str(row.get("chain", "")).strip()
            if not chain:
                continue
            filename = str(row.get("filename", "")).strip()
            source_path = str(row.get("source_path", "")).strip()
            if not source_path:
                raise ValueError(
                    "barrel decisions CSV must contain source_path for each gated row; "
                    "basename-only matching is intentionally disabled"
                )
            basename_sources.setdefault((filename or Path(source_path).name, chain), set()).add(
                _resolved_source_key(source_path)
            )
            keys = {source_path, _resolved_source_key(source_path)}
            for key in sorted(item for item in keys if item):
                decision_key = (key, chain)
                if decision_key in decisions:
                    raise ValueError(
                        f"duplicate barrel decision for source_path={key!r}, chain={chain!r}"
                    )
                decisions[decision_key] = dict(row)
    ambiguous = [
        (filename, chain, sorted(sources))
        for (filename, chain), sources in basename_sources.items()
        if filename and len(sources) > 1
    ]
    if ambiguous:
        filename, chain, sources = ambiguous[0]
        raise ValueError(
            "ambiguous barrel decisions for duplicate filename/chain "
            f"({filename!r}, {chain!r}); use unique source_path values. Sources: {sources}"
        )
    return decisions


def _apply_barrel_decisions(
    rows: list[dict[str, object]],
    decisions: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    preserved_filtered_fields = {
        "filename",
        "source_path",
        "chain",
        "result",
        "result_stage",
        "score_type",
        "calibration_status",
        "config_profile",
        "barrel_gate_enabled",
        "barrel_gate_passed",
        "barrel_gate_result",
        "barrel_gate_score",
        "barrel_gate_reason",
        "reason",
    }
    string_filtered_fields = {
        "confidence_basis",
        "layer_count_mode",
        "barrel_wall_graph_variant",
        "barrel_wall_graph_visibility_regime",
        "barrel_wall_graph_selection_strategy",
        "barrel_wall_graph_selected_ids",
        "barrel_wall_graph_raw_curve",
        "barrel_wall_graph_outer_curve",
        "barrel_wall_graph_outer_gap_curve",
        "axis_hypothesis_name",
        "axis_hypothesis_candidates",
        "run_window_core_candidate_windows",
        "run_window_core_reanalyzed_windows",
    }

    def clear_filtered_count_fields(item: dict[str, object]) -> None:
        for field in DEFAULT_RESULT_COLUMNS:
            if field in preserved_filtered_fields or field not in item:
                continue
            item[field] = "" if field in string_filtered_fields else 0
        for field in item:
            if field in preserved_filtered_fields or field in DEFAULT_RESULT_COLUMNS:
                continue
            if field.startswith(("barrel_wall_graph_", "run_window_core_", "axis_hypothesis_")):
                item[field] = "" if field in string_filtered_fields else 0
        for field in ("strand_count", "confidence", "candidate_strands", "persistent_strands"):
            if field in item:
                item[field] = 0

    gated_rows: list[dict[str, object]] = []
    for row in rows:
        chain = str(row.get("chain", ""))
        keys = [
            str(row.get("source_path", "")),
            _resolved_source_key(str(row.get("source_path", ""))),
        ]
        decision = next((decisions.get((key, chain)) for key in keys if key and (key, chain) in decisions), None)
        updated = dict(row)
        updated["barrel_gate_enabled"] = True
        if str(updated.get("result", "")).upper() == "ERROR" or str(updated.get("result_stage", "")).lower() in {
            "prepare",
            "error",
        }:
            updated.update(
                {
                    "barrel_gate_passed": False,
                    "barrel_gate_result": "NOT_EVALUATED",
                    "barrel_gate_score": 0.0,
                    "barrel_gate_reason": "Preparation or analysis failed before barrel gate evaluation.",
                }
            )
            gated_rows.append(updated)
            continue
        if decision is None:
            updated.update(
                {
                    "result": RESULT_FILTERED_OUT,
                    "result_stage": "barrel_gate",
                    "strand_count": 0,
                    "confidence": 0.0,
                    "barrel_gate_passed": False,
                    "barrel_gate_result": "MISSING",
                    "barrel_gate_score": 0.0,
                    "barrel_gate_reason": "No matching beta-barrel detection decision row.",
                    "reason": "Filtered by barrel decisions: no matching detection row.",
                }
            )
            clear_filtered_count_fields(updated)
            gated_rows.append(updated)
            continue
        result = str(decision.get("result", ""))
        score = decision.get("decision_score", decision.get("score", 0.0))
        passed = result == "BARREL"
        updated.update(
            {
                "barrel_gate_passed": passed,
                "barrel_gate_result": result or "UNKNOWN",
                "barrel_gate_score": score,
                "barrel_gate_reason": str(decision.get("reason", "")),
            }
        )
        if result.upper() == "ERROR":
            updated.update(
                {
                    "result": RESULT_ERROR,
                    "result_stage": "barrel_gate",
                    "strand_count": 0,
                    "confidence": 0.0,
                    "reason": "Barrel gate source row was ERROR; candidate stave count not trusted.",
                }
            )
            clear_filtered_count_fields(updated)
            gated_rows.append(updated)
            continue
        if not passed:
            updated.update(
                {
                    "result": RESULT_FILTERED_OUT,
                    "result_stage": "barrel_gate",
                    "strand_count": 0,
                    "confidence": 0.0,
                    "reason": f"Filtered by barrel decisions: result={result or 'UNKNOWN'}.",
                }
            )
            clear_filtered_count_fields(updated)
        gated_rows.append(updated)
    return gated_rows


def _write_gated_metadata(
    *,
    cfg: object,
    result: object,
    rows: list[dict[str, object]],
    output_csv: str,
    barrel_decisions_csv: str,
) -> None:
    metadata_path = cfg.output.metadata_path or default_metadata_path(output_csv)
    metadata = build_run_metadata(
        cfg,
        input_files=list(result.input_files),
        output_csv=output_csv,
        row_count=len(rows),
    )
    metadata["barrel_gate"] = {
        "enabled": True,
        "decisions_csv": file_state(barrel_decisions_csv),
        "passed_rows": int(sum(bool(row.get("barrel_gate_passed")) for row in rows)),
        "filtered_rows": int(
            sum(str(row.get("result", "")).upper() == RESULT_FILTERED_OUT for row in rows)
        ),
    }
    write_run_metadata(metadata, metadata_path)


def _failure_reason_summary(rows: list[dict[str, object]], *, limit: int = 3) -> str:
    reasons = [str(row.get("reason", "")).strip() for row in rows if str(row.get("reason", "")).strip()]
    if not reasons:
        return ""
    shown = "; ".join(reasons[:limit])
    more = f"; plus {len(reasons) - limit} more failure(s)" if len(reasons) > limit else ""
    return f"{shown}{more}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="betlas readout beta-barrel-staves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Count beta-strand staves from PDB/mmCIF inputs and write a CSV "
            "summary. Advanced KEY=VALUE arguments are treated as Hydra overrides."
        ),
        epilog=(
            "Examples:\n"
            "  betlas readout beta-barrel-staves structure.cif --barrel-decisions runs/detection.csv --out runs/staves.csv\n"
            "  betlas readout beta-barrel-staves structure.cif --chain A --allow-ungated --out runs/staves_A.csv\n"
            "  betlas readout beta-barrel-staves structures/ --barrel-decisions runs/detection.csv --workers 8 --prep 4 --out runs/staves.csv\n"
            "  betlas readout beta-barrel-staves input.path=structures/ output.csv=runs/staves.csv runtime.dssp_bin_path=mkdssp --allow-ungated\n"
            "  betlas readout beta-barrel-staves --check-env\n\n"
            "Output CSV: one row per analyzed chain with candidate stave count, uncalibrated heuristic confidence, slice, geometry, and runtime-status columns.\n"
            "Stdout is progress/status text, not CSV; pass --out or output.csv=... for a CSV path. Default CSV: beta_barrel_staves_results.csv.\n"
            "Multi-model structures are analyzed using the first model exposed by Biopython/DSSP.\n"
            "Biological gate: pass --barrel-decisions from beta-barrel-detection, or pass --allow-ungated for explicit exploratory ungated counting.\n"
            "DSSP: pass runtime.dssp_bin_path=/path/to/mkdssp or use --check-env to verify availability.\n"
            "Hydra overrides: KEY=VALUE tokens are forwarded after CLI flags and can set nested config values."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Input structure file or directory. Required unless input.path=... is provided."
        ),
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of analysis worker processes.",
    )
    parser.add_argument(
        "--prepare-workers",
        "--prep",
        type=int,
        default=None,
        help="Number of preparation worker processes (default: follows --workers).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Write results CSV to this path. Stdout is status text, not CSV.",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="Analyze only this chain id. By default, all chains in each input structure are analyzed.",
    )
    parser.add_argument(
        "--barrel-decisions",
        default=None,
        help="Optional beta-barrel-detection CSV used to filter candidate stave counts to BARREL chains.",
    )
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="Run candidate stave counting without a beta-barrel detection gate. Use only for exploratory analysis.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check whether Python and DSSP are available, then exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    args, hydra_overrides = parser.parse_known_args(argv)
    if args.path and _looks_like_hydra_override(args.path):
        hydra_overrides = [args.path, *hydra_overrides]
        args.path = None
    args.path, hydra_overrides = _recover_positional_path(
        parser,
        args.path,
        hydra_overrides,
    )
    _reject_unknown_options(parser, hydra_overrides)

    if (
        args.path is None
        and not args.check_env
        and not _has_override(hydra_overrides, "input.path")
        and not _has_true_override(hydra_overrides, "runtime.check_env")
    ):
        parser.error("the input path is required (or pass input.path=...)")

    if (
        not args.check_env
        and not _has_true_override(hydra_overrides, "runtime.check_env")
        and not args.allow_ungated
        and not args.barrel_decisions
    ):
        parser.error(
            "beta-barrel-staves requires --barrel-decisions DETECTION.csv or explicit --allow-ungated"
        )

    try:
        cfg = build_config(hydra_overrides)
        cfg = apply_runtime_overrides(
            cfg,
            input_path=args.path,
            chain_id=args.chain,
            workers=args.workers,
            prepare_workers=args.prepare_workers,
            out_csv=args.out,
        )

        configure_thread_environment()

        if args.check_env or cfg.runtime.check_env:
            summary = runtime_summary(cfg.runtime.dssp_bin_path, require_dssp=False)
            print(f"Python: {summary['python']} ({summary['python_executable']})")
            print(f"DSSP: {summary['dssp']}")
            if summary["dssp"] == "not found":
                raise SystemExit(2)
            return

        if args.barrel_decisions:
            decisions = _load_barrel_decisions(args.barrel_decisions)
            result = run_pipeline_result(
                cfg,
                write_csv=False,
                print_summary=False,
                show_progress=sys.stderr.isatty(),
                strict_input=True,
                raise_on_all_prepare_failures=False,
            )
            rows = _apply_barrel_decisions(result.raw_rows(), decisions)
            write_results_csv(rows, cfg.output.csv_path)
            _write_gated_metadata(
                cfg=cfg,
                result=result,
                rows=rows,
                output_csv=cfg.output.csv_path,
                barrel_decisions_csv=args.barrel_decisions,
            )
            print_results_summary(
                rows,
                cfg.output.csv_path,
                summary_limit=cfg.output.summary_limit,
                write_csv=False,
                output_written=True,
            )
            if rows and all(str(row.get("result", "")).upper() == "ERROR" for row in rows):
                reason_summary = _failure_reason_summary(rows)
                if reason_summary:
                    print(f"Error: all gated stave rows failed. Reasons: {reason_summary}", file=sys.stderr)
                raise SystemExit(2)
            return

        run_pipeline_result(
            cfg,
            write_csv=True,
            print_summary=True,
            show_progress=sys.stderr.isatty(),
            strict_input=True,
            raise_on_all_prepare_failures=True,
        )
    except (
        BetaBarrelStavesReadoutError,
        FileNotFoundError,
        HydraException,
        OmegaConfBaseException,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
