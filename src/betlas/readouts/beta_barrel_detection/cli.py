from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException

if __package__ in {None, ""}:  # pragma: no cover - path execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from betlas.readouts.beta_barrel_detection.bootstrap import configure_thread_environment
    from betlas.readouts.beta_barrel_detection.config import build_config
    from betlas.readouts.beta_barrel_detection.exceptions import BetlasBetaError
    from betlas.readouts.beta_barrel_detection.runtime import (
        dssp_requirement_message,
        runtime_summary,
    )
else:
    from .bootstrap import configure_thread_environment
    from .config import build_config
    from .exceptions import BetlasBetaError
    from .runtime import dssp_requirement_message, runtime_summary


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="betlas readout beta-barrel-detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Detect beta-barrel-like protein chains from PDB/mmCIF inputs and write a CSV "
            "summary. Advanced KEY=VALUE arguments are treated as Hydra overrides."
        ),
        epilog=(
            "Examples:\n"
            "  betlas readout beta-barrel-detection structure.cif --out runs/detection.csv\n"
            "  betlas readout beta-barrel-detection structure.cif --chain A --out runs/detection_A.csv\n"
            "  betlas readout beta-barrel-detection structures/ --workers 8 --prep 4 --out runs/detection.csv\n"
            "  betlas readout beta-barrel-detection input.path=structures/ output.csv=runs/detection.csv runtime.dssp_bin_path=mkdssp\n"
            "  betlas readout beta-barrel-detection --check-env\n\n"
            "Output CSV: one row per analyzed chain with beta-barrel-like geometry decision, uncalibrated heuristic score, slice, geometry, and runtime-status columns.\n"
            "Stdout is progress/status text, not CSV; pass --out or output.csv=... for a CSV path. Default CSV: beta_barrel_detection_results.csv.\n"
            "Multi-model structures are analyzed using the first model exposed by Biopython/DSSP.\n"
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

    try:
        configure_thread_environment()

        cfg = build_config(hydra_overrides)
        if __package__ in {None, ""}:  # pragma: no cover - path execution convenience
            from betlas.readouts.beta_barrel_detection.pipeline import (
                apply_runtime_overrides,
                run_pipeline_result,
            )
        else:
            from .pipeline import (
                apply_runtime_overrides,
                run_pipeline_result,
            )

        cfg = apply_runtime_overrides(
            cfg,
            input_path=args.path,
            chain_id=args.chain,
            workers=args.workers,
            prepare_workers=args.prepare_workers,
            out_csv=args.out,
        )

        if args.check_env or cfg.runtime.check_env:
            summary = runtime_summary(
                cfg.runtime.dssp_bin_path,
                require_dssp=bool(cfg.runtime.dssp_bin_path),
            )
            print(f"Python: {summary['python']} ({summary['python_executable']})")
            print(f"DSSP: {summary['dssp']}")
            if summary["dssp"] == "not found":
                print(dssp_requirement_message(), file=sys.stderr)
                raise SystemExit(2)
            return

        run_pipeline_result(
            cfg,
            write_csv=True,
            print_summary=True,
            show_progress=sys.stderr.isatty(),
            strict_input=True,
        )
    except (
        BetlasBetaError,
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
