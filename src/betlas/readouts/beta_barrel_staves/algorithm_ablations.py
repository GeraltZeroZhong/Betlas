#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .publication import require_publication_gold_provenance

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VARIANT_CATALOG_RESOURCE = "conf/publication/algorithm_ablations.yaml"

SUMMARY_FIELDS = [
    "variant",
    "overrides",
    "runtime_seconds",
    "gold_pass_counted_n",
    "gold_pass_missing_n",
    "exact",
    "exact_rate",
    "within_2",
    "within_2_rate",
    "mae",
    "bias",
    "small_exact_rate",
    "large_exact_rate",
    "small_mae",
    "large_mae",
    "counted_rows",
    "low_confidence_rows",
    "filtered_rows",
    "error_rows",
    "basis_counts",
    "bucket_metrics",
]


def _resource_text(resource: str) -> str:
    return (
        files("betlas.readouts.beta_barrel_staves")
        .joinpath(*resource.split("/"))
        .read_text(encoding="utf-8")
    )


def _as_variant(item: Any, *, index: int) -> tuple[str, list[str]]:
    if not isinstance(item, dict):
        raise ValueError(f"Variant catalog entry {index} must be a mapping.")
    name = str(item.get("name", "")).strip()
    if not name:
        raise ValueError(f"Variant catalog entry {index} is missing `name`.")
    overrides = item.get("overrides", [])
    if overrides is None:
        overrides = []
    if not isinstance(overrides, list) or not all(
        isinstance(override, str) for override in overrides
    ):
        raise ValueError(f"Variant catalog entry `{name}` must define string overrides.")
    return name, list(overrides)


def load_variant_catalog(
    resource: str = DEFAULT_VARIANT_CATALOG_RESOURCE,
) -> list[tuple[str, list[str]]]:
    cfg = OmegaConf.to_container(OmegaConf.create(_resource_text(resource)), resolve=True)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("variants"), list):
        raise ValueError(f"Variant catalog `{resource}` must contain a `variants` list.")

    variants = [_as_variant(item, index=index) for index, item in enumerate(cfg["variants"])]
    names = [name for name, _overrides in variants]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Variant catalog `{resource}` contains duplicate names: {', '.join(duplicates)}"
        )
    return variants


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def betlas_key(row: dict[str, str]) -> str:
    return f"{Path(row['filename']).stem.upper()}_{row['chain']}"


def format_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.000"
    return f"{numerator / denominator:.3f}"


def format_mean(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def summarize_variant(
    *,
    variant: str,
    overrides: list[str],
    runtime_seconds: float,
    betlas_csv: Path,
    gold_csv: Path,
    joined_csv: Path,
    allow_opm_derived_gold: bool = False,
    allow_internal_stress_test: bool = False,
) -> dict[str, object]:
    betlas_rows = read_csv(betlas_csv)
    gold_rows = read_csv(gold_csv)
    require_publication_gold_provenance(
        gold_rows,
        dataset=gold_csv,
        allow_opm_derived_gold=allow_opm_derived_gold,
        allow_internal_stress_test=allow_internal_stress_test,
    )
    betlas_by_key = {betlas_key(row): row for row in betlas_rows}
    result_counts = Counter(row.get("result", "") for row in betlas_rows)

    pairs: list[tuple[int, int, str]] = []
    small_errors: list[int] = []
    large_errors: list[int] = []
    joined_rows: list[dict[str, object]] = []
    basis_counts: Counter[str] = Counter()
    bucket_errors: dict[int, list[int]] = {}
    missing = 0
    for row in gold_rows:
        if row.get("evidence_level") != "gold" or row.get("qc_status") != "pass":
            continue
        expected = parse_int(row.get("strand_count_final", ""))
        if expected is None:
            continue
        key = row.get("record_id", "")
        betlas_row = betlas_by_key.get(key, {})
        observed = parse_int(betlas_row.get("strand_count", ""))
        result = betlas_row.get("result", "")
        if observed is None or result != "COUNTED":
            missing += 1
            joined_rows.append(
                {
                    "record_id": key,
                    "gold": expected,
                    "pred": "",
                    "err": "",
                    "result": result or "MISSING",
                    "confidence_basis": betlas_row.get("confidence_basis", ""),
                }
            )
            continue
        basis = betlas_row.get("confidence_basis", "")
        error = observed - expected
        pairs.append((expected, observed, basis))
        basis_counts[basis or "unknown"] += 1
        bucket_errors.setdefault(expected, []).append(error)
        if 8 <= expected <= 14:
            small_errors.append(error)
        if expected >= 16:
            large_errors.append(error)
        joined_rows.append(
            {
                "record_id": key,
                "gold": expected,
                "pred": observed,
                "err": error,
                "result": result,
                "confidence_basis": basis,
            }
        )
    write_csv(
        joined_csv,
        joined_rows,
        ["record_id", "gold", "pred", "err", "result", "confidence_basis"],
    )

    n = len(pairs)
    exact = sum(1 for expected, observed, _basis in pairs if expected == observed)
    within_2 = sum(1 for expected, observed, _basis in pairs if abs(expected - observed) <= 2)
    mae = None if not pairs else sum(abs(expected - observed) for expected, observed, _basis in pairs) / n
    bias = None if not pairs else sum(observed - expected for expected, observed, _basis in pairs) / n
    small_exact = sum(1 for error in small_errors if error == 0)
    large_exact = sum(1 for error in large_errors if error == 0)
    small_mae = None if not small_errors else sum(abs(error) for error in small_errors) / len(small_errors)
    large_mae = None if not large_errors else sum(abs(error) for error in large_errors) / len(large_errors)

    bucket_parts = []
    for bucket in sorted(bucket_errors):
        errors = bucket_errors[bucket]
        bucket_n = len(errors)
        bucket_exact = sum(1 for error in errors if error == 0)
        bucket_mae = sum(abs(error) for error in errors) / bucket_n
        bucket_bias = sum(errors) / bucket_n
        bucket_parts.append(
            f"{bucket}:n={bucket_n},exact={bucket_exact},mae={bucket_mae:.2f},bias={bucket_bias:+.2f}"
        )

    return {
        "variant": variant,
        "overrides": " ".join(overrides),
        "runtime_seconds": f"{runtime_seconds:.1f}",
        "gold_pass_counted_n": n,
        "gold_pass_missing_n": missing,
        "exact": exact,
        "exact_rate": format_rate(exact, n),
        "within_2": within_2,
        "within_2_rate": format_rate(within_2, n),
        "mae": format_mean(mae),
        "bias": format_mean(bias),
        "small_exact_rate": format_rate(small_exact, len(small_errors)),
        "large_exact_rate": format_rate(large_exact, len(large_errors)),
        "small_mae": format_mean(small_mae),
        "large_mae": format_mean(large_mae),
        "counted_rows": result_counts.get("COUNTED", 0),
        "low_confidence_rows": result_counts.get("LOW_CONFIDENCE", 0),
        "filtered_rows": result_counts.get("FILTERED_OUT", 0),
        "error_rows": result_counts.get("ERROR", 0),
        "basis_counts": "; ".join(f"{key}={value}" for key, value in basis_counts.most_common()),
        "bucket_metrics": " | ".join(bucket_parts),
    }


def run_command(command: list[str], *, env: dict[str, str], reuse: bool, output_path: Path) -> None:
    if reuse and output_path.exists():
        print(f"[reuse] {display_path(output_path)}", flush=True)
        return
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def selected_variants(names: list[str] | None) -> list[tuple[str, list[str]]]:
    variants = load_variant_catalog()
    if not names:
        return variants
    wanted = set(names)
    selected = [item for item in variants if item[0] in wanted]
    missing = sorted(wanted - {item[0] for item in selected})
    if missing:
        raise SystemExit(f"Unknown ablation variant(s): {', '.join(missing)}")
    return selected


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "variant",
        "n",
        "exact",
        "within_2",
        "mae",
        "large_mae",
        "runtime_s",
        "top_basis",
    ]
    lines = [
        "# Strand Count Ablation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        top_basis = str(row["basis_counts"]).split("; ")[0] if row["basis_counts"] else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["gold_pass_counted_n"]),
                    f"{row['exact']} ({row['exact_rate']})",
                    f"{row['within_2']} ({row['within_2_rate']})",
                    str(row["mae"]),
                    str(row["large_mae"]),
                    str(row["runtime_seconds"]),
                    top_basis,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="runs/readouts/beta_barrel_staves/algorithm_ablations",
    )
    parser.add_argument("--workers", default="7")
    parser.add_argument("--prepare-workers", default="7")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--variant", action="append", help="Run only a named variant; repeatable.")
    parser.add_argument(
        "--allow-opm-derived-gold",
        action="store_true",
        help="Allow legacy OPM-derived gold/pass labels without manual chain-count provenance for compatibility checks.",
    )
    parser.add_argument(
        "--allow-internal-stress-test",
        action="store_true",
        help="Allow AFDB/non-release stress-test rows for compatibility checks.",
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    gold_csv = (
        REPO_ROOT
        / "data/readouts/beta_barrel_staves/processed/beta_barrel_publication_gold.csv"
    )
    try:
        require_publication_gold_provenance(
            read_csv(gold_csv),
            dataset=gold_csv,
            allow_opm_derived_gold=bool(args.allow_opm_derived_gold),
            allow_internal_stress_test=bool(args.allow_internal_stress_test),
        )
    except ValueError as exc:
        parser.error(str(exc))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    rows: list[dict[str, object]] = []

    for variant, overrides in selected_variants(args.variant):
        variant_dir = out_dir / variant
        betlas_csv = variant_dir / "beta_barrel_staves.csv"
        joined_csv = variant_dir / "gold_pass_joined.csv"
        reused = args.reuse_existing and betlas_csv.exists()
        started = time.time()
        barrel_gate_override = "barrel_gate.enabled=false"
        run_command(
            [
                sys.executable,
                "-m",
                "betlas",
                "readout",
                "beta-barrel-staves",
                "data/raw/pdb",
                "--workers",
                args.workers,
                "--prepare-workers",
                args.prepare_workers,
                "--out",
                str(betlas_csv),
                barrel_gate_override,
                "analyzer.count.min_confidence=0.0",
                "runtime.fail_on_dssp_error=false",
                *overrides,
            ],
            env=env,
            reuse=args.reuse_existing,
            output_path=betlas_csv,
        )
        runtime_seconds = 0.0 if reused else time.time() - started
        rows.append(
            summarize_variant(
                variant=variant,
                overrides=overrides,
                runtime_seconds=runtime_seconds,
                betlas_csv=betlas_csv,
                gold_csv=gold_csv,
                joined_csv=joined_csv,
                allow_opm_derived_gold=bool(args.allow_opm_derived_gold),
                allow_internal_stress_test=bool(args.allow_internal_stress_test),
            )
        )

    summary_csv = out_dir / "summary.csv"
    write_csv(summary_csv, rows, SUMMARY_FIELDS)
    write_markdown(out_dir / "summary.md", rows)
    print(f"Wrote {display_path(summary_csv)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
