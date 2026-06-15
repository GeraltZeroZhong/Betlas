#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
SRC_DIR = REPO / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betlas.readouts.beta_barrel_staves.external_baselines import (  # noqa: E402
    build_external_baseline_manifest,
    write_json,
)
from betlas.readouts.beta_barrel_staves.publication import (  # noqa: E402
    require_publication_gold_provenance,
)

DEFAULT_DATASET = (
    REPO
    / "data"
    / "readouts"
    / "beta_barrel_staves"
    / "processed"
    / "beta_barrel_publication_gold.csv"
)
DEFAULT_EVIDENCE_LEVELS = ("gold",)
DEFAULT_QC_STATUSES = ("pass",)
PROFILE_DIAGNOSTIC_SCOPE = "diagnostic_self_fasta"


MODEL_KEYS = {
    "Internal: direct geometry count": "beta_barrel_staves_geometry",
    "PolarBearal3": "polarbearal3",
    "TMbed": "tmbed",
    "PRED-TMBB2 HMM (JUCHMME)": "pred_tmbb2_hmm_juchmme",
    "PROFtmb": "proftmb",
    "PRED-TMBB2+HNN (diagnostic)": "pred_tmbb2_hnn_diagnostic",
    "BETAWARE (diagnostic)": "betaware_diagnostic",
}


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"\\0", "\0", "NA", "nan"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    values = [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]
    return values or list(default)


def load_gold_rows(
    dataset: Path,
    *,
    evidence_levels: set[str],
    qc_statuses: set[str],
    allow_opm_derived_gold: bool = False,
    allow_internal_stress_test: bool = False,
) -> list[dict[str, object]]:
    with dataset.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("evidence_level") in evidence_levels and row.get("qc_status") in qc_statuses
        ]
    require_publication_gold_provenance(
        rows,
        dataset=dataset,
        allow_opm_derived_gold=allow_opm_derived_gold,
        allow_internal_stress_test=allow_internal_stress_test,
    )
    rows.sort(key=lambda row: row["record_id"])
    for row in rows:
        row["gold_strand_count"] = parse_int(row.get("strand_count_final"))
        if row["gold_strand_count"] is None:
            raise ValueError(f"Missing gold strand_count_final for {row['record_id']}")
    return rows


def count_segments(labels: str, segment_chars: set[str]) -> int:
    count = 0
    in_segment = False
    for char in labels:
        is_segment = char in segment_chars
        if is_segment and not in_segment:
            count += 1
        in_segment = is_segment
    return count


def empty_predictions(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(row["record_id"]): {"pred": None, "status": "missing", "raw": ""}
        for row in rows
    }


def parse_beta_barrel_staves(
    rows: list[dict[str, object]],
    path: Path,
) -> dict[str, dict[str, object]]:
    predictions = empty_predictions(rows)
    if not path.exists():
        return predictions

    by_chain: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = Path(row.get("filename", "")).stem.upper()
            chain = row.get("chain", "")
            if pdb_id and chain:
                by_chain[(pdb_id, chain)] = row

    for gold in rows:
        pdb_id = str(gold["pdb_id"]).upper()
        structure_id = str(gold.get("structure_id", "")).upper()
        structure_key = structure_id or pdb_id
        record_id = str(gold["record_id"])
        record_key = record_id.upper()
        auth_chain = str(gold.get("auth_chain_id", ""))
        asym_id = str(gold.get("asym_id", ""))
        row = by_chain.get((structure_key, auth_chain)) or by_chain.get((structure_key, asym_id))
        row = row or by_chain.get((pdb_id, auth_chain)) or by_chain.get((pdb_id, asym_id))
        row = row or by_chain.get((record_key, auth_chain)) or by_chain.get((record_key, asym_id))
        if not row:
            continue
        status = row.get("result") or "missing"
        pred = parse_int(row.get("strand_count"))
        if status not in {"COUNTED", "LOW_CONFIDENCE"}:
            pred = None
        predictions[record_id] = {"pred": pred, "status": status, "raw": row.get("reason", "")}
    return predictions


def parse_tabular_counts(
    rows: list[dict[str, object]],
    path: Path,
    record_column: str,
    count_column: str,
    status_column: str | None = None,
    delimiter: str = "\t",
    suffix: str = "",
    numeric_statuses: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    predictions = empty_predictions(rows)
    if not path.exists():
        return predictions

    gold_ids = set(predictions)
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            record_id = (row.get(record_column) or "").strip()
            if suffix and record_id.endswith(suffix):
                record_id = record_id[: -len(suffix)]
            if record_id not in gold_ids:
                continue
            status = (row.get(status_column) if status_column else "ok") or "ok"
            pred = parse_int(row.get(count_column))
            if numeric_statuses is not None and status not in numeric_statuses:
                pred = None
            predictions[record_id] = {"pred": pred, "status": status, "raw": ""}
    return predictions


def parse_tmbed(rows: list[dict[str, object]], path: Path) -> dict[str, dict[str, object]]:
    predictions = empty_predictions(rows)
    if not path.exists():
        return predictions

    gold_ids = set(predictions)
    current_id: str | None = None
    body: list[str] = []

    def flush() -> None:
        if current_id is None or current_id not in gold_ids:
            return
        labels = body[-1].strip() if body else ""
        pred = count_segments(labels, {"B", "b"}) if labels else None
        predictions[current_id] = {"pred": pred, "status": "ok" if labels else "failed", "raw": labels}

    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].split()[0]
                body = []
            else:
                body.append(line)
        flush()
    return predictions


def parse_juchmme(rows: list[dict[str, object]], path: Path) -> dict[str, dict[str, object]]:
    predictions = empty_predictions(rows)
    if not path.exists():
        return predictions

    gold_ids = set(predictions)
    current_id: str | None = None
    with path.open(errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("ID: >"):
                current_id = line.split(">", 1)[1].split()[0]
                continue
            if current_id in gold_ids and line.startswith("VP:"):
                labels = line.split(":", 1)[1].strip()
                predictions[current_id] = {
                    "pred": count_segments(labels, {"M"}),
                    "status": "ok",
                    "raw": labels,
                }
    return predictions


def summarize_model(
    model: str,
    rows: list[dict[str, object]],
    predictions: dict[str, dict[str, object]],
    note: str,
) -> dict[str, object]:
    gold_n = len(rows)
    status_counts = Counter()
    valid_errors: list[int] = []
    exact_n = 0
    within2_n = 0
    attempted = 0

    for row in rows:
        record_id = str(row["record_id"])
        gold = int(row["gold_strand_count"])
        prediction = predictions.get(record_id, {"pred": None, "status": "missing"})
        status = str(prediction.get("status") or "missing")
        pred = prediction.get("pred")
        status_counts[status] += 1
        if status != "missing":
            attempted += 1
        if pred is None:
            continue
        error = int(pred) - gold
        valid_errors.append(error)
        if error == 0:
            exact_n += 1
        if abs(error) <= 2:
            within2_n += 1

    valid_n = len(valid_errors)
    missing = status_counts.get("missing", 0)
    statuses = "; ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items()) if status != "missing"
    )
    if not statuses:
        statuses = "none"
    return {
        "model": model,
        "gold_n": gold_n,
        "attempted": attempted,
        "valid_n": valid_n,
        "exact_n": exact_n,
        "exact_all": exact_n / gold_n if gold_n else 0.0,
        "exact_valid": exact_n / valid_n if valid_n else 0.0,
        "within2_n": within2_n,
        "within2_valid": within2_n / valid_n if valid_n else 0.0,
        "mae": sum(abs(error) for error in valid_errors) / valid_n if valid_n else 0.0,
        "bias": sum(valid_errors) / valid_n if valid_n else 0.0,
        "statuses": statuses,
        "missing": missing,
        "note": note,
    }


def format_percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def format_float(value: object) -> str:
    return f"{float(value):.2f}"


def format_runtime(seconds: float | None) -> str:
    if seconds is None:
        return ""
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def load_runtime_rows(run_dir: Path) -> dict[str, float]:
    path = run_dir / "runtime_breakdown.tsv"
    if not path.exists():
        return {}
    rows: dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                rows[row["component"]] = float(row["runtime_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def load_run_manifest(run_dir: Path) -> dict[str, object]:
    manifest = run_dir / "gold_run_manifest.json"
    if not manifest.exists():
        return {}
    return json.loads(manifest.read_text(encoding="utf-8"))


def profile_database_scope_for_run(run_dir: Path) -> str:
    run_manifest = load_run_manifest(run_dir)
    scope = str(run_manifest.get("profile_database_scope", ""))
    if scope:
        return scope
    if (run_dir / "profiles" / "blastdb" / "gold.pin").exists():
        return PROFILE_DIAGNOSTIC_SCOPE
    return ""


def apply_profile_scope_annotations(
    summaries: list[dict[str, object]],
    profile_database_scope: str,
) -> None:
    if profile_database_scope != PROFILE_DIAGNOSTIC_SCOPE:
        return
    for summary in summaries:
        if summary.get("model") == "PROFtmb":
            summary["model"] = "PROFtmb (diagnostic)"


def add_runtime_fields(summaries: list[dict[str, object]], run_dir: Path) -> None:
    runtime_rows = load_runtime_rows(run_dir)
    profile_seconds = runtime_rows.get("psiblast_profiles", 0.0)
    model_runtime = {
        "PolarBearal3": (runtime_rows.get("polarbearal3"), "PolarBearal3 wall runtime"),
        "TMbed": (runtime_rows.get("tmbed_total"), "TMbed embedding + prediction runtime"),
        "PRED-TMBB2 HMM (JUCHMME)": (runtime_rows.get("pred_tmbb2_hmm"), "JUCHMME HMM runtime"),
        "PROFtmb": (
            profile_seconds + runtime_rows["proftmb"] if "proftmb" in runtime_rows else None,
            "PSI-BLAST profile runtime + PROFtmb runtime",
        ),
        "PRED-TMBB2+HNN (diagnostic)": (runtime_rows.get("pred_tmbb2_hnn"), "JUCHMME HNN runtime"),
        "BETAWARE (diagnostic)": (
            profile_seconds + runtime_rows["betaware"] if "betaware" in runtime_rows else None,
            "PSI-BLAST profile runtime + BETAWARE runtime",
        ),
    }
    for summary in summaries:
        runtime, basis = model_runtime.get(str(summary["model"]), (None, "not recorded"))
        summary["runtime_seconds"] = "" if runtime is None else f"{runtime:.3f}"
        summary["runtime_hms"] = format_runtime(runtime)
        summary["runtime_basis"] = basis


def write_summary_tsv(path: Path, summaries: list[dict[str, object]]) -> None:
    fields = [
        "model",
        "gold_n",
        "attempted",
        "valid_n",
        "exact_n",
        "exact_all",
        "exact_valid",
        "within2_n",
        "within2_valid",
        "mae",
        "bias",
        "statuses",
        "missing",
        "note",
        "runtime_seconds",
        "runtime_hms",
        "runtime_basis",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            for key in ("exact_all", "exact_valid", "within2_valid", "mae", "bias"):
                row[key] = f"{float(row[key]):.4f}"
            writer.writerow(row)


def write_per_record_tsv(
    path: Path,
    rows: list[dict[str, object]],
    model_predictions: dict[str, dict[str, dict[str, object]]],
) -> None:
    fields = ["record_id", "gold_strand_count"]
    for model in MODEL_KEYS:
        key = MODEL_KEYS[model]
        fields.extend([f"{key}_pred", f"{key}_status"])

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for gold in rows:
            record_id = str(gold["record_id"])
            out: dict[str, object] = {
                "record_id": record_id,
                "gold_strand_count": gold["gold_strand_count"],
            }
            for model in MODEL_KEYS:
                key = MODEL_KEYS[model]
                prediction = model_predictions[model].get(record_id, {"pred": None, "status": "missing"})
                out[f"{key}_pred"] = "" if prediction.get("pred") is None else prediction.get("pred")
                out[f"{key}_status"] = prediction.get("status", "missing")
            writer.writerow(out)


def write_markdown(
    path: Path,
    summaries: list[dict[str, object]],
    dataset: Path,
    run_dir: Path,
    evidence_levels: set[str],
    qc_statuses: set[str],
) -> None:
    main = [summary for summary in summaries if "diagnostic" not in str(summary["model"])]
    diagnostics = [summary for summary in summaries if "diagnostic" in str(summary["model"])]

    def table_lines(items: list[dict[str, object]]) -> list[str]:
        lines = [
            "| Model | Attempted | Numeric predictions | Exact / all | Exact / valid | Within +/-2 / valid | MAE | Bias | Runtime | Statuses / note |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for summary in items:
            lines.append(
                "| {model} | {attempted}/{gold_n} | {valid_n}/{gold_n} | "
                "{exact_n}/{gold_n} ({exact_all}) | {exact_n}/{valid_n} ({exact_valid}) | "
                "{within2_n}/{valid_n} ({within2_valid}) | {mae} | {bias} | {runtime} | "
                "{statuses}; {note} |".format(
                    model=summary["model"],
                    attempted=summary["attempted"],
                    gold_n=summary["gold_n"],
                    valid_n=summary["valid_n"],
                    exact_n=summary["exact_n"],
                    exact_all=format_percent(summary["exact_all"]),
                    exact_valid=format_percent(summary["exact_valid"]),
                    within2_n=summary["within2_n"],
                    within2_valid=format_percent(summary["within2_valid"]),
                    mae=format_float(summary["mae"]),
                    bias=format_float(summary["bias"]),
                    runtime=summary.get("runtime_hms", ""),
                    statuses=summary["statuses"],
                    note=summary["note"],
                )
            )
        return lines

    lines = [
        "# Current Gold Dataset Model Summary",
        "",
        f"- Dataset: `{dataset}` filtered to `evidence_level={','.join(sorted(evidence_levels))}`, "
        f"`qc_status={','.join(sorted(qc_statuses))}`.",
        f"- Run directory: `{run_dir}`.",
        "- Evaluation unit: chain-level records; gold label = `strand_count_final`.",
        "- Failed/no-count outputs are misses in `Exact / all`; `Exact / valid` only uses numeric predictions.",
        "",
        "## Main Comparison",
        "",
        *table_lines(main),
        "",
        "## Diagnostic Runs",
        "",
        *table_lines(diagnostics),
        "",
        f"- Machine-readable summary: `{path.with_suffix('.tsv')}`",
        f"- Per-record predictions: `{path.with_name('gold_model_per_record.tsv')}`",
        "",
    ]
    path.write_text("\n".join(lines))


def build_predictions(
    rows: list[dict[str, object]],
    run_dir: Path,
    betlas_csv: Path | None,
    *,
    profile_database_scope: str = "",
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, str]]:
    if betlas_csv is None:
        betlas_csv = run_dir / "beta_barrel_staves" / "beta_barrel_staves_gold.csv"

    predictions = {
        "Internal: direct geometry count": parse_beta_barrel_staves(rows, betlas_csv),
        "PolarBearal3": parse_tabular_counts(
            rows,
            run_dir / "polar_bearal3" / "polarbearal3_gold_summary.tsv",
            "record_id",
            "strand_count",
            "status",
            numeric_statuses={"ok", "ok_reused"},
        ),
        "TMbed": parse_tmbed(rows, run_dir / "tmbed" / "gold.pred"),
        "PRED-TMBB2 HMM (JUCHMME)": parse_juchmme(
            rows, run_dir / "juchmme" / "pred_tmbb2_hmm_gold.out"
        ),
        "PROFtmb": parse_tabular_counts(
            rows,
            run_dir / "proftmb" / "gold_proftmb_tabular.txt",
            "SeqID",
            "#_Predicted_Strands",
            suffix=".Q",
        ),
        "PRED-TMBB2+HNN (diagnostic)": parse_juchmme(
            rows, run_dir / "juchmme" / "pred_tmbb2_hnn_gold.out"
        ),
        "BETAWARE (diagnostic)": parse_tabular_counts(
            rows,
            run_dir / "betaware" / "betaware_gold_summary.tsv",
            "record_id",
            "strand_count",
            "status",
            numeric_statuses={"ok"},
        ),
    }
    proftmb_note = "profile-based HMM; count = #_Predicted_Strands"
    if profile_database_scope == PROFILE_DIAGNOSTIC_SCOPE:
        proftmb_note = (
            "diagnostic only; PSI-BLAST profiles were built against the selected "
            "gold FASTA itself, not an independent sequence database"
        )
    notes = {
        "Internal: direct geometry count": "diagnostic rule-based geometry output; numeric = COUNTED + LOW_CONFIDENCE",
        "PolarBearal3": "SluskyLab/PolarBearal3",
        "TMbed": "count = contiguous B/b beta-barrel segments in TMbed labels",
        "PRED-TMBB2 HMM (JUCHMME)": "JUCHMME 1.0.6; count = contiguous M segments in VP labels",
        "PROFtmb": proftmb_note,
        "PRED-TMBB2+HNN (diagnostic)": "excluded from main comparison; local FASTA/HNN path fails sanity checks",
        "BETAWARE (diagnostic)": "excluded from main comparison; local gold-only profile regime is not a usable BETAWARE baseline",
    }
    return predictions, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a current gold baseline run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--betlas-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--evidence-level", action="append", default=None)
    parser.add_argument("--qc-status", action="append", default=None)
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

    run_dir = args.run_dir.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    out_dir = (args.out_dir or run_dir / "summary").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_levels = set(args.evidence_level or env_list("GOLD_EVIDENCE_LEVELS", DEFAULT_EVIDENCE_LEVELS))
    qc_statuses = set(args.qc_status or env_list("GOLD_QC_STATUSES", DEFAULT_QC_STATUSES))
    try:
        rows = load_gold_rows(
            dataset,
            evidence_levels=evidence_levels,
            qc_statuses=qc_statuses,
            allow_opm_derived_gold=bool(args.allow_opm_derived_gold),
            allow_internal_stress_test=bool(args.allow_internal_stress_test),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    profile_database_scope = profile_database_scope_for_run(run_dir)
    predictions, notes = build_predictions(
        rows,
        run_dir,
        args.betlas_csv,
        profile_database_scope=profile_database_scope,
    )
    summaries = [
        summarize_model(model, rows, predictions[model], notes[model])
        for model in MODEL_KEYS
    ]
    add_runtime_fields(summaries, run_dir)
    apply_profile_scope_annotations(summaries, profile_database_scope)

    summary_tsv = out_dir / "gold_model_summary.tsv"
    per_record_tsv = out_dir / "gold_model_per_record.tsv"
    summary_md = out_dir / "gold_model_summary.md"
    write_summary_tsv(summary_tsv, summaries)
    write_per_record_tsv(per_record_tsv, rows, predictions)
    write_markdown(summary_md, summaries, dataset, run_dir, evidence_levels, qc_statuses)
    summary_manifest = out_dir / "gold_model_summary_manifest.json"
    write_json(
        summary_manifest,
        build_external_baseline_manifest(
            run_name="beta_barrel_staves_gold_external_summary",
            status="ok",
            command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            inputs={
                "dataset": dataset,
                "gold_run_manifest": run_dir / "gold_run_manifest.json",
                "external_baseline_manifest": run_dir / "gold_external_baseline_manifest.json",
                "betlas_csv": args.betlas_csv
                or run_dir / "beta_barrel_staves" / "beta_barrel_staves_gold.csv",
                "polarbearal3_summary": run_dir / "polar_bearal3" / "polarbearal3_gold_summary.tsv",
                "tmbed_predictions": run_dir / "tmbed" / "gold.pred",
                "pred_tmbb2_hmm": run_dir / "juchmme" / "pred_tmbb2_hmm_gold.out",
                "pred_tmbb2_hnn": run_dir / "juchmme" / "pred_tmbb2_hnn_gold.out",
                "proftmb_tabular": run_dir / "proftmb" / "gold_proftmb_tabular.txt",
                "betaware_summary": run_dir / "betaware" / "betaware_gold_summary.tsv",
                "runtime_breakdown": run_dir / "runtime_breakdown.tsv",
            },
            outputs={
                "summary_tsv": summary_tsv,
                "per_record_tsv": per_record_tsv,
                "summary_markdown": summary_md,
            },
            env=os.environ,
            parameters={
                "evidence_levels": sorted(evidence_levels),
                "qc_statuses": sorted(qc_statuses),
                "profile_database_scope": profile_database_scope,
                "allow_opm_derived_gold": bool(args.allow_opm_derived_gold),
                "allow_internal_stress_test": bool(args.allow_internal_stress_test),
            },
            metrics={"gold_rows": len(rows), "model_count": len(summaries)},
        ),
    )

    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
