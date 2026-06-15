from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ...provenance import build_run_manifest, write_json

INTERNAL_STRESS_TEST_SCOPE = "internal_stress_test_not_publication_final"
OPM_DERIVED_STRAND_COUNT_SOURCES = frozenset(
    {
        "OPM family n",
        "OPM family n (membership-supported)",
        "OPM family n (assembly-level)",
    }
)
MANUAL_CHAIN_COUNT_SOURCES = frozenset(
    {
        "manual_chain_review",
        "manual_chain_count",
        "manual_seed",
        "manual_annotation",
    }
)


def _parse_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        return None
    return int(numeric)


def _record_id(row: Mapping[str, object]) -> str:
    return str(row.get("record_id") or row.get("pdb_id") or "<unknown>").strip()


def has_manual_chain_count_provenance(row: Mapping[str, object]) -> bool:
    if _parse_int(row.get("manual_expected_strand_count")) is not None:
        return True
    return any(
        str(row.get(field, "") or "").strip() in MANUAL_CHAIN_COUNT_SOURCES
        for field in ("strand_count_source", "manual_expected_source")
    )


def is_publication_gold_pass_row(row: Mapping[str, object]) -> bool:
    return row.get("evidence_level") == "gold" and row.get("qc_status") == "pass"


def is_opm_derived_gold_without_manual_count(row: Mapping[str, object]) -> bool:
    source = str(row.get("strand_count_source", "")).strip()
    return (
        is_publication_gold_pass_row(row)
        and source in OPM_DERIVED_STRAND_COUNT_SOURCES
        and not has_manual_chain_count_provenance(row)
    )


def is_internal_stress_test_row(row: Mapping[str, object]) -> bool:
    publication_eligible = str(row.get("publication_eligible", "") or "").strip().lower()
    record_id = str(row.get("record_id", "") or "").strip().upper()
    structure_method = str(row.get("structure_method", "") or "").strip().lower()
    return (
        str(row.get("benchmark_scope", "")).strip() == INTERNAL_STRESS_TEST_SCOPE
        or publication_eligible in {"0", "false", "no"}
        or record_id.startswith("AFDBV6_")
        or structure_method == "alphafolddb v6"
    )


def publication_guard_violations(
    rows: Iterable[Mapping[str, object]],
    *,
    allow_opm_derived_gold: bool = False,
    allow_internal_stress_test: bool = False,
) -> dict[str, list[str]]:
    internal_ids: list[str] = []
    opm_ids: list[str] = []
    for row in rows:
        if is_internal_stress_test_row(row) and not allow_internal_stress_test:
            internal_ids.append(_record_id(row))
        if is_opm_derived_gold_without_manual_count(row) and not allow_opm_derived_gold:
            opm_ids.append(_record_id(row))
    return {
        "internal_stress_test_rows": internal_ids,
        "opm_derived_gold_without_manual_count": opm_ids,
    }


def is_publication_eligible_gold_row(row: Mapping[str, object]) -> bool:
    return (
        is_publication_gold_pass_row(row)
        and not is_internal_stress_test_row(row)
        and not is_opm_derived_gold_without_manual_count(row)
    )


def publication_eligible_gold_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if is_publication_eligible_gold_row(row)]


def read_table_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_table_rows(path: str | Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output


def write_publication_gold_dataset(
    source_csv: str | Path,
    output_csv: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    rows = read_table_rows(source_csv)
    fieldnames = list(rows[0]) if rows else []
    selected = publication_eligible_gold_rows(rows)
    output = write_table_rows(output_csv, selected, fieldnames)
    require_publication_gold_provenance(selected, dataset=output)
    if manifest_path is not None:
        write_json(
            manifest_path,
            build_run_manifest(
                command="betlas curate-staves-publication-set",
                parameters={"source_csv": str(source_csv), "output_csv": str(output_csv)},
                inputs={"source_csv": source_csv},
                outputs={"publication_gold_csv": output},
                metrics={"source_rows": len(rows), "publication_gold_rows": len(selected)},
                extra={
                    "selection_policy": (
                        "gold/pass rows only; excludes non-release stress-test rows and OPM-derived "
                        "gold rows without manual chain-count provenance"
                    ),
                    "source_guard_violations": {
                        key: len(values) for key, values in publication_guard_violations(rows).items()
                    },
                },
            ),
        )
    return output


def require_publication_gold_provenance(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str | Path = "",
    allow_opm_derived_gold: bool = False,
    allow_internal_stress_test: bool = False,
) -> None:
    row_list = list(rows)
    violations = publication_guard_violations(
        row_list,
        allow_opm_derived_gold=allow_opm_derived_gold,
        allow_internal_stress_test=allow_internal_stress_test,
    )
    messages: list[str] = []
    if violations["internal_stress_test_rows"]:
        sample = ", ".join(violations["internal_stress_test_rows"][:10])
        messages.append(
            "dataset contains AFDB/non-release stress-test rows "
            f"({len(violations['internal_stress_test_rows'])}; sample: {sample})"
        )
    if violations["opm_derived_gold_without_manual_count"]:
        sample = ", ".join(violations["opm_derived_gold_without_manual_count"][:10])
        messages.append(
            "publication gold/pass rows include OPM-derived labels without manual "
            "chain-count provenance "
            f"({len(violations['opm_derived_gold_without_manual_count'])}; sample: {sample})"
        )
    if not messages:
        return

    prefix = f"{dataset}: " if dataset else ""
    raise ValueError(
        prefix
        + "; ".join(messages)
        + ". Rebuild the dataset with the current evidence policy or pass the explicit "
        "compatibility override only for non-release validation runs."
    )
