from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...provenance import build_run_manifest, write_json

DEFAULT_SCHEMA_PATH = (
    Path("data")
    / "readouts"
    / "beta_barrel_staves"
    / "schemas"
    / "beta_barrel_strand_dataset.schema.json"
)

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    row_number: int
    record_id: str
    field: str
    message: str


def _read_schema_required(schema_path: str | Path) -> list[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return [str(field) for field in schema.get("required", [])]


def _resolve_artifact_path(value: str, *, csv_path: Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [repo_root / path, csv_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_beta_barrel_staves_dataset(
    csv_path: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    csv_file = Path(csv_path)
    root = Path(repo_root)
    required = _read_schema_required(schema_path)
    issues: list[DatasetIssue] = []
    row_count = 0

    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        for field in required:
            if field not in columns:
                issues.append(
                    DatasetIssue(
                        severity="error",
                        row_number=0,
                        record_id="",
                        field=field,
                        message="required column is absent",
                    )
                )

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            record_id = str(row.get("record_id") or "").strip()
            for field in required:
                if field in columns and str(row.get(field) or "").strip() == "":
                    issues.append(
                        DatasetIssue(
                            severity="error",
                            row_number=row_number,
                            record_id=record_id,
                            field=field,
                            message="required value is empty",
                        )
                    )

            for field in ("raw_structure_sha256", "source_structure_sha256", "input_mmcif_sha256"):
                value = str(row.get(field) or "").strip()
                if value and not SHA256_RE.match(value):
                    issues.append(
                        DatasetIssue(
                            severity="error",
                            row_number=row_number,
                            record_id=record_id,
                            field=field,
                            message="sha256 must be 64 hexadecimal characters",
                        )
                    )

            manifest = str(row.get("audit_manifest") or "").strip()
            if manifest:
                resolved = _resolve_artifact_path(manifest, csv_path=csv_file, repo_root=root)
                if not resolved.exists():
                    issues.append(
                        DatasetIssue(
                            severity="warning",
                            row_number=row_number,
                            record_id=record_id,
                            field="audit_manifest",
                            message=f"manifest path does not exist: {manifest}",
                        )
                    )

    severity_counts: dict[str, int] = {}
    for issue in issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    return {
        "schema": "betlas.validation.beta-barrel-staves-dataset",
        "csv_path": str(csv_file),
        "schema_path": str(schema_path),
        "row_count": row_count,
        "issue_count": len(issues),
        "severity_counts": severity_counts,
        "issues": [asdict(issue) for issue in issues],
    }


def write_beta_barrel_staves_validation_report(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    repo_root: str | Path = ".",
) -> Path:
    report = validate_beta_barrel_staves_dataset(csv_path, schema_path=schema_path, repo_root=repo_root)
    manifest = build_run_manifest(
        command="betlas validate-staves-data",
        parameters={
            "csv": str(csv_path),
            "schema": str(schema_path),
            "repo_root": str(repo_root),
        },
        inputs={"csv": csv_path, "schema": schema_path},
        outputs={"validation_report": output_path},
        metrics={
            "rows": int(report["row_count"]),
            "issues": int(report["issue_count"]),
            **{f"{severity}_issues": count for severity, count in report["severity_counts"].items()},
        },
        extra={"validation": report},
    )
    return write_json(output_path, manifest)
