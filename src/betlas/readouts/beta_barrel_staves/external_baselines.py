from __future__ import annotations

import json
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...provenance import file_state, runtime_state, utc_now_iso, write_json

EXTERNAL_BASELINE_MANIFEST_SCHEMA = "betlas.beta-barrel-staves.external-baselines.v1"
EXTERNAL_METHOD_MANIFEST_SCHEMA = "betlas.beta-barrel-staves.external-method.v1"

DEFAULT_EXTERNAL_BASELINE_ENV = (
    "BETAWARE_SENSITIVITY",
    "BETAWARE_THRESHOLD",
    "DOTNET_BIN",
    "GOLD_DATASET",
    "GOLD_EVIDENCE_LEVELS",
    "GOLD_QC_STATUSES",
    "GOLD_RUN_DIR",
    "HF_HUB_DISABLE_XET",
    "JAVA_BIN",
    "LD_LIBRARY_PATH",
    "POLARBEARAL_REBUILD",
    "POLARBEARAL_REUSE",
    "POLARBEARAL_TIMEOUT",
    "PSIBLAST_THREADS",
    "PYTHON_BIN",
    "TMBED_BATCH_SIZE",
    "TMBED_PYTHON",
    "TMBED_THREADS",
    "TMBED_TIMEOUT",
)


def selected_environment(
    names: Sequence[str] = DEFAULT_EXTERNAL_BASELINE_ENV,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = source or os.environ
    return {name: env[name] for name in names if name in env}


def command_state(command: str | Sequence[str] | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(command, str):
        return {"line": command}
    if isinstance(command, Mapping):
        return {str(key): command_state(value) for key, value in command.items()}
    argv = [str(part) for part in command]
    return {"argv": argv, "line": shlex.join(argv)}


def artifact_state(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return {
            "path": str(path),
            "type": "missing",
            "exists": False,
            "size": 0,
            "mtime_ns": 0,
            "sha256": "",
        }
    if resolved.is_dir():
        files = []
        total_size = 0
        for child in sorted(resolved.rglob("*")):
            if not child.is_file():
                continue
            row = file_state(child)
            row["relative_path"] = str(child.relative_to(resolved))
            files.append(row)
            total_size += int(row.get("size", 0))
        tree_payload = [
            {
                "relative_path": row["relative_path"],
                "sha256": row.get("sha256", ""),
                "size": row.get("size", 0),
            }
            for row in files
        ]
        tree_sha256 = ""
        if files:
            import hashlib

            tree_sha256 = hashlib.sha256(
                json.dumps(tree_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return {
            "path": str(resolved),
            "type": "directory",
            "exists": True,
            "file_count": len(files),
            "size": total_size,
            "sha256": tree_sha256,
            "files": files,
        }
    row = file_state(resolved)
    row["type"] = "file"
    return row


def artifact_states(paths: Mapping[str, str | Path] | None) -> dict[str, dict[str, Any]]:
    return {label: artifact_state(path) for label, path in (paths or {}).items()}


def build_external_method_manifest(
    *,
    method: str,
    command: str | Sequence[str] | Mapping[str, Any],
    inputs: Mapping[str, str | Path] | None = None,
    outputs: Mapping[str, str | Path] | None = None,
    env: Mapping[str, str] | None = None,
    env_names: Sequence[str] = DEFAULT_EXTERNAL_BASELINE_ENV,
    parameters: Mapping[str, Any] | None = None,
    status: str = "ok",
    returncode: int | None = None,
    runtime_seconds: float | None = None,
    message: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": EXTERNAL_METHOD_MANIFEST_SCHEMA,
        "created_utc": utc_now_iso(),
        "method": method,
        "status": status,
        "returncode": returncode,
        "runtime_seconds": None if runtime_seconds is None else round(float(runtime_seconds), 6),
        "message": message,
        "command": command_state(command),
        "parameters": dict(parameters or {}),
        "env": selected_environment(env_names, env or {}),
        "inputs": artifact_states(inputs),
        "outputs": artifact_states(outputs),
        "extra": dict(extra or {}),
    }
    return payload


def build_external_baseline_manifest(
    *,
    run_name: str,
    status: str,
    methods: Mapping[str, Mapping[str, Any]] | None = None,
    command: str | Sequence[str] | Mapping[str, Any] | None = None,
    inputs: Mapping[str, str | Path] | None = None,
    outputs: Mapping[str, str | Path] | None = None,
    env: Mapping[str, str] | None = None,
    env_names: Sequence[str] = DEFAULT_EXTERNAL_BASELINE_ENV,
    parameters: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": EXTERNAL_BASELINE_MANIFEST_SCHEMA,
        "created_utc": utc_now_iso(),
        "run_name": run_name,
        "status": status,
        "parameters": dict(parameters or {}),
        "metrics": dict(metrics or {}),
        "env": selected_environment(env_names, env or {}),
        "inputs": artifact_states(inputs),
        "outputs": artifact_states(outputs),
        "methods": dict(methods or {}),
        "runtime": runtime_state(),
        "extra": dict(extra or {}),
    }
    if command is not None:
        payload["command"] = command_state(command)
    return payload


__all__ = [
    "DEFAULT_EXTERNAL_BASELINE_ENV",
    "EXTERNAL_BASELINE_MANIFEST_SCHEMA",
    "EXTERNAL_METHOD_MANIFEST_SCHEMA",
    "artifact_state",
    "artifact_states",
    "build_external_baseline_manifest",
    "build_external_method_manifest",
    "command_state",
    "selected_environment",
    "write_json",
]
