from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA = "betlas.run-manifest.v1"
DATA_MANIFEST_SCHEMA = "betlas.data-manifest.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not-installed"


def source_repo_root(start: Path | None = None) -> Path | None:
    base = (start or Path.cwd()).resolve()
    candidates = [base, *base.parents]
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def git_value(args: list[str], repo_root: Path | None = None) -> str:
    root = repo_root or source_repo_root()
    if root is None:
        return ""
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=root,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or source_repo_root()
    if root is None:
        return {
            "available": False,
            "root": "",
            "commit": "",
            "dirty": None,
            "note": "No .git directory was found; use source_tree for source-level traceability.",
        }
    return {
        "available": True,
        "root": str(root) if root else "",
        "commit": git_value(["rev-parse", "HEAD"], root),
        "dirty": bool(git_value(["status", "--short"], root)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError:
        return {
            "path": str(path),
            "exists": False,
            "size": 0,
            "mtime_ns": 0,
            "sha256": "",
        }
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def directory_manifest(
    root: str | Path,
    *,
    patterns: Iterable[str] = ("**/*",),
    max_file_bytes: int | None = None,
) -> list[dict[str, Any]]:
    base = Path(root).expanduser()
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                if max_file_bytes is not None and path.stat().st_size > max_file_bytes:
                    continue
                row = file_state(path)
            except OSError:
                continue
            row["relative_path"] = str(path.relative_to(base))
            rows.append(row)
    return rows


def source_tree_state(
    paths: Iterable[str | Path] = (
        "src",
        "scripts",
        "configs",
        "docs",
        "tests",
        "external_baselines",
        "pyproject.toml",
        "environment.yml",
        "LICENSE",
        "README.md",
    ),
) -> dict[str, Any]:
    base = Path.cwd()
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        files.extend(
            child
            for child in sorted(path.rglob("*"))
            if child.is_file()
            and "__pycache__" not in child.parts
            and ".pytest_cache" not in child.parts
            and ".ruff_cache" not in child.parts
        )

    digest = hashlib.sha256()
    total_bytes = 0
    relative_paths: list[str] = []
    for path in sorted(set(files)):
        try:
            relative = str(path.relative_to(base))
        except ValueError:
            relative = str(path)
        state = file_state(path)
        relative_paths.append(relative)
        total_bytes += int(state.get("size", 0))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(state.get("sha256", "")).encode("ascii"))
        digest.update(b"\0")

    return {
        "root": str(base),
        "sha256": digest.hexdigest(),
        "file_count": len(relative_paths),
        "total_bytes": total_bytes,
        "paths": relative_paths,
    }


def runtime_state() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "packages": {
            name: package_version(name)
            for name in (
                "betlas",
                "numpy",
                "pandas",
                "scipy",
                "biopython",
                "scikit-learn",
                "xgboost",
                "hydra-core",
            )
        },
    }


def build_run_manifest(
    *,
    command: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, str | Path] | None = None,
    outputs: Mapping[str, str | Path] | None = None,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "created_utc": utc_now_iso(),
        "command": command,
        "parameters": dict(parameters),
        "config": dict(config or {}),
        "inputs": {name: file_state(path) for name, path in (inputs or {}).items()},
        "outputs": {name: file_state(path) for name, path in (outputs or {}).items()},
        "metrics": dict(metrics or {}),
        "runtime": runtime_state(),
        "git": git_state(),
        "source_tree": source_tree_state(),
        "extra": dict(extra or {}),
    }


def build_data_manifest(
    *,
    name: str,
    roots: Mapping[str, str | Path],
    artifacts: Mapping[str, str | Path] | None = None,
    patterns: Iterable[str] = ("**/*",),
    max_file_bytes: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": DATA_MANIFEST_SCHEMA,
        "created_utc": utc_now_iso(),
        "name": name,
        "artifacts": {label: file_state(path) for label, path in (artifacts or {}).items()},
        "roots": {
            label: {
                "root": str(Path(root).expanduser()),
                "files": directory_manifest(root, patterns=patterns, max_file_bytes=max_file_bytes),
            }
            for label, root in roots.items()
        },
        "runtime": runtime_state(),
        "git": git_state(),
        "source_tree": source_tree_state(),
        "extra": dict(extra or {}),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
