from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..constants import THREAD_ENV_DEFAULTS
from ..runtime import runtime_summary

RUN_METADATA_SCHEMA = "betlas.run-metadata.v1"


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not-installed"


def _source_repo_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_value(args: list[str]) -> str:
    repo_root = _source_repo_root()
    if repo_root is None:
        return ""
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_file_state(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError:
        return {
            "path": str(path_value),
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
        "sha256": _sha256_file(resolved),
    }


def _config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)  # pragma: no cover - AppConfig is the supported type.


def default_metadata_path(output_csv: str) -> str:
    output = Path(output_csv).expanduser()
    return str(output.with_suffix(f"{output.suffix}.metadata.json"))


def build_run_metadata(
    cfg: AppConfig,
    *,
    input_files: list[str],
    output_csv: str | None,
    row_count: int,
) -> dict[str, Any]:
    return {
        "schema": RUN_METADATA_SCHEMA,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "versions": {
            "betlas.readouts.beta_barrel_staves": _package_version("betlas"),
            "biopython": _package_version("biopython"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "hydra_core": _package_version("hydra-core"),
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "dirty": bool(_git_value(["status", "--short"])),
        },
        "runtime": runtime_summary(cfg.runtime.dssp_bin_path, require_dssp=False),
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in THREAD_ENV_DEFAULTS
        },
        "config": _config_to_dict(cfg),
        "input_files": [_input_file_state(path) for path in input_files],
        "output_csv": output_csv or "",
        "row_count": int(row_count),
    }


def write_run_metadata(metadata: dict[str, Any], metadata_path: str) -> None:
    output = Path(metadata_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=str(output.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, output)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
