from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..config import AppConfig

PREPARE_CACHE_VERSION = 5
PREPARE_CACHE_SCHEMA = "betlas.prepare-cache.v1"


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not-installed"


def default_prepare_cache_dir() -> str:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache_home) if xdg_cache_home else (Path.home() / ".cache")
    return str(cache_root / "betlas" / "prepare")


def resolve_prepare_cache_dir(configured_dir: str | None = None) -> Path:
    cache_dir = configured_dir or default_prepare_cache_dir()
    return Path(cache_dir).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_state(file_path: str) -> dict[str, int | str]:
    path = Path(file_path).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _path_state(path_value: str | None) -> dict[str, int | str | bool]:
    if not path_value:
        return {"path": "", "exists": False}

    raw_path = str(path_value)
    try:
        executable_path = shutil.which(raw_path) if os.path.basename(raw_path) == raw_path else None
        path = Path(executable_path or raw_path).expanduser().resolve(strict=True)
        stat = path.stat()
    except OSError:
        return {"path": raw_path, "exists": False}

    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _prepare_config_state(cfg: AppConfig) -> dict[str, object]:
    return {
        "cache_version": PREPARE_CACHE_VERSION,
        "betlas.readouts.beta_barrel_staves_version": _package_version("betlas"),
        "biopython_version": _package_version("biopython"),
        "chain_id": str(getattr(cfg.input, "chain_id", "") or ""),
        "min_chain_residues": int(cfg.input.min_chain_residues),
        "dssp_bin_path": str(cfg.runtime.dssp_bin_path or ""),
        "dssp_binary_state": _path_state(cfg.runtime.dssp_bin_path),
        "fail_on_dssp_error": bool(cfg.runtime.fail_on_dssp_error),
        "barrel_gate_enabled": bool(cfg.barrel_gate.enabled),
        "barrel_gate_fail_on_error": bool(cfg.barrel_gate.fail_on_error),
        "barrel_gate_pass_results": [str(value) for value in cfg.barrel_gate.pass_results],
        "barrel_gate_overrides": [str(value) for value in cfg.barrel_gate.overrides],
    }


def build_prepare_cache_key(file_path: str, cfg: AppConfig) -> str:
    payload = {
        "file": _file_state(file_path),
        "prepare": _prepare_config_state(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_cache_path(file_path: str, cfg: AppConfig) -> Path:
    cache_dir = resolve_prepare_cache_dir(cfg.runtime.prepare_cache_dir)
    cache_key = build_prepare_cache_key(file_path, cfg)
    return cache_dir / cache_key[:2] / f"{cache_key}.json"


def _unlink_corrupt_cache(cache_path: Path) -> None:
    try:
        cache_path.unlink()
    except OSError:
        pass


def _payloads_from_cache_record(
    cache_record: object,
    *,
    expected_key: str,
) -> list[dict[str, object]] | None:
    if not isinstance(cache_record, dict):
        return None
    if cache_record.get("schema") != PREPARE_CACHE_SCHEMA:
        return None
    if cache_record.get("cache_version") != PREPARE_CACHE_VERSION:
        return None
    if cache_record.get("cache_key") != expected_key:
        return None

    payloads = cache_record.get("payloads")
    if isinstance(payloads, list):
        return payloads
    return None


def load_prepare_payloads(file_path: str, cfg: AppConfig) -> list[dict[str, object]] | None:
    if not cfg.runtime.prepare_cache_enabled:
        return None

    cache_path = prepare_cache_path(file_path, cfg)
    cache_key = build_prepare_cache_key(file_path, cfg)
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache_record = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        _unlink_corrupt_cache(cache_path)
        return None

    payloads = _payloads_from_cache_record(cache_record, expected_key=cache_key)
    if payloads is None:
        _unlink_corrupt_cache(cache_path)
        return None
    return payloads


def store_prepare_payloads(file_path: str, cfg: AppConfig, payloads: list[dict[str, object]]) -> None:
    if not cfg.runtime.prepare_cache_enabled:
        return

    cache_path = prepare_cache_path(file_path, cfg)
    cache_key = build_prepare_cache_key(file_path, cfg)
    cache_record = {
        "schema": PREPARE_CACHE_SCHEMA,
        "cache_version": PREPARE_CACHE_VERSION,
        "cache_key": cache_key,
        "file_state": _file_state(file_path),
        "prepare_state": _prepare_config_state(cfg),
        "payloads": payloads,
    }
    tmp_path: str | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f"{cache_path.stem}.", suffix=".tmp", dir=cache_path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(cache_record, handle, sort_keys=True, separators=(",", ":"))
        os.replace(tmp_path, cache_path)
        tmp_path = None
    except OSError:
        pass
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
