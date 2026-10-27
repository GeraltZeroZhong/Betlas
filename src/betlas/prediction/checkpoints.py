"""Version-pinned installation of published numerical predictor checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Any

RELEASE = "v1.1.0"
TASKS = ("solenoid", "barrel", "barrel-strands")
DEFAULT_BASE_URL = f"https://github.com/GeraltZeroZhong/Betlas/releases/download/{RELEASE}"


class CheckpointError(ValueError):
    """A requested predictor could not be installed or validated."""


def _task(task: str) -> str:
    if task not in TASKS:
        raise CheckpointError(f"Unknown prediction task {task!r}; choose {', '.join(TASKS)}")
    return task


def _manifest() -> dict[str, dict[str, Any]]:
    payload = json.loads(files(__package__).joinpath("manifest.json").read_text())
    return payload["checkpoints"]


def _cache(cache_dir: str | Path | None) -> Path:
    selected = cache_dir or os.environ.get("BETLAS_CHECKPOINT_DIR", "~/.cache/betlas/checkpoints")
    return Path(selected).expanduser() / RELEASE


def _verified(path: Path, entry: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != int(entry["byte_size"]):
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == entry["sha256"]


def list_checkpoints(*, cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Describe the three release models and verified local availability."""
    manifest = _manifest()
    return [
        {
            "task": task,
            "release": RELEASE,
            **manifest[task],
            "installed": _verified(_cache(cache_dir) / manifest[task]["filename"], manifest[task]),
        }
        for task in TASKS
    ]


def install_checkpoint(
    task: str,
    *,
    cache_dir: str | Path | None = None,
    base_url: str | Path | None = None,
    allow_download: bool = True,
) -> Path:
    """Install and verify one release checkpoint, reusing an intact cached copy.

    ``base_url`` or ``BETLAS_CHECKPOINT_BASE_URL`` selects a release mirror. A
    local directory supports offline release installation. Every source must
    match the package's release manifest.
    """
    task = _task(task)
    entry = _manifest()[task]
    target = _cache(cache_dir) / entry["filename"]
    if _verified(target, entry):
        return target
    source = str(base_url or os.environ.get("BETLAS_CHECKPOINT_BASE_URL", DEFAULT_BASE_URL))
    scheme = urllib.parse.urlparse(source).scheme
    local = scheme == "" or scheme == "file"
    if not local and not allow_download:
        raise CheckpointError(
            f"The {task} checkpoint is not installed. Run 'betlas checkpoints install {task}' "
            "with network access, provide a local release mirror, or pass --checkpoint."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".json", dir=target.parent)
    staging = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output:
            if local:
                directory = (
                    Path(urllib.request.url2pathname(urllib.parse.urlparse(source).path))
                    if scheme == "file"
                    else Path(source).expanduser()
                )
                with (directory / entry["filename"]).open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            else:
                if scheme != "https":
                    raise CheckpointError(
                        "Checkpoint URLs must use HTTPS; local directories are also supported"
                    )
                url = source.rstrip("/") + "/" + urllib.parse.quote(entry["filename"])
                with urllib.request.urlopen(url, timeout=60) as stream:
                    shutil.copyfileobj(stream, output)
        if not _verified(staging, entry):
            raise CheckpointError(
                f"Checkpoint size or SHA-256 does not match the {RELEASE} manifest: {task}"
            )
        os.replace(staging, target)
    except (OSError, urllib.error.URLError) as exc:
        raise CheckpointError(
            f"Could not install the {task} checkpoint for {RELEASE}: {exc}. "
            "Check release availability or provide --checkpoint / BETLAS_CHECKPOINT_BASE_URL."
        ) from exc
    finally:
        staging.unlink(missing_ok=True)
    return target


def resolve_checkpoint(task: str, *, checkpoint: str | Path | None = None, **kwargs: Any) -> Path:
    """Resolve a caller-supplied JSON model or a verified release model."""
    _task(task)
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise CheckpointError(f"Checkpoint file does not exist: {path}")
        return path
    return install_checkpoint(task, **kwargs)
