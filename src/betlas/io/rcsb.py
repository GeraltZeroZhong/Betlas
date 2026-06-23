from __future__ import annotations

import os
import re
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..constants import DEFAULT_MMCIF_DIR

_URL_TIMEOUT_SECONDS = 30
_PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")


def validate_pdb_id(pdb_id: str) -> str:
    """Return a canonical lower-case RCSB PDB id or raise on unsafe input."""

    value = str(pdb_id).strip()
    if not _PDB_ID_RE.fullmatch(value):
        raise ValueError(
            f"invalid pdb_id {pdb_id!r}; expected exactly four letters/digits with no path separators"
        )
    return value.lower()


def _download_atomic(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp") as handle:
            tmp_path = Path(handle.name)
        with urllib.request.urlopen(url, timeout=_URL_TIMEOUT_SECONDS) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if tmp_path.stat().st_size == 0:
            raise OSError(f"downloaded empty file from {url}")
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def mmcif_path_for(pdb_id: str, mmcif_dir: Path = DEFAULT_MMCIF_DIR) -> Path:
    canonical = validate_pdb_id(pdb_id)
    base = Path(mmcif_dir).expanduser()
    path = base / f"{canonical}.cif.gz"
    resolved_base = base.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_base):
        raise ValueError(f"resolved mmCIF path escapes mmcif_dir: {path}")
    return path


def download_mmcif(pdb_id: str, mmcif_dir: Path = DEFAULT_MMCIF_DIR, *, force: bool = False) -> Path:
    canonical = validate_pdb_id(pdb_id)
    mmcif_dir.mkdir(parents=True, exist_ok=True)
    path = mmcif_path_for(canonical, mmcif_dir)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    url = f"https://files.rcsb.org/download/{canonical.upper()}.cif.gz"
    _download_atomic(url, path)
    return path


def download_mmcifs(
    pdb_ids: list[str],
    mmcif_dir: Path = DEFAULT_MMCIF_DIR,
    *,
    workers: int = 8,
    force: bool = False,
) -> dict[str, str]:
    unique_ids = sorted({validate_pdb_id(pdb_id) for pdb_id in pdb_ids if str(pdb_id).strip()})
    results: dict[str, str] = {}
    if workers <= 1:
        for pdb_id in tqdm(unique_ids, desc="Downloading mmCIF"):
            try:
                results[pdb_id] = str(download_mmcif(pdb_id, mmcif_dir, force=force))
            except Exception as exc:  # pragma: no cover - network failures are environment-specific
                results[pdb_id] = f"ERROR: {exc}"
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_id = {
            pool.submit(download_mmcif, pdb_id, mmcif_dir, force=force): pdb_id
            for pdb_id in unique_ids
        }
        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Downloading mmCIF"):
            pdb_id = future_to_id[future]
            try:
                results[pdb_id] = str(future.result())
            except Exception as exc:  # pragma: no cover - network failures are environment-specific
                results[pdb_id] = f"ERROR: {exc}"
    return results
