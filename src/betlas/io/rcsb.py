from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..constants import DEFAULT_MMCIF_DIR


def mmcif_path_for(pdb_id: str, mmcif_dir: Path = DEFAULT_MMCIF_DIR) -> Path:
    return mmcif_dir / f"{pdb_id.lower()}.cif.gz"


def download_mmcif(pdb_id: str, mmcif_dir: Path = DEFAULT_MMCIF_DIR, *, force: bool = False) -> Path:
    mmcif_dir.mkdir(parents=True, exist_ok=True)
    path = mmcif_path_for(pdb_id, mmcif_dir)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
    urllib.request.urlretrieve(url, path)
    return path


def download_mmcifs(
    pdb_ids: list[str],
    mmcif_dir: Path = DEFAULT_MMCIF_DIR,
    *,
    workers: int = 8,
    force: bool = False,
) -> dict[str, str]:
    unique_ids = sorted({pdb_id.lower() for pdb_id in pdb_ids if str(pdb_id).strip()})
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
