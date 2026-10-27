"""Bounded file reads for local research tables and compressed coordinate input."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_bytes(path: str | Path, chunk_size: int = 8192) -> bytes:
    """Avoid large individual reads on mounted filesystems with limited buffers."""
    chunks = []
    with Path(path).open("rb", buffering=0) as handle:
        while block := handle.read(chunk_size):
            chunks.append(block)
    return b"".join(chunks)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    content = read_bytes(path)
    if content.startswith(b"\x1f\x8b") and "compression" not in kwargs:
        kwargs["compression"] = "gzip"
    return pd.read_csv(io.BytesIO(content), **kwargs)


def read_json(path: str | Path) -> Any:
    return json.loads(read_bytes(path))


def read_mmcif(path: str | Path) -> dict[str, Any]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    content = read_bytes(path)
    if content.startswith(b"\x1f\x8b"):
        content = gzip.decompress(content)
    return dict(MMCIF2Dict(io.StringIO(content.decode("utf-8"))))


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
