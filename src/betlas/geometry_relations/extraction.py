"""Parse mmCIF bytes with interchangeable local parsing backends."""
from __future__ import annotations

import gzip
import io
from typing import Any

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

def parse_mmcif(content: bytes, *, parser: str) -> dict[str, Any]:
    """Decode an mmCIF with scalar/list behavior matching Biopython."""
    if content.startswith(b"\x1f\x8b"):
        content = gzip.decompress(content)
    text = content.decode("utf-8")
    if parser == "biopython":
        return dict(MMCIF2Dict(io.StringIO(text)))
    if parser != "gemmi":
        raise ValueError(f"Unsupported mmCIF parser: {parser}")
    import gemmi

    block = gemmi.cif.read_string(text).sole_block()
    result = {"data_": block.name}

    def unquote(value):
        decoded = gemmi.cif.as_string(value) if value.startswith(('"', "'", ";")) else value
        # MMCIF2Dict strips trailing whitespace on each semicolon-text line;
        # leading whitespace and missing-value tokens remain meaningful.
        return "\n".join(line.rstrip() for line in decoded.split("\n")) if value.startswith(";") else decoded

    for item in block:
        if item.pair:
            key, value = item.pair
            result[key] = [unquote(value)]
        elif item.loop:
            loop = item.loop
            width = len(loop.tags)
            for offset, key in enumerate(loop.tags):
                result[key] = [unquote(value) for value in loop.values[offset::width]]
    return result
