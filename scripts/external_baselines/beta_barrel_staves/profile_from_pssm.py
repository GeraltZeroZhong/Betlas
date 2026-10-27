#!/usr/bin/env python3
"""Convert PSI-BLAST ASCII PSSM observed percentages to a BETAWARE profile."""

from __future__ import annotations

import argparse
from pathlib import Path

PSIBLAST_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


def pssm_to_betaware_profile(pssm: Path, output: Path) -> None:
    """Write twenty frequency columns in PSI-BLAST's amino-acid order."""
    rows = []
    for line in pssm.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 42 or not parts[0].isdigit():
            continue
        rows.append([float(value) / 100.0 for value in parts[22:42]])
    if not rows:
        raise ValueError(f"No ASCII PSSM rows parsed from {pssm}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(" ".join(f"{value:.2f}" for value in row) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        pssm_to_betaware_profile(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Set BETAWARE_AA_ORDER={PSIBLAST_AA_ORDER} when running BETAWARE with this profile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
