"""Join fixed-release CATH domain sequences without a DSSP dependency.

Published domain sequences may include residues absent from current coordinates.
They are a distinct input representation from observed-coordinate sequences.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser

_HEADER = re.compile(r"cath\|([0-9]+_[0-9]+_[0-9]+)\|([A-Za-z0-9]{5}[0-9]{2})/(\S+)\Z")
_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBXZOUJ")
_OUTPUT_COLUMNS = (
    "domain_sequence", "sequence_source", "sequence_status", "sequence_reason",
    "sequence_header", "sequence_boundary_provenance", "cath_sequence_length",
    "sequence_length_matches_cath_domain_length", "sequence_length_difference",
)


def build_sequence_table(manifest: pd.DataFrame, fasta: Path) -> pd.DataFrame:
    """Preserve every manifest record and join official domain IDs case-sensitively.

    FASTA header versions must agree with the single manifest ``cath_version``.
    Duplicate FASTA IDs, including unselected IDs, are input conflicts. Missing
    records and multi-chain domains remain explicit failures with empty sequences.
    Length disagreement is reported, not silently corrected or used for exclusion.
    """
    required = {"record_id", "cath_version", "cath_domain_length"}
    if required - set(manifest):
        raise ValueError(f"Manifest lacks required columns: {sorted(required - set(manifest))}")
    frame = manifest.copy().reset_index(drop=True)
    if frame.record_id.isna().any() or frame.record_id.astype(str).str.strip().eq("").any():
        raise ValueError("Manifest record_id must be nonempty")
    if frame.record_id.duplicated().any():
        raise ValueError("Manifest record_id must be unique")
    if "domain_id" not in frame:
        frame["domain_id"] = frame.record_id
    if frame.domain_id.isna().any() or frame.domain_id.duplicated().any():
        raise ValueError("Manifest domain_id must be nonempty and unique")
    if not frame.record_id.astype(str).eq(frame.domain_id.astype(str)).all():
        raise ValueError("Fixed-release record_id must equal the exact CATH domain_id")
    if frame.empty:
        for column in _OUTPUT_COLUMNS:
            frame[column] = pd.Series(dtype=object)
        return frame
    versions = frame.cath_version.astype(str).unique().tolist()
    if len(versions) != 1 or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", versions[0]):
        raise ValueError("Manifest must declare one fixed cath_version")
    version = versions[0]
    wanted = set(frame.domain_id.astype(str))
    retained: dict[str, tuple[str, str, str]] = {}
    seen: set[str] = set()
    opener = gzip.open if Path(fasta).suffix == ".gz" else open
    with opener(fasta, "rt", encoding="utf-8") as handle:
        for title, sequence in SimpleFastaParser(handle):
            match = _HEADER.fullmatch(title)
            if match is None:
                raise ValueError(f"Invalid fixed-release CATH FASTA header: {title!r}")
            release, domain_id, boundaries = match.groups()
            if release.replace("_", ".") != version:
                raise ValueError(f"FASTA release {release} differs from manifest cath_version {version}")
            if domain_id in seen:
                raise ValueError(f"Duplicate FASTA domain ID: {domain_id}")
            seen.add(domain_id)
            if domain_id in wanted:
                retained[domain_id] = (title, boundaries, sequence)
    rows = []
    for record in frame.to_dict("records"):
        domain_id = str(record["domain_id"])
        length_text = str(record["cath_domain_length"])
        if not re.fullmatch(r"[1-9][0-9]*", length_text):
            raise ValueError(f"Invalid positive cath_domain_length for {domain_id}")
        declared_length = int(length_text)
        source = retained.get(domain_id)
        status, reason = "ok", ""
        title, boundaries, sequence = source if source else ("", "", "")
        if source is None:
            status, reason = "missing_sequence", "Domain ID is absent from the fixed-release FASTA"
        elif not sequence or set(sequence) - _RESIDUES:
            status, reason = "invalid_sequence", "Published sequence is empty or has unsupported residue symbols"
        if "segments_json" in record and str(record["segments_json"]).strip():
            segments = json.loads(record["segments_json"])
            if not isinstance(segments, list) or any(not isinstance(segment, dict) or "chain_id" not in segment for segment in segments):
                raise ValueError(f"Invalid segments_json for {domain_id}")
            if len({str(segment["chain_id"]) for segment in segments}) > 1:
                status, reason = "multichain_requires_explicit_pooling", "No per-chain sequence assignment is supplied"
        rows.append({
            "domain_sequence": sequence if status == "ok" else "",
            "sequence_source": "CATH release sequence",
            "sequence_status": status,
            "sequence_reason": reason,
            "sequence_header": title,
            "sequence_boundary_provenance": boundaries,
            "cath_sequence_length": len(sequence) if source else None,
            "sequence_length_matches_cath_domain_length": len(sequence) == declared_length if source else None,
            "sequence_length_difference": len(sequence) - declared_length if source else None,
        })
    for column in _OUTPUT_COLUMNS:
        frame[column] = [row[column] for row in rows]
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_csv.resolve() in {args.manifest.resolve(), args.fasta.resolve()}:
        parser.error("Output must differ from both source files")
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    result = build_sequence_table(manifest, args.fasta)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False)
    print(json.dumps({"records": len(result), "sequence_status_counts": result.sequence_status.value_counts().to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
