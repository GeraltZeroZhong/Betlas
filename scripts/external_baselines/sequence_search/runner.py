"""Run externally installed protein-search tools against user-supplied inputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path

BLAST_FIELDS = (
    "qseqid", "saccver", "pident", "length", "qcovs", "evalue", "bitscore",
    "sscinames", "sskingdoms", "stitle",
)
MMSEQS_FIELDS = (
    "query", "target", "pident", "alnlen", "qstart", "qend", "tstart", "tend",
    "evalue", "bits",
)


def _validate_search(evalue: float, threads: int) -> None:
    if not math.isfinite(evalue) or evalue <= 0:
        raise ValueError("The E-value threshold must be finite and positive.")
    if threads < 1:
        raise ValueError("The thread count must be positive.")


def blastp_command(
    query: str | Path,
    database: str | Path,
    output: str | Path,
    *,
    executable: str | Path = "blastp",
    evalue: float = 1e-5,
    max_targets: int = 10,
    threads: int = 1,
) -> list[str]:
    """Construct a local BLASTp search with a fixed, documented TSV schema."""
    _validate_search(evalue, threads)
    if max_targets < 1:
        raise ValueError("The maximum target count must be positive.")
    return [
        str(executable), "-query", str(query), "-out", str(output),
        "-outfmt", "6 " + " ".join(BLAST_FIELDS), "-evalue", str(evalue),
        "-max_target_seqs", str(max_targets), "-db", str(database),
        "-num_threads", str(threads),
    ]


def mmseqs_command(
    query: str | Path,
    target: str | Path,
    output: str | Path,
    temporary: str | Path,
    *,
    executable: str | Path = "mmseqs",
    evalue: float = 1e-3,
    sensitivity: float = 7.5,
    threads: int = 1,
) -> list[str]:
    """Construct MMseqs2 easy-search for local query and target sequences."""
    _validate_search(evalue, threads)
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("The sensitivity must be finite and positive.")
    return [
        str(executable), "easy-search", str(query), str(target), str(output),
        str(temporary), "--format-output", ",".join(MMSEQS_FIELDS),
        "-e", str(evalue), "-s", str(sensitivity), "--threads", str(threads),
    ]


def _prepare_output(query: str | Path, output: str | Path) -> Path:
    query_path = Path(query).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not query_path.is_file():
        raise FileNotFoundError(f"Query FASTA does not exist: {query_path}")
    if query_path == output_path:
        raise ValueError("The output path must differ from the query FASTA.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def run_blastp(
    query: str | Path,
    database: str | Path,
    output: str | Path,
    **options: object,
) -> Path:
    """Run BLASTp locally; database creation is the caller's responsibility."""
    output_path = _prepare_output(query, output)
    query_path = Path(query).expanduser().resolve()
    database_path = Path(database).expanduser().resolve()
    command = blastp_command(query_path, database_path, output_path, **options)
    env = os.environ.copy()
    database_parent = str(database_path.parent)
    env["BLASTDB"] = os.pathsep.join(
        part for part in (database_parent, env.get("BLASTDB", "")) if part
    )
    subprocess.run(command, check=True, env=env)
    if not output_path.is_file():
        raise FileNotFoundError(f"BLASTp produced no output file: {output_path}")
    return output_path


def run_mmseqs(
    query: str | Path,
    target: str | Path,
    output: str | Path,
    temporary: str | Path,
    **options: object,
) -> Path:
    """Run local MMseqs2 easy-search with an explicit working directory."""
    output_path = _prepare_output(query, output)
    query_path = Path(query).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if output_path == target_path:
        raise ValueError("The output path must differ from the target input.")
    temporary_path = Path(temporary).expanduser().resolve()
    temporary_path.mkdir(parents=True, exist_ok=True)
    command = mmseqs_command(
        query_path, target_path, output_path, temporary_path, **options
    )
    subprocess.run(command, check=True)
    if not output_path.is_file():
        raise FileNotFoundError(f"MMseqs2 produced no output file: {output_path}")
    return output_path


def read_hits(path: str | Path, *, method: str) -> list[dict[str, str]]:
    """Read unfiltered TSV hits, preserving tool output values as strings.

    Empty files represent searches without reported hits. Only query IDs present
    in the TSV can be recovered here; retain the query FASTA to count no-hit queries.
    """
    schemas = {"blastp": BLAST_FIELDS, "mmseqs": MMSEQS_FIELDS}
    if method not in schemas:
        raise ValueError("Method must be 'blastp' or 'mmseqs'.")
    fields = schemas[method]
    hits = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not values:
                continue
            if len(values) != len(fields):
                raise ValueError(
                    f"Line {line_number}: expected {len(fields)} fields, got {len(values)}."
                )
            hits.append(dict(zip(fields, values, strict=True)))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="method", required=True)
    for name in ("blastp", "mmseqs"):
        subparser = subparsers.add_parser(name, help=f"Run a local {name} search.")
        subparser.add_argument("--query", type=Path, required=True, help="Query protein FASTA.")
        subparser.add_argument("--output", type=Path, required=True, help="Output hit TSV.")
        subparser.add_argument("--executable", default=name, help="Installed tool executable.")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument("--evalue", type=float, default=1e-5 if name == "blastp" else 1e-3)
        if name == "blastp":
            subparser.add_argument("--database", type=Path, required=True, help="Local BLAST database prefix.")
            subparser.add_argument("--max-targets", type=int, default=10)
        else:
            subparser.add_argument("--target", type=Path, required=True, help="Target protein FASTA or local MMseqs2 database prefix.")
            subparser.add_argument("--temporary", type=Path, required=True, help="MMseqs2 temporary directory.")
            subparser.add_argument("--sensitivity", type=float, default=7.5)
    args = vars(parser.parse_args())
    method = args.pop("method")
    result = run_blastp(**args) if method == "blastp" else run_mmseqs(**args)
    print(f"Wrote {len(read_hits(result, method=method))} hits to {result}")
