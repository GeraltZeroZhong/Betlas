"""Bounded local runs using verified upstream CLIs; no input generation service."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from .formats import (
    parse_bbcontacts,
    parse_bcov,
    parse_betapro_map,
    parse_betapro_str,
    read_bcov_strands,
)


def _input_path(path: str | Path) -> Path:
    result = Path(path).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _identity(sample_id: str, sequence: str, ss_source: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", sample_id):
        raise ValueError("Use a nonempty identifier without whitespace")
    if not re.fullmatch(r"[A-Z]+", sequence):
        raise ValueError("Sequence must be an ungapped uppercase amino-acid string")
    if ss_source not in ("predicted", "observed"):
        raise ValueError("Declare ss_source as predicted or observed")


def _execute(argv: Sequence[str], directory: Path, timeout: float, tag: str) -> dict:
    """Keep a local command receipt and kill its owned process group on timeout."""
    if isinstance(argv, (str, bytes)) or not argv or timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("A command and finite positive timeout are required")
    record = dict(command=list(map(str, argv)), timeout_seconds=timeout, status="starting")
    start = time.monotonic()
    try:
        with (directory / f"{tag}.stdout.log").open("w") as out, (
                directory / f"{tag}.stderr.log").open("w") as err:
            process = subprocess.Popen(list(map(str, argv)), cwd=directory,
                                       stdout=out, stderr=err, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record.update(status="timeout", returncode=process.returncode)
                raise
            record.update(status="ok" if process.returncode == 0 else "failed",
                          returncode=process.returncode)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, argv)
    except OSError as error:
        record.update(status="launch_failed", error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        (directory / f"{tag}.execution.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _finish(result: dict, directory: Path, ss_source: str, inputs: dict) -> dict:
    result.update(secondary_structure_source=ss_source,
                  input_provenance=inputs, omitted_pairs="not_emitted_by_native_method")
    (directory / "normalized.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_psicov_msa(path: str | Path, sequence: str) -> int:
    """Require one equal-width sequence per line and the ungapped target first."""
    count = 0
    with Path(path).open() as handle:
        for line in handle:
            row = line.rstrip("\r\n")
            if len(row) != len(sequence) or not re.fullmatch(r"[A-Z-]+", row):
                raise ValueError("PSICOV MSA must contain uppercase equal-width aligned rows, no headers")
            if count == 0 and row != sequence:
                raise ValueError("The first MSA row must equal the ungapped target sequence")
            count += 1
    if count == 0:
        raise ValueError("Empty MSA")
    return count


def run_bcov(*, command: Sequence[str], sample_id: str, sequence: str,
             strands_file: str | Path, ss_source: str, output_dir: str | Path,
             timeout: float, msa: str | Path | None = None,
             scores: str | Path | None = None, minimum_parallel_separation: int = 6) -> dict:
    """Native BCov MSA mode or explicitly supplied sparse coupling-score mode."""
    _identity(sample_id, sequence, ss_source)
    if (msa is None) == (scores is None):
        raise ValueError("Supply exactly one of msa or scores")
    if minimum_parallel_separation < 0:
        raise ValueError("Minimum parallel separation cannot be negative")
    strands = _input_path(strands_file)
    if len(read_bcov_strands(strands, len(sequence))) < 2:
        raise ValueError("BCov requires at least two strands for its score normalization")
    source = _input_path(msa if msa is not None else scores)
    count = validate_psicov_msa(source, sequence) if msa is not None else None
    if scores is not None:
        rows = [x.split() for x in source.read_text().splitlines() if x.strip()]
        if not rows or rows[0] != [str(len(sequence))]:
            raise ValueError("BCov sparse matrix must start with target length")
        seen = set()
        for row in rows[1:]:
            if len(row) != 3:
                raise ValueError("BCov input score matrix has three columns")
            i, j = map(int, row[:2])
            if not 0 <= i < j < len(sequence) or (i, j) in seen or not math.isfinite(float(row[2])):
                raise ValueError("Invalid or duplicate BCov sparse matrix entry")
            seen.add((i, j))
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    argv = [*command, "-a" if msa is not None else "-m", str(source), "-s", str(strands),
            "-o", str(directory / "prediction.topo"), "-c", str(directory / "prediction.bmap"),
            "-n", str(minimum_parallel_separation)]
    _execute(argv, directory, timeout, "bcov")
    result = parse_bcov(directory / "prediction.topo", directory / "prediction.bmap", strands,
                        sample_id=sample_id, length=len(sequence))
    return _finish(result, directory, ss_source, dict(strands=str(strands),
        msa=str(source) if msa is not None else None, score_matrix=str(source) if scores else None,
        msa_sequence_count=count, minimum_parallel_separation=minimum_parallel_separation))


def run_bbcontacts(*, command: Sequence[str], sample_id: str, sequence: str,
                   coupling_matrix: str | Path, secondary_structure: str | Path,
                   ss_source: str, alignment_sequence_count: int, output_dir: str | Path,
                   timeout: float, config: str | Path | None = None) -> dict:
    """Use sqrt(N)/L diversity with unchanged upstream smoothing/shortening defaults."""
    _identity(sample_id, sequence, ss_source)
    if alignment_sequence_count < 1:
        raise ValueError("The alignment must contain at least one sequence")
    matrix, secondary = _input_path(coupling_matrix), _input_path(secondary_structure)
    ss = "".join(line.strip() for line in secondary.read_text().splitlines()
                 if not line.startswith(">"))
    if len(ss) != len(sequence) or set(ss) - set("HEC"):
        raise ValueError("bbcontacts secondary structure must contain exactly L H/E/C states")
    rows = 0
    with matrix.open() as handle:
        for line in handle:
            if line.startswith("#"):
                # Native reader enumerates source lines: comments may appear only at the end.
                if rows != len(sequence):
                    raise ValueError("bbcontacts matrix comments must follow all matrix rows")
                continue
            values = line.strip().split("\t")
            if len(values) != len(sequence) or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("bbcontacts requires a finite L by L tab-delimited matrix")
            rows += 1
    if rows != len(sequence):
        raise ValueError("bbcontacts coupling matrix has the wrong row count")
    diversity = math.sqrt(alignment_sequence_count) / len(sequence)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    prefix = directory / "prediction"
    argv = [*command, str(matrix), repr(diversity), str(prefix),
            "-d" if ss_source == "observed" else "-p", str(secondary), "-n", sample_id]
    if config is not None:
        argv += ["-c", str(_input_path(config))]
    _execute(argv, directory, timeout, "bbcontacts")
    suffix = ".DSSP.filteredoutput.txt" if ss_source == "observed" else ".filteredoutput.txt"
    result = parse_bbcontacts(str(prefix) + suffix, sample_id=sample_id,
                              length=len(sequence), successful_exit=True)
    return _finish(result, directory, ss_source, dict(coupling_matrix=str(matrix),
        secondary_structure=str(secondary), alignment_sequence_count=alignment_sequence_count,
        supplied_diversity=diversity, config=str(config) if config else "upstream_default"))


def run_betapro(*, installation: str | Path, sample_id: str, sequence: str,
               input_file: str | Path, input_kind: str, ss_source: str,
               output_dir: str | Path, timeout: float) -> dict:
    """Run a separately configured BetaPro installation; never patch it silently.

    input_kind is fasta, supplied_ss, or probability_map. Even supplied_ss
    still needs SSpro4 for solvent accessibility and sequence profiles.
    """
    _identity(sample_id, sequence, ss_source)
    if input_kind not in ("fasta", "supplied_ss", "probability_map"):
        raise ValueError("Unknown BetaPro input kind")
    if input_kind == "fasta" and ss_source != "predicted":
        raise ValueError("FASTA mode uses predicted secondary structure")
    root, source = Path(installation).resolve(), _input_path(input_file)
    # Upstream shell and Perl scripts interpolate paths without quoting.
    if re.search(r"[^A-Za-z0-9_./-]", str(root)):
        raise ValueError("Legacy BetaPro installation path must not contain shell metacharacters or spaces")
    script_names = ["predict_strand.sh"]
    if input_kind != "probability_map":
        script_names += ["predict_beta_fasta.sh" if input_kind == "fasta" else "predict_beta_ss.sh"]
    for name in script_names:
        script = _input_path(root / "bin" / name)
        if "/home/baldig/jianlin/" in script.read_text():
            raise ValueError("BetaPro still has historical installation paths; run upstream configure.pl first")
    directory = Path(output_dir).resolve()
    if re.search(r"[^A-Za-z0-9_./-]", str(directory)):
        raise ValueError("Legacy BetaPro output path must not contain shell metacharacters or spaces")
    directory.mkdir(parents=True, exist_ok=False)
    staged = directory / "input.txt"
    staged.write_bytes(source.read_bytes())
    matrix = directory / "prediction.map"
    start = time.monotonic()
    if input_kind == "probability_map":
        matrix.write_bytes(staged.read_bytes())
    else:
        input_lines = staged.read_text().splitlines()
        if len(input_lines) != (2 if input_kind == "fasta" else 3):
            raise ValueError("BetaPro input must have a header, one-line sequence, and optional H/E/C line")
        if input_lines[0].lstrip(">") != sample_id or input_lines[1] != sequence:
            raise ValueError("BetaPro input identifier/sequence mismatch")
        if input_kind == "supplied_ss" and (len(input_lines[2]) != len(sequence)
                                               or set(input_lines[2]) - set("HEC.GIST-")):
            raise ValueError("Invalid supplied secondary structure")
        _execute(["/bin/sh", str(root / "bin" / script_names[-1]), "input.txt", "prediction.map"],
                 directory, timeout, "betapro_map")
    probability = parse_betapro_map(matrix, sample_id=sample_id, expected_sequence=sequence)
    remaining = timeout - (time.monotonic() - start)
    if remaining <= 0:
        raise subprocess.TimeoutExpired("BetaPro total walltime", timeout)
    _execute(["/bin/sh", str(root / "bin" / "predict_strand.sh"), "prediction.map", "prediction"],
             directory, remaining, "betapro_strands")
    if "error happens in strand pairing prediction" in (directory / "betapro_strands.stdout.log").read_text():
        raise RuntimeError("BetaPro Perl wrapper reported a failed predictor despite a zero exit code")
    result = parse_betapro_str(directory / "prediction.str", sample_id=sample_id,
                               expected_sequence=sequence)
    result["score_matrix"] = probability["score_matrix"]
    result["secondary_structure"] = probability["secondary_structure"]
    return _finish(result, directory, ss_source, dict(input_file=str(source), input_kind=input_kind,
                                                    installation=str(root)))
