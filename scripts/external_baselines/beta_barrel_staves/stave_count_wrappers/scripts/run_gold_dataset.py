#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
SRC_DIR = REPO / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betlas.readouts.beta_barrel_staves.external_baselines import (  # noqa: E402
    build_external_baseline_manifest,
    build_external_method_manifest,
    write_json,
)
from betlas.readouts.beta_barrel_staves.publication import (  # noqa: E402
    require_publication_gold_provenance,
)

RUN = Path(os.environ.get("GOLD_RUN_DIR", ROOT / "runs" / "gold")).expanduser().resolve()
DATASET = Path(
    os.environ.get(
        "GOLD_DATASET",
        REPO
        / "data"
        / "readouts"
        / "beta_barrel_staves"
        / "processed"
        / "beta_barrel_publication_gold.csv",
    )
).expanduser().resolve()
RCSB = REPO / "data" / "raw" / "rcsb"
PSIBLAST_PROFILE_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
DEFAULT_EVIDENCE_LEVELS = ("gold",)
DEFAULT_QC_STATUSES = ("pass",)
METHODS = (
    "betaware",
    "pred_tmbb2_hmm",
    "pred_tmbb2_hnn",
    "proftmb",
    "tmbed",
    "polarbearal3",
)
PROFILE_METHODS = {"betaware", "proftmb"}


def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("text", True)
    return subprocess.run(cmd, **kwargs)


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def env_list(name: str, default: tuple[str, ...]) -> set[str]:
    raw = os.environ.get(name)
    if not raw:
        return set(default)
    values = [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]
    return set(values or default)


def split_values(values: list[str] | None, default: tuple[str, ...]) -> set[str]:
    if not values:
        return set(default)
    parsed = [
        item.strip()
        for value in values
        for item in re.split(r"[,;]", value)
        if item.strip()
    ]
    return set(parsed or default)


def selected_methods(values: list[str] | None) -> set[str]:
    if not values:
        return set(METHODS)
    methods = split_values(values, ("all",))
    unknown = methods.difference((*METHODS, "all"))
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(unknown))}")
    if "all" in methods:
        return set(METHODS)
    return methods


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external baseline tools on the gold/pass PDB dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument(
        "--evidence-level",
        action="append",
        default=None,
        help="Evidence level to include. Repeat or comma-separate. Default: gold.",
    )
    parser.add_argument(
        "--qc-status",
        action="append",
        default=None,
        help="QC status to include. Repeat or comma-separate. Default: pass.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help=f"Baseline subset: {','.join(METHODS)},all. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write the manifest and return 0 even if a selected baseline fails.",
    )
    parser.add_argument(
        "--allow-opm-derived-gold",
        action="store_true",
        help="Allow legacy OPM-derived gold/pass labels without manual chain-count provenance for compatibility checks.",
    )
    parser.add_argument(
        "--allow-internal-stress-test",
        action="store_true",
        help="Allow AFDB/non-release stress-test rows for compatibility checks.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_row(
    component: str,
    seconds: float,
    *,
    status: str = "ok",
    detail: str = "",
) -> dict[str, str]:
    return {
        "component": component,
        "runtime_seconds": f"{seconds:.3f}",
        "status": status,
        "detail": detail,
    }


def write_runtime_breakdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["component", "runtime_seconds", "status", "detail"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_gold_rows(
    *,
    evidence_levels: set[str] | None = None,
    qc_statuses: set[str] | None = None,
    allow_opm_derived_gold: bool = False,
    allow_internal_stress_test: bool = False,
) -> list[dict[str, str]]:
    evidence_levels = evidence_levels or env_list("GOLD_EVIDENCE_LEVELS", DEFAULT_EVIDENCE_LEVELS)
    qc_statuses = qc_statuses or env_list("GOLD_QC_STATUSES", DEFAULT_QC_STATUSES)
    with DATASET.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("evidence_level") in evidence_levels and row.get("qc_status") in qc_statuses
        ]
    require_publication_gold_provenance(
        rows,
        dataset=DATASET,
        allow_opm_derived_gold=allow_opm_derived_gold,
        allow_internal_stress_test=allow_internal_stress_test,
    )
    return sorted(rows, key=lambda row: row["record_id"])


def sequence_for(row: dict[str, str]) -> str:
    if row.get("sequence"):
        return re.sub(r"\s+", "", row["sequence"]).upper()

    entity_json = RCSB / f"{row['pdb_id']}_{row['entity_id']}_entity.json"
    with entity_json.open() as handle:
        payload = json.load(handle)
    sequence = payload["entity_poly"]["pdbx_seq_one_letter_code_can"]
    return re.sub(r"\s+", "", sequence).upper()


def wrap_fasta(seq: str, width: int = 80) -> str:
    return "\n".join(seq[index : index + width] for index in range(0, len(seq), width))


def prepare_gold_sequences(rows: list[dict[str, str]]) -> Path:
    input_dir = RUN / "inputs"
    single_dir = input_dir / "single_fasta"
    single_dir.mkdir(parents=True, exist_ok=True)
    fasta = input_dir / "gold_sequences.fasta"

    with fasta.open("w") as out:
        for row in rows:
            seq = sequence_for(row)
            out.write(f">{row['record_id']}\n{wrap_fasta(seq)}\n")
            (single_dir / f"{row['record_id']}.fasta").write_text(
                f">{row['record_id']}\n{wrap_fasta(seq)}\n"
            )
    return fasta


def make_profiles(rows: list[dict[str, str]], fasta: Path) -> tuple[Path, Path]:
    psiblast = shutil.which("psiblast")
    makeblastdb = shutil.which("makeblastdb")
    if not psiblast or not makeblastdb:
        raise RuntimeError("psiblast and makeblastdb are required for profile-based baselines")

    profile_dir = RUN / "profiles"
    proftmb_dir = profile_dir / "proftmb_pssm"
    betaware_dir = profile_dir / "betaware_freq"
    db_prefix = profile_dir / "blastdb" / "gold"
    proftmb_dir.mkdir(parents=True, exist_ok=True)
    betaware_dir.mkdir(parents=True, exist_ok=True)
    db_prefix.parent.mkdir(parents=True, exist_ok=True)
    profile_errors = profile_dir / "profile_errors.tsv"

    if not (db_prefix.with_suffix(".pin")).exists():
        result = run_cmd(
            [makeblastdb, "-in", str(fasta), "-dbtype", "prot", "-out", str(db_prefix)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)

    with profile_errors.open("w", newline="") as error_handle:
        error_writer = csv.DictWriter(
            error_handle,
            fieldnames=["record_id", "stage", "message"],
            delimiter="\t",
        )
        error_writer.writeheader()
        for row in rows:
            record_id = row["record_id"]
            query = RUN / "inputs" / "single_fasta" / f"{record_id}.fasta"
            pssm = proftmb_dir / f"{record_id}.Q"
            if not pssm.exists() or pssm.stat().st_size == 0:
                result = run_cmd(
                    [
                        psiblast,
                        "-query",
                        str(query),
                        "-db",
                        str(db_prefix),
                        "-num_iterations",
                        "2",
                        "-evalue",
                        "0.001",
                        "-out_ascii_pssm",
                        str(pssm),
                        "-out",
                        str(profile_dir / f"{record_id}.psiblast.out"),
                        "-num_threads",
                        os.environ.get("PSIBLAST_THREADS", "2"),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if result.returncode:
                    error_writer.writerow(
                        {
                            "record_id": record_id,
                            "stage": "psiblast",
                            "message": result.stderr.strip()[:500],
                        }
                    )
                    continue
            if not pssm.exists() or pssm.stat().st_size == 0:
                error_writer.writerow(
                    {
                        "record_id": record_id,
                        "stage": "psiblast",
                        "message": "PSSM file was not created.",
                    }
                )
                continue
            try:
                pssm_to_betaware_profile(pssm, betaware_dir / f"{record_id}.prof")
            except Exception as exc:
                error_writer.writerow(
                    {
                        "record_id": record_id,
                        "stage": "betaware_profile",
                        "message": str(exc)[:500],
                    }
                )
    return proftmb_dir, betaware_dir


def pssm_to_betaware_profile(pssm: Path, out_path: Path) -> None:
    rows: list[list[float]] = []
    for line in pssm.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 42 or not parts[0].isdigit():
            continue
        observed_percentages = [float(value) / 100.0 for value in parts[22:42]]
        rows.append(observed_percentages)

    if not rows:
        raise RuntimeError(f"No PSSM rows parsed from {pssm}")

    with out_path.open("w") as handle:
        for values in rows:
            handle.write(" ".join(f"{value:.2f}" for value in values) + "\n")


def parse_betaware_topology(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^(Topology|TMB Strands)\s+:", line):
            continue
        topology = [line.split(":", 1)[1].strip()]
        for continuation in lines[index + 1 :]:
            if not re.match(r"^\s+[0-9,\-\s]+$", continuation):
                break
            topology.append(continuation.strip())
        return re.sub(r"\s+", "", "".join(topology))
    return ""


def run_betaware(rows: list[dict[str, str]], betaware_profiles: Path) -> Path:
    tool = ROOT / "tools" / "betaware" / "betaware.py"
    out_dir = RUN / "betaware"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "betaware_gold_summary.tsv"
    env = os.environ.copy()
    env["BETAWARE_ROOT"] = str(ROOT / "tools" / "betaware")

    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "status", "predicted_tmbb", "score", "strand_count", "topology", "output"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            record_id = row["record_id"]
            out_file = out_dir / f"{record_id}.out"
            profile = betaware_profiles / f"{record_id}.prof"
            if not profile.exists():
                writer.writerow(
                    {
                        "record_id": record_id,
                        "status": "failed",
                        "predicted_tmbb": "",
                        "score": "",
                        "strand_count": "",
                        "topology": "",
                        "output": f"missing profile: {profile}",
                    }
                )
                continue
            result = run_cmd(
                [
                    sys.executable,
                    str(tool),
                    "-f",
                    str(RUN / "inputs" / "single_fasta" / f"{record_id}.fasta"),
                    "-p",
                    str(profile),
                    "-a",
                    PSIBLAST_PROFILE_AA_ORDER,
                    "-s",
                    os.environ.get("BETAWARE_SENSITIVITY", "0.5"),
                    "-t",
                    "-o",
                    str(out_file),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode:
                out_file.with_suffix(".err").write_text(result.stderr)
                writer.writerow(
                    {
                        "record_id": record_id,
                        "status": "failed",
                        "predicted_tmbb": "",
                        "score": "",
                        "strand_count": "",
                        "topology": "",
                        "output": str(out_file),
                    }
                )
                continue

            text = out_file.read_text(errors="replace")
            pred_match = re.search(r"Predicted TMBB\s+:\s+(Yes|No)\s+([0-9.]+)", text)
            topology = parse_betaware_topology(text)
            writer.writerow(
                {
                    "record_id": record_id,
                    "status": "ok",
                    "predicted_tmbb": pred_match.group(1) if pred_match else "",
                    "score": pred_match.group(2) if pred_match else "",
                    "strand_count": len([seg for seg in topology.split(",") if seg]),
                    "topology": topology,
                    "output": str(out_file),
                }
            )
    return summary


def run_juchmme(fasta: Path, mode: str) -> Path:
    out = RUN / "juchmme" / f"pred_tmbb2_{mode}_gold.out"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd(
        [
            "bash",
            str(ROOT / "scripts" / "run_juchmme_pred_tmbb2.sh"),
            mode,
            str(fasta),
            str(out),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(f"JUCHMME {mode} failed:\n{result.stderr}")
    return out


def run_proftmb(profile_dir: Path) -> Path:
    out_prefix = RUN / "proftmb" / "gold"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd(
        [
            "bash",
            str(ROOT / "scripts" / "run_proftmb.sh"),
            str(profile_dir),
            str(out_prefix),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(f"PROFtmb failed:\n{result.stderr}")
    return Path(f"{out_prefix}_proftmb_tabular.txt")


def run_tmbed(fasta: Path) -> tuple[Path, list[dict[str, str]]]:
    out_dir = RUN / "tmbed"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = out_dir / "gold_embeddings.h5"
    predictions = out_dir / "gold.pred"
    py = Path(os.environ.get("TMBED_PYTHON", ROOT / ".conda" / "tmbed" / "bin" / "python"))
    tool = ROOT / "tools" / "TMbed"
    model_dir = out_dir / "prot_t5_model"
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    runtime_rows: list[dict[str, str]] = []

    if not embeddings.exists():
        started = time.perf_counter()
        result = run_cmd(
            [
                str(py),
                "-m",
                "tmbed",
                "embed",
                "-f",
                str(fasta),
                "-e",
                str(embeddings),
                "--no-use-gpu",
                "--threads",
                os.environ.get("TMBED_THREADS", "4"),
                "--batch-size",
                os.environ.get("TMBED_BATCH_SIZE", "1200"),
                "--model-dir",
                str(model_dir),
            ],
            cwd=tool,
            env=env,
            stdout=(out_dir / "embed.stdout").open("w"),
            stderr=(out_dir / "embed.stderr").open("w"),
            timeout=int(os.environ.get("TMBED_TIMEOUT", "7200")),
        )
        runtime_rows.append(
            runtime_row(
                "tmbed_embedding",
                time.perf_counter() - started,
                status="ok" if result.returncode == 0 else "error",
                detail="TMbed ProtT5 embedding stage; skipped only when gold_embeddings.h5 already exists",
            )
        )
        if result.returncode:
            raise RuntimeError(f"TMbed embedding failed; see {out_dir / 'embed.stderr'}")
    else:
        runtime_rows.append(
            runtime_row(
                "tmbed_embedding",
                0.0,
                status="cached",
                detail=f"reused existing embedding cache: {embeddings}",
            )
        )

    started = time.perf_counter()
    result = run_cmd(
        [
            str(py),
            "-m",
            "tmbed",
            "predict",
            "-f",
            str(fasta),
            "-e",
            str(embeddings),
            "-p",
            str(predictions),
            "--out-format",
            "1",
            "--no-use-gpu",
            "--threads",
            os.environ.get("TMBED_THREADS", "4"),
        ],
        cwd=tool,
        env=env,
        stdout=(out_dir / "predict.stdout").open("w"),
        stderr=(out_dir / "predict.stderr").open("w"),
        timeout=int(os.environ.get("TMBED_TIMEOUT", "7200")),
    )
    runtime_rows.append(
        runtime_row(
            "tmbed_prediction",
            time.perf_counter() - started,
            status="ok" if result.returncode == 0 else "error",
            detail="TMbed prediction from embeddings",
        )
    )
    if result.returncode:
        raise RuntimeError(f"TMbed prediction failed; see {out_dir / 'predict.stderr'}")
    runtime_rows.append(
        runtime_row(
            "tmbed_total",
            sum(float(row["runtime_seconds"]) for row in runtime_rows),
            detail="TMbed embedding plus prediction",
        )
    )
    return predictions, runtime_rows


def filter_pdb_chain(pdb_path: Path, chain_id: str, out_path: Path) -> int:
    count = 0
    saw_model = False
    in_first_model = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pdb_path.open(errors="replace") as src, out_path.open("w") as dst:
        for line in src:
            if line.startswith("MODEL"):
                if saw_model:
                    break
                saw_model = True
                in_first_model = True
                dst.write(line)
                continue
            if line.startswith("ENDMDL"):
                if saw_model:
                    dst.write(line)
                    break
                continue
            if saw_model and not in_first_model:
                continue
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21 and line[21] == chain_id:
                dst.write(line)
                count += 1
        dst.write("END\n")
    return count


def row_structure_path(row: dict[str, str]) -> Path | None:
    for field in ("structure_path", "afdb_model_path"):
        value = row.get(field, "").strip()
        if value:
            return resolve_repo_path(value)
    return None


def row_structure_chain(row: dict[str, str]) -> str:
    for field in ("structure_chain_id", "afdb_chain_id", "auth_chain_id", "asym_id"):
        value = row.get(field, "").strip()
        if value:
            return value
    return "A"


def row_pdb_chain_id(row: dict[str, str]) -> str:
    """Return a fixed-column PDB chain ID for tools that cannot read mmCIF IDs."""
    for field in ("structure_chain_id", "afdb_chain_id", "auth_chain_id", "asym_id"):
        value = row.get(field, "").strip()
        if len(value) == 1:
            return value
    for field in ("structure_chain_id", "afdb_chain_id", "auth_chain_id", "asym_id"):
        value = row.get(field, "").strip()
        if value:
            return value[:1]
    return "A"


def parse_polarbearal_stdout(stdout_path: Path, pdb_id: str) -> dict[str, str]:
    if not stdout_path.exists():
        return {"strand_count": "", "axis_length": "", "avg_radius": ""}
    text = stdout_path.read_text(errors="replace")
    match = re.search(rf"^{pdb_id}\t(\d+)\t([0-9.]+)\t([0-9.]+)", text, re.MULTILINE)
    if not match:
        return {"strand_count": "", "axis_length": "", "avg_radius": ""}
    return {
        "strand_count": match.group(1),
        "axis_length": match.group(2),
        "avg_radius": match.group(3),
    }


def parse_polarbearal_strands(out_dir: Path) -> str:
    strand_files = sorted((out_dir / "betaBarrelStrands").glob("*.txt"))
    if not strand_files:
        return ""
    strands: set[str] = set()
    with strand_files[0].open(errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            strand = row.get("_res_strand", "")
            if strand:
                strands.add(strand)
    return str(len(strands)) if strands else ""


def single_line_message(value: object, *, limit: int = 500) -> str:
    text = re.sub(r"[\t\r\n]+", " ", str(value or ""))
    text = re.sub(r" {2,}", " ", text).strip()
    return text[:limit]


def load_polarbearal_previous(summary: Path) -> dict[str, dict[str, str]]:
    if not summary.exists() or os.environ.get("POLARBEARAL_REUSE", "1") != "1":
        return {}
    with summary.open(newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def run_polarbearal(rows: list[dict[str, str]]) -> Path:
    pdb_cache = RUN / "pdb_cache"
    chain_dir = RUN / "polar_bearal3" / "chain_inputs"
    out_root = RUN / "polar_bearal3" / "outputs"
    summary = RUN / "polar_bearal3" / "polarbearal3_gold_summary.tsv"
    pdb_cache.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    previous = load_polarbearal_previous(summary)

    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "status", "strand_count", "axis_length", "avg_radius", "output_dir", "message"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            record_id = row["record_id"]
            structure_path = row_structure_path(row)
            if structure_path is None:
                pdb_id = row["pdb_id"].upper()
                pdb = pdb_cache / f"{pdb_id}.pdb"
                parse_id = pdb_id
                chain_id = row_pdb_chain_id(row)
            else:
                pdb = structure_path
                parse_id = record_id.upper()
                chain_id = row_pdb_chain_id(row)
            if not pdb.exists() and structure_path is None:
                try:
                    urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb)
                except Exception as exc:
                    writer.writerow(
                        {
                            "record_id": record_id,
                            "status": "failed",
                            "strand_count": "",
                            "axis_length": "",
                            "avg_radius": "",
                            "output_dir": str(out_root / record_id),
                            "message": single_line_message(f"PDB download failed: {exc}"),
                        }
                    )
                    handle.flush()
                    continue
            if not pdb.exists():
                writer.writerow(
                    {
                        "record_id": record_id,
                        "status": "failed",
                        "strand_count": "",
                        "axis_length": "",
                        "avg_radius": "",
                        "output_dir": str(out_root / record_id),
                        "message": single_line_message(f"structure file not found: {pdb}"),
                    }
                )
                handle.flush()
                continue

            filtered = chain_dir / record_id / f"{parse_id}.pdb"
            atom_count = filter_pdb_chain(pdb, chain_id, filtered)
            if atom_count == 0:
                writer.writerow(
                    {
                        "record_id": record_id,
                        "status": "failed",
                        "strand_count": "",
                        "axis_length": "",
                        "avg_radius": "",
                        "output_dir": str(out_root / record_id),
                        "message": single_line_message("no atoms for requested chain in PDB file"),
                    }
                )
                continue

            out_dir = out_root / record_id
            stdout_path = out_dir / "polarbearal.stdout"
            stderr_path = out_dir / "polarbearal.stderr"
            previous_row = previous.get(record_id)
            if previous_row and previous_row.get("status") == "timeout" and not parse_polarbearal_strands(out_dir):
                previous_row["message"] = single_line_message(previous_row.get("message", ""))
                writer.writerow(previous_row)
                handle.flush()
                continue

            reused = False
            if os.environ.get("POLARBEARAL_REUSE", "1") == "1" and parse_polarbearal_strands(out_dir):
                reused = True
                result = subprocess.CompletedProcess([], 0, "", "")
            else:
                env = os.environ.copy()
                env["POLARBEARAL_STDOUT"] = str(stdout_path)
                env["POLARBEARAL_STDERR"] = str(stderr_path)
                cmd = ["bash", str(ROOT / "scripts" / "run_polarbearal.sh"), str(filtered), str(out_dir)]
                timeout_seconds = int(os.environ.get("POLARBEARAL_TIMEOUT", "300"))
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = proc.communicate(timeout=timeout_seconds)
                    result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.communicate()
                    writer.writerow(
                        {
                            "record_id": record_id,
                            "status": "timeout",
                            "strand_count": "",
                            "axis_length": "",
                            "avg_radius": "",
                            "output_dir": str(out_dir),
                            "message": single_line_message(f"timed out after {timeout_seconds} seconds"),
                        }
                    )
                    handle.flush()
                    continue

            parsed = parse_polarbearal_stdout(stdout_path, parse_id)
            if not parsed["strand_count"]:
                parsed["strand_count"] = parse_polarbearal_strands(out_dir)
            ok = result.returncode == 0 and bool(parsed["strand_count"])
            message = ""
            if result.returncode != 0:
                message = stderr_path.read_text(errors="replace").strip() if stderr_path.exists() else ""
                if not message:
                    message = result.stderr.strip()
            writer.writerow(
                {
                    "record_id": record_id,
                    "status": "ok_reused" if ok and reused else "ok" if ok else "failed",
                    "strand_count": parsed["strand_count"],
                    "axis_length": parsed["axis_length"],
                    "avg_radius": parsed["avg_radius"],
                    "output_dir": str(out_dir),
                    "message": single_line_message(message),
                }
            )
            handle.flush()
    return summary


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def profile_input_outputs(fasta: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    return (
        {"gold_fasta": fasta},
        {
            "proftmb_profiles": RUN / "profiles" / "proftmb_pssm",
            "betaware_profiles": RUN / "profiles" / "betaware_freq",
            "profile_errors": RUN / "profiles" / "profile_errors.tsv",
        },
    )


def method_manifest(
    *,
    method: str,
    command: object,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    seconds: float,
    status: str = "ok",
    message: str = "",
    parameters: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_external_method_manifest(
        method=method,
        command=command,  # type: ignore[arg-type]
        inputs=inputs,
        outputs=outputs,
        env=os.environ,
        parameters=parameters,
        status=status,
        runtime_seconds=seconds,
        message=single_line_message(message, limit=1000),
        extra=extra,
    )


def betaware_command_template(betaware_profiles: Path | None) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools" / "betaware" / "betaware.py"),
        "-f",
        str(RUN / "inputs" / "single_fasta" / "<record_id>.fasta"),
        "-p",
        str((betaware_profiles or RUN / "profiles" / "betaware_freq") / "<record_id>.prof"),
        "-a",
        PSIBLAST_PROFILE_AA_ORDER,
        "-s",
        os.environ.get("BETAWARE_SENSITIVITY", "0.5"),
        "-t",
        "-o",
        str(RUN / "betaware" / "<record_id>.out"),
    ]


def tmbed_command(fasta: Path) -> dict[str, list[str]]:
    py = Path(os.environ.get("TMBED_PYTHON", ROOT / ".conda" / "tmbed" / "bin" / "python"))
    embeddings = RUN / "tmbed" / "gold_embeddings.h5"
    predictions = RUN / "tmbed" / "gold.pred"
    model_dir = RUN / "tmbed" / "prot_t5_model"
    return {
        "embed": [
            str(py),
            "-m",
            "tmbed",
            "embed",
            "-f",
            str(fasta),
            "-e",
            str(embeddings),
            "--no-use-gpu",
            "--threads",
            os.environ.get("TMBED_THREADS", "4"),
            "--batch-size",
            os.environ.get("TMBED_BATCH_SIZE", "1200"),
            "--model-dir",
            str(model_dir),
        ],
        "predict": [
            str(py),
            "-m",
            "tmbed",
            "predict",
            "-f",
            str(fasta),
            "-e",
            str(embeddings),
            "-p",
            str(predictions),
            "--out-format",
            "1",
            "--no-use-gpu",
            "--threads",
            os.environ.get("TMBED_THREADS", "4"),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    global DATASET, RUN

    args = parse_args(argv)
    RUN = args.run_dir.expanduser().resolve()
    DATASET = args.dataset.expanduser().resolve()
    evidence_levels = split_values(args.evidence_level, DEFAULT_EVIDENCE_LEVELS)
    qc_statuses = split_values(args.qc_status, DEFAULT_QC_STATUSES)
    os.environ["GOLD_EVIDENCE_LEVELS"] = ",".join(sorted(evidence_levels))
    os.environ["GOLD_QC_STATUSES"] = ",".join(sorted(qc_statuses))
    try:
        methods = selected_methods(args.method)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    total_started = time.perf_counter()
    start = iso_now()
    runtime_rows: list[dict[str, str]] = []
    error_keys: list[str] = []
    method_runs: dict[str, dict[str, object]] = {}
    RUN.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        rows = load_gold_rows(
            evidence_levels=evidence_levels,
            qc_statuses=qc_statuses,
            allow_opm_derived_gold=bool(args.allow_opm_derived_gold),
            allow_internal_stress_test=bool(args.allow_internal_stress_test),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    fasta = prepare_gold_sequences(rows)
    runtime_rows.append(
        runtime_row(
            "prepare_sequences",
            time.perf_counter() - started,
            detail="load dataset rows and write combined/single FASTA inputs",
        )
    )
    proftmb_profiles: Path | None = None
    betaware_profiles: Path | None = None

    results: dict[str, object] = {
        "started_at": start,
        "dataset": str(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "evidence_levels": ",".join(sorted(evidence_levels)),
        "qc_statuses": ",".join(sorted(qc_statuses)),
        "methods": ",".join(sorted(methods)),
        "gold_rows": str(len(rows)),
        "gold_fasta": str(fasta),
    }

    if not rows:
        results["dataset_error"] = "no rows matched requested evidence/QC filters"
        error_keys.append("dataset_error")

    if methods.intersection(PROFILE_METHODS):
        started = time.perf_counter()
        try:
            proftmb_profiles, betaware_profiles = make_profiles(rows, fasta)
            profile_seconds = time.perf_counter() - started
            results["proftmb_profiles"] = str(proftmb_profiles)
            results["betaware_profiles"] = str(betaware_profiles)
            results["profile_database_scope"] = "diagnostic_self_fasta"
            profile_inputs, profile_outputs = profile_input_outputs(fasta)
            method_runs["psiblast_profiles"] = method_manifest(
                method="psiblast_profiles",
                command={
                    "makeblastdb": [
                        "makeblastdb",
                        "-in",
                        str(fasta),
                        "-dbtype",
                        "prot",
                        "-out",
                        str(RUN / "profiles" / "blastdb" / "gold"),
                    ],
                    "psiblast": [
                        "psiblast",
                        "-query",
                        str(RUN / "inputs" / "single_fasta" / "<record_id>.fasta"),
                        "-db",
                        str(RUN / "profiles" / "blastdb" / "gold"),
                        "-num_iterations",
                        "2",
                        "-evalue",
                        "0.001",
                        "-out_ascii_pssm",
                        str(RUN / "profiles" / "proftmb_pssm" / "<record_id>.Q"),
                        "-num_threads",
                        os.environ.get("PSIBLAST_THREADS", "2"),
                    ],
                },
                inputs=profile_inputs,
                outputs=profile_outputs,
                seconds=profile_seconds,
                extra={"profile_database_scope": "diagnostic_self_fasta"},
            )
            runtime_rows.append(
                runtime_row(
                    "psiblast_profiles",
                    profile_seconds,
                    detail=(
                        "make BLAST DB from selected gold FASTA, run PSI-BLAST profiles, "
                        "and convert BETAWARE profile inputs"
                    ),
                )
            )
        except Exception as exc:
            profile_seconds = time.perf_counter() - started
            results["psiblast_profiles_error"] = str(exc)
            error_keys.append("psiblast_profiles_error")
            profile_inputs, profile_outputs = profile_input_outputs(fasta)
            method_runs["psiblast_profiles"] = method_manifest(
                method="psiblast_profiles",
                command="makeblastdb + per-record psiblast profile generation",
                inputs=profile_inputs,
                outputs=profile_outputs,
                seconds=profile_seconds,
                status="error",
                message=str(exc),
                extra={"profile_database_scope": "diagnostic_self_fasta"},
            )
            runtime_rows.append(
                runtime_row(
                    "psiblast_profiles",
                    profile_seconds,
                    status="error",
                    detail=str(exc)[:500],
                )
            )

    if "betaware" in methods:
        started = time.perf_counter()
        try:
            if betaware_profiles is None:
                raise RuntimeError("BETAWARE profiles were not generated")
            betaware_summary = run_betaware(rows, betaware_profiles)
            betaware_seconds = time.perf_counter() - started
            results["betaware_summary"] = str(betaware_summary)
            method_runs["betaware"] = method_manifest(
                method="betaware",
                command=betaware_command_template(betaware_profiles),
                inputs={"gold_fasta": fasta, "betaware_profiles": betaware_profiles},
                outputs={"summary": betaware_summary, "per_record_outputs": RUN / "betaware"},
                seconds=betaware_seconds,
                parameters={"command_scope": "per_record_template"},
            )
            runtime_rows.append(
                runtime_row("betaware", betaware_seconds, detail="BETAWARE after profile generation")
            )
        except Exception as exc:
            betaware_seconds = time.perf_counter() - started
            results["betaware_error"] = str(exc)
            error_keys.append("betaware_error")
            method_runs["betaware"] = method_manifest(
                method="betaware",
                command=betaware_command_template(betaware_profiles),
                inputs={"gold_fasta": fasta, "betaware_profiles": betaware_profiles or RUN / "profiles" / "betaware_freq"},
                outputs={"summary": RUN / "betaware" / "betaware_gold_summary.tsv"},
                seconds=betaware_seconds,
                status="error",
                message=str(exc),
                parameters={"command_scope": "per_record_template"},
            )
            runtime_rows.append(
                runtime_row("betaware", betaware_seconds, status="error", detail=str(exc)[:500])
            )

    for mode, method_name in (("hmm", "pred_tmbb2_hmm"), ("hnn", "pred_tmbb2_hnn")):
        if method_name not in methods:
            continue
        started = time.perf_counter()
        out_path = RUN / "juchmme" / f"pred_tmbb2_{mode}_gold.out"
        cmd = [
            "bash",
            str(ROOT / "scripts" / "run_juchmme_pred_tmbb2.sh"),
            mode,
            str(fasta),
            str(out_path),
        ]
        try:
            out_path = run_juchmme(fasta, mode)
            juchmme_seconds = time.perf_counter() - started
            results[f"pred_tmbb2_{mode}"] = str(out_path)
            method_runs[method_name] = method_manifest(
                method=method_name,
                command=cmd,
                inputs={"gold_fasta": fasta},
                outputs={"predictions": out_path, "stderr": out_path.with_suffix(out_path.suffix + ".err")},
                seconds=juchmme_seconds,
                parameters={"mode": mode},
            )
            runtime_rows.append(
                runtime_row(
                    f"pred_tmbb2_{mode}",
                    juchmme_seconds,
                    detail=f"JUCHMME PRED-TMBB2 {mode} mode",
                )
            )
        except Exception as exc:
            juchmme_seconds = time.perf_counter() - started
            results[f"pred_tmbb2_{mode}_error"] = str(exc)
            error_keys.append(f"pred_tmbb2_{mode}_error")
            method_runs[method_name] = method_manifest(
                method=method_name,
                command=cmd,
                inputs={"gold_fasta": fasta},
                outputs={"predictions": out_path, "stderr": out_path.with_suffix(out_path.suffix + ".err")},
                seconds=juchmme_seconds,
                status="error",
                message=str(exc),
                parameters={"mode": mode},
            )
            runtime_rows.append(
                runtime_row(
                    f"pred_tmbb2_{mode}",
                    juchmme_seconds,
                    status="error",
                    detail=str(exc)[:500],
                )
            )

    if "proftmb" in methods:
        started = time.perf_counter()
        out_prefix = RUN / "proftmb" / "gold"
        cmd = ["bash", str(ROOT / "scripts" / "run_proftmb.sh"), str(proftmb_profiles or RUN / "profiles" / "proftmb_pssm"), str(out_prefix)]
        proftmb_tabular = Path(f"{out_prefix}_proftmb_tabular.txt")
        try:
            if proftmb_profiles is None:
                raise RuntimeError("PROFtmb profiles were not generated")
            proftmb_tabular = run_proftmb(proftmb_profiles)
            proftmb_seconds = time.perf_counter() - started
            results["proftmb_tabular"] = str(proftmb_tabular)
            method_runs["proftmb"] = method_manifest(
                method="proftmb",
                command=cmd,
                inputs={"profiles": proftmb_profiles},
                outputs={"tabular": proftmb_tabular},
                seconds=proftmb_seconds,
            )
            runtime_rows.append(
                runtime_row("proftmb", proftmb_seconds, detail="PROFtmb after profile generation")
            )
        except Exception as exc:
            proftmb_seconds = time.perf_counter() - started
            results["proftmb_error"] = str(exc)
            error_keys.append("proftmb_error")
            method_runs["proftmb"] = method_manifest(
                method="proftmb",
                command=cmd,
                inputs={"profiles": proftmb_profiles or RUN / "profiles" / "proftmb_pssm"},
                outputs={"tabular": proftmb_tabular},
                seconds=proftmb_seconds,
                status="error",
                message=str(exc),
            )
            runtime_rows.append(
                runtime_row("proftmb", proftmb_seconds, status="error", detail=str(exc)[:500])
            )

    if "tmbed" in methods:
        started = time.perf_counter()
        tmbed_predictions = RUN / "tmbed" / "gold.pred"
        try:
            tmbed_predictions, tmbed_runtime_rows = run_tmbed(fasta)
            tmbed_seconds = time.perf_counter() - started
            results["tmbed_predictions"] = str(tmbed_predictions)
            method_runs["tmbed"] = method_manifest(
                method="tmbed",
                command=tmbed_command(fasta),
                inputs={"gold_fasta": fasta, "embeddings": RUN / "tmbed" / "gold_embeddings.h5"},
                outputs={
                    "predictions": tmbed_predictions,
                    "embed_stdout": RUN / "tmbed" / "embed.stdout",
                    "embed_stderr": RUN / "tmbed" / "embed.stderr",
                    "predict_stdout": RUN / "tmbed" / "predict.stdout",
                    "predict_stderr": RUN / "tmbed" / "predict.stderr",
                },
                seconds=tmbed_seconds,
                extra={"stage_runtimes": tmbed_runtime_rows},
            )
            runtime_rows.extend(tmbed_runtime_rows)
            runtime_rows.append(
                runtime_row("tmbed_wrapper", tmbed_seconds, detail="wall time around complete TMbed wrapper")
            )
        except Exception as exc:
            tmbed_seconds = time.perf_counter() - started
            results["tmbed_error"] = str(exc)
            error_keys.append("tmbed_error")
            method_runs["tmbed"] = method_manifest(
                method="tmbed",
                command=tmbed_command(fasta),
                inputs={"gold_fasta": fasta, "embeddings": RUN / "tmbed" / "gold_embeddings.h5"},
                outputs={"predictions": tmbed_predictions},
                seconds=tmbed_seconds,
                status="error",
                message=str(exc),
            )
            runtime_rows.append(
                runtime_row("tmbed_wrapper", tmbed_seconds, status="error", detail=str(exc)[:500])
            )

    if "polarbearal3" in methods:
        started = time.perf_counter()
        polar_summary = RUN / "polar_bearal3" / "polarbearal3_gold_summary.tsv"
        polar_cmd = [
            "bash",
            str(ROOT / "scripts" / "run_polarbearal.sh"),
            str(RUN / "polar_bearal3" / "chain_inputs" / "<record_id>" / "<parse_id>.pdb"),
            str(RUN / "polar_bearal3" / "outputs" / "<record_id>"),
        ]
        try:
            polar_summary = run_polarbearal(rows)
            polar_seconds = time.perf_counter() - started
            results["polarbearal_summary"] = str(polar_summary)
            method_runs["polarbearal3"] = method_manifest(
                method="polarbearal3",
                command=polar_cmd,
                inputs={"gold_dataset": DATASET, "chain_inputs": RUN / "polar_bearal3" / "chain_inputs"},
                outputs={"summary": polar_summary, "outputs": RUN / "polar_bearal3" / "outputs"},
                seconds=polar_seconds,
                parameters={"command_scope": "per_record_template"},
            )
            runtime_rows.append(
                runtime_row("polarbearal3", polar_seconds, detail="PolarBearal3 chain-level run")
            )
        except Exception as exc:
            polar_seconds = time.perf_counter() - started
            results["polarbearal_error"] = str(exc)
            error_keys.append("polarbearal_error")
            method_runs["polarbearal3"] = method_manifest(
                method="polarbearal3",
                command=polar_cmd,
                inputs={"gold_dataset": DATASET, "chain_inputs": RUN / "polar_bearal3" / "chain_inputs"},
                outputs={"summary": polar_summary, "outputs": RUN / "polar_bearal3" / "outputs"},
                seconds=polar_seconds,
                status="error",
                message=str(exc),
                parameters={"command_scope": "per_record_template"},
            )
            runtime_rows.append(
                runtime_row("polarbearal3", polar_seconds, status="error", detail=str(exc)[:500])
            )

    total_runtime = time.perf_counter() - total_started
    runtime_rows.append(runtime_row("total", total_runtime, detail="end-to-end run_gold_dataset wall runtime"))
    runtime_breakdown = RUN / "runtime_breakdown.tsv"
    write_runtime_breakdown(runtime_breakdown, runtime_rows)
    results["finished_at"] = iso_now()
    results["runtime_seconds"] = f"{total_runtime:.3f}"
    results["runtime_breakdown"] = str(runtime_breakdown)
    results["error_keys"] = ",".join(error_keys)
    results["method_manifests"] = method_runs
    manifest = RUN / "gold_run_manifest.json"
    structured_manifest = RUN / "gold_external_baseline_manifest.json"
    results["structured_manifest"] = str(structured_manifest)
    manifest.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    structured_payload = build_external_baseline_manifest(
        run_name="beta_barrel_staves_gold_external_baselines",
        status="error" if error_keys else "ok",
        command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        inputs={"dataset": DATASET, "gold_fasta": fasta},
        outputs={
            "legacy_manifest": manifest,
            "runtime_breakdown": runtime_breakdown,
            "betaware_summary": RUN / "betaware" / "betaware_gold_summary.tsv",
            "pred_tmbb2_hmm": RUN / "juchmme" / "pred_tmbb2_hmm_gold.out",
            "pred_tmbb2_hnn": RUN / "juchmme" / "pred_tmbb2_hnn_gold.out",
            "proftmb_tabular": RUN / "proftmb" / "gold_proftmb_tabular.txt",
            "tmbed_predictions": RUN / "tmbed" / "gold.pred",
            "polarbearal3_summary": RUN / "polar_bearal3" / "polarbearal3_gold_summary.tsv",
        },
        env=os.environ,
        methods=method_runs,
        parameters={
            "evidence_levels": sorted(evidence_levels),
            "qc_statuses": sorted(qc_statuses),
            "methods": sorted(methods),
            "allow_opm_derived_gold": bool(args.allow_opm_derived_gold),
            "allow_internal_stress_test": bool(args.allow_internal_stress_test),
            "allow_errors": bool(args.allow_errors),
        },
        metrics={"gold_rows": len(rows), "runtime_seconds": round(total_runtime, 6)},
        extra={"error_keys": error_keys},
    )
    write_json(structured_manifest, structured_payload)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if args.allow_errors or not error_keys else 1


if __name__ == "__main__":
    raise SystemExit(main())
