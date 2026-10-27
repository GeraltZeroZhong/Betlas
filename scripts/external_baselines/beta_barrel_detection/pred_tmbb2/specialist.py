"""Run one declared polymer sequence and retain a topology-derived hard call."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from .runner import run_baseline
from .sequences import declared_polymer_sequence


def run_sample(sequence: str, *, sample_id: str, sequence_source: str,
               output_dir: str | Path, juchmme_dir: str | Path,
               java_executable: str = "java", timeout: float = 120.0) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"method": "pred_tmbb2_single_juchmme", "sample_id": sample_id,
              "task": "membrane_beta_barrel", "input_regime": "sequence", "status": "unavailable",
              "score": None, "prediction": None, "reason": "", "sequence_source": sequence_source,
              "score_semantics": "native_topology_count_hard_call_only",
              "native_scope": "transmembrane_beta_barrel_topology",
              "juchmme_dir": str(Path(juchmme_dir).resolve())}
    try:
        if not sample_id or any(c.isspace() for c in sample_id) or sample_id.startswith(">"):
            raise ValueError("Sample ID must be a single FASTA identifier")
        sequence = "".join(sequence.split()).upper()
        if not sequence_source.strip():
            raise ValueError("Explicit sequence provenance is required")
        if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("JUCHMME requires an unambiguous canonical amino-acid sequence")
        result["sequence_length"] = len(sequence)
        fasta = output / "input.fasta"
        fasta.write_text(f">{sample_id}\n{sequence}\n")
        rows = run_baseline(fasta, juchmme_dir=juchmme_dir, work_dir=output / "native",
                            output_path=output / "native.csv", java_executable=java_executable,
                            prediction_field="LP", min_tm_strands=3, timeout=timeout)
        if len(rows) != 1 or rows[0].sample_id != sample_id:
            raise ValueError("Native output does not match the single supplied sequence")
        native = rows[0]
        if len(native.topology) != len(sequence):
            raise ValueError("Native topology length does not match the supplied sequence")
        result.update(status="ok", prediction=int(native.result == "BARREL"),
                      native_result=native.as_row())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result.update(status="unavailable", prediction=None, score=None, reason=str(error))
    result["elapsed_seconds"] = time.monotonic() - started
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", help="PDB/mmCIF containing declared SEQRES/entity polymer sequence")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--juchmme-dir", required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        sequence = declared_polymer_sequence(args.structure, chain_id=args.chain)
    except (OSError, ValueError, KeyError) as error:
        output = Path(args.out_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        result = {"method": "pred_tmbb2_single_juchmme", "sample_id": args.sample_id,
                  "task": "membrane_beta_barrel", "input_regime": "sequence", "status": "unavailable",
                  "prediction": None, "score": None, "reason": str(error)}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2
    result = run_sample(sequence["sequence"], sample_id=args.sample_id,
                        sequence_source=f"{sequence['sequence_source']}:{sequence['source_path']}:{args.chain}",
                        output_dir=args.out_dir, juchmme_dir=args.juchmme_dir,
                        java_executable=args.java, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
