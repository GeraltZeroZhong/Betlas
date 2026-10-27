"""Run native IsItABarrel on one exact structure-derived contact map."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from Bio.PDB.PDBExceptions import PDBConstructionException, PDBException

from scripts.external_baselines.beta_barrel_detection.cooper_beta.inputs import select_structure

from .contact_maps import generate_structure_contact_maps
from .runner import run_baseline


def run_sample(structure_path: str | Path, *, sample_id: str, chain_id: str,
               output_dir: str | Path, script_path: str | Path,
               python_executable: str = sys.executable, model_index: int = 0,
               timeout: float = 120.0) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"method": "isitabarrel_structure_map", "sample_id": sample_id,
              "task": "barrel_geometry", "input_regime": "coordinate_contact_map",
              "status": "unavailable", "score": None, "prediction": None, "reason": "",
              "score_semantics": "hybrid_native_score_hard_call_only",
              "native_scope": "bacterial_transmembrane_beta_barrel_prediction",
              "script_path": str(Path(script_path).resolve()),
              "contact_map": {"atom": "CA", "cutoff_angstrom": 8.0, "local_exclusion": 2,
                              "index_policy": "observed_ca_residue_order"}}
    try:
        selected, selection = select_structure(structure_path, output / "input", chain_id=chain_id,
                                               model_index=model_index)
        result["input"] = selection
        generated = generate_structure_contact_maps(selected, output / "contact_maps")
        if len(generated.records) != 1 or generated.records[0].chain_id != chain_id:
            raise ValueError("Contact-map generation did not produce exactly the selected author chain")
        native_id = generated.records[0].sample_id
        rows = run_baseline(generated.protid_list_path, generated.map_dir, script_path=script_path,
                            work_dir=output / "native", output_path=output / "native.csv",
                            python_executable=python_executable, timeout=timeout)
        if len(rows) != 1 or rows[0].sample_id != native_id:
            raise ValueError("Native output does not match the single selected contact map")
        native = rows[0]
        result.update(status="ok", prediction=int(native.result == "BARREL"),
                      native_result=native.as_row())
    except (OSError, ValueError, RuntimeError, PDBException, PDBConstructionException,
            subprocess.TimeoutExpired) as error:
        result.update(status="unavailable", prediction=None, score=None, reason=str(error))
    result["elapsed_seconds"] = time.monotonic() - started
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    result = run_sample(args.structure, sample_id=args.sample_id, chain_id=args.chain,
                        output_dir=args.out_dir, script_path=args.script,
                        python_executable=args.python, model_index=args.model_index, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
