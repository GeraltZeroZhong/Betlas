#!/usr/bin/env python
"""Build aligned ESM-C mean embeddings for a beta-barrel detection cohort."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import MMCIFParser
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betlas.assets import resolve_esmc_weights  # noqa: E402

DEFAULT_COHORT = REPO_ROOT / "data/readouts/beta_barrel_detection/full_mpstruc_767_neg800/benchmark_cohort.csv"
DEFAULT_OUT_NPZ = (
    REPO_ROOT
    / "data/readouts/beta_barrel_detection/full_mpstruc_767_neg800/"
    "betlas_151_layer_radial16_official/esmc_mean_embeddings_aligned.npz"
)
DEFAULT_WEIGHTS = REPO_ROOT / "data/external/esmc_weights/esmc_600m_2024_12_v0.pth"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_chain(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def residue_to_one_letter(residue: Any) -> str:
    hetfield = residue.id[0] if isinstance(residue.id, tuple) else ""
    if str(hetfield).strip():
        return ""
    resname = str(residue.get_resname()).upper().strip()
    return protein_letters_3to1_extended.get(resname, "X")


def chain_sequence(chain: Any) -> str:
    letters = [residue_to_one_letter(residue) for residue in chain.get_residues()]
    return "".join(letter for letter in letters if letter)


def candidate_chains(row: pd.Series) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for column in ["selected_chain_id", "label_chain_id", "fallback_chain_id"]:
        chain_id = safe_chain(row.get(column, ""))
        if chain_id and chain_id not in seen:
            seen.add(chain_id)
            out.append(chain_id)
    return out


def extract_sequence(row: pd.Series, *, min_residues: int) -> dict[str, Any]:
    pdb_id = str(row["pdb_id"])
    structure_path = Path(str(row["structure_path"]))
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(structure_path))
        model = next(structure.get_models(), None)
    except Exception as exc:
        return {
            "record_id": row["record_id"],
            "pdb_id": pdb_id,
            "filename": row["filename"],
            "status": "parse_error",
            "sequence": "",
            "sequence_length": 0,
            "chain_id": "",
            "reason": str(exc)[:240],
        }
    if model is None:
        return {
            "record_id": row["record_id"],
            "pdb_id": pdb_id,
            "filename": row["filename"],
            "status": "no_model",
            "sequence": "",
            "sequence_length": 0,
            "chain_id": "",
            "reason": "no model",
        }

    by_chain = {str(chain.id): chain_sequence(chain) for chain in model}
    selected = ""
    sequence = ""
    for chain_id in candidate_chains(row):
        seq = by_chain.get(chain_id, "")
        if len(seq) >= min_residues:
            selected = chain_id
            sequence = seq
            break
    if not sequence:
        candidates = sorted(
            ((chain_id, seq) for chain_id, seq in by_chain.items() if len(seq) >= min_residues),
            key=lambda item: (-len(item[1]), item[0]),
        )
        if candidates:
            selected, sequence = candidates[0]

    if not sequence:
        return {
            "record_id": row["record_id"],
            "pdb_id": pdb_id,
            "filename": row["filename"],
            "status": "no_sequence",
            "sequence": "",
            "sequence_length": 0,
            "chain_id": "",
            "reason": f"no chain with >= {min_residues} residues",
        }
    return {
        "record_id": row["record_id"],
        "pdb_id": pdb_id,
        "filename": row["filename"],
        "status": "ok",
        "sequence": sequence,
        "sequence_length": len(sequence),
        "chain_id": selected,
        "reason": "",
    }


def extract_sequences(cohort: pd.DataFrame, *, min_residues: int) -> pd.DataFrame:
    rows = [extract_sequence(row, min_residues=min_residues) for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Extracting sequences")]
    return pd.DataFrame(rows)


def load_esmc_model(*, weights_path: Path, device: str, precision: str):
    import torch
    from esm.models.esmc import ESMC
    from esm.tokenization import EsmSequenceTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmSequenceTokenizer()
    model = ESMC(tokenizer=tokenizer, d_model=1152, n_layers=36, n_heads=18).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {key.replace("module.", "").replace("model.", ""): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=False)
    if precision == "fp16" and str(device).startswith("cuda"):
        model = model.half()
    elif precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    model.eval()
    return model, tokenizer, device


def build_embeddings(
    sequence_meta: pd.DataFrame,
    *,
    weights_path: Path,
    device: str,
    precision: str,
    max_length: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    import torch
    from esm.sdk.api import ESMProtein

    model, _tokenizer, resolved_device = load_esmc_model(weights_path=weights_path, device=device, precision=precision)
    ok = sequence_meta.loc[sequence_meta["status"].astype(str) == "ok"].reset_index(drop=True)
    embeddings: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for row in tqdm(ok.itertuples(index=False), total=len(ok), desc="ESM-C embeddings"):
            raw_sequence = str(row.sequence)
            sequence = raw_sequence[:max_length]
            protein = ESMProtein(sequence=sequence)
            tokenized = model.encode(protein).sequence.unsqueeze(0).to(resolved_device)
            out = model(tokenized)
            residue = out.embeddings[0, 1:-1].detach().float().cpu()
            mean = residue.mean(dim=0).numpy().astype(np.float32)
            if not np.all(np.isfinite(mean)):
                mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            embeddings.append(mean)
            rows.append(
                {
                    "record_id": row.record_id,
                    "pdb_id": row.pdb_id,
                    "filename": row.filename,
                    "chain_id": row.chain_id,
                    "sequence_length": int(row.sequence_length),
                    "embedded_length": len(sequence),
                    "truncated": int(len(raw_sequence) > len(sequence)),
                    "embedding_index": len(embeddings) - 1,
                }
            )
    matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 1152), dtype=np.float32)
    details = {
        "embedding_runtime_seconds": round(time.perf_counter() - start, 3),
        "device": str(resolved_device),
        "precision": precision,
        "max_length": int(max_length),
    }
    return matrix, pd.DataFrame(rows), details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "ESM-C weights must already exist locally; this script never downloads or redistributes "
            "third-party model weights.\n"
            "Example:\n"
            "  PYTHONPATH=.:src python scripts/reproducibility/readout_benchmarks/"
            "beta_barrel_detection/build_esmc_embeddings_for_cohort.py "
            "--cohort-csv runs/detection/benchmark_cohort.csv "
            "--weights /path/to/esmc_600m_2024_12_v0.pth "
            "--out-npz runs/detection/esmc_mean_embeddings_aligned.npz"
        ),
    )
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Local ESM-C weights file; no download is attempted.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--min-residues", type=int, default=20)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--reuse-sequences",
        action="store_true",
        help="Reuse esmc_sequence_metadata.csv if present, then rebuild the embedding cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    out_npz = args.out_npz.expanduser().resolve()
    weights_path = resolve_esmc_weights(args.weights, required=True)
    if weights_path is None:  # pragma: no cover - required=True raises before this branch
        raise RuntimeError("ESM-C weights path could not be resolved")
    out_dir = out_npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence_meta_path = out_dir / "esmc_sequence_metadata.csv"
    embedding_meta_path = out_dir / "esmc_embedding_metadata.csv"
    run_meta_path = out_dir / "esmc_embedding_run_metadata.json"

    if args.reuse and out_npz.exists() and sequence_meta_path.exists() and embedding_meta_path.exists():
        print(f"Reusing {out_npz}")
        return 0

    cohort = pd.read_csv(args.cohort_csv.expanduser().resolve())
    if args.reuse_sequences and sequence_meta_path.exists():
        sequence_meta = pd.read_csv(sequence_meta_path)
    else:
        sequence_meta = extract_sequences(cohort, min_residues=int(args.min_residues))
        sequence_meta.to_csv(sequence_meta_path, index=False)
    failures = sequence_meta.loc[sequence_meta["status"].astype(str) != "ok"]
    if not failures.empty:
        failures.to_csv(out_dir / "esmc_sequence_failures.csv", index=False)
    else:
        failure_path = out_dir / "esmc_sequence_failures.csv"
        if failure_path.exists():
            failure_path.unlink()

    embeddings, embedding_meta, details = build_embeddings(
        sequence_meta,
        weights_path=weights_path,
        device=str(args.device),
        precision=str(args.precision),
        max_length=int(args.max_length),
    )
    record_ids = cohort["record_id"].astype(str).tolist()
    embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[1] else 1152
    aligned_embeddings = np.zeros((len(record_ids), embedding_dim), dtype=np.float32)
    esmc_available = np.zeros(len(record_ids), dtype=np.float32)
    row_index = {record_id: idx for idx, record_id in enumerate(record_ids)}
    for row in embedding_meta.itertuples(index=False):
        record_id = str(row.record_id)
        target = row_index.get(record_id)
        if target is None:
            continue
        aligned_embeddings[target, :] = embeddings[int(row.embedding_index), :]
        esmc_available[target] = 1.0
    if int(esmc_available.sum()) != len(embedding_meta):
        raise RuntimeError(
            f"Aligned embedding coverage mismatch: available={int(esmc_available.sum())} embedded={len(embedding_meta)}"
        )

    np.savez_compressed(
        out_npz,
        record_id=np.asarray(record_ids, dtype=str),
        embeddings=aligned_embeddings,
        esmc_available=esmc_available,
    )

    embedded_by_id = embedding_meta.set_index("record_id", drop=False).to_dict(orient="index") if not embedding_meta.empty else {}
    aligned_rows: list[dict[str, Any]] = []
    for idx, row in sequence_meta.reset_index(drop=True).iterrows():
        record_id = str(row["record_id"])
        embedded = embedded_by_id.get(record_id, {})
        aligned_rows.append(
            {
                "record_id": record_id,
                "pdb_id": row.get("pdb_id", ""),
                "filename": row.get("filename", ""),
                "status": row.get("status", ""),
                "chain_id": embedded.get("chain_id", row.get("chain_id", "")),
                "sequence_length": int(row.get("sequence_length", 0) or 0),
                "embedded_length": int(embedded.get("embedded_length", 0) or 0),
                "truncated": int(embedded.get("truncated", 0) or 0),
                "embedding_index": int(embedded.get("embedding_index", -1) or -1),
                "aligned_index": int(idx),
                "esmc_available": float(esmc_available[idx]),
                "reason": row.get("reason", ""),
            }
        )
    aligned_embedding_meta = pd.DataFrame(aligned_rows)
    aligned_embedding_meta.to_csv(embedding_meta_path, index=False)

    run_meta = {
        "schema": "betlas.beta-barrel-detection.esmc-mean-embeddings.v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohort_csv": str(args.cohort_csv.expanduser().resolve()),
        "out_npz": str(out_npz),
        "weights": str(weights_path),
        "n_records": int(len(cohort)),
        "embedding_dim": int(aligned_embeddings.shape[1]) if aligned_embeddings.ndim == 2 else 0,
        "embedded_records": int(esmc_available.sum()),
        "missing_sequence_records": int(len(failures)),
        "sequence_status_counts": sequence_meta["status"].value_counts().to_dict(),
        "truncated": int(aligned_embedding_meta["truncated"].sum()),
        "runtime_seconds": round(time.perf_counter() - start, 3),
        **details,
        "python": sys.version,
    }
    write_json(run_meta_path, run_meta)
    print(f"Wrote {out_npz}")
    print(f"Rows: {len(record_ids)}; embedded: {int(esmc_available.sum())}; dim: {aligned_embeddings.shape[1] if aligned_embeddings.ndim == 2 else 0}")
    print(f"Runtime seconds: {run_meta['runtime_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
