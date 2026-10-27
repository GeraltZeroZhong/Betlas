#!/usr/bin/env python
"""Extract ESM-C 600M embeddings from explicit sequences or a structure table."""

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

EMBEDDING_DIM = 1152
SEQUENCE_COLUMNS = (
    "record_id", "pdb_id", "filename", "status", "sequence",
    "sequence_length", "chain_id", "reason",
)
EMBEDDING_COLUMNS = (
    "record_id", "pdb_id", "filename", "chain_id", "sequence_length",
    "embedded_length", "truncated", "embedding_index",
)


def load_structure_table(path: Path) -> pd.DataFrame:
    """Load records with unique IDs and explicit structure paths.

    Relative structure paths are resolved against the input table's directory.
    Optional metadata columns use the input filename when omitted.
    """
    records = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("record_id", "structure_path"):
        if column not in records or records[column].str.strip().eq("").any():
            raise ValueError(f"Input CSV requires nonempty {column} values")
    if records["record_id"].duplicated().any():
        raise ValueError("Input CSV record_id values must be unique")
    records["structure_path"] = records["structure_path"].map(
        lambda value: str((path.parent / Path(value).expanduser()).resolve())
    )
    if "pdb_id" not in records:
        records["pdb_id"] = records["structure_path"].map(lambda value: Path(value).stem)
    if "filename" not in records:
        records["filename"] = records["structure_path"].map(lambda value: Path(value).name)
    return records


def align_embeddings(record_ids: list[str], embeddings: np.ndarray, metadata: pd.DataFrame):
    """Align successful embeddings to the original row order with a presence mask."""
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record_ids must be unique")
    aligned = np.zeros((len(record_ids), embeddings.shape[1]), dtype=np.float32)
    available = np.zeros(len(record_ids), dtype=np.float32)
    positions = {record_id: index for index, record_id in enumerate(record_ids)}
    for row in metadata.itertuples(index=False):
        if str(row.record_id) not in positions:
            raise ValueError(f"Embedding record is absent from input: {row.record_id}")
        target = positions[str(row.record_id)]
        if available[target]:
            raise ValueError(f"Duplicate embedding record: {row.record_id}")
        aligned[target] = embeddings[int(row.embedding_index)]
        available[target] = 1.0
    return aligned, available


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
    for column in ["chain_id", "selected_chain_id", "label_chain_id", "fallback_chain_id"]:
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


def extract_sequences(records: pd.DataFrame, *, min_residues: int) -> pd.DataFrame:
    rows = [extract_sequence(row, min_residues=min_residues) for _, row in tqdm(records.iterrows(), total=len(records), desc="Extracting sequences")]
    return pd.DataFrame(rows, columns=SEQUENCE_COLUMNS)


def load_sequence_table(path: Path, *, sequence_column: str, min_residues: int):
    """Use exact caller-defined domain sequences without any chain fallback.

    Empty/invalid/short sequences stay in the output ledger as failures. Source
    geometry status does not gate independent sequence representation baselines.
    """
    records = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"record_id", sequence_column}.issubset(records):
        raise ValueError("Sequence CSV requires record_id and the selected sequence column")
    if records["record_id"].str.strip().eq("").any() or records["record_id"].duplicated().any():
        raise ValueError("Sequence CSV record_id must be nonempty and unique")
    rows = []
    for record in records.to_dict("records"):
        sequence = record[sequence_column].strip().upper()
        reason = ""
        if not sequence:
            reason = "Empty explicit domain sequence"
        elif set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZOU"):
            reason = "Sequence contains unsupported residue symbols"
        elif len(sequence) < min_residues:
            reason = f"Explicit domain sequence has fewer than {min_residues} residues"
        rows.append({"record_id": record["record_id"], "pdb_id": record.get("pdb_id", ""),
                     "filename": "", "status": "invalid_sequence" if reason else "ok",
                     "sequence": sequence, "sequence_length": len(sequence),
                     "chain_id": record.get("chain_id", ""), "reason": reason})
    return records, pd.DataFrame(rows, columns=SEQUENCE_COLUMNS)


def apply_length_policy(sequences: pd.DataFrame, *, max_length: int, policy: str) -> pd.DataFrame:
    """Keep overlength records visible when full-domain embedding is required."""
    if max_length < 1 or policy not in {"reject", "truncate"}:
        raise ValueError("Expected a positive length and reject or truncate policy")
    result = sequences.copy()
    if policy == "reject":
        mask = result.status.eq("ok") & result.sequence_length.gt(max_length)
        result.loc[mask, "status"] = "overlength"
        result.loc[mask, "reason"] = "Full domain exceeds the frozen maximum length; no truncation permitted"
    return result


def load_esmc_model(*, weights_path: Path, device: str, precision: str):
    try:
        import torch
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional esm==3.2.1 environment described in the embeddings README") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmSequenceTokenizer()
    model = ESMC(
        tokenizer=tokenizer, d_model=1152, n_layers=36, n_heads=18,
        use_flash_attn=str(device).startswith("cuda") and precision == "fp16",
    ).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {key.replace("module.", "").replace("model.", ""): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
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
    if max_length < 1:
        raise ValueError("max_length must be positive")
    ok = sequence_meta.loc[sequence_meta["status"].astype(str) == "ok"].reset_index(drop=True)
    if ok.empty:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), pd.DataFrame(columns=EMBEDDING_COLUMNS), {
            "embedding_runtime_seconds": 0.0, "device": device, "precision": precision, "max_length": max_length,
        }
    model, _tokenizer, resolved_device = load_esmc_model(weights_path=weights_path, device=device, precision=precision)
    import torch
    from esm.sdk.api import ESMProtein
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
                raise FloatingPointError(f"Non-finite ESM-C embedding for record_id={row.record_id}; no successful vector was written")
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
    matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    details = {
        "embedding_runtime_seconds": round(time.perf_counter() - start, 3),
        "device": str(resolved_device),
        "precision": "fp16" if str(resolved_device).startswith("cuda") and precision == "fp16" else "fp32",
        "requested_precision": precision,
        "max_length": int(max_length),
    }
    return matrix, pd.DataFrame(rows, columns=EMBEDDING_COLUMNS), details


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-csv", type=Path, help="Structure table with record_id and structure_path; optional chain_id.")
    inputs.add_argument("--sequence-csv", type=Path, help="Explicit domain sequences with record_id; no structure or chain fallback.")
    parser.add_argument("--sequence-column", default="domain_sequence")
    parser.add_argument("--long-sequence-policy", choices=["reject", "truncate"], help="Default: reject for explicit domain sequences; truncate for legacy structure input.")
    parser.add_argument("--out-npz", type=Path, required=True, help="Output NPZ; metadata files are written beside it.")
    parser.add_argument("--weights", type=Path, required=True, help="Existing local ESM-C 600M checkpoint.")
    parser.add_argument("--device", default="auto", help="Torch device; auto selects CUDA when available, otherwise CPU.")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--max-length", type=int, default=1022, help="Maximum sequence length; longer sequences are explicitly marked truncated.")
    parser.add_argument("--min-residues", type=int, default=20)
    args = parser.parse_args(argv)
    if args.min_residues < 1 or args.max_length < 1:
        parser.error("--min-residues and --max-length must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    out_npz = args.out_npz.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Local ESM-C weights file does not exist: {weights_path}")
    out_dir = out_npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence_meta_path = out_dir / "esmc_sequence_metadata.csv"
    embedding_meta_path = out_dir / "esmc_embedding_metadata.csv"
    run_meta_path = out_dir / "esmc_embedding_run_metadata.json"

    input_path = (args.sequence_csv or args.input_csv).expanduser().resolve()
    if args.sequence_csv is not None:
        records, sequence_meta = load_sequence_table(input_path, sequence_column=args.sequence_column, min_residues=int(args.min_residues))
    else:
        records = load_structure_table(input_path)
        sequence_meta = extract_sequences(records, min_residues=int(args.min_residues))
    length_policy = args.long_sequence_policy or ("reject" if args.sequence_csv is not None else "truncate")
    sequence_meta = apply_length_policy(sequence_meta, max_length=args.max_length, policy=length_policy)
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
    record_ids = records["record_id"].astype(str).tolist()
    aligned_embeddings, esmc_available = align_embeddings(record_ids, embeddings, embedding_meta)

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
                "embedding_index": int(embedded.get("embedding_index", -1)),
                "aligned_index": int(idx),
                "esmc_available": float(esmc_available[idx]),
                "reason": row.get("reason", ""),
            }
        )
    aligned_embedding_meta = pd.DataFrame(aligned_rows, columns=["record_id", "pdb_id", "filename", "status", "chain_id", "sequence_length", "embedded_length", "truncated", "embedding_index", "aligned_index", "esmc_available", "reason"])
    aligned_embedding_meta.to_csv(embedding_meta_path, index=False)

    run_meta = {
        "schema": "betlas.esmc-mean-embeddings.v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_csv": str(input_path),
        "sequence_source": "explicit_domain_sequence" if args.sequence_csv is not None else "structure_chain",
        "sequence_column": args.sequence_column if args.sequence_csv is not None else None,
        "long_sequence_policy": length_policy,
        "out_npz": str(out_npz),
        "weights": str(weights_path),
        "n_records": int(len(records)),
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
