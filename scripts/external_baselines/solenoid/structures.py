"""Inspect exact coordinate inputs without renumbering residues or closing gaps."""
from __future__ import annotations

import math
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa


def is_protein_residue(residue) -> bool:
    """Include genetically encoded SEC/PYL omitted by Biopython's lookup."""
    return is_aa(residue, standard=False) or residue.resname in {"SEC", "PYL"}


def load_structure(path: str | Path):
    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif path.suffix.lower() in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=True, PERMISSIVE=False)
    else:
        raise ValueError("Supply an uncompressed PDB, ENT, CIF, or mmCIF coordinate file")
    return parser.get_structure("input", str(path))


def matrix_cost(span: int) -> dict[str, int]:
    """Source-derived array sizes, not a prediction of TensorFlow peak RSS."""
    if span < 1:
        raise ValueError("Residue-number span must be positive")
    # Native float64 distance matrix; padded diagonal tiles of 128 with stride 64.
    padded = math.ceil(span / 128) * 128 + 64
    tiles = (padded - 64) // 64
    reconstructed = tiles * 64
    return {
        "matrix_span": span,
        "distance_matrix_bytes": 8 * span**2,
        "padded_distance_matrix_bytes": 8 * padded**2,
        "diagonal_tiles": tiles,
        "tile_input_bytes": 8 * tiles * 128**2,
        "native_prediction_matrix_bytes": 8 * reconstructed**2 * 6,
        "principal_array_bytes": (
            8 * span**2 + 8 * padded**2 + 8 * tiles * 128**2
            + 8 * reconstructed**2 * 6
        ),
    }


def inspect_structure(
    path: str | Path, *, chain_id: str | None = None, max_span: int = 2048
) -> dict:
    """Check representability by upstream's integer-indexed distance matrix.

    Missing integer positions remain zero-padded as in upstream. Insertion codes,
    ambiguous residue identities and filtered CA residues are unsupported rather
    than silently renumbered. No coordinates or source files are modified.
    """
    if max_span < 1:
        raise ValueError("max_span must be positive")
    path = Path(path).resolve()
    structure = load_structure(path)
    model = next(structure.get_models())
    chains = list(model.get_chains())
    result = {
        "structure_path": str(path), "model_id": model.id,
        "model_policy": "first_model", "n_models": len(list(structure.get_models())),
        "status": "unavailable", "reason": "", "residues": [],
    }
    if chain_id is None:
        if len(chains) != 1:
            result["reason"] = "explicit_chain_required_for_multiple_chains"
            return result
        chain_id = chains[0].id
    result["chain_id"] = chain_id
    if chain_id not in model:
        result["reason"] = "chain_not_found"
        return result
    chain = model[chain_id]
    residues = list(chain)
    observed = [r for r in residues if is_protein_residue(r) and "CA" in r]
    result["n_observed_ca_residues"] = len(observed)
    if not observed:
        result["reason"] = "no_protein_ca_residues"
        return result
    if any(r.is_disordered() == 2 for r in residues):
        result["reason"] = "disordered_residue_identity"
        return result
    if any(r.id[2].strip() for r in observed):
        result["reason"] = "insertion_codes_not_representable_by_native_matrix"
        return result
    # Exactly match Protein.set_atom_lists rather than substituting a CA-only model.
    eligible = [r for r in residues if len(r) >= 4 and "CA" in r]
    if [r.id for r in eligible] != [r.id for r in observed]:
        result["reason"] = "native_atom_filter_does_not_match_observed_protein_residues"
        return result
    numbers = [int(r.id[1]) for r in observed]
    if len(set(numbers)) != len(numbers) or numbers != sorted(numbers):
        result["reason"] = "nonunique_or_nonmonotonic_author_numbers"
        return result
    if residues[0].id != observed[0].id:
        result["reason"] = "native_first_residue_is_not_first_eligible_residue"
        return result
    coords = [tuple(float(x) for x in r["CA"].coord) for r in observed]
    if any(not math.isfinite(x) for xyz in coords for x in xyz):
        result["reason"] = "nonfinite_coordinates"
        return result
    if len(set(coords)) < 2:
        result["reason"] = "zero_distance_normalization"
        return result
    result.update(matrix_cost(numbers[-1] - numbers[0] + 1))
    result["n_missing_integer_positions"] = result["matrix_span"] - len(observed)
    result["residues"] = [
        {"chain_id": chain_id, "residue_number": int(r.id[1]),
         "insertion_code": r.id[2].strip(), "residue_name": r.resname,
         "native_span_index": int(r.id[1]) - numbers[0]}
        for r in observed
    ]
    if result["matrix_span"] > max_span:
        result["reason"] = "matrix_span_exceeds_explicit_resource_limit"
        return result
    result.update(status="ready", reason="")
    return result
