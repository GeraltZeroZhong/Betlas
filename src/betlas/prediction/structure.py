"""Full-model DSSP annotation and chain measurements for task prediction."""

from __future__ import annotations

import gzip
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from Bio.PDB import MMCIFIO, PDBParser

from betlas.features.extract import extract_classifier_features
from betlas.io.domain_geometry import (
    _ca_residues,
    _select_model,
    annotate_structure,
    build_domain_geometry,
    export_domain_from_source,
)
from betlas.io.mmcif import parse_residue_ranges, read_mmcif_dict
from betlas.models import DomainCandidate


def _model_input(source: Path, destination: Path, model_id: int) -> Path:
    """Write one complete coordinate model in mmCIF, preserving author IDs."""
    suffix = source.name.lower().removesuffix(".gz")
    if not suffix.endswith((".cif", ".mmcif", ".pdb")):
        raise ValueError("Structure input must be PDB or mmCIF, optionally gzip-compressed")
    if suffix.endswith(".pdb"):
        stream = gzip.open(source, "rt") if source.name.lower().endswith(".gz") else source.open()
        with stream:
            structure = PDBParser(QUIET=True, PERMISSIVE=False, get_header=False).get_structure(
                "input", stream
            )
        models = list(structure.get_models())
        if model_id >= len(models):
            raise ValueError(
                f"model_id {model_id} is unavailable; structure has {len(models)} models"
            )
        writer = MMCIFIO()
        writer.set_structure(models[model_id])
        writer.save(str(destination))
        data, _ = _select_model(read_mmcif_dict(destination), 0)
    else:
        data, _ = _select_model(read_mmcif_dict(source), model_id)
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(destination))
    return destination


def _check_annotation_ids(data: dict[str, Any]) -> None:
    """Reject identifier collisions produced by affected DSSP mmCIF writers."""
    keys = (
        ("_struct_sheet", ("id",)),
        ("_struct_sheet_range", ("sheet_id", "id")),
        ("_dssp_struct_ladder", ("id",)),
    )
    for category, fields in keys:
        columns = []
        for field in fields:
            values = data.get(f"{category}.{field}", [])
            columns.append(values if isinstance(values, list) else [values])
        if len({len(column) for column in columns}) != 1:
            raise ValueError(f"DSSP annotation has inconsistent {category} identifiers")
        identifiers = list(zip(*columns, strict=True))
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                f"DSSP annotation contains duplicate {category} identifiers; "
                "rerun with DSSP 4.6.1 or a newer corrected writer"
            )


@contextmanager
def measured_structure(
    structure: str | Path,
    *,
    task: str,
    chain_id: str,
    model_id: int = 0,
    residue_ranges: str = "",
    dssp_binary: str | Path = "mkdssp",
    timeout: float = 300,
) -> Iterator[tuple[dict[str, Any], Path, dict[str, Any]]]:
    """Yield current descriptors, selected full model, and annotation metadata."""
    source = Path(structure).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Structure file does not exist: {source}")
    if model_id < 0:
        raise ValueError("model_id must be nonnegative")
    if not str(chain_id):
        raise ValueError("An author chain identifier is required")
    with tempfile.TemporaryDirectory(prefix="betlas-predict-") as folder:
        directory = Path(folder)
        selected = _model_input(source, directory / "model.cif", model_id)
        data = read_mmcif_dict(selected)
        residues, _ = _ca_residues(data, "input")
        chain = [r for r in residues if r.chain_id == chain_id]
        if not chain:
            choices = ", ".join(sorted({r.chain_id for r in residues}))
            raise ValueError(
                f"Chain {chain_id!r} contains no protein C-alpha coordinates; available chains: {choices}"
            )
        if residue_ranges:
            ranges = parse_residue_ranges(residue_ranges, fallback_chain_id=chain_id)
            if any(c != chain_id for c, _, _ in ranges):
                raise ValueError("All residue ranges must select the requested author chain")
            segments = [{"chain_id": c, "start": start, "end": end} for c, start, end in ranges]
        else:
            first, last = chain[0], chain[-1]
            segments = [
                {
                    "chain_id": chain_id,
                    "start": first.auth_seq_id,
                    "start_ins": first.insertion_code,
                    "end": last.auth_seq_id,
                    "end_ins": last.insertion_code,
                }
            ]
        domain = DomainCandidate(
            "input", "input", chain_id, "input", residue_ranges, "", "", "user_input"
        )
        annotation_input = selected
        if task in ("solenoid", "barrel"):
            annotation_input = directory / "selected-chain.cif"
            export_domain_from_source(domain, selected, annotation_input, segments=segments)
        annotated = directory / "annotated.cif"
        annotation = annotate_structure(
            annotation_input, annotated, dssp_binary=dssp_binary, timeout=timeout
        )
        annotation_data = read_mmcif_dict(annotated)
        _check_annotation_ids(annotation_data)
        geometry = build_domain_geometry(domain, annotation_data, segments=segments)
        if not geometry.residues:
            raise ValueError("No protein residues remain in the selected structure range")
        features = extract_classifier_features(geometry)
        metadata = {
            "chain_id": chain_id,
            "model_id": model_id,
            "residue_count": len(geometry.residues),
            "beta_strand_count": len(geometry.beta_segments),
            "secondary_structure_source": "DSSP",
            "dssp_version": annotation["dssp_version"],
            "annotation_scope": "full_model"
            if task == "barrel-strands"
            else "selected_domain_or_chain",
            "warnings": list(geometry.warnings),
        }
        yield features, selected, metadata


def layer_radial_features(
    source: Path, chain_id: str, columns: list[str], dssp_binary: str | Path
) -> dict[str, float]:
    """Measure native stave-specific layer/radial features used in training."""
    from betlas.readouts.beta_barrel_staves.pipeline import count_strands

    result = count_strands(
        str(source),
        overrides={
            "input.chain_id": chain_id,
            "runtime.dssp_bin_path": str(dssp_binary),
            "runtime.fail_on_dssp_error": True,
            "runtime.prepare_cache_enabled": False,
        },
        workers=1,
        prepare_workers=1,
        write_csv=False,
        print_summary=False,
        show_progress=False,
        allow_ungated=True,
    )
    rows = result.raw_rows()
    if len(rows) != 1 or rows[0].get("chain") != chain_id:
        raise ValueError("Layer/radial extraction did not return the requested chain")
    row = rows[0]
    if row.get("result") in {"ERROR", "SKIP", "FILTERED_OUT"}:
        raise ValueError(f"Layer/radial measurements are unavailable: {row.get('result')}")
    values = {}
    for column in columns:
        if not column.startswith("layer_radial16__"):
            raise ValueError(f"Unrecognised task-specific feature column: {column}")
        key = column.removeprefix("layer_radial16__")
        if key not in row:
            raise ValueError(f"Layer/radial extraction is missing {key}")
        values[column] = float(row[key])
    return values
