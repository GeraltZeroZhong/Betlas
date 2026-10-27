"""Exact domain geometry from full-model DSSP annotated mmCIF structures.

Domain boundaries use author residue keys; ordering and continuity use polymer
label sequence identifiers. DSSP is run on the full model before domain slicing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFIO

from ..constants import STANDARD_AMINO_ACIDS
from ..models import (
    DomainCandidate,
    ResidueRecord,
    SecondaryStructureElement,
    SheetPatch,
    StructureGeometry,
)
from .mmcif import read_mmcif_dict


def _values(data: dict[str, Any], key: str, count: int = 0, default: str = "") -> list[str]:
    value = data.get(key)
    if value is None:
        return [default] * count
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def _ins(value: Any) -> str:
    text = str(value).strip()
    return "" if text in {"", ".", "?"} else text


def _select_model(data: dict[str, Any], model_id: int) -> tuple[dict[str, Any], str]:
    """Select a model by numeric serial order and normalize its atom serial to 1."""
    if model_id < 0:
        raise ValueError("model_id must be nonnegative")
    count = len(_values(data, "_atom_site.id"))
    if not count:
        raise ValueError("coordinate input has no atom_site records")
    model_nums = _values(data, "_atom_site.pdbx_PDB_model_num", count, "1")
    if len(model_nums) != count:
        raise ValueError("inconsistent atom_site model column")
    models = sorted(set(model_nums), key=int)
    if model_id >= len(models):
        raise ValueError(f"model_id {model_id} is unavailable; structure has {len(models)} models")
    selected_model = models[model_id]
    indices = [i for i, number in enumerate(model_nums) if number == selected_model]
    selected = dict(data)
    for key in data:
        if key.startswith("_atom_site."):
            values = _values(data, key)
            if len(values) != count:
                raise ValueError(f"inconsistent atom_site column: {key}")
            selected[key] = [values[i] for i in indices]
    selected["_atom_site.pdbx_PDB_model_num"] = ["1"] * len(indices)
    return selected, selected_model


def annotate_structure(
    mmcif_path: str | Path,
    output_path: str | Path,
    *,
    dssp_binary: str | Path,
    model_id: int = 0,
    timeout: float = 300,
) -> dict[str, Any]:
    """Annotate a full selected model using an explicitly selected DSSP 4 binary.

``model_id`` is a zero-based index into model numbers sorted numerically. The
selected model is normalized to model number 1 in the annotation input. Output
is replaced only after successful parsing and required-category verification.
    """
    if model_id < 0 or timeout <= 0:
        raise ValueError("model_id must be nonnegative and timeout must be positive")
    source, target = Path(mmcif_path), Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError("annotation output must differ from the coordinate original")
    binary = shutil.which(str(dssp_binary))
    if binary is None:
        raise FileNotFoundError(f"DSSP executable is not executable: {dssp_binary}")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             check=True, timeout=min(timeout, 30)).stdout.strip()
    if not version.startswith("mkdssp version 4."):
        raise ValueError(f"annotated mmCIF requires DSSP 4, found {version!r}")
    data, selected_model = _select_model(read_mmcif_dict(source), model_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dssp-", dir=target.parent) as temporary:
        full_model = Path(temporary) / "model.cif"
        annotated = Path(temporary) / "annotated.cif"
        writer = MMCIFIO()
        writer.set_dict(data)
        writer.save(str(full_model))
        command = [binary, "--output-format", "mmcif"]
        # Older DSSP 4 requires opting into residue and ladder categories.
        parts = version.removeprefix("mkdssp version ").split(".")
        if int(parts[1]) < 5:
            command.append("--write-experimental")
        command.extend([str(full_model), str(annotated)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"DSSP exited {result.returncode}: {result.stderr.strip()}")
        if not annotated.is_file():
            raise RuntimeError("DSSP did not create annotated mmCIF")
        checked = read_mmcif_dict(annotated)
        if not _values(checked, "_dssp_struct_summary.label_seq_id"):
            raise RuntimeError("DSSP annotation contains no residue summary category")
        if not _values(checked, "_atom_site.id"):
            raise RuntimeError("DSSP annotation contains no coordinates")
        os.replace(annotated, target)
    return {"status": "ok", "output_path": str(target), "model_id": model_id,
            "source_model_num": selected_model, "dssp_binary": binary,
            "dssp_version": version, "command": command, "warnings": result.stderr.strip()}


def _ca_residues(data: dict[str, Any], pdb_id: str) -> tuple[list[ResidueRecord], list[str]]:
    keys = ("group_PDB", "auth_asym_id", "auth_seq_id", "label_seq_id", "label_asym_id",
            "label_atom_id", "label_comp_id", "pdbx_PDB_ins_code", "label_alt_id",
            "occupancy", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")
    n = len(_values(data, "_atom_site.id"))
    if not n:
        raise ValueError("annotated structure has no atom_site records")
    columns = {key: _values(data, f"_atom_site.{key}", n) for key in keys}
    if any(len(values) != n for values in columns.values()):
        raise ValueError("inconsistent atom_site columns")
    if len(set(columns["pdbx_PDB_model_num"]) - {"", ".", "?"}) > 1:
        raise ValueError("annotated structure must contain exactly one model")
    chosen: dict[tuple[str, int, str], tuple[tuple[float, int], ResidueRecord, str]] = {}
    for i in range(n):
        row = {key: values[i] for key, values in columns.items()}
        if row["label_atom_id"] != "CA" or row["label_comp_id"] not in STANDARD_AMINO_ACIDS:
            continue
        if row["group_PDB"] not in {"ATOM", "HETATM"}:
            continue
        if row["label_seq_id"] in {"", ".", "?"}:
            continue
        auth, label = int(row["auth_seq_id"]), int(row["label_seq_id"])
        insertion = _ins(row["pdbx_PDB_ins_code"])
        coords = tuple(float(row[key]) for key in ("Cartn_x", "Cartn_y", "Cartn_z"))
        if not np.isfinite(coords).all():
            continue
        residue = ResidueRecord(pdb_id, _ins(row["auth_asym_id"]), auth, label, insertion,
                                row["label_comp_id"], coords)
        key = (residue.chain_id, auth, insertion)
        alt = _ins(row["label_alt_id"])
        rank = (float(row["occupancy"] or "1"), 2 if not alt else 1 if alt == "A" else 0)
        previous = chosen.get(key)
        if previous is not None and (previous[1].label_seq_id != label or previous[2] != row["label_asym_id"]):
            raise ValueError(f"ambiguous author residue key: {key}")
        if previous is None or rank > previous[0]:
            chosen[key] = (rank, residue, row["label_asym_id"])
    ordered = sorted(chosen.values(), key=lambda value: (value[2], value[1].label_seq_id))
    return [entry[1] for entry in ordered], [entry[2] for entry in ordered]


def _cath_zero_chain_context(domain: DomainCandidate) -> bool:
    """Recognize an explicit CATH zero token retained in its native domain ID."""
    native_id = domain.domain_id
    explicit_cath = bool(domain.cath_code or domain.cath_architecture_code)
    return (explicit_cath and len(native_id) == 7
            and native_id[:4].lower() == domain.pdb_id.lower()
            and native_id[4] == "0" and native_id[5:].isdigit())


def _resolve_spans(data, residues, label_chains, segments, *, cath_zero_chain: bool = False):
    """Resolve coordinate numbering, including unobserved polymer endpoints.

The wwPDB scheme's pdb_seq_num is coordinate/PDB numbering. auth_seq_num
may differ from coordinate numbering, so it is only a fallback after checking
agreement against observed coordinate anchors for that polymer chain.

CATH's zero chain token can denote a literal numeric author chain or a blank
chain. In explicit CATH context, literal ``0`` takes precedence; blank is an
alias only when no literal zero chain exists. If both chains independently
support the requested endpoints, the alias is ambiguous and is rejected.
Generic user-provided blank segments never acquire this CATH-specific alias.
    """
    observed = {(r.chain_id, r.auth_seq_id, r.insertion_code): (chain, int(r.label_seq_id))
                for r, chain in zip(residues, label_chains, strict=True)}
    by_label = {(chain, int(r.label_seq_id)): (r.chain_id, r.auth_seq_id, r.insertion_code)
                for r, chain in zip(residues, label_chains, strict=True)}
    mappings = {key: {value} for key, value in observed.items()}
    n = len(_values(data, "_pdbx_poly_seq_scheme.seq_id"))
    fields = {name: _values(data, f"_pdbx_poly_seq_scheme.{name}", n)
              for name in ("seq_id", "asym_id", "pdb_strand_id", "pdb_ins_code", "pdb_seq_num", "auth_seq_num")}
    if any(len(values) != n for values in fields.values()):
        raise ValueError("inconsistent polymer sequence scheme columns")
    scheme_rows: dict[str, list[int]] = defaultdict(list)
    for i, chain in enumerate(fields["asym_id"]):
        scheme_rows[chain].append(i)
    for chain, rows in scheme_rows.items():
        chosen_field = None
        for field in ("pdb_seq_num", "auth_seq_num"):
            agreements, conflicts, candidates = 0, 0, 0
            for i in rows:
                try:
                    seq, number = int(fields["seq_id"][i]), int(fields[field][i])
                except ValueError:
                    continue
                candidates += 1
                anchor = by_label.get((chain, seq))
                key = (_ins(fields["pdb_strand_id"][i]), number, _ins(fields["pdb_ins_code"][i]))
                if anchor is not None:
                    agreements += anchor == key
                    conflicts += anchor != key
            if candidates and not conflicts and (agreements or field == "pdb_seq_num"):
                chosen_field = field
                break
        if chosen_field is None:
            # Existing exact CA keys remain usable; unobserved endpoints do not
            # acquire an unverified alternative numbering convention.
            continue
        for i in rows:
            try:
                seq, number = int(fields["seq_id"][i]), int(fields[chosen_field][i])
            except ValueError:
                continue
            key = (_ins(fields["pdb_strand_id"][i]), number, _ins(fields["pdb_ins_code"][i]))
            mappings.setdefault(key, set()).add((chain, seq))
    spans = []
    requested: set[tuple[str, int]] = set()
    missing_boundaries = 0
    author_chains = {_ins(value) for value in _values(data, "_atom_site.auth_asym_id")}
    author_chains.update(_ins(value) for value in fields["pdb_strand_id"])
    for segment in segments:
        chain_id = _ins(segment["chain_id"])
        if cath_zero_chain and chain_id in {"", "0"}:
            if "0" in author_chains:
                if "" in author_chains:
                    candidates = {}
                    for candidate in ("0", ""):
                        endpoints = [mappings.get((candidate, int(segment[side]),
                                                   _ins(segment.get(f"{side}_ins", ""))), set())
                                     for side in ("start", "end")]
                        if all(len(matches) == 1 for matches in endpoints):
                            first, last = [next(iter(matches)) for matches in endpoints]
                            if first[0] == last[0] and first[1] <= last[1]:
                                candidates[candidate] = (first, last)
                    if len(candidates) == 2:
                        raise ValueError("ambiguous CATH zero-chain token: both literal zero and blank chains support the boundaries")
                chain_id = "0"
            elif "" in author_chains:
                chain_id = ""
        endpoints = []
        for side in ("start", "end"):
            key = (chain_id, int(segment[side]), _ins(segment.get(f"{side}_ins", "")))
            matches = mappings.get(key, set())
            if len(matches) != 1:
                reason = "ambiguous" if matches else "unmapped"
                raise ValueError(f"domain boundary is {reason} in observed coordinates/polymer scheme: {key}")
            endpoints.append(next(iter(matches)))
            missing_boundaries += key not in observed
        (chain, start), (end_chain, end) = endpoints
        if chain != end_chain:
            raise ValueError("domain segment endpoints refer to different polymer chains")
        if start > end:
            raise ValueError("domain segment endpoints are reversed in polymer sequence order")
        positions = {(chain, seq) for seq in range(start, end + 1)}
        if requested & positions:
            raise ValueError("domain segments overlap in polymer sequence")
        requested.update(positions)
        spans.append((chain, start, end))
    return spans, requested, missing_boundaries


def build_domain_geometry(
    domain: DomainCandidate,
    annotated_mmcif: str | Path | Mapping[str, Any],
    *,
    segments: list[dict[str, Any]],
) -> StructureGeometry:
    """Select exact author+insertion boundaries and preserve DSSP sheet edges.

    Unmapped or ambiguous endpoints raise ``ValueError``. Unobserved endpoints
    can resolve through the verified polymer sequence scheme, with explicit QC
    counts. Absence of beta segments is represented by a valid geometry with
    ``no_dssp_beta_segments`` warning.
    """
    if not segments:
        raise ValueError("at least one exact domain segment is required")
    # Parsed input permits several domains to share one full-model annotation.
    data = dict(annotated_mmcif) if isinstance(annotated_mmcif, Mapping) else read_mmcif_dict(Path(annotated_mmcif))
    if not _values(data, "_dssp_struct_summary.label_seq_id"):
        raise ValueError("expected DSSP annotated mmCIF with residue summaries")
    all_residues, label_chains = _ca_residues(data, domain.pdb_id)
    spans, requested, missing_boundaries = _resolve_spans(
        data, all_residues, label_chains, segments, cath_zero_chain=_cath_zero_chain_context(domain))
    selected: dict[int, int] = {}
    for segment_index, (chain, start, end) in enumerate(spans):
        for i, residue in enumerate(all_residues):
            if label_chains[i] == chain and start <= residue.label_seq_id <= end:
                selected[i] = segment_index
    ordered_indices = sorted(selected)
    residues = [all_residues[i] for i in ordered_indices]
    source_index = {old: new for new, old in enumerate(ordered_indices)}
    fragment_ids: dict[int, str] = {}
    fragment_number = 0
    previous = None
    for old in ordered_indices:
        if (previous is None or selected[old] != selected[previous]
                or label_chains[old] != label_chains[previous]
                or all_residues[old].label_seq_id != all_residues[previous].label_seq_id + 1):
            fragment_number += 1
        fragment_ids[source_index[old]] = f"F{fragment_number}"
        previous = old

    def range_indices(category: str, row: int) -> list[int]:
        # DSSP's label endpoints avoid assumptions about author numbering.
        chains = _values(data, f"{category}.beg_label_asym_id")
        ends = _values(data, f"{category}.end_label_asym_id")
        starts = _values(data, f"{category}.beg_label_seq_id")
        stops = _values(data, f"{category}.end_label_seq_id")
        if row >= min(len(chains), len(ends), len(starts), len(stops)):
            raise ValueError(f"DSSP range lacks complete polymer endpoints: {category}")
        if chains[row] != ends[row]:
            raise ValueError(f"DSSP range spans multiple polymer chains: {category}")
        start, stop = int(starts[row]), int(stops[row])
        if start > stop:
            raise ValueError("DSSP range has reversed polymer endpoints")
        return [source_index[old] for old in ordered_indices
                if label_chains[old] == chains[row] and start <= all_residues[old].label_seq_id <= stop]

    beta: list[SecondaryStructureElement] = []
    helices: list[SecondaryStructureElement] = []
    raw_to_elements: dict[tuple[str, str], list[str]] = defaultdict(list)
    for category, element_type, destination in (("_struct_sheet_range", "strand", beta),
                                                ("_struct_conf", "helix", helices)):
        ids = _values(data, f"{category}.id")
        sheets = _values(data, f"{category}.sheet_id", len(ids))
        types = _values(data, f"{category}.conf_type_id", len(ids))
        for row, range_id in enumerate(ids):
            if element_type == "helix" and types[row] not in {"HELX_RH_AL_P", "HELX_RH_3T_P", "HELX_RH_PI_P"}:
                continue
            indices = range_indices(category, row)
            fragments: dict[str, list[int]] = defaultdict(list)
            for index in indices:
                fragments[fragment_ids[index]].append(index)
            for fragment, indices_in_fragment in fragments.items():
                if len(indices_in_fragment) < (2 if element_type == "strand" else 3):
                    continue
                prefix = "B" if element_type == "strand" else "H"
                element_id = f"{prefix}{len(destination) + 1:03d}"
                first, last = residues[indices_in_fragment[0]], residues[indices_in_fragment[-1]]
                element = SecondaryStructureElement(element_id, element_type, first.chain_id,
                    first.auth_seq_id, last.auth_seq_id, tuple(indices_in_fragment), sheets[row],
                    range_id, fragment_id=fragment)
                destination.append(element)
                if element_type == "strand":
                    raw_to_elements[(sheets[row], range_id)].append(element_id)

    # Keep only contacts whose ladder endpoint intervals intersect both retained
    # strand pieces; clipping a domain must not retain an entirely external edge.
    edges: set[tuple[str, str, str]] = set()
    clipped_ladders = False
    ladder_ids = _values(data, "_dssp_struct_ladder.id")
    ladder_columns = {name: _values(data, f"_dssp_struct_ladder.{name}", len(ladder_ids))
                      for name in ("sheet_id", "range_id_1", "range_id_2", "type")}
    by_element = {element.element_id: element for element in beta}
    for i in range(len(ladder_ids)):
        sheet = ladder_columns["sheet_id"][i]
        sense = ladder_columns["type"][i]
        if sense not in {"parallel", "anti-parallel"}:
            raise ValueError(f"unsupported DSSP ladder sense: {sense}")
        sides = []
        for side in (1, 2):
            starts = _values(data, f"_dssp_struct_ladder.beg_{side}_label_seq_id")
            ends = _values(data, f"_dssp_struct_ladder.end_{side}_label_seq_id")
            chains = _values(data, f"_dssp_struct_ladder.beg_{side}_label_asym_id")
            if i >= min(len(starts), len(ends), len(chains)):
                raise ValueError("DSSP ladder lacks exact polymer endpoints")
            start, end = sorted((int(starts[i]), int(ends[i])))
            candidates = raw_to_elements[(sheet, ladder_columns[f"range_id_{side}"][i])]
            retained = []
            for element_id in candidates:
                positions = {residues[idx].label_seq_id for idx in by_element[element_id].residue_indices
                             if label_chains[ordered_indices[idx]] == chains[i]}
                if start in positions and end in positions:
                    retained.append(element_id)
                elif any(start <= position <= end for position in positions):
                    clipped_ladders = True
            sides.append(retained)
        for left in sides[0]:
            for right in sides[1]:
                if left != right:
                    edges.add((*sorted((left, right)), sense))
    warnings = []
    if not residues:
        warnings.append("no_selected_ca_residues")
    if missing_boundaries:
        warnings.append("boundary_ca_missing_resolved_by_polymer_scheme")
    if len(residues) < len(requested):
        warnings.append("missing_ca_within_domain")
    if clipped_ladders:
        warnings.append("clipped_ladder_pairing_unavailable")
    if not beta:
        warnings.append("no_dssp_beta_segments")
    if beta and not edges:
        warnings.append("no_retained_dssp_ladders")
    by_sheet: dict[str, list[str]] = defaultdict(list)
    for strand in beta:
        by_sheet[strand.sheet_id].append(strand.element_id)
    return StructureGeometry(domain, tuple(residues), tuple(beta), tuple(helices),
        tuple(SheetPatch(sheet, tuple(ids)) for sheet, ids in sorted(by_sheet.items())),
        tuple(warnings), tuple(sorted(edges)), "dssp", len(segments), len(requested), missing_boundaries)


def export_domain_mmcif(
    domain: DomainCandidate,
    annotated_mmcif: str | Path,
    output_path: str | Path,
    *,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Export exact domain polymer atoms for structure-search baselines.

The coordinate-only mmCIF preserves original atom/residue/chain identifiers and
all atom sites within the same resolved domain boundaries used by geometry.
Full-structure SSE and sequence categories are omitted to avoid stale ranges.
    """
    source, target = Path(annotated_mmcif), Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError("domain output must differ from its annotated structure")
    return _export_domain_data(domain, read_mmcif_dict(source), target, segments)


def export_domain_from_source(
    domain: DomainCandidate,
    raw_mmcif: str | Path,
    output_path: str | Path,
    *,
    segments: list[dict[str, Any]],
    model_id: int = 0,
) -> dict[str, Any]:
    """Export exact domain atoms directly from coordinates without requiring DSSP.

Model selection and author-to-polymer boundary mapping match the feature input
pipeline. Unobserved endpoints can be resolved from the polymer scheme, and
discontinuous or multichain segments retain their original identifiers.
    """
    source, target = Path(raw_mmcif), Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError("domain output must differ from the coordinate original")
    data, source_model = _select_model(read_mmcif_dict(source), model_id)
    result = _export_domain_data(domain, data, target, segments)
    return {**result, "model_id": model_id, "source_model_num": source_model}


def _export_domain_data(domain, data, target, segments):
    if not segments:
        raise ValueError("at least one exact domain segment is required")
    residues, chains = _ca_residues(data, domain.pdb_id)
    _spans, requested, missing_boundaries = _resolve_spans(
        data, residues, chains, segments, cath_zero_chain=_cath_zero_chain_context(domain))
    label_chains = _values(data, "_atom_site.label_asym_id")
    label_numbers = _values(data, "_atom_site.label_seq_id")
    indices = []
    for i, (chain, label) in enumerate(zip(label_chains, label_numbers, strict=True)):
        if label not in {"", ".", "?"} and (chain, int(label)) in requested:
            indices.append(i)
    if not indices:
        raise ValueError("domain has no observed polymer atom sites")
    output = {"data_": "domain"}
    for key in data:
        if key.startswith("_atom_site."):
            values = _values(data, key)
            if len(values) != len(label_numbers):
                raise ValueError(f"inconsistent atom_site column: {key}")
            output[key] = [values[i] for i in indices]
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".domain-", dir=target.parent) as temporary:
        path = Path(temporary) / "domain.cif"
        writer = MMCIFIO()
        writer.set_dict(output)
        writer.save(str(path))
        os.replace(path, target)
    return {"status": "ok", "output_path": str(target), "atom_count": len(indices),
            "requested_residue_count": len(requested), "missing_boundary_ca_count": missing_boundaries}
