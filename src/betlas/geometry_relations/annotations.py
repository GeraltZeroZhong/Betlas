"""Exact-domain deposited annotation adapter and paired annotation comparison.

Both adapters use the existing full-model polymer/CA selection contract.
Deposited sheet IDs are identifiers, not spatial order; complete declared
pairing edges are retained instead of the legacy per-strand sense projection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from betlas.io.domain_geometry import (
    _ca_residues,
    _cath_zero_chain_context,
    _ins,
    _resolve_spans,
    _select_model,
    _values,
    build_domain_geometry,
)
from betlas.models import DomainCandidate, SecondaryStructureElement, SheetPatch, StructureGeometry

from .io import read_mmcif

SCHEMA = "betlas.annotation-bridge.v1"


def build_dssp_domain_geometry(
    domain: DomainCandidate, annotated_mmcif: str | Path | Mapping[str, Any],
    *, segments: list[dict[str, Any]],
) -> StructureGeometry:
    """Use the original DSSP domain constructor, including its clipping rules."""
    data = annotated_mmcif if isinstance(annotated_mmcif, Mapping) else read_mmcif(annotated_mmcif)
    return build_domain_geometry(domain, data, segments=segments)


def build_deposited_domain_geometry(
    domain: DomainCandidate, source_mmcif: str | Path | Mapping[str, Any],
    *, segments: list[dict[str, Any]], model_id: int = 0,
) -> StructureGeometry:
    """Read deposited ranges and complete order edges on the exact domain.

    A partial range has no known within-range pairing location in deposited
    order records, so its affected pairing is unavailable rather than guessed.
    All selected residues are retained even where no beta annotation exists.
    """
    if not segments:
        raise ValueError("At least one exact domain segment is required")
    data = dict(source_mmcif) if isinstance(source_mmcif, Mapping) else read_mmcif(source_mmcif)
    data, _selected_model = _select_model(data, model_id)
    all_residues, label_chains = _ca_residues(data, domain.pdb_id)
    spans, requested, missing_boundaries = _resolve_spans(
        data, all_residues, label_chains, segments,
        cath_zero_chain=_cath_zero_chain_context(domain))
    selected = {}
    for segment_index, (chain, start, end) in enumerate(spans):
        for old, residue in enumerate(all_residues):
            if label_chains[old] == chain and start <= residue.label_seq_id <= end:
                selected[old] = segment_index
    ordered_indices = sorted(selected)
    residues = tuple(all_residues[old] for old in ordered_indices)
    fragment_ids = {}
    fragment_number, previous = 0, None
    for new, old in enumerate(ordered_indices):
        if (previous is None or selected[old] != selected[previous]
                or label_chains[old] != label_chains[previous]
                or all_residues[old].label_seq_id != all_residues[previous].label_seq_id + 1):
            fragment_number += 1
        fragment_ids[new] = f"F{fragment_number}"
        previous = old

    def annotation_span(category: str, row: int) -> tuple[str, int, int]:
        # Prefer declared polymer coordinates. Fall back to the same verified
        # author/polymer mapping used for CATH boundaries, including insertions.
        try:
            chain = _values(data, f"{category}.beg_label_asym_id")[row]
            end_chain = _values(data, f"{category}.end_label_asym_id")[row]
            if chain in {"", ".", "?"} or end_chain in {"", ".", "?"}:
                raise ValueError("Polymer chain identifiers are unavailable")
            start = int(_values(data, f"{category}.beg_label_seq_id")[row])
            end = int(_values(data, f"{category}.end_label_seq_id")[row])
        except (IndexError, ValueError):
            count = len(_values(data, f"{category}.id"))
            first_chain = _values(data, f"{category}.beg_auth_asym_id", count)[row]
            last_chain = _values(data, f"{category}.end_auth_asym_id", count)[row]
            if first_chain != last_chain:
                raise ValueError("Deposited range spans multiple author chains") from None
            segment = {"chain_id": _ins(first_chain),
                       "start": int(_values(data, f"{category}.beg_auth_seq_id", count)[row]),
                       "end": int(_values(data, f"{category}.end_auth_seq_id", count)[row]),
                       "start_ins": _ins(_values(data, f"{category}.pdbx_beg_PDB_ins_code", count)[row]),
                       "end_ins": _ins(_values(data, f"{category}.pdbx_end_PDB_ins_code", count)[row])}
            resolved, _, _ = _resolve_spans(data, all_residues, label_chains, [segment])
            return resolved[0]
        if chain != end_chain or start > end:
            raise ValueError("Deposited range has incompatible polymer endpoints")
        return chain, start, end

    beta, helices = [], []
    raw_to_elements = defaultdict(list)
    complete_ranges = set()
    for category, element_type, destination in (("_struct_sheet_range", "strand", beta),
                                                ("_struct_conf", "helix", helices)):
        ids = _values(data, f"{category}.id")
        sheets = _values(data, f"{category}.sheet_id", len(ids))
        types = _values(data, f"{category}.conf_type_id", len(ids))
        for row, range_id in enumerate(ids):
            if element_type == "helix" and not types[row].startswith("HELX"):
                continue
            selected_label_chains = {span[0] for span in spans}
            selected_auth_chains = {residue.chain_id for residue in residues}
            label_chain = _values(data, f"{category}.beg_label_asym_id", len(ids))[row]
            author_chain = _ins(_values(data, f"{category}.beg_auth_asym_id", len(ids))[row])
            if label_chain not in {"", ".", "?"}:
                if label_chain not in selected_label_chains:
                    continue
            elif author_chain not in selected_auth_chains:
                continue
            chain, start, end = annotation_span(category, row)
            fragments = defaultdict(list)
            for new, old in enumerate(ordered_indices):
                if label_chains[old] == chain and start <= all_residues[old].label_seq_id <= end:
                    fragments[fragment_ids[new]].append(new)
            for fragment, indices in fragments.items():
                if len(indices) < (2 if element_type == "strand" else 3):
                    continue
                first, last = residues[indices[0]], residues[indices[-1]]
                element_id = ("B" if element_type == "strand" else "H") + f"{len(destination) + 1:03d}"
                destination.append(SecondaryStructureElement(
                    element_id, element_type, first.chain_id, first.auth_seq_id,
                    last.auth_seq_id, tuple(indices), sheets[row], range_id,
                    fragment_id=fragment))
                if element_type == "strand":
                    raw_to_elements[(sheets[row], range_id)].append(element_id)
                    if first.label_seq_id == start and last.label_seq_id == end:
                        complete_ranges.add((sheets[row], range_id, element_id))
    order_sheets = _values(data, "_struct_sheet_order.sheet_id")
    order_columns = {name: _values(data, f"_struct_sheet_order.{name}", len(order_sheets))
                     for name in ("range_id_1", "range_id_2", "sense")}
    if any(len(column) != len(order_sheets) for column in order_columns.values()):
        raise ValueError("Inconsistent deposited sheet order columns")
    edges, partial = set(), False
    for i, sheet in enumerate(order_sheets):
        first = (sheet, order_columns["range_id_1"][i])
        second = (sheet, order_columns["range_id_2"][i])
        lefts, rights = raw_to_elements[first], raw_to_elements[second]
        if not lefts or not rights:
            continue
        sense = order_columns["sense"][i]
        if sense not in {"parallel", "anti-parallel"}:
            sense = "unknown"
        for left in lefts:
            for right in rights:
                if (*first, left) in complete_ranges and (*second, right) in complete_ranges:
                    if left != right:
                        edges.add((*sorted((left, right)), sense))
                else:
                    partial = True
    warnings = []
    if missing_boundaries:
        warnings.append("boundary_ca_missing_resolved_by_polymer_scheme")
    if len(residues) < len(requested):
        warnings.append("missing_ca_within_domain")
    if not _values(data, "_struct_sheet_range.id"):
        warnings.append("missing_deposited_sheet_annotation")
    if not order_sheets:
        warnings.append("missing_deposited_sheet_order")
    if partial:
        warnings.append("clipped_deposited_pairing_unavailable")
    if not beta:
        warnings.append("no_deposited_beta_segments")
    by_sheet = defaultdict(list)
    for element in beta:
        by_sheet[element.sheet_id].append(element.element_id)
    return StructureGeometry(domain, residues, tuple(beta), tuple(helices),
                             tuple(SheetPatch(sheet, tuple(ids)) for sheet, ids in sorted(by_sheet.items())),
                             tuple(warnings), tuple(sorted(edges)), "deposited", len(segments),
                             len(requested), missing_boundaries)


def compare_annotations(left: StructureGeometry, right: StructureGeometry) -> dict[str, Any]:
    """Compare unchanged selected coordinates and annotation-specific nodes.

    Node correspondence is all positive residue overlaps, including one-to-many
    splits. It does not force equal beta subsets or use node/sheet identifier
    equality. A failed coordinate identity check invalidates a paired bridge.
    """
    def residue_key(residue):
        return (residue.chain_id, residue.auth_seq_id, residue.insertion_code, residue.label_seq_id)

    lres = {residue_key(r): r for r in left.residues}
    rres = {residue_key(r): r for r in right.residues}
    shared = set(lres) & set(rres)
    max_delta = max((float(np.linalg.norm(np.array(lres[key].coord_ca) - rres[key].coord_ca))
                     for key in shared), default=None)
    same_coordinates = set(lres) == set(rres) and max_delta is not None and max_delta <= 1e-8
    lnodes = {n.element_id: {residue_key(left.residues[i]) for i in n.residue_indices}
              for n in left.beta_segments}
    rnodes = {n.element_id: {residue_key(right.residues[i]) for i in n.residue_indices}
              for n in right.beta_segments}
    lbeta = set().union(*lnodes.values()) if lnodes else set()
    rbeta = set().union(*rnodes.values()) if rnodes else set()
    overlaps = []
    for lid, lkeys in lnodes.items():
        for rid, rkeys in rnodes.items():
            intersection = len(lkeys & rkeys)
            if intersection:
                overlaps.append({"left_node": lid, "right_node": rid,
                                 "shared_residue_count": intersection,
                                 "left_fraction": intersection / len(lkeys),
                                 "right_fraction": intersection / len(rkeys),
                                 "jaccard": intersection / len(lkeys | rkeys)})
    return {"schema": SCHEMA, "record_id": left.domain.record_id,
            "left_source": left.secondary_structure_source,
            "right_source": right.secondary_structure_source,
            "same_selected_residue_identity": set(lres) == set(rres),
            "same_selected_coordinates": same_coordinates,
            "max_ca_displacement_angstrom": max_delta,
            "left_residue_count": len(lres), "right_residue_count": len(rres),
            "left_beta_residue_count": len(lbeta), "right_beta_residue_count": len(rbeta),
            "beta_residue_intersection": len(lbeta & rbeta),
            "beta_residue_jaccard": len(lbeta & rbeta) / len(lbeta | rbeta) if lbeta | rbeta else None,
            "left_node_count": len(lnodes), "right_node_count": len(rnodes),
            "node_residue_overlaps": overlaps,
            "left_warnings": list(left.warnings), "right_warnings": list(right.warnings)}
