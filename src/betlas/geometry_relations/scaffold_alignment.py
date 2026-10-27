"""Direct beta-scaffold correspondence under a single rigid transformation.

USalign receives the complete domain. Beta labels are applied *after* alignment;
they never create a concatenated beta-only chain. Every correspondence is decoded
from native ``-do`` output, including non-sequential alignments. Coverage uses all
annotated beta residues on each side, not only the aligned subset.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFIO

from .validation import (
    _atom_rows,
    _canonical_data,
    _manifest,
    _pairs,
    _read_text,
    _residue_key,
    _write_csv,
    _write_json,
    parse_alignment_output,
)

SCHEMA = "betlas.scaffold-correspondence.v1"
MODES = {"SQ": 0, "sNS": 6, "fNS": 5}
THRESHOLDS = (2.0, 3.5, 5.0)


def load_graphs(source: Any, ids=None) -> dict[str, dict]:
    """Read existing graph records, bounded to requested IDs when supplied."""
    wanted = set(ids) if ids is not None else None
    if isinstance(source, Mapping):
        records = source.items()
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".json":
            data = json.loads(_read_text(path))
            records = data.items() if isinstance(data, dict) and "record_id" not in data else (
                (r["record_id"], r) for r in (data if isinstance(data, list) else [data]))
        else:
            def stream():
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as handle:
                    buffer = ""
                    while chunk := handle.read(8192):
                        buffer += chunk
                        lines = buffer.split("\n")
                        buffer = lines.pop()
                        for line in lines:
                            if line.strip():
                                row = json.loads(line)
                                yield row["record_id"], row
                    if buffer.strip():
                        row = json.loads(buffer)
                        yield row["record_id"], row
            records = stream()
    else:
        records = ((r["record_id"], r) for r in source)
    result = {}
    for key, row in records:
        if wanted is not None and key not in wanted:
            continue
        if isinstance(row, (str, Path)):
            row = json.loads(_read_text(Path(row)))
        if key in result:
            raise ValueError(f"Duplicate connectivity record: {key}")
        result[str(key)] = row.get("graph", row)
    return result


def export_coordinates(ids, coordinate_manifest, output_dir) -> dict[str, dict]:
    """Export selected exact domain coordinates and an original-identity CA map.

    The native aligner uses one-character chains and fixed-width residue IDs in
    its correspondence output. Execution copies therefore use chain A and CA
    positions 1..N. Original author/label identity, sequence order, missing-label
    gaps, coordinates and selected conformers remain in each JSON. This is an
    execution identifier change, never a concatenation of multiple chains.
    """
    manifest = _manifest(coordinate_manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for record_id in sorted(set(map(str, ids))):
        record = {"record_id": record_id, "status": "coordinate_unavailable", "reason": ""}
        prepared[record_id] = record
        source = manifest.get(record_id)
        if source is None or source.get("export_status", "ok") != "ok":
            record["reason"] = "Missing coordinate manifest row or upstream export unavailable"
            continue
        record["source_path"] = str(source.get("structure_path", ""))
        try:
            data, inspected = _canonical_data(Path(record["source_path"]))
            record.update(inspected)
            if record["status"] != "ok":
                continue
            identities = record["ca_identities"]
            if len(identities) > 9999:
                record.update(status="unsupported_native_residue_range",
                              reason="Native correspondence field supports at most 9999 indexed residues")
                continue
            for index, residue in enumerate(identities):
                residue.update(index0=index, execution_residue_number=index + 1,
                               residue_uid=f"{residue['chain_id']}:{residue['author_residue_number']}:{residue['insertion_code']}")
            index_map = {(str(r["model_serial"]), r["chain_id"], r["author_residue_number"],
                          r["insertion_code"]): str(i + 1) for i, r in enumerate(identities)}
            rows = _atom_rows(data)
            data["_atom_site.auth_seq_id"] = [index_map[_residue_key(row)] for row in rows]
            data["_atom_site.auth_asym_id"] = ["A"] * len(rows)
            data["_atom_site.label_asym_id"] = ["A"] * len(rows)
            data["_atom_site.pdbx_PDB_ins_code"] = ["?"] * len(rows)
            stem = f"structure-{record_id.encode().hex()}"
            destination = output / f"{stem}.cif"
            writer = MMCIFIO()
            writer.set_dict(data)
            writer.save(str(destination))
            record.update(execution_path=str(destination), identity_map_path=str(output / f"{stem}.json"),
                          execution_identity_policy="chain A, auth index 1..N; original identity in ca_identities")
        except (OSError, ValueError, KeyError, IndexError) as error:
            record.update(status="coordinate_error", reason=f"{type(error).__name__}: {error}")
        _write_json(output / f"structure-{record_id.encode().hex()}.json", record)
    _write_json(output / "coordinate-manifest.json", prepared)
    return prepared


def parse_transform(text: str) -> dict[str, list]:
    """Parse a proper rigid transform with convention y = R @ x + t."""
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0] in {"0", "1", "2"}:
            try:
                values = [float(x) for x in fields[1:]]
            except ValueError:
                continue
            if int(fields[0]) in rows:
                raise ValueError("Duplicate matrix row")
            rows[int(fields[0])] = values
    if set(rows) != {0, 1, 2}:
        raise ValueError("Missing complete rotation/translation matrix")
    matrix = np.array([rows[i] for i in range(3)])
    rotation, translation = matrix[:, 1:], matrix[:, 0]
    if (not np.isfinite(matrix).all() or
            not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError("Transformation is not a finite proper rigid rotation")
    return {"rotation": rotation.tolist(), "translation": translation.tolist()}


def parse_residue_correspondence(stdout: str, query_residues, target_residues) -> list[dict]:
    """Decode native -do records from execution-indexed complete domain inputs."""
    result = []
    used_query, used_target = set(), set()
    for line in stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].startswith(" CA "):
            continue
        decoded = []
        for field, residues in zip(fields[:2], (query_residues, target_residues), strict=True):
            if len(field) != 15 or field[9] != "A" or field[14].strip():
                raise ValueError("Unexpected native residue identity format")
            index = int(field[10:14]) - 1
            if index < 0 or index >= len(residues):
                raise ValueError("Native residue index outside inspected input")
            if field[5:8] != residues[index]["residue_name"]:
                raise ValueError("Native residue name disagrees with exact identity map")
            decoded.append(index)
        qi, ti = decoded
        if qi in used_query or ti in used_target:
            raise ValueError("Native correspondence is not one-to-one")
        distance = float(fields[2])
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Invalid native residue-pair distance")
        used_query.add(qi)
        used_target.add(ti)
        result.append({"query_index0": qi, "target_index0": ti,
                       "query_residue_uid": query_residues[qi]["residue_uid"],
                       "target_residue_uid": target_residues[ti]["residue_uid"],
                       "native_distance_angstrom": distance})
    if not result:
        raise ValueError("No native residue correspondences found; sequence display is not a mapping")
    return result


def _fit_rigid(query, target):
    query, target = np.asarray(query, dtype=float), np.asarray(target, dtype=float)
    if query.shape != target.shape or len(query) < 3:
        return None
    qcenter, tcenter = query.mean(axis=0), target.mean(axis=0)
    if min(np.linalg.matrix_rank(query - qcenter), np.linalg.matrix_rank(target - tcenter)) < 2:
        return None
    u, _, vt = np.linalg.svd((query - qcenter).T @ (target - tcenter))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = (u @ correction @ vt).T
    translation = tcenter - rotation @ qcenter
    distances = np.linalg.norm(query @ rotation.T + translation - target, axis=1)
    return {"rotation": rotation.tolist(), "translation": translation.tolist(),
            "rmsd_angstrom": float(np.sqrt(np.mean(distances ** 2))),
            "distances": distances.tolist()}


def _beta_annotations(residues, graph):
    if not graph or "nodes" not in graph:
        raise ValueError("Connectivity graph with beta nodes is required")
    uid_to_index = {r["residue_uid"]: i for i, r in enumerate(residues)}
    if len(uid_to_index) != len(residues):
        raise ValueError("Duplicate CA residue identity")
    annotation, nodes, sheets = {}, {}, defaultdict(set)
    for node in graph["nodes"]:
        key = str(node["node_id"])
        if key in nodes:
            raise ValueError("Duplicate beta strand ID")
        members = []
        uids = node.get("residue_uids", [])
        positions = node.get("label_positions", [])
        if len(uids) != len(positions) or not uids:
            raise ValueError("Beta strand lacks a complete residue identity list")
        for uid, label in zip(uids, positions, strict=True):
            if uid not in uid_to_index:
                raise ValueError(f"Annotated beta residue missing from exact domain input: {uid}")
            i = uid_to_index[uid]
            residue = residues[i]
            if residue["chain_id"] != node["chain_id"] or residue["label_seq_id"] != int(label):
                raise ValueError("Beta graph and coordinate polymer identities disagree")
            if i in annotation:
                raise ValueError("Residue assigned to multiple beta strands")
            annotation[i] = {"strand": key, "sheet": str(node.get("sheet_id", "")),
                             "label": int(label)}
            members.append(i)
        nodes[key] = members
        if node.get("sheet_id"):
            sheets[str(node["sheet_id"])].update(members)
    return annotation, nodes, dict(sheets)


def _threshold_name(value):
    return f"{value:g}".replace(".", "p")


def _runs(pairs):
    """Contiguous correspondence runs; polymer gaps break runs on both sides."""
    ordered = sorted(pairs)
    if not ordered:
        return 0, 0
    sizes, size = [], 1
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        if right[0] - left[0] == 1 and abs(right[1] - left[1]) == 1:
            size += 1
        else:
            sizes.append(size)
            size = 1
    sizes.append(size)
    return len(sizes), max(sizes)


def measure_scaffold(query_residues, target_residues, correspondences, transform,
                     query_graph, target_graph, *, thresholds=THRESHOLDS,
                     strand_min_fraction=0.5, graph_comparator=None) -> dict:
    """Measure actual one-to-one correspondences, without reassigning atoms.

    Primary spatial coverage is beta-to-beta at each threshold under the native
    full-domain transform. A covered strand has at least ``strand_min_fraction``
    of its own beta residues covered. Raw per-strand fractions are also retained.
    Independent sheet fits keep the same correspondences as the joint fit.
    """
    thresholds = tuple(float(x) for x in thresholds)
    if not thresholds or any(not math.isfinite(x) or x <= 0 for x in thresholds):
        raise ValueError("Distance thresholds must be finite and positive")
    if not 0 < strand_min_fraction <= 1:
        raise ValueError("strand_min_fraction must be in (0, 1]")
    qa, qnodes, qsheets = _beta_annotations(query_residues, query_graph)
    ta, tnodes, tsheets = _beta_annotations(target_residues, target_graph)
    qxyz = np.array([r["coordinate"] for r in query_residues], dtype=float)
    txyz = np.array([r["coordinate"] for r in target_residues], dtype=float)
    rotation, translation = np.asarray(transform["rotation"]), np.asarray(transform["translation"])
    moved = qxyz @ rotation.T + translation
    pairs, beta_pairs, by_strand, by_sheet = [], [], defaultdict(list), defaultdict(list)
    seen_q, seen_t = set(), set()
    for source in correspondences:
        qi, ti = int(source["query_index0"]), int(source["target_index0"])
        if qi in seen_q or ti in seen_t or not 0 <= qi < len(qxyz) or not 0 <= ti < len(txyz):
            raise ValueError("Invalid or non-unique actual atom correspondence")
        seen_q.add(qi)
        seen_t.add(ti)
        distance = float(np.linalg.norm(moved[qi] - txyz[ti]))
        row = {**source, "distance_angstrom": distance,
               "query_beta": qi in qa, "target_beta": ti in ta,
               "query_strand": qa.get(qi, {}).get("strand"),
               "target_strand": ta.get(ti, {}).get("strand"),
               "query_sheet": qa.get(qi, {}).get("sheet"),
               "target_sheet": ta.get(ti, {}).get("sheet")}
        pairs.append(row)
        if qi in qa and ti in ta:
            beta_pairs.append(row)
            by_strand[(qa[qi]["strand"], ta[ti]["strand"])].append(row)
            if qa[qi]["sheet"] and ta[ti]["sheet"]:
                by_sheet[(qa[qi]["sheet"], ta[ti]["sheet"])].append(row)
    summary = {"query_beta_count": len(qa), "target_beta_count": len(ta),
               "query_strand_count": len(qnodes), "target_strand_count": len(tnodes),
               "beta_to_beta_pair_count": len(beta_pairs), "correspondence_count": len(pairs),
               "beta_annotation_status": "ok" if qa and ta else "no_beta_on_one_or_both_sides"}
    summary.update(beta_to_nonbeta_pair_count=sum(p["query_beta"] and not p["target_beta"] for p in pairs),
                   nonbeta_to_beta_pair_count=sum(not p["query_beta"] and p["target_beta"] for p in pairs),
                   nonbeta_to_nonbeta_pair_count=sum(not p["query_beta"] and not p["target_beta"] for p in pairs))
    # These contributions refer to this one saved transform and the returned
    # correspondence set. They are not a decomposition of separately optimized
    # native TM-scores, which can use different normalizations/transforms.
    for role, length in (("query", len(query_residues)), ("target", len(target_residues))):
        d0 = max(0.5, 1.24 * max(length - 15, 0) ** (1 / 3) - 1.8)
        weights = [(1 / (1 + (p["distance_angstrom"] / d0) ** 2)) / length for p in pairs]
        summary[f"fixed_transform_{role}_tm_style_beta_contribution"] = sum(
            weight for p, weight in zip(pairs, weights, strict=True) if p[f"{role}_beta"])
        summary[f"fixed_transform_{role}_tm_style_other_contribution"] = sum(
            weight for p, weight in zip(pairs, weights, strict=True) if not p[f"{role}_beta"])
        summary[f"fixed_transform_{role}_tm_style_total"] = sum(weights)
    for threshold in thresholds:
        name = _threshold_name(threshold)
        close = [p for p in beta_pairs if p["distance_angstrom"] <= threshold]
        qcount, tcount = Counter(p["query_strand"] for p in close), Counter(p["target_strand"] for p in close)
        for role, beta_annotations, nodes, counts in (("query", qa, qnodes, qcount), ("target", ta, tnodes, tcount)):
            summary[f"{role}_beta_coverage_{name}"] = len(close) / len(beta_annotations) if beta_annotations else None
            summary[f"{role}_strand_coverage_{name}"] = (
                sum(counts[key] / len(members) >= strand_min_fraction for key, members in nodes.items()) / len(nodes)
                if nodes else None)
            summary[f"{role}_full_strand_coverage_{name}"] = (
                sum(counts[key] == len(members) for key, members in nodes.items()) / len(nodes) if nodes else None)
        for kind in ("beta", "strand", "full_strand"):
            values = [summary[f"{role}_{kind}_coverage_{name}"] for role in ("query", "target")]
            summary[f"min_{kind}_coverage_{name}"] = min(values) if None not in values else None
    strand_rows = []
    for (query, target), members in sorted(by_strand.items()):
        labels = [(qa[p["query_index0"]]["label"], ta[p["target_index0"]]["label"]) for p in members]
        n_runs, longest = _runs(labels)
        ordered = sorted(labels)
        differences = np.diff([p[1] for p in ordered])
        direction = "forward" if len(differences) and np.all(differences > 0) else (
            "reverse" if len(differences) and np.all(differences < 0) else "mixed_or_single")
        item = {"query_strand": query, "target_strand": target, "n_correspondences": len(members),
                "query_fraction": len(members) / len(qnodes[query]),
                "target_fraction": len(members) / len(tnodes[target]),
                "correspondence_run_count": n_runs, "longest_run_residues": longest,
                "orientation": direction}
        for threshold in thresholds:
            name = _threshold_name(threshold)
            close = [p for p in members if p["distance_angstrom"] <= threshold]
            item[f"n_within_{name}"] = len(close)
            item[f"query_fraction_{name}"] = len(close) / len(qnodes[query])
            item[f"target_fraction_{name}"] = len(close) / len(tnodes[target])
            item[f"run_count_{name}"], item[f"longest_run_{name}"] = _runs([
                (qa[p["query_index0"]]["label"], ta[p["target_index0"]]["label"]) for p in close])
        strand_rows.append(item)
    # A partial spatial map must never become a full bijection by optimization.
    spatial_map = {}
    for row in strand_rows:
        query, target = row["query_strand"], row["target_strand"]
        candidates_q = [r for r in strand_rows if r["query_strand"] == query]
        candidates_t = [r for r in strand_rows if r["target_strand"] == target]
        # 3.5 A is the prespecified mapping readout, alongside the full curves.
        count = sum(p["distance_angstrom"] <= 3.5 for p in by_strand[(query, target)])
        counts_q = [sum(p["distance_angstrom"] <= 3.5 for p in by_strand[(r["query_strand"], r["target_strand"])]) for r in candidates_q]
        counts_t = [sum(p["distance_angstrom"] <= 3.5 for p in by_strand[(r["query_strand"], r["target_strand"])]) for r in candidates_t]
        dominant = (count > 0 and count == max(counts_q) == max(counts_t)
                    and counts_q.count(count) == 1 and counts_t.count(count) == 1)
        row["reciprocal_dominant"] = dominant
        row["spatial_mapping_eligible"] = (dominant and count / len(qnodes[query]) >= strand_min_fraction
                                             and count / len(tnodes[target]) >= strand_min_fraction)
        if row["spatial_mapping_eligible"]:
            spatial_map[query] = target
    for role, nodes, key in (("query", qnodes, "query_strand"), ("target", tnodes, "target_strand")):
        multiplicity = Counter(row[key] for row in strand_rows)
        summary[f"{role}_split_strand_fraction"] = sum(multiplicity[n] > 1 for n in nodes) / len(nodes) if nodes else None
        summary[f"{role}_spatially_mapped_strand_fraction"] = len(spatial_map) / len(nodes) if nodes else None
    summary["strand_correspondence_run_count"] = sum(r["correspondence_run_count"] for r in strand_rows)
    summary["strand_correspondence_longest_run_fraction"] = (
        sum(r["longest_run_residues"] for r in strand_rows) / len(beta_pairs) if beta_pairs else None)
    sheet_rows, fit_members = [], []
    for (query, target), members in sorted(by_sheet.items()):
        qi, ti = [p["query_index0"] for p in members], [p["target_index0"] for p in members]
        fitted = _fit_rigid(qxyz[qi], txyz[ti])
        sheet = {"query_sheet": query, "target_sheet": target, "n_correspondences": len(members),
                 "query_sheet_fraction": len(members) / len(qsheets[query]),
                 "target_sheet_fraction": len(members) / len(tsheets[target]),
                 "fit_status": "ok" if fitted else "insufficient_noncollinear_correspondences",
                 "full_domain_transform_rmsd": float(np.sqrt(np.mean([p["distance_angstrom"] ** 2 for p in members]))),
                 "independent_fit": fitted}
        if fitted:
            fit_members.extend(members)
        sheet_rows.append(sheet)
    joint = _fit_rigid([qxyz[p["query_index0"]] for p in fit_members],
                       [txyz[p["target_index0"]] for p in fit_members]) if fit_members else None
    fitted_sheets = [row for row in sheet_rows if row["independent_fit"]]
    independent_rmsd = (math.sqrt(sum(row["n_correspondences"] * row["independent_fit"]["rmsd_angstrom"] ** 2
                                    for row in fitted_sheets) / len(fit_members)) if fit_members else None)
    summary.update(sheet_pair_count=len(sheet_rows), fitted_sheet_pair_count=len(fitted_sheets),
                   sheet_fit_residue_pair_count=len(fit_members),
                   common_sheet_fit_rmsd=joint["rmsd_angstrom"] if joint else None,
                   independent_sheet_fit_rmsd=independent_rmsd,
                   common_minus_independent_sheet_rmsd=(joint["rmsd_angstrom"] - independent_rmsd) if joint else None)
    query_multiplicity = Counter(row["query_sheet"] for row in sheet_rows)
    target_multiplicity = Counter(row["target_sheet"] for row in sheet_rows)
    one_to_one_members, one_to_one_sheets = [], []
    for row in sheet_rows:
        query, target, count = row["query_sheet"], row["target_sheet"], row["n_correspondences"]
        qcounts = [r["n_correspondences"] for r in sheet_rows if r["query_sheet"] == query]
        tcounts = [r["n_correspondences"] for r in sheet_rows if r["target_sheet"] == target]
        dominant = (count == max(qcounts) == max(tcounts) and qcounts.count(count) == 1 and tcounts.count(count) == 1)
        eligible = (dominant and row["query_sheet_fraction"] >= strand_min_fraction
                    and row["target_sheet_fraction"] >= strand_min_fraction and row["fit_status"] == "ok")
        row.update(query_sheet_multiplicity=query_multiplicity[query],
                   target_sheet_multiplicity=target_multiplicity[target],
                   reciprocal_dominant=dominant, one_to_one_sheet_eligible=eligible)
        if eligible:
            one_to_one_members.extend(by_sheet[(query, target)])
            one_to_one_sheets.append(row)
    whole_sheet_bijection = bool(qsheets and tsheets and len(one_to_one_sheets) == len(qsheets) == len(tsheets))
    bijective_joint = _fit_rigid([qxyz[p["query_index0"]] for p in one_to_one_members],
                                 [txyz[p["target_index0"]] for p in one_to_one_members]) if one_to_one_members else None
    bijective_independent = math.sqrt(sum(row["n_correspondences"] * row["independent_fit"]["rmsd_angstrom"] ** 2
        for row in one_to_one_sheets) / len(one_to_one_members)) if one_to_one_members else None
    multiple = len(one_to_one_sheets) >= 2
    summary.update(query_sheet_count=len(qsheets), target_sheet_count=len(tsheets),
                   query_split_sheet_count=sum(count > 1 for count in query_multiplicity.values()),
                   target_split_sheet_count=sum(count > 1 for count in target_multiplicity.values()),
                   whole_sheet_bijection=whole_sheet_bijection,
                   one_to_one_sheet_pair_count=len(one_to_one_sheets),
                   one_to_one_sheet_fit_residue_pair_count=len(one_to_one_members),
                   one_to_one_sheet_fit_status="multiple_sheets" if multiple else (
                       "single_sheet_only" if one_to_one_sheets else "no_eligible_sheet_pairs"),
                   one_to_one_sheet_common_fit_rmsd=bijective_joint["rmsd_angstrom"] if bijective_joint else None,
                   one_to_one_sheet_independent_fit_rmsd=bijective_independent,
                   one_to_one_sheet_assembly_rmsd_gap=(bijective_joint["rmsd_angstrom"] - bijective_independent)
                   if multiple and bijective_joint else None)
    compatibility = graph_comparator(query_graph, target_graph, spatial_map) if graph_comparator else None
    if compatibility is not None:
        summary.update({f"graph_{key}": value for key, value in compatibility.items()
                        if value is None or isinstance(value, (str, bool, int, float))})
    return {"summary": summary, "residue_correspondences": pairs, "strand_correspondences": strand_rows,
            "spatial_strand_mapping": spatial_map, "graph_mapping_comparison": compatibility,
            "sheet_correspondences": sheet_rows, "common_sheet_fit": joint,
            "one_to_one_sheet_common_fit": bijective_joint,
            "settings": {"thresholds_angstrom": list(thresholds), "strand_min_fraction": strand_min_fraction,
                         "spatial_mapping_threshold_angstrom": 3.5,
                         "sheet_fit_scope": "all sheet-pair fragments; secondary descriptive fit, may split sheets",
                         "primary_sheet_fit_scope": "reciprocal unique dominant sheet pairs, both >=50% coverage; assembly gap requires >=2 pairs",
                         "tm_style_contribution_scope": "saved common transform and returned correspondences, not native separately optimized TM decomposition"}}


def run_scaffold_alignments(pairs, coordinate_manifest, connectivity, output_dir, *,
                           modes=("SQ", "sNS"), binary="USalign", workers=4, timeout=120,
                           prepared_coordinates=None, graph_comparator=None) -> list[dict]:
    """Run a bounded frozen panel, retaining failures and complete native evidence.

    Each pair can specify a single ``mode``; otherwise all requested modes run.
    ``prepared_coordinates`` may reuse the return value of export_coordinates.
    Outputs are new-only. Runtime inputs are staged in /tmp, preserving source
    coordinate tokens and the original-identity sidecar in the output directory.
    """
    rows = _pairs(pairs)
    modes = tuple(modes)
    if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("Modes must be unique SQ, sNS or fNS values")
    if workers < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Workers and timeout must be positive")
    jobs = [(row, mode) for row in rows for mode in ([row["mode"]] if row.get("mode") else modes)]
    if any(mode not in MODES for _, mode in jobs):
        raise ValueError("Unknown pair-specific alignment mode")
    executable = shutil.which(str(binary))
    if executable is None:
        raise FileNotFoundError(f"USalign executable unavailable: {binary}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    ids = {row[key] for row in rows for key in ("query_id", "target_id")}
    graphs = load_graphs(connectivity, ids)
    prepared = (export_coordinates(ids, coordinate_manifest, output / "coordinates")
                if prepared_coordinates is None else dict(prepared_coordinates))
    _write_json(output / "coordinate-inspection.json", {key: prepared.get(key) for key in sorted(ids)})
    version = subprocess.run([executable, "-v"], capture_output=True, text=True, timeout=20, check=False)
    (output / "version.stdout.txt").write_text(version.stdout)
    (output / "version.stderr.txt").write_text(version.stderr)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="betlas-scaffold-") as scratch:
        staged = {}
        for key in sorted(ids):
            record = prepared.get(key, {})
            if record.get("status") == "ok":
                destination = Path(scratch) / f"{key.encode().hex()}.cif"
                destination.write_text(_read_text(Path(record["execution_path"])), encoding="utf-8")
                staged[key] = str(destination)

        def execute(item):
            index, (pair, mode) = item
            stem = f"{index:05d}-{mode}"
            relative = f"raw/{stem}"
            row = {"pair_id": pair["pair_id"], "query_id": pair["query_id"], "target_id": pair["target_id"],
                   "mode": mode, "status": "not_run", "reason": "", "elapsed_seconds": 0.0,
                   "native_elapsed_seconds": 0.0, "returncode": None,
                   "stdout_path": relative + ".stdout.txt", "stderr_path": relative + ".stderr.txt",
                   "matrix_path": relative + ".matrix.txt", "command_path": relative + ".command.json",
                   "correspondence_path": relative + ".correspondence.json"}
            command, stdout, stderr, matrix_text, details = [], "", "", "", None
            query, target = prepared.get(pair["query_id"], {}), prepared.get(pair["target_id"], {})
            tick = time.monotonic()
            if query.get("status") != "ok" or target.get("status") != "ok":
                row.update(status="input_unavailable", reason=f"query={query.get('status')}; target={target.get('status')}")
            else:
                matrix = Path(scratch) / f"{stem}.matrix.txt"
                command = [executable, staged[pair["query_id"]], staged[pair["target_id"]],
                           "-ter", "1", "-split", "0", "-het", "1", "-mol", "prot", "-outfmt", "0",
                           "-mm", str(MODES[mode]), "-do", "-m", str(matrix)]
                native_tick = time.monotonic()
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
                    row["native_elapsed_seconds"] = time.monotonic() - native_tick
                    stdout, stderr = result.stdout, result.stderr
                    row["returncode"] = result.returncode
                    if result.returncode:
                        row.update(status="execution_failed", reason=f"Exit code {result.returncode}")
                    else:
                        native_metrics = parse_alignment_output(stdout)
                        if (native_metrics["query_length"] != query["ca_count"] or
                                native_metrics["target_length"] != target["ca_count"]):
                            raise ValueError("Native aligner did not consume all inspected domain CA residues")
                        matrix_text = matrix.read_text()
                        transform = parse_transform(matrix_text)
                        correspondences = parse_residue_correspondence(stdout, query["ca_identities"], target["ca_identities"])
                        if len(correspondences) != native_metrics["aligned_length"]:
                            raise ValueError("Native aligned length differs from actual correspondence count")
                        # Save alignment even if beta/graph annotation subsequently proves unusable.
                        details = {"schema": SCHEMA, "mode": mode, "transform": transform,
                                   "native_metrics": native_metrics, "residue_correspondences": correspondences}
                        row.update(native_metrics, alignment_status="ok")
                        qgraph, tgraph = graphs.get(pair["query_id"]), graphs.get(pair["target_id"])
                        if qgraph is None or tgraph is None:
                            row.update(status="annotation_unavailable", reason="Missing retained connectivity graph")
                        else:
                            try:
                                measured = measure_scaffold(query["ca_identities"], target["ca_identities"],
                                                            correspondences, transform, qgraph, tgraph,
                                                            graph_comparator=graph_comparator)
                                differences = [abs(p["distance_angstrom"] - p["native_distance_angstrom"])
                                               for p in measured["residue_correspondences"]]
                                # Native distances have three decimals; input text may retain more.
                                if max(differences, default=0) > 0.005:
                                    raise ValueError("Exact-coordinate transformed distance differs from native output by >0.005 A")
                                details.update(measured)
                                row.update(measured["summary"], status="ok",
                                           max_native_distance_difference=max(differences, default=0))
                            except (ValueError, KeyError, IndexError) as error:
                                row.update(status="measurement_failed", reason=f"{type(error).__name__}: {error}")
                except subprocess.TimeoutExpired as error:
                    stdout, stderr = error.stdout or "", error.stderr or ""
                    row.update(status="timeout", reason=f"Exceeded {timeout:g} seconds",
                               native_elapsed_seconds=time.monotonic() - native_tick)
                except (ValueError, OSError, IndexError) as error:
                    row.update(status="alignment_parse_failed", reason=f"{type(error).__name__}: {error}")
            row["elapsed_seconds"] = time.monotonic() - tick
            for content, field in ((stdout, "stdout_path"), (stderr, "stderr_path"), (matrix_text, "matrix_path")):
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                (output / row[field]).write_text(content, encoding="utf-8")
            _write_json(output / row["command_path"], {"argv": command, "mode": mode,
                        "query_source": query.get("source_path"), "target_source": target.get("source_path"),
                        "execution_identity_policy": "chain A, residue index; original-identity sidecar retained"})
            _write_json(output / row["correspondence_path"], details)
            _write_json(output / f"{relative}.result.json", row)
            return row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(execute, enumerate(jobs)))
    fields = list(dict.fromkeys(key for row in results for key in row))
    _write_csv(output / "scaffold-alignments.csv", results, fields or ["pair_id", "mode", "status"])
    _write_json(output / "summary.json", {"schema": SCHEMA, "pair_count": len(rows), "alignment_count": len(jobs),
                "statuses": dict(Counter(row["status"] for row in results)), "binary": executable,
                "workers": workers, "timeout_seconds": timeout, "elapsed_seconds": time.monotonic() - started,
                "modes": MODES, "thresholds_angstrom": list(THRESHOLDS), "strand_min_fraction": 0.5,
                "coverage_definition": "actual beta-to-beta correspondences / all retained beta residues on each side",
                "human_review": "not_performed", "missing_policy": "unavailable, never zero similarity"})
    return results
