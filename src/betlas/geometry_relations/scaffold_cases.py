"""Bounded, reproducible coordinate illustrations of scaffold measurements.

Case selection is for explaining measured phenomena, not estimating prevalence.
Rendered coordinates are never rescaled per structure. Pair overlays use the
saved common rigid transformation; separately oriented fixed-K series explicitly
do not represent pairwise structural superpositions or expert adjudication.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .scaffold_alignment import _beta_annotations, load_graphs
from .validation import _read_text, _records, _write_csv, _write_json

CASE_PROTOCOL = {
    "schema": "betlas.scaffold-illustrations.v1",
    "selection_purpose": "fixed descriptive case selection, not a prevalence sample",
    "maximum_per_kind": 4,
    "maximum_pair_cases": 12,
    "maximum_fixed_graph_series": 3,
    "maximum_total_cases": 15,
    "shape_shared_graph_different": {
        "mode": "sNS", "min_beta_coverage_3p5_at_least": 0.60,
        "min_strand_coverage_3p5_at_least": 0.50,
        "complete_spatial_strand_mapping": True,
        "pair_or_chain_compatibility_explicitly_false": True,
        "rank": "descending minimum beta coverage, then minimum strand coverage",
    },
    "sheet_placement_difference": {
        "mode": "sNS", "one_to_one_sheet_pairs_at_least": 2,
        "whole_sheet_bijection": True,
        "independent_sheet_fit_rmsd_at_most": 2.0,
        "assembly_rmsd_gap_at_least": 1.5,
        "rank": "descending assembly RMSD gap, then minimum beta coverage",
    },
    "freer_alignment_more_fragmented": {
        "mode": "fNS", "reference_mode": "sNS",
        "min_beta_coverage_3p5_gain_at_least": 0.15,
        "mean_split_strand_fraction_increase_at_least_or": 0.15,
        "longest_run_fraction_decline_at_least_or": 0.15,
        "rank": "descending beta coverage gain, then fragmentation change",
    },
    "fixed_graph_series": {
        "source": "already frozen pattern/elongation-quantile representative table",
        "additional_structure_alignments": 0,
        "camera": "each domain's own beta PCA frame, shared angstrom scale; not a common superposition",
    },
    "deduplication": "unordered domain pair across all pair-case kinds",
    "tie_break": "query ID, target ID, alignment directory, pair ID",
    "unsupported_kind_policy": "leave empty; do not lower thresholds or generate additional pairs",
    "coordinate_display": "actual CA coordinates; equal physical aspect; preserve missing polymer gaps",
    "review_status": "computational measurements; expert and author review not performed",
}

PAIR_KINDS = tuple(key for key in CASE_PROTOCOL if key in {
    "shape_shared_graph_different", "sheet_placement_difference", "freer_alignment_more_fragmented"})


def freeze_case_protocol(path):
    """Save the selection definition before inspecting formal panel outcomes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(_read_text(path)) != CASE_PROTOCOL:
            raise ValueError("Existing frozen case protocol differs")
    else:
        _write_json(path, CASE_PROTOCOL)
    return dict(CASE_PROTOCOL)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _boolean(value):
    if value is True or value == "True" or value == "true":
        return True
    if value is False or value == "False" or value == "false":
        return False
    return None


def _at_least(row, key, threshold):
    value = _number(row.get(key))
    return value is not None and value >= threshold


def read_alignment_tables(paths):
    result = []
    for source in paths:
        path = Path(source)
        if path.is_dir():
            path = path / "scaffold-alignments.csv"
        for row in _records(path):
            result.append({**row, "alignment_directory": str(path.parent),
                           "correspondence_file": str(path.parent / row["correspondence_path"])})
    return result


def _split_fraction(row):
    values = [_number(row.get(f"{side}_split_strand_fraction")) for side in ("query", "target")]
    return sum(values) / 2 if None not in values else None


def select_cases(alignment_rows, *, maximum_per_kind=4):
    """Apply the frozen rule and retain an eligibility ledger, including zeros."""
    if not 0 <= maximum_per_kind <= CASE_PROTOCOL["maximum_per_kind"]:
        raise ValueError("Case count exceeds the frozen per-kind bound")
    rows = [dict(row) for row in alignment_rows]
    reference = {(r["query_id"], r["target_id"], r.get("alignment_directory", "")): r
                 for r in rows if r.get("mode") == "sNS" and r.get("status") == "ok"}
    pools = {kind: [] for kind in PAIR_KINDS}
    ledger = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        base = {key: row.get(key, "") for key in ("pair_id", "query_id", "target_id", "mode", "alignment_directory")}
        if row.get("mode") == "sNS":
            shared = (_at_least(row, "min_beta_coverage_3p5", .60)
                      and _at_least(row, "min_strand_coverage_3p5", .50)
                      and _boolean(row.get("graph_mapping_complete")) is True
                      and (_boolean(row.get("graph_pair_mapping_compatible")) is False
                           or _boolean(row.get("graph_chain_mapping_compatible")) is False))
            independent = _number(row.get("one_to_one_sheet_independent_fit_rmsd"))
            assembly = (_at_least(row, "one_to_one_sheet_pair_count", 2)
                        and _boolean(row.get("whole_sheet_bijection")) is True
                        and independent is not None and independent <= 2
                        and _at_least(row, "one_to_one_sheet_assembly_rmsd_gap", 1.5))
            for kind, eligible, score in (
                ("shape_shared_graph_different", shared,
                 (_number(row.get("min_beta_coverage_3p5")) or 0,
                  _number(row.get("min_strand_coverage_3p5")) or 0)),
                ("sheet_placement_difference", assembly,
                 (_number(row.get("one_to_one_sheet_assembly_rmsd_gap")) or 0,
                  _number(row.get("min_beta_coverage_3p5")) or 0)),
            ):
                ledger.append({**base, "case_kind": kind, "eligible": eligible,
                               "primary_rank_value": score[0], "secondary_rank_value": score[1]})
                if eligible:
                    pools[kind].append((score, row, None))
        elif row.get("mode") == "fNS":
            comparison = reference.get((row["query_id"], row["target_id"], row.get("alignment_directory", "")))
            gain, split_gain, run_decline = None, None, None
            if comparison:
                left, right = _number(row.get("min_beta_coverage_3p5")), _number(comparison.get("min_beta_coverage_3p5"))
                gain = left - right if left is not None and right is not None else None
                left, right = _split_fraction(row), _split_fraction(comparison)
                split_gain = left - right if left is not None and right is not None else None
                left = _number(row.get("strand_correspondence_longest_run_fraction"))
                right = _number(comparison.get("strand_correspondence_longest_run_fraction"))
                run_decline = right - left if left is not None and right is not None else None
            fragmented = ((split_gain is not None and split_gain >= .15)
                          or (run_decline is not None and run_decline >= .15))
            eligible = gain is not None and gain >= .15 and fragmented
            kind = "freer_alignment_more_fragmented"
            ledger.append({**base, "case_kind": kind, "eligible": eligible,
                           "beta_coverage_gain": gain, "mean_split_strand_fraction_gain": split_gain,
                           "longest_run_fraction_decline": run_decline})
            if eligible:
                pools[kind].append(((gain, max(split_gain or 0, run_decline or 0)), row, comparison))
    selected, used = [], set()
    for kind in PAIR_KINDS:
        candidates = sorted(pools[kind], key=lambda item: (
            -item[0][0], -item[0][1], str(item[1]["query_id"]), str(item[1]["target_id"]),
            str(item[1].get("alignment_directory", "")), str(item[1]["pair_id"])))
        count = 0
        for score, row, reference_row in candidates:
            pair_key = tuple(sorted((row["query_id"], row["target_id"])))
            if pair_key in used or count >= maximum_per_kind:
                continue
            count += 1
            used.add(pair_key)
            selected.append({"case_id": f"C{len(selected)+1:02d}", "case_kind": kind,
                             "alignment": row, "reference_alignment": reference_row,
                             "rank_values": list(score)})
    return selected, ledger


def load_prepared_coordinates(paths):
    """Merge selected-domain caches without silently replacing different atoms."""
    result = {}
    for source in paths:
        data = source if isinstance(source, Mapping) else json.loads(_read_text(Path(source)))
        for key, row in data.items():
            if key in result and result[key].get("ca_identities") != row.get("ca_identities"):
                raise ValueError(f"Coordinate identity differs between caches: {key}")
            result[key] = row
    return result


def _camera(residues, graph):
    beta, _, _ = _beta_annotations(residues, graph)
    xyz = np.asarray([r["coordinate"] for r in residues], dtype=float)
    points = xyz[sorted(beta)] if beta else xyz
    center = points.mean(axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    if basis.shape != (3, 3):
        raise ValueError("A three-dimensional camera requires at least three CA points")
    for axis in range(3):
        if basis[axis, np.argmax(np.abs(basis[axis]))] < 0:
            basis[axis] *= -1
    if np.linalg.det(basis) < 0:
        basis[-1] *= -1
    return center, basis


def project_pair(query_residues, target_residues, target_graph, transform):
    """One common physical frame: native query-to-target transform then camera."""
    center, basis = _camera(target_residues, target_graph)
    query = np.asarray([r["coordinate"] for r in query_residues], dtype=float)
    target = np.asarray([r["coordinate"] for r in target_residues], dtype=float)
    rotation, translation = np.asarray(transform["rotation"]), np.asarray(transform["translation"])
    query = query @ rotation.T + translation
    qp, tp = (query - center) @ basis.T, (target - center) @ basis.T
    combined = np.vstack((qp, tp))
    minimum, maximum = combined.min(axis=0), combined.max(axis=0)
    span = max(float(np.max(maximum - minimum)) * 1.12, 1)
    midpoint = (minimum + maximum) / 2
    return qp, tp, {"camera_origin": center.tolist(), "camera_basis_rows": basis.tolist(),
                    "axis_centers": midpoint.tolist(), "common_span_angstrom": span,
                    "query_to_target_transform": transform,
                    "policy": "target beta PCA camera; one native common transform; no geometric scaling"}


def trace_segments(residues, indices=None):
    """Break traces at unobserved polymer positions, rather than joining gaps."""
    indices = list(range(len(residues))) if indices is None else sorted(indices)
    segments, current = [], []
    for index in indices:
        residue = residues[index]
        if current:
            previous = residues[current[-1]]
            if (residue["chain_id"] != previous["chain_id"]
                    or residue["label_seq_id"] != previous["label_seq_id"] + 1):
                segments.append(current)
                current = []
        current.append(index)
    if current:
        segments.append(current)
    return segments


def _plot_coordinate_view(ax, qp, tp, qr, tr, qg, tg, camera, axes, *, title, labels=False):
    from matplotlib.lines import Line2D
    for points, residues, graph, color, style, prefix in (
        (qp, qr, qg, "#D55E00", "-", "Q"), (tp, tr, tg, "#0072B2", "--", "T")
    ):
        for segment in trace_segments(residues):
            ax.plot(points[segment, axes[0]], points[segment, axes[1]], color=color, alpha=.22,
                    lw=.65, linestyle=style)
        _, nodes, _ = _beta_annotations(residues, graph)
        for key, members in nodes.items():
            for segment in trace_segments(residues, members):
                ax.plot(points[segment, axes[0]], points[segment, axes[1]], color=color,
                        lw=1.9, linestyle=style, marker="." if len(segment) == 1 else None)
            if labels:
                centroid = points[members].mean(axis=0)
                ax.text(centroid[axes[0]], centroid[axes[1]], f"{prefix}:{key}", color=color,
                        fontsize=5.5, alpha=.85)
    half = camera["common_span_angstrom"] / 2
    center = camera["axis_centers"]
    ax.set(xlim=(center[axes[0]] - half, center[axes[0]] + half),
           ylim=(center[axes[1]] - half, center[axes[1]] + half), aspect="equal",
           xlabel=f"Camera axis {axes[0]+1} (Å)", ylabel=f"Camera axis {axes[1]+1} (Å)", title=title)
    ax.legend(handles=[Line2D([0], [0], color="#D55E00", label="Query", lw=2),
                       Line2D([0], [0], color="#0072B2", label="Target", lw=2, linestyle="--")],
              loc="upper right", fontsize=7, framealpha=.8)
    ax.spines[["top", "right"]].set_visible(False)


def _ordered_nodes(graph):
    return sorted(graph["nodes"], key=lambda n: (n["chain_id"], n["label_start"] if "label_start" in n else min(n["label_positions"])))


def _mapping_matrix(ax, detail, query_graph, target_graph):
    query = [n["node_id"] for n in _ordered_nodes(query_graph)]
    target = [n["node_id"] for n in _ordered_nodes(target_graph)]
    if not query or not target:
        ax.text(.5, .5, "No beta strand correspondence", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    matrix = np.zeros((len(query), len(target)))
    counts = {}
    for row in detail["strand_correspondences"]:
        i, j = query.index(row["query_strand"]), target.index(row["target_strand"])
        matrix[i, j] = min(row["query_fraction_3p5"], row["target_fraction_3p5"])
        counts[(i, j)] = row["n_within_3p5"]
    plot = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis", aspect="auto", interpolation="nearest")
    ax.set(xticks=range(len(target)), xticklabels=target, yticks=range(len(query)), yticklabels=query,
           xlabel="Target strand (polymer order)", ylabel="Query strand (polymer order)",
           title="Strand correspondence at 3.5 Å")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    if matrix.size <= 144:
        for (i, j), count in counts.items():
            if count:
                ax.text(j, i, str(count), ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] < .5 else "black")
    ax.figure.colorbar(plot, ax=ax, shrink=.8, label="Minimum of two strand coverage fractions")


def _pairing_diagram(ax, query_graph, target_graph=None, mapping=None):
    """Graph layout is explicitly abstract; its positions are not coordinates."""
    from matplotlib.patches import FancyArrowPatch
    mapping = mapping or {}
    positions = []
    graphs = [query_graph] if target_graph is None else [query_graph, target_graph]
    for side, graph in enumerate(graphs):
        nodes = _ordered_nodes(graph)
        xs = 0.0 if side == 0 else 1.0
        ys = np.linspace(.92, .08, len(nodes)) if len(nodes) > 1 else [.5]
        xy = {node["node_id"]: (xs, float(y)) for node, y in zip(nodes, ys, strict=True)}
        positions.append(xy)
        color = "#D55E00" if side == 0 else "#0072B2"
        for edge in graph.get("pairing_edges", []):
            sense = edge.get("sense", "")
            first, last = sorted((xy[edge["source"]], xy[edge["target"]]), key=lambda p: -p[1])
            separation = abs(first[1]-last[1])
            # Give even adjacent strands a visible outward arc. A fixed radius
            # collapses short edges into the node column in larger graphs.
            radius = 2 * (.12 + .08*separation) / max(separation, .01)
            ax.add_patch(FancyArrowPatch(first, last, arrowstyle="-",
                connectionstyle=f"arc3,rad={radius if side == 0 else -radius}", lw=1.3,
                color="#7B3294" if sense == "parallel" else "#444444",
                linestyle="-" if sense in {"parallel", "anti-parallel"} else ":"))
        for edge in graph.get("sequence_relations", []):
            first, last = xy[edge["source"]], xy[edge["target"]]
            offset = .045 if side == 0 else -.045
            ax.add_patch(FancyArrowPatch((first[0]+offset, first[1]), (last[0]+offset, last[1]),
                                        arrowstyle="->", mutation_scale=5, color="#999999", lw=.6,
                                        linestyle="-" if edge.get("observed_chain_path") is True else ":"))
        for key, (x, y) in xy.items():
            ax.scatter([x], [y], color=color, s=14, zorder=3)
            ax.text(x - .04 if side == 0 else x + .04, y, key,
                    ha="right" if side == 0 else "left", va="center", fontsize=6, color=color)
    if len(positions) == 2:
        for query, target in mapping.items():
            if query in positions[0] and target in positions[1]:
                x, y = zip(positions[0][query], positions[1][target], strict=True)
                ax.plot(x, y, color="#009E73", alpha=.55, lw=.8, zorder=0)
    statuses = [str(graph.get("pairing_status", "unspecified")) for graph in graphs]
    status_line = f"Q: {statuses[0]}; T: {statuses[1]}" if target_graph else f"Status: {statuses[0]}"
    ax.set(xlim=(-.55, 1.55 if target_graph else .55), ylim=(0, 1),
           title=f"Retained pairing + polymer order\n{status_line}")
    ax.set_axis_off()
    legend = "Purple: parallel  |  Gray: antiparallel\nArrows: polymer order; dotted paths: unobserved"
    if target_graph is not None:
        legend += "\nGreen: spatial strand map"
    ax.text(.5, -.035, legend,
            transform=ax.transAxes, ha="center", va="top", fontsize=6)


def _mapping_disagreements(query_graph, target_graph, mapping):
    targets = {frozenset((e["source"], e["target"])): e for e in target_graph.get("pairing_edges", [])}
    result = []
    for edge in query_graph.get("pairing_edges", []):
        a, b = edge["source"], edge["target"]
        if a not in mapping or b not in mapping:
            continue
        destination = targets.get(frozenset((mapping[a], mapping[b])))
        if destination is None:
            result.append(f"Query pairing {a}–{b} maps to {mapping[a]}–{mapping[b]}, without a retained target pairing edge.")
        elif edge.get("sense") != destination.get("sense"):
            result.append(f"Mapped pairing {a}–{b} / {mapping[a]}–{mapping[b]} has senses {edge.get('sense')} / {destination.get('sense')}.")
    target_order = {(e["source"], e["target"]) for e in target_graph.get("sequence_relations", [])}
    for edge in query_graph.get("sequence_relations", []):
        a, b = edge["source"], edge["target"]
        if a in mapping and b in mapping and (mapping[a], mapping[b]) not in target_order:
            result.append(f"Query consecutive-strand order {a}→{b} maps to {mapping[a]}→{mapping[b]}, which is not consecutive target N→C order.")
    return result


def _observations(case, detail, query_graph, target_graph):
    row, summary = case["alignment"], detail["summary"]
    lines = [f"{row['mode']} provides {summary['beta_to_beta_pair_count']} beta-to-beta residue correspondences; "
             f"coverage within 3.5 Å is {summary['query_beta_coverage_3p5']:.1%} of query beta residues and "
             f"{summary['target_beta_coverage_3p5']:.1%} of target beta residues.",
             f"The inputs contain {summary['query_strand_count']} / {summary['target_strand_count']} retained strands "
             f"and {summary['query_sheet_count']} / {summary['target_sheet_count']} annotated sheets."]
    lines.append(f"Retained pairing annotations are {query_graph.get('pairing_status', 'unspecified')} / "
                 f"{target_graph.get('pairing_status', 'unspecified')}, containing "
                 f"{len(query_graph.get('pairing_edges', []))} / {len(target_graph.get('pairing_edges', []))} pairing edges.")
    for sheet in sorted(detail["sheet_correspondences"], key=lambda s: -s["n_correspondences"])[:3]:
        fitted = sheet.get("independent_fit")
        text = (f"Sheet {sheet['query_sheet']}→{sheet['target_sheet']}: {sheet['n_correspondences']} corresponding residues, "
                f"{sheet['query_sheet_fraction']:.1%}/{sheet['target_sheet_fraction']:.1%} sheet coverage; "
                f"RMSD under the full-domain transform {sheet['full_domain_transform_rmsd']:.2f} Å")
        if fitted:
            text += f", independently fitted RMSD {fitted['rmsd_angstrom']:.2f} Å."
        else:
            text += "; insufficient non-collinear correspondences for an independent rigid fit."
        lines.append(text)
    if summary["one_to_one_sheet_assembly_rmsd_gap"] is not None:
        lines.append(f"On {summary['one_to_one_sheet_pair_count']} eligible one-to-one sheet pairs, common-fit / "
                     f"independent-fit RMSD is {summary['one_to_one_sheet_common_fit_rmsd']:.2f} / "
                     f"{summary['one_to_one_sheet_independent_fit_rmsd']:.2f} Å using the same residue correspondences.")
    else:
        lines.append(f"Whole-sheet assembly comparison status: {summary['one_to_one_sheet_fit_status']}.")
    lines.extend(_mapping_disagreements(query_graph, target_graph, detail["spatial_strand_mapping"])[:6])
    if case.get("reference_alignment"):
        reference = case["reference_alignment"]
        lines.append(f"The sNS reference has minimum beta coverage {_number(reference['min_beta_coverage_3p5']):.1%}; "
                     f"fNS has {summary['min_beta_coverage_3p5']:.1%}. Mean split-strand fraction changes from "
                     f"{_split_fraction(reference):.1%} to {_split_fraction(summary):.1%}.")
    lines.append("These are coordinate-derived computational observations. No independent human or expert adjudication is implied.")
    return lines


def render_pair_case(case, prepared, graphs, output_dir):
    """Render two orthographic projections, a strand map and annotated graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    row = case["alignment"]
    detail = json.loads(_read_text(Path(row["correspondence_file"])))
    qr, tr = prepared[row["query_id"]]["ca_identities"], prepared[row["target_id"]]["ca_identities"]
    qg, tg = graphs[row["query_id"]], graphs[row["target_id"]]
    qp, tp, camera = project_pair(qr, tr, tg, detail["transform"])
    reference = case.get("reference_alignment")
    nrows = 3 if reference else 2
    with plt.rc_context({"font.size": 8, "axes.titlesize": 10, "figure.titlesize": 12,
                         "savefig.bbox": None, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.5*nrows), layout="constrained")
        for axis, view in zip(axes[0], ((0, 1), (0, 2)), strict=True):
            _plot_coordinate_view(axis, qp, tp, qr, tr, qg, tg, camera, view,
                                  title=f"{row['mode']}: common rigid transform, view {view[1]}", labels=False)
        if reference:
            refdetail = json.loads(_read_text(Path(reference["correspondence_file"])))
            rq, rt, rcamera = project_pair(qr, tr, tg, refdetail["transform"])
            # All four projections have the same physical span, even when the
            # transforms place unmatched domain regions differently.
            combined = np.vstack((qp, tp, rq, rt))
            lo, hi = combined.min(axis=0), combined.max(axis=0)
            span = max(float(np.max(hi-lo))*1.12, 1)
            camera["common_span_angstrom"] = rcamera["common_span_angstrom"] = span
            camera["axis_centers"] = rcamera["axis_centers"] = ((lo+hi)/2).tolist()
            camera["reference_mode_camera"] = {**rcamera}
            for axis, view in zip(axes[1], ((0, 1), (0, 2)), strict=True):
                _plot_coordinate_view(axis, rq, rt, qr, tr, qg, tg, rcamera, view,
                                      title=f"sNS reference: view {view[1]}")
            for axis, view in zip(axes[0], ((0, 1), (0, 2)), strict=True):
                center, half = camera["axis_centers"], span/2
                axis.set(xlim=(center[view[0]]-half, center[view[0]]+half),
                         ylim=(center[view[1]]-half, center[view[1]]+half))
        _mapping_matrix(axes[-1, 0], detail, qg, tg)
        _pairing_diagram(axes[-1, 1], qg, tg, detail["spatial_strand_mapping"])
        fig.suptitle(f"{case['case_id']} | {case['case_kind'].replace('_', ' ')}\n"
                     f"{row['query_id']} (query) vs {row['target_id']} (target)\n"
                     "Actual CA coordinates; thick: beta strands, faint: full CA trace; target beta PCA camera")
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{case['case_id']}.{extension}", dpi=180, facecolor="white")
        plt.close(fig)
    observations = _observations(case, detail, qg, tg)
    evidence = {**case, "camera": camera, "observations": observations,
                "observation_type": "automatic coordinate/graph-derived descriptions",
                "visual_review_status": "pending_computational_image_inspection",
                "human_review_status": "not_performed"}
    _write_json(out / f"{case['case_id']}.json", evidence)
    return evidence


def render_fixed_graph_series(case_id, representatives, prepared, graphs, output_dir):
    """Show frozen quantile representatives with independent labeled cameras."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(representatives, key=lambda r: (_number(r["target_quantile"]), r["record_id"]))
    views, span, measurements = [], 1.0, []
    for row in rows:
        record_id = row["record_id"]
        residues = prepared[record_id]["ca_identities"]
        center, basis = _camera(residues, graphs[record_id])
        points = (np.asarray([r["coordinate"] for r in residues]) - center) @ basis.T
        span = max(span, float(np.max(points.max(axis=0)-points.min(axis=0)))*1.12)
        views.append((row, residues, points, center, basis))
        beta, nodes, sheets = _beta_annotations(residues, graphs[record_id])
        beta_points = points[sorted(beta)]
        sheet_centers = {key: points[sorted(indices)].mean(axis=0) for key, indices in sheets.items()}
        separations = [{"sheet_1": left, "sheet_2": right,
                        "centroid_separation_angstrom": float(np.linalg.norm(sheet_centers[left]-sheet_centers[right]))}
                       for i, left in enumerate(sorted(sheet_centers)) for right in sorted(sheet_centers)[i+1:]]
        measurements.append({"record_id": record_id, "ca_count": len(residues), "beta_ca_count": len(beta),
                             "strand_count": len(nodes), "sheet_count": len(sheets),
                             "beta_axis_1_extent_angstrom": float(np.ptp(beta_points[:,0])),
                             "beta_radius_of_gyration_angstrom": float(np.sqrt(np.mean(np.sum(beta_points**2,axis=1)))),
                             "sheet_centroid_separations": separations,
                             "pairing_edges": graphs[record_id].get("pairing_edges", [])})
    with plt.rc_context({"font.size": 7, "axes.titlesize": 9, "figure.titlesize": 12, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(3, len(rows), figsize=(3.8*len(rows), 11), layout="constrained", squeeze=False)
        for col, (row, residues, points, _center, _basis) in enumerate(views):
            graph = graphs[row["record_id"]]
            _, nodes, _ = _beta_annotations(residues, graph)
            midpoint = (points.max(axis=0)+points.min(axis=0))/2
            for ax, plane in zip(axes[:2, col], ((0,1),(0,2)), strict=True):
                for segment in trace_segments(residues):
                    ax.plot(points[segment,plane[0]], points[segment,plane[1]], color="#777777", alpha=.3, lw=.6)
                for members in nodes.values():
                    for segment in trace_segments(residues, members):
                        ax.plot(points[segment,plane[0]], points[segment,plane[1]], color="#0072B2", lw=1.8)
                ax.set(xlim=(midpoint[plane[0]]-span/2, midpoint[plane[0]]+span/2),
                       ylim=(midpoint[plane[1]]-span/2, midpoint[plane[1]]+span/2), aspect="equal",
                       xlabel=f"Own PCA axis {plane[0]+1} (Å)", ylabel=f"Own PCA axis {plane[1]+1} (Å)")
            axes[0,col].set_title(f"{row['record_id']}\nQuantile {float(row['target_quantile']):g}; elongation {float(row['axis_value']):.3f}")
            _pairing_diagram(axes[2,col], graph)
        pattern_id = rows[0]["pattern_id"]
        fig.suptitle(f"{case_id} | Fixed K-chain pattern {pattern_id}\n"
                     "Frozen elongation-quantile representatives; each domain has its OWN PCA camera\n"
                     "Same angstrom scale throughout; these are NOT pairwise superpositions or a trajectory")
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{case_id}.{extension}", dpi=180, facecolor="white")
        plt.close(fig)
    observations = [f"The {len(rows)} frozen representatives of {pattern_id} span elongation "
                    f"{min(float(r['axis_value']) for r in rows):.3f} to {max(float(r['axis_value']) for r in rows):.3f}."]
    for measurement in measurements:
        text = (f"{measurement['record_id']}: {measurement['beta_ca_count']} beta CA atoms in "
                f"{measurement['strand_count']} strands / {measurement['sheet_count']} sheets; "
                f"beta extent along its own first PCA axis {measurement['beta_axis_1_extent_angstrom']:.2f} Å; "
                f"beta radius of gyration {measurement['beta_radius_of_gyration_angstrom']:.2f} Å.")
        for separation in sorted(measurement["sheet_centroid_separations"],
                                 key=lambda r: -r["centroid_separation_angstrom"])[:2]:
            text += (f" Sheet {separation['sheet_1']}–{separation['sheet_2']} centroid separation "
                     f"{separation['centroid_separation_angstrom']:.2f} Å.")
        observations.append(text)
    observations.extend(["The plotted CA coordinates show actual physical sizes at one angstrom scale; domain orientations are independent.",
                         "Representative points illustrate the selected pattern's measured range; they do not replace its full member distribution or establish a trajectory."])
    evidence = {"case_id": case_id, "case_kind": "fixed_graph_morphology_series", "pattern_id": pattern_id,
                "representatives": rows, "common_span_angstrom": span,
                "coordinate_measurements": measurements,
                "cameras": [{"record_id": row["record_id"], "origin": center.tolist(), "basis_rows": basis.tolist()}
                            for row, _, _, center, basis in views],
                "observations": observations,
                "observation_type": "automatic coordinate/graph-derived descriptions",
                "visual_review_status": "pending_computational_image_inspection", "human_review_status": "not_performed"}
    _write_json(out / f"{case_id}.json", evidence)
    return evidence


def render_case_package(alignment_rows, prepared, graphs, output_dir, *, frozen_protocol,
                        representatives=None, pair_metadata=None):
    """Write bounded illustrations and explicit automated observations."""
    if json.loads(_read_text(Path(frozen_protocol))) != CASE_PROTOCOL:
        raise ValueError("Case protocol differs from the frozen file")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    selected, ledger = select_cases(alignment_rows)
    metadata_by_id, metadata_by_pair = {}, {}
    for row in _records(pair_metadata) if pair_metadata is not None else []:
        if row["pair_id"] in metadata_by_id:
            raise ValueError(f"Duplicate metadata pair ID: {row['pair_id']}")
        metadata_by_id[row["pair_id"]] = row
        metadata_by_pair.setdefault((row["query_id"], row["target_id"]), []).append(row)
    selected_manifest, evidence, failures = [], [], []
    for case in selected:
        row = case["alignment"]
        annotation = metadata_by_id.get(row["pair_id"], {})
        endpoint_matches = metadata_by_pair.get((row["query_id"], row["target_id"]), [])
        if not annotation and len(endpoint_matches) == 1:
            annotation = endpoint_matches[0]
        if annotation and (annotation["query_id"], annotation["target_id"]) != (row["query_id"], row["target_id"]):
            raise ValueError("Metadata pair ID has different domain endpoints")
        case["pair_metadata"] = annotation
        selected_manifest.append({"case_id": case["case_id"], "case_kind": case["case_kind"],
                                  "query_id": row["query_id"], "target_id": row["target_id"],
                                  "mode": row["mode"], "pair_id": annotation.get("pair_id", row["pair_id"]),
                                  "source_alignment_pair_id": row["pair_id"],
                                  "packing_distance": annotation.get("packing_distance"),
                                  "alignment_directory": row.get("alignment_directory"),
                                  "packing_distance_legacy": annotation.get("packing_distance_legacy")})
        try:
            evidence.append(render_pair_case(case, prepared, graphs, out))
        except (ValueError, KeyError, OSError) as error:
            failures.append({"case_id": case["case_id"], "reason": f"{type(error).__name__}: {error}"})
    groups = {}
    for row in _records(representatives) if representatives is not None else []:
        groups.setdefault(row["pattern_id"], []).append(row)
    if len(groups) > CASE_PROTOCOL["maximum_fixed_graph_series"]:
        raise ValueError("Frozen series count exceeds case protocol")
    for pattern_id, members in sorted(groups.items()):
        case_id = f"C{len(selected_manifest)+1:02d}"
        selected_manifest.append({"case_id": case_id, "case_kind": "fixed_graph_morphology_series",
                                  "pattern_id": pattern_id, "record_ids": ";".join(r["record_id"] for r in members)})
        try:
            evidence.append(render_fixed_graph_series(case_id, members, prepared, graphs, out))
        except (ValueError, KeyError, OSError) as error:
            failures.append({"case_id": case_id, "reason": f"{type(error).__name__}: {error}"})
    _write_json(out / "selection-protocol.json", CASE_PROTOCOL)
    _write_csv(out / "case-selection.csv", selected_manifest,
               list(dict.fromkeys(key for row in selected_manifest for key in row)) or ["case_id", "case_kind"])
    _write_csv(out / "case-eligibility.csv", ledger,
               list(dict.fromkeys(key for row in ledger for key in row)) or ["pair_id", "case_kind", "eligible"])
    _write_json(out / "case-evidence.json", evidence)
    summary = {"case_count": len(selected_manifest), "rendered_cases": len(evidence), "failures": failures,
               "selected_per_kind": dict(Counter(row["case_kind"] for row in selected_manifest)),
               "eligible_per_kind": dict(Counter(row["case_kind"] for row in ledger if row["eligible"])),
               "review_status": "computational measurement descriptions; visual inspection pending; no human adjudication",
               "additional_structure_alignments": 0}
    _write_json(out / "summary.json", summary)
    lines = ["# Coordinate-based computational case observations", "",
             "These cases were selected by the saved fixed rule to illustrate measured phenomena. They do not estimate occurrence rates. "
             "Descriptions below are computed from actual coordinates and retained graphs; independent expert review and author review were not performed.", "",
             "Pair views use one saved native rigid transformation and two orthographic projections at equal physical scale. "
             "Fixed-graph series use each domain's own beta PCA camera at a shared angstrom scale; these are not superpositions or trajectories.", "",
             f"Selected: {len(selected_manifest)}; rendered: {len(evidence)}; failed: {len(failures)}. "
             "Unsupported categories remain empty; selection thresholds were not relaxed.", ""]
    for case in evidence:
        lines.extend([f"## {case['case_id']}: {case['case_kind'].replace('_', ' ')}", "",
                      f"![Coordinate illustration]({case['case_id']}.png)", ""])
        lines.extend(f"- {observation}" for observation in case["observations"])
        lines.append("")
    (out / "COMPUTATIONAL_OBSERVATIONS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol")
    parser.add_argument("--alignment-table", action="append", default=[])
    parser.add_argument("--coordinates", action="append", default=[])
    parser.add_argument("--graphs")
    parser.add_argument("--protocol")
    parser.add_argument("--representatives")
    parser.add_argument("--pair-metadata")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.freeze_protocol:
        freeze_case_protocol(args.freeze_protocol)
        return
    if not args.output or not args.protocol or not args.graphs or not args.coordinates:
        parser.error("Rendering requires --output, --protocol, --graphs and --coordinates")
    prepared = load_prepared_coordinates(args.coordinates)
    graphs = load_graphs(args.graphs, prepared)
    result = render_case_package(read_alignment_tables(args.alignment_table), prepared, graphs, args.output,
                                 frozen_protocol=args.protocol, representatives=args.representatives,
                                 pair_metadata=args.pair_metadata)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
