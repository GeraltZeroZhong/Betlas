"""Coordinate measurements for fixed pairing/order patterns.

Lengths are Angstroms and angles are degrees. Plane residual is not curvature;
CA pseudodihedral is a local backbone descriptor, not beta-sheet twist. All
measurements use observed coordinates and preserve their applicability status.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .connection_patterns import _pattern_distributions, _payload, load_graph_records
from .io import read_csv, read_json, write_json

PHYSICAL_COLUMNS = (
    "beta_rg_angstrom", "sheet_plane_rms_mean_angstrom", "sheet_plane_rms_max_angstrom",
    "largest_two_sheet_centroid_distance_angstrom", "largest_two_sheet_normal_angle_degrees",
    "largest_two_sheet_centroid_normal_separation_angstrom", "strand_axis_bend_mean_degrees",
    "strand_ca_pseudodihedral_mean_absolute_degrees",
)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    magnitude = float(np.linalg.norm(vector))
    return vector / magnitude if magnitude > 1e-12 else None


def _directed_axis(points: np.ndarray) -> np.ndarray | None:
    centered = points - points.mean(axis=0)
    if len(points) < 2 or np.linalg.norm(centered) <= 1e-12:
        return None
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    if len(singular) >= 2 and singular[0] - singular[1] <= 1e-8 * max(singular[0], 1):
        return None
    axis = right[0]
    dot = float(np.dot(axis, points[-1] - points[0]))
    if abs(dot) <= 1e-12:
        return None
    return axis if dot > 0 else -axis


def _angle(first: np.ndarray, second: np.ndarray, *, unoriented: bool = False) -> float:
    cosine = float(np.dot(first, second))
    return float(np.degrees(np.arccos(np.clip(abs(cosine) if unoriented else cosine, -1, 1))))


def _pseudodihedral(points: np.ndarray) -> float | None:
    """Signed CA virtual torsion for four consecutive N-to-C CA positions."""
    b0, b1, b2 = points[0] - points[1], points[2] - points[1], points[3] - points[2]
    axis = _unit(b1)
    if axis is None:
        return None
    first, second = b0 - np.dot(b0, axis) * axis, b2 - np.dot(b2, axis) * axis
    if min(np.linalg.norm(first), np.linalg.norm(second)) <= 1e-12:
        return None
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, first), second), np.dot(first, second))))


def measure_pattern_geometry(
    raw_graph: Mapping[str, Any], ca_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure one domain from its exact retained beta nodes and CA identities."""
    graph = _payload(raw_graph)
    nodes = list(graph.get("nodes", []))
    coordinates = {}
    for residue in ca_identities:
        uid = str(residue["residue_uid"])
        if uid in coordinates:
            raise ValueError(f"Duplicate residue identity: {uid}")
        point = np.asarray(residue["coordinate"], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("CA coordinates must be finite three-vectors")
        coordinates[uid] = point
    beta_ids = {str(uid) for node in nodes for uid in node.get("residue_uids", [])}
    missing = beta_ids - set(coordinates)
    domain: dict[str, Any] = {column: None for column in PHYSICAL_COLUMNS}
    domain.update(beta_residues_expected=len(beta_ids), beta_residues_missing=len(missing),
                  coordinate_status="no_beta_nodes" if not beta_ids else "partial_beta_coordinates" if missing else "complete_beta_coordinates",
                  n_strands=len(nodes), n_strands_bend_measured=0, n_strands_dihedral_measured=0,
                  n_sheets_plane_measured=0, n_sheet_pairs_measured=0)
    if missing or not beta_ids:
        return {"domain": domain, "strands": [], "sheets": [], "sheet_pairs": []}
    all_points = np.array([coordinates[uid] for uid in sorted(beta_ids)])
    centered = all_points - all_points.mean(axis=0)
    domain["beta_rg_angstrom"] = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    strands, grouped_sheets = [], defaultdict(list)
    for node in nodes:
        points = np.array([coordinates[str(uid)] for uid in node["residue_uids"]])
        positions = node.get("label_positions")
        continuous = (positions is not None and len(positions) == len(points)
                      and all(int(b) - int(a) == 1 for a, b in zip(positions, positions[1:], strict=False)))
        bend = None
        if continuous and len(points) >= 6:
            smoothed = (points[:-1] + points[1:]) / 2
            first, last = _directed_axis(smoothed[:3]), _directed_axis(smoothed[-3:])
            if first is not None and last is not None:
                bend = _angle(first, last)
        torsions = []
        if continuous:
            for start in range(len(points) - 3):
                torsion = _pseudodihedral(points[start:start + 4])
                if torsion is not None:
                    torsions.append(torsion)
        strands.append({"node_id": node["node_id"], "sheet_id": node.get("sheet_id", ""),
                        "residue_count": len(points), "polymer_positions_contiguous": continuous,
                        "axis_bend_degrees": bend, "ca_pseudodihedral_count": len(torsions),
                        "ca_pseudodihedral_mean_absolute_degrees": float(np.mean(np.abs(torsions))) if torsions else None,
                        "ca_pseudodihedral_values_degrees": torsions,
                        "bend_status": "measured" if bend is not None else "requires_contiguous_6_CA_and_defined_axes"})
        if str(node.get("sheet_id", "")):
            grouped_sheets[str(node["sheet_id"])].append(node)
    sheets, plane_lookup = [], {}
    for sheet_id, members in sorted(grouped_sheets.items()):
        identities = {str(uid) for node in members for uid in node["residue_uids"]}
        points = np.array([coordinates[uid] for uid in sorted(identities)])
        centroid = points.mean(axis=0)
        row = {"sheet_id": sheet_id, "n_strands": len(members), "n_residues": len(points),
               "plane_status": "requires_two_strands_and_nondegenerate_plane", "plane_rms_angstrom": None,
               "centroid": centroid.tolist(), "plane_normal": None}
        if len(members) >= 2 and len(points) >= 3:
            delta = points - centroid
            _u, singular, vectors = np.linalg.svd(delta, full_matrices=False)
            # A plane is undefined for collinear points or tied second/third axes.
            if len(singular) == 3 and singular[1] > 1e-8 * max(singular[0], 1) and singular[1] - singular[2] > 1e-8 * max(singular[0], 1):
                normal = vectors[-1]
                rms = float(np.sqrt(np.mean((delta @ normal) ** 2)))
                row.update(plane_status="measured", plane_rms_angstrom=rms, plane_normal=normal.tolist())
                plane_lookup[sheet_id] = (centroid, normal)
        sheets.append(row)
    sheet_pairs = []
    for left, right in combinations(sheets, 2):
        if left["sheet_id"] not in plane_lookup or right["sheet_id"] not in plane_lookup:
            continue
        c1, n1 = plane_lookup[left["sheet_id"]]
        c2, n2 = plane_lookup[right["sheet_id"]]
        difference = c2 - c1
        sheet_pairs.append({"left_sheet": left["sheet_id"], "right_sheet": right["sheet_id"],
                            "centroid_distance_angstrom": float(np.linalg.norm(difference)),
                            "normal_angle_degrees": _angle(n1, n2, unoriented=True),
                            "centroid_normal_separation_angstrom": float((abs(difference @ n1) + abs(difference @ n2)) / 2)})
    measured_planes = [sheet["plane_rms_angstrom"] for sheet in sheets if sheet["plane_status"] == "measured"]
    bends = [strand["axis_bend_degrees"] for strand in strands if strand["axis_bend_degrees"] is not None]
    torsion_means = [strand["ca_pseudodihedral_mean_absolute_degrees"] for strand in strands if strand["ca_pseudodihedral_count"]]
    domain.update(sheet_plane_rms_mean_angstrom=float(np.mean(measured_planes)) if measured_planes else None,
                  sheet_plane_rms_max_angstrom=max(measured_planes) if measured_planes else None,
                  strand_axis_bend_mean_degrees=float(np.mean(bends)) if bends else None,
                  strand_ca_pseudodihedral_mean_absolute_degrees=float(np.mean(torsion_means)) if torsion_means else None,
                  n_strands_bend_measured=len(bends), n_strands_dihedral_measured=len(torsion_means),
                  n_sheets_plane_measured=len(measured_planes), n_sheet_pairs_measured=len(sheet_pairs))
    ranked = sorted(sheets, key=lambda sheet: (-sheet["n_strands"], -sheet["n_residues"], sheet["sheet_id"]))
    if len(ranked) >= 2:
        wanted = {ranked[0]["sheet_id"], ranked[1]["sheet_id"]}
        for row in sheet_pairs:
            if {row["left_sheet"], row["right_sheet"]} == wanted:
                for key in ("centroid_distance_angstrom", "normal_angle_degrees", "centroid_normal_separation_angstrom"):
                    domain[f"largest_two_sheet_{key}"] = row[key]
                break
    return {"domain": domain, "strands": strands, "sheets": sheets, "sheet_pairs": sheet_pairs}


def run_pattern_geometry(
    graphs_jsonl_gz: str | Path, coordinate_source: str | Path | Mapping[str, Any], output_dir: str | Path,
) -> dict[str, Any]:
    """Measure the previously selected full pattern population or fixed sample."""
    destination = Path(output_dir)
    if (destination / "pattern-physical-domains.csv").exists():
        raise FileExistsError("Physical pattern results already exist")
    requests = read_csv(destination / "coordinate-requests.csv", dtype={A: str, T: str, H: str})
    if isinstance(coordinate_source, Mapping):
        prepared = coordinate_source
    else:
        path = Path(coordinate_source)
        prepared = read_json(path / "coordinate-manifest.json" if path.is_dir() else path)
    raw = load_graph_records(graphs_jsonl_gz)
    domains, strands, sheets, pairs = [], [], [], []
    for request in requests.to_dict("records"):
        record_id = str(request["record_id"])
        inspection = prepared.get(record_id)
        identity = {"record_id": record_id, "chain_pattern": request["chain_pattern"], H: request[H]}
        try:
            if not inspection or not inspection.get("ca_identities"):
                raise ValueError("Prepared CA identity map is unavailable")
            measured = measure_pattern_geometry(raw[record_id], inspection["ca_identities"])
            domains.append(request | measured["domain"] | {"measurement_error": ""})
            strands.extend(identity | row for row in measured["strands"])
            sheets.extend(identity | row for row in measured["sheets"])
            pairs.extend(identity | row for row in measured["sheet_pairs"])
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
            domains.append(request | {column: None for column in PHYSICAL_COLUMNS}
                           | {"coordinate_status": "measurement_unavailable", "measurement_error": f"{type(error).__name__}: {error}"})
    domain_table = pd.DataFrame(domains)
    summary, by_h = _pattern_distributions(domain_table, PHYSICAL_COLUMNS)
    artifacts = {"pattern-physical-domains.csv": domain_table, "pattern-physical-distributions.csv": summary,
                 "pattern-physical-per-H.csv": by_h, "pattern-strands.csv": pd.DataFrame(strands),
                 "pattern-sheets.csv": pd.DataFrame(sheets), "pattern-sheet-pairs.csv": pd.DataFrame(pairs)}
    for name, table in artifacts.items():
        table.to_csv(destination / name, index=False)
    protocol = {
        "schema": "betlas.pattern-geometry.v1", "status": "complete", "requested_domains": len(requests),
        "coordinate_status": domain_table.coordinate_status.value_counts().to_dict(),
        "population": "The frozen coordinate-requests table; not the five-per-pattern display series. Full population unless catalogue specified H-balanced sampling.",
        "beta_Rg": "Root mean squared CA distance from the centroid, using each retained beta residue exactly once, Angstroms.",
        "sheet_planes": "All annotation sheet groups with >=2 strands, >=3 CA and a nondegenerate least-squares plane; RMS perpendicular residual, not curvature.",
        "sheet_pairs": "All valid sheet-plane pairs. Unoriented normal angle in [0,90] degrees; centroid distance; mean absolute projection of centroid displacement onto each normal, Angstroms.",
        "largest_two_sheets": "Rank by retained strand count, then residue count, then sheet ID. Only report if both actual largest sheets have valid planes; all pairs remain available separately.",
        "strand_bend": "For >=6 contiguous CA, average adjacent CA pairs to suppress alternation, then fit N-to-C PCA axes to the first and last three averaged positions. Report the angle [0,180] degrees, not curvature.",
        "CA_pseudodihedral": "Virtual torsions of four consecutive CA within each contiguous retained strand; retain signed raw angles and summarize mean absolute angle per strand, then equal strand mean. Not beta-sheet twist.",
        "missing": "Any missing retained beta CA makes domain physical values unavailable; insufficient or degenerate local support stays NA. Observed zero remains zero.",
        "interpretation": "Coordinate summaries at fixed graph resolution; no interpolation, dynamics, causal or full-topology inference.",
        "artifacts": {name: len(table) for name, table in artifacts.items()},
    }
    write_json(destination / "physical-protocol.json", protocol)
    return protocol
