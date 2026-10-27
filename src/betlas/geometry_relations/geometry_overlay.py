"""Attach reference-label descriptions to already frozen packing graphs.

Distances and short mutual neighborhoods precede this overlay. Label overlap
and proximity to reference architecture 2.40 are label-conditioned descriptions,
not independent geometric measurements. Each phase uses its own graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, write_json


def _boolean(values: pd.Series) -> pd.Series:
    parsed = values.astype("string").str.strip().str.lower()
    absent = parsed.isna() | parsed.isin(["", "na", "nan", "none", "<na>"])
    known = parsed.isin(["true", "false", "1", "0", "1.0", "0.0"])
    if (~(absent | known)).any():
        raise ValueError("Graph boolean must be true, false, or NA")
    result = parsed.isin(["true", "1", "1.0"]).astype("boolean")
    return result.mask(absent)


def _unique(table: pd.DataFrame, name: str) -> None:
    if "record_id" not in table or table.record_id.isna().any() or table.record_id.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} requires record_id")
    if table.record_id.duplicated().any():
        raise ValueError(f"{name} contains duplicate record_id")


def _phase_overlay(cohort: pd.DataFrame, domains: pd.DataFrame, edges: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Reconstruct the frozen short mutual adjacency and check its denominator."""
    _unique(domains, phase)
    identity = cohort.set_index("record_id")
    nodes = domains.set_index("record_id").copy()
    if not nodes.index.isin(identity.index).all():
        raise ValueError(f"{phase} graph contains records outside the cohort")
    expected_ids = identity.index if phase == "all" else identity.index[identity.role.eq(phase)]
    if set(nodes.index) != set(expected_ids):
        raise ValueError(f"{phase} graph membership does not match the frozen cohort role")
    if not nodes.label.eq(identity.loc[nodes.index, A]).all() or not nodes.homology.eq(identity.loc[nodes.index, H]).all():
        raise ValueError(f"{phase} graph labels do not match cohort identities")
    eligible = _boolean(nodes.packing_eligible)
    if eligible.isna().any():
        raise ValueError("Packing availability must be known for graph nodes")
    selected = edges.loc[(_boolean(edges.mutual) & _boolean(edges.short_edge)).fillna(False)]
    neighbors: dict[str, set[str]] = {record_id: set() for record_id in nodes.index}
    for edge in selected.itertuples(index=False):
        source, target = str(edge.source_id), str(edge.target_id)
        if source not in neighbors or target not in neighbors or source == target:
            raise ValueError("Selected edge has missing or identical endpoints")
        if not eligible.at[source] or not eligible.at[target]:
            continue
        distance = float(edge.distance)
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Selected edge distance must be finite and nonnegative")
        neighbors[source].add(target)
        neighbors[target].add(source)
    n_neighbors = pd.to_numeric(nodes.n_neighbors, errors="coerce")
    overlap = _boolean(nodes.label_overlap)
    fraction = pd.to_numeric(nodes.cross_label_fraction, errors="coerce")
    near = pd.Series(pd.NA, index=nodes.index, dtype="boolean")
    for record_id, adjacent in neighbors.items():
        expected_count = len(adjacent) if eligible.at[record_id] else np.nan
        recorded_count = n_neighbors.at[record_id]
        if not ((pd.isna(expected_count) and pd.isna(recorded_count)) or expected_count == recorded_count):
            raise ValueError(f"{phase} adjacency denominator differs for {record_id}")
        if not adjacent:
            if pd.notna(overlap.at[record_id]) or pd.notna(fraction.at[record_id]):
                raise ValueError("Empty or unavailable neighborhoods must retain NA label overlap")
            continue
        labels = identity.loc[sorted(adjacent), A]
        expected_fraction = float(labels.ne(identity.at[record_id, A]).mean())
        if pd.isna(overlap.at[record_id]) or bool(overlap.at[record_id]) != (expected_fraction > 0):
            raise ValueError("Recorded label overlap differs from the selected adjacency")
        if not np.isclose(fraction.at[record_id], expected_fraction):
            raise ValueError("Recorded cross-label fraction differs from the selected adjacency")
        near.at[record_id] = labels.eq("2.40").any()
    result = pd.DataFrame({
        "packing_eligible": eligible,
        "neighborhood_status": nodes.neighborhood_status,
        "n_neighbors": n_neighbors,
        "label_overlap": overlap,
        "label_overlap_fraction": fraction,
        "near_barrel": near,
    })
    for column in ("n_neighbor_h", "n_neighbor_labels", "neighbor_distance_median", "neighbor_distance_min", "neighbor_distance_max", "target_h_reweighted_cross_label_fraction"):
        if column in nodes:
            result[column] = pd.to_numeric(nodes[column], errors="coerce")
    return result.reset_index()


def run_geometry_overlay(cohort_csv: str | Path | pd.DataFrame, atlas_root: str | Path, output_csv: str | Path) -> dict[str, Any]:
    """Join all/phase RMS k30 reference-label overlays without changing graphs."""
    cohort = cohort_csv.copy() if isinstance(cohort_csv, pd.DataFrame) else read_csv(cohort_csv, dtype=str, keep_default_na=False)
    _unique(cohort, "cohort")
    if missing := {A, T, H, "role"} - set(cohort):
        raise ValueError(f"Cohort lacks identity/role fields: {sorted(missing)}")
    metadata = [column for column in (
        "record_id", "pdb_id", A, T, H, "role", "strict_label", "legacy_label", "feature_status",
        "engineering_record", "engineering_h", "mother_h_pdb_component", "mother_s35_domain_count",
    ) if column in cohort]
    output = cohort[metadata].copy()
    phase_summary = {}
    for phase in ("all", "discovery", "confirmation"):
        directory = Path(atlas_root) / phase / "rms"
        domain_path = directory / "relations-k30" / "architecture-per_domain.csv"
        edge_path = directory / "edges-k30.csv.gz"
        domains = read_csv(domain_path, dtype=str, keep_default_na=False)
        edges = read_csv(edge_path, dtype=str, keep_default_na=False)
        overlay = _phase_overlay(cohort, domains, edges, phase)
        phase_summary[phase] = {
            "domains": len(overlay), "packing_eligible": int(overlay.packing_eligible.sum()),
            "measured_neighborhoods": int(overlay.near_barrel.notna().sum()),
            "near_reference_barrel": int(overlay.near_barrel.sum()),
            "domain_input": str(domain_path), "edge_input": str(edge_path),
        }
        if phase != "all":
            overlay = overlay[["record_id", "near_barrel", "n_neighbors", "neighborhood_status"]].rename(
                columns={column: f"{column}_{phase}" for column in ("near_barrel", "n_neighbors", "neighborhood_status")}
            )
        output = output.merge(overlay, on="record_id", how="left", validate="one_to_one")
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    protocol = {
        "schema": "betlas.geometry-overlay.v1", "status": "complete", "domains": len(output),
        "primary_graph": "Previously frozen label-free packing RMS distance, k30, short mutual edges, with upstream H/PDB exclusions.",
        "label_overlap": "All-graph reference-A overlap and cross-A neighbor fraction; these are label-conditioned overlays.",
        "near_barrel": "At least one short mutual neighbor has reference A=2.40; defined only for nonempty neighborhoods. This is not a barrel prediction.",
        "phase_isolation": "Discovery and confirmation use their own phase graphs. Outside-phase and empty/unavailable neighborhoods remain NA; all-graph relations never fill phase missingness.",
        "phases": phase_summary,
    }
    write_json(destination.with_suffix(".json"), protocol)
    return protocol
