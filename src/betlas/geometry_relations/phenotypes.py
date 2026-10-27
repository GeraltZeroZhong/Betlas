"""Descriptive, hierarchy-aware packing and pairing phenotypes.

Reference labels define reporting groups, not measurements. Complete pairing
graphs, retained partial graphs, and unavailable annotations are kept separate.
These summaries contain no tests, confidence intervals, causal estimates, or
newly fitted classifiers. A graph cycle does not establish a geometric barrel.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, write_json
from .packing import PACKING_BLOCKS, prepare_packing_features

P_COLUMNS = tuple(column for block in PACKING_BLOCKS.values() for column in block)
TWO_SHEET_COLUMNS = (
    "betlas_sheet_pair_centroid_distance_norm",
    "betlas_sheet_pair_valid_normal_abs_dot",
    "betlas_sheet_pair_valid_face_alignment",
    "betlas_sheet_pair_angular_separation_fraction",
    "betlas_sheet_pair_size_balance",
)
K_COLUMNS = tuple(f"pairing_{field}" for field in (
    "node_count", "edge_count", "component_count", "largest_component_fraction",
    "degree_mean", "degree_max", "branch_fraction", "degree_two_fraction", "isolated_fraction",
    "cycle_rank", "cycle_rank_per_node", "simple_cycle_component_node_fraction",
    "known_sense_fraction", "parallel_fraction_known", "antiparallel_fraction_known",
    "cross_chain_fraction", "cross_fragment_fraction", "sequence_span_available_fraction",
    "strand_rank_gap_mean", "strand_rank_gap_max", "nonlocal_fraction_same_chain",
    "observed_path_fraction_same_chain",
))
CLOSURE_COLUMNS = (
    "betlas_axis_best_angular_coverage", "betlas_axis_best_slice_coverage_median",
    "betlas_axis_best_slice_coverage_std", "betlas_z_continuity_fraction", "betlas_axis_best_radius_cv",
)
CLOSURE_K_COLUMNS = tuple(f"pairing_{field}" for field in (
    "cycle_rank", "cycle_rank_per_node", "simple_cycle_component_node_fraction",
    "branch_fraction", "largest_component_fraction", "degree_two_fraction", "isolated_fraction",
))


def _read(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    return source.copy() if isinstance(source, pd.DataFrame) else read_csv(source, dtype=str, keep_default_na=False)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _boolean(series: pd.Series) -> pd.Series:
    def value(item: Any) -> Any:
        if item is None or pd.isna(item) or str(item).strip().lower() in {"", "na", "nan", "none", "<na>"}:
            return pd.NA
        if str(item).strip().lower() in {"true", "1", "1.0"}:
            return True
        if str(item).strip().lower() in {"false", "0", "0.0"}:
            return False
        raise ValueError("near_barrel must be boolean or NA, not a continuous distance or fraction")
    return pd.Series(pd.array(series.map(value), dtype="boolean"), index=series.index)


def _unique(frame: pd.DataFrame, name: str) -> None:
    if "record_id" not in frame or frame.record_id.isna().any() or frame.record_id.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} requires nonmissing record_id")
    frame["record_id"] = frame.record_id.astype(str)
    if frame.record_id.duplicated().any():
        raise ValueError(f"{name} contains duplicate record_id")


def _prepare(
    cohort: pd.DataFrame, connectivity: pd.DataFrame, geometry: pd.DataFrame,
) -> pd.DataFrame:
    for frame, name in ((cohort, "cohort"), (connectivity, "connectivity"), (geometry, "geometry")):
        _unique(frame, name)
    if missing := {A, T, H} - set(cohort):
        raise ValueError(f"Cohort lacks CATH identity columns: {sorted(missing)}")
    for column in (A, T, H):
        if cohort[column].isna().any() or cohort[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Cohort has missing {column}")
        cohort[column] = cohort[column].astype(str)
    if missing := {"status", "pairing_status"} - set(connectivity):
        raise ValueError(f"Connectivity table lacks status columns: {sorted(missing)}")
    metadata = [column for column in (
        "record_id", "pdb_id", A, T, H, "role", "feature_status", "engineering_record",
        "engineering_h", "mother_h_pdb_component", "mother_s35_domain_count",
    ) if column in cohort]
    data = cohort[metadata].copy()
    if "role" not in data:
        data["role"] = "unspecified"
    data["phenotype_group"] = np.select(
        [data[A].eq("2.40"), data[T].eq("2.60.120"), data[A].eq("2.60")],
        ["B", "J", "S_other"], default="other",
    )
    data["within_260_group"] = np.where(data[T].eq("2.60.120"), "J", data[T])
    physical = prepare_packing_features(cohort)
    for column in P_COLUMNS:
        data[column] = physical[column]
    parse_ok = _numeric(cohort, "betlas_parse_ok").eq(1)
    if "feature_status" in cohort:
        parse_ok &= cohort.feature_status.eq("ok")
    data.loc[~parse_ok, list(P_COLUMNS)] = np.nan
    two_sheet = parse_ok & _numeric(cohort, "betlas_sheet_count").ge(2) & _numeric(cohort, "betlas_sheet_face_count").ge(2)
    data["two_sheet_applicable"] = two_sheet
    for column in TWO_SHEET_COLUMNS:
        data[column] = _numeric(cohort, column).where(two_sheet)

    graph_columns = [column for column in ("record_id", "status", "pairing_status", "warnings", *K_COLUMNS) if column in connectivity]
    graph = connectivity[graph_columns].rename(columns={"status": "connectivity_status", "warnings": "connectivity_warnings"})
    data = data.merge(graph, on="record_id", how="left", validate="one_to_one")
    data["connectivity_status"] = data.connectivity_status.fillna("not_extracted").replace("", "not_extracted")
    data["pairing_status"] = data.pairing_status.fillna("unavailable").replace("", "unavailable")
    for column in K_COLUMNS:
        data[column] = _numeric(data, column)
    extracted = data.connectivity_status.eq("ok") & data.pairing_node_count.gt(0)
    data["pairing_complete"] = extracted & data.pairing_status.isin(["available", "no_retained_pairings"])
    data["pairing_partial"] = extracted & data.pairing_status.eq("partial_clipped_pairings")
    data["pairing_unavailable"] = ~(data.pairing_complete | data.pairing_partial)

    geometric_columns = [column for column in (
        "record_id", "near_barrel", "near_barrel_discovery", "near_barrel_confirmation",
        "label_overlap", "label_overlap_fraction", "topology_overlap",
        "topology_overlap_fraction", "packing_eligible",
    ) if column in geometry]
    data = data.merge(geometry[geometric_columns], on="record_id", how="left", validate="one_to_one")
    for column in ("near_barrel", "near_barrel_discovery", "near_barrel_confirmation"):
        if column not in data:
            data[column] = pd.NA
        data[column] = _boolean(data[column])

    # Fixed descriptive bins of observed domain quantities, not adjustment models.
    data["observed_length"] = _numeric(cohort, "betlas_residue_count").where(parse_ok).to_numpy()
    data["beta_fraction"] = _numeric(cohort, "betlas_beta_residue_fraction").where(parse_ok).to_numpy()
    data["strand_count"] = _numeric(cohort, "betlas_beta_strand_count").where(parse_ok).to_numpy()
    data["length_bin"] = np.select(
        [data.observed_length.gt(0) & data.observed_length.le(100),
         data.observed_length.gt(100) & data.observed_length.le(200), data.observed_length.gt(200)],
        ["le100", "101_200", "gt200"], default="unavailable",
    )
    data["beta_fraction_bin"] = np.select(
        [data.beta_fraction.between(0, 0.5), data.beta_fraction.gt(0.5) & data.beta_fraction.le(1)],
        ["le0.5", "gt0.5"], default="unavailable",
    )
    data["strand_count_bin"] = np.select(
        [data.strand_count.gt(0) & data.strand_count.le(8), data.strand_count.gt(8)],
        ["le8", "gt8"], default="unavailable",
    )
    return data


def _scopes(data: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    yield "all", data
    for role, part in data.groupby("role", dropna=False, sort=True):
        if role in {"discovery", "confirmation"}:
            part = part.copy()
            part["near_barrel"] = part[f"near_barrel_{role}"]
        yield f"role:{role}", part


def _measurements(data: pd.DataFrame, *, closure_only: bool) -> Iterator[tuple[str, str, pd.Series]]:
    for column in CLOSURE_COLUMNS if closure_only else P_COLUMNS:
        yield "P_applicable", column, data[column]
    for column in TWO_SHEET_COLUMNS:
        yield "two_sheet_applicable", column, data[column]
    for column in CLOSURE_K_COLUMNS if closure_only else K_COLUMNS:
        yield "K_complete", column, data[column].where(data.pairing_complete)
        yield "K_retained_sensitivity", column, data[column].where(data.pairing_complete | data.pairing_partial)
        yield "K_partial_retained_only", column, data[column].where(data.pairing_partial)


def _finite_summary(values: pd.Series, groups: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.DataFrame({H: groups, "value": pd.to_numeric(values, errors="coerce")})
    by_h = frame.groupby(H, sort=True).agg(n_total=("value", "size"), n_observed=("value", "count"), value_mean=("value", "mean"))
    by_h["n_missing"] = by_h.n_total - by_h.n_observed
    observed = frame.value.dropna()
    h_means = by_h.loc[by_h.n_observed.gt(0), "value_mean"]
    summary = {
        "n_total": len(frame), "n_observed": len(observed), "n_missing": len(frame) - len(observed),
        "H_total": len(by_h), "H_observed": len(h_means),
        "domain_median": float(observed.median()) if len(observed) else np.nan,
        "domain_mean": float(observed.mean()) if len(observed) else np.nan,
        "domain_q25": float(observed.quantile(0.25)) if len(observed) else np.nan,
        "domain_q75": float(observed.quantile(0.75)) if len(observed) else np.nan,
        "H_equal_mean": float(h_means.mean()) if len(h_means) else np.nan,
        "max_H_observed_fraction": float(by_h.n_observed.max() / len(observed)) if len(observed) else np.nan,
    }
    return summary, by_h.reset_index()


def _distributions(data: pd.DataFrame, *, question: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if question == "H1":
        selected, group_column = data.loc[data[A].eq("2.60")], "within_260_group"
    elif question == "H2":
        selected, group_column = data.loc[data.phenotype_group.isin(["B", "J", "S_other"])], "phenotype_group"
    else:
        raise ValueError("Distribution question must be H1 or H2")
    rows, h_rows = [], []
    for scope, scope_data in _scopes(selected):
        for group, part in scope_data.groupby(group_column, sort=True):
            for view, column, values in _measurements(part, closure_only=question == "H2"):
                stats, by_h = _finite_summary(values, part[H])
                identity = {"question": question, "scope": scope, "group": group, "view": view, "measurement": column}
                rows.append(identity | stats | {
                    "pairing_complete_n": int(part.pairing_complete.sum()),
                    "pairing_partial_n": int(part.pairing_partial.sum()),
                    "pairing_unavailable_n": int(part.pairing_unavailable.sum()),
                    "two_sheet_applicable_n": int(part.two_sheet_applicable.sum()),
                })
                h_rows.extend(identity | row for row in by_h.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(h_rows)


def _contrasts(distributions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if distributions.empty:
        return pd.DataFrame(columns=["scope", "group", "view", "measurement", "comparison_status"])
    keys = ["scope", "view", "measurement"]
    baseline = distributions.loc[distributions.group.eq("J")].set_index(keys)
    for row in distributions.loc[~distributions.group.eq("J")].to_dict("records"):
        key = tuple(row[column] for column in keys)
        if key not in baseline.index:
            continue
        j = baseline.loc[key]
        comparable = row["n_observed"] > 0 and j.n_observed > 0
        status = "not_comparable" if not comparable else (
            "descriptive_multiple_H_support" if min(row["H_observed"], j.H_observed) >= 2 else "descriptive_single_H_support"
        )
        rows.append({
            **{column: row[column] for column in keys}, "group": row["group"], "reference": "J",
            "comparison_status": status, "J_n_observed": j.n_observed, "other_n_observed": row["n_observed"],
            "J_H_observed": j.H_observed, "other_H_observed": row["H_observed"],
            "difference_H_equal_mean_other_minus_J": row["H_equal_mean"] - j.H_equal_mean if comparable else np.nan,
            "difference_domain_median_other_minus_J": row["domain_median"] - j.domain_median if comparable else np.nan,
        })
    return pd.DataFrame(rows)


def _near_summary(part: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    values = pd.to_numeric(part.near_barrel.astype("Float64"), errors="coerce")
    stats, by_h = _finite_summary(values, part[H])
    return {
        "n_total": stats["n_total"], "near_barrel_available_n": stats["n_observed"],
        "near_barrel_unavailable_n": stats["n_missing"], "near_barrel_n": int(values.sum()),
        "near_barrel_fraction_available": stats["domain_mean"],
        "H_equal_near_barrel_fraction_available": stats["H_equal_mean"],
        "H_total": stats["H_total"], "H_observed": stats["H_observed"],
        "max_H_observed_fraction": stats["max_H_observed_fraction"],
    }, by_h


def _near_barrel_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = data.loc[data[A].eq("2.60")]
    rows, h_rows = [], []
    for scope, scope_data in _scopes(selected):
        for group, group_data in scope_data.groupby("within_260_group", sort=True):
            subsets: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", group_data)]
            for column, levels in (
                ("length_bin", ("le100", "101_200", "gt200", "unavailable")),
                ("beta_fraction_bin", ("le0.5", "gt0.5", "unavailable")),
                ("strand_count_bin", ("le8", "gt8", "unavailable")),
            ):
                subsets.extend((column, level, group_data.loc[group_data[column].eq(level)]) for level in levels)
            for labels, part in group_data.groupby(["length_bin", "beta_fraction_bin", "strand_count_bin"], sort=True):
                subsets.append(("joint_fixed_bins", ";".join(labels), part))
            for stratifier, stratum, part in subsets:
                identity = {"scope": scope, "group": group, "stratifier": stratifier, "stratum": stratum}
                stats, by_h = _near_summary(part)
                rows.append(identity | stats)
                h_rows.extend(identity | row for row in by_h.to_dict("records"))
    summary = pd.DataFrame(rows)
    contrasts = []
    if not summary.empty:
        keys = ["scope", "stratifier", "stratum"]
        j_lookup = summary.loc[summary.group.eq("J")].set_index(keys)
        for row in summary.loc[~summary.group.eq("J")].to_dict("records"):
            key = tuple(row[column] for column in keys)
            if key not in j_lookup.index:
                continue
            j = j_lookup.loc[key]
            available = row["near_barrel_available_n"] > 0 and j.near_barrel_available_n > 0
            contrasts.append({
                **{column: row[column] for column in keys}, "group": row["group"], "reference": "J",
                "comparison_status": "descriptive" if available else "not_comparable",
                "J_available_n": j.near_barrel_available_n, "other_available_n": row["near_barrel_available_n"],
                "J_H_observed": j.H_observed, "other_H_observed": row["H_observed"],
                "difference_fraction_other_minus_J": row["near_barrel_fraction_available"] - j.near_barrel_fraction_available if available else np.nan,
                "difference_H_equal_fraction_other_minus_J": row["H_equal_near_barrel_fraction_available"] - j.H_equal_near_barrel_fraction_available if available else np.nan,
            })
    return summary, pd.DataFrame(h_rows), pd.DataFrame(contrasts)


def run_phenotype_analysis(
    cohort_csv: str | Path | pd.DataFrame,
    connectivity_csv: str | Path | pd.DataFrame,
    geometry_domain_csv: str | Path | pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write fixed H1/H2/H3 summaries, leaving graph discovery to the caller.

    Required cohort identities are record_id and the native CATH A/T/H columns;
    physical inputs are the existing Betlas feature columns. Connectivity uses
    summarize_connectivity's flattened status and pairing_* fields. The geometry
    table supplies a previously frozen boolean/NA near_barrel relation. No graph,
    cutoff, selected pair, grouping role, or model is optimized by this function.
    """
    cohort, connectivity, geometry = _read(cohort_csv), _read(connectivity_csv), _read(geometry_domain_csv)
    data = _prepare(cohort, connectivity, geometry)
    h1, h1_h = _distributions(data, question="H1")
    h2, h2_h = _distributions(data, question="H2")
    near, near_h, near_contrasts = _near_barrel_tables(data)
    coverage = data.groupby(["role", A, "phenotype_group", "connectivity_status", "pairing_status"], dropna=False).agg(
        domains=("record_id", "size"), H_groups=(H, "nunique"),
        complete_pairing_domains=("pairing_complete", "sum"), partial_pairing_domains=("pairing_partial", "sum"),
        two_sheet_applicable_domains=("two_sheet_applicable", "sum"),
    ).reset_index()
    artifacts = {
        "domain-phenotypes.csv": data, "coverage.csv": coverage,
        "H1-distributions.csv": h1, "H1-per-H.csv": h1_h, "H1-contrasts.csv": _contrasts(h1),
        "H2-closure-scales.csv": h2, "H2-per-H.csv": h2_h,
        "H3-near-barrel.csv": near, "H3-per-H.csv": near_h, "H3-contrasts.csv": near_contrasts,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in artifacts.items():
        table.to_csv(destination / name, index=False)
    protocol = {
        "schema": "betlas.geometry-phenotypes.v1", "status": "complete", "domains": len(data),
        "questions": ["H1: J versus individual other 2.60 T", "H2: B/J/S_other closure at distinct scales", "H3: near-barrel within 2.60"],
        "grouping": "B=A2.40; J=T2.60.120; S_other=A2.60 excluding J. No 2.70/2.102 pooling.",
        "aggregation": "Domain summaries plus the equally weighted mean of within-H means; counts and missing support retained.",
        "scopes": "All rows are descriptive; each frozen geometry role is also reported separately.",
        "K_primary": "status=ok, positive node count, pairing_status available or no_retained_pairings; absent edges may be observed zeros.",
        "K_sensitivity": "K_retained_sensitivity adds partial_clipped_pairings; K_partial_retained_only shows partial graphs separately.",
        "K_edge_conditioned": "Direction/span quantities without eligible edges stay NA, even in a complete zero-edge graph.",
        "two_sheet": "parse/feature status successful, sheet_count>=2 and sheet_face_count>=2; both largest node-count sheets then contain >=2 beta nodes.",
        "near_barrel": "Input label-conditioned boolean on the previously fixed geometric relation. All scope uses the all graph; discovery/confirmation scopes use their corresponding phase graph only. Missing phase measurements remain NA and never fall back to the all graph.",
        "fixed_bins": {"observed_length": [100, 200], "beta_fraction": [0.5], "strand_count": [8]},
        "bin_interpretation": "Unadjusted fixed-bin conditional descriptions, including unknown-bin rows; not confounder removal or causal effects.",
        "limits": "Graph cycles do not establish barrel geometry; scalar graph summaries do not uniquely establish topology. No p-values or CIs.",
        "artifacts": {name: len(table) for name, table in artifacts.items()},
    }
    write_json(destination / "protocol.json", protocol)
    return protocol
