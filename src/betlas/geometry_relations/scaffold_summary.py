"""Descriptive scaffold-study summaries with explicit repeated-anchor units.

No independent-pair significance tests or confidence intervals are produced.
Spatial graph compatibility is distinct from abstract graph isomorphism. Native
SQ results inherited without atom correspondences remain scalar-only records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json

NATIVE_METRICS = (
    "query_tm_score", "target_tm_score", "min_tm_score", "rmsd_angstrom",
    "query_coverage", "target_coverage", "min_coverage", "aligned_length",
    "short_long_length_ratio", "aligned_sequence_identity",
)
COVERAGE_METRICS = tuple(
    f"{role}_{kind}_coverage_{threshold}"
    for threshold in ("2", "3p5", "5")
    for kind in ("beta", "strand", "full_strand")
    for role in ("query", "target", "min")
)
STRUCTURAL_METRICS = COVERAGE_METRICS + (
    "query_split_strand_fraction", "target_split_strand_fraction",
    "query_spatially_mapped_strand_fraction", "target_spatially_mapped_strand_fraction",
    "strand_correspondence_longest_run_fraction", "whole_sheet_bijection",
    "one_to_one_sheet_pair_count", "one_to_one_sheet_fit_residue_pair_count",
    "one_to_one_sheet_common_fit_rmsd", "one_to_one_sheet_independent_fit_rmsd",
    "one_to_one_sheet_assembly_rmsd_gap", "common_sheet_fit_rmsd",
    "independent_sheet_fit_rmsd", "common_minus_independent_sheet_rmsd",
    "graph_mapping_complete", "graph_mapped_pair_edge_precision",
    "graph_mapped_pair_edge_recall",
)
DISTANCE_METRICS = (
    "packing_distance", "packing_distance_legacy", "distance_global_shape",
    "distance_sheet_shape", "distance_axis_organization", "distance_angular_repetition",
)
METRICS = NATIVE_METRICS + STRUCTURAL_METRICS + DISTANCE_METRICS
COMPARISON_METRICS = (
    "min_tm_score", "min_beta_coverage_2", "min_beta_coverage_3p5",
    "min_beta_coverage_5", "min_strand_coverage_3p5",
    "query_spatially_mapped_strand_fraction", "target_spatially_mapped_strand_fraction",
    "one_to_one_sheet_assembly_rmsd_gap", "packing_distance",
)
BOOL_METRICS = {"whole_sheet_bijection", "graph_mapping_complete"}


def _bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if value in (True, 1, "True", "true", "1", "1.0"):
        return True
    if value in (False, 0, "False", "false", "0", "0.0"):
        return False
    return None


def graph_relation(row: pd.Series | dict) -> str:
    """Classify only the actual complete spatial strand correspondence.

    The chain graph includes polymer order, fragment partition and observed-path
    attributes; a changed chain relation does not by itself prove rewiring.
    """
    if row.get("measurement_scope") == "native_scalar_only":
        return "not_measured_scalar_only"
    complete = _bool(row.get("graph_mapping_complete"))
    if complete is False:
        return "partial_spatial_mapping"
    if complete is None:
        return "spatial_mapping_unavailable"
    pair = _bool(row.get("graph_pair_mapping_compatible"))
    chain = _bool(row.get("graph_chain_mapping_compatible"))
    if pair is None:
        return "complete_mapping_pair_graph_unavailable"
    if not pair:
        if chain is True:
            raise ValueError("Chain compatibility cannot hold when pairing is incompatible")
        return "pairing_changed"
    if chain is None:
        return "complete_mapping_chain_graph_unavailable"
    return "connection_preserved" if chain else "pairing_preserved_chain_changed"


def load_alignment_source(path: str | Path, *, legacy: bool = False,
                          scalar_only: bool = False) -> pd.DataFrame:
    """Keep original source IDs and turn relative raw pointers into absolute paths."""
    path = Path(path).resolve()
    data = read_csv(path, dtype=str)
    data["source_pair_id_alignment"] = data.pair_id
    if legacy:
        data["pair_id"] = data.pair_id.map(lambda x: x if x.startswith("legacy_") else f"legacy_{x}")
    if scalar_only:
        data["mode"] = "SQ"
        data["alignment_status"] = data.status
        # Protect the scalar-only contract even if a future input gains columns.
        data = data.drop(columns=[c for c in data if c in STRUCTURAL_METRICS or c.startswith("graph_")])
    data["measurement_scope"] = "native_scalar_only" if scalar_only else "actual_correspondences"
    data["source_csv"] = str(path)
    data["source_dir"] = str(path.parent)
    data["raw_dir"] = str(path.parent / "raw")
    for column in list(data):
        if column.endswith("_path"):
            data[f"source_{column}"] = data[column]
            data[column] = data[column].map(
                lambda x: str((path.parent / str(x)).resolve()) if pd.notna(x) else np.nan)
    return data


def join_alignments(pairs: pd.DataFrame, requests: pd.DataFrame,
                    cohort: pd.DataFrame, sources: list[pd.DataFrame], *,
                    include_legacy_sq: bool = True) -> pd.DataFrame:
    """Retain every planned pair/mode and reject duplicate or mismatched results."""
    pairs, requests, cohort = pairs.copy(), requests.copy(), cohort.copy()
    if pairs.pair_id.duplicated().any() or cohort.record_id.duplicated().any():
        raise ValueError("Pair and cohort IDs must be unique")
    if include_legacy_sq:
        legacy = pairs.loc[pairs.panel.eq("legacy"), ["pair_id", "query_id", "target_id"]].copy()
        legacy["mode"] = "SQ"
        requests = pd.concat([requests, legacy], ignore_index=True)
    if requests.duplicated(["pair_id", "mode"]).any():
        raise ValueError("Alignment requests must be unique per pair and mode")
    observed = pd.concat(sources, ignore_index=True, sort=False) if sources else pd.DataFrame(
        columns=["pair_id", "mode", "query_id", "target_id", "status"])
    if observed.duplicated(["pair_id", "mode"]).any():
        raise ValueError("Multiple raw results for one pair/mode; do not silently prefer a run")
    keys = set(map(tuple, requests[["pair_id", "mode"]].to_numpy()))
    if any(key not in keys for key in map(tuple, observed[["pair_id", "mode"]].to_numpy())):
        raise ValueError("Raw results contain unplanned pair/mode combinations")
    data = requests.merge(observed, on=["pair_id", "mode"], how="left",
                          suffixes=("", "_result"), validate="one_to_one")
    for name in ("query_id", "target_id"):
        if (data[f"{name}_result"].notna() & data[name].ne(data[f"{name}_result"])).any():
            raise ValueError(f"Raw {name} does not match planned orientation")
        data = data.drop(columns=f"{name}_result")
    meta = pairs.drop(columns=["query_id", "target_id"])
    collision = set(meta).intersection(data) - {"pair_id"}
    # For example, frozen strand_count and actually annotated strand_count are
    # distinct measurements and must both remain visible.
    meta = meta.rename(columns={column: f"panel_{column}" for column in collision})
    data = data.merge(meta, on="pair_id", how="left", validate="many_to_one")
    if data.panel.isna().any():
        raise ValueError("A requested alignment has no study-pair metadata")
    for role in ("query", "target"):
        fields = [c for c in (A, T, H, "pdb_id", "role") if c in cohort]
        names = {"record_id": f"{role}_id", **{c: f"{role}_{c}" for c in fields}}
        labels = cohort[["record_id", *fields]].rename(columns=names)
        # The new cohort is authoritative, while old panel metadata stays visible.
        for column in names.values():
            if column != f"{role}_id" and column in data:
                data = data.rename(columns={column: f"panel_{column}"})
        data = data.merge(labels, on=f"{role}_id", how="left", validate="many_to_one")
        data[f"{role}_A"] = data[f"{role}_{A}"]
        data[f"{role}_T"] = data[f"{role}_{T}"]
        data[f"{role}_H"] = data[f"{role}_{H}"]
    data["actual_topology_relation"] = np.where(
        data.query_T.isna() | data.target_T.isna(), "unknown",
        np.where(data.query_T.eq(data.target_T), "same_T", "different_T"))
    if "anchor_id" not in data:
        data["anchor_id"] = np.nan
    data["unit_anchor"] = data.anchor_id.fillna(data.query_id)
    data["status"] = data.status.fillna("not_run")
    if "alignment_status" not in data:
        data["alignment_status"] = np.nan
    data["native_available"] = data.alignment_status.eq("ok") | data.status.eq("ok")
    if "measurement_scope" not in data:
        data["measurement_scope"] = np.nan
    data["measurement_scope"] = data.measurement_scope.fillna("not_run")
    data["correspondence_available"] = data.status.eq("ok") & data.measurement_scope.eq("actual_correspondences")
    for name in METRICS:
        values = data[name] if name in data else pd.Series(np.nan, index=data.index)
        if name in BOOL_METRICS:
            values = values.map(_bool)
        data[name] = pd.to_numeric(values, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        if name in NATIVE_METRICS:
            data.loc[~data.native_available, name] = np.nan
        elif name in STRUCTURAL_METRICS:
            data.loc[~data.correspondence_available, name] = np.nan
    data["graph_relation"] = data.apply(graph_relation, axis=1)
    return data


def _statistics(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {"n_observed": len(values), "mean": values.mean(), "median": values.median(),
            "q10": values.quantile(.1), "q90": values.quantile(.9),
            "minimum": values.min(), "maximum": values.max()}


def summarize_metrics(data: pd.DataFrame, group_columns: list[str],
                      metrics: tuple | list = METRICS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Domain-pair descriptions plus anchor means and equal-query-H anchor means.

    Pair medians and quantiles describe the selected pairs, not independent
    observations. H summaries average anchor means within H before averaging H.
    """
    summaries, anchors, hs = [], [], []
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for keys, group in data.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys, strict=True))
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
            observed = group.assign(metric_value=values).loc[values.notna()]
            per_anchor = observed.groupby(["unit_anchor", "query_H"], dropna=False).metric_value.agg(["mean", "size"])
            if not per_anchor.empty and per_anchor.reset_index().unit_anchor.duplicated().any():
                raise ValueError("One anchor cannot belong to multiple query H groups")
            anchor_frame = per_anchor.reset_index()
            per_h = anchor_frame.dropna(subset=["query_H"]).groupby("query_H")["mean"].agg(["mean", "size"])
            stats = _statistics(values)
            summaries.append({**base, "metric": metric, "n_requested": len(group),
                              "n_native_available": int(group.native_available.sum()),
                              "n_correspondence_available": int(group.correspondence_available.sum()),
                              **stats, "n_missing": int(values.isna().sum()),
                              "n_anchors_requested": group.unit_anchor.nunique(),
                              "n_anchors_observed": len(per_anchor), "n_query_H_observed": len(per_h),
                              "anchor_equal_mean": per_anchor["mean"].mean(),
                              "query_H_equal_anchor_mean": per_h["mean"].mean()})
            for (anchor, h), row in per_anchor.iterrows():
                anchors.append({**base, "metric": metric, "unit_anchor": anchor,
                                "query_H": h, "n_pairs_observed": int(row["size"]),
                                "anchor_mean": row["mean"]})
            for h, row in per_h.iterrows():
                hs.append({**base, "metric": metric, "query_H": h,
                           "n_anchors_observed": int(row["size"]), "H_mean_of_anchor_means": row["mean"]})
    return pd.DataFrame(summaries), pd.DataFrame(anchors), pd.DataFrame(hs)


def tier_random_differences(atlas: pd.DataFrame, metrics: tuple = COMPARISON_METRICS) -> pd.DataFrame:
    """Subtract the random target from each tier within the same anchor and mode."""
    rows = []
    if atlas.empty:
        return pd.DataFrame()
    if atlas.duplicated(["unit_anchor", "mode", "distance_tier"]).any():
        raise ValueError("Multiple targets in one anchor/mode/tier")
    random = atlas.loc[atlas.distance_tier.eq("random")].set_index(["unit_anchor", "mode"])
    for _, row in atlas.loc[~atlas.distance_tier.eq("random")].iterrows():
        key = row.unit_anchor, row["mode"]
        reference = random.loc[key] if key in random.index else None
        if reference is not None and reference.actual_topology_relation != row.actual_topology_relation:
            raise ValueError("Tier/random comparison changes the anchor's topology stratum")
        for metric in metrics:
            left = row.get(metric, np.nan)
            right = reference.get(metric, np.nan) if reference is not None else np.nan
            eligible = pd.notna(left) and pd.notna(right)
            rows.append({"pair_id": row.pair_id, "random_pair_id": reference.pair_id if reference is not None else None,
                         "unit_anchor": row.unit_anchor, "query_H": row.query_H,
                         "query_A": row.query_A, "mode": row["mode"], "distance_tier": row.distance_tier,
                         "actual_topology_relation": row.actual_topology_relation, "metric": metric,
                         "tier_value": left, "random_value": right,
                         "tier_minus_random": left - right if eligible else np.nan,
                         "status": "available" if eligible else ("no_random_target" if reference is None else "measurement_missing")})
    return pd.DataFrame(rows)


def summarize_differences(differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if differences.empty:
        return pd.DataFrame()
    for keys, group in differences.groupby(["mode", "actual_topology_relation", "distance_tier", "metric"], dropna=False):
        h = group.groupby("query_H").tier_minus_random.mean().dropna()
        rows.append(dict(zip(["mode", "actual_topology_relation", "distance_tier", "metric"], keys, strict=True)) | {
            "n_requested": len(group), **_statistics(group.tier_minus_random),
            "n_query_H_observed": len(h), "query_H_equal_mean": h.mean(),
            "n_positive": int(group.tier_minus_random.gt(0).sum()),
            "n_negative": int(group.tier_minus_random.lt(0).sum()),
            "n_zero": int(group.tier_minus_random.eq(0).sum())})
    return pd.DataFrame(rows)


def _rho(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    rho = (values.x.rank().corr(values.y.rank())
           if len(values) >= 3 and values.x.nunique() >= 2 and values.y.nunique() >= 2 else np.nan)
    return len(values), rho


def correlation_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Descriptive Spearman coefficients; no significance from reused pairs."""
    pooled, within = [], []
    metrics = ("min_tm_score", "min_beta_coverage_2", "min_beta_coverage_3p5", "min_beta_coverage_5")
    for keys, group in data.groupby(["panel", "mode", "actual_topology_relation"], dropna=False):
        base = dict(zip(["panel", "mode", "actual_topology_relation"], keys, strict=True))
        for metric in metrics:
            n, rho = _rho(group.packing_distance, group[metric])
            pooled.append({**base, "metric": metric, "n_requested": len(group), "n_observed": n,
                           "spearman_rho": rho, "unit": "selected_pairs_descriptive_only"})
        if base["panel"] != "atlas":
            continue
        for anchor, subset in group.groupby("unit_anchor"):
            for metric in metrics:
                n, rho = _rho(subset.packing_distance, subset[metric])
                within.append({**base, "unit_anchor": anchor, "query_H": subset.query_H.iloc[0],
                               "metric": metric, "n_requested": len(subset), "n_observed": n,
                               "spearman_rho": rho,
                               "status": "available" if pd.notna(rho) else "too_few_or_constant_values"})
    per_h = []
    within_frame = pd.DataFrame(within)
    if not within_frame.empty:
        for keys, group in within_frame.groupby(["mode", "actual_topology_relation", "metric", "query_H"], dropna=False):
            per_h.append(dict(zip(["mode", "actual_topology_relation", "metric", "query_H"], keys, strict=True)) | {
                "n_anchors_requested": len(group), **_statistics(group.spearman_rho)})
    return pd.DataFrame(pooled), within_frame, pd.DataFrame(per_h)


def graph_relation_counts(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(["panel", "mode"], dropna=False):
        eligible = group.graph_relation.isin(("connection_preserved", "pairing_preserved_chain_changed", "pairing_changed"))
        for relation in sorted(data.graph_relation.unique()):
            indicator = group.graph_relation.eq(relation).astype(float)
            anchor = group.assign(indicator=indicator).groupby(["unit_anchor", "query_H"], dropna=False).indicator.mean()
            h = anchor.reset_index().dropna(subset=["query_H"]).groupby("query_H").indicator.mean()
            count = int(indicator.sum())
            rows.append({"panel": keys[0], "mode": keys[1], "graph_relation": relation,
                         "n_requested": len(group), "n_relation": count,
                         "n_fully_classifiable": int(eligible.sum()), "fraction_of_requested": count / len(group),
                         "fraction_of_classifiable": count / int(eligible.sum()) if eligible.any() and relation in (
                             "connection_preserved", "pairing_preserved_chain_changed", "pairing_changed") else np.nan,
                         "anchor_equal_fraction": anchor.mean(), "query_H_equal_anchor_fraction": h.mean()})
    return pd.DataFrame(rows)


def legacy_sq_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Compare paired full-domain native scalars; never fabricate legacy SQ beta coverage."""
    legacy = data.loc[data.panel.eq("legacy")]
    sq = legacy.loc[legacy["mode"].eq("SQ")].set_index("pair_id")
    rows = []
    for _, row in legacy.loc[legacy["mode"].isin(("sNS", "fNS"))].iterrows():
        if row.pair_id not in sq.index:
            continue
        reference = sq.loc[row.pair_id]
        result = {k: row.get(k) for k in ("pair_id", "query_id", "target_id", "mode", "role", "relation_kind",
                                         "query_H", "target_H", "packing_distance", "packing_distance_legacy")}
        result.update(SQ_status=reference.status, correspondence_status=row.status,
                      SQ_source_csv=reference.source_csv, SQ_stdout_path=reference.get("stdout_path"),
                      comparison_scope="native_full_domain_scalars_only")
        for metric in NATIVE_METRICS:
            result[f"SQ_{metric}"] = reference[metric]
            result[f"{metric}"] = row[metric]
            result[f"minus_SQ_{metric}"] = row[metric] - reference[metric]
        rows.append(result)
    return pd.DataFrame(rows)


def common_pair_readouts(data: pd.DataFrame, readouts: pd.DataFrame) -> pd.DataFrame:
    common = data.loc[data.panel.eq("common_matched") & data["mode"].eq("sNS")]
    if readouts.duplicated(["pair_id", "field"]).any():
        raise ValueError("Duplicate common-pair readout")
    if set(readouts.pair_id) != set(common.pair_id):
        raise ValueError("Common P/K readouts and structural panel use different pairs")
    labels = readouts.drop(columns=[c for c in ("query_h", "target_h") if c in readouts])
    joint = labels.merge(common, on="pair_id", how="left", validate="many_to_one")
    for field in ("query_value", "target_value", "query_minus_target"):
        joint[field] = pd.to_numeric(joint[field], errors="coerce")
    return joint


def run_scaffold_summary(config_path: str | Path, output_dir: str | Path | None = None) -> Path:
    config = read_json(config_path)
    derived, outputs = Path(config["derived"]), Path(config["outputs"])
    target = Path(output_dir) if output_dir is not None else outputs / "scaffold-summary"
    if target.exists():
        raise FileExistsError(f"Refusing to replace an existing summary: {target}")
    cohort_path = Path(config["measurement_cohort"])
    pairs = read_csv(derived / "study-pairs.csv", dtype=str)
    requests = read_csv(derived / "alignment-requests.csv", dtype=str)
    cohort = read_csv(cohort_path, dtype=str)
    common_sources = [outputs / x / "scaffold-alignments.csv" for x in ("scaffold-common", "scaffold-common77")]
    if all(path.exists() for path in common_sources):
        raise ValueError("Both common panel source names exist; choose one explicitly")
    source_specs = [(outputs / "scaffold-legacy-smoke/scaffold-alignments.csv", True, False),
                    (outputs / "scaffold-legacy/scaffold-alignments.csv", True, False),
                    (next((p for p in common_sources if p.exists()), common_sources[0]), False, False),
                    (outputs / "scaffold-atlas/scaffold-alignments.csv", False, False),
                    (Path(config["previous_outputs"]) / "structure-validation/pair-alignments.csv", True, True)]
    sources = [load_alignment_source(path, legacy=legacy, scalar_only=scalar)
               for path, legacy, scalar in source_specs if path.exists()]
    data = join_alignments(pairs, requests, cohort, sources)
    target.mkdir(parents=True, exist_ok=False)
    data.to_csv(target / "alignments-joined.csv.gz", index=False)
    panel, _, _ = summarize_metrics(data, ["panel", "mode"])
    panel.to_csv(target / "panel-mode-distributions.csv", index=False)
    legacy = data.loc[data.panel.eq("legacy")]
    legacy_stats, _, _ = summarize_metrics(legacy, ["mode", "role", "relation_kind"])
    legacy_stats.to_csv(target / "legacy-relation-distributions.csv", index=False)
    atlas = data.loc[data.panel.eq("atlas")]
    atlas_stats, anchors, hs = summarize_metrics(atlas, ["mode", "actual_topology_relation", "distance_tier"])
    atlas_stats.to_csv(target / "atlas-tier-distributions.csv", index=False)
    anchors.to_csv(target / "atlas-anchor-metrics.csv", index=False)
    hs.to_csv(target / "atlas-H-metrics.csv", index=False)
    differences = tier_random_differences(atlas)
    differences.to_csv(target / "atlas-tier-minus-random.csv", index=False)
    summarize_differences(differences).to_csv(target / "atlas-tier-minus-random-summary.csv", index=False)
    graph_relation_counts(data).to_csv(target / "graph-relations.csv", index=False)
    legacy_sq_comparison(data).to_csv(target / "legacy-SQ-comparison.csv", index=False)
    data.loc[data.panel.eq("common_matched")].to_csv(target / "common77-joint.csv", index=False)
    common_path = outputs / "common-matched/paired-readouts.csv.gz"
    if common_path.exists():
        common_pair_readouts(data, read_csv(common_path, dtype=str)).to_csv(
            target / "common77-paired-readouts.csv.gz", index=False)
    pooled, within, correlations_h = correlation_tables(data)
    pooled.to_csv(target / "packing-coverage-correlations.csv", index=False)
    within.to_csv(target / "within-anchor-correlations.csv", index=False)
    correlations_h.to_csv(target / "within-anchor-correlations-per-H.csv", index=False)
    anchor_path = derived / "atlas-anchors.csv"
    anchor_inventory = read_csv(anchor_path, dtype=str) if anchor_path.exists() else pd.DataFrame()
    if not anchor_inventory.empty:
        anchor_inventory.to_csv(target / "atlas-anchor-coverage.csv", index=False)
    write_json(target / "summary-protocol.json", {
        "config": str(Path(config_path).resolve()), "study_pairs": str((derived / "study-pairs.csv").resolve()),
        "measurement_cohort": str(cohort_path.resolve()), "measurement_scaler": config.get("measurement_scaler"),
        "sources": [{"path": str(p.resolve()), "available": p.exists(), "legacy_prefix": legacy,
                     "scalar_only": scalar} for p, legacy, scalar in source_specs],
        "n_study_pairs": len(pairs), "n_requested_new_alignments": len(requests),
        "n_rows_including_inherited_SQ": len(data), "status_counts": data.status.value_counts().to_dict(),
        "n_atlas_anchors_selected": len(anchor_inventory) if not anchor_inventory.empty else None,
        "n_atlas_anchors_with_pairs": atlas.unit_anchor.nunique(),
        "definitions": {
            "packing_distance": "The study-pair v2 distance; legacy distance remains a separate column.",
            "coverage": "Actual beta-to-beta correspondences within 2, 3.5 and 5 angstrom; min means the smaller directional coverage.",
            "native_TM": "Native full-domain scores; not beta-scaffold-specific scores.",
            "unit": "Anchor (query domain), with H means of anchor means. Pair quantiles and correlations are descriptive.",
            "missing": "All planned pair/mode rows retained. Scalar-only inherited SQ has no beta, strand, sheet or spatial-graph measurements.",
            "tier_minus_random": "Within the same anchor and mode; NA when no random target or either measurement is missing.",
            "whole_sheet": "One-to-one sheet assembly gap requires at least two eligible sheet pairs; not synonymous with whole_sheet_bijection.",
            "graph_relation": "Three relations require complete spatial strand mapping and known relevant graph compatibility; partial/unknown stay separate.",
            "chain_change": "K_chain includes polymer order, chain/fragment partitions and observed paths; a change is not by itself proven rewiring.",
            "rank_correlation": "Spearman from average ranks, at least 3 finite pairs and nonconstant values; no p-values or intervals.",
            "selection": "Atlas anchor/topology/distance sampling and previous selected panels limit population generalization.",
        },
    })
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(run_scaffold_summary(args.config, args.output_dir))


if __name__ == "__main__":
    main()
