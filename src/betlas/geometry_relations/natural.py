"""Full-catalog measurement sensitivity without a dense all-domain graph."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json
from .packing import PackingScaler, prepare_packing_features


class BoundedReader(io.RawIOBase):
    def __init__(self, path):
        super().__init__()
        self.source = Path(path).open("rb", buffering=0)

    def readable(self):
        return True

    def readinto(self, target):
        block = self.source.read(min(len(target), 8192))
        target[: len(block)] = block
        return len(block)

    def close(self):
        if not self.closed:
            self.source.close()
        super().close()


def run_natural(config_path):
    cfg = read_json(config_path)
    derived, outputs = Path(cfg["derived"]), Path(cfg["outputs"]) / "natural"
    outputs.mkdir(parents=True, exist_ok=False)
    scaler = PackingScaler.from_dict(read_json(derived / "packing-scaler.json"))
    representatives = read_csv(derived / "cohort.csv", dtype=str, keep_default_na=False)
    rep_matrix = scaler.transform(representatives)
    rep_lookup = {key: i for i, key in enumerate(representatives.cath_s35_cluster_id)}
    fields = list(rep_matrix.columns)
    identities = ["record_id", A, T, H, "cath_s35_cluster_id", "feature_status"]
    applicability = [
        "betlas_parse_ok",
        "betlas_beta_residue_count",
        "betlas_sheet_face_count",
        "betlas_axis_best_slice_count",
        "betlas_beta_strand_count",
    ]
    columns = list(dict.fromkeys(identities + fields + applicability))
    all_rows = []
    source = Path(cfg["source_derived"]) / "features/all.csv"
    with BoundedReader(source) as raw:
        reader = pd.read_csv(raw, dtype=str, keep_default_na=False, usecols=columns, chunksize=4000)
        for chunk in reader:
            matrix = scaler.transform(chunk)
            indices = chunk.cath_s35_cluster_id.map(rep_lookup).fillna(-1).to_numpy(dtype=int)
            valid_ref = indices >= 0
            target = rep_matrix.values[np.maximum(indices, 0)]
            diff = matrix.values.astype(float) - target
            squared = np.zeros(len(chunk), dtype=float)
            for block_indices in matrix.blocks.values():
                squared += np.mean(diff[:, block_indices] ** 2, axis=1)
            distance = np.sqrt(squared / len(matrix.blocks))
            distance[~valid_ref] = np.nan
            measured = prepare_packing_features(chunk)
            row = chunk[identities].reset_index(drop=True).copy()
            row["packing_eligible"] = np.isfinite(matrix.values).all(axis=1)
            row["distance_to_s35_representative"] = distance
            row["representative_id"] = [
                representatives.record_id.iloc[i] if i >= 0 else "" for i in indices
            ]
            for field in fields:
                row[field] = measured[field].to_numpy()
            all_rows.append(row)
    frame = pd.concat(all_rows, ignore_index=True)
    frame.to_csv(outputs / "domain-measurements.csv.gz", index=False)
    h_means = frame.groupby([A, H])[fields].mean().reset_index()
    h_means.to_csv(outputs / "h-means.csv", index=False)
    summaries = []
    for a, group in frame.groupby(A):
        d = group.distance_to_s35_representative.dropna()
        summaries.append(
            {
                A: a,
                "domains": len(group),
                "h_groups": group[H].nunique(),
                "packing_eligible": int(group.packing_eligible.sum()),
                "rep_distance_available": len(d),
                "rep_distance_median": d.median(),
                "rep_distance_q90": d.quantile(0.9),
                "rep_distance_q99": d.quantile(0.99),
            }
        )
    pd.DataFrame(summaries).to_csv(outputs / "architecture-support.csv", index=False)
    comparisons = []
    rep_raw = prepare_packing_features(representatives)
    rep_raw[A], rep_raw[H] = representatives[A], representatives[H]
    rep_h = rep_raw.groupby([A, H])[fields].mean().reset_index()
    for a, group in h_means.groupby(A):
        for field in fields:
            previous = rep_h.loc[rep_h[A].eq(a), field]
            comparisons.append(
                {
                    A: a,
                    "field": field,
                    "full_equal_h_mean": group[field].mean(),
                    "s35_equal_h_mean": previous.mean(),
                    "difference": group[field].mean() - previous.mean(),
                    "full_h_available": group[field].notna().sum(),
                    "s35_h_available": previous.notna().sum(),
                }
            )
    pd.DataFrame(comparisons).to_csv(outputs / "full-vs-s35-h-means.csv", index=False)
    cuts = read_json(derived / "short-distance-cuts.json")
    available = frame.distance_to_s35_representative.notna()
    summary = {
        "domains": len(frame),
        "architectures": frame[A].nunique(),
        "h_groups": frame[H].nunique(),
        "feature_status": frame.feature_status.value_counts().to_dict(),
        "packing_eligible": int(frame.packing_eligible.sum()),
        "distance_available": int(available.sum()),
        "within_cluster_distance_over_discovery_edge_cut": int(
            frame.loc[available, "distance_to_s35_representative"].gt(cuts["rms"]["q75"]).sum()
        ),
        "interpretation": "Full-domain physical distributions and within-S35 heterogeneity; no all-domain neighbor graph or independent replication claim.",
    }
    write_json(outputs / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    print(json.dumps(run_natural(parser.parse_args().config), indent=2), flush=True)
