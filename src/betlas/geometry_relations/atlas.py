"""Packing graphs with frozen discovery measurements and explicit phase roles."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json
from .packing import PACKING_BLOCKS, PackingScaler, knn_from_distances, pairwise_block_distances


def export_relations(
    nodes: pd.DataFrame, edges: pd.DataFrame, directory: Path, architectures: list[str]
) -> dict:
    from .relationships import summarize_relationships

    directory.mkdir(parents=True, exist_ok=True)
    result = summarize_relationships(nodes, edges, A, all_labels=architectures)
    for key in ("directed", "undirected", "per_domain"):
        pd.DataFrame(result[key]).to_csv(directory / f"architecture-{key}.csv", index=False)
    write_json(directory / "summary.json", result["summary"])
    return result


def run_atlas(config_path: str | Path, *, phase: str) -> dict:
    cfg = read_json(config_path)
    derived = Path(cfg["derived"])
    result_root = Path(cfg["outputs"]) / "atlas" / phase
    if result_root.exists():
        raise FileExistsError(f"Atlas phase already exists: {result_root}")
    if phase != "discovery" and not (derived / "frozen-candidates.json").exists():
        raise RuntimeError("Candidate selection must be frozen before confirmation/all atlas")
    source = read_csv(derived / "cohort.csv", dtype=str, keep_default_na=False)
    discovery = source.loc[source.role.eq("discovery")]
    scaler_path = derived / "packing-scaler.json"
    if phase == "discovery":
        scaler = PackingScaler.fit(discovery)
        write_json(scaler_path, scaler.to_dict())
    else:
        scaler = PackingScaler.from_dict(read_json(scaler_path))
    frame = source if phase == "all" else source.loc[source.role.eq(phase)]
    frame = frame.reset_index(drop=True)
    matrix = scaler.transform(frame)
    result_root.mkdir(parents=True)
    np.savez_compressed(
        result_root / "scaled-measurements.npz",
        record_id=frame.record_id.to_numpy(),
        columns=np.array(matrix.columns),
        values=matrix.values,
    )
    availability = matrix.availability_frame()
    availability.insert(0, "record_id", frame.record_id)
    availability.to_csv(result_root / "availability.csv", index=False)
    architectures = sorted(source[A].unique(), key=lambda v: tuple(map(int, v.split("."))))
    thresholds = {} if phase == "discovery" else read_json(derived / "short-distance-cuts.json")
    summary = {"phase": phase, "domains": len(frame), "variants": {}, "started": time.time()}
    for metric in ("rms", "l1"):
        cache = pairwise_block_distances(matrix, metrics=(metric,), block_size=256)
        drops = [()] + [(block,) for block in PACKING_BLOCKS] if metric == "rms" else [()]
        for drop in drops:
            name = "drop_" + drop[0] if drop else metric
            folder = result_root / name
            folder.mkdir()
            distances = cache.combine(metric=metric, drop_blocks=drop)
            neighbors = knn_from_distances(distances, max_k=60)
            eligible = np.isfinite(np.diag(distances))
            nodes = frame[
                [
                    "record_id",
                    "pdb_id",
                    A,
                    T,
                    H,
                    "role",
                    "strict_label",
                    "legacy_label",
                    "mother_h_pdb_component",
                    "mother_s35_domain_count",
                ]
            ].copy()
            nodes["packing_eligible"] = eligible
            nodes.to_csv(folder / "nodes.csv", index=False)
            # These preserve every raw top-k relation, including later excluded pairs.
            np.savez_compressed(
                folder / "neighbors.npz",
                record_id=frame.record_id.to_numpy(),
                indices=neighbors.indices,
                distances=neighbors.distances,
                finite_candidates=neighbors.finite_candidate_counts,
            )
            if phase == "discovery":
                base = neighbors.neighborhood(30, group_ids=frame[H], pdb_ids=frame.pdb_id)
                finite = base.edges.loc[base.edges.mutual, "distance"].to_numpy()
                thresholds[name] = {
                    "q50": float(np.quantile(finite, 0.5)) if len(finite) else None,
                    "q75": float(np.quantile(finite, 0.75)) if len(finite) else None,
                    "q90": float(np.quantile(finite, 0.9)) if len(finite) else None,
                    "n_directed_edges": len(finite),
                }
            threshold = thresholds[name]["q75"]
            variant_summary = {"packing_eligible": int(eligible.sum()), "distance_cut": threshold}
            for k in (10, 30, 60):
                neighborhood = neighbors.neighborhood(k, group_ids=frame[H], pdb_ids=frame.pdb_id)
                edges = neighborhood.edges.copy()
                edges["source_id"] = frame.record_id.to_numpy()[edges.source.to_numpy(dtype=int)]
                edges["target_id"] = frame.record_id.to_numpy()[edges.target.to_numpy(dtype=int)]
                edges["short_edge"] = (
                    edges.distance.le(threshold) if threshold is not None else False
                )
                edges.to_csv(folder / f"edges-k{k}.csv.gz", index=False)
                neighborhood.coverage.assign(record_id=frame.record_id).to_csv(
                    folder / f"coverage-k{k}.csv", index=False
                )
                if name == "rms" or k == 30:
                    relations = export_relations(
                        nodes, edges, folder / f"relations-k{k}", architectures
                    )
                    variant_summary[f"k{k}"] = relations["summary"]
                if name == "rms" and k == 30:
                    # Explicit same-A topology analysis and historical remapping.
                    from .relationships import summarize_relationships

                    for label, col in (("strict", "strict_label"), ("legacy", "legacy_label")):
                        rel = summarize_relationships(nodes, edges, col)
                        pd.DataFrame(rel["directed"]).to_csv(
                            folder / f"{label}-directed.csv", index=False
                        )
                        pd.DataFrame(rel["per_domain"]).to_csv(
                            folder / f"{label}-per-domain.csv", index=False
                        )
                    for a, subset in nodes.groupby(A, sort=True):
                        ids = set(subset.record_id)
                        retained = edges.loc[edges.source_id.isin(ids) & edges.target_id.isin(ids)]
                        rel = summarize_relationships(subset, retained, T)
                        pd.DataFrame(rel["directed"]).to_csv(
                            folder / f"within-{a}-topology.csv", index=False
                        )
                    # Additional absolute cuts are not selected by outcome.
                    for cut in ("q50", "q90"):
                        changed = edges.copy()
                        changed["short_edge"] = changed.distance.le(thresholds[name][cut])
                        export_relations(
                            nodes, changed, folder / f"relations-k30-{cut}", architectures
                        )
            summary["variants"][name] = variant_summary
            del distances, neighbors
            gc.collect()
        del cache
        gc.collect()
    if phase == "discovery":
        write_json(derived / "short-distance-cuts.json", thresholds)
    summary["elapsed_seconds"] = time.time() - summary.pop("started")
    write_json(result_root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation", "all"], required=True)
    args = parser.parse_args()
    print(json.dumps(run_atlas(args.config, phase=args.phase), indent=2), flush=True)


if __name__ == "__main__":
    main()
