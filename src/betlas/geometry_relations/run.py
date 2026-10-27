"""Reproducible local geometry experiment stages from an explicit run config."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from .cohorts import A, H, T, allocate_roles, category, engineering_panel
from .io import read_csv, read_json, write_json
from .packing import PACKING_BLOCKS, packing_feature_contract


def prepare(config_path: str | Path) -> dict:
    config = read_json(config_path)
    source, dest = Path(config["source_derived"]), Path(config["derived"])
    dest.mkdir(parents=True, exist_ok=True)
    frame = read_csv(source / "features/s35.csv", dtype=str, keep_default_na=False)
    mother = read_csv(
        source / "catalog/domains.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["record_id", "pdb_id", A, T, H, "cath_s35_cluster_id"],
    )
    panel = engineering_panel(frame)
    roles, dependency = allocate_roles(frame, mother, panel)
    frame = frame.merge(
        roles.drop(columns=["pdb_id", A, T, H, "feature_status"]),
        on="record_id",
        validate="one_to_one",
    )
    frame["strict_label"] = [category(a, t) for a, t in zip(frame[A], frame[T], strict=True)]
    frame["legacy_label"] = [
        category(a, t, legacy=True) for a, t in zip(frame[A], frame[T], strict=True)
    ]
    sizes = mother.groupby("cath_s35_cluster_id").size()
    frame["mother_s35_domain_count"] = frame.cath_s35_cluster_id.map(sizes).astype(int)
    frame.to_csv(dest / "cohort.csv", index=False)
    roles.to_csv(dest / "roles.csv", index=False)
    panel.to_csv(dest / "engineering-panel.csv", index=False)
    coverage = (
        frame.groupby([A, "role", "feature_status"], dropna=False)
        .agg(domains=("record_id", "size"), h_groups=(H, "nunique"), topologies=(T, "nunique"))
        .reset_index()
    )
    coverage.to_csv(dest / "cohort-coverage.csv", index=False)
    protocol = {
        "schema": "betlas.geometry-relations.protocol.v1",
        "run_id": config["run_id"],
        "seed": 13,
        "packing": packing_feature_contract(),
        "neighborhood_sizes": [10, 30, 60],
        "primary_metric": "rms",
        "primary_k": 30,
        "sensitivity": ["l1", *["drop_" + k for k in PACKING_BLOCKS]],
        "short_distance_cut": "75th percentile of discovery-only k30 mutual different-H/different-PDB edge distances; fixed for confirmation",
        "confirmation": "Independent within-confirmation graph with discovery-fitted measurements and distance cut; not unseen historical data",
        "node_support": "All eligible rows retained. Dependency-excluded top-k edges are not replaced by farther neighbors.",
        "candidates": "Historical pairs plus at most three new A-pairs ranked by H support and stability using discovery only",
        "inference": "Descriptive H-weighted effects and support/deletion sensitivity; no domain-independent p-values or CIs",
        "model_uncertainty": "Top-two margin <= 0.2; a raw model diagnostic, not calibrated physical uncertainty",
        "model_overlap": "Boolean presence of a different-A short mutual neighbor; label-conditioned; unavailable neighborhoods stay NA",
        "allocation": dependency,
        "software": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "biopython", "gemmi"]
        },
        "python": sys.version,
    }
    write_json(dest / "protocol.json", protocol)
    write_json(Path(config["internal"]) / "PROTOCOL.json", protocol)
    return {"domains": len(frame), "panel": len(panel), **dependency}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2), flush=True)


if __name__ == "__main__":
    main()
