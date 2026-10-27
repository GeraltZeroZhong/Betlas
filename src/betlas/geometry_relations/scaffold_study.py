"""Prepare explicit, outcome-independent panels for a scaffold study.

Reference labels control sampling coverage and reporting strata. Neither the
packing metric nor structural correspondence is fitted to these labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json
from .packing import PackingScaler, prepare_packing_features
from .phenotypes import K_COLUMNS


def metadata_pool(frame: pd.DataFrame, query: pd.Series) -> pd.Series:
    """The fixed size/completeness calipers, excluding same H and same PDB."""
    ratio = frame.observed_length / query.observed_length
    return (
        ratio.between(0.8, 1.25)
        & frame.beta_fraction.sub(query.beta_fraction).abs().le(0.15)
        & frame.strand_count.sub(query.strand_count).abs().le(2)
        & frame.missing_fraction.sub(query.missing_fraction).abs().le(0.05)
        & frame[H].ne(query[H]) & frame.pdb_id.ne(query.pdb_id)
        & frame.record_id.ne(query.record_id)
    )


def block_distances(values: np.ndarray, blocks: dict, index: int) -> dict[str, np.ndarray]:
    """Per-block RMS distances, retaining unavailable values."""
    return {
        name: np.sqrt(np.mean((values[:, indices] - values[index, indices]) ** 2, axis=1))
        for name, indices in blocks.items()
    }


def sample_panel(frame: pd.DataFrame, scaler: PackingScaler, *, seed: int = 13,
                 anchors_per_architecture: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose length-spread anchors and five targets per supported anchor.

    Alternate same-T/different-T target pools across anchors. If the preferred
    pool is empty, use the other before any structural outcomes are observed.
    This is recorded explicitly. Targets are unique within an anchor, with no
    replacement between distance strata and the random baseline.
    """
    data = frame.sort_values("record_id").reset_index(drop=True).copy()
    matrix = scaler.transform(data)
    data["packing_eligible"] = np.isfinite(matrix.values).all(axis=1)
    rng = np.random.default_rng(seed)
    anchors, pairs = [], []
    for architecture, members in data.groupby(A, sort=True):
        members = members.loc[members.packing_eligible & members.observed_length.gt(0)]
        used, used_h = set(), set()
        targets = np.quantile(members.observed_length, [0.1, 0.35, 0.65, 0.9]) if len(members) else []
        for slot, target_length in enumerate(targets[:anchors_per_architecture]):
            available = members.loc[~members.index.isin(used)]
            if available.empty:
                break
            diverse = available.loc[~available[H].isin(used_h)]
            if not diverse.empty:
                available = diverse
            i = int(available.observed_length.sub(target_length).abs().idxmin())
            query = data.loc[i]
            used.add(i)
            used_h.add(query[H])
            allowed = metadata_pool(data, query) & data.packing_eligible
            same = allowed & data[T].eq(query[T])
            different = allowed & data[T].ne(query[T])
            prefer_same = slot % 2 == 0
            chosen = same if prefer_same else different
            fallback = not bool(chosen.any())
            if fallback:
                chosen = different if prefer_same else same
            stratum = "same_T" if (prefer_same != fallback) else "different_T"
            block = block_distances(matrix.values, matrix.blocks, i)
            distances = np.sqrt(np.mean(np.stack([x ** 2 for x in block.values()]), axis=0))
            pool = data.loc[chosen].copy()
            pool["distance"] = distances[chosen]
            pool = pool.sort_values(["distance", "record_id"])
            anchor_id = f"anchor_{len(anchors)+1:03d}"
            anchor = {"anchor_id": anchor_id, "record_id": query.record_id, A: architecture,
                      T: query[T], H: query[H], "slot": slot, "observed_length": query.observed_length,
                      "same_t_candidates": int(same.sum()), "different_t_candidates": int(different.sum()),
                      "target_stratum": stratum, "fallback_before_alignment": fallback,
                      "eligible_targets": len(pool), "status": "available" if len(pool) else "no_matched_targets"}
            picked = set()
            n = len(pool)
            for tier in ("nearest", "q25_50", "q50_75", "q75_100", "random"):
                if not n:
                    continue
                if tier == "nearest":
                    candidates = pool.iloc[:1]
                elif tier == "random":
                    candidates = pool
                else:
                    lo, hi = {"q25_50": (0.25, 0.5), "q50_75": (0.5, 0.75), "q75_100": (0.75, 1)}[tier]
                    candidates = pool.iloc[int(np.floor(lo*n)):int(np.ceil(hi*n))]
                candidates = candidates.loc[~candidates.index.isin(picked)]
                if candidates.empty:
                    continue
                j = int(candidates.index[0] if tier == "nearest" else rng.choice(candidates.index))
                picked.add(j)
                target = data.loc[j]
                row = {"pair_id": f"atlas_{len(pairs)+1:04d}", "panel": "atlas",
                       "anchor_id": anchor_id, "query_id": query.record_id, "target_id": target.record_id,
                       "distance_tier": tier, "target_stratum": stratum, "pool_size": n,
                       "packing_distance": float(distances[j]), "target_pool_rank": int(pool.index.get_loc(j)+1)}
                for side, record in (("query", query), ("target", target)):
                    for col in (A, T, H, "pdb_id", "observed_length", "beta_fraction", "strand_count", "missing_fraction"):
                        row[f"{side}_{col}"] = record[col]
                for name, distance in block.items():
                    row[f"distance_{name}"] = float(distance[j])
                pairs.append(row)
            anchor["selected_pairs"] = len(picked)
            anchors.append(anchor)
    return pd.DataFrame(anchors), pd.DataFrame(pairs)


def paired_readouts(pairs: pd.DataFrame, cohort: pd.DataFrame, phenotypes: pd.DataFrame,
                    scaler: PackingScaler) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Common-set raw P/K differences and two-sided family summaries."""
    cohort = cohort.set_index("record_id", drop=False)
    phenotype = phenotypes.set_index("record_id", drop=False)
    p = prepare_packing_features(cohort, scaler.blocks)
    mat = scaler.transform(cohort)
    lookup = {rid: i for i, rid in enumerate(cohort.index)}
    records, metrics = [], []
    for pair in pairs.to_dict("records"):
        qid, tid = pair["query_id"], pair["target_id"]
        i, j = lookup[qid], lookup[tid]
        row = dict(pair)
        squares = []
        for name, indices in mat.blocks.items():
            square = float(np.mean((mat.values[i, indices] - mat.values[j, indices]) ** 2))
            row[f"distance_{name}"] = np.sqrt(square)
            squares.append(square)
        row["packing_distance"] = np.sqrt(np.mean(squares))
        row["query_h"], row["target_h"] = cohort.loc[qid, H], cohort.loc[tid, H]
        records.append(row)
        for field in [*p.columns, *K_COLUMNS]:
            source = p if field in p else phenotype
            q, t = source.loc[qid, field], source.loc[tid, field]
            metrics.append({"pair_id": row["pair_id"], "field": field, "query_h": row["query_h"],
                            "target_h": row["target_h"], "query_value": q, "target_value": t,
                            "query_minus_target": q-t if pd.notna(q) and pd.notna(t) else np.nan})
    metrics = pd.DataFrame(metrics)
    summaries = []
    for field, group in metrics.groupby("field", sort=True):
        valid = group.dropna(subset=["query_minus_target"])
        qmeans, tmeans = valid.groupby("query_h").query_minus_target.mean(), valid.groupby("target_h").query_minus_target.mean()
        summaries.append({"field": field, "pairs_total": len(group), "pairs_observed": len(valid),
                          "query_h_count": len(qmeans), "target_h_count": len(tmeans),
                          "pair_mean_difference": valid.query_minus_target.mean(),
                          "pair_median_difference": valid.query_minus_target.median(),
                          "pair_q10_difference": valid.query_minus_target.quantile(.1),
                          "pair_q90_difference": valid.query_minus_target.quantile(.9),
                          "query_h_equal_mean_difference": qmeans.mean(),
                          "target_h_equal_mean_difference": tmeans.mean(),
                          "query_h_positive_fraction": qmeans.gt(0).mean(),
                          "target_h_positive_fraction": tmeans.gt(0).mean()})
    return pd.DataFrame(records), metrics, pd.DataFrame(summaries)


def prepare(config_path: str | Path) -> dict:
    cfg = read_json(config_path)
    previous_d, previous_o = Path(cfg["previous_derived"]), Path(cfg["previous_outputs"])
    dest, out = Path(cfg["derived"]), Path(cfg["outputs"])/"common-matched"
    dest.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    types = {A: str, T: str, H: str}
    cohort_path = Path(cfg.get("measurement_cohort", previous_d/"cohort.csv"))
    scaler_path = Path(cfg.get("measurement_scaler", previous_d/"packing-scaler.json"))
    cohort = read_csv(cohort_path, dtype=types)
    pheno = read_csv(previous_o/"phenotypes/domain-phenotypes.csv", dtype=types)
    frame = pheno.merge(cohort[["record_id", "observed_residue_count", "requested_residue_count"]],
                        on="record_id", validate="one_to_one")
    frame["missing_fraction"] = 1-frame.observed_residue_count/frame.requested_residue_count
    # Include applicability metadata from the original extractor, not placeholders.
    fields = [c for c in cohort if c.startswith("betlas_") and c not in frame]
    frame = frame.merge(cohort[["record_id", *fields]], on="record_id", validate="one_to_one")
    scaler = PackingScaler.from_dict(read_json(scaler_path))
    anchors, atlas = sample_panel(frame, scaler, seed=cfg["seed"])
    anchors.to_csv(dest/"atlas-anchors.csv", index=False)
    atlas.to_csv(dest/"atlas-pairs.csv", index=False)
    old_pairs = read_csv(previous_d/"validation-pairs.csv", dtype=str)
    old_pairs["pair_id"] = "legacy_"+old_pairs.pair_id
    old_pairs["panel"] = "legacy"
    old_pairs["packing_distance_legacy"] = old_pairs.packing_distance
    old_matrix = scaler.transform(cohort)
    index = {rid:i for i,rid in enumerate(cohort.record_id)}
    for row_index, row in old_pairs.iterrows():
        i,j=index[row.query_id],index[row.target_id]
        squares=[]
        for name,positions in old_matrix.blocks.items():
            square=float(np.mean((old_matrix.values[i,positions]-old_matrix.values[j,positions])**2))
            old_pairs.loc[row_index,f"distance_{name}"]=np.sqrt(square)
            squares.append(square)
        old_pairs.loc[row_index,"packing_distance"]=str(np.sqrt(np.mean(squares)))
    matched = read_csv(previous_o/"matched-phenotypes/matched-pairs.csv", dtype={"target_topology":str})
    matched = matched.loc[matched.phase.eq(cfg["matched_phase"]) & matched.target_topology.eq(cfg["matched_target_topology"])].copy()
    complete = pheno.set_index("record_id").pairing_complete
    matched["both_complete_k"] = matched.j_record_id.map(complete) & matched.target_record_id.map(complete)
    matched["source_pair_id"] = matched.pair_id
    matched["pair_id"] = [f"matched_{i+1:03d}" for i in range(len(matched))]
    matched = matched.rename(columns={"j_record_id":"query_id", "target_record_id":"target_id"})
    matched["panel"] = "common_matched"
    matched.to_csv(out/"all-matched-pairs.csv", index=False)
    common = matched.loc[matched.both_complete_k].copy()
    wide, long, summary = paired_readouts(common, cohort, pheno, scaler)
    wide.to_csv(out/"common-pairs.csv", index=False)
    long.to_csv(out/"paired-readouts.csv.gz", index=False)
    summary.to_csv(out/"summary.csv", index=False)
    for side in ("query_h", "target_h"):
        long.groupby(["field",side],sort=True).query_minus_target.agg(["count","mean","median"]).reset_index().to_csv(out/f"{side}-differences.csv",index=False)
    pairs = pd.concat([old_pairs, wide, atlas], ignore_index=True)
    pairs.to_csv(dest/"study-pairs.csv", index=False)
    requests=[]
    for pair in pairs.to_dict("records"):
        modes = {"legacy":["sNS","fNS"], "common_matched":["sNS"], "atlas":["SQ","sNS"]}[pair["panel"]]
        for mode in modes:
            requests.append({"pair_id":pair["pair_id"],"query_id":pair["query_id"],"target_id":pair["target_id"],"mode":mode})
    pd.DataFrame(requests).to_csv(dest/"alignment-requests.csv", index=False)
    ids = sorted(set(pairs.query_id)|set(pairs.target_id))
    pd.DataFrame({"record_id":ids}).to_csv(dest/"coordinate-requests.csv",index=False)
    coverage=cohort.groupby(A).agg(s35_domains=("record_id","size"),h_groups=(H,"nunique")).reset_index()
    coverage=coverage.merge(anchors.groupby(A).agg(anchors=("record_id","size"),anchors_with_targets=("selected_pairs",lambda x:int(x.gt(0).sum())),pairs=("selected_pairs","sum")).reset_index(),how="left",on=A)
    coverage.to_csv(dest/"atlas-coverage.csv",index=False)
    protocol={"schema":"betlas.scaffold-study.v1","run_id":cfg["run_id"],"seed":cfg["seed"],
              "cohort_source":str(cohort_path),"scaler_source":str(scaler_path),"labels":"Sampling strata and reporting only; no structural-outcome fitting",
              "anchor_selection":"Per A up to four observed-length quantiles 0.1/0.35/0.65/0.9, prefer unused H, deterministic ID ties",
              "target_strata":"Alternate same-T/different-T by anchor slot; switch only when preferred metadata pool empty, before alignment",
              "calipers":{"target_query_length_ratio":[.8,1.25],"beta_absolute":.15,"strand_absolute":2,"missing_absolute":.05},
              "target_selection":"Nearest; random within rank quantiles 25–50,50–75,75–100%; random from remaining matched pool; no replacement within anchor",
              "distance":"Configured frozen 18-field equal-block RMS, complete active fields; all distances and selected ranks saved",
              "interpretation":"Exploratory stratified retrieval curves, not prevalence or independent-pair inference",
              "counts":{"legacy_pairs":len(old_pairs),"matched_original":len(matched),"matched_complete":len(common),"anchors":len(anchors),"atlas_pairs":len(atlas),"coordinate_domains":len(ids),"alignment_requests":len(requests)}}
    write_json(dest/"protocol.json",protocol)
    write_json(Path(cfg["internal"])/"PROTOCOL.json",protocol)
    return protocol


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    print(prepare(parser.parse_args().config))
