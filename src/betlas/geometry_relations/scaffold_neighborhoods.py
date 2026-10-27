"""Rebuild measured neighborhoods and attach saved predictions without refitting."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json
from .packing import PackingScaler, knn_from_distances, pairwise_block_distances
from .relationships import summarize_relationships


def pattern_neighbor_stats(nodes, edges):
    """Compare complete observed chain graphs; partial graphs have no fraction."""
    lookup = nodes.set_index("record_id")
    primary = edges.loc[edges.mutual & edges.short_edge]
    rows = []
    for query, group in primary.groupby("source_id", sort=True):
        targets = lookup.loc[group.target_id]
        query_row = lookup.loc[query]
        valid = (targets.chain_primary_eligible.fillna(False)
                 & bool(query_row.chain_primary_eligible)
                 & targets.chain_pattern.notna() & pd.notna(query_row.chain_pattern))
        rows.append({"record_id": query, "chain_pattern_neighbor_count": int(valid.sum()),
                     "same_chain_pattern_fraction": targets.loc[valid, "chain_pattern"].eq(query_row.chain_pattern).mean() if valid.any() else np.nan})
    return pd.DataFrame(rows, columns=["record_id", "chain_pattern_neighbor_count", "same_chain_pattern_fraction"])


def refresh_pattern_overlay(cfg, root):
    """Refresh graph eligibility without recomputing frozen geometric neighbors."""
    patterns = read_csv(Path(cfg["outputs"])/"connection-patterns/domain-patterns.csv").set_index("record_id")
    for phase in ("discovery", "confirmation", "all"):
        folder = root/phase
        nodes = read_csv(folder/"nodes.csv", dtype={A:str, T:str, H:str})
        nodes["chain_primary_eligible"] = nodes.record_id.map(patterns.chain_primary_eligible)
        nodes.to_csv(folder/"nodes.csv", index=False)
        edges = read_csv(folder/"edges.csv.gz")
        geometry = read_csv(folder/"architecture-per_domain.csv", dtype={"label":str, "homology":str})
        geometry = geometry.merge(pattern_neighbor_stats(nodes, edges), on="record_id", how="left", validate="one_to_one")
        geometry.to_csv(folder/"domain-geometry.csv", index=False)
        if phase == "all":
            topology = pd.DataFrame(summarize_relationships(nodes, edges, T, pair_tables=False)["per_domain"])
            topology = topology.merge(pattern_neighbor_stats(nodes, edges), on="record_id", how="left", validate="one_to_one")
            topology.to_csv(folder/"topology-domain-geometry.csv", index=False)
    overlay_models(cfg, root)


def run(config_path):
    cfg = read_json(config_path)
    frame = read_csv(cfg["measurement_cohort"], dtype={A:str,T:str,H:str})
    scaler = PackingScaler.from_dict(read_json(cfg["measurement_scaler"]))
    root = Path(cfg["outputs"])/"invariant-neighborhoods"
    root.mkdir(parents=True,exist_ok=False)
    patterns=read_csv(Path(cfg["outputs"])/"connection-patterns/domain-patterns.csv",dtype={A:str,T:str,H:str}).set_index("record_id")
    summaries={}
    cutoff=None
    for phase in ("discovery","confirmation","all"):
        folder=root/phase
        folder.mkdir()
        scope=(frame if phase=="all" else frame.loc[frame.role.eq(phase)]).reset_index(drop=True)
        matrix=scaler.transform(scope)
        cache=pairwise_block_distances(matrix,metrics=("rms",))
        distances=cache.combine(metric="rms")
        del cache
        neighbors=knn_from_distances(distances,max_k=30)
        eligible=np.isfinite(np.diag(distances))
        del distances
        neighborhood=neighbors.neighborhood(30,group_ids=scope[H],pdb_ids=scope.pdb_id)
        edges=neighborhood.edges.copy()
        edges["source_id"]=scope.record_id.to_numpy()[edges.source.to_numpy(dtype=int)]
        edges["target_id"]=scope.record_id.to_numpy()[edges.target.to_numpy(dtype=int)]
        if phase=="discovery":
            cutoff=float(edges.loc[edges.mutual,"distance"].quantile(.75))
        edges["short_edge"]=edges.distance.le(cutoff)
        nodes=scope[["record_id","pdb_id",A,T,H,"role"]].copy()
        nodes["packing_eligible"]=eligible
        nodes["pair_pattern"]=nodes.record_id.map(patterns.pair_pattern)
        nodes["chain_pattern"]=nodes.record_id.map(patterns.chain_pattern)
        nodes["chain_primary_eligible"]=nodes.record_id.map(patterns.chain_primary_eligible)
        nodes.to_csv(folder/"nodes.csv",index=False)
        edges.to_csv(folder/"edges.csv.gz",index=False)
        neighborhood.coverage.assign(record_id=scope.record_id).to_csv(folder/"coverage.csv",index=False)
        np.savez_compressed(folder/"neighbors.npz",record_id=scope.record_id.to_numpy(),indices=neighbors.indices,distances=neighbors.distances)
        relation=summarize_relationships(nodes,edges,A,all_labels=sorted(frame[A].unique()))
        for key in ("per_domain","directed","undirected"):
            pd.DataFrame(relation[key]).to_csv(folder/f"architecture-{key}.csv",index=False)
        per_domain=pd.DataFrame(relation["per_domain"])
        primary=edges.loc[edges.mutual & edges.short_edge]
        per_domain=per_domain.merge(pattern_neighbor_stats(nodes,edges),on="record_id",how="left",validate="one_to_one")
        per_domain.to_csv(folder/"domain-geometry.csv",index=False)
        if phase == "all":
            topology = pd.DataFrame(summarize_relationships(nodes, edges, T, pair_tables=False)["per_domain"])
            topology = topology.merge(pattern_neighbor_stats(nodes, edges), on="record_id", how="left", validate="one_to_one")
            topology.to_csv(folder/"topology-domain-geometry.csv", index=False)
        # Family concentration and an induced-subgraph contribution diagnostic.
        contribution=primary.merge(nodes[["record_id",A,H]],left_on="source_id",right_on="record_id").rename(columns={A:"source_A",H:"source_H"}).drop(columns="record_id")
        contribution=contribution.merge(nodes[["record_id",A,H]],left_on="target_id",right_on="record_id").rename(columns={A:"target_A",H:"target_H"})
        contribution.groupby(["source_A","target_A","source_H","target_H"]).size().rename("directed_edges").reset_index().to_csv(folder/"family-contributions.csv",index=False)
        if cfg.get("contribution_remove_h"):
            remove=cfg["contribution_remove_h"]
            retained=nodes.loc[nodes[H].ne(remove)]
            kept_ids=set(retained.record_id)
            after=edges.loc[edges.source_id.isin(kept_ids)&edges.target_id.isin(kept_ids)]
            deleted=summarize_relationships(retained,after,A,all_labels=sorted(frame[A].unique()))
            pd.DataFrame(deleted["directed"]).to_csv(folder/"after-family-deletion-directed.csv",index=False)
            write_json(folder/"family-deletion.json",{"removed_h":remove,"before":relation["summary"],"after":deleted["summary"],"rule":"Induced subgraph of frozen graph; no replacement neighbors"})
        summaries[phase]={**relation["summary"],"cutoff":cutoff,"nodes":len(nodes),"eligible":int(eligible.sum())}
        del matrix,neighbors,neighborhood,edges
        gc.collect()
    write_json(root/"protocol.json",{"measurement_scaler":cfg["measurement_scaler"],"metric":"equal-block RMS","k":30,
               "cutoff":cutoff,"cutoff_source":"Discovery cross-H/PDB mutual k30 distance q75, frozen for all phases",
               "filter":"Original top30, then exclude same H/PDB, no refill","scope":"Exploratory recalculation after measured rotational-invariance correction","phases":summaries})
    overlay_models(cfg,root)
    return summaries


def overlay_models(cfg,root):
    """Use only saved OOF predictions; replace all geometry with current values."""
    old=read_csv(Path(cfg["previous_outputs"])/"model-overlay/domain-summaries.csv",dtype={A:str,T:str,H:str,"target_label":str,"prediction":str})
    columns=["record_id","pdb_id",A,T,H,"task","system","target_label","eligible_test","prediction_available",
             "probability_available","prediction","top2_margin","normalized_entropy","correct","split_id"]
    old=old[columns].copy()
    architecture_geometry=read_csv(root/"all/domain-geometry.csv",dtype={"label":str,"homology":str})
    topology_geometry=read_csv(root/"all/topology-domain-geometry.csv",dtype={"label":str,"homology":str})
    geometry=pd.concat([architecture_geometry.assign(task="architecture"),
                        topology_geometry.assign(task="topology")],ignore_index=True)
    gcolumns=["record_id","packing_eligible","neighborhood_status","n_neighbors","n_neighbor_h","n_neighbor_labels",
              "cross_label_fraction","target_h_reweighted_cross_label_fraction","neighbor_distance_min","neighbor_distance_median",
              "h_phase_node_count","same_chain_pattern_fraction","chain_pattern_neighbor_count"]
    merged=old.merge(geometry[gcolumns+["task"]],on=["record_id","task"],validate="many_to_one")
    pattern=read_csv(Path(cfg["outputs"])/"connection-patterns/domain-patterns.csv",usecols=["record_id","pair_pattern","chain_pattern","pair_primary_eligible","chain_primary_eligible"])
    merged=merged.merge(pattern,on="record_id",validate="many_to_one")
    merged["margin_band"]=np.select([merged.top2_margin.le(.2),merged.top2_margin.gt(.2)],["le_0.2","gt_0.2"],default="unavailable")
    merged["mixing_band"]=pd.cut(merged.cross_label_fraction,[-.001,0,.25,.5,.75,1],labels=["zero","0_to_.25",".25_to_.5",".5_to_.75",".75_to_1"]).astype("string").fillna("unavailable")
    merged["family_support_band"]=pd.cut(merged.h_phase_node_count,[0,1,5,np.inf],labels=["one","2_to_5","gt_5"]).astype("string")
    merged.to_csv(root/"model-domain-overlay.csv.gz",index=False)
    tables=[]
    observed=merged.loc[merged.prediction_available & merged.eligible_test].copy()
    for scope,groupcols in (("all",["task","system","margin_band","mixing_band"]),
                            ("within_label",["task","system","target_label","margin_band","mixing_band"]),
                            ("within_label_support",["task","system","target_label","family_support_band","margin_band","mixing_band"]),
                            ("connection_pattern",["task","system","chain_pattern","margin_band"])):
        rows=[]
        population=observed.loc[observed.chain_primary_eligible] if scope=="connection_pattern" else observed
        for keys,g in population.groupby(groupcols,dropna=False,observed=True,sort=True):
            correct=pd.to_numeric(g.correct,errors="coerce")
            gh=g.assign(correct_numeric=correct).groupby(H).correct_numeric.mean()
            rows.append({**dict(zip(groupcols,keys if isinstance(keys,tuple) else (keys,),strict=True)),
                         "domains":len(g),"H":g[H].nunique(),"accuracy":correct.mean(),"h_equal_accuracy":gh.mean(),
                         "mean_margin":g.top2_margin.mean(),"mean_nearest_distance":g.neighbor_distance_min.mean(),
                         "mean_cross_label_fraction":g.cross_label_fraction.mean(),"mean_same_chain_pattern_fraction":g.same_chain_pattern_fraction.mean()})
        table=pd.DataFrame(rows)
        table.to_csv(root/f"model-{scope}.csv",index=False)
        tables.append(scope)
    prediction=observed.pivot(index=["record_id","task"],columns="system",values="prediction")
    disagreement=prediction.notna().sum(axis=1).rename("observed_heads").to_frame()
    disagreement["distinct_predictions"]=prediction.nunique(axis=1)
    disagreement["any_head_disagreement"]=disagreement.distinct_predictions.gt(1).where(disagreement.observed_heads.eq(3))
    disagreement=disagreement.reset_index().merge(geometry[gcolumns+["task"]],on=["record_id","task"],validate="many_to_one")
    disagreement.to_csv(root/"head-disagreement-overlay.csv",index=False)
    write_json(root/"model-protocol.json",{"prediction_source":"Saved role=test OOF, no refit/calibration", "geometry_source":"Current invariant neighborhood graph and pairing patterns",
               "estimand":"Descriptive accuracy and H-equal accuracy in measured strata; within-label/support tables retained",
               "chain_pattern_neighbor_fraction":"Both query and target require chain_primary_eligible; denominator is eligible observed neighbors",
               "connection_pattern_population":"Only chain_primary_eligible domains",
               "head_disagreement":"Three saved heads required; unavailable predictions remain NA",
               "label_scope":"Architecture outcomes use A neighborhood composition; topology outcomes use T composition on the same frozen geometry graph",
               "missing":"Unavailable neighborhood remains separate; no zero filling", "tables":tables})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    print(run(parser.parse_args().config))
