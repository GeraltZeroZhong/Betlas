"""Freeze discovery candidates and select a bounded structural comparison panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json


def freeze_candidates(config_path):
    cfg = read_json(config_path)
    derived, out = Path(cfg["derived"]), Path(cfg["outputs"])
    root = out / "atlas/discovery"
    nodes = read_csv(root / "rms/nodes.csv", dtype={A: str, H: str})
    eligible_nodes = nodes.loc[nodes.packing_eligible.eq(True)]
    h_prevalence = eligible_nodes.groupby(A)[H].nunique() / eligible_nodes[H].nunique()
    primary = read_csv(
        root / "rms/relations-k30/architecture-directed.csv",
        dtype={"source_label": str, "target_label": str},
    )
    variants = [
        "rms",
        "l1",
        "drop_global_shape",
        "drop_sheet_shape",
        "drop_axis_organization",
        "drop_angular_repetition",
    ]
    present = {}
    for variant in variants:
        rel = read_csv(
            root / variant / "relations-k30/architecture-directed.csv",
            dtype={"source_label": str, "target_label": str},
        )
        present[variant] = set(
            tuple(sorted((r.source_label, r.target_label)))
            for r in rel.itertuples()
            if r.source_label != r.target_label
            and r.n_support_source_h >= 2
            and r.n_support_target_h >= 2
        )
    records = []
    lookup = primary.set_index(["source_label", "target_label"])
    for row in primary.itertuples():
        a, b = row.source_label, row.target_label
        if a >= b:
            continue
        reverse = lookup.loc[(b, a)]
        stability = sum((a, b) in supported for supported in present.values())
        # This rank corrects target opportunity descriptively; it is not a null-model test.
        a_lift = row.target_h_reweighted_source_h_equal_fraction / h_prevalence.get(b, np.nan)
        b_lift = reverse.target_h_reweighted_source_h_equal_fraction / h_prevalence.get(a, np.nan)
        score = min(a_lift, b_lift) if np.isfinite([a_lift, b_lift]).all() else np.nan
        feasible = (
            row.n_directed_edges >= 10
            and min(row.n_support_source_h, row.n_support_target_h) >= 2
            and stability >= 3
        )

        def count(value):
            return int(value) if pd.notna(value) else None

        records.append(
            {
                "a": a,
                "b": b,
                "n_directed_edges": count(row.n_directed_edges),
                "support_h_a": count(row.n_support_source_h),
                "support_h_b": count(row.n_support_target_h),
                "variants_supported": stability,
                "descriptive_min_lift": score,
                "eligible_for_panel": feasible,
                "historical_architecture_pair": set((a, b)) == {"2.40", "2.60"},
            }
        )
    table = pd.DataFrame(records).sort_values(
        ["variants_supported", "descriptive_min_lift", "n_directed_edges"], ascending=False
    )
    chosen = table.loc[table.eligible_for_panel & ~table.historical_architecture_pair].head(3)
    table["selected"] = table.index.isin(chosen.index)
    table.to_csv(derived / "discovery-candidate-ranking.csv", index=False)
    frozen = {
        "schema": "betlas.geometry-candidates.v1",
        "source_phase": "discovery_only",
        "criteria": "At least 10 directed edges, >=2 supporting H on each side, >=3 of 6 fixed views; ranks are descriptive, not significance certificates",
        "ranking": "view stability, then minimum bidirectional target-H-reweighted fraction divided by eligible target-H prevalence, then edge count",
        "maximum_new_pairs": 3,
        "new_architecture_pairs": chosen.to_dict("records"),
        "historical_strict_pairs": [
            ["barrel", "jelly_roll"],
            ["barrel", "other_sandwich"],
            ["jelly_roll", "other_sandwich"],
        ],
    }
    write_json(derived / "frozen-candidates.json", frozen)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            select_validation_panel(args.config) if args.panel else freeze_candidates(args.config),
            indent=2,
        ),
        flush=True,
    )


def select_validation_panel(config_path):
    cfg = read_json(config_path)
    derived, outputs = Path(cfg["derived"]), Path(cfg["outputs"])
    frame = read_csv(derived / "cohort.csv", dtype=str, keep_default_na=False).set_index(
        "record_id", drop=False
    )
    manifest = read_csv(
        Path(cfg["source_derived"]) / "domain-structures/domain_structures.csv",
        dtype=str,
        keep_default_na=False,
    )
    allowed_ids = set(manifest.loc[manifest.export_status.eq("ok"), "record_id"])
    allowed_ids &= set(
        frame.loc[
            frame.domain_chain_count.eq("1")
            & frame.feature_status.eq("ok")
            & frame.engineering_h.eq("False"),
            "record_id",
        ]
    )
    frozen = read_json(derived / "frozen-candidates.json")
    themes = [
        ("historical_" + a + "--" + b, "strict_label", a, b)
        for a, b in frozen["historical_strict_pairs"]
    ]
    themes += [
        ("architecture_" + item["a"] + "--" + item["b"], A, item["a"], item["b"])
        for item in frozen["new_architecture_pairs"]
    ]
    rng = np.random.default_rng(13)
    selected, used_pairs, panel_ids, themes_summary = [], set(), set(), []
    numeric = pd.DataFrame(index=frame.index)
    for column in (
        "observed_residue_count",
        "betlas_beta_residue_fraction",
        "betlas_beta_strand_count",
        "cath_resolution",
        "requested_residue_count",
    ):
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric["missing_fraction"] = (
        1 - numeric.observed_residue_count / numeric.requested_residue_count
    )

    def add_pair(q, t, role, theme, phase, matched="", distance=np.nan):
        key = tuple(sorted((q, t)))
        if key in used_pairs or q == t:
            return False
        used_pairs.add(key)
        panel_ids.update(key)
        selected.append(
            {
                "pair_id": f"pair_{len(selected) + 1:04d}",
                "query_id": q,
                "target_id": t,
                "role": role,
                "relation_kind": theme,
                "phase": phase,
                "matched_pair_id": matched,
                "packing_distance": distance,
                "queryH": frame.at[q, H],
                "targetH": frame.at[t, H],
                "queryA": frame.at[q, A],
                "targetA": frame.at[t, A],
                "queryT": frame.at[q, T],
                "targetT": frame.at[t, T],
            }
        )
        return True

    for theme, column, left, right in themes:
        chosen, used_query_h, confirmation_count = 0, set(), 0
        for phase in ("confirmation", "discovery"):
            edges = read_csv(outputs / "atlas" / phase / "rms/edges-k30.csv.gz")
            edges = edges.loc[edges.mutual.eq(True) & edges.short_edge.eq(True)].sort_values(
                "distance"
            )
            for edge in edges.itertuples():
                if chosen >= 6:
                    break
                q, t = edge.source_id, edge.target_id
                if (
                    q not in allowed_ids
                    or t not in allowed_ids
                    or frame.at[q, column] != left
                    or frame.at[t, column] != right
                ):
                    continue
                if frame.at[q, H] in used_query_h or tuple(sorted((q, t))) in used_pairs:
                    continue
                # Match the target's size, beta fraction, completeness and label;
                # do not require the control to be geometrically far.
                candidates = frame.loc[
                    frame.role.eq(phase) & frame[column].eq(right) & frame.index.isin(allowed_ids)
                ]
                candidates = candidates.loc[
                    ~candidates[H].isin([frame.at[q, H], frame.at[t, H]])
                    & ~candidates.pdb_id.isin([frame.at[q, "pdb_id"], frame.at[t, "pdb_id"]])
                ]
                ratios = (
                    numeric.loc[candidates.index, "observed_residue_count"]
                    / numeric.at[t, "observed_residue_count"]
                )
                beta_delta = (
                    numeric.loc[candidates.index, "betlas_beta_residue_fraction"]
                    - numeric.at[t, "betlas_beta_residue_fraction"]
                ).abs()
                missing_delta = (
                    numeric.loc[candidates.index, "missing_fraction"]
                    - numeric.at[t, "missing_fraction"]
                ).abs()
                candidates = candidates.loc[
                    ratios.between(0.8, 1.25) & beta_delta.le(0.15) & missing_delta.le(0.05)
                ]
                if candidates.empty:
                    continue
                ids = candidates.index.to_numpy().copy()
                rng.shuffle(ids)
                control = next(
                    (
                        c
                        for c in ids
                        if tuple(sorted((q, c))) not in used_pairs
                        and len(panel_ids | {q, t, c}) <= 120
                    ),
                    None,
                )
                if control is None:
                    continue
                match = f"match_{len(selected) // 2 + 1:03d}"
                add_pair(q, t, "candidate", theme, phase, match, float(edge.distance))
                add_pair(q, control, "matched_control", theme, phase, match)
                used_query_h.add(frame.at[q, H])
                chosen += 1
                confirmation_count += phase == "confirmation"
        themes_summary.append(
            {
                "theme": theme,
                "matched_sets": chosen,
                "confirmation_sets": confirmation_count,
                "discovery_only_sets": chosen - confirmation_count,
            }
        )
    # Ensure a usable blinded panel even when few candidate relations are supported.
    pool = frame.loc[frame.role.eq("confirmation") & frame.index.isin(allowed_ids)].copy()
    pool["draw"] = rng.random(len(pool))
    for _a, group in pool.sort_values("draw").groupby(A, sort=True):
        for record in group.head(5).record_id:
            if len(panel_ids) < 80:
                panel_ids.add(record)
    if len(panel_ids) < 80:
        for record in pool.sort_values("draw").record_id:
            panel_ids.add(record)
            if len(panel_ids) >= 80:
                break
    ids = sorted(panel_ids)
    backgrounds = [
        (q, t)
        for i, q in enumerate(ids)
        for t in ids[i + 1 :]
        if frame.at[q, H] != frame.at[t, H] and frame.at[q, "pdb_id"] != frame.at[t, "pdb_id"]
    ]
    rng.shuffle(backgrounds)
    for q, t in backgrounds:
        if len(selected) >= 200:
            break
        add_pair(q, t, "panel_background", "within_selected_panel", "mixed_selected_panel")
    pairs = pd.DataFrame(selected)
    pairs.to_csv(derived / "validation-pairs.csv", index=False)
    frame.loc[sorted(set(pairs.query_id) | set(pairs.target_id))].to_csv(
        derived / "validation-domains.csv", index=False
    )
    result = {
        "pairs": len(pairs),
        "domains": len(set(pairs.query_id) | set(pairs.target_id)),
        "themes": themes_summary,
        "seed": 13,
        "candidate_rule": "Up to six matched sets per frozen theme, confirmation first; discovery fallback explicitly exploratory",
        "matching": "Same target label/phase; different H/PDB; length ratio [0.8,1.25], beta fraction difference <=0.15, missing fraction difference <=0.05; no geometric-farness requirement",
        "background": "Random cross-H/PDB pairs within the selected panel; not a population prevalence sample",
        "human_review": "not performed",
    }
    write_json(derived / "validation-panel-protocol.json", result)
    return result


if __name__ == "__main__":
    main()
