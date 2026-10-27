"""Metadata-only cohort allocation for geometry discovery and confirmation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

A = "cath_architecture_code"
T = "cath_topology_code"
H = "cath_homology_code"


def category(architecture: str, topology: str, *, legacy: bool = False) -> str:
    if topology == "2.60.120":
        return "jelly_roll"
    if architecture == "2.40":
        return "barrel"
    if architecture == "2.60" or (legacy and architecture in {"2.70", "2.102"}):
        return "other_sandwich"
    return "other_architecture"


def engineering_panel(frame: pd.DataFrame, *, seed: int = 13, per_group: int = 12) -> pd.DataFrame:
    """Select a bounded measurement panel without model scores or geometry distances."""
    work = frame.loc[frame.feature_status.eq("ok")].copy()
    work["panel_group"] = [category(a, t) for a, t in zip(work[A], work[T], strict=True)]
    rng = np.random.default_rng(seed)
    work["selection_tiebreak"] = rng.random(len(work))
    segments = work.segments_json.map(lambda x: len(json.loads(x)))
    clipped = work.betlas_ladder_pairing_status.eq("partial_clipped_ladders")
    missing = pd.to_numeric(work.missing_boundary_ca_count, errors="coerce").fillna(0).gt(0)
    work["engineering_priority"] = (
        segments.gt(1).astype(int) + clipped.astype(int) + missing.astype(int)
    )
    used_h, used_pdb, selected = set(), set(), []
    for group in ("barrel", "jelly_roll", "other_sandwich", "other_architecture"):
        pool = work.loc[work.panel_group.eq(group)].sort_values(
            ["engineering_priority", "selection_tiebreak"], ascending=[False, True]
        )
        seen_a, chosen = set(), 0
        # Prefer architecture diversity in the outside control group.
        for diversity in (True, False):
            for idx, row in pool.iterrows():
                if chosen >= per_group:
                    break
                if row[H] in used_h or row.pdb_id in used_pdb:
                    continue
                if group == "other_architecture" and diversity and row[A] in seen_a:
                    continue
                selected.append(idx)
                used_h.add(row[H])
                used_pdb.add(row.pdb_id)
                seen_a.add(row[A])
                chosen += 1
    return work.loc[
        selected, ["record_id", "pdb_id", A, T, H, "panel_group", "engineering_priority"]
    ].reset_index(drop=True)


def allocate_roles(
    frame: pd.DataFrame, mother: pd.DataFrame, panel: pd.DataFrame, *, seed: int = 13
) -> tuple[pd.DataFrame, dict]:
    """Allocate whole H groups; expose direct-PDB purge and broad dependence."""
    rng = np.random.default_rng(seed)
    engineering_h = set(panel[H])
    h_meta = frame[[A, T, H]].drop_duplicates().sort_values([A, T, H])
    roles = {}
    for _topology, group in h_meta.groupby(T, sort=True):
        hs = sorted(group[H].tolist())
        if len(hs) == 1:
            roles[hs[0]] = "discovery" if hs[0] in engineering_h else "descriptive_single_h"
            continue
        remaining = np.array([h for h in hs if h not in engineering_h], dtype=object)
        rng.shuffle(remaining)
        count = min(len(remaining), max(1, int(round(len(hs) / 3))))
        confirmation = set(remaining[:count])
        roles.update({h: "confirmation" if h in confirmation else "discovery" for h in hs})
    out = frame[["record_id", "pdb_id", A, T, H, "feature_status"]].copy()
    out["planned_role"] = out[H].map(roles)
    discovery_h = {h for h, role in roles.items() if role == "discovery"}
    discovery_pdb = set(mother.loc[mother[H].isin(discovery_h), "pdb_id"])
    out["shared_pdb_with_discovery_mother"] = out.pdb_id.isin(discovery_pdb)
    out["role"] = out.planned_role
    purge = out.planned_role.eq("confirmation") & out.shared_pdb_with_discovery_mother
    out.loc[purge, "role"] = "confirmation_pdb_purged"
    out["engineering_h"] = out[H].isin(engineering_h)
    out["engineering_record"] = out.record_id.isin(panel.record_id)
    # Full-mother H/PDB components reveal dependence that direct purging misses.
    parent = {h: h for h in mother[H].unique()}

    def find(h):
        while parent[h] != h:
            parent[h] = parent[parent[h]]
            h = parent[h]
        return h

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for _, group in mother[["pdb_id", H]].drop_duplicates().groupby("pdb_id", sort=False):
        hs = list(group[H])
        for h in hs[1:]:
            union(hs[0], h)
    out["mother_h_pdb_component"] = out[H].map(lambda h: find(h))
    counts = out.mother_h_pdb_component.value_counts()
    summary = {
        "seed": seed,
        "allocation": "approximately two discovery H per confirmation H, stratified by T",
        "purge": "confirmation representative PDB occurs in any full-mother domain of a discovery H",
        "roles": out.role.value_counts().to_dict(),
        "engineering_domains": len(panel),
        "engineering_h": len(engineering_h),
        "full_mother_components": int(len(counts)),
        "largest_component_domains": int(counts.iloc[0]),
        "largest_component_fraction": float(counts.iloc[0] / len(out)),
        "history": "Existing data have been used previously; this is prospective analysis allocation, not unseen external validation.",
    }
    return out, summary
