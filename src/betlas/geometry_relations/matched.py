"""Covariate-matched within-architecture contrasts, without outcome selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import A, H, T
from .io import read_csv, read_json, write_json
from .phenotypes import K_COLUMNS, P_COLUMNS, TWO_SHEET_COLUMNS


def match_metadata(left: pd.DataFrame, right: pd.DataFrame) -> list[tuple[int, int, float]]:
    """Deterministic greedy matching without replacement within one comparison.

    Only length, beta fraction, strand count, completeness and identity enter
    matching. The returned indices are positional; no outcomes enter the cost.
    """
    if left.empty or right.empty:
        return []
    length_ratio = (
        left.observed_length.to_numpy()[:, None] / right.observed_length.to_numpy()[None, :]
    )
    beta = np.abs(left.beta_fraction.to_numpy()[:, None] - right.beta_fraction.to_numpy()[None, :])
    strands = np.abs(left.strand_count.to_numpy()[:, None] - right.strand_count.to_numpy()[None, :])
    missing = np.abs(
        left.missing_fraction.to_numpy()[:, None] - right.missing_fraction.to_numpy()[None, :]
    )
    allowed = (
        (length_ratio >= 0.8)
        & (length_ratio <= 1.25)
        & (beta <= 0.15)
        & (strands <= 2)
        & (missing <= 0.05)
    )
    allowed &= left[H].to_numpy()[:, None] != right[H].to_numpy()[None, :]
    allowed &= left.pdb_id.to_numpy()[:, None] != right.pdb_id.to_numpy()[None, :]
    cost = np.abs(np.log(length_ratio)) / np.log(1.25) + beta / 0.15 + strands / 2 + missing / 0.05
    li, ri = np.where(allowed & np.isfinite(cost))
    # Input ID sorting resolves ties independently of the measured outcomes.
    order = np.lexsort((ri, li, cost[li, ri]))
    used_left, used_right, result = set(), set(), []
    for k in order:
        i, j = int(li[k]), int(ri[k])
        if i not in used_left and j not in used_right:
            result.append((i, j, float(cost[i, j])))
            used_left.add(i)
            used_right.add(j)
    return result


def run_matched(config_path: str | Path) -> dict:
    cfg = read_json(config_path)
    output = Path(cfg["outputs"]) / "matched-phenotypes"
    output.mkdir(parents=True, exist_ok=False)
    cohort = read_csv(Path(cfg["derived"]) / "cohort.csv", dtype={A: str, T: str, H: str})
    data = read_csv(
        Path(cfg["outputs"]) / "phenotypes/domain-phenotypes.csv", dtype={A: str, T: str, H: str}
    )
    metadata = cohort[["record_id", "observed_residue_count", "requested_residue_count"]].copy()
    metadata["missing_fraction"] = (
        1 - metadata.observed_residue_count / metadata.requested_residue_count
    )
    data = data.merge(
        metadata[["record_id", "missing_fraction"]], on="record_id", validate="one_to_one"
    )
    data = data.loc[data[A].eq("2.60")].sort_values("record_id")
    pairs, coverage, contrasts = [], [], []
    fields = [*P_COLUMNS, *TWO_SHEET_COLUMNS, *K_COLUMNS]
    for phase in ("all", "discovery", "confirmation"):
        scope = data if phase == "all" else data.loc[data.role.eq(phase)]
        left = scope.loc[scope[T].eq("2.60.120")]
        for target in sorted(set(data[T]) - {"2.60.120"}):
            right = scope.loc[scope[T].eq(target)]
            matched = match_metadata(left, right)
            coverage.append(
                {
                    "phase": phase,
                    "target_topology": target,
                    "j_domains": len(left),
                    "target_domains": len(right),
                    "matched_pairs": len(matched),
                    "j_h": left[H].nunique(),
                    "target_h": right[H].nunique(),
                }
            )
            measurements = []
            for i, j, cost in matched:
                left_row, right_row = left.iloc[i], right.iloc[j]
                pair_id = f"{phase}:{target}:{left_row.record_id}:{right_row.record_id}"
                pair = {
                    "pair_id": pair_id,
                    "phase": phase,
                    "target_topology": target,
                    "j_record_id": left_row.record_id,
                    "target_record_id": right_row.record_id,
                    "j_h": left_row[H],
                    "target_h": right_row[H],
                    "matching_cost": cost,
                }
                for field in (
                    "observed_length",
                    "beta_fraction",
                    "strand_count",
                    "missing_fraction",
                ):
                    pair[f"j_{field}"] = left_row[field]
                    pair[f"target_{field}"] = right_row[field]
                pairs.append(pair)
                for field in fields:
                    available = pd.notna(left_row[field]) and pd.notna(right_row[field])
                    if field in K_COLUMNS:
                        available &= bool(left_row.pairing_complete) and bool(
                            right_row.pairing_complete
                        )
                    measurements.append(
                        {
                            "pair_id": pair_id,
                            "field": field,
                            "j_h": left_row[H],
                            "target_h": right_row[H],
                            "j_value": left_row[field] if available else np.nan,
                            "target_value": right_row[field] if available else np.nan,
                            "j_minus_target": left_row[field] - right_row[field]
                            if available
                            else np.nan,
                        }
                    )
            measured = pd.DataFrame(measurements)
            if measured.empty:
                continue
            measured["phase"], measured["target_topology"] = phase, target
            for field, group in measured.groupby("field", sort=True):
                valid = group.dropna(subset=["j_minus_target"])
                h_means = valid.groupby("j_h").j_minus_target.mean()
                contrasts.append(
                    {
                        "phase": phase,
                        "target_topology": target,
                        "field": field,
                        "n_pairs": len(group),
                        "n_observed_pairs": len(valid),
                        "j_h_observed": valid.j_h.nunique(),
                        "target_h_observed": valid.target_h.nunique(),
                        "median_paired_difference": valid.j_minus_target.median(),
                        "j_h_equal_mean_difference": h_means.mean(),
                        "two_sided_multi_h_support": valid.j_h.nunique() >= 2
                        and valid.target_h.nunique() >= 2,
                    }
                )
            # One compressed aggregate per phase avoids a large small-file tree.
            phase_file = output / f"{phase}-paired-measurements.csv.gz"
            measured.to_csv(
                phase_file,
                index=False,
                mode="a",
                header=not phase_file.exists(),
                compression="gzip",
            )
    pd.DataFrame(pairs).to_csv(output / "matched-pairs.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)
    protocol = {
        "pairs_by_phase": pd.DataFrame(pairs).phase.value_counts().to_dict(),
        "selection": "Greedy minimum metadata-cost matching without replacement within each topology/phase; deterministic ID ties",
        "calipers": {
            "length_ratio": [0.8, 1.25],
            "beta_fraction_absolute": 0.15,
            "strand_count_absolute": 2,
            "missing_fraction_absolute": 0.05,
        },
        "cost": "Sum of absolute log-length ratio, beta difference, strand difference and missing difference divided by their calipers",
        "scope": "J=2.60.120 versus each other 2.60 T; all, discovery and confirmation separately; H/PDB different",
        "connectivity": "Both domains must have complete pairing for a primary K contrast; otherwise NA",
        "inference": "Descriptive conditioning on size/composition; no causal effect or independent-pair inference. J domains recur across T comparisons. Matching can condition away meaningful geometry.",
    }
    write_json(output / "protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    print(run_matched(parser.parse_args().config))
