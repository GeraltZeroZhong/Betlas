"""Rigid-motion, block-distance and representative-selection controls."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from betlas.features.extract import _axis_slice_features
from betlas.features.geometry import beta_element_geometries, build_axis_hypotheses, stable_axis
from betlas.models import DomainCandidate, ResidueRecord, SecondaryStructureElement, SheetPatch, StructureGeometry

from .cohorts import A, H

S35 = "cath_s35_cluster_id"

def geometry_from_dict(data):
    """Restore a serialized geometry without selection or annotation changes."""
    def segment(row):
        return SecondaryStructureElement(**{**row, "residue_indices": tuple(row["residue_indices"])})
    return StructureGeometry(
        domain=DomainCandidate.from_mapping(data["domain"]),
        residues=tuple(ResidueRecord(**{**row, "coord_ca": tuple(row["coord_ca"])}) for row in data["residues"]),
        beta_segments=tuple(segment(row) for row in data["beta_segments"]),
        helices=tuple(segment(row) for row in data["helices"]),
        sheet_patches=tuple(SheetPatch(row["sheet_id"], tuple(row["strand_ids"])) for row in data["sheet_patches"]),
        warnings=tuple(data.get("warnings", [])),
        sheet_connections=tuple(tuple(row) for row in data.get("sheet_connections", [])),
        secondary_structure_source=data.get("secondary_structure_source", "deposited"),
        domain_segment_count=data.get("domain_segment_count", 0),
        requested_residue_count=data.get("requested_residue_count", 0),
        missing_boundary_ca_count=data.get("missing_boundary_ca_count", 0),
    )


def rotation_matrices(count=12, seed=13):
    """Uniform SO(3) draws from normalized Gaussian quaternions; identity first."""
    if count < 1:
        raise ValueError("At least one random rotation is required")
    rng = np.random.default_rng(seed)
    result = [np.eye(3)]
    for q in rng.normal(size=(count, 4)):
        w, x, y, z = q / np.linalg.norm(q)
        result.append(np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ]))
    return result


def rotate_geometry(geometry, rotation):
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12):
        raise ValueError("A proper orthogonal rotation is required")
    if not np.isclose(np.linalg.det(rotation), 1., atol=1e-12):
        raise ValueError("Reflections are not rigid rotation controls")
    coordinates = np.array([row.coord_ca for row in geometry.residues])
    center = coordinates.mean(axis=0)
    transformed = (coordinates - center) @ rotation.T + center
    residues = tuple(replace(row, coord_ca=tuple(xyz)) for row, xyz in zip(geometry.residues, transformed, strict=True))
    return replace(geometry, residues=residues)


def axis_candidates(geometry):
    """Evaluate exactly the existing candidate set and its frozen score formula."""
    elements = beta_element_geometries(geometry)
    points = np.concatenate([item.coords for item in elements]) if elements else np.empty((0, 3))
    midpoints = np.array([item.midpoint for item in elements]) if elements else np.empty((0, 3))
    rows = []
    for candidate in build_axis_hypotheses(geometry):
        direction = stable_axis(np.asarray(candidate.direction))
        features = _axis_slice_features(points, midpoints, np.asarray(candidate.origin), direction)
        score = (.40 * features["slice_coverage_median"] + .25 * features["angular_coverage"]
                 + .20 * features["z_continuity_fraction"] + .15 * (1 - features["largest_gap_fraction"]))
        rows.append({"axis_name": candidate.name, "score": float(score),
                     "direction_x": direction[0], "direction_y": direction[1], "direction_z": direction[2],
                     **features})
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["score_rank"] = rank
    return rows


def scaled_prepared(frame, scaler):
    """Scale fields already applicability-masked by the previous run.

    Natural tables omit applicability metadata. Calling scaler.transform there
    would wrongly mask valid fields a second time. The original masks are kept.
    """
    columns = [field for fields in scaler.blocks.values() for field in fields if field in scaler.scale]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = (values - np.array([scaler.center[field] for field in columns])) / np.array([scaler.scale[field] for field in columns])
    blocks = {name: [columns.index(field) for field in fields if field in columns] for name, fields in scaler.blocks.items()}
    return values.astype(np.float32), {name: indices for name, indices in blocks.items() if indices}


def block_decomposition(left, right, blocks):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    delta = left - right
    result = {}
    for name, indices in blocks.items():
        squared = np.mean(delta[:, indices] ** 2, axis=1)
        result[name + "_rms"] = np.sqrt(squared)
        result[name + "_squared_contribution"] = squared / len(blocks)
    total = np.sum([result[name + "_squared_contribution"] for name in blocks], axis=0)
    result["packing_distance"] = np.sqrt(total)
    for name in blocks:
        result[name + "_fraction_of_squared_distance"] = np.divide(
            result[name + "_squared_contribution"], total,
            out=np.full(len(total), np.nan), where=np.isfinite(total) & (total > 0))
    return pd.DataFrame(result)


def weighted_quantile(values, weights, quantiles):
    """Interpolate weighted empirical midpoint CDF; endpoints clamp to extrema."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values[valid], kind="stable")
    values, weights = values[valid][order], weights[valid][order]
    cdf = (np.cumsum(weights) - weights / 2) / weights.sum()
    return np.interp(quantiles, cdf, values)


def weighted_summaries(frame, columns):
    rows = []
    for scheme, group in (("domain_equal", None), ("s35_equal", S35), ("h_equal", H)):
        for column in columns:
            selected = frame.loc[np.isfinite(frame[column])]
            weights = np.ones(len(selected)) if group is None else 1 / selected.groupby(group)[group].transform("size").to_numpy()
            values = selected[column].to_numpy(float)
            quantiles = weighted_quantile(values, weights, [.1, .5, .9, .99])
            rows.append({"weighting": scheme, "measure": column, "domains": len(selected),
                         "s35_clusters": selected[S35].nunique(), "h_groups": selected[H].nunique(),
                         "mean": np.average(values, weights=weights) if len(values) else np.nan,
                         "q10": quantiles[0], "median": quantiles[1], "q90": quantiles[2], "q99": quantiles[3]})
    return pd.DataFrame(rows)


def representative_controls(frame, values, representatives, rep_values, blocks, *, seed=13, max_clusters_per_a=2,
                            auxiliary_blocks=None):
    """Compare fixed representatives on held-out members of bounded clusters."""
    rng = np.random.default_rng(seed)
    valid = frame.loc[frame.packing_distance.notna() & ~frame.self_reference].copy()
    sizes = valid.groupby([A, S35]).size().reset_index(name="nonself_members")
    choices, coverage = [], []
    for architecture in sorted(frame[A].unique()):
        pool = sizes.loc[sizes[A].eq(architecture) & sizes.nonself_members.ge(8)].sort_values(S35)
        indices = rng.permutation(len(pool))[:max_clusters_per_a]
        chosen = pool.iloc[indices]
        coverage.append({A: architecture, "eligible_clusters": len(pool), "selected_clusters": len(chosen)})
        choices.extend(chosen[S35])
    rep_indices = {key: i for i, key in enumerate(representatives[S35])}
    pair_rows, manifests = [], []
    for cluster in choices:
        subset = valid.loc[valid[S35].eq(cluster)].sort_values("record_id")
        selected = rng.permutation(subset.index.to_numpy())[:64]
        training, evaluation = selected[:len(selected)//2], selected[len(selected)//2:]
        candidate_members = rng.permutation(training)[:7]
        candidate_ids = [representatives.iloc[rep_indices[cluster]].record_id] + frame.loc[candidate_members, "record_id"].tolist()
        candidate_values = np.vstack([rep_values[rep_indices[cluster]], values[candidate_members]])
        candidate_training = []
        for candidate in candidate_values:
            candidate_training.append(block_decomposition(values[training], np.broadcast_to(candidate, values[training].shape), blocks).packing_distance.mean())
        medoid_index = int(np.argmin(candidate_training))
        random_index = int(rng.integers(1, len(candidate_values)))
        methods = {"original": 0, "random_training_member": random_index, "training_candidate_medoid": medoid_index}
        for method, candidate_index in methods.items():
            distances = block_decomposition(values[evaluation], np.broadcast_to(candidate_values[candidate_index], values[evaluation].shape), blocks)
            auxiliary = (block_decomposition(values[evaluation], np.broadcast_to(candidate_values[candidate_index], values[evaluation].shape), auxiliary_blocks)
                         if auxiliary_blocks else None)
            for position, member_index in enumerate(evaluation):
                pair_rows.append({A: frame.at[member_index, A], H: frame.at[member_index, H], S35: cluster,
                                  "record_id": frame.at[member_index, "record_id"], "method": method,
                                  "candidate_id": candidate_ids[candidate_index],
                                  "packing_distance": float(distances.packing_distance.iloc[position]),
                                  "auxiliary_block_distance": float(auxiliary.packing_distance.iloc[position]) if auxiliary is not None else np.nan})
        manifests.append({S35: cluster, "training_ids": frame.loc[training, "record_id"].tolist(),
                          "evaluation_ids": frame.loc[evaluation, "record_id"].tolist(),
                          "candidate_ids": candidate_ids, "training_mean_distances": list(map(float, candidate_training)),
                          "selected_medoid_id": candidate_ids[medoid_index], "random_candidate_id": candidate_ids[random_index]})
    return pd.DataFrame(pair_rows), manifests, pd.DataFrame(coverage)


