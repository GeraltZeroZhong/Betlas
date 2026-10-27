"""Label-free packing distances with explicit measurement availability.

The default contract selects existing, directly interpretable measurements.  It
does not measure curvature, define a biological boundary, or use class labels.
The selected axis is the existing geometry-optimized axis, not an unbiased axis.
Missing measurements are never imputed.  A distance requires complete active
measurements in every included block; removing a block can restore coverage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PACKING_BLOCKS: dict[str, tuple[str, ...]] = {
    "global_shape": ("betlas_pca_elongation", "betlas_pca_flatness"),
    "sheet_shape": (
        "betlas_sheet_planarity_residual_mean",
        "betlas_sheet_axis_dispersion_mean",
        "betlas_sheet_angular_span_mean",
    ),
    "axis_organization": (
        "betlas_axis_best_angular_coverage",
        "betlas_axis_best_radius_cv",
        "betlas_z_continuity_fraction",
        "betlas_axis_best_slice_coverage_median",
        "betlas_axis_best_slice_coverage_std",
    ),
    "angular_repetition": tuple(f"betlas_angular_circular_moment_k{k}" for k in range(1, 9)),
}


def packing_feature_contract() -> dict[str, Any]:
    """Return the fixed default measurement and applicability specification."""
    return {
        "schema": "betlas.packing-contract.v2",
        "blocks": {key: list(value) for key, value in PACKING_BLOCKS.items()},
        "weights": "Equal block weights; equal active feature weights within each block.",
        "rms": "sqrt(mean_blocks(mean_active_features((scaled_i-scaled_j)^2)))",
        "l1": "mean_blocks(mean_active_features(abs(scaled_i-scaled_j)))",
        "missing": "No imputation; every active field in included blocks must be finite.",
        "applicability": {
            "all": "betlas_parse_ok == 1 when provided",
            "betlas_sheet_planarity_residual_mean": "betlas_beta_residue_count >= 3",
            "betlas_sheet_axis_dispersion_mean": "betlas_sheet_face_count >= 1",
            "slice_coverage_median_and_std": "betlas_axis_best_slice_count >= 1",
            "circular_moments_k1_to_k8": "betlas_beta_strand_count >= 4",
        },
        "metadata_missing": "Missing applicability metadata makes its dependent fields unavailable.",
        "interpretation": (
            "Sheet planarity is a PCA residual, not curvature. Axis measurements use the "
            "existing optimized axis. Sheet identity depends on the input annotation."
        ),
    }


def prepare_packing_features(
    frame: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]] = PACKING_BLOCKS,
    *,
    apply_applicability: bool = True,
) -> pd.DataFrame:
    """Select only contract fields and mask unsupported extractor placeholders.

    Additional metadata, identifiers, reference labels, and predictions are not
    selected. Applicability predicates apply only to the known physical fields;
    custom fields must already encode unavailable values as NaN.
    """
    columns = [column for fields in blocks.values() for column in fields]
    if len(set(columns)) != len(columns):
        raise ValueError("A packing feature must belong to exactly one physical block")
    values = pd.DataFrame(index=frame.index)
    for column in columns:
        source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        values[column] = pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not apply_applicability:
        return values

    def require(fields: Sequence[str], metadata: str, minimum: float) -> None:
        selected = [field for field in fields if field in values]
        if not selected:
            return
        available = (
            pd.to_numeric(frame[metadata], errors="coerce").ge(minimum)
            if metadata in frame else pd.Series(False, index=frame.index)
        )
        values.loc[~available, selected] = np.nan

    require(("betlas_sheet_planarity_residual_mean",), "betlas_beta_residue_count", 3)
    require(("betlas_sheet_axis_dispersion_mean",), "betlas_sheet_face_count", 1)
    require(
        ("betlas_axis_best_slice_coverage_median", "betlas_axis_best_slice_coverage_std"),
        "betlas_axis_best_slice_count", 1,
    )
    require(tuple(f"betlas_angular_circular_moment_k{k}" for k in (1, 2)), "betlas_beta_strand_count", 4)
    require(tuple(f"betlas_angular_circular_moment_k{k}" for k in range(3, 9)), "betlas_beta_strand_count", 4)
    if "betlas_parse_ok" in frame:
        values.loc[pd.to_numeric(frame["betlas_parse_ok"], errors="coerce").ne(1), :] = np.nan
    return values


@dataclass
class PackingMatrix:
    """Scaled fields and masks in input row order; no labels are stored."""

    values: np.ndarray
    columns: tuple[str, ...]
    blocks: dict[str, tuple[int, ...]]

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.values)

    def block_availability(self) -> dict[str, np.ndarray]:
        return {name: self.observed[:, indices].all(axis=1) for name, indices in self.blocks.items()}

    def availability_frame(self) -> pd.DataFrame:
        result = pd.DataFrame(self.block_availability())
        result["observed_active_fields"] = self.observed.sum(axis=1)
        result["active_fields"] = len(self.columns)
        return result


@dataclass
class PackingScaler:
    """Discovery-only median/IQR scaling, preserving all missing values.

    A zero IQR falls back to the discovery population standard deviation; this
    exception is recorded per field. Constant, empty, or single-observation
    fields are excluded using discovery data alone. No confirmation data alter
    this choice. This is an explicit fit/transform interface, not a global fit.
    """

    blocks: dict[str, tuple[str, ...]]
    center: dict[str, float]
    scale: dict[str, float]
    field_status: dict[str, dict[str, Any]]
    apply_applicability: bool = True

    @classmethod
    def fit(
        cls,
        discovery: pd.DataFrame,
        blocks: Mapping[str, Sequence[str]] = PACKING_BLOCKS,
        *,
        apply_applicability: bool = True,
    ) -> PackingScaler:
        normalized = {str(name): tuple(fields) for name, fields in blocks.items()}
        if not normalized or any(not fields for fields in normalized.values()):
            raise ValueError("Packing requires nonempty physical blocks")
        frame = prepare_packing_features(discovery, normalized, apply_applicability=apply_applicability)
        centers: dict[str, float] = {}
        scales: dict[str, float] = {}
        status: dict[str, dict[str, Any]] = {}
        for column in frame:
            finite = frame[column].dropna().to_numpy(dtype=float)
            entry: dict[str, Any] = {"n_discovery_finite": len(finite)}
            if len(finite) < 2:
                entry["status"] = "excluded_insufficient_discovery_observations"
            elif np.ptp(finite) == 0:
                entry["status"] = "excluded_constant_in_discovery"
            else:
                q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
                scale = float(q75 - q25)
                entry["status"] = "active_iqr"
                if scale <= 0:
                    scale = float(np.std(finite, ddof=0))
                    entry["status"] = "active_zero_iqr_std_fallback"
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Non-finite discovery scale for {column}")
                centers[column] = float(median)
                scales[column] = scale
                entry.update(center=float(median), scale=scale)
            status[column] = entry
        if not scales:
            raise ValueError("No variable packing measurements in discovery data")
        return cls(normalized, centers, scales, status, apply_applicability)

    def transform(self, frame: pd.DataFrame) -> PackingMatrix:
        selected = prepare_packing_features(frame, self.blocks, apply_applicability=self.apply_applicability)
        columns = tuple(column for fields in self.blocks.values() for column in fields if column in self.scale)
        values = selected.loc[:, columns].to_numpy(dtype=float)
        values = (values - np.array([self.center[column] for column in columns])) / np.array(
            [self.scale[column] for column in columns]
        )
        positions = {column: index for index, column in enumerate(columns)}
        blocks = {
            name: tuple(positions[column] for column in fields if column in positions)
            for name, fields in self.blocks.items()
        }
        blocks = {name: indices for name, indices in blocks.items() if indices}
        return PackingMatrix(values.astype(np.float32), columns, blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "betlas.packing-scaler.v1",
            "blocks": {name: list(fields) for name, fields in self.blocks.items()},
            "center": dict(self.center), "scale": dict(self.scale),
            "field_status": self.field_status,
            "apply_applicability": self.apply_applicability,
            "excluded_blocks": [
                name for name, fields in self.blocks.items() if not any(field in self.scale for field in fields)
            ],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> PackingScaler:
        if state.get("schema") != "betlas.packing-scaler.v1":
            raise ValueError("Unsupported packing scaler schema")
        return cls(
            {name: tuple(fields) for name, fields in state["blocks"].items()},
            dict(state["center"]), dict(state["scale"]), dict(state["field_status"]),
            bool(state["apply_applicability"]),
        )


@dataclass
class BlockDistanceCache:
    """Mean feature differences per block, reusable across block removal views."""

    distances: dict[str, dict[str, np.ndarray]]

    def combine(self, *, metric: str = "rms", drop_blocks: Sequence[str] = ()) -> np.ndarray:
        if metric not in self.distances:
            raise ValueError(f"Metric {metric!r} has not been computed")
        matrices = self.distances[metric]
        unknown = set(drop_blocks) - set(matrices)
        if unknown:
            raise ValueError(f"Unknown blocks: {sorted(unknown)}")
        retained = [matrix for name, matrix in matrices.items() if name not in drop_blocks]
        if not retained:
            raise ValueError("At least one active physical block must remain")
        combined = np.zeros_like(retained[0])
        for matrix in retained:
            combined += matrix
        combined /= len(retained)
        if metric == "rms":
            np.sqrt(combined, out=combined)
        return combined


def pairwise_block_distances(
    matrix: PackingMatrix,
    *,
    metrics: Sequence[str] = ("rms", "l1"),
    block_size: int = 256,
) -> BlockDistanceCache:
    """Compute float32 symmetric distances in bounded row blocks.

    RMS caches squared distances; the outer square root is applied only after
    averaging physical blocks. Invalid pairs remain infinity, including invalid
    diagonals. No pairwise deletion of fields or missing-value similarity occurs.
    For 8,232 rows each cached physical-block/metric matrix uses about 271 MB.
    Call with one metric at a time if retaining both exceeds the memory budget.
    """
    if block_size < 1 or not metrics or set(metrics) - {"rms", "l1"}:
        raise ValueError("Use a positive block_size and rms and/or l1 metrics")
    n = len(matrix.values)
    output: dict[str, dict[str, np.ndarray]] = {metric: {} for metric in dict.fromkeys(metrics)}
    for name, indices in matrix.blocks.items():
        values = matrix.values[:, indices].astype(float)
        valid = np.isfinite(values).all(axis=1)
        safe_values = np.where(np.isfinite(values), values, 0.0)
        for metric in output:
            result = np.full((n, n), np.inf, dtype=np.float32)
            native_metric = "sqeuclidean" if metric == "rms" else "cityblock"
            for start in range(0, n, block_size):
                stop = min(start + block_size, n)
                distances = cdist(safe_values[start:stop], safe_values, metric=native_metric)
                distances /= len(indices)
                distances[~valid[start:stop], :] = np.inf
                distances[:, ~valid] = np.inf
                result[start:stop] = distances
            output[metric][name] = result
    return BlockDistanceCache(output)


@dataclass
class PackingNeighborhood:
    edges: pd.DataFrame
    coverage: pd.DataFrame
    excluded: pd.DataFrame


@dataclass
class PackingNeighbors:
    indices: np.ndarray
    distances: np.ndarray
    finite_candidate_counts: np.ndarray

    def neighborhood(
        self,
        k: int,
        *,
        group_ids: Sequence[Any] | None = None,
        pdb_ids: Sequence[Any] | None = None,
    ) -> PackingNeighborhood:
        """Filter each *original* top-k set without replacing excluded neighbors.

        Mutual means both filtered directed edges exist at this same k. Local
        scale is the last available original top-k distance, before dependency
        filtering; retained radius and both coverage denominators are separate.
        IDs never affect distances. Unknown dependency IDs do not match.
        """
        if k < 1 or k > self.indices.shape[1]:
            raise ValueError("k must be between 1 and the cached maximum k")
        n = len(self.indices)

        def keys(items: Sequence[Any] | None, *, lower: bool = False) -> list[str]:
            if items is None:
                return [""] * n
            if len(items) != n:
                raise ValueError("Dependency metadata must match the number of rows")
            result = []
            for item in items:
                value = "" if item is None or pd.isna(item) else str(item).strip()
                result.append(value.lower() if lower else value)
            return result

        groups, pdbs = keys(group_ids), keys(pdb_ids, lower=True)
        retained: list[list[tuple[int, int, float]]] = [[] for _ in range(n)]
        excluded: list[dict[str, Any]] = []
        base_count = np.sum(self.indices[:, :k] >= 0, axis=1)
        scales = np.full(n, np.nan)
        for source in range(n):
            if base_count[source]:
                scales[source] = self.distances[source, base_count[source] - 1]
            for rank, target in enumerate(self.indices[source, :k], start=1):
                if target < 0:
                    continue
                distance = float(self.distances[source, rank - 1])
                same_group = bool(groups[source] and groups[source] == groups[target])
                same_pdb = bool(pdbs[source] and pdbs[source] == pdbs[target])
                if same_group or same_pdb:
                    excluded.append({
                        "source": source, "target": int(target), "rank": rank,
                        "distance": distance, "same_group": same_group, "same_pdb": same_pdb,
                    })
                else:
                    retained[source].append((int(target), rank, distance))
        membership = [{target for target, _rank, _distance in row} for row in retained]
        edges = []
        coverage = []
        for source, row in enumerate(retained):
            mutual_count = 0
            for target, rank, distance in row:
                mutual = source in membership[target]
                mutual_count += int(mutual)
                denominator = np.sqrt(scales[source] * scales[target])
                edges.append({
                    "source": source, "target": target, "rank": rank, "k": k,
                    "distance": distance, "mutual": mutual,
                    "source_local_scale": scales[source], "target_local_scale": scales[target],
                    "relative_distance": distance / denominator if denominator > 0 else np.nan,
                })
            coverage.append({
                "row": source, "k": k, "finite_candidates": int(self.finite_candidate_counts[source]),
                "base_neighbors": int(base_count[source]), "retained_neighbors": len(row),
                "mutual_neighbors": mutual_count, "local_scale": scales[source],
                "retained_radius": max((item[2] for item in row), default=np.nan),
                "coverage_requested": len(row) / k,
                "coverage_available": len(row) / base_count[source] if base_count[source] else np.nan,
            })
        return PackingNeighborhood(
            pd.DataFrame(edges, columns=[
                "source", "target", "rank", "k", "distance", "mutual",
                "source_local_scale", "target_local_scale", "relative_distance",
            ]),
            pd.DataFrame(coverage),
            pd.DataFrame(excluded, columns=["source", "target", "rank", "distance", "same_group", "same_pdb"]),
        )


def knn_from_distances(distances: np.ndarray, *, max_k: int = 60) -> PackingNeighbors:
    """Cache finite nonself top-k neighbors with input-position tie breaking.

    Dependency filtering happens later and cannot reach beyond each requested
    original k. The input matrix is never mutated. Absent neighbors use -1/inf.
    """
    distances = np.asarray(distances)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Expected a square distance matrix")
    if max_k < 1 or max_k > 60:
        raise ValueError("max_k must be between 1 and 60")
    n = len(distances)
    indices = np.full((n, max_k), -1, dtype=np.int32)
    values = np.full((n, max_k), np.inf, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    for source in range(n):
        row = distances[source]
        eligible = np.flatnonzero(np.isfinite(row) & (np.arange(n) != source))
        if np.any(row[eligible] < 0):
            raise ValueError("Distances must be nonnegative")
        counts[source] = len(eligible)
        ordered = eligible[np.argsort(row[eligible], kind="stable")[:max_k]]
        indices[source, :len(ordered)] = ordered
        values[source, :len(ordered)] = row[ordered]
    return PackingNeighbors(indices, values, counts)
