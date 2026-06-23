from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features.geometry import (
    EPS,
    angular_coverage,
    beta_element_geometries,
    build_axis_hypotheses,
    project_to_axis,
    stable_axis,
)
from .io.mmcif import available_auth_chain_ids, build_structure_geometry, is_mmcif_path
from .models import AxisHypothesis, DomainCandidate, StructureGeometry


@dataclass(frozen=True)
class SliceConfig:
    """Configuration for Betlas axis-aligned beta-structure slicing."""

    min_points_per_slice: int = 4
    target_bin_width: float = 4.0
    min_bins: int = 4
    max_bins: int = 24


@dataclass(frozen=True)
class SlicePoint:
    """Projected beta-structure point assigned to one axis-aligned slice."""

    x: float
    y: float
    z: float
    angle: float
    point_index: int
    slice_index: int | None = None
    included: bool = True
    exclusion_reason: str = ""
    chain_id: str = ""
    auth_seq_id: int | None = None
    label_seq_id: int | None = None
    insertion_code: str = ""
    residue_name: str = ""
    residue_uid: str = ""
    strand_id: str = ""
    sheet_id: str = ""
    sheet_range_id: str = ""


@dataclass(frozen=True)
class SliceBin:
    """One informative z-bin used by Betlas slice-closure summaries."""

    index: int
    z_left: float
    z_right: float
    z_center: float
    point_count: int
    angular_coverage: float
    largest_gap_fraction: float
    points: tuple[SlicePoint, ...] = ()


@dataclass(frozen=True)
class SliceBundle:
    """Axis selection, slice bins, and summary values for one structure."""

    axis_name: str
    axis_origin: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    axis_score: float
    summary: dict[str, float] = field(default_factory=dict)
    slices: tuple[SliceBin, ...] = ()
    points: tuple[SlicePoint, ...] = ()
    warnings: tuple[str, ...] = ()


def _beta_arrays(geometry: StructureGeometry) -> tuple[np.ndarray, np.ndarray]:
    beta_points, segment_midpoints, _point_records = _beta_arrays_with_records(geometry)
    return beta_points, segment_midpoints


def _beta_arrays_with_records(
    geometry: StructureGeometry,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
    beta_geoms = beta_element_geometries(geometry)
    beta_points = (
        np.concatenate([item.coords for item in beta_geoms], axis=0)
        if beta_geoms
        else np.empty((0, 3))
    )
    segment_midpoints = (
        np.array([item.midpoint for item in beta_geoms], dtype=float)
        if beta_geoms
        else np.empty((0, 3))
    )
    point_records: list[dict[str, object]] = []
    for item in beta_geoms:
        for residue_index in item.element.residue_indices:
            residue = geometry.residues[residue_index]
            point_records.append(
                {
                    "chain_id": residue.chain_id,
                    "auth_seq_id": residue.auth_seq_id,
                    "label_seq_id": residue.label_seq_id,
                    "insertion_code": residue.insertion_code,
                    "residue_name": residue.residue_name,
                    "residue_uid": residue.residue_uid,
                    "strand_id": item.element.element_id,
                    "sheet_id": item.element.sheet_id,
                    "sheet_range_id": item.element.sheet_range_id,
                }
            )
    return beta_points, segment_midpoints, tuple(point_records)


def _empty_slice_summary() -> dict[str, float]:
    return {
        "angular_coverage": 0.0,
        "largest_gap_fraction": 1.0,
        "radius_cv": 0.0,
        "z_range": 0.0,
        "z_continuity_fraction": 0.0,
        "slice_count": 0.0,
        "slice_coverage_mean": 0.0,
        "slice_coverage_median": 0.0,
        "slice_coverage_std": 0.0,
        "slice_high_coverage_fraction": 0.0,
        "slice_largest_gap_fraction_mean": 1.0,
    }


def _compute_axis_slice_summary(
    beta_points: np.ndarray,
    segment_midpoints: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    config: SliceConfig | None = None,
    include_points: bool = False,
    point_records: tuple[dict[str, object], ...] = (),
) -> tuple[dict[str, float], tuple[SliceBin, ...]]:
    cfg = config or SliceConfig()
    xy_mid, _z_mid, _xyz_mid = project_to_axis(segment_midpoints, origin, axis)
    angles_mid = np.arctan2(xy_mid[:, 1], xy_mid[:, 0]) if len(xy_mid) else np.array([])
    coverage_all, largest_gap = angular_coverage(angles_mid)
    radii_mid = np.linalg.norm(xy_mid, axis=1) if len(xy_mid) else np.array([])

    xy_points, z_points, _xyz_points = project_to_axis(beta_points, origin, axis)
    if len(z_points) == 0:
        return _empty_slice_summary(), ()

    z_min, z_max = float(np.min(z_points)), float(np.max(z_points))
    z_range = max(0.0, z_max - z_min)
    bins = max(
        int(cfg.min_bins),
        min(
            int(cfg.max_bins),
            int(math.ceil(z_range / float(cfg.target_bin_width))) if z_range > EPS else int(cfg.min_bins),
        ),
    )
    edges = np.linspace(z_min, z_max + EPS, bins + 1)
    coverages: list[float] = []
    gap_fracs: list[float] = []
    slice_bins: list[SliceBin] = []
    point_angles = np.arctan2(xy_points[:, 1], xy_points[:, 0])
    min_points = int(cfg.min_points_per_slice)

    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        mask = (z_points >= left) & (z_points < right)
        point_count = int(np.sum(mask))
        if point_count < min_points:
            continue
        cov, gap = angular_coverage(point_angles[mask])
        gap_fraction = float(gap / (2.0 * math.pi))
        coverages.append(cov)
        gap_fracs.append(gap_fraction)

        points: tuple[SlicePoint, ...] = ()
        if include_points:
            point_indices = np.flatnonzero(mask)
            points = tuple(
                SlicePoint(
                    x=float(xy_points[point_index, 0]),
                    y=float(xy_points[point_index, 1]),
                    z=float(z_points[point_index]),
                    angle=float(point_angles[point_index]),
                    point_index=int(point_index),
                    slice_index=int(index),
                    **(
                        point_records[point_index]
                        if point_index < len(point_records)
                        else {}
                    ),
                )
                for point_index in point_indices
            )
        slice_bins.append(
            SliceBin(
                index=index,
                z_left=float(left),
                z_right=float(right),
                z_center=float((left + right) / 2.0),
                point_count=point_count,
                angular_coverage=float(cov),
                largest_gap_fraction=gap_fraction,
                points=points,
            )
        )

    radius_mean = float(np.mean(radii_mid)) if len(radii_mid) else 0.0
    radius_std = float(np.std(radii_mid)) if len(radii_mid) else 0.0
    summary = {
        "angular_coverage": float(coverage_all),
        "largest_gap_fraction": float(largest_gap / (2.0 * math.pi)),
        "radius_cv": radius_std / (radius_mean + EPS),
        "z_range": z_range,
        "z_continuity_fraction": len(slice_bins) / max(1, bins),
        "slice_count": float(len(slice_bins)),
        "slice_coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "slice_coverage_median": float(np.median(coverages)) if coverages else 0.0,
        "slice_coverage_std": float(np.std(coverages)) if coverages else 0.0,
        "slice_high_coverage_fraction": float(np.mean(np.asarray(coverages) >= 0.75)) if coverages else 0.0,
        "slice_largest_gap_fraction_mean": float(np.mean(gap_fracs)) if gap_fracs else 1.0,
    }
    return summary, tuple(slice_bins)


def _axis_score(summary: dict[str, float]) -> float:
    return float(
        0.40 * summary["slice_coverage_median"]
        + 0.25 * summary["angular_coverage"]
        + 0.20 * summary["z_continuity_fraction"]
        + 0.15 * (1.0 - summary["largest_gap_fraction"])
    )


def _project_slice_points(
    beta_points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    bins: tuple[SliceBin, ...],
    point_records: tuple[dict[str, object], ...],
) -> tuple[SlicePoint, ...]:
    if len(beta_points) == 0:
        return ()
    included_by_index = {
        point.point_index: item.index
        for item in bins
        for point in item.points
    }
    xy_points, z_points, _xyz_points = project_to_axis(beta_points, origin, axis)
    point_angles = np.arctan2(xy_points[:, 1], xy_points[:, 0])
    exclusion_reason = "" if bins else "no_informative_slices"
    rows: list[SlicePoint] = []
    for point_index in range(len(beta_points)):
        included = point_index in included_by_index
        rows.append(
            SlicePoint(
                x=float(xy_points[point_index, 0]),
                y=float(xy_points[point_index, 1]),
                z=float(z_points[point_index]),
                angle=float(point_angles[point_index]),
                point_index=int(point_index),
                slice_index=included_by_index.get(point_index),
                included=included,
                exclusion_reason="" if included else (exclusion_reason or "below_min_points_per_slice"),
                **(
                    point_records[point_index]
                    if point_index < len(point_records)
                    else {}
                ),
            )
        )
    return tuple(rows)


def _resolve_axis(
    geometry: StructureGeometry,
    axis: str | AxisHypothesis = "best",
    *,
    config: SliceConfig | None = None,
) -> tuple[str, np.ndarray, np.ndarray, float, dict[str, float]]:
    beta_points, segment_midpoints = _beta_arrays(geometry)
    hypotheses = build_axis_hypotheses(geometry)
    by_name = {item.name: item for item in hypotheses}

    if isinstance(axis, AxisHypothesis):
        origin = np.array(axis.origin, dtype=float)
        direction = stable_axis(np.array(axis.direction, dtype=float))
        summary, _bins = _compute_axis_slice_summary(beta_points, segment_midpoints, origin, direction, config=config)
        return axis.name, origin, direction, _axis_score(summary), summary

    if axis != "best":
        try:
            hypothesis = by_name[axis]
        except KeyError as exc:
            available = ", ".join(["best", *sorted(by_name)])
            raise ValueError(f"unknown Betlas slice axis {axis!r}; available axes: {available}") from exc
        origin = np.array(hypothesis.origin, dtype=float)
        direction = stable_axis(np.array(hypothesis.direction, dtype=float))
        summary, _bins = _compute_axis_slice_summary(beta_points, segment_midpoints, origin, direction, config=config)
        return hypothesis.name, origin, direction, _axis_score(summary), summary

    rows: list[tuple[float, str, np.ndarray, np.ndarray, dict[str, float]]] = []
    for hypothesis in hypotheses:
        origin = np.array(hypothesis.origin, dtype=float)
        direction = stable_axis(np.array(hypothesis.direction, dtype=float))
        summary, _bins = _compute_axis_slice_summary(beta_points, segment_midpoints, origin, direction, config=config)
        rows.append((_axis_score(summary), hypothesis.name, origin, direction, summary))
    if not rows:
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, 1.0])
        summary, _bins = _compute_axis_slice_summary(beta_points, segment_midpoints, origin, direction, config=config)
        return "none", origin, direction, 0.0, summary
    rows.sort(key=lambda row: row[0], reverse=True)
    score, name, origin, direction, summary = rows[0]
    return name, origin, direction, score, summary


def list_slice_axes(geometry: StructureGeometry) -> tuple[str, ...]:
    """Return available axis hypothesis names for Betlas slicing."""

    return tuple(hypothesis.name for hypothesis in build_axis_hypotheses(geometry))


def slice_structure(
    geometry: StructureGeometry,
    *,
    axis: str | AxisHypothesis = "best",
    config: SliceConfig | None = None,
    include_points: bool = True,
) -> SliceBundle:
    """Slice beta-structure points along a selected Betlas axis."""

    if not geometry.beta_segments:
        warnings = "; ".join(geometry.warnings) if geometry.warnings else "no beta-sheet segments"
        raise ValueError(
            "Betlas slicing requires parsed beta-sheet segments; none were found "
            f"for chain {geometry.domain.chain_id!r}. Parser warnings: {warnings}"
        )
    beta_points, segment_midpoints, point_records = _beta_arrays_with_records(geometry)
    axis_name, origin, direction, axis_score, _summary = _resolve_axis(geometry, axis, config=config)
    summary, bins = _compute_axis_slice_summary(
        beta_points,
        segment_midpoints,
        origin,
        direction,
        config=config,
        include_points=include_points,
        point_records=point_records,
    )
    points = _project_slice_points(beta_points, origin, direction, bins, point_records) if include_points else ()
    return SliceBundle(
        axis_name=axis_name,
        axis_origin=tuple(float(value) for value in origin),
        axis_direction=tuple(float(value) for value in direction),
        axis_score=float(axis_score),
        summary=summary,
        slices=bins,
        points=points,
        warnings=tuple(geometry.warnings),
    )


def summarize_slices(bundle: SliceBundle) -> dict[str, float]:
    """Return unprefixed Betlas slice summary values for a slice bundle."""

    return dict(bundle.summary)


def slice_feature_row(
    geometry: StructureGeometry,
    *,
    axis: str | AxisHypothesis = "best",
    config: SliceConfig | None = None,
) -> dict[str, float | str]:
    """Return Betlas feature-column names for the selected slice axis."""

    bundle = slice_structure(geometry, axis=axis, config=config, include_points=False)
    if axis == "best":
        row: dict[str, float | str] = {
            "betlas_axis_best_name": bundle.axis_name,
            "betlas_axis_best_score": bundle.axis_score,
            "betlas_axis_best_is_strand_axis": 1.0 if bundle.axis_name == "strand_axis" else 0.0,
            "betlas_axis_best_is_point_pc1": 1.0 if bundle.axis_name == "point_pc1" else 0.0,
            "betlas_axis_best_is_point_pc2": 1.0 if bundle.axis_name == "point_pc2" else 0.0,
            "betlas_axis_best_is_point_pc3": 1.0 if bundle.axis_name == "point_pc3" else 0.0,
            "betlas_z_continuity_fraction": float(bundle.summary["z_continuity_fraction"]),
        }
        row.update({f"betlas_axis_best_{key}": float(value) for key, value in bundle.summary.items()})
        return row

    prefix = f"betlas_axis_{bundle.axis_name}_"
    return {
        "betlas_axis_slice_name": bundle.axis_name,
        "betlas_axis_slice_score": bundle.axis_score,
        **{f"{prefix}{key}": float(value) for key, value in bundle.summary.items()},
    }


def slice_mmcif(
    mmcif_path: str | Path,
    *,
    chain_id: str,
    residue_ranges: str = "",
    record_id: str | None = None,
    domain_id: str | None = None,
    pdb_id: str | None = None,
    axis: str = "best",
    config: SliceConfig | None = None,
) -> SliceBundle:
    """Build geometry from an mmCIF file and return a Betlas slice bundle."""

    path = Path(mmcif_path)
    if not is_mmcif_path(path):
        raise ValueError(
            "Betlas slicing currently accepts mmCIF files (.cif, .mmcif, .cif.gz, .mmcif.gz)."
        )
    if not path.exists():
        raise FileNotFoundError(f"structure file does not exist: {path}")
    stem = path.name.split(".", 1)[0].lower()
    domain = DomainCandidate(
        record_id=record_id or f"{stem}_{chain_id}",
        pdb_id=(pdb_id or stem).lower(),
        chain_id=chain_id,
        domain_id=domain_id or f"{stem}_{chain_id}",
        residue_ranges=residue_ranges,
        fold_label_final="unlabeled",
        evidence_level="user_input",
        label_source_primary="user_input",
    )
    geometry = build_structure_geometry(domain, path)
    if not geometry.residues:
        chains = available_auth_chain_ids(path)
        available = ", ".join(chains) if chains else "<none>"
        raise ValueError(
            f"no residues were selected for chain {chain_id!r}; available author chain ids: {available}"
        )
    return slice_structure(geometry, axis=axis, config=config)


__all__ = [
    "SliceBin",
    "SliceBundle",
    "SliceConfig",
    "SlicePoint",
    "list_slice_axes",
    "slice_feature_row",
    "slice_mmcif",
    "slice_structure",
    "summarize_slices",
]
