from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..models import AxisHypothesis, SecondaryStructureElement, StructureGeometry

EPS = 1e-9


@dataclass(frozen=True)
class ElementGeometry:
    element: SecondaryStructureElement
    coords: np.ndarray
    midpoint: np.ndarray
    axis: np.ndarray
    directed_axis: np.ndarray
    length: float


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def stable_axis(vector: np.ndarray) -> np.ndarray:
    axis = unit(vector)
    if not np.any(axis):
        return np.array([0.0, 0.0, 1.0])
    idx = int(np.argmax(np.abs(axis)))
    if axis[idx] < 0:
        axis = -axis
    return axis


def residue_coords(geometry: StructureGeometry, indices: tuple[int, ...]) -> np.ndarray:
    return np.array([geometry.residues[index].coord_ca for index in indices], dtype=float)


def element_geometry(geometry: StructureGeometry, element: SecondaryStructureElement) -> ElementGeometry:
    coords = residue_coords(geometry, element.residue_indices)
    midpoint = coords.mean(axis=0) if len(coords) else np.zeros(3)
    directed_axis = unit(coords[-1] - coords[0]) if len(coords) >= 2 else np.array([0.0, 0.0, 1.0])
    axis = stable_axis(directed_axis) if len(coords) >= 2 else np.array([0.0, 0.0, 1.0])
    length = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) >= 2 else 0.0
    return ElementGeometry(
        element=element,
        coords=coords,
        midpoint=midpoint,
        axis=axis,
        directed_axis=directed_axis,
        length=length,
    )


def beta_element_geometries(geometry: StructureGeometry) -> list[ElementGeometry]:
    return [element_geometry(geometry, segment) for segment in geometry.beta_segments]


def helix_element_geometries(geometry: StructureGeometry) -> list[ElementGeometry]:
    return [element_geometry(geometry, helix) for helix in geometry.helices]


def pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 2:
        return np.zeros(3), np.eye(3), np.zeros(3)
    origin = points.mean(axis=0)
    centered = points - origin
    cov = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return origin, vectors[:, order], values[order]


def orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = stable_axis(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, w))) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(w, helper))
    v = unit(np.cross(w, u))
    return u, v, w


def project_to_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, w = orthonormal_frame(axis)
    centered = points - origin
    x = centered @ u
    y = centered @ v
    z = centered @ w
    return np.column_stack([x, y]), z, np.column_stack([x, y, z])


def circular_gaps(angles: np.ndarray) -> np.ndarray:
    if len(angles) <= 1:
        return np.array([2.0 * math.pi])
    sorted_angles = np.sort(np.mod(angles, 2.0 * math.pi))
    return np.diff(np.r_[sorted_angles, sorted_angles[0] + 2.0 * math.pi])


def angular_coverage(angles: np.ndarray) -> tuple[float, float]:
    if len(angles) <= 1:
        return 0.0, 2.0 * math.pi
    gaps = circular_gaps(angles)
    max_gap = float(np.max(gaps))
    return float(max(0.0, 1.0 - max_gap / (2.0 * math.pi))), max_gap


def entropy(values: list[str] | np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) <= 1:
        return 0.0
    probs = counts.astype(float) / float(np.sum(counts))
    value = -np.sum(probs * np.log2(probs + EPS)) / math.log2(len(unique))
    return float(min(1.0, max(0.0, value)))


def dominant_strand_axis(elements: list[ElementGeometry]) -> np.ndarray:
    if not elements:
        return np.array([0.0, 0.0, 1.0])
    axes = np.array([item.axis for item in elements], dtype=float)
    dyadic = axes.T @ axes / max(1, len(axes))
    values, vectors = np.linalg.eigh(dyadic)
    return stable_axis(vectors[:, int(np.argmax(values))])


def build_axis_hypotheses(geometry: StructureGeometry) -> list[AxisHypothesis]:
    beta_geoms = beta_element_geometries(geometry)
    beta_coords = np.concatenate([item.coords for item in beta_geoms], axis=0) if beta_geoms else np.empty((0, 3))
    if len(beta_coords) == 0:
        all_coords = np.array([residue.coord_ca for residue in geometry.residues], dtype=float)
        beta_coords = all_coords if len(all_coords) else np.zeros((1, 3))
    origin, vectors, values = pca(beta_coords)
    strand_axis = dominant_strand_axis(beta_geoms)
    hypotheses = [
        AxisHypothesis("strand_axis", tuple(origin), tuple(strand_axis), 0.0),
        AxisHypothesis("point_pc1", tuple(origin), tuple(stable_axis(vectors[:, 0])), float(values[0])),
        AxisHypothesis("point_pc2", tuple(origin), tuple(stable_axis(vectors[:, 1])), float(values[1] if len(values) > 1 else 0.0)),
        AxisHypothesis("point_pc3", tuple(origin), tuple(stable_axis(vectors[:, 2])), float(values[2] if len(values) > 2 else 0.0)),
    ]
    unique: list[AxisHypothesis] = []
    seen: list[np.ndarray] = []
    for hypothesis in hypotheses:
        direction = np.array(hypothesis.direction, dtype=float)
        if any(abs(float(np.dot(direction, prev))) > 0.995 for prev in seen):
            continue
        seen.append(direction)
        unique.append(hypothesis)
    return unique
