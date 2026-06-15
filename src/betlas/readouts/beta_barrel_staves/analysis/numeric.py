from __future__ import annotations

from collections import Counter
from math import atan2, exp, pi

import numpy as np


def clamp_ratio(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def mode_count(counts: list[int]) -> int:
    if not counts:
        return 0
    median_count = float(np.median(np.asarray(counts, dtype=float)))
    histogram = Counter(counts)
    return int(
        min(
            histogram,
            key=lambda count: (
                -histogram[count],
                abs(float(count) - median_count),
                -count,
            ),
        )
    )


def quantile_count(counts: list[int], quantile: float) -> int:
    if not counts:
        return 0
    return int(round(float(np.quantile(np.asarray(counts, dtype=float), quantile))))


def angle_distance(first: float, second: float) -> float:
    return float(abs((first - second + pi) % (2.0 * pi) - pi))


def circular_mean(angles: list[float]) -> float:
    if not angles:
        return 0.0
    sine = float(np.sum(np.sin(np.asarray(angles, dtype=float))))
    cosine = float(np.sum(np.cos(np.asarray(angles, dtype=float))))
    return float(atan2(sine, cosine))


def estimate_similarity(count: int, estimate: int, *, scale_fraction: float = 0.18) -> float:
    if count <= 0 or estimate <= 0:
        return 0.0
    sigma = max(1.5, float(estimate) * scale_fraction)
    return float(exp(-abs(count - estimate) / sigma))
