from __future__ import annotations

from pathlib import Path

from .core import TopologyDiagnosticsResult, run_topology_diagnostics


def score_topology_ambiguity(
    features_csv: str | Path,
    *,
    predictions_csv: str | Path | None = None,
    model: str | None = "hist_gradient_boosting",
    out_csv: str | Path | None = None,
    k_neighbors: int = 12,
) -> TopologyDiagnosticsResult:
    return run_topology_diagnostics(
        features_csv=features_csv,
        predictions_csv=predictions_csv,
        model=model,
        out_csv=out_csv,
        mode="ambiguity",
        k_neighbors=k_neighbors,
    )


def score_fold_continuous_signals(
    features_csv: str | Path,
    *,
    predictions_csv: str | Path | None = None,
    model: str | None = "hist_gradient_boosting",
    out_csv: str | Path | None = None,
    k_neighbors: int = 12,
) -> TopologyDiagnosticsResult:
    return run_topology_diagnostics(
        features_csv=features_csv,
        predictions_csv=predictions_csv,
        model=model,
        out_csv=out_csv,
        mode="continuous",
        k_neighbors=k_neighbors,
    )


def detect_mixed_topology(
    features_csv: str | Path,
    *,
    predictions_csv: str | Path | None = None,
    model: str | None = "hist_gradient_boosting",
    out_csv: str | Path | None = None,
    k_neighbors: int = 12,
) -> TopologyDiagnosticsResult:
    return run_topology_diagnostics(
        features_csv=features_csv,
        predictions_csv=predictions_csv,
        model=model,
        out_csv=out_csv,
        mode="mixed",
        k_neighbors=k_neighbors,
    )
