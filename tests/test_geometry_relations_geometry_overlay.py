from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from betlas.geometry_relations.cohorts import A, H, T
from betlas.geometry_relations.geometry_overlay import run_geometry_overlay
from betlas.geometry_relations.relationships import summarize_relationships


def _inputs(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    cohort = pd.DataFrame({
        "record_id": ["d1", "d2", "c1", "c2", "b", "n", "u"],
        A: ["2.60"] * 4 + ["2.40", "2.70", "2.60"],
        T: ["2.60.120"] * 4 + ["2.40.10", "2.70.10", "2.60.40"],
        H: [f"h{i}" for i in range(7)],
        "role": ["discovery"] * 2 + ["confirmation"] * 3 + ["descriptive_single_h"] * 2,
        "packing_eligible": [True] * 6 + [False],
    })
    root = tmp_path / "atlas"
    for phase, pairs in (
        ("all", [("d1", "b"), ("c1", "b"), ("c2", "b")]),
        ("discovery", [("d1", "d2")]), ("confirmation", [("c1", "c2")]),
    ):
        nodes = cohort if phase == "all" else cohort.loc[cohort.role.eq(phase)]
        rows = []
        for source, target in pairs:
            for left, right in ((source, target), (target, source)):
                rows.append({"source_id": left, "target_id": right, "distance": 0.2, "mutual": True, "short_edge": True})
        if phase == "all":
            rows.extend([
                {"source_id": "n", "target_id": "b", "distance": 5, "mutual": True, "short_edge": False},
                {"source_id": "d2", "target_id": "b", "distance": 0.1, "mutual": False, "short_edge": True},
            ])
        edges = pd.DataFrame(rows)
        relations = summarize_relationships(nodes, edges, A)
        directory = root / phase / "rms"
        (directory / "relations-k30").mkdir(parents=True)
        pd.DataFrame(relations["per_domain"]).to_csv(directory / "relations-k30/architecture-per_domain.csv", index=False)
        edges.to_csv(directory / "edges-k30.csv.gz", index=False, compression="gzip")
    return cohort, root


def test_phase_relations_do_not_fall_back_to_all_and_empty_is_NA(tmp_path: Path) -> None:
    cohort, atlas = _inputs(tmp_path)
    protocol = run_geometry_overlay(cohort, atlas, tmp_path / "geometry.csv")
    result = pd.read_csv(tmp_path / "geometry.csv").set_index("record_id")
    assert result.loc["c1", "near_barrel"]
    assert not result.loc["c1", "near_barrel_confirmation"]
    assert pd.isna(result.loc["c1", "near_barrel_discovery"])
    assert result.loc["d1", "near_barrel"]
    assert not result.loc["d1", "near_barrel_discovery"]
    assert pd.isna(result.loc["b", "near_barrel_confirmation"])
    for record_id in ("d2", "n", "u"):
        assert pd.isna(result.loc[record_id, "near_barrel"])
        assert pd.isna(result.loc[record_id, "label_overlap"])
        assert pd.isna(result.loc[record_id, "label_overlap_fraction"])
    assert result.loc["n", "n_neighbors"] == 0
    assert pd.isna(result.loc["u", "n_neighbors"])
    assert result.loc["b", "n_neighbors"] == 3  # reciprocal rows are deduplicated
    assert result.loc["c1", "label_overlap_fraction"] == 1
    assert protocol["phases"]["confirmation"]["measured_neighborhoods"] == 2


def test_inconsistent_saved_adjacency_is_rejected(tmp_path: Path) -> None:
    cohort, atlas = _inputs(tmp_path)
    path = atlas / "all/rms/relations-k30/architecture-per_domain.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table.loc[table.record_id.eq("b"), "n_neighbors"] = "6"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="adjacency denominator"):
        run_geometry_overlay(cohort, atlas, tmp_path / "invalid.csv")


def test_confirmation_graph_requires_exact_phase_membership(tmp_path: Path) -> None:
    cohort, atlas = _inputs(tmp_path)
    cohort.loc[cohort.record_id.eq("c1"), "role"] = "discovery"
    with pytest.raises(ValueError, match="membership"):
        run_geometry_overlay(cohort, atlas, tmp_path / "invalid.csv")
