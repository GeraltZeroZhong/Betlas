from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pandas as pd
import pytest

import betlas
import betlas.readouts.beta_barrel_detection as detection_api
import betlas.readouts.beta_barrel_detection.cli as detection_cli
import betlas.readouts.beta_barrel_detection.pipeline as detection_pipeline
import betlas.readouts.beta_barrel_staves as staves_api
import betlas.readouts.beta_barrel_staves.cli as staves_cli
import betlas.readouts.beta_barrel_staves.readout as staves_readout
from betlas.ml.benchmark import numeric_feature_columns
from betlas.readouts import get_readout, list_readouts
from betlas.readouts.beta_barrel_detection.config import build_config as build_detection_config
from betlas.readouts.beta_barrel_detection.pipeline import discover_input_files
from betlas.readouts.beta_barrel_detection.pipeline_workers import analyze_chain_payload
from betlas.readouts.beta_barrel_detection.prepare_cache import (
    _normalize_payloads,
    prepare_cache_path,
    store_prepare_payloads,
)
from betlas.readouts.beta_barrel_detection.results import (
    write_results_csv as write_detection_results_csv,
)
from betlas.readouts.beta_barrel_staves import (
    build_config,
    count_beta_barrel_staves,
    count_strands,
)
from betlas.readouts.beta_barrel_staves.analysis.analyzer import StrandCountAnalyzer
from betlas.readouts.beta_barrel_staves.cli import _apply_barrel_decisions, _load_barrel_decisions
from betlas.readouts.beta_barrel_staves.config import AnalyzerConfig
from betlas.readouts.beta_barrel_staves.exceptions import DsspNotFoundError
from betlas.readouts.beta_barrel_staves.geometry.slicer import ProteinSlicer
from betlas.readouts.beta_barrel_staves.io.prepare_cache import (
    prepare_cache_path as staves_prepare_cache_path,
)
from betlas.readouts.beta_barrel_staves.io.prepare_cache import (
    store_prepare_payloads as store_staves_prepare_payloads,
)
from betlas.readouts.beta_barrel_staves.models import StrandCountResult
from betlas.readouts.beta_barrel_staves.pipeline import (
    _prepare_error_rows as staves_prepare_error_rows,
)
from betlas.readouts.beta_barrel_staves.pipeline.chain import (
    analyze_chain_payload as analyze_staves_chain_payload,
)
from betlas.readouts.beta_barrel_staves.runtime import find_dssp_binary, require_dssp_binary
from betlas.readouts.topology_diagnostics import (
    compute_topology_diagnostics,
    run_topology_diagnostics,
)
from betlas.readouts.topology_diagnostics.cli import main as topology_diagnostics_cli
from betlas.specs import list_readout_column_specs


def _ring_layer(count: int, *, missing: set[int] | None = None) -> list[tuple[float, float, float, float]]:
    missing = missing or set()
    points = []
    for strand_id in range(count):
        if strand_id in missing:
            continue
        theta = (2.0 * math.pi * strand_id) / count
        points.append(
            (
                10.0 * math.cos(theta),
                10.0 * math.sin(theta),
                float(strand_id),
                float(strand_id),
            )
        )
    return points


def _ring_layer_with_terminal_outliers(count: int) -> list[tuple[float, float, float, float]]:
    return [
        (10.5, 0.0, -10.0, -1.0),
        *_ring_layer(count),
        (-10.5, 0.0, float(count + 10), float(count)),
    ]


def _ring_layer_with_inner_plug(count: int) -> list[tuple[float, float, float, float]]:
    points = _ring_layer(count)
    for offset in range(5):
        theta = (2.0 * math.pi * offset) / 5.0
        points.append(
            (
                2.0 * math.cos(theta),
                2.0 * math.sin(theta),
                float(count + offset),
                float(count + offset),
            )
        )
    return points


def test_readout_registry_exposes_beta_barrel_staves():
    names = [spec.name for spec in list_readouts()]

    assert names == [
        "beta-barrel-detection",
        "beta-barrel-staves",
        "fold-continuous-scores",
        "mixed-topology",
        "topology-ambiguity",
        "topology-diagnostics",
    ]
    assert "bfvd-viral-beta-fold-scan" not in names
    assert get_readout("beta-barrel-detection").summary.startswith("Betlas native")
    assert get_readout("beta-barrel-staves").summary.startswith("Secondary readout")
    assert get_readout("topology-ambiguity").summary.startswith("Boundary-region ambiguity")
    assert "uncalibrated heuristic score" in get_readout("beta-barrel-detection").output_protocol
    assert "uncalibrated heuristic confidence" in get_readout("beta-barrel-staves").output_protocol
    assert "calibration status" in get_readout("topology-ambiguity").output_protocol


def test_top_level_detection_api_alias_is_public() -> None:
    assert isinstance(betlas.__version__, str)
    assert "__version__" in betlas.__all__
    assert betlas.detect_beta_barrel_like is detection_pipeline.detect


def test_readout_subpackage_all_keeps_implementation_helpers_private() -> None:
    implementation_helpers = {"ProteinLoader", "PCAAligner", "ProteinSlicer", "BarrelAnalyzer"}

    assert not implementation_helpers.intersection(detection_api.__all__)
    assert not implementation_helpers.intersection(staves_api.__all__)


def test_staves_result_exposes_typed_source_path() -> None:
    result = StrandCountResult.from_row(
        {
            "filename": "example.cif",
            "source_path": "/tmp/example.cif",
            "chain": "A",
            "result": "FILTERED_OUT",
            "result_stage": "prefilter",
            "reason": "unit",
        }
    )

    assert result.source_path == "/tmp/example.cif"
    assert result.to_dict()["source_path"] == "/tmp/example.cif"


def test_beta_barrel_staves_config_defaults_are_betlas_owned():
    cfg = build_config()

    assert cfg.output.csv_path == "beta_barrel_staves_results.csv"
    assert cfg.barrel_gate.enabled is False
    assert cfg.analyzer.rules.barrel_wall_graph.enabled is True


def test_count_beta_barrel_staves_python_api_requires_gate_or_explicit_ungated() -> None:
    with pytest.raises(ValueError, match="requires barrel_decisions"):
        count_beta_barrel_staves("structure.cif", write_csv=False, print_summary=False)


def test_count_beta_barrel_staves_python_api_defaults_do_not_write_current_directory() -> None:
    signature = inspect.signature(count_beta_barrel_staves)

    assert signature.parameters["write_csv"].default is None
    assert signature.parameters["print_summary"].default is False


def test_count_beta_barrel_staves_python_api_writes_gated_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "structure.cif"
    source.write_text("data_unit\n", encoding="utf-8")
    out_csv = tmp_path / "staves.csv"
    decisions_csv = tmp_path / "decisions.csv"
    pd.DataFrame(
        [
            {
                "filename": source.name,
                "source_path": str(source),
                "chain": "A",
                "result": "BARREL",
                "decision_score": 0.9,
                "reason": "unit",
            }
        ]
    ).to_csv(decisions_csv, index=False)
    seen_kwargs: dict[str, object] = {}

    class FakeResult:
        input_files = [str(source)]

        @staticmethod
        def raw_rows() -> list[dict[str, object]]:
            return [
                {
                    "filename": source.name,
                    "source_path": str(source),
                    "chain": "A",
                    "result": "COUNTED",
                    "result_stage": "analysis",
                    "strand_count": 8,
                    "confidence": 0.5,
                    "score_type": "heuristic",
                    "calibration_status": "uncalibrated",
                    "config_profile": "native",
                    "reason": "",
                }
            ]

    def fake_run_pipeline_result(*args: object, **kwargs: object) -> FakeResult:
        seen_kwargs.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(staves_readout, "run_pipeline_result", fake_run_pipeline_result)

    count_beta_barrel_staves(
        source,
        output=out_csv,
        barrel_decisions=decisions_csv,
        print_summary=False,
        workers=1,
        prepare_workers=1,
    )

    assert seen_kwargs["show_progress"] is False
    assert out_csv.exists()
    metadata_path = out_csv.with_suffix(".csv.metadata.json")
    manifest_path = Path(f"{out_csv}.manifest.json")
    assert metadata_path.exists()
    assert manifest_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metadata["barrel_gate"]["decisions_csv"]["exists"] is True
    assert manifest["inputs"]["barrel_decisions_csv"]["exists"] is True
    assert manifest["extra"]["barrel_gate"]["matching_contract"] == "exact resolved source_path plus chain"


def test_beta_barrel_staves_compat_api_requires_explicit_ungated() -> None:
    with pytest.raises(ValueError, match="allow_ungated=True"):
        count_strands("structure.cif", write_csv=False, print_summary=False)


def test_beta_barrel_detection_discovers_compressed_structure_suffixes(tmp_path: Path) -> None:
    for name in ["a.pdb.gz", "b.cif.gz", "c.mmcif.gz", "ignored.txt"]:
        (tmp_path / name).write_text("", encoding="utf-8")

    files = discover_input_files(
        str(tmp_path),
        [".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz"],
        strict=True,
    )

    assert [Path(path).name for path in files] == ["a.pdb.gz", "b.cif.gz", "c.mmcif.gz"]
    assert discover_input_files(str(tmp_path / "a.pdb.gz"), [".pdb.gz"], strict=True) == [
        str(tmp_path / "a.pdb.gz")
    ]


def test_beta_barrel_detection_reports_dssp_error_before_sheet_prefilter() -> None:
    cfg = build_detection_config([])
    residues = [
        {"coord": (float(index), 0.0, 0.0), "is_sheet": False}
        for index in range(int(cfg.input.min_chain_residues) + 1)
    ]

    row = analyze_chain_payload(
        {
            "filename": "failed.cif",
            "source_path": "/tmp/failed.cif",
            "chain": "A",
            "dssp_error": "DSSP failed for failed.cif: missing executable",
            "residues_data": residues,
        },
        cfg,
    )

    assert row["result"] == "ERROR"
    assert row["result_stage"] == "dssp"
    assert "DSSP failed" in row["reason"]


def test_beta_barrel_staves_reports_dssp_error_before_sheet_prefilter() -> None:
    cfg = build_config()
    residues = [
        {"coord": (float(index), 0.0, 0.0), "is_sheet": False}
        for index in range(int(cfg.input.min_chain_residues) + 1)
    ]

    row = analyze_staves_chain_payload(
        {
            "filename": "failed.cif",
            "source_path": "/tmp/failed.cif",
            "chain": "A",
            "dssp_error": "DSSP failed for failed.cif: missing executable",
            "residues_data": residues,
        },
        cfg,
    )

    assert row["result"] == "ERROR"
    assert row["result_stage"] == "dssp"
    assert "DSSP failed" in row["reason"]


def test_staves_slicer_keeps_filled_gap_beta_runs_separate() -> None:
    slicer = ProteinSlicer(step_size=1.0, fill_sheet_hole_length=1)
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 2.0),
    ]
    residues = [{"is_sheet": True}, {"is_sheet": False}, {"is_sheet": True}]

    slices = slicer.slice_structure(coords, residues)

    assert {point[3] for points in slices.values() for point in points} == {0.0, 1.0}
    assert {point[3] for point in slices[1.0]} == {0.0, 1.0}


def test_beta_barrel_detection_prepare_error_rows_mark_score_not_applicable() -> None:
    rows = detection_pipeline._prepare_error_rows(["/tmp/missing.cif: parser failed"])

    assert rows[0]["score_type"] == "not_applicable"
    assert rows[0]["calibration_status"] == "not_applicable"


def test_beta_barrel_staves_prepare_error_rows_mark_score_not_applicable() -> None:
    rows = staves_prepare_error_rows(["/tmp/missing.cif: parser failed"])

    assert rows[0]["score_type"] == "not_applicable"
    assert rows[0]["calibration_status"] == "not_applicable"


def test_beta_barrel_detection_prepare_cache_preserves_or_skips_dssp_errors(tmp_path: Path) -> None:
    cfg = build_detection_config([])
    cfg.runtime.prepare_cache_enabled = True
    cfg.runtime.prepare_cache_dir = str(tmp_path / "cache")
    structure = tmp_path / "failed.cif"
    structure.write_text("data_failed\n", encoding="utf-8")
    payload = {
        "filename": "failed.cif",
        "source_path": str(structure),
        "chain": "A",
        "dssp_status": "error",
        "dssp_error": "DSSP failed",
        "residues_data": [
            {
                "coord": (0.0, 0.0, 0.0),
                "is_sheet": False,
                "chain": "A",
                "resseq": 1,
                "icode": "",
                "hetfield": " ",
            }
        ],
    }

    normalized = _normalize_payloads([payload])
    assert normalized is not None
    assert normalized[0]["dssp_status"] == "error"
    assert normalized[0]["dssp_error"] == "DSSP failed"

    store_prepare_payloads(str(structure), cfg, [payload])
    assert not prepare_cache_path(str(structure), cfg).exists()


def test_beta_barrel_staves_prepare_cache_skips_dssp_errors(tmp_path: Path) -> None:
    cfg = build_config()
    cfg.runtime.prepare_cache_enabled = True
    cfg.runtime.prepare_cache_dir = str(tmp_path / "cache")
    structure = tmp_path / "failed.cif"
    structure.write_text("data_failed\n", encoding="utf-8")
    payload = {
        "filename": "failed.cif",
        "source_path": str(structure),
        "chain": "A",
        "dssp_error": "DSSP failed",
        "residues_data": [{"coord": (0.0, 0.0, 0.0), "is_sheet": False}],
    }

    store_staves_prepare_payloads(str(structure), cfg, [payload])

    assert not staves_prepare_cache_path(str(structure), cfg).exists()


def test_beta_barrel_staves_readout_counts_persistent_ring():
    slices = {float(index): _ring_layer(8) for index in range(6)}

    report = StrandCountAnalyzer().analyze(slices)

    assert report["strand_count"] == 8
    assert report["persistent_strands"] == 8
    assert report["candidate_strands"] == 8
    assert report["support_consensus_strands"] == 8
    assert report["run_window_core_applied"] == 0


def test_beta_barrel_staves_readout_trims_terminal_outliers():
    slices = {float(index): _ring_layer_with_terminal_outliers(8) for index in range(6)}

    report = StrandCountAnalyzer().analyze(slices)

    assert report["strand_count"] == 8
    assert report["trimmed_layers"] == 6


def test_beta_barrel_staves_readout_filters_inner_plug_points():
    slices = {float(index): _ring_layer_with_inner_plug(16) for index in range(6)}
    cfg = AnalyzerConfig()
    cfg.rules.radial_outlier.min_radius_ratio = 0.55

    report = StrandCountAnalyzer(cfg).analyze(slices)

    assert report["strand_count"] == 16
    assert report["radial_outlier_points"] == 30


def test_staves_barrel_decisions_require_source_path_and_do_not_basename_match(tmp_path: Path) -> None:
    decisions_csv = tmp_path / "decisions.csv"
    source_a = tmp_path / "a" / "same.cif"
    source_b = tmp_path / "b" / "same.cif"
    decisions_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "filename": "same.cif",
                "source_path": str(source_a),
                "chain": "A",
                "result": "BARREL",
                "decision_score": 0.9,
            }
        ]
    ).to_csv(decisions_csv, index=False)

    decisions = _load_barrel_decisions(str(decisions_csv))
    rows = [
        {
            "filename": "same.cif",
            "source_path": str(source_b),
            "chain": "A",
            "result": "COUNTED",
            "strand_count": 8,
            "confidence": 0.8,
        }
    ]
    gated = _apply_barrel_decisions(rows, decisions)
    assert gated[0]["barrel_gate_result"] == "MISSING"
    assert gated[0]["result"] == "FILTERED_OUT"


def test_staves_barrel_decisions_fail_on_ambiguous_duplicate_filename_chain(tmp_path: Path) -> None:
    decisions_csv = tmp_path / "decisions.csv"
    pd.DataFrame(
        [
            {
                "filename": "same.cif",
                "source_path": str(tmp_path / "a" / "same.cif"),
                "chain": "A",
                "result": "BARREL",
            },
            {
                "filename": "same.cif",
                "source_path": str(tmp_path / "b" / "same.cif"),
                "chain": "A",
                "result": "BARREL",
            },
        ]
    ).to_csv(decisions_csv, index=False)

    with pytest.raises(ValueError, match="ambiguous barrel decisions"):
        _load_barrel_decisions(str(decisions_csv))


def test_staves_gated_filtered_rows_clear_candidate_counts(tmp_path: Path) -> None:
    source = tmp_path / "nonbarrel.cif"
    decisions = {
        (str(source), "A"): {
            "source_path": str(source),
            "chain": "A",
            "result": "NON_BARREL",
            "decision_score": "0",
            "reason": "Too few scored slices",
        }
    }
    rows = [
        {
            "filename": source.name,
            "source_path": str(source),
            "chain": "A",
            "result": "COUNTED",
            "result_stage": "count",
            "strand_count": 16,
            "confidence": 0.92,
            "candidate_strands": 16,
            "persistent_strands": 16,
            "support_consensus_strands": 16,
            "count_decision_score": 1.12,
            "barrel_wall_graph_count": 16,
            "barrel_wall_graph_selected_ids": "1;2;3",
            "run_window_core_base_strand_count": 18,
            "axis_hypothesis_name": "strand_axis",
        }
    ]

    gated = _apply_barrel_decisions(rows, decisions)

    assert gated[0]["result"] == "FILTERED_OUT"
    assert gated[0]["strand_count"] == 0
    assert gated[0]["candidate_strands"] == 0
    assert gated[0]["persistent_strands"] == 0
    assert gated[0]["support_consensus_strands"] == 0
    assert gated[0]["count_decision_score"] == 0
    assert gated[0]["barrel_wall_graph_count"] == 0
    assert gated[0]["barrel_wall_graph_selected_ids"] == ""
    assert gated[0]["run_window_core_base_strand_count"] == 0
    assert gated[0]["axis_hypothesis_name"] == ""


def test_staves_gated_prepare_errors_are_not_rewritten_as_missing_gate(tmp_path: Path) -> None:
    source = tmp_path / "broken.cif"
    rows = [
        {
            "filename": source.name,
            "source_path": str(source),
            "chain": "",
            "result": "ERROR",
            "result_stage": "prepare",
            "reason": "malformed structure",
        }
    ]

    gated = _apply_barrel_decisions(rows, decisions={})

    assert gated[0]["result"] == "ERROR"
    assert gated[0]["result_stage"] == "prepare"
    assert gated[0]["barrel_gate_result"] == "NOT_EVALUATED"
    assert gated[0]["reason"] == "malformed structure"


def test_staves_explicit_bad_dssp_path_does_not_fallback_to_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mkdssp = bin_dir / "mkdssp"
    mkdssp.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mkdssp.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert find_dssp_binary("mkdssp") == str(mkdssp)
    assert find_dssp_binary(str(tmp_path / "missing" / "mkdssp")) is None
    with pytest.raises(DsspNotFoundError, match="Configured DSSP executable"):
        require_dssp_binary(str(tmp_path / "missing" / "mkdssp"))


def test_staves_gated_all_prepare_failures_still_write_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_csv = tmp_path / "staves.csv"
    decisions_csv = tmp_path / "decisions.csv"
    pd.DataFrame(
        [
            {
                "filename": "broken.cif",
                "source_path": str(tmp_path / "broken.cif"),
                "chain": "A",
                "result": "BARREL",
                "decision_score": 0.9,
            }
        ]
    ).to_csv(decisions_csv, index=False)

    class FakeResult:
        input_files = [str(tmp_path / "broken.cif")]

        @staticmethod
        def raw_rows() -> list[dict[str, object]]:
            return [
                {
                    "filename": "broken.cif",
                    "source_path": str(tmp_path / "broken.cif"),
                    "chain": "",
                    "result": "ERROR",
                    "result_stage": "prepare",
                    "reason": "malformed structure",
                }
            ]

    monkeypatch.setattr(staves_cli, "run_pipeline_result", lambda *args, **kwargs: FakeResult())

    with pytest.raises(SystemExit) as exc:
        staves_cli.main(
            [
                str(tmp_path / "broken.cif"),
                "--barrel-decisions",
                str(decisions_csv),
                "--out",
                str(out_csv),
            ]
        )

    assert exc.value.code == 2
    assert out_csv.exists()
    assert out_csv.with_suffix(".csv.metadata.json").exists()
    row = pd.read_csv(out_csv).iloc[0]
    assert row["result"] == "ERROR"
    assert row["result_stage"] == "prepare"
    assert row["barrel_gate_result"] == "NOT_EVALUATED"


def test_detection_cli_disables_progress_when_stderr_is_not_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs: dict[str, object] = {}

    def fake_run_pipeline_result(*args: object, **kwargs: object) -> object:
        seen_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(detection_pipeline, "run_pipeline_result", fake_run_pipeline_result)
    monkeypatch.setattr(detection_cli.sys.stderr, "isatty", lambda: False)

    detection_cli.main([str(tmp_path / "input.cif"), "--out", str(tmp_path / "detection.csv")])

    assert seen_kwargs["show_progress"] is False


def test_staves_cli_disables_progress_when_stderr_is_not_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_csv = tmp_path / "staves.csv"
    decisions_csv = tmp_path / "decisions.csv"
    source = tmp_path / "input.cif"
    pd.DataFrame(
        [
            {
                "filename": source.name,
                "source_path": str(source),
                "chain": "A",
                "result": "BARREL",
                "decision_score": 0.9,
            }
        ]
    ).to_csv(decisions_csv, index=False)
    seen_kwargs: dict[str, object] = {}

    class FakeResult:
        input_files = [str(source)]

        @staticmethod
        def raw_rows() -> list[dict[str, object]]:
            return [
                {
                    "filename": source.name,
                    "source_path": str(source),
                    "chain": "A",
                    "result": "OK",
                    "result_stage": "analysis",
                    "strand_count": 8,
                    "confidence": 0.8,
                }
            ]

    def fake_run_pipeline_result(*args: object, **kwargs: object) -> FakeResult:
        seen_kwargs.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(staves_cli, "run_pipeline_result", fake_run_pipeline_result)
    monkeypatch.setattr(staves_cli.sys.stderr, "isatty", lambda: False)

    staves_cli.main(
        [
            str(source),
            "--barrel-decisions",
            str(decisions_csv),
            "--out",
            str(out_csv),
        ]
    )

    assert seen_kwargs["show_progress"] is False


def test_detection_failure_only_csv_uses_stable_schema(tmp_path: Path) -> None:
    out_csv = tmp_path / "detection.csv"

    write_detection_results_csv(
        [
            {
                "filename": "broken.cif",
                "source_path": str(tmp_path / "broken.cif"),
                "chain": "A",
                "result": "ERROR",
                "result_stage": "prepare",
                "reason": "malformed structure",
            }
        ],
        str(out_csv),
    )

    columns = pd.read_csv(out_csv, nrows=0).columns
    assert "decision_score" in columns
    assert "score_type" in columns
    assert "calibration_status" in columns


def test_readout_score_metadata_columns_are_documented() -> None:
    detection = {spec.name for spec in list_readout_column_specs("beta-barrel-detection")}
    staves = {spec.name for spec in list_readout_column_specs("beta-barrel-staves")}

    assert {"score_type", "calibration_status", "config_profile"} <= detection
    assert {"score_type", "calibration_status", "config_profile"} <= staves


def _topology_row(record_id: str, label: str, **updates: float | str) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "record_id": record_id,
        "pdb_id": record_id[:4],
        "domain_id": record_id,
        "fold_label_final": label,
        "betlas_parse_ok": 1,
        "betlas_top_fold": label,
        "betlas_sheet_count": 2.0,
        "betlas_sheet_face_count": 2.0,
        "betlas_sheet_pair_top2_fraction": 0.9,
        "betlas_sheet_pair_size_balance": 0.9,
        "betlas_sheet_pair_bilayer_score": 0.15,
        "betlas_sandwich_lobe_guard_score": 0.16,
        "betlas_sheet_pair_face_alignment": 0.8,
        "betlas_sheet_pair_normal_abs_dot": 0.8,
        "betlas_sheet_pair_cross_contact_density8": 0.25,
        "betlas_axis_best_slice_coverage_median": 0.72,
        "betlas_axis_best_angular_coverage": 0.76,
        "betlas_axis_best_largest_gap_fraction": 0.24,
        "betlas_axis_best_slice_largest_gap_fraction_mean": 0.32,
        "betlas_barrel_wall_continuity_score": 0.12,
        "betlas_contact8_cycle_rank_norm": 0.08,
        "betlas_contact8_degree2_fraction": 0.65,
        "betlas_angular_sector_occupancy12": 0.65,
        "betlas_axis_best_slice_high_coverage_fraction": 0.4,
        "betlas_sheet_seq_top2_interleave_score": 0.1,
        "betlas_sheet_seq_top2_order_displacement": 0.2,
        "betlas_top2_sheet_order_nonlocal_fraction": 0.2,
        "betlas_top2_sheet_order_inversion_fraction_mean": 0.1,
        "betlas_sheet_seq_greek_key_proxy": 0.05,
        "betlas_jelly_roll_order_nonlocal_score": 0.03,
        "betlas_beta_run_eight_score": 0.4,
        "betlas_angular_fft_k3_8_max": 0.1,
        "betlas_angular_fft_k3_8_best_k": 6.0,
        "betlas_axis_periodicity_score": 0.0,
        "betlas_pca_elongation": 1.2,
        "betlas_beta_alpha_alternation_fraction": 0.1,
        "betlas_alpha_shell_radial_delta": 0.0,
    }
    for fold_label in (
        "beta_barrel",
        "beta_prism",
        "beta_propeller",
        "jelly_roll",
        "beta_solenoid",
        "beta_sandwich",
        "tim_like_beta_alpha_barrel",
    ):
        row[f"betlas_rule_score_{fold_label}"] = 0.0
    row[f"betlas_rule_score_{label}"] = 2.0
    row.update(updates)
    return row


def test_topology_diagnostics_scores_continuous_and_mixed_signals():
    rows = [
        _topology_row(
            "ambiguousA",
            "beta_sandwich",
            betlas_top_fold="jelly_roll",
            betlas_rule_score_jelly_roll=3.4,
            betlas_rule_score_beta_sandwich=3.2,
            betlas_sheet_seq_top2_interleave_score=0.8,
            betlas_sheet_seq_top2_order_displacement=0.7,
            betlas_top2_sheet_order_nonlocal_fraction=0.9,
            betlas_top2_sheet_order_inversion_fraction_mean=0.6,
            betlas_sheet_seq_greek_key_proxy=0.6,
            betlas_jelly_roll_order_nonlocal_score=0.5,
            betlas_beta_run_eight_score=1.0,
        ),
        _topology_row(
            "barrelB",
            "beta_barrel",
            betlas_rule_score_beta_barrel=3.0,
            betlas_axis_best_slice_coverage_median=0.9,
            betlas_axis_best_angular_coverage=0.9,
            betlas_axis_best_largest_gap_fraction=0.1,
            betlas_axis_best_slice_largest_gap_fraction_mean=0.18,
            betlas_barrel_wall_continuity_score=0.55,
            betlas_angular_sector_occupancy12=1.0,
            betlas_axis_best_slice_high_coverage_fraction=0.9,
            betlas_sandwich_lobe_guard_score=0.04,
        ),
        _topology_row("sandC", "beta_sandwich", betlas_rule_score_beta_sandwich=3.0),
    ]

    diagnostics = compute_topology_diagnostics(pd.DataFrame(rows), k_neighbors=2)
    ambiguous = diagnostics[diagnostics["record_id"] == "ambiguousA"].iloc[0]
    barrel = diagnostics[diagnostics["record_id"] == "barrelB"].iloc[0]

    assert ambiguous["betlas_jelly_rollness"] > 0.55
    assert ambiguous["betlas_sandwichness"] > 0.45
    assert ambiguous["betlas_topology_ambiguity_score"] > 0.4
    assert ambiguous["betlas_mixed_topology_flag"] == 1
    assert "jelly_roll_like_sandwich_subtopology" in ambiguous["betlas_mixed_topology_types"]
    assert barrel["betlas_barrel_likeness"] > 0.7


def test_topology_diagnostics_columns_are_excluded_from_ml_features():
    df = pd.DataFrame(
        [
            {
                "record_id": "x",
                "fold_label_final": "beta_barrel",
                "betlas_parse_ok": 1,
                "betlas_axis_best_angular_coverage": 0.8,
                "betlas_topology_ambiguity_score": 0.9,
                "betlas_boundary_region_flag": 1,
                "betlas_jelly_rollness": 0.7,
                "betlas_mixed_topology_score": 0.6,
                "betlas_neighbor_label_entropy": 0.5,
            }
        ]
    )

    columns = numeric_feature_columns(df)

    assert "betlas_axis_best_angular_coverage" in columns
    assert "betlas_topology_ambiguity_score" not in columns
    assert "betlas_boundary_region_flag" not in columns
    assert "betlas_jelly_rollness" not in columns
    assert "betlas_mixed_topology_score" not in columns
    assert "betlas_neighbor_label_entropy" not in columns


def test_topology_diagnostics_errors_on_missing_explicit_predictions(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    pd.DataFrame([_topology_row("sandC", "beta_sandwich")]).to_csv(features, index=False)

    with pytest.raises(FileNotFoundError, match="prediction CSV does not exist"):
        run_topology_diagnostics(
            features_csv=features,
            predictions_csv=tmp_path / "missing_predictions.csv",
            out_csv=tmp_path / "topology.csv",
            write_manifest=False,
            predictions_required=True,
        )


def test_topology_diagnostics_errors_when_explicit_predictions_do_not_match(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame([_topology_row("sandC", "beta_sandwich")]).to_csv(features, index=False)
    pd.DataFrame(
        [
            {
                "record_id": "other",
                "prob_beta_sandwich": 1.0,
                "prob_beta_barrel": 0.0,
                "pred_probability": 1.0,
            }
        ]
    ).to_csv(predictions, index=False)

    with pytest.raises(ValueError, match="did not match every parse-ok feature row"):
        run_topology_diagnostics(
            features_csv=features,
            predictions_csv=predictions,
            out_csv=tmp_path / "topology.csv",
            write_manifest=False,
            predictions_required=True,
        )


def test_topology_diagnostics_errors_when_explicit_predictions_lack_probability_signal(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame([_topology_row("sandC", "beta_sandwich")]).to_csv(features, index=False)
    pd.DataFrame([{"record_id": "sandC", "model": "hist_gradient_boosting"}]).to_csv(predictions, index=False)

    with pytest.raises(ValueError, match="lacks usable probability"):
        run_topology_diagnostics(
            features_csv=features,
            predictions_csv=predictions,
            out_csv=tmp_path / "topology.csv",
            write_manifest=False,
            predictions_required=True,
        )


def test_topology_diagnostics_marks_rule_softmax_as_uncalibrated() -> None:
    diagnostics = compute_topology_diagnostics(pd.DataFrame([_topology_row("sandC", "beta_sandwich")]))

    assert diagnostics.loc[0, "betlas_probability_source"] == "rule_softmax"
    assert diagnostics.loc[0, "betlas_probability_calibration_status"] == "uncalibrated_rule_softmax"


def test_topology_diagnostics_parse_failed_rows_are_status_only() -> None:
    diagnostics = compute_topology_diagnostics(
        pd.DataFrame(
            [
                {
                    "record_id": "bad",
                    "pdb_id": "bad1",
                    "domain_id": "bad",
                    "betlas_parse_ok": 0,
                    "betlas_error": "no beta-sheet segments",
                    "betlas_top_fold": "beta_sandwich",
                    "betlas_rule_score_beta_sandwich": 10.0,
                }
            ]
        )
    )

    assert diagnostics.loc[0, "betlas_topology_status"] == "parse_failed"
    assert "no beta-sheet" in diagnostics.loc[0, "betlas_topology_error"]
    assert "betlas_topology_ambiguity_score" in diagnostics.columns
    assert "betlas_jelly_rollness" in diagnostics.columns
    assert diagnostics.loc[0, "betlas_topology_ambiguity_score"] == ""
    assert diagnostics.loc[0, "betlas_jelly_rollness"] == ""


def test_topology_diagnostics_no_informative_slices_rows_are_status_only() -> None:
    row = _topology_row(
        "zeroSlice",
        "beta_sandwich",
        betlas_score_status="no_informative_slices",
        betlas_axis_best_slice_count=0.0,
    )

    diagnostics = compute_topology_diagnostics(pd.DataFrame([row]))

    assert diagnostics.loc[0, "betlas_topology_status"] == "no_informative_slices"
    assert "betlas_score_status" in diagnostics.loc[0, "betlas_topology_error"]
    assert diagnostics.loc[0, "betlas_probability_top1"] == ""
    assert diagnostics.loc[0, "betlas_mixed_topology_score"] == ""


def test_topology_diagnostics_rule_score_only_rows_are_status_only() -> None:
    row = {
        "record_id": "rules_only",
        "pdb_id": "rule",
        "domain_id": "rules_only",
        "fold_label_final": "beta_barrel",
        "betlas_parse_ok": 1,
        "betlas_score_status": "ok",
        "betlas_top_fold": "beta_barrel",
        "betlas_rule_score_beta_barrel": 3.0,
        "betlas_rule_score_beta_prism": 0.0,
        "betlas_rule_score_beta_propeller": 0.0,
        "betlas_rule_score_jelly_roll": 0.0,
        "betlas_rule_score_beta_solenoid": 0.0,
        "betlas_rule_score_beta_sandwich": 0.0,
        "betlas_rule_score_tim_like_beta_alpha_barrel": 0.0,
    }

    diagnostics = compute_topology_diagnostics(pd.DataFrame([row]))

    assert diagnostics.loc[0, "betlas_topology_status"] == "missing_topology_geometry"
    assert "raw Betlas geometry columns" in diagnostics.loc[0, "betlas_topology_error"]
    assert diagnostics.loc[0, "betlas_barrel_likeness"] == ""
    assert diagnostics.loc[0, "betlas_probability_source"] == ""


def test_topology_diagnostics_explicit_out_controls_default_manifest_path(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    out_csv = tmp_path / "requested" / "topology.csv"
    stale_manifest = tmp_path / "stale" / "topology.manifest.json"
    config = tmp_path / "topology.yaml"
    pd.DataFrame([_topology_row("sandC", "beta_sandwich")]).to_csv(features, index=False)
    config.write_text(
        "\n".join(
            [
                "io:",
                f"  manifest_path: {stale_manifest}",
                "  output_csv: should_not_be_used.csv",
            ]
        ),
        encoding="utf-8",
    )

    result = run_topology_diagnostics(
        features_csv=features,
        out_csv=out_csv,
        config_path=config,
    )

    assert result.manifest_path == out_csv.with_suffix(".csv.manifest.json")
    assert result.manifest_path.exists()
    assert not stale_manifest.exists()


def test_topology_cli_does_not_load_config_predictions_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features = tmp_path / "features.csv"
    predictions = tmp_path / "predictions.csv"
    out_csv = tmp_path / "topology.csv"
    config = tmp_path / "topology.yaml"
    pd.DataFrame([_topology_row("sandC", "beta_sandwich")]).to_csv(features, index=False)
    pd.DataFrame(
        [
            {
                "record_id": "sandC",
                "model": "hist_gradient_boosting",
                "prob_beta_barrel": 1.0,
                "prob_beta_sandwich": 0.0,
            }
        ]
    ).to_csv(predictions, index=False)
    config.write_text(
        "\n".join(
            [
                "io:",
                f"  predictions_csv: {predictions}",
            ]
        ),
        encoding="utf-8",
    )

    topology_diagnostics_cli(
        [
            "--config",
            str(config),
            "--features",
            str(features),
            "--out",
            str(out_csv),
        ]
    )

    capsys.readouterr()
    manifest = json.loads(out_csv.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["predictions_loaded"] is False
