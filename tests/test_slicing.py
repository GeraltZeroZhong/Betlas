from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from betlas import cli
from betlas.constants import FOLD_LABELS
from betlas.features.extract import extract_signature
from betlas.models import (
    DomainCandidate,
    ResidueRecord,
    SecondaryStructureElement,
    SheetPatch,
    StructureGeometry,
)
from betlas.slicing import list_slice_axes, slice_feature_row, slice_structure, summarize_slices


def _write_minimal_mmcif(path: Path) -> None:
    atom_rows: list[str] = []
    coords = [
        (1, 0.0, 0.0, 0.0),
        (2, 0.0, 0.0, 2.0),
        (3, 0.0, 0.0, 4.0),
        (4, 0.0, 0.0, 6.0),
        (11, 8.0, 0.0, 0.0),
        (12, 8.0, 0.0, 2.0),
        (13, 8.0, 0.0, 4.0),
        (14, 8.0, 0.0, 6.0),
    ]
    for index, (seq_id, x, y, z) in enumerate(coords, start=1):
        atom_rows.append(
            f"ATOM {index} C CA CA . ALA A A {seq_id} {seq_id} ? {x:.3f} {y:.3f} {z:.3f} 1.00"
        )
    path.write_text(
        "\n".join(
            [
                "data_betlas_test",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.auth_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.label_asym_id",
                "_atom_site.auth_seq_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                *atom_rows,
                "loop_",
                "_struct_sheet_range.sheet_id",
                "_struct_sheet_range.id",
                "_struct_sheet_range.beg_auth_asym_id",
                "_struct_sheet_range.end_auth_asym_id",
                "_struct_sheet_range.beg_auth_seq_id",
                "_struct_sheet_range.end_auth_seq_id",
                "S1 1 A A 1 4",
                "S1 2 A A 11 14",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_no_sheet_mmcif(path: Path) -> None:
    atom_rows = [
        f"ATOM {index} C CA CA . ALA A A {index} {index} ? {float(index):.3f} 0.000 0.000 1.00"
        for index in range(1, 9)
    ]
    path.write_text(
        "\n".join(
            [
                "data_betlas_no_sheet",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.auth_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.label_asym_id",
                "_atom_site.auth_seq_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                *atom_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _ring_geometry() -> StructureGeometry:
    domain = DomainCandidate(
        record_id="ring",
        pdb_id="ring",
        chain_id="A",
        domain_id="ringA",
        residue_ranges="",
        fold_label_final=FOLD_LABELS[0],
        evidence_level="unit",
        label_source_primary="unit",
    )
    residues: list[ResidueRecord] = []
    segments: list[SecondaryStructureElement] = []
    radius = 10.0
    z_values = [0.0, 2.0, 4.0, 6.0]
    angles = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    for strand_index, angle in enumerate(angles):
        indices: list[int] = []
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        for z_index, z in enumerate(z_values):
            indices.append(len(residues))
            residues.append(
                ResidueRecord(
                    pdb_id="ring",
                    chain_id="A",
                    auth_seq_id=1 + strand_index * 10 + z_index,
                    label_seq_id=1 + strand_index * 10 + z_index,
                    insertion_code="",
                    residue_name="ALA",
                    coord_ca=(x, y, z),
                )
            )
        segments.append(
            SecondaryStructureElement(
                element_id=f"B{strand_index + 1:03d}",
                element_type="strand",
                chain_id="A",
                start_auth_seq_id=residues[indices[0]].auth_seq_id,
                end_auth_seq_id=residues[indices[-1]].auth_seq_id,
                residue_indices=tuple(indices),
                sheet_id="S1",
                sheet_range_id=str(strand_index + 1),
            )
        )
    return StructureGeometry(
        domain=domain,
        residues=tuple(residues),
        beta_segments=tuple(segments),
        helices=(),
        sheet_patches=(SheetPatch(sheet_id="S1", strand_ids=tuple(segment.element_id for segment in segments)),),
    )


def _single_strand_geometry() -> StructureGeometry:
    geometry = _ring_geometry()
    segment = geometry.beta_segments[0]
    segment = SecondaryStructureElement(
        element_id=segment.element_id,
        element_type=segment.element_type,
        chain_id=segment.chain_id,
        start_auth_seq_id=segment.start_auth_seq_id,
        end_auth_seq_id=geometry.residues[2].auth_seq_id,
        residue_indices=(0, 1, 2),
        sheet_id=segment.sheet_id,
        sheet_range_id=segment.sheet_range_id,
    )
    return StructureGeometry(
        domain=geometry.domain,
        residues=geometry.residues[:3],
        beta_segments=(segment,),
        helices=(),
        sheet_patches=(SheetPatch(sheet_id="S1", strand_ids=(segment.element_id,)),),
    )


def test_slice_structure_reports_axis_aligned_ring_slices() -> None:
    bundle = slice_structure(_ring_geometry(), axis="best")
    summary = summarize_slices(bundle)

    assert bundle.axis_name in list_slice_axes(_ring_geometry())
    assert len(bundle.slices) == 4
    assert summary["slice_count"] == 4.0
    assert summary["z_continuity_fraction"] == 1.0
    assert summary["slice_coverage_median"] == pytest.approx(0.75)
    assert all(item.point_count == 4 for item in bundle.slices)
    assert all(len(item.points) == 4 for item in bundle.slices)
    first_point = bundle.slices[0].points[0]
    assert first_point.auth_seq_id in {1, 11, 21, 31}
    assert first_point.residue_uid.startswith("A:")
    assert first_point.strand_id.startswith("B")
    assert first_point.sheet_id == "S1"


def test_slice_feature_row_matches_existing_extraction_columns() -> None:
    geometry = _ring_geometry()
    row = slice_feature_row(geometry)
    features = extract_signature(geometry).features

    for key in [
        "betlas_axis_best_name",
        "betlas_axis_best_score",
        "betlas_axis_best_angular_coverage",
        "betlas_axis_best_largest_gap_fraction",
        "betlas_axis_best_slice_count",
        "betlas_axis_best_slice_coverage_median",
        "betlas_axis_best_slice_largest_gap_fraction_mean",
        "betlas_z_continuity_fraction",
    ]:
        assert row[key] == features[key]


def test_extract_signature_zero_informative_slices_is_not_rule_scored() -> None:
    features = extract_signature(_single_strand_geometry()).features

    assert features["betlas_parse_ok"] == 1
    assert features["betlas_axis_best_slice_count"] == 0.0
    assert features["betlas_score_status"] == "no_informative_slices"
    assert "betlas_top_fold" not in features
    assert not any(column.startswith("betlas_rule_score_") for column in features)


def test_slice_cli_help_is_public(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["slice", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "slice-closure summaries" in out
    assert "--chain" in out
    assert "--summary-out" in out


def test_extract_features_single_structure_cli_feeds_grammar_score(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    structure = tmp_path / "mini.cif"
    features = tmp_path / "features.csv"
    scores = tmp_path / "scores.csv"
    _write_minimal_mmcif(structure)

    cli.main(["extract-features", "--structure", str(structure), "--chain", "A", "--out", str(features)])
    row = pd.read_csv(features).iloc[0]

    assert row["record_id"] == "mini_A"
    assert row["betlas_parse_ok"] == 1
    assert "Wrote 1 feature row" in capsys.readouterr().out

    cli.main(["grammar", "score", "--features", str(features), "--out", str(scores)])
    scored = pd.read_csv(scores)
    assert "betlas_top_fold" in scored
    assert "betlas_rule_score_beta_barrel" in scored

    points = tmp_path / "points.csv"
    cli.main(["slice", str(structure), "--chain", "A", "--points-out", str(points)])
    point_columns = set(pd.read_csv(points).columns)
    assert {"auth_seq_id", "residue_uid", "strand_id", "sheet_id", "sheet_range_id"} <= point_columns


def test_extract_features_structure_no_sheet_fails_or_writes_status_only_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    structure = tmp_path / "no_sheet.cif"
    features = tmp_path / "no_sheet_features.csv"
    _write_no_sheet_mmcif(structure)

    with pytest.raises(SystemExit) as exc:
        cli.main(["extract-features", "--structure", str(structure), "--chain", "A", "--out", str(features)])
    assert exc.value.code == 2
    assert "requires parsed beta-sheet segments" in capsys.readouterr().err
    assert not features.exists()

    cli.main(
        [
            "extract-features",
            "--structure",
            str(structure),
            "--chain",
            "A",
            "--write-failed-row",
            "--out",
            str(features),
        ]
    )
    row = pd.read_csv(features).iloc[0]
    assert row["betlas_parse_ok"] == 0
    assert "betlas_top_fold" not in row.index
    assert not any(column.startswith("betlas_rule_score_") for column in row.index)
    assert row["source_mmcif_sha256"]


def test_examples_cli_copies_packaged_mini_structure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["examples", "list"])
    assert "mini" in capsys.readouterr().out

    cli.main(["examples", "copy", "mini", "--out-dir", str(tmp_path)])
    copied = tmp_path / "mini.cif"

    assert copied.exists()
    assert "data_betlas_mini" in copied.read_text(encoding="utf-8")


def test_chains_cli_inspects_author_and_label_chains(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    structure = tmp_path / "mini.cif"
    _write_minimal_mmcif(structure)

    cli.main(["chains", str(structure)])
    out = capsys.readouterr().out
    assert "auth_chain_id" in out
    assert "insertion_code_ca_count" in out
    assert "nonpolymer_atom_rows" in out
    assert "feature_extraction_supported" in out

    cli.main(["structure", "inspect", str(structure), "--format", "json"])
    json_out = capsys.readouterr().out
    assert '"auth_chain_id": "A"' in json_out
    assert '"sheet_annotation_available": true' in json_out


def test_slice_cli_zero_informative_slices_writes_status_and_headers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    structure = tmp_path / "mini.cif"
    slices = tmp_path / "slices.csv"
    points = tmp_path / "points.csv"
    summary = tmp_path / "summary.json"
    _write_minimal_mmcif(structure)

    cli.main(
        [
            "slice",
            str(structure),
            "--chain",
            "A",
            "--min-points-per-slice",
            "99",
            "--out",
            str(slices),
            "--points-out",
            str(points),
            "--summary-out",
            str(summary),
        ]
    )
    assert "Wrote 0 Betlas slice rows" in capsys.readouterr().out
    slice_frame = pd.read_csv(slices)
    assert slice_frame.empty
    assert "status" in slice_frame.columns
    payload = summary.read_text(encoding="utf-8")
    assert '"status": "no_informative_slices"' in payload
    point_frame = pd.read_csv(points)
    assert set(point_frame["included"]) == {0}
    assert set(point_frame["exclusion_reason"]) == {"no_informative_slices"}


def test_grammar_score_strict_rejects_incomplete_feature_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features = tmp_path / "incomplete.csv"
    permissive = tmp_path / "permissive.csv"
    pd.DataFrame(
        [{"record_id": "r1", "betlas_axis_best_slice_coverage_median": 0.5}]
    ).to_csv(features, index=False)

    with pytest.raises(SystemExit) as exc:
        cli.main(["grammar", "score", "--features", str(features)])
    assert exc.value.code == 2
    assert "requires betlas_parse_ok" in capsys.readouterr().err

    cli.main(["grammar", "score", "--features", str(features), "--no-strict", "--out", str(permissive)])
    assert "betlas_top_fold" in pd.read_csv(permissive).columns


def test_grammar_score_rejects_parse_failed_rows_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features = tmp_path / "parse_failed.csv"
    permissive = tmp_path / "parse_failed_scored.csv"
    pd.DataFrame(
        [
            {
                "record_id": "r1",
                "chain_id": "A",
                "betlas_parse_ok": 0,
                "betlas_error": "no beta-sheet segments",
                "betlas_axis_best_slice_coverage_median": 0.5,
            }
        ]
    ).to_csv(features, index=False)

    with pytest.raises(SystemExit) as exc:
        cli.main(["grammar", "score", "--features", str(features), "--no-strict"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "refused parse-failed feature rows" in captured.err
    assert "Traceback" not in captured.err

    cli.main(
        [
            "grammar",
            "score",
            "--features",
            str(features),
            "--no-strict",
            "--allow-parse-fail",
            "--out",
            str(permissive),
        ]
    )
    scored = pd.read_csv(permissive)
    assert scored.loc[0, "betlas_parse_ok"] == 0
    assert scored.loc[0, "betlas_score_status"] == "parse_failed"
    assert "betlas_top_fold" not in scored.columns
    assert not any(column.startswith("betlas_rule_score_") for column in scored.columns)


def test_slice_structure_fails_when_no_beta_segments() -> None:
    domain = DomainCandidate(
        record_id="empty",
        pdb_id="test",
        chain_id="A",
        domain_id="A",
        residue_ranges="",
        fold_label_final=FOLD_LABELS[0],
        evidence_level="unit",
        label_source_primary="unit",
    )
    geometry = StructureGeometry(
        domain=domain,
        residues=(),
        beta_segments=(),
        helices=(),
        sheet_patches=(),
        warnings=("no sheet annotations",),
    )

    with pytest.raises(ValueError, match="parsed beta-sheet segments"):
        slice_structure(geometry)


def test_slice_cli_wrong_chain_reports_user_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    structure = tmp_path / "mini.cif"
    _write_minimal_mmcif(structure)

    with pytest.raises(SystemExit) as exc:
        cli.main(["slice", str(structure), "--chain", "Z"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "available author chain ids: A" in captured.err
