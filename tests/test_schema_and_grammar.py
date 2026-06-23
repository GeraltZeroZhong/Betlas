from __future__ import annotations

import pandas as pd
import pytest

from betlas import cli
from betlas.constants import FOLD_LABELS
from betlas.features.rules import GRAMMAR_RULE_INPUT_COLUMNS, grammar_rule_scores
from betlas.grammars import (
    compute_grammar_features,
    get_grammar,
    list_grammars,
    score_fold_grammar,
)
from betlas.models import DomainCandidate, StructureGeometry
from betlas.schema import (
    SchemaCompatibilityWarning,
    normalize_feature_columns,
)
from betlas.specs import (
    describe_column,
    describe_feature,
    describe_readout_column,
    list_column_specs,
    list_feature_specs,
    list_readout_column_specs,
    list_rule_specs,
)


def test_normalize_feature_columns_accepts_compatibility_input() -> None:
    df = pd.DataFrame(
        {
            "record_id": ["r1"],
            "cz_parse_ok": ["1"],
            "cz_axis_best_angular_coverage": ["0.8"],
        }
    )

    normalized = normalize_feature_columns(df)

    assert "cz_parse_ok" not in normalized
    assert normalized.loc[0, "betlas_parse_ok"] == "1"
    assert normalized.loc[0, "betlas_axis_best_angular_coverage"] == "0.8"


def test_normalize_feature_columns_prefers_betlas_and_warns_on_conflict() -> None:
    df = pd.DataFrame({"betlas_parse_ok": ["1"], "cz_parse_ok": ["0"]})

    with pytest.warns(SchemaCompatibilityWarning):
        normalized = normalize_feature_columns(df)

    assert normalized.loc[0, "betlas_parse_ok"] == "1"
    assert "cz_parse_ok" not in normalized
    with pytest.raises(ValueError, match="conflicts"):
        normalize_feature_columns(df, strict=True)


def test_score_fold_grammar_matches_existing_rule_scores_for_compatibility_input() -> None:
    betlas_features = {
        "betlas_axis_best_slice_coverage_median": 0.9,
        "betlas_axis_best_angular_coverage": 0.85,
        "betlas_axis_best_largest_gap_fraction": 0.1,
        "betlas_beta_strand_count": 8,
    }
    legacy_features = {
        key.replace("betlas_", "cz_", 1): value for key, value in betlas_features.items()
    }

    with pytest.raises(ValueError, match="canonical betlas_"):
        score_fold_grammar(legacy_features)
    assert score_fold_grammar(legacy_features, strict=False) == grammar_rule_scores(
        betlas_features,
        strict=False,
    )


def test_score_fold_grammar_strict_rejects_parse_failed_feature_rows() -> None:
    row = {column: 1.0 for column in GRAMMAR_RULE_INPUT_COLUMNS}
    row["betlas_axis_best_slice_count"] = 1.0
    row["betlas_parse_ok"] = 0

    with pytest.raises(ValueError, match="parse-failed"):
        score_fold_grammar(row)
    with pytest.raises(ValueError, match="parse-failed"):
        grammar_rule_scores(row)

    assert "beta_barrel" in score_fold_grammar(row, strict=False)


def test_score_fold_grammar_strict_requires_parse_ok_and_informative_slices() -> None:
    row = {column: 1.0 for column in GRAMMAR_RULE_INPUT_COLUMNS}
    row["betlas_axis_best_slice_count"] = 1.0

    with pytest.raises(ValueError, match="requires betlas_parse_ok"):
        score_fold_grammar(row)

    row["betlas_parse_ok"] = 1
    row["betlas_axis_best_slice_count"] = 0
    with pytest.raises(ValueError, match="no informative slices"):
        score_fold_grammar(row)


def test_grammar_registry_covers_public_grammar_families() -> None:
    expected = {
        "composition",
        "axis_closure",
        "sheet_geometry",
        "beta_run_topology",
        "sheet_pair_packing",
        "sheet_sequence_topology",
        "sheet_order_topology",
        "contact_graph",
        "contact_sequence_topology",
        "axis_periodicity",
        "strand_order",
        "alpha_shell",
        "angular_lobes",
        "fold_rule_scores",
        "continuous_topology",
        "topology_ambiguity",
        "mixed_topology",
    }

    specs = list_grammars()

    assert {spec.name for spec in specs} == expected
    assert all(spec.columns or spec.column_prefixes for spec in specs)
    assert all(spec.math_summary for spec in specs)
    assert all(spec.inputs for spec in specs)
    assert all(spec.outputs for spec in specs)
    assert get_grammar("composition").name == "composition"


def test_rule_specs_use_fold_specific_inputs_and_exact_terms() -> None:
    specs = {spec.fold_label: spec for spec in list_rule_specs()}

    assert "topology-order terms" not in specs["jelly_roll"].formula
    assert "betlas_sheet_seq_greek_key_proxy" in specs["jelly_roll"].inputs
    assert "betlas_helix_count" not in specs["beta_barrel"].inputs
    assert len({spec.inputs for spec in specs.values()}) > 1


def test_topology_readout_specs_describe_string_and_unit_interval_columns() -> None:
    competitor = describe_readout_column("betlas_boundary_neighbor_top_competitor", "topology-diagnostics")
    ambiguity = describe_readout_column("betlas_topology_ambiguity_score", "topology-diagnostics")
    jelly = describe_readout_column("betlas_jelly_rollness", "topology-diagnostics")
    rule_margin = describe_readout_column("betlas_rule_probability_margin", "topology-diagnostics")

    assert competitor.dtype == "string"
    assert competitor.value_range.startswith("one Betlas fold label")
    assert ambiguity.value_range == "[0, 1]"
    assert "rule-softmax" in ambiguity.definition
    assert jelly.value_range == "[0, 1]"
    assert "calibrated probability" in rule_margin.definition
    assert "deterministic geometry extraction pipeline" not in rule_margin.definition


def test_compute_grammar_features_filters_by_registry_prefixes() -> None:
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
    )

    features = compute_grammar_features(geometry, "composition")

    assert "betlas_residue_count" in features
    assert "betlas_axis_best_score" not in features


def test_cli_help_exposes_only_public_core_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in [
        "build-dataset",
        "extract-features",
        "benchmark",
        "ablate",
        "grammar",
        "slice",
        "readout",
        "assets",
        "examples",
    ]:
        assert command in out
    for internal in [
        "science-report",
        "publication-evidence",
        "external-baselines",
        "validate-staves-data",
        "curate-staves-publication-set",
        "repair-feature-table",
        "data-manifest",
    ]:
        assert internal not in out


def test_cli_grammar_and_readout_list_commands(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["grammar", "list"])
    grammar_out = capsys.readouterr().out
    assert "composition" in grammar_out
    assert "mixed_topology" in grammar_out

    cli.main(["readout", "list"])
    readout_out = capsys.readouterr().out
    assert "topology-diagnostics" in readout_out
    assert "bfvd-viral-beta-fold-scan" not in readout_out


def test_cli_grammar_describe_json_includes_math_summary(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["grammar", "describe", "axis_closure", "--format", "json"])

    out = capsys.readouterr().out
    assert '"name": "axis_closure"' in out
    assert "angular coverage" in out
    assert "resolved_columns" in out


def test_continuous_topology_grammar_matches_overlap_implementation_summary() -> None:
    spec = get_grammar("continuous_topology")

    assert "min(score_a, score_b) * (1 - abs(score_a - score_b))" in spec.math_summary
    assert "betlas_jelly_rollness_basis_json" in spec.columns
    assert "betlas_barrel_jelly_overlap" in spec.columns
    assert "betlas_barrel_jelly_overlap" in spec.to_dict()["resolved_columns"]


def test_compute_grammar_features_rejects_readout_only_grammars() -> None:
    with pytest.raises(ValueError, match="readout-only grammars"):
        compute_grammar_features(StructureGeometry.__new__(StructureGeometry), grammars="continuous_topology")


def test_feature_and_rule_specs_cover_core_columns() -> None:
    specs = {spec.name: spec for spec in list_feature_specs()}

    assert "betlas_axis_best_score" in specs
    assert specs["betlas_axis_best_score"].family == "axis_closure"
    assert "0.40 * slice_coverage_median" in specs["betlas_axis_best_score"].formula
    assert describe_feature("betlas_axis_best_score").name == "betlas_axis_best_score"

    rule_specs = list_rule_specs()
    assert {spec.name for spec in rule_specs} >= {"betlas_rule_score_beta_barrel"}
    assert all(spec.inputs for spec in rule_specs)


def test_column_specs_cover_protocol_and_readout_columns() -> None:
    names = {spec.name for spec in list_column_specs()}

    assert "source_mmcif_sha256" in names
    assert "chain_id" in names
    assert "betlas_angular_fft_k8" in names
    assert "betlas_axis_point_pc1_slice_count" in names
    assert "decision_score" in {spec.name for spec in list_readout_column_specs("beta-barrel-detection")}
    assert "confidence" in {spec.name for spec in list_readout_column_specs("beta-barrel-staves")}
    assert "betlas_topology_ambiguity_score" in {
        spec.name for spec in list_readout_column_specs("topology-ambiguity")
    }
    assert "betlas_jelly_rollness" in {
        spec.name for spec in list_readout_column_specs("fold-continuous-scores")
    }
    assert "not a calibrated probability" in describe_readout_column(
        "decision_score",
        "beta-barrel-detection",
    ).definition
    assert describe_column("record_id").required is True
    assert describe_readout_column(
        "betlas_jelly_sandwich_overlap",
        "fold-continuous-scores",
    ).definition.startswith("Continuous topology overlap")


def test_feature_specs_assign_all_betlas_columns_to_public_families() -> None:
    specs = list_feature_specs()
    unassigned = [spec.name for spec in specs if spec.family == "unassigned"]

    assert unassigned == []
    assert describe_feature("betlas_axis_slice_name").dtype == "string"
    assert describe_feature("betlas_axis_point_pc1_slice_count").family == "axis_closure"
    assert "betlas_jelly_rollness" not in {spec.name for spec in specs}
    assert "rule_softmax" in describe_readout_column(
        "betlas_probability_top1",
        "topology-diagnostics",
    ).definition
