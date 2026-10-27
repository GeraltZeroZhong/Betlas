from __future__ import annotations

import pandas as pd
import pytest

from betlas import cli
from betlas.constants import FOLD_LABELS
from betlas.features.current import FEATURE_COLUMNS, FEATURE_FAMILIES
from betlas.grammars import compute_grammar_features, get_grammar, list_grammars
from betlas.models import DomainCandidate, StructureGeometry
from betlas.schema import SchemaCompatibilityWarning, normalize_feature_columns
from betlas.specs import describe_column, describe_feature, list_column_specs, list_feature_specs

def test_normalize_feature_columns_accepts_explicit_column_aliases() -> None:
    df = pd.DataFrame(
        {
            "record_id": ["r1"],
            "input_status": ["1"],
            "input_coverage": ["0.8"],
        }
    )
    aliases = {
        "input_status": "betlas_parse_ok",
        "input_coverage": "betlas_axis_best_angular_coverage",
    }

    normalized = normalize_feature_columns(df, aliases=aliases)

    assert "input_status" not in normalized
    assert normalized.loc[0, "betlas_parse_ok"] == "1"
    assert normalized.loc[0, "betlas_axis_best_angular_coverage"] == "0.8"
    assert df.columns.tolist() == ["record_id", "input_status", "input_coverage"]


def test_normalize_feature_columns_prefers_canonical_column_and_warns_on_conflict() -> None:
    df = pd.DataFrame({"betlas_parse_ok": ["1"], "input_status": ["0"]})
    aliases = {"input_status": "betlas_parse_ok"}

    with pytest.warns(SchemaCompatibilityWarning):
        normalized = normalize_feature_columns(df, aliases=aliases)

    assert normalized.loc[0, "betlas_parse_ok"] == "1"
    assert "input_status" not in normalized
    with pytest.raises(ValueError, match="conflicts"):
        normalize_feature_columns(df, aliases=aliases, strict=True)


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


def test_registry_is_an_exact_partition_of_current_descriptors():
    specs=list_grammars()
    assert {spec.name for spec in specs}==set(FEATURE_FAMILIES)
    columns=[column for spec in specs for column in spec.columns]
    assert len(columns)==len(set(columns))==136
    assert set(columns)==set(FEATURE_COLUMNS)
    assert all(spec.math_summary and spec.inputs and spec.outputs for spec in specs)
    assert get_grammar("axis_periodicity").columns == tuple(f"betlas_angular_circular_moment_k{k}" for k in range(3,9))


def test_feature_dictionary_uses_fixed_current_order_and_rejects_unknowns():
    specs=list_feature_specs()
    assert tuple(spec.name for spec in specs)==FEATURE_COLUMNS
    assert all(spec.family in FEATURE_FAMILIES for spec in specs)
    assert "0.40 * slice_coverage_median" in describe_feature("betlas_axis_best_score").formula
    assert describe_column("record_id").required
    assert "feature_schema" in {spec.name for spec in list_column_specs()}
    with pytest.raises(KeyError):
        describe_feature("unknown_column")


def test_cli_grammar_description_resolves_current_columns(capsys):
    cli.main(["grammar","describe","axis_periodicity","--format","json"])
    text=capsys.readouterr().out
    assert "betlas_angular_circular_moment_k8" in text
    assert "fft" not in text
