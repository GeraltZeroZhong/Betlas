from __future__ import annotations

import inspect
import re
from dataclasses import asdict, dataclass
from typing import Any

from .constants import FOLD_LABELS
from .models import DomainCandidate


@dataclass(frozen=True)
class FeatureSpec:
    """Public data-dictionary entry for one Betlas output column."""

    name: str
    family: str
    dtype: str
    units: str
    value_range: str
    definition: str
    formula: str
    missing_value: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleSpec:
    """Public data-dictionary entry for one transparent fold-rule score."""

    name: str
    fold_label: str
    definition: str
    formula: str
    inputs: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["inputs"] = list(self.inputs)
        return out


@dataclass(frozen=True)
class ColumnSpec:
    """Public table-contract entry for an input or output column."""

    name: str
    table: str
    role: str
    dtype: str
    required: bool
    definition: str
    units: str
    value_range: str
    missing_value: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SUMMARY_PREFIXES = (
    "betlas_sheet_strand_count",
    "betlas_sheet_angular_span",
    "betlas_sheet_planarity_residual",
    "betlas_sheet_axis_dispersion",
    "betlas_beta_run_length",
    "betlas_strand_length",
    "betlas_helix_length",
)
_SUMMARY_SUFFIXES = ("mean", "std", "min", "median", "max")
_AXIS_SLICE_KEYS = (
    "angular_coverage",
    "largest_gap_fraction",
    "radius_cv",
    "z_range",
    "z_continuity_fraction",
    "slice_count",
    "slice_coverage_mean",
    "slice_coverage_median",
    "slice_coverage_std",
    "slice_high_coverage_fraction",
    "slice_largest_gap_fraction_mean",
)
_SLICE_AXIS_NAMES = ("strand_axis", "point_pc1", "point_pc2", "point_pc3")
_SCHEMA_COLUMNS = {
    "betlas_parse_ok",
    "betlas_error",
    "betlas_warnings",
    "betlas_top_fold",
    "betlas_rule_margin",
    "betlas_fold_scores_json",
    "betlas_score_status",
    "betlas_topology_status",
    "betlas_topology_error",
}
_COLUMN_OVERRIDES: dict[str, dict[str, str]] = {
    "betlas_parse_ok": {
        "dtype": "integer",
        "range": "{0, 1}",
        "definition": "Whether Betlas parsed enough residues and beta segments for geometry scoring.",
        "missing": "0 for failed or unavailable geometry rows",
    },
    "betlas_error": {
        "dtype": "string",
        "range": "string",
        "definition": "Machine-readable parse or extraction error text for failed feature rows.",
        "missing": "empty string when no error was recorded",
    },
    "betlas_warnings": {
        "dtype": "string",
        "range": "semicolon-delimited warning string",
        "definition": "Semicolon-delimited nonfatal parser or geometry warnings.",
        "missing": "empty string when no warnings were recorded",
    },
    "betlas_top_fold": {
        "dtype": "string",
        "range": "one Betlas fold label",
        "definition": "Top-scoring fold label from transparent Betlas fold-rule scores.",
        "missing": "empty string when scores are unavailable",
    },
    "betlas_rule_margin": {
        "dtype": "numeric",
        "range": "real-valued",
        "definition": "Difference between the highest and second-highest transparent fold-rule score.",
        "missing": "0 when scores are unavailable",
    },
    "betlas_score_status": {
        "dtype": "string",
        "range": "ok|parse_failed",
        "definition": "Status for grammar scoring rows; parse-failed feature rows are not assigned fold calls or rule scores.",
        "missing": "empty string only for older score tables",
    },
    "betlas_topology_status": {
        "dtype": "string",
        "range": "ok|parse_failed",
        "definition": "Status for topology diagnostic rows; parse-failed feature rows are status-only and not assigned topology readouts.",
        "missing": "empty string only for older topology diagnostic tables",
    },
    "betlas_topology_error": {
        "dtype": "string",
        "range": "string",
        "definition": "Machine-readable reason for topology diagnostic rows that could not be computed.",
        "missing": "empty string when topology diagnostics were computed",
    },
    "betlas_angular_fft_k3_8_best_k": {
        "dtype": "integer",
        "range": "{0, 3, 4, 5, 6, 7, 8}",
        "definition": "Harmonic index with the largest k=3..8 angular Fourier response; 0 when unavailable.",
        "missing": "0 when fewer than four beta-element midpoints are available",
    },
    "betlas_topology_ambiguity_level": {
        "dtype": "string",
        "range": "categorical/string",
        "definition": "Categorical topology-ambiguity level emitted by topology diagnostics.",
        "missing": "empty string when not computed",
    },
    "betlas_topology_ambiguity_reasons": {
        "dtype": "string",
        "range": "semicolon-delimited reason string",
        "definition": "Reasons contributing to the topology-ambiguity status.",
        "missing": "empty string when not computed",
    },
    "betlas_boundary_neighbor_top_competitor": {
        "dtype": "string",
        "range": "one Betlas fold label or empty string",
        "definition": "Most frequent neighboring topology label that differs from the row label in boundary-neighbor diagnostics.",
        "missing": "empty string when no competitor label is available",
    },
    "betlas_probability_top1_label": {
        "dtype": "string",
        "range": "one Betlas fold label or empty string",
        "definition": "Top fold label from explicit prediction probabilities or from uncalibrated rule-softmax weights.",
        "missing": "empty string when probability-like weights are unavailable",
    },
    "betlas_probability_top2_label": {
        "dtype": "string",
        "range": "one Betlas fold label or empty string",
        "definition": "Second fold label from explicit prediction probabilities or from uncalibrated rule-softmax weights.",
        "missing": "empty string when probability-like weights are unavailable",
    },
    "betlas_probability_source": {
        "dtype": "string",
        "range": "prediction|rule_softmax",
        "definition": "Source of betlas_probability_* columns: explicit prediction CSV or softmax-normalized transparent rule scores.",
        "missing": "empty string when topology diagnostics were not computed",
    },
    "betlas_probability_calibration_status": {
        "dtype": "string",
        "range": "model_reported_uncalibrated|uncalibrated_rule_softmax",
        "definition": "Calibration status for betlas_probability_* columns; rule-softmax fallback values are not calibrated probabilities.",
        "missing": "empty string when topology diagnostics were not computed",
    },
    "betlas_rule_top1_label": {
        "dtype": "string",
        "range": "one Betlas fold label or empty string",
        "definition": "Top fold label after softmax normalization of transparent rule scores.",
        "missing": "empty string when rule-softmax weights are unavailable",
    },
    "betlas_rule_top2_label": {
        "dtype": "string",
        "range": "one Betlas fold label or empty string",
        "definition": "Second fold label after softmax normalization of transparent rule scores.",
        "missing": "empty string when rule-softmax weights are unavailable",
    },
    "betlas_mixed_topology_types": {
        "dtype": "string",
        "range": "semicolon-delimited topology labels",
        "definition": "Topology labels participating in a mixed-topology diagnostic.",
        "missing": "empty string when no mixed topology is reported",
    },
    "betlas_manual_boundary_audit_priority": {
        "dtype": "string",
        "range": "categorical/string",
        "definition": "Manual audit-priority label for boundary or mixed-topology review.",
        "missing": "empty string when no audit priority is assigned",
    },
    "betlas_axis_slice_name": {
        "dtype": "string",
        "range": "axis hypothesis name",
        "definition": "Name of the explicitly selected non-best Betlas axis used for slice summaries.",
        "missing": "empty string when no explicit-axis slice summary is emitted",
    },
}
_TOPOLOGY_DIAGNOSTIC_ZERO_FILL_COLUMNS = {
    "betlas_topology_ambiguity_score",
    "betlas_boundary_region_flag",
    "betlas_probability_top1",
    "betlas_probability_top2",
    "betlas_probability_top2_margin",
    "betlas_probability_entropy",
    "betlas_rule_probability_margin",
    "betlas_rule_probability_entropy",
    "betlas_rule_label_conflict",
    "betlas_neighbor_label_entropy",
    "betlas_neighbor_disagreement_fraction",
    "betlas_boundary_neighbor_fraction",
    "betlas_boundary_neighbor_similarity",
    "betlas_jelly_rollness",
    "betlas_sandwichness",
    "betlas_barrel_likeness",
    "betlas_jelly_sandwich_overlap",
    "betlas_barrel_sandwich_overlap",
    "betlas_barrel_jelly_overlap",
    "betlas_mixed_topology_score",
    "betlas_mixed_topology_flag",
    "betlas_secondary_topology_score",
}
for _column in _TOPOLOGY_DIAGNOSTIC_ZERO_FILL_COLUMNS:
    _COLUMN_OVERRIDES.setdefault(_column, {})["missing"] = (
        "missing, blank, or non-finite numeric inputs are zero-filled before "
        "topology-diagnostics normalization"
    )
    _COLUMN_OVERRIDES.setdefault(_column, {})["range"] = "[0, 1]"
_COLUMN_OVERRIDES["betlas_secondary_topology_score"]["range"] = "[0, 1]"
_PLACEHOLDER_COLUMNS = {
    "betlas_angular_fft_k",
    "betlas_axis_",
    "betlas_axis_best_",
    "betlas_rule_score_",
}
_PROTOCOL_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "record_id",
        "feature_csv",
        "identifier",
        "string",
        True,
        "Stable row identifier used to join feature, score, benchmark, and readout tables.",
        "not applicable",
        "non-empty string",
        "not allowed for rows that should be benchmarked or joined",
        "Betlas dataset and single-structure feature extraction",
    ),
    ColumnSpec(
        "pdb_id",
        "feature_csv",
        "identifier",
        "string",
        False,
        "Lowercase PDB/mmCIF accession or user-provided structure identifier.",
        "not applicable",
        "string",
        "empty string when unknown",
        "Betlas dataset and single-structure feature extraction",
    ),
    ColumnSpec(
        "chain_id",
        "feature_csv",
        "identifier",
        "string",
        False,
        "Author chain identifier used during feature extraction.",
        "not applicable",
        "string",
        "empty string when not provided",
        "Betlas dataset and single-structure feature extraction",
    ),
    ColumnSpec(
        "domain_id",
        "feature_csv",
        "identifier",
        "string",
        False,
        "Domain or chain-level identifier used by benchmark and readout joins.",
        "not applicable",
        "string",
        "empty string when not provided",
        "Betlas dataset and single-structure feature extraction",
    ),
    ColumnSpec(
        "residue_ranges",
        "feature_csv",
        "input_scope",
        "string",
        False,
        "Optional residue range expression used to select a domain or chain subset.",
        "residue author numbering",
        "range expression string",
        "empty string means all residues in the selected chain",
        "Betlas feature extraction",
    ),
    ColumnSpec(
        "fold_label_final",
        "feature_csv",
        "label",
        "string",
        True,
        "Fold label consumed by benchmark and ablation workflows.",
        "not applicable",
        "one configured Betlas fold label",
        "not allowed for benchmark rows",
        "Betlas label table or user-provided label table",
    ),
    ColumnSpec(
        "source_mmcif_path",
        "feature_csv",
        "provenance",
        "string",
        False,
        "Structure path used for feature extraction.",
        "not applicable",
        "path string",
        "empty string when unavailable",
        "Betlas feature extraction",
    ),
    ColumnSpec(
        "source_mmcif_sha256",
        "feature_csv",
        "provenance",
        "string",
        False,
        "SHA-256 digest of the structure file used for feature extraction.",
        "not applicable",
        "64-character hexadecimal digest",
        "empty string when unavailable",
        "Betlas feature extraction",
    ),
    ColumnSpec(
        "source_mmcif_size",
        "feature_csv",
        "provenance",
        "integer",
        False,
        "Byte size of the structure file used for feature extraction.",
        "bytes",
        "non-negative integer",
        "empty string or zero when unavailable",
        "Betlas feature extraction",
    ),
    ColumnSpec(
        "source_mmcif_exists",
        "feature_csv",
        "provenance",
        "integer",
        False,
        "Whether the structure file existed when feature extraction wrote the table.",
        "not applicable",
        "{0, 1}",
        "0 when unavailable",
        "Betlas feature extraction",
    ),
)


def _domain_candidate_column_specs() -> tuple[ColumnSpec, ...]:
    already = {spec.name for spec in _PROTOCOL_COLUMNS}
    specs: list[ColumnSpec] = []
    for field_name, field_info in DomainCandidate.__dataclass_fields__.items():
        if field_name in already:
            continue
        dtype = "boolean" if field_info.type is bool else "integer" if field_info.type is int else "string"
        role = "provenance"
        if field_name.startswith("cath_"):
            role = "external_label_provenance"
        elif field_name in {"evidence_level", "label_source_primary", "label_source_supporting", "label_conflict_notes"}:
            role = "label_provenance"
        elif field_name in {"allowed_for_benchmark", "qc_status", "discovered_by_betlas"}:
            role = "qc_status"
        specs.append(
            ColumnSpec(
                field_name,
                "feature_csv",
                role,
                dtype,
                False,
                f"Betlas feature-row metadata field `{field_name}` copied from the input label/source candidate.",
                "not applicable",
                "{true, false}" if dtype == "boolean" else "string" if dtype == "string" else "integer",
                "default dataclass value when unavailable",
                "betlas.models.DomainCandidate",
            )
        )
    return tuple(specs)


def _column_spec_from_feature(feature: FeatureSpec) -> ColumnSpec:
    return ColumnSpec(
        name=feature.name,
        table="feature_csv",
        role="feature",
        dtype=feature.dtype,
        required=False,
        definition=feature.definition,
        units=feature.units,
        value_range=feature.value_range,
        missing_value=feature.missing_value,
        source=feature.source,
    )


def _source_columns() -> set[str]:
    from .features import extract, rules
    from .readouts.topology_diagnostics import core as topology_core

    text = "\n".join(
        [
            inspect.getsource(extract),
            inspect.getsource(rules),
            inspect.getsource(topology_core),
        ]
    )
    columns = set(re.findall(r"betlas_[A-Za-z0-9_]+", text))
    for prefix in _SUMMARY_PREFIXES:
        columns.update(f"{prefix}_{suffix}" for suffix in _SUMMARY_SUFFIXES)
    columns.update(f"betlas_axis_best_{key}" for key in _AXIS_SLICE_KEYS)
    columns.update(f"betlas_axis_{axis}_{key}" for axis in _SLICE_AXIS_NAMES for key in _AXIS_SLICE_KEYS)
    columns.update({"betlas_axis_slice_name", "betlas_axis_slice_score"})
    columns.update(f"betlas_angular_fft_k{k}" for k in range(3, 9))
    columns.update(f"betlas_rule_score_{label}" for label in FOLD_LABELS)
    for names in topology_core.MODE_COLUMNS.values():
        columns.update(str(name) for name in names if str(name).startswith("betlas_"))
    columns.update(_SCHEMA_COLUMNS)
    return columns


def _family_for_column(column: str) -> str:
    if column in _SCHEMA_COLUMNS:
        return "schema"
    from .grammars import list_grammars

    for grammar in list_grammars():
        if grammar.matches_column(column):
            return grammar.name
    return "unassigned"


def _dtype_for_column(column: str) -> str:
    override = _COLUMN_OVERRIDES.get(column, {})
    if "dtype" in override:
        return override["dtype"]
    if column.endswith("_json"):
        return "json-string"
    if column.endswith("_label") or column in {
        "betlas_top_fold",
        "betlas_axis_best_name",
        "betlas_axis_slice_name",
        "betlas_error",
        "betlas_warnings",
    }:
        return "string"
    return "numeric"


def _range_for_column(column: str) -> str:
    override = _COLUMN_OVERRIDES.get(column, {})
    if "range" in override:
        return override["range"]
    if column.endswith("_fraction") or column.endswith("_coverage") or column.endswith("_probability"):
        return "[0, 1]"
    if column.endswith("_flag") or column.endswith("_ok") or "_is_" in column:
        return "{0, 1}"
    if "_count" in column or column.endswith("_rank"):
        return "non-negative"
    if column.endswith("_label") or column.endswith("_json") or column in {
        "betlas_top_fold",
        "betlas_axis_best_name",
        "betlas_axis_slice_name",
        "betlas_error",
        "betlas_warnings",
    }:
        return "categorical/string"
    return "real-valued"


def _definition_for_column(column: str, family: str) -> str:
    override = _COLUMN_OVERRIDES.get(column, {})
    if "definition" in override:
        return override["definition"]
    if family == "schema":
        return "Betlas parse/status or aggregate schema column."
    if column.startswith("betlas_rule_score_"):
        return "Transparent fold-rule score computed from Betlas geometry features."
    if column.startswith("betlas_axis_best_") or column == "betlas_z_continuity_fraction":
        return "Best-axis closure or z-slice descriptor computed from projected beta-structure coordinates."
    return f"Betlas {family} feature emitted by the deterministic geometry extraction pipeline."


def _formula_for_column(column: str, family: str) -> str:
    if column == "betlas_axis_best_score":
        return "0.40 * slice_coverage_median + 0.25 * angular_coverage + 0.20 * z_continuity_fraction + 0.15 * (1 - largest_gap_fraction)"
    if column.startswith("betlas_rule_score_"):
        return "See RuleSpec for the transparent fold-rule score components."
    if column in {
        "betlas_jelly_sandwich_overlap",
        "betlas_barrel_sandwich_overlap",
        "betlas_barrel_jelly_overlap",
    }:
        return "clip01(min(score_a, score_b) * (1 - abs(score_a - score_b))) for the named topology-score pair"
    if column in {"betlas_jelly_rollness", "betlas_sandwichness", "betlas_barrel_likeness"}:
        return (
            "clip01(weighted_mean(clip01(component evidence))) using topology-diagnostics "
            "YAML weights/ranges; missing or non-finite component inputs are zero-filled"
        )
    if column == "betlas_topology_ambiguity_score":
        return (
            "clip01(weighted_mean(probability margin/entropy ambiguity, rule conflict, "
            "boundary-neighbor evidence, and configured label-pair checks)); missing or "
            "non-finite inputs are zero-filled"
        )
    if column == "betlas_mixed_topology_score":
        return (
            "clip01(weighted evidence for simultaneously high continuous topology scores, "
            "overlap terms, secondary topology support, and ambiguity context); missing or "
            "non-finite inputs are zero-filled"
        )
    if column.startswith("betlas_rule_probability_"):
        return (
            "Softmax summary of transparent rule scores with configured temperature; "
            "missing or non-finite rule scores are zero-filled before softmax"
        )
    if column.endswith("_basis_json"):
        return "JSON object recording weighted evidence terms emitted by the matching readout implementation."
    if column.startswith("betlas_axis_") and any(column.startswith(f"betlas_axis_{axis}_") for axis in _SLICE_AXIS_NAMES):
        return "Same z-slice summary formula as the best-axis columns, evaluated on the named axis hypothesis."
    return "Deterministic formula implemented in betlas.features.extract, betlas.slicing, or the matching readout module."


def list_feature_specs(columns: list[str] | tuple[str, ...] | None = None) -> tuple[FeatureSpec, ...]:
    """Return public data-dictionary entries for Betlas output columns."""

    selected = set(columns) if columns is not None else _source_columns()
    specs: list[FeatureSpec] = []
    for column in sorted(str(item) for item in selected if str(item).startswith("betlas_")):
        if column.endswith("_") or column in _PLACEHOLDER_COLUMNS:
            continue
        family = _family_for_column(column)
        override = _COLUMN_OVERRIDES.get(column, {})
        specs.append(
            FeatureSpec(
                name=column,
                family=family,
                dtype=_dtype_for_column(column),
                units="dimensionless unless the definition names Angstrom distances",
                value_range=_range_for_column(column),
                definition=_definition_for_column(column, family),
                formula=_formula_for_column(column, family),
                missing_value=override.get(
                    "missing",
                    "0 or empty string for unavailable geometry/status fields unless documented otherwise",
                ),
                source="betlas.features.extract / betlas.slicing / betlas.readouts",
            )
        )
    return tuple(specs)


def describe_feature(name: str) -> FeatureSpec:
    """Return one feature data-dictionary entry by column name."""

    for spec in list_feature_specs([name]):
        if spec.name == name:
            return spec
    raise KeyError(f"unknown Betlas feature column {name!r}")


def list_rule_specs() -> tuple[RuleSpec, ...]:
    """Return public data-dictionary entries for transparent fold-rule scores."""

    formulas = {
        "beta_barrel": "1.1*closure + 1.0*all_coverage + 0.7*barrel_wall + 0.4*(1-gap) + 0.2*sheet_span - 0.35*lobe_guard - 0.20*k2_dominance",
        "beta_prism": "1.2*bell(face_count,3,1) + 0.7*bell(best_k,3,1) + 0.4*flatness + 0.3*all_coverage",
        "beta_propeller": "1.1*prop_fft + 0.9*bell(best_k,6,2.5) + 0.5*all_coverage + 0.3*flatness",
        "jelly_roll": "0.8*bell(strand_count,8,2.2) + 0.9*bell(run_count,8,2.4) + 0.6*bell(sheet_count,2,1.2) + 0.5*top2_fraction + 0.6*greek_proxy + 0.7*nonlocal_order + 0.5*interleave + 0.4*order_displacement + 0.3*(1-closure)",
        "beta_solenoid": "1.3*axis_periodicity_score + 1.0*min(1.5, elongation/3) + 0.5*z_continuity_fraction",
        "beta_sandwich": "0.8*bell(sheet_count,2,1.2) + 1.0*bilayer + 0.7*lobe_guard + 0.4*top2_balance + 0.5*(1-closure) + 0.4*flatness + 0.2*min(1, face_count/3)",
        "tim_like_beta_alpha_barrel": "1.0*bell(strand_count,8,2.5) + 0.9*min(1, helix_count/8) + 0.8*beta_alpha_alternation + 0.5*max(0, alpha_shell/6) + 0.4*closure",
    }
    inputs = {
        "beta_barrel": (
            "betlas_axis_best_slice_coverage_median",
            "betlas_axis_best_angular_coverage",
            "betlas_axis_best_largest_gap_fraction",
            "betlas_barrel_wall_continuity_score",
            "betlas_sheet_angular_span_max",
            "betlas_sandwich_lobe_guard_score",
            "betlas_angular_fft_k2_dominance",
        ),
        "beta_prism": (
            "betlas_sheet_face_count",
            "betlas_angular_fft_k3_8_best_k",
            "betlas_pca_flatness",
            "betlas_axis_best_angular_coverage",
        ),
        "beta_propeller": (
            "betlas_angular_fft_k3_8_max",
            "betlas_angular_fft_k3_8_best_k",
            "betlas_axis_best_angular_coverage",
            "betlas_pca_flatness",
        ),
        "jelly_roll": (
            "betlas_beta_strand_count",
            "betlas_beta_run_count",
            "betlas_sheet_count",
            "betlas_sheet_pair_top2_fraction",
            "betlas_sheet_seq_greek_key_proxy",
            "betlas_jelly_roll_order_nonlocal_score",
            "betlas_sheet_seq_top2_interleave_score",
            "betlas_sheet_seq_top2_order_displacement",
            "betlas_axis_best_slice_coverage_median",
        ),
        "beta_solenoid": (
            "betlas_axis_periodicity_score",
            "betlas_pca_elongation",
            "betlas_z_continuity_fraction",
        ),
        "beta_sandwich": (
            "betlas_sheet_count",
            "betlas_sheet_pair_bilayer_score",
            "betlas_sandwich_lobe_guard_score",
            "betlas_sheet_pair_size_balance",
            "betlas_axis_best_slice_coverage_median",
            "betlas_pca_flatness",
            "betlas_sheet_face_count",
        ),
        "tim_like_beta_alpha_barrel": (
            "betlas_beta_strand_count",
            "betlas_helix_count",
            "betlas_beta_alpha_alternation_fraction",
            "betlas_alpha_shell_radial_delta",
            "betlas_axis_best_slice_coverage_median",
        ),
    }
    return tuple(
        RuleSpec(
            name=f"betlas_rule_score_{label}",
            fold_label=label,
            definition=f"Transparent Betlas rule score for the {label} fold label.",
            formula=formulas[label],
            inputs=inputs[label],
            source="betlas.features.rules.grammar_rule_scores",
        )
        for label in FOLD_LABELS
    )


def _readout_column_definition(readout: str, column: str) -> tuple[str, str, str, str]:
    if column in {"filename", "source_path"}:
        return ("provenance", "string", "Input structure filename or path.", "path/string")
    if column == "chain":
        return ("identifier", "string", "Chain identifier reported by the structure parser.", "string")
    if column == "result":
        return ("result", "string", "Readout result enum for the analyzed chain.", "categorical/string")
    if column == "result_stage":
        return ("status", "string", "Pipeline stage that produced the row status.", "categorical/string")
    if column == "score_type":
        return ("score", "string", "Declares whether the readout score is heuristic or calibrated.", "categorical/string")
    if column == "calibration_status":
        return ("score", "string", "Calibration status for the readout score; native readout scores are uncalibrated.", "categorical/string")
    if column == "config_profile":
        return ("provenance", "string", "Named configuration profile used for the readout run.", "string")
    if column in {"reason", "decision_basis", "confidence_basis", "barrel_gate_reason"}:
        return ("status", "string", "Human-readable basis or failure/status reason.", "string")
    if column == "decision_score":
        return (
            "score",
            "numeric",
            "Heuristic positive-decision support score used by the native detection readout; non-BARREL rows use 0 and raw geometry remains in score_raw/score_adjust. It is not a calibrated probability.",
            "[0, +inf)",
        )
    if column == "confidence":
        return (
            "score",
            "numeric",
            "Heuristic candidate-stave confidence from slice-derived support and agreement terms; not a calibrated probability.",
            "[0, 1]",
        )
    if column == "strand_count":
        return (
            "readout",
            "integer",
            "Candidate beta-barrel stave or strand count reported by the staves readout.",
            "non-negative integer",
        )
    if column.startswith("betlas_"):
        feature = describe_feature(column)
        return ("readout", feature.dtype, feature.definition, feature.value_range)
    if column.endswith("_flag") or column.startswith("guard_") or column.startswith("barrel_gate_"):
        return ("status", "numeric/string", "Readout guard, gate, or status field.", "categorical/string")
    if "count" in column or column.endswith("_layers") or column.endswith("_residues"):
        return ("readout", "numeric", "Count or layer-support field emitted by the readout pipeline.", "non-negative")
    if "score" in column or "fraction" in column:
        return ("score", "numeric", "Heuristic geometry or support score emitted by the readout pipeline.", "real-valued")
    return ("readout", "string", f"{readout} output column.", "string or numeric")


def list_readout_column_specs(readout: str | None = None) -> tuple[ColumnSpec, ...]:
    """Return public column specs for Betlas readout output tables."""

    from .readouts.beta_barrel_detection.constants import (
        DEFAULT_RESULT_COLUMNS as DETECTION_COLUMNS,
    )
    from .readouts.beta_barrel_staves.constants import DEFAULT_RESULT_COLUMNS as STAVES_COLUMNS
    from .readouts.topology_diagnostics.core import ID_COLUMNS, MODE_COLUMNS

    readout_columns = {
        "beta-barrel-detection": tuple(DETECTION_COLUMNS),
        "beta_barrel_detection": tuple(DETECTION_COLUMNS),
        "beta-barrel-staves": tuple(STAVES_COLUMNS),
        "beta_barrel_staves": tuple(STAVES_COLUMNS),
        "topology-diagnostics": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["all"]),
        "topology_diagnostics": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["all"]),
        "topology-ambiguity": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["ambiguity"]),
        "topology_ambiguity": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["ambiguity"]),
        "fold-continuous-scores": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["continuous"]),
        "fold_continuous_scores": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["continuous"]),
        "mixed-topology": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["mixed"]),
        "mixed_topology": tuple(ID_COLUMNS) + tuple(MODE_COLUMNS["mixed"]),
    }
    if readout is not None:
        try:
            selected = {readout: readout_columns[readout]}
        except KeyError as exc:
            available = ", ".join(sorted(readout_columns))
            raise KeyError(f"unknown Betlas readout {readout!r}; available readouts: {available}") from exc
    else:
        selected = {
            "beta-barrel-detection": readout_columns["beta-barrel-detection"],
            "beta-barrel-staves": readout_columns["beta-barrel-staves"],
            "topology-diagnostics": readout_columns["topology-diagnostics"],
            "topology-ambiguity": readout_columns["topology-ambiguity"],
            "fold-continuous-scores": readout_columns["fold-continuous-scores"],
            "mixed-topology": readout_columns["mixed-topology"],
        }

    specs: list[ColumnSpec] = []
    for readout_name, columns in selected.items():
        for column in columns:
            role, dtype, definition, value_range = _readout_column_definition(readout_name, str(column))
            specs.append(
                ColumnSpec(
                    name=str(column),
                    table=f"{readout_name}_csv",
                    role=role,
                    dtype=dtype,
                    required=True,
                    definition=definition,
                    units="dimensionless unless the definition names Angstrom distances, residues, layers, or bytes",
                    value_range=value_range,
                    missing_value="empty string for unavailable status/provenance fields; numeric fields use 0 or NaN only as documented by the readout",
                    source=f"betlas.readouts.{readout_name.replace('-', '_')}",
                )
            )
    return tuple(specs)


def describe_readout_column(name: str, readout: str | None = None) -> ColumnSpec:
    """Return one readout column data-dictionary entry."""

    for spec in list_readout_column_specs(readout):
        if spec.name == name:
            return spec
    raise KeyError(f"unknown Betlas readout column {name!r}")


def list_column_specs(table: str | None = None) -> tuple[ColumnSpec, ...]:
    """Return public table-contract entries for feature and readout outputs."""

    specs = [
        *_PROTOCOL_COLUMNS,
        *_domain_candidate_column_specs(),
        *(_column_spec_from_feature(spec) for spec in list_feature_specs()),
    ]
    specs.extend(list_readout_column_specs())
    if table is not None:
        specs = [spec for spec in specs if spec.table == table]
    return tuple(specs)


def describe_column(name: str, table: str | None = None) -> ColumnSpec:
    """Return one public table-contract entry."""

    for spec in list_column_specs(table):
        if spec.name == name:
            return spec
    raise KeyError(f"unknown Betlas column {name!r}")


__all__ = [
    "FeatureSpec",
    "RuleSpec",
    "ColumnSpec",
    "describe_column",
    "describe_feature",
    "describe_readout_column",
    "list_feature_specs",
    "list_column_specs",
    "list_readout_column_specs",
    "list_rule_specs",
]
