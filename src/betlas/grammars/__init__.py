from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..constants import FOLD_LABELS
from ..features.extract import extract_signature
from ..features.rules import grammar_rule_scores
from ..models import StructureGeometry
from ..schema import _is_compatibility_column, normalize_feature_mapping


@dataclass(frozen=True)
class GrammarSpec:
    name: str
    summary: str
    column_prefixes: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    math_summary: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    implementation_notes: tuple[str, ...] = ()

    def matches_column(self, column: str) -> bool:
        return column in self.columns or any(column.startswith(prefix) for prefix in self.column_prefixes)

    def to_dict(self) -> dict[str, Any]:
        from ..specs import list_feature_specs

        resolved_columns = [
            feature.name for feature in list_feature_specs() if self.matches_column(feature.name)
        ]
        return {
            "name": self.name,
            "summary": self.summary,
            "math_summary": self.math_summary,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "column_prefixes": list(self.column_prefixes),
            "columns": list(self.columns),
            "resolved_columns": resolved_columns,
            "implementation_notes": list(self.implementation_notes),
        }


_GRAMMAR_SPECS: tuple[GrammarSpec, ...] = (
    GrammarSpec(
        "composition",
        "Residue, beta-strand, helix, SSE-count, and global composition features.",
        (
            "betlas_residue_",
            "betlas_beta_residue",
            "betlas_beta_strand",
            "betlas_strand_length",
            "betlas_helix_",
            "betlas_sse_count",
            "betlas_pca_",
        ),
    ),
    GrammarSpec(
        "axis_closure",
        "Best-axis angular closure, slice continuity, and barrel-wall evidence.",
        (
            "betlas_axis_best_",
            "betlas_axis_strand_axis_",
            "betlas_axis_point_pc1_",
            "betlas_axis_point_pc2_",
            "betlas_axis_point_pc3_",
            "betlas_z_continuity",
            "betlas_angular_gap",
            "betlas_barrel_wall_",
        ),
        ("betlas_axis_slice_name", "betlas_axis_slice_score"),
    ),
    GrammarSpec(
        "sheet_geometry",
        "Sheet counts, sizes, angular spans, planarity, and sheet-level geometry.",
        (
            "betlas_sheet_count",
            "betlas_sheet_face_count",
            "betlas_largest_sheet",
            "betlas_sheet_strand_count",
            "betlas_sheet_size_entropy",
            "betlas_sheet_angular_span",
            "betlas_sheet_planarity",
            "betlas_sheet_axis",
            "betlas_parallel_sheet",
            "betlas_antiparallel_sheet",
        ),
    ),
    GrammarSpec(
        "beta_run_topology",
        "Merged sequence-level beta-run topology and run angular coverage.",
        ("betlas_beta_run_", "betlas_beta_segment_to_run_"),
    ),
    GrammarSpec(
        "sheet_pair_packing",
        "Top-two sheet packing, face alignment, cross contacts, and bilayer evidence.",
        ("betlas_sheet_pair_",),
    ),
    GrammarSpec(
        "sheet_sequence_topology",
        "Top-two sheet sequence alternation, interleave, displacement, and Greek-key proxy.",
        ("betlas_sheet_seq_",),
    ),
    GrammarSpec(
        "sheet_order_topology",
        "Sheet range order, nonlocality, inversion, and jelly-roll ordering evidence.",
        ("betlas_sheet_order_", "betlas_top2_sheet_order_", "betlas_jelly_roll_order_"),
    ),
    GrammarSpec(
        "contact_graph",
        "Beta-strand contact graph density, components, cycle rank, and degree structure.",
        (
            "betlas_contact8_edge_",
            "betlas_contact10_edge_",
            "betlas_contact8_cross_sheet_fraction",
            "betlas_contact8_component_",
            "betlas_contact8_largest_component_",
            "betlas_contact8_cycle_",
            "betlas_contact8_degree",
        ),
    ),
    GrammarSpec(
        "contact_sequence_topology",
        "Sequence separation and nonlocality of contact edges.",
        (
            "betlas_contact8_seq_",
            "betlas_contact8_nonlocal",
            "betlas_contact8_cross_sheet_nonlocal",
            "betlas_contact10_seq_",
            "betlas_contact10_nonlocal",
        ),
    ),
    GrammarSpec(
        "axis_periodicity",
        "Periodic angular harmonics around the best structural axis.",
        (),
        (
            "betlas_axis_periodicity_score",
            "betlas_angular_fft_k3_8_max",
            "betlas_angular_fft_k3_8_best_k",
            "betlas_angular_fft_k3",
            "betlas_angular_fft_k4",
            "betlas_angular_fft_k5",
            "betlas_angular_fft_k6",
            "betlas_angular_fft_k7",
            "betlas_angular_fft_k8",
        ),
    ),
    GrammarSpec(
        "strand_order",
        "Strand order, direction, N-to-C axis relation, and strand-axis descriptors.",
        (
            "betlas_strand_order_",
            "betlas_strand_direction_",
            "betlas_strand_ntc_",
            "betlas_strand_axis_",
        ),
    ),
    GrammarSpec(
        "alpha_shell",
        "Helix-beta alternation and radial alpha-shell context around beta structures.",
        ("betlas_alpha_shell_", "betlas_helix_beta_", "betlas_beta_alpha_"),
    ),
    GrammarSpec(
        "angular_lobes",
        "Low-order angular lobes, sector occupancy, and sandwich-lobe guard signals.",
        (
            "betlas_angular_fft_k1",
            "betlas_angular_fft_k2",
            "betlas_angular_sector_",
            "betlas_sandwich_lobe_guard",
        ),
    ),
    GrammarSpec(
        "fold_rule_scores",
        "Transparent fold grammar rule scores and top-rule margin.",
        ("betlas_rule_score_",),
        ("betlas_top_fold", "betlas_rule_margin", "betlas_fold_scores_json"),
    ),
    GrammarSpec(
        "continuous_topology",
        "Continuous jelly-roll, sandwich, and barrel-likeness topology readouts.",
        (),
        (
            "betlas_jelly_rollness",
            "betlas_sandwichness",
            "betlas_barrel_likeness",
            "betlas_jelly_rollness_basis_json",
            "betlas_sandwichness_basis_json",
            "betlas_barrel_likeness_basis_json",
            "betlas_jelly_sandwich_overlap",
            "betlas_barrel_sandwich_overlap",
            "betlas_barrel_jelly_overlap",
        ),
    ),
    GrammarSpec(
        "topology_ambiguity",
        "Topology ambiguity, model probability, boundary-neighbor, and conflict readouts.",
        (
            "betlas_topology_ambiguity_",
            "betlas_probability_",
            "betlas_rule_top",
            "betlas_rule_probability_",
            "betlas_neighbor_",
            "betlas_boundary_neighbor_",
        ),
        ("betlas_boundary_region_flag", "betlas_rule_label_conflict"),
    ),
    GrammarSpec(
        "mixed_topology",
        "Mixed-topology flag, secondary topology, and manual boundary-audit priority.",
        ("betlas_mixed_topology_", "betlas_secondary_topology_", "betlas_manual_boundary_audit_"),
    ),
)

_GRAMMAR_DETAILS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "composition": {
        "math_summary": "Counts and normalized fractions are computed over parsed residues and secondary-structure elements; PCA descriptors summarize the coordinate cloud by principal-axis elongation and flatness.",
        "inputs": (
            "Parsed residues, beta segments, helices, and Cartesian coordinates from StructureGeometry.",
            "Per-domain secondary-structure assignments and residue-level geometry.",
        ),
        "outputs": (
            "Residue, beta, helix, strand-length, SSE-count, and PCA composition columns.",
        ),
        "implementation_notes": (
            "Implemented in betlas.features.extract without learned parameters.",
        ),
    },
    "axis_closure": {
        "math_summary": "Candidate structural axes are scored as 0.40*slice_coverage_median + 0.25*angular coverage + 0.20*z_continuity_fraction + 0.15*(1-largest_gap_fraction), where angular coverage is the occupied angular fraction around the axis. Barrel-wall continuity is computed after best-axis slice summaries and is not part of axis selection.",
        "inputs": (
            "Beta-strand C-alpha coordinates and candidate axes from strand and point-cloud geometry.",
            "Slice occupancy along the selected axis.",
        ),
        "outputs": (
            "Best-axis closure, angular gap, z-continuity, and barrel-wall continuity columns.",
        ),
        "implementation_notes": (
            "Axis selection and slice summaries are deterministic functions of StructureGeometry.",
        ),
    },
    "sheet_geometry": {
        "math_summary": "Beta sheets are summarized by count, face count, strand membership, size entropy, angular span around the best axis, and planarity. Parallel/antiparallel fractions are derived from parsed mmCIF sheet sense_to_previous labels when present.",
        "inputs": (
            "Sheet patches, strand assignments, contact-derived sheet membership, and best-axis coordinates.",
        ),
        "outputs": (
            "Sheet count, face count, largest-sheet, sheet-size entropy, angular span, planarity, and orientation columns.",
        ),
        "implementation_notes": (
            "Sheet descriptors reuse the same parsed sheet patches used by downstream packing grammars.",
        ),
    },
    "beta_run_topology": {
        "math_summary": "Sequence-adjacent beta segments are merged into beta runs; run count, run length, run-to-segment mapping, and angular run coverage describe local beta topology.",
        "inputs": (
            "Ordered beta segments and residue indices.",
            "Best-axis angular coordinates for segment and run centroids.",
        ),
        "outputs": (
            "Beta-run count, length, coverage, and segment-to-run mapping columns.",
        ),
        "implementation_notes": (
            "The merge rule follows existing sequence adjacency thresholds in feature extraction.",
        ),
    },
    "sheet_pair_packing": {
        "math_summary": "The two largest sheets are compared by size balance, top-two fraction, face alignment, cross-sheet contact density, and a bilayer packing score.",
        "inputs": (
            "Sheet-patch membership, beta-strand centroids, and residue contact distances.",
        ),
        "outputs": (
            "Top-two sheet packing, cross-contact, alignment, balance, and bilayer score columns.",
        ),
        "implementation_notes": (
            "The bilayer score is a transparent geometry score, not a fitted model.",
        ),
    },
    "sheet_sequence_topology": {
        "math_summary": "Top-two sheet ranges are compared in sequence order to measure alternation, interleave, displacement, and a Greek-key-like topology proxy.",
        "inputs": (
            "Sequence positions for strands assigned to the two largest sheets.",
        ),
        "outputs": (
            "Sheet-sequence alternation, interleave, displacement, and Greek-key proxy columns.",
        ),
        "implementation_notes": (
            "The proxy summarizes ordering evidence and does not assign a discrete fold label by itself.",
        ),
    },
    "sheet_order_topology": {
        "math_summary": "Sheet strand ranges are compared across sequence and sheet order to quantify nonlocality, order inversion, and jelly-roll ordering evidence.",
        "inputs": (
            "Sheet range order, top-two sheet assignments, and beta-strand sequence positions.",
        ),
        "outputs": (
            "Sheet-order nonlocality, inversion, and jelly-roll order evidence columns.",
        ),
        "implementation_notes": (
            "This grammar provides inputs to continuous topology and fold-rule scores.",
        ),
    },
    "contact_graph": {
        "math_summary": "A beta-strand contact graph is built from residue-distance cutoffs; density, component count, largest component fraction, cycle rank, and degree statistics summarize graph topology.",
        "inputs": (
            "Beta-strand residue coordinates and distance-threshold contact edges.",
        ),
        "outputs": (
            "Contact-edge density, cross-sheet fraction, component, cycle, and degree columns.",
        ),
        "implementation_notes": (
            "Cutoff-specific columns use the existing 8 A and 10 A contact definitions.",
        ),
    },
    "contact_sequence_topology": {
        "math_summary": "Contact edges are projected onto sequence positions to compute edge separation, nonlocal contact fractions, and cross-sheet nonlocality.",
        "inputs": (
            "Contact graph edges and beta-strand sequence indices.",
        ),
        "outputs": (
            "Contact sequence-separation and nonlocality columns.",
        ),
        "implementation_notes": (
            "The readout is derived from the same contact graph used by contact_graph.",
        ),
    },
    "axis_periodicity": {
        "math_summary": "Angular occupancy around the selected axis is decomposed into Fourier harmonics k=3..8; a helical angle-versus-z linear-fit R2 is averaged with the strongest harmonic to produce betlas_axis_periodicity_score.",
        "inputs": (
            "Best-axis angular coordinates for beta-strand or run centroids.",
        ),
        "outputs": (
            "Axis periodicity score, k=3..8 harmonic amplitudes, best-k, and maximum harmonic columns.",
        ),
        "implementation_notes": (
            "The harmonic calculation is deterministic and uses the existing angular-bin representation.",
        ),
    },
    "strand_order": {
        "math_summary": "Strands are ordered by sequence and by axis projection; direction, N-to-C relation, and strand-axis descriptors encode relative orientation.",
        "inputs": (
            "Beta-strand endpoints, sequence order, and selected structural axis.",
        ),
        "outputs": (
            "Strand order, direction, N-to-C, and strand-axis descriptor columns.",
        ),
        "implementation_notes": (
            "Orientation values are geometry descriptors, not post-hoc label corrections.",
        ),
    },
    "alpha_shell": {
        "math_summary": "Helix-beta alternation and radial helix placement are summarized by beta-alpha sequence alternation and alpha-shell radial context around beta structure.",
        "inputs": (
            "Helix segments, beta segments, radial distances, and sequence adjacency.",
        ),
        "outputs": (
            "Alpha-shell, helix-beta, and beta-alpha alternation columns.",
        ),
        "implementation_notes": (
            "These columns contribute to the tim-like beta-alpha barrel rule score.",
        ),
    },
    "angular_lobes": {
        "math_summary": "Low-order angular Fourier harmonics and sector occupancy quantify one- and two-lobed beta distributions; the sandwich-lobe guard penalizes closed-barrel interpretations when bilayer lobes dominate.",
        "inputs": (
            "Best-axis angular occupancy of beta elements.",
        ),
        "outputs": (
            "k=1/k=2 angular harmonic, angular-sector occupancy, and sandwich-lobe guard columns.",
        ),
        "implementation_notes": (
            "The guard enters fold-rule scores as an explicit negative term for beta-barrel evidence.",
        ),
    },
    "fold_rule_scores": {
        "math_summary": "Transparent fold scores are weighted sums of grammar features plus bell-shaped terms such as exp(-((x-center)/width)^2); the top fold and margin are derived from the sorted score vector.",
        "inputs": (
            "Canonical Betlas feature columns consumed by betlas.features.rules.grammar_rule_scores.",
        ),
        "outputs": (
            "Per-fold rule scores, top fold, top-rule margin, and JSON score vector columns.",
        ),
        "implementation_notes": (
            "Weights and bell centers are the constants already defined in betlas.features.rules.",
        ),
    },
    "continuous_topology": {
        "math_summary": "Jelly-rollness, sandwichness, and barrel-likeness are weighted means of normalized grammar evidence. Each pairwise overlap is clip01(min(score_a, score_b) * (1 - abs(score_a - score_b))), so overlap is high only when both topology scores are high and similar.",
        "inputs": (
            "Grammar-rule probabilities, sheet topology features, packing features, and closure features.",
        ),
        "outputs": (
            "Continuous topology scores, basis JSON columns, and pairwise overlap columns.",
        ),
        "implementation_notes": (
            "Implemented by topology diagnostics with configurable weights loaded from package YAML.",
        ),
    },
    "topology_ambiguity": {
        "math_summary": "Ambiguity combines probability entropy, top-two probability margin, rule/model disagreement, boundary-neighbor evidence, and YAML-defined boundary label-pair checks.",
        "inputs": (
            "Betlas feature tables, optional prediction tables, grammar-rule probabilities, and nearest-neighbor feature space.",
        ),
        "outputs": (
            "Topology ambiguity score, level, reasons, probability summaries, rule summaries, and boundary-neighbor columns.",
        ),
        "implementation_notes": (
            "When prediction tables are absent, grammar-rule probabilities are used as the probability source.",
        ),
    },
    "mixed_topology": {
        "math_summary": "Mixed topology evidence is computed from simultaneous high continuous topology scores and secondary topology support, then thresholded into a flag and audit-priority columns.",
        "inputs": (
            "Continuous topology scores, overlap columns, probability summaries, and topology diagnostics configuration.",
        ),
        "outputs": (
            "Mixed-topology score, flag, type labels, secondary topology, and audit-priority columns.",
        ),
        "implementation_notes": (
            "This readout reports boundary evidence while leaving source labels unchanged.",
        ),
    },
}

_GRAMMAR_SPECS = tuple(
    GrammarSpec(
        name=spec.name,
        summary=spec.summary,
        column_prefixes=spec.column_prefixes,
        columns=spec.columns,
        math_summary=str(_GRAMMAR_DETAILS[spec.name]["math_summary"]),
        inputs=tuple(_GRAMMAR_DETAILS[spec.name]["inputs"]),
        outputs=tuple(_GRAMMAR_DETAILS[spec.name]["outputs"]),
        implementation_notes=tuple(_GRAMMAR_DETAILS[spec.name]["implementation_notes"]),
    )
    for spec in _GRAMMAR_SPECS
)

_GRAMMAR_BY_NAME = {spec.name: spec for spec in _GRAMMAR_SPECS}


def list_grammars() -> tuple[GrammarSpec, ...]:
    return _GRAMMAR_SPECS


def get_grammar(name: str) -> GrammarSpec:
    try:
        return _GRAMMAR_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown Betlas grammar {name!r}; expected one of {sorted(_GRAMMAR_BY_NAME)}") from exc


def compute_grammar_features(
    geometry: StructureGeometry,
    grammars: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, float | int | str]:
    features = normalize_feature_mapping(extract_signature(geometry).features)
    if grammars is None:
        return dict(features)

    names = (grammars,) if isinstance(grammars, str) else tuple(grammars)
    specs = tuple(get_grammar(name) for name in names)
    return {
        column: value
        for column, value in features.items()
        if any(spec.matches_column(column) for spec in specs)
    }


def score_fold_grammar(features: dict[str, Any], *, strict: bool = True) -> dict[str, float]:
    if strict and any(_is_compatibility_column(str(key)) for key in features):
        raise ValueError(
            "score_fold_grammar requires canonical betlas_* columns by default; "
            "pass strict=False only for compatibility scoring of older feature names"
        )
    return grammar_rule_scores(normalize_feature_mapping(features), strict=strict)


def explain_fold_grammar(
    features: dict[str, Any],
    fold: str | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    scores = score_fold_grammar(features, strict=strict)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top = ranked[0] if ranked else ("", 0.0)
    second = ranked[1] if len(ranked) > 1 else ("", 0.0)
    explanation: dict[str, Any] = {
        "scores": scores,
        "scores_json": json.dumps(scores, sort_keys=True),
        "top_fold": top[0],
        "top_score": float(top[1]),
        "second_fold": second[0],
        "second_score": float(second[1]),
        "margin": float(top[1] - second[1]) if ranked else 0.0,
    }
    if fold is not None:
        if fold not in FOLD_LABELS:
            raise ValueError(f"unknown fold {fold!r}; expected one of {list(FOLD_LABELS)}")
        explanation["requested_fold"] = fold
        explanation["requested_score"] = float(scores.get(fold, 0.0))
        explanation["requested_rank"] = next(
            (index for index, (label, _score) in enumerate(ranked, start=1) if label == fold),
            None,
        )
    return explanation


__all__ = [
    "GrammarSpec",
    "compute_grammar_features",
    "explain_fold_grammar",
    "get_grammar",
    "list_grammars",
    "score_fold_grammar",
]
