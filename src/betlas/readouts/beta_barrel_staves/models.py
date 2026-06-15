from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from typing import Any, get_type_hints

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"", "0", "false", "f", "no", "n", "off", "none", "null", "nan"}


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        return False
    return False


def _field_default(dataclass_field: Field[object]) -> object:
    if dataclass_field.default is not MISSING:
        return dataclass_field.default
    if dataclass_field.default_factory is not MISSING:
        return dataclass_field.default_factory()
    return ""


def _coerce_scalar(value: object, target_type: object, default: object) -> object:
    if target_type is bool:
        return _coerce_bool(value)
    if target_type is int:
        try:
            return int(value if value not in (None, "") else default)
        except (TypeError, ValueError):
            return int(default or 0)
    if target_type is float:
        try:
            return float(value if value not in (None, "") else default)
        except (TypeError, ValueError):
            return float(default or 0.0)
    if target_type is str:
        return str(value if value is not None else default)
    return value if value is not None else default


def _coerced_dataclass_kwargs(
    cls: type[object],
    row: Mapping[str, object],
    *,
    skip: set[str] | None = None,
) -> dict[str, object]:
    skipped = skip or set()
    type_hints = get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for dataclass_field in fields(cls):
        name = dataclass_field.name
        if name in skipped:
            continue
        kwargs[name] = _coerce_scalar(
            row.get(name, _field_default(dataclass_field)),
            type_hints.get(name, object),
            _field_default(dataclass_field),
        )
    return kwargs


@dataclass(frozen=True)
class ResidueRecord:
    """C-alpha residue record used by the public loader API."""

    res_id: int
    coord: Any
    is_sheet: bool
    chain: str | None = None
    insertion_code: str = ""
    hetflag: str = ""
    res_uid: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedChainPayload:
    """Prepared per-chain payload passed into the chain analyzer."""

    filename: str
    chain: str
    residues_data: list[dict[str, object]]
    source_path: str = ""
    chain_index: int | None = None
    barrel_gate: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreparedChainPayload:
        known = {"filename", "chain", "residues_data", "source_path", "_chain_index", "barrel_gate"}
        chain_index: int | None
        try:
            chain_index = int(payload["_chain_index"]) if "_chain_index" in payload else None
        except (TypeError, ValueError):
            chain_index = None
        return cls(
            filename=str(payload.get("filename", "")),
            chain=str(payload.get("chain", "")),
            residues_data=list(payload.get("residues_data", []) or []),
            source_path=str(payload.get("source_path", "")),
            chain_index=chain_index,
            barrel_gate=dict(payload.get("barrel_gate", {}) or {}),
            extra={str(key): value for key, value in payload.items() if key not in known},
        )

    def to_dict(self) -> dict[str, object]:
        data = dict(self.extra)
        data.update(
            {
                "filename": self.filename,
                "chain": self.chain,
                "residues_data": self.residues_data,
            }
        )
        if self.source_path:
            data["source_path"] = self.source_path
        if self.chain_index is not None:
            data["_chain_index"] = self.chain_index
        if self.barrel_gate:
            data["barrel_gate"] = self.barrel_gate
        return data


@dataclass(frozen=True)
class LayerDiagnostic:
    """Per-slice diagnostic returned by the strand-count analyzer."""

    z: float
    raw_points: int
    collapsed_points: int
    selected_points: int
    layer_count: int
    usable: bool
    geometric_pass: bool
    reason: str
    seq_core_trim_left: int = 0
    seq_core_trim_right: int = 0
    radial_outliers: int = 0
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> LayerDiagnostic:
        kwargs = _coerced_dataclass_kwargs(cls, row, skip={"raw"})
        return cls(**kwargs, raw=dict(row))

    def to_dict(self) -> dict[str, object]:
        data = dict(self.raw)
        data.update(
            {
                "z": self.z,
                "raw_points": self.raw_points,
                "collapsed_points": self.collapsed_points,
                "selected_points": self.selected_points,
                "layer_count": self.layer_count,
                "usable": self.usable,
                "geometric_pass": self.geometric_pass,
                "reason": self.reason,
                "seq_core_trim_left": self.seq_core_trim_left,
                "seq_core_trim_right": self.seq_core_trim_right,
                "radial_outliers": self.radial_outliers,
            }
        )
        return data


@dataclass(frozen=True)
class StrandCountReport:
    """Structured report for one analyzed chain before CSV row serialization."""

    strand_count: int
    confidence: float
    confidence_basis: str
    layer_count_mode: int = 0
    layer_count_median: float = 0.0
    supporting_layers: int = 0
    usable_layers: int = 0
    total_layers: int = 0
    layer_support_fraction: float = 0.0
    usable_layer_fraction: float = 0.0
    geometric_pass_layers: int = 0
    geometric_pass_fraction: float = 0.0
    trimmed_layers: int = 0
    radial_outlier_layers: int = 0
    radial_outlier_points: int = 0
    candidate_strands: int = 0
    persistent_strands: int = 0
    trajectory_candidate_strands: int = 0
    trajectory_merge_count: int = 0
    support_consensus_strands: int = 0
    support_consensus_threshold: int = 0
    layer_count_q75: int = 0
    layer_count_q90: int = 0
    layer_count_max: int = 0
    selected_strand_support_mean: float = 0.0
    selected_strand_support_min: float = 0.0
    count_decision_score: float = 0.0
    layer_count_max_plateau: int = 0
    layer_count_largest_drop: int = 0
    absent_from_max_strands: int = 0
    terminal_strand_support: int = 0
    terminal_consensus_strands: int = 0
    terminal_support_fraction: float = 0.0
    barrel_wall_graph_applied: int = 0
    barrel_wall_graph_count: int = 0
    barrel_wall_graph_variant: str = ""
    barrel_wall_graph_edge_min_count: int = 0
    barrel_wall_graph_visibility_regime: str = ""
    barrel_wall_graph_selection_strategy: str = ""
    barrel_wall_graph_support_ratio: float = 0.0
    barrel_wall_graph_node_count: int = 0
    barrel_wall_graph_edge_count: int = 0
    barrel_wall_graph_component_edge_count: int = 0
    barrel_wall_graph_component_avg_degree: float = 0.0
    barrel_wall_graph_component_cycle_rank: float = 0.0
    barrel_wall_graph_component_degree2_fraction: float = 0.0
    barrel_wall_graph_component_branch_fraction: float = 0.0
    barrel_wall_graph_component_radial_rank_mean: float = 0.0
    barrel_wall_graph_component_radial_rank_min: float = 0.0
    barrel_wall_graph_edge_density: float = 0.0
    barrel_wall_graph_membership_score: float = 0.0
    barrel_wall_graph_membership_block_count: int = 0
    barrel_wall_graph_membership_largest_block_fraction: float = 0.0
    barrel_wall_graph_membership_plateau_edges: int = 0
    barrel_wall_graph_membership_next_drop_fraction: float = 0.0
    barrel_wall_graph_membership_variant_agreement: int = 0
    barrel_wall_graph_observed_layers: int = 0
    barrel_wall_graph_selected_ids: str = ""
    barrel_wall_graph_raw_curve: str = ""
    barrel_wall_graph_outer_curve: str = ""
    barrel_wall_graph_outer_gap_curve: str = ""
    run_window_core_applied: int = 0
    run_window_core_score: float = 0.0
    run_window_core_start: int = -1
    run_window_core_stop: int = -1
    run_window_core_trim_left: int = 0
    run_window_core_trim_right: int = 0
    run_window_core_candidate_windows: int = 0
    run_window_core_reanalyzed_windows: int = 0
    run_window_core_base_strand_count: int = 0
    run_window_core_base_support_consensus: int = 0
    run_window_core_base_layer_q90: int = 0
    run_window_core_base_layer_max: int = 0
    run_window_core_layer_max_preservation: float = 0.0
    run_window_core_layer_q90_preservation: float = 0.0
    run_window_core_outside_support_fraction: float = 0.0
    run_window_core_z_coverage_fraction: float = 0.0
    run_window_core_max_z_gap: int = 0
    run_window_core_centrality: float = 0.0
    run_window_core_count_layer_improvement: int = 0
    run_window_core_absent_improvement: float = 0.0
    run_window_core_drop_improvement: float = 0.0
    run_window_core_count_margin: float = 0.0
    layer_details: list[LayerDiagnostic] = field(default_factory=list)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, report: Mapping[str, object]) -> StrandCountReport:
        layers = [
            LayerDiagnostic.from_mapping(layer)
            for layer in list(report.get("layer_details", []) or [])
            if isinstance(layer, Mapping)
        ]
        kwargs = _coerced_dataclass_kwargs(cls, report, skip={"layer_details", "raw"})
        return cls(**kwargs, layer_details=layers, raw=dict(report))

    def to_dict(self) -> dict[str, object]:
        data = dict(self.raw)
        for key, value in asdict(self).items():
            if key == "raw":
                continue
            if key == "layer_details":
                data[key] = [layer.to_dict() for layer in self.layer_details]
            else:
                data[key] = value
        return data


@dataclass(frozen=True)
class StrandCountResult:
    """Stable public result for one structure chain."""

    filename: str
    chain: str
    result: str
    result_stage: str
    reason: str
    strand_count: int = 0
    confidence: float = 0.0
    confidence_basis: str = ""
    count_threshold: float = 0.0
    barrel_gate_enabled: bool = False
    barrel_gate_passed: bool = False
    barrel_gate_result: str = ""
    barrel_gate_score: float = 0.0
    barrel_gate_reason: str = ""
    layer_count_mode: int = 0
    layer_count_median: float = 0.0
    supporting_layers: int = 0
    usable_layers: int = 0
    total_layers: int = 0
    layer_support_fraction: float = 0.0
    usable_layer_fraction: float = 0.0
    geometric_pass_layers: int = 0
    geometric_pass_fraction: float = 0.0
    trimmed_layers: int = 0
    radial_outlier_layers: int = 0
    radial_outlier_points: int = 0
    candidate_strands: int = 0
    persistent_strands: int = 0
    trajectory_candidate_strands: int = 0
    trajectory_merge_count: int = 0
    support_consensus_strands: int = 0
    support_consensus_threshold: int = 0
    layer_count_q75: int = 0
    layer_count_q90: int = 0
    layer_count_max: int = 0
    selected_strand_support_mean: float = 0.0
    selected_strand_support_min: float = 0.0
    count_decision_score: float = 0.0
    layer_count_max_plateau: int = 0
    layer_count_largest_drop: int = 0
    absent_from_max_strands: int = 0
    terminal_strand_support: int = 0
    terminal_consensus_strands: int = 0
    terminal_support_fraction: float = 0.0
    barrel_wall_graph_applied: int = 0
    barrel_wall_graph_count: int = 0
    barrel_wall_graph_variant: str = ""
    barrel_wall_graph_edge_min_count: int = 0
    barrel_wall_graph_visibility_regime: str = ""
    barrel_wall_graph_selection_strategy: str = ""
    barrel_wall_graph_support_ratio: float = 0.0
    barrel_wall_graph_node_count: int = 0
    barrel_wall_graph_edge_count: int = 0
    barrel_wall_graph_component_edge_count: int = 0
    barrel_wall_graph_component_avg_degree: float = 0.0
    barrel_wall_graph_component_cycle_rank: float = 0.0
    barrel_wall_graph_component_degree2_fraction: float = 0.0
    barrel_wall_graph_component_branch_fraction: float = 0.0
    barrel_wall_graph_component_radial_rank_mean: float = 0.0
    barrel_wall_graph_component_radial_rank_min: float = 0.0
    barrel_wall_graph_edge_density: float = 0.0
    barrel_wall_graph_membership_score: float = 0.0
    barrel_wall_graph_membership_block_count: int = 0
    barrel_wall_graph_membership_largest_block_fraction: float = 0.0
    barrel_wall_graph_membership_plateau_edges: int = 0
    barrel_wall_graph_membership_next_drop_fraction: float = 0.0
    barrel_wall_graph_membership_variant_agreement: int = 0
    barrel_wall_graph_observed_layers: int = 0
    barrel_wall_graph_selected_ids: str = ""
    barrel_wall_graph_raw_curve: str = ""
    barrel_wall_graph_outer_curve: str = ""
    barrel_wall_graph_outer_gap_curve: str = ""
    axis_hypothesis_name: str = ""
    axis_hypothesis_score: float = 0.0
    axis_hypothesis_proxy_rank: int = 0
    axis_hypothesis_candidates: str = ""
    run_window_core_applied: int = 0
    run_window_core_score: float = 0.0
    run_window_core_start: int = -1
    run_window_core_stop: int = -1
    run_window_core_trim_left: int = 0
    run_window_core_trim_right: int = 0
    run_window_core_candidate_windows: int = 0
    run_window_core_reanalyzed_windows: int = 0
    run_window_core_base_strand_count: int = 0
    run_window_core_base_support_consensus: int = 0
    run_window_core_base_layer_q90: int = 0
    run_window_core_base_layer_max: int = 0
    run_window_core_layer_max_preservation: float = 0.0
    run_window_core_layer_q90_preservation: float = 0.0
    run_window_core_outside_support_fraction: float = 0.0
    run_window_core_z_coverage_fraction: float = 0.0
    run_window_core_max_z_gap: int = 0
    run_window_core_centrality: float = 0.0
    run_window_core_count_layer_improvement: int = 0
    run_window_core_absent_improvement: float = 0.0
    run_window_core_drop_improvement: float = 0.0
    run_window_core_count_margin: float = 0.0
    chain_residues: int = 0
    sheet_residues: int = 0
    informative_slices: int = 0
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> StrandCountResult:
        kwargs = _coerced_dataclass_kwargs(cls, row, skip={"raw"})
        return cls(**kwargs, raw=dict(row))

    def to_dict(self) -> dict[str, object]:
        data = dict(self.raw)
        for key, value in asdict(self).items():
            if key != "raw":
                data[key] = value
        return data


DetectionResult = StrandCountResult
AnalysisReport = StrandCountReport


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured result for a complete Betlas beta-barrel-staves readout run."""

    rows: list[StrandCountResult]
    input_files: list[str] = field(default_factory=list)
    output_path: str | None = None
    config: object | None = None

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        input_files: Iterable[str] | None = None,
        output_path: str | None = None,
        config: object | None = None,
    ) -> PipelineRunResult:
        return cls(
            rows=[StrandCountResult.from_row(row) for row in rows],
            input_files=list(input_files or []),
            output_path=output_path,
            config=config,
        )

    @property
    def result_counts(self) -> dict[str, int]:
        return dict(Counter(row.result for row in self.rows))

    def to_rows(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def raw_rows(self) -> list[dict[str, object]]:
        """Return original row mappings for backward-compatible callers."""
        return [dict(row.raw) if row.raw else row.to_dict() for row in self.rows]
