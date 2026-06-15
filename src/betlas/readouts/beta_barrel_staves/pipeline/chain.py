from __future__ import annotations

import numpy as np

from ..analysis.analyzer import StrandCountAnalyzer
from ..config import AppConfig
from ..constants import (
    RESULT_COUNTED,
    RESULT_ERROR,
    RESULT_FILTERED_OUT,
    RESULT_LOW_CONFIDENCE,
)
from ..gates.barrel_gate import gate_from_payload
from ..geometry.alignment import PCAAligner
from ..geometry.axis_search import AxisSliceCandidate, alignment_slice_candidates
from ..geometry.slicer import ProteinSlicer


def format_below_threshold_reason(metric_name: str, observed: int, threshold: int) -> str:
    return f"{metric_name} below threshold ({observed} < {threshold})"


def _report_float(report: dict[str, object], key: str) -> float:
    return float(report.get(key, 0.0) or 0.0)


def axis_report_energy(report: dict[str, object], cfg: AppConfig) -> float:
    axis_cfg = cfg.analyzer.axis_search
    return (
        float(axis_cfg.report_confidence_weight) * _report_float(report, "confidence")
        + float(axis_cfg.report_decision_weight)
        * _report_float(report, "count_decision_score")
    )


def _candidate_pool_for_report_scoring(
    candidates: list[AxisSliceCandidate],
    cfg: AppConfig,
) -> list[AxisSliceCandidate]:
    if not bool(cfg.analyzer.axis_search.report_scoring_enabled):
        return [max(candidates, key=lambda candidate: candidate.score_key)]

    base_candidates = [candidate for candidate in candidates if candidate.name == "base_pca"]
    ranked = sorted(candidates, key=lambda candidate: candidate.score_key, reverse=True)
    limit = int(cfg.analyzer.axis_search.report_candidate_limit)
    selected: list[AxisSliceCandidate] = [*base_candidates, *ranked[:limit]]
    unique: list[AxisSliceCandidate] = []
    seen: set[str] = set()
    for candidate in selected:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        unique.append(candidate)
    return unique or ranked[:1]


def select_axis_report(
    candidates: list[AxisSliceCandidate],
    analyzer: StrandCountAnalyzer,
    cfg: AppConfig,
    *,
    min_informative_slices: int,
) -> tuple[dict[str, object] | None, AxisSliceCandidate]:
    scored_candidates = _candidate_pool_for_report_scoring(candidates, cfg)
    analyzable = [
        candidate
        for candidate in scored_candidates
        if len(candidate.slices) >= min_informative_slices
    ]
    if not analyzable:
        return None, max(candidates, key=lambda candidate: len(candidate.slices))

    evaluated: list[tuple[float, dict[str, object], AxisSliceCandidate]] = []
    for candidate in analyzable:
        report = dict(analyzer.analyze(candidate.slices))
        score = axis_report_energy(report, cfg)
        evaluated.append((score, report, candidate))

    score, report, candidate = max(
        evaluated,
        key=lambda item: (
            item[0],
            _report_float(item[1], "confidence"),
            -item[2].proxy_rank,
        ),
    )
    report.update(
        {
            "axis_hypothesis_name": candidate.name,
            "axis_hypothesis_score": float(score),
            "axis_hypothesis_proxy_rank": int(candidate.proxy_rank),
            "axis_hypothesis_candidates": ";".join(
                item.name for item in scored_candidates
            ),
        }
    )
    return report, candidate


def analyze_chain_payload(payload: dict[str, object], cfg: AppConfig) -> dict[str, object]:
    """Analyze one prepared chain payload and return one result-table row."""
    filename = str(payload.get("filename", ""))
    source_path = str(payload.get("source_path", filename))
    chain_id = str(payload.get("chain", ""))
    try:
        chain_index = int(payload.get("_chain_index", 0) or 0)
    except (TypeError, ValueError):
        chain_index = 0
    residues_data = list(payload.get("residues_data", []) or [])
    min_chain_residues = int(cfg.input.min_chain_residues)
    min_sheet_residues = int(cfg.input.min_sheet_residues)
    min_informative_slices = int(cfg.input.min_informative_slices)
    count_cfg = cfg.analyzer.count
    chain_residue_count = len(residues_data)
    dssp_error = str(payload.get("dssp_error", "") or "")
    sheet_residue_coords = [
        residue["coord"] for residue in residues_data if residue.get("is_sheet", False)
    ]
    sheet_residue_count = len(sheet_residue_coords)

    def build_row(
        result: str,
        reason: str,
        report: dict[str, object] | None = None,
        *,
        result_stage: str,
        informative_slices: int | None = None,
    ) -> dict[str, object]:
        report = report or {}
        total_layers = int(report.get("total_layers", informative_slices or 0) or 0)
        usable_layers = int(report.get("usable_layers", 0) or 0)
        supporting_layers = int(report.get("supporting_layers", 0) or 0)
        return {
            "_source_path": source_path,
            "_chain_index": chain_index,
            "filename": filename,
            "chain": chain_id,
            "result": result,
            "result_stage": result_stage,
            "strand_count": int(report.get("strand_count", 0) or 0),
            "confidence": float(report.get("confidence", 0.0) or 0.0),
            "confidence_basis": str(report.get("confidence_basis", "")),
            "count_threshold": float(count_cfg.min_confidence),
            "barrel_gate_enabled": bool(report.get("barrel_gate_enabled", cfg.barrel_gate.enabled)),
            "barrel_gate_passed": bool(report.get("barrel_gate_passed", False)),
            "barrel_gate_result": str(report.get("barrel_gate_result", "")),
            "barrel_gate_score": float(report.get("barrel_gate_score", 0.0) or 0.0),
            "barrel_gate_reason": str(report.get("barrel_gate_reason", "")),
            "layer_count_mode": int(report.get("layer_count_mode", 0) or 0),
            "layer_count_median": float(report.get("layer_count_median", 0.0) or 0.0),
            "supporting_layers": supporting_layers,
            "usable_layers": usable_layers,
            "total_layers": total_layers,
            "layer_support_fraction": float(report.get("layer_support_fraction", 0.0) or 0.0),
            "usable_layer_fraction": float(report.get("usable_layer_fraction", 0.0) or 0.0),
            "geometric_pass_layers": int(report.get("geometric_pass_layers", 0) or 0),
            "geometric_pass_fraction": float(report.get("geometric_pass_fraction", 0.0) or 0.0),
            "trimmed_layers": int(report.get("trimmed_layers", 0) or 0),
            "radial_outlier_layers": int(report.get("radial_outlier_layers", 0) or 0),
            "radial_outlier_points": int(report.get("radial_outlier_points", 0) or 0),
            "candidate_strands": int(report.get("candidate_strands", 0) or 0),
            "persistent_strands": int(report.get("persistent_strands", 0) or 0),
            "trajectory_candidate_strands": int(report.get("trajectory_candidate_strands", 0) or 0),
            "trajectory_merge_count": int(report.get("trajectory_merge_count", 0) or 0),
            "support_consensus_strands": int(report.get("support_consensus_strands", 0) or 0),
            "support_consensus_threshold": int(report.get("support_consensus_threshold", 0) or 0),
            "layer_count_q75": int(report.get("layer_count_q75", 0) or 0),
            "layer_count_q90": int(report.get("layer_count_q90", 0) or 0),
            "layer_count_max": int(report.get("layer_count_max", 0) or 0),
            "selected_strand_support_mean": float(report.get("selected_strand_support_mean", 0.0) or 0.0),
            "selected_strand_support_min": float(report.get("selected_strand_support_min", 0.0) or 0.0),
            "count_decision_score": float(report.get("count_decision_score", 0.0) or 0.0),
            "layer_count_max_plateau": int(report.get("layer_count_max_plateau", 0) or 0),
            "layer_count_largest_drop": int(report.get("layer_count_largest_drop", 0) or 0),
            "absent_from_max_strands": int(report.get("absent_from_max_strands", 0) or 0),
            "terminal_strand_support": int(report.get("terminal_strand_support", 0) or 0),
            "terminal_consensus_strands": int(report.get("terminal_consensus_strands", 0) or 0),
            "terminal_support_fraction": float(report.get("terminal_support_fraction", 0.0) or 0.0),
            "barrel_wall_graph_applied": int(report.get("barrel_wall_graph_applied", 0) or 0),
            "barrel_wall_graph_count": int(report.get("barrel_wall_graph_count", 0) or 0),
            "barrel_wall_graph_variant": str(report.get("barrel_wall_graph_variant", "")),
            "barrel_wall_graph_edge_min_count": int(
                report.get("barrel_wall_graph_edge_min_count", 0) or 0
            ),
            "barrel_wall_graph_visibility_regime": str(
                report.get("barrel_wall_graph_visibility_regime", "")
            ),
            "barrel_wall_graph_selection_strategy": str(
                report.get("barrel_wall_graph_selection_strategy", "")
            ),
            "barrel_wall_graph_support_ratio": float(
                report.get("barrel_wall_graph_support_ratio", 0.0) or 0.0
            ),
            "barrel_wall_graph_node_count": int(
                report.get("barrel_wall_graph_node_count", 0) or 0
            ),
            "barrel_wall_graph_edge_count": int(
                report.get("barrel_wall_graph_edge_count", 0) or 0
            ),
            "barrel_wall_graph_component_edge_count": int(
                report.get("barrel_wall_graph_component_edge_count", 0) or 0
            ),
            "barrel_wall_graph_component_avg_degree": float(
                report.get("barrel_wall_graph_component_avg_degree", 0.0) or 0.0
            ),
            "barrel_wall_graph_component_cycle_rank": float(
                report.get("barrel_wall_graph_component_cycle_rank", 0.0) or 0.0
            ),
            "barrel_wall_graph_component_degree2_fraction": float(
                report.get("barrel_wall_graph_component_degree2_fraction", 0.0) or 0.0
            ),
            "barrel_wall_graph_component_branch_fraction": float(
                report.get("barrel_wall_graph_component_branch_fraction", 0.0) or 0.0
            ),
            "barrel_wall_graph_component_radial_rank_mean": float(
                report.get("barrel_wall_graph_component_radial_rank_mean", 0.0) or 0.0
            ),
            "barrel_wall_graph_component_radial_rank_min": float(
                report.get("barrel_wall_graph_component_radial_rank_min", 0.0) or 0.0
            ),
            "barrel_wall_graph_edge_density": float(
                report.get("barrel_wall_graph_edge_density", 0.0) or 0.0
            ),
            "barrel_wall_graph_membership_score": float(
                report.get("barrel_wall_graph_membership_score", 0.0) or 0.0
            ),
            "barrel_wall_graph_membership_block_count": int(
                report.get("barrel_wall_graph_membership_block_count", 0) or 0
            ),
            "barrel_wall_graph_membership_largest_block_fraction": float(
                report.get(
                    "barrel_wall_graph_membership_largest_block_fraction",
                    0.0,
                )
                or 0.0
            ),
            "barrel_wall_graph_membership_plateau_edges": int(
                report.get("barrel_wall_graph_membership_plateau_edges", 0) or 0
            ),
            "barrel_wall_graph_membership_next_drop_fraction": float(
                report.get("barrel_wall_graph_membership_next_drop_fraction", 0.0)
                or 0.0
            ),
            "barrel_wall_graph_membership_variant_agreement": int(
                report.get("barrel_wall_graph_membership_variant_agreement", 0) or 0
            ),
            "barrel_wall_graph_observed_layers": int(
                report.get("barrel_wall_graph_observed_layers", 0) or 0
            ),
            "barrel_wall_graph_selected_ids": str(
                report.get("barrel_wall_graph_selected_ids", "")
            ),
            "barrel_wall_graph_raw_curve": str(report.get("barrel_wall_graph_raw_curve", "")),
            "barrel_wall_graph_outer_curve": str(
                report.get("barrel_wall_graph_outer_curve", "")
            ),
            "barrel_wall_graph_outer_gap_curve": str(
                report.get("barrel_wall_graph_outer_gap_curve", "")
            ),
            "axis_hypothesis_name": str(report.get("axis_hypothesis_name", "")),
            "axis_hypothesis_score": float(
                report.get("axis_hypothesis_score", 0.0) or 0.0
            ),
            "axis_hypothesis_proxy_rank": int(
                report.get("axis_hypothesis_proxy_rank", 0) or 0
            ),
            "axis_hypothesis_candidates": str(
                report.get("axis_hypothesis_candidates", "")
            ),
            "run_window_core_applied": int(report.get("run_window_core_applied", 0) or 0),
            "run_window_core_score": float(report.get("run_window_core_score", 0.0) or 0.0),
            "run_window_core_start": int(report.get("run_window_core_start", -1)),
            "run_window_core_stop": int(report.get("run_window_core_stop", -1)),
            "run_window_core_trim_left": int(report.get("run_window_core_trim_left", 0) or 0),
            "run_window_core_trim_right": int(report.get("run_window_core_trim_right", 0) or 0),
            "run_window_core_candidate_windows": int(
                report.get("run_window_core_candidate_windows", 0) or 0
            ),
            "run_window_core_reanalyzed_windows": int(
                report.get("run_window_core_reanalyzed_windows", 0) or 0
            ),
            "run_window_core_base_strand_count": int(
                report.get("run_window_core_base_strand_count", 0) or 0
            ),
            "run_window_core_base_support_consensus": int(
                report.get("run_window_core_base_support_consensus", 0) or 0
            ),
            "run_window_core_base_layer_q90": int(
                report.get("run_window_core_base_layer_q90", 0) or 0
            ),
            "run_window_core_base_layer_max": int(
                report.get("run_window_core_base_layer_max", 0) or 0
            ),
            "run_window_core_layer_max_preservation": float(
                report.get("run_window_core_layer_max_preservation", 0.0) or 0.0
            ),
            "run_window_core_layer_q90_preservation": float(
                report.get("run_window_core_layer_q90_preservation", 0.0) or 0.0
            ),
            "run_window_core_outside_support_fraction": float(
                report.get("run_window_core_outside_support_fraction", 0.0) or 0.0
            ),
            "run_window_core_z_coverage_fraction": float(
                report.get("run_window_core_z_coverage_fraction", 0.0) or 0.0
            ),
            "run_window_core_max_z_gap": int(report.get("run_window_core_max_z_gap", 0) or 0),
            "run_window_core_centrality": float(
                report.get("run_window_core_centrality", 0.0) or 0.0
            ),
            "run_window_core_count_layer_improvement": int(
                report.get("run_window_core_count_layer_improvement", 0) or 0
            ),
            "run_window_core_absent_improvement": float(
                report.get("run_window_core_absent_improvement", 0.0) or 0.0
            ),
            "run_window_core_drop_improvement": float(
                report.get("run_window_core_drop_improvement", 0.0) or 0.0
            ),
            "run_window_core_count_margin": float(
                report.get("run_window_core_count_margin", 0.0) or 0.0
            ),
            "chain_residues": int(report.get("chain_residues", chain_residue_count)),
            "sheet_residues": int(report.get("sheet_residues", sheet_residue_count)),
            "informative_slices": int(
                report.get(
                    "informative_slices",
                    informative_slices if informative_slices is not None else total_layers,
                )
            ),
            "reason": reason,
        }

    if chain_residue_count < min_chain_residues:
        return build_row(
            RESULT_FILTERED_OUT,
            format_below_threshold_reason("Chain residues", chain_residue_count, min_chain_residues),
            result_stage="prefilter",
        )

    all_coordinates = np.array([residue["coord"] for residue in residues_data], dtype=float)
    if sheet_residue_count < min_sheet_residues:
        reason = format_below_threshold_reason(
            "Beta-sheet residues",
            sheet_residue_count,
            min_sheet_residues,
        )
        if dssp_error:
            reason = f"DSSP failed during preparation; {reason}. DSSP error: {dssp_error}"
        return build_row(
            RESULT_FILTERED_OUT,
            reason,
            result_stage="prefilter",
        )

    gate_report: dict[str, object] = {}
    barrel_gate = gate_from_payload(payload)
    if cfg.barrel_gate.enabled and barrel_gate is None:
        return build_row(
            RESULT_ERROR,
            "Missing external barrel-gate payload; prepare the chain before analysis.",
            {
                "barrel_gate_enabled": True,
                "barrel_gate_passed": False,
                "barrel_gate_result": "ERROR",
            },
            result_stage="barrel_gate",
        )
    if barrel_gate is not None:
        gate_report = {
            "barrel_gate_enabled": barrel_gate.enabled,
            "barrel_gate_passed": barrel_gate.passed,
            "barrel_gate_result": barrel_gate.result,
            "barrel_gate_score": barrel_gate.score,
            "barrel_gate_reason": barrel_gate.reason,
        }
        if barrel_gate.enabled and not barrel_gate.passed:
            return build_row(
                RESULT_FILTERED_OUT,
                f"External barrel gate did not pass ({barrel_gate.result}: {barrel_gate.reason})",
                gate_report,
                result_stage="barrel_gate",
            )

    sheet_coordinates = np.array(sheet_residue_coords, dtype=float)
    slicer = ProteinSlicer(
        step_size=cfg.slicer.step_size,
        fill_sheet_hole_length=cfg.slicer.fill_sheet_hole_length,
    )

    try:
        aligner = PCAAligner()
        aligner.fit(sheet_coordinates)
        candidates = alignment_slice_candidates(
            aligner,
            all_coordinates,
            residues_data,
            slicer,
            cfg,
        )
    except Exception as exc:
        return build_row(RESULT_ERROR, f"Alignment failed: {exc}", result_stage="error")

    if not candidates:
        return build_row(RESULT_ERROR, "Alignment produced no slice candidates.", result_stage="error")

    analyzer = StrandCountAnalyzer(cfg.analyzer)
    try:
        report, selected_candidate = select_axis_report(
            candidates,
            analyzer,
            cfg,
            min_informative_slices=min_informative_slices,
        )
    except Exception as exc:
        total_layers = max((len(candidate.slices) for candidate in candidates), default=0)
        error_report = {"total_layers": total_layers}
        return build_row(
            RESULT_ERROR,
            f"Slice analyzer crashed unexpectedly: {exc}",
            error_report,
            result_stage="error",
            informative_slices=total_layers,
        )

    informative_slice_count = len(selected_candidate.slices)
    if report is None:
        return build_row(
            RESULT_FILTERED_OUT,
            format_below_threshold_reason(
                "Informative slices",
                informative_slice_count,
                min_informative_slices,
            ),
            result_stage="prefilter",
            informative_slices=informative_slice_count,
        )

    report.update(
        {
            "chain_residues": chain_residue_count,
            "sheet_residues": sheet_residue_count,
            "informative_slices": informative_slice_count,
            **gate_report,
        }
    )

    confidence = float(report.get("confidence", 0.0) or 0.0)
    strand_count = int(report.get("strand_count", 0) or 0)
    usable_layers = int(report.get("usable_layers", 0) or 0)
    if (
        strand_count > 0
        and usable_layers >= int(count_cfg.min_usable_layers)
        and confidence >= float(count_cfg.min_confidence)
    ):
        return build_row(
            RESULT_COUNTED,
            "OK",
            report,
            result_stage="count",
            informative_slices=informative_slice_count,
        )

    if usable_layers < int(count_cfg.min_usable_layers):
        reason = (
            f"Too few usable slices for a stable count "
            f"({usable_layers} < {int(count_cfg.min_usable_layers)})"
        )
    elif strand_count <= 0:
        reason = "No persistent strand set could be estimated"
    else:
        reason = (
            f"Low-confidence strand count "
            f"({confidence:.2f} < {float(count_cfg.min_confidence):.2f})"
        )

    return build_row(
        RESULT_LOW_CONFIDENCE,
        reason,
        report,
        result_stage="count",
        informative_slices=informative_slice_count,
    )
