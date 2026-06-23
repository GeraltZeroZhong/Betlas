from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import ceil

import numpy as np

from ..config import AnalyzerConfig
from .barrel_wall import (
    BarrelWallGraphCoreSelector,
    BarrelWallGraphSelection,
    empty_barrel_wall_feature_table,
)
from .confidence import ConfidenceEstimator
from .count_decision import CountDecisionEngine
from .layer import LayerAnalyzer
from .numeric import mode_count, quantile_count
from .run_window_core import RunWindowCoreSelector, report_feature_table
from .sequence_core import GlobalSequenceCoreSelector
from .trajectory import TrajectoryMerger

COMPAT_ANALYZER_OVERRIDE_PATHS = {
    "min_points": ("count", "min_points_per_layer"),
    "min_points_per_layer": ("count", "min_points_per_layer"),
    "min_usable_layers": ("count", "min_usable_layers"),
    "min_strand_support_layers": ("count", "min_strand_support_layers"),
    "min_strand_support_fraction": ("count", "min_strand_support_fraction"),
    "consensus_min_strand_support_layers": ("count", "consensus_min_strand_support_layers"),
    "consensus_min_strand_support_fraction": ("count", "consensus_min_strand_support_fraction"),
    "trajectory_merge_enabled": ("count", "trajectory_merge_enabled"),
    "trajectory_merge_max_run_id_gap": ("count", "trajectory_merge_max_run_id_gap"),
    "trajectory_merge_max_layer_gap": ("count", "trajectory_merge_max_layer_gap"),
    "trajectory_merge_max_overlap_fraction": ("count", "trajectory_merge_max_overlap_fraction"),
    "trajectory_merge_angle_tolerance_deg": ("count", "trajectory_merge_angle_tolerance_deg"),
    "trajectory_merge_slope_tolerance_deg_per_layer": (
        "count",
        "trajectory_merge_slope_tolerance_deg_per_layer",
    ),
    "trajectory_merge_extended_enabled": ("count", "trajectory_merge_extended_enabled"),
    "trajectory_merge_extended_max_run_id_gap": (
        "count",
        "trajectory_merge_extended_max_run_id_gap",
    ),
    "trajectory_merge_extended_angle_tolerance_deg": (
        "count",
        "trajectory_merge_extended_angle_tolerance_deg",
    ),
    "trajectory_merge_extended_min_candidate_strands": (
        "count",
        "trajectory_merge_extended_min_candidate_strands",
    ),
    "trajectory_merge_extended_min_layer_q90": (
        "count",
        "trajectory_merge_extended_min_layer_q90",
    ),
    "trajectory_merge_extended_min_support_gap": (
        "count",
        "trajectory_merge_extended_min_support_gap",
    ),
    "min_confidence": ("count", "min_confidence"),
    "min_count": ("count", "min_count"),
    "max_count": ("count", "max_count"),
    "require_geometric_consistency": ("count", "require_geometric_consistency"),
    "nn_rule_enabled": ("rules", "nearest_neighbor", "enabled"),
    "nn_max_robust_cv": ("rules", "nearest_neighbor", "max_robust_cv"),
    "nn_min_inlier_frac": ("rules", "nearest_neighbor", "min_inlier_frac"),
    "angle_rule_enabled": ("rules", "angle", "enabled"),
    "angle_max_gap_deg": ("rules", "angle", "max_gap_deg"),
    "angle_order_rule_enabled": ("rules", "angle", "order", "enabled"),
    "angle_order_local_step_max": ("rules", "angle", "order", "local_step_max"),
    "angle_order_min_local_frac": ("rules", "angle", "order", "min_local_frac"),
    "angle_order_max_mean_circ_dist_norm": (
        "rules",
        "angle",
        "order",
        "max_mean_circ_dist_norm",
    ),
    "sequence_core_rule_enabled": ("rules", "sequence_core", "enabled"),
    "sequence_core_max_exhaustive_points": ("rules", "sequence_core", "max_exhaustive_points"),
    "sequence_core_global_enabled": ("rules", "sequence_core", "global_enabled"),
    "sequence_core_global_min_layer_count_max": (
        "rules",
        "sequence_core",
        "global_min_layer_count_max",
    ),
    "sequence_core_global_min_trajectory_strands": (
        "rules",
        "sequence_core",
        "global_min_trajectory_strands",
    ),
    "sequence_core_global_min_support_gap": (
        "rules",
        "sequence_core",
        "global_min_support_gap",
    ),
    "sequence_core_global_min_prediction_gap": (
        "rules",
        "sequence_core",
        "global_min_prediction_gap",
    ),
    "sequence_core_global_layer_count_allowance": (
        "rules",
        "sequence_core",
        "global_layer_count_allowance",
    ),
    "radial_outlier_rule_enabled": ("rules", "radial_outlier", "enabled"),
}


class StrandCountAnalyzer:
    """
    Estimate beta-barrel stave count from DSSP-derived slice points.

    The analyzer is intentionally a coordinator. Scientific mechanisms live in
    dedicated modules so each slice-only assumption can be audited separately.
    """

    def __init__(self, config: AnalyzerConfig | None = None, **compat_overrides):
        self.config = deepcopy(config or AnalyzerConfig())
        self._apply_compat_overrides(compat_overrides)
        self.layer_analyzer = LayerAnalyzer(self.config)
        self.window_config = deepcopy(self.config)
        self.window_config.rules.sequence_core.enabled = False
        self.window_config.rules.sequence_core.global_enabled = False
        self.window_config.rules.sequence_core.run_window_enabled = False
        self.window_layer_analyzer = LayerAnalyzer(self.window_config)
        self.trajectory_merger = TrajectoryMerger(self.config.count)
        self.count_decision = CountDecisionEngine(self.config.count)
        self.global_sequence_core = GlobalSequenceCoreSelector(self.config.rules.sequence_core)
        self.run_window_core = RunWindowCoreSelector(self.config.rules.sequence_core)
        self.barrel_wall_graph = BarrelWallGraphCoreSelector(
            self.config.rules.barrel_wall_graph
        )
        self.confidence_estimator = ConfidenceEstimator(self.config.confidence)

    def _apply_compat_overrides(self, overrides: dict[str, object]) -> None:
        for override_name, override_value in overrides.items():
            if override_name not in COMPAT_ANALYZER_OVERRIDE_PATHS:
                raise TypeError(f"Unknown analyzer override: {override_name}")

            target = self.config
            path = COMPAT_ANALYZER_OVERRIDE_PATHS[override_name]
            for part in path[:-1]:
                target = getattr(target, part)
            setattr(target, path[-1], override_value)

    def _empty_report(self, reason: str) -> dict[str, object]:
        return {
            "strand_count": 0,
            "confidence": 0.0,
            "confidence_basis": reason,
            "layer_count_mode": 0,
            "layer_count_median": 0.0,
            "supporting_layers": 0,
            "usable_layers": 0,
            "total_layers": 0,
            "layer_support_fraction": 0.0,
            "usable_layer_fraction": 0.0,
            "geometric_pass_layers": 0,
            "geometric_pass_fraction": 0.0,
            "trimmed_layers": 0,
            "radial_outlier_layers": 0,
            "radial_outlier_points": 0,
            "candidate_strands": 0,
            "persistent_strands": 0,
            "trajectory_candidate_strands": 0,
            "trajectory_merge_count": 0,
            "support_consensus_strands": 0,
            "support_consensus_threshold": 0,
            "layer_count_q75": 0,
            "layer_count_q90": 0,
            "layer_count_max": 0,
            "selected_strand_support_mean": 0.0,
            "selected_strand_support_min": 0.0,
            "count_decision_score": 0.0,
            "global_sequence_core_strands": 0,
            "global_sequence_core_trim_left": 0,
            "global_sequence_core_trim_right": 0,
            "layer_count_max_plateau": 0,
            "layer_count_largest_drop": 0,
            "absent_from_max_strands": 0,
            "terminal_strand_support": 0,
            "terminal_consensus_strands": 0,
            "terminal_support_fraction": 0.0,
            **empty_barrel_wall_feature_table(),
            "run_window_core_applied": 0,
            "run_window_core_score": 0.0,
            "run_window_core_start": -1,
            "run_window_core_stop": -1,
            "run_window_core_trim_left": 0,
            "run_window_core_trim_right": 0,
            "run_window_core_candidate_windows": 0,
            "run_window_core_reanalyzed_windows": 0,
            "run_window_core_base_strand_count": 0,
            "run_window_core_base_support_consensus": 0,
            "run_window_core_base_layer_q90": 0,
            "run_window_core_base_layer_max": 0,
            "run_window_core_layer_max_preservation": 0.0,
            "run_window_core_layer_q90_preservation": 0.0,
            "run_window_core_outside_support_fraction": 0.0,
            "run_window_core_z_coverage_fraction": 0.0,
            "run_window_core_max_z_gap": 0,
            "run_window_core_centrality": 0.0,
            "run_window_core_count_layer_improvement": 0,
            "run_window_core_absent_improvement": 0.0,
            "run_window_core_drop_improvement": 0.0,
            "run_window_core_count_margin": 0.0,
            "layer_details": [],
        }

    def _layer_support_thresholds(self, usable_layer_count: int) -> tuple[int, int]:
        count_cfg = self.config.count
        support_threshold = max(
            int(count_cfg.min_strand_support_layers),
            int(ceil(usable_layer_count * float(count_cfg.min_strand_support_fraction))),
        )
        consensus_threshold = max(
            int(count_cfg.consensus_min_strand_support_layers),
            int(ceil(usable_layer_count * float(count_cfg.consensus_min_strand_support_fraction))),
        )
        return support_threshold, consensus_threshold

    def _apply_extended_trajectory_merge(
        self,
        *,
        strand_support: dict[int, int],
        usable_layers: list[dict[str, object]],
        trajectory_support: dict[int, int],
        trajectory_fragments: dict[int, list[int]],
        trajectory_merge_count: int,
        support_layer_threshold: int,
        consensus_layer_threshold: int,
        support_consensus_ids: list[int],
        candidate_strands: int,
        layer_count_q90: int,
        layer_count_max: int,
    ) -> tuple[dict[int, int], dict[int, list[int]], int, list[int], list[int]]:
        if not self.trajectory_merger.should_use_extended_merge(
            candidate_strands=candidate_strands,
            support_consensus_strands=len(support_consensus_ids),
            layer_count_q90=layer_count_q90,
            layer_count_max=layer_count_max,
        ):
            persistent_ids = sorted(
                strand_id
                for strand_id, support in trajectory_support.items()
                if support >= support_layer_threshold
            )
            return (
                trajectory_support,
                trajectory_fragments,
                trajectory_merge_count,
                persistent_ids,
                support_consensus_ids,
            )

        extended = self.trajectory_merger.merge(
            strand_support,
            usable_layers,
            max_run_id_gap=int(self.config.count.trajectory_merge_extended_max_run_id_gap),
            angle_tolerance_deg=float(self.config.count.trajectory_merge_extended_angle_tolerance_deg),
        )
        extended_support_consensus_ids = sorted(
            strand_id
            for strand_id, support in extended.support.items()
            if support >= consensus_layer_threshold
        )
        if len(extended_support_consensus_ids) >= len(support_consensus_ids):
            persistent_ids = sorted(
                strand_id
                for strand_id, support in trajectory_support.items()
                if support >= support_layer_threshold
            )
            return (
                trajectory_support,
                trajectory_fragments,
                trajectory_merge_count,
                persistent_ids,
                support_consensus_ids,
            )

        persistent_ids = sorted(
            strand_id
            for strand_id, support in extended.support.items()
            if support >= support_layer_threshold
        )
        return (
            extended.support,
            extended.fragments,
            extended.merge_count,
            persistent_ids,
            extended_support_consensus_ids,
        )

    def analyze(self, slices_dict: dict[float, list[tuple[float, ...]]]) -> dict[str, object]:
        """Analyze active slices and return a strand-count report."""
        return self._analyze_once(slices_dict, apply_run_window_core=True)

    def _analyze_once(
        self,
        slices_dict: dict[float, list[tuple[float, ...]]],
        *,
        apply_run_window_core: bool,
        apply_global_sequence_core: bool = True,
    ) -> dict[str, object]:
        active_layers = [z_value for z_value in sorted(slices_dict) if slices_dict[z_value]]
        total_layers = len(active_layers)
        if total_layers == 0:
            return self._empty_report("no_active_slices")

        layer_analyzer = self.layer_analyzer
        if not apply_run_window_core and not apply_global_sequence_core:
            layer_analyzer = self.window_layer_analyzer

        layer_details = [
            layer_analyzer.analyze_layer(
                z_value,
                np.asarray(slices_dict[z_value], dtype=float),
            )
            for z_value in active_layers
        ]
        usable_layers = [layer for layer in layer_details if bool(layer.get("usable", False))]
        usable_layer_count = len(usable_layers)

        geometric_pass_layers = sum(
            1 for layer in layer_details if bool(layer.get("geometric_pass", False))
        )
        trimmed_layers = sum(
            1
            for layer in layer_details
            if int(layer.get("seq_core_trim_left", 0) or 0)
            or int(layer.get("seq_core_trim_right", 0) or 0)
        )
        radial_outlier_points = sum(int(layer.get("radial_outliers", 0) or 0) for layer in layer_details)
        radial_outlier_layers = sum(
            1 for layer in layer_details if int(layer.get("radial_outliers", 0) or 0) > 0
        )
        geometric_pass_fraction = geometric_pass_layers / total_layers if total_layers else 0.0
        usable_layer_fraction = usable_layer_count / total_layers if total_layers else 0.0

        layer_counts = [int(layer["layer_count"]) for layer in usable_layers]
        layer_count_mode = mode_count(layer_counts)
        layer_count_median = float(np.median(np.asarray(layer_counts, dtype=float))) if layer_counts else 0.0
        layer_count_q75 = quantile_count(layer_counts, 0.75)
        layer_count_q90 = quantile_count(layer_counts, 0.90)
        layer_count_max = max(layer_counts) if layer_counts else 0

        strand_support: dict[int, int] = defaultdict(int)
        for layer in usable_layers:
            for strand_id in list(layer.get("strand_ids", []) or []):
                strand_support[int(strand_id)] += 1
        raw_strand_support = dict(strand_support)

        merged = self.trajectory_merger.merge(raw_strand_support, usable_layers)
        trajectory_support = merged.support
        trajectory_fragments = merged.fragments
        trajectory_merge_count = merged.merge_count

        support_layer_threshold, consensus_layer_threshold = self._layer_support_thresholds(
            usable_layer_count
        )
        support_consensus_ids = sorted(
            strand_id
            for strand_id, support in trajectory_support.items()
            if support >= consensus_layer_threshold
        )
        candidate_strands = len(raw_strand_support)

        (
            trajectory_support,
            trajectory_fragments,
            trajectory_merge_count,
            persistent_ids,
            support_consensus_ids,
        ) = self._apply_extended_trajectory_merge(
            strand_support=raw_strand_support,
            usable_layers=usable_layers,
            trajectory_support=trajectory_support,
            trajectory_fragments=trajectory_fragments,
            trajectory_merge_count=trajectory_merge_count,
            support_layer_threshold=support_layer_threshold,
            consensus_layer_threshold=consensus_layer_threshold,
            support_consensus_ids=support_consensus_ids,
            candidate_strands=candidate_strands,
            layer_count_q90=layer_count_q90,
            layer_count_max=layer_count_max,
        )
        trajectory_candidate_strands = len(trajectory_support)
        persistent_strands = len(persistent_ids)

        decision = self.count_decision.select(
            strand_support=dict(trajectory_support),
            support_consensus_ids=support_consensus_ids,
            persistent_ids=persistent_ids,
            consensus_threshold=consensus_layer_threshold,
            layer_counts=layer_counts,
            layer_count_mode=layer_count_mode,
            layer_count_q75=layer_count_q75,
            layer_count_q90=layer_count_q90,
            layer_count_max=layer_count_max,
            usable_layer_count=usable_layer_count,
        )
        strand_count = int(decision.strand_count)
        confidence_basis = decision.confidence_basis
        selected_ids = [int(value) for value in decision.selected_ids]
        count_decision_score = float(decision.decision_score)

        global_core = None
        if apply_global_sequence_core:
            global_core = self.global_sequence_core.select(
                trajectory_support=dict(trajectory_support),
                trajectory_fragments=trajectory_fragments,
                support_consensus_ids=support_consensus_ids,
                usable_layers=usable_layers,
                current_count=strand_count,
                layer_count_q90=layer_count_q90,
                layer_count_max=layer_count_max,
                usable_layer_count=usable_layer_count,
            )
        global_sequence_core_strands = 0
        global_sequence_core_trim_left = 0
        global_sequence_core_trim_right = 0
        if global_core is not None and 0 < global_core.strand_count < strand_count:
            strand_count = int(global_core.strand_count)
            confidence_basis = "global_sequence_core"
            selected_ids = [int(value) for value in global_core.selected_ids]
            count_decision_score = max(count_decision_score, float(global_core.score))
            global_sequence_core_strands = int(global_core.strand_count)
            global_sequence_core_trim_left = int(global_core.trim_left)
            global_sequence_core_trim_right = int(global_core.trim_right)

        supporting_layers = self.confidence_estimator.supporting_layer_count(
            layer_counts=layer_counts,
            strand_count=strand_count,
            selected_ids=selected_ids,
            trajectory_fragments=trajectory_fragments,
            usable_layers=usable_layers,
            min_layer_coverage_fraction=float(self.config.count.min_layer_coverage_fraction),
        )
        layer_support_fraction = supporting_layers / usable_layer_count if usable_layer_count else 0.0

        if strand_count <= 0 or usable_layer_count < int(self.config.count.min_usable_layers):
            confidence = 0.0
            selected_strand_support_mean = 0.0
            selected_strand_support_min = 0.0
        else:
            confidence_result = self.confidence_estimator.estimate(
                strand_count=strand_count,
                selected_ids=selected_ids,
                trajectory_support=trajectory_support,
                usable_layer_count=usable_layer_count,
                layer_counts=layer_counts,
                support_consensus_count=len(support_consensus_ids),
                trajectory_candidate_count=trajectory_candidate_strands,
                layer_count_q90=layer_count_q90,
                count_decision_score=count_decision_score,
                layer_support_fraction=layer_support_fraction,
                usable_layer_fraction=usable_layer_fraction,
                geometric_pass_fraction=geometric_pass_fraction,
                radial_outlier_layers=radial_outlier_layers,
                trimmed_layers=trimmed_layers,
            )
            confidence = confidence_result.confidence
            selected_strand_support_mean = confidence_result.selected_support_mean
            selected_strand_support_min = confidence_result.selected_support_min

        report = {
            "strand_count": int(strand_count),
            "confidence": float(confidence),
            "confidence_basis": confidence_basis,
            "layer_count_mode": int(layer_count_mode),
            "layer_count_median": float(layer_count_median),
            "supporting_layers": int(supporting_layers),
            "usable_layers": int(usable_layer_count),
            "total_layers": int(total_layers),
            "layer_support_fraction": float(layer_support_fraction),
            "usable_layer_fraction": float(usable_layer_fraction),
            "geometric_pass_layers": int(geometric_pass_layers),
            "geometric_pass_fraction": float(geometric_pass_fraction),
            "trimmed_layers": int(trimmed_layers),
            "radial_outlier_layers": int(radial_outlier_layers),
            "radial_outlier_points": int(radial_outlier_points),
            "candidate_strands": int(candidate_strands),
            "persistent_strands": int(persistent_strands),
            "trajectory_candidate_strands": int(trajectory_candidate_strands),
            "trajectory_merge_count": int(trajectory_merge_count),
            "support_consensus_strands": int(len(support_consensus_ids)),
            "support_consensus_threshold": int(consensus_layer_threshold),
            "layer_count_q75": int(layer_count_q75),
            "layer_count_q90": int(layer_count_q90),
            "layer_count_max": int(layer_count_max),
            "selected_strand_support_mean": float(selected_strand_support_mean),
            "selected_strand_support_min": float(selected_strand_support_min),
            "count_decision_score": float(count_decision_score),
            "global_sequence_core_strands": int(global_sequence_core_strands),
            "global_sequence_core_trim_left": int(global_sequence_core_trim_left),
            "global_sequence_core_trim_right": int(global_sequence_core_trim_right),
            "strand_support": {str(key): int(value) for key, value in sorted(raw_strand_support.items())},
            "trajectory_strand_support": {
                str(key): int(value) for key, value in sorted(trajectory_support.items())
            },
            "trajectory_fragments": {
                str(key): [int(value) for value in values]
                for key, values in sorted(trajectory_fragments.items())
            },
            "layer_details": [
                {key: value for key, value in layer.items() if key != "_points"}
                for layer in layer_details
            ],
        }
        report.update(
            report_feature_table(
                report,
                terminal_band_size=int(
                    self.config.rules.sequence_core.run_window_terminal_band_size
                ),
            )
        )
        report.update(empty_barrel_wall_feature_table())
        report.update(self._empty_run_window_core_report_fields(report))
        if apply_run_window_core:
            graph_selection = self.barrel_wall_graph.select(slices_dict, report)
            if graph_selection is not None:
                report.update(graph_selection.feature_table)
                if int(graph_selection.strand_count) != int(report.get("strand_count", 0) or 0):
                    return self._replace_with_barrel_wall_graph_count(
                        report,
                        graph_selection,
                        raw_strand_support=raw_strand_support,
                        usable_layer_count=usable_layer_count,
                        layer_counts=layer_counts,
                        usable_layers=usable_layers,
                        support_consensus_count=len(support_consensus_ids),
                        trajectory_candidate_strands=trajectory_candidate_strands,
                        layer_count_q90=layer_count_q90,
                        usable_layer_fraction=usable_layer_fraction,
                        geometric_pass_fraction=geometric_pass_fraction,
                        radial_outlier_layers=radial_outlier_layers,
                        trimmed_layers=trimmed_layers,
                    )

            selection = self.run_window_core.select(
                slices_dict,
                report,
                lambda filtered_slices: self._analyze_once(
                    filtered_slices,
                    apply_run_window_core=False,
                    apply_global_sequence_core=False,
                ),
            )
            if selection is not None:
                selected_report = dict(selection.report)
                selected_report["confidence_basis"] = "run_window_core"
                selected_report["run_window_core_applied"] = 1
                selected_report.update(selection.feature_table)
                selected_report.update(self._base_run_window_core_report_fields(report))
                return selected_report
        return report

    def _replace_with_barrel_wall_graph_count(
        self,
        report: dict[str, object],
        graph_selection: BarrelWallGraphSelection,
        *,
        raw_strand_support: dict[int, int],
        usable_layer_count: int,
        layer_counts: list[int],
        usable_layers: list[dict[str, object]],
        support_consensus_count: int,
        trajectory_candidate_strands: int,
        layer_count_q90: int,
        usable_layer_fraction: float,
        geometric_pass_fraction: float,
        radial_outlier_layers: int,
        trimmed_layers: int,
    ) -> dict[str, object]:
        selected_ids = [int(value) for value in graph_selection.selected_ids]
        graph_fragments = {strand_id: [strand_id] for strand_id in selected_ids}
        decision_score = max(
            float(report.get("count_decision_score", 0.0) or 0.0),
            float(graph_selection.score),
        )
        supporting_layers = self.confidence_estimator.supporting_layer_count(
            layer_counts=layer_counts,
            strand_count=int(graph_selection.strand_count),
            selected_ids=selected_ids,
            trajectory_fragments=graph_fragments,
            usable_layers=usable_layers,
            min_layer_coverage_fraction=float(self.config.count.min_layer_coverage_fraction),
        )
        layer_support_fraction = supporting_layers / usable_layer_count if usable_layer_count else 0.0
        if (
            int(graph_selection.strand_count) <= 0
            or usable_layer_count < int(self.config.count.min_usable_layers)
        ):
            confidence = 0.0
            selected_support_mean = 0.0
            selected_support_min = 0.0
        else:
            confidence_result = self.confidence_estimator.estimate(
                strand_count=int(graph_selection.strand_count),
                selected_ids=selected_ids,
                trajectory_support=raw_strand_support,
                usable_layer_count=usable_layer_count,
                layer_counts=layer_counts,
                support_consensus_count=support_consensus_count,
                trajectory_candidate_count=trajectory_candidate_strands,
                layer_count_q90=layer_count_q90,
                count_decision_score=decision_score,
                layer_support_fraction=layer_support_fraction,
                usable_layer_fraction=usable_layer_fraction,
                geometric_pass_fraction=geometric_pass_fraction,
                radial_outlier_layers=radial_outlier_layers,
                trimmed_layers=trimmed_layers,
            )
            confidence = confidence_result.confidence
            selected_support_mean = confidence_result.selected_support_mean
            selected_support_min = confidence_result.selected_support_min

        report.update(
            {
                "strand_count": int(graph_selection.strand_count),
                "confidence": float(confidence),
                "confidence_basis": graph_selection.confidence_basis,
                "supporting_layers": int(supporting_layers),
                "layer_support_fraction": float(layer_support_fraction),
                "selected_strand_support_mean": float(selected_support_mean),
                "selected_strand_support_min": float(selected_support_min),
                "count_decision_score": float(decision_score),
            }
        )
        return report

    @staticmethod
    def _base_run_window_core_report_fields(report: dict[str, object]) -> dict[str, object]:
        return {
            "run_window_core_base_strand_count": int(report.get("strand_count", 0) or 0),
            "run_window_core_base_support_consensus": int(
                report.get("support_consensus_strands", 0) or 0
            ),
            "run_window_core_base_layer_q90": int(report.get("layer_count_q90", 0) or 0),
            "run_window_core_base_layer_max": int(report.get("layer_count_max", 0) or 0),
        }

    def _empty_run_window_core_report_fields(self, report: dict[str, object]) -> dict[str, object]:
        return {
            "run_window_core_applied": 0,
            "run_window_core_score": 0.0,
            "run_window_core_start": -1,
            "run_window_core_stop": -1,
            "run_window_core_trim_left": 0,
            "run_window_core_trim_right": 0,
            "run_window_core_candidate_windows": 0,
            "run_window_core_reanalyzed_windows": 0,
            **self._base_run_window_core_report_fields(report),
            "run_window_core_layer_max_preservation": 0.0,
            "run_window_core_layer_q90_preservation": 0.0,
            "run_window_core_outside_support_fraction": 0.0,
            "run_window_core_z_coverage_fraction": 0.0,
            "run_window_core_max_z_gap": 0,
            "run_window_core_centrality": 0.0,
            "run_window_core_count_layer_improvement": 0,
            "run_window_core_absent_improvement": 0.0,
            "run_window_core_drop_improvement": 0.0,
            "run_window_core_count_margin": 0.0,
        }

BarrelAnalyzer = StrandCountAnalyzer
