from __future__ import annotations

import argparse

from .core import MODE_COLUMNS, run_topology_diagnostics

_MODE_HELP = {
    "all": {
        "description": (
            "Compute topology ambiguity, continuous fold-organization scores, "
            "and mixed-topology flags from a Betlas feature table."
        ),
        "examples": (
            "Examples:\n"
            "  {prog} --features runs/betlas_features.csv --out runs/topology.csv\n"
            "  {prog} --mode ambiguity --features runs/betlas_features.csv --out runs/topology_ambiguity.csv\n"
            "  {prog} --features runs/betlas_features.csv --predictions runs/oof_predictions.csv --neighbors 15 --out runs/topology.csv"
        ),
        "output": "Output CSV: ambiguity, continuous topology, and mixed-topology readout columns selected by --mode.",
    },
    "ambiguity": {
        "description": "Compute topology ambiguity and boundary-region audit columns from a Betlas feature table.",
        "examples": (
            "Examples:\n"
            "  {prog} --features runs/betlas_features.csv --out runs/topology_ambiguity.csv\n"
            "  {prog} --features runs/betlas_features.csv --predictions runs/oof_predictions.csv --neighbors 15 --out runs/topology_ambiguity.csv"
        ),
        "output": "Output CSV: topology ambiguity, probability-source, rule-conflict, and neighbor-evidence columns.",
    },
    "continuous": {
        "description": "Compute continuous jelly-rollness, sandwichness, barrel-likeness, and overlap scores.",
        "examples": (
            "Examples:\n"
            "  {prog} --features runs/betlas_features.csv --out runs/fold_continuous_scores.csv\n"
            "  {prog} --features runs/betlas_features.csv --include-features --out runs/features_with_continuous_scores.csv"
        ),
        "output": "Output CSV: continuous fold-organization scores and basis JSON columns.",
    },
    "mixed": {
        "description": "Compute mixed-topology and manual boundary-audit priority columns from Betlas features.",
        "examples": (
            "Examples:\n"
            "  {prog} --features runs/betlas_features.csv --out runs/mixed_topology.csv\n"
            "  {prog} --features runs/betlas_features.csv --predictions runs/oof_predictions.csv --out runs/mixed_topology.csv"
        ),
        "output": "Output CSV: mixed-topology score, flag, type labels, secondary topology, and audit priority.",
    },
}


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "betlas readout topology-diagnostics",
    fixed_mode: str | None = None,
) -> None:
    mode_for_help = fixed_mode or "all"
    help_text = _MODE_HELP[mode_for_help]
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=help_text["description"],
        epilog=(
            f"{help_text['examples'].format(prog=prog)}\n\n"
            "Input feature CSV: canonical Betlas feature columns plus record identifiers.\n"
            f"{help_text['output']}\n"
            "Stdout is status text, not CSV; pass --out for the CSV path.\n"
            "Predictions: optional OOF prediction CSV; when absent, rule-derived softmax weights are used "
            "and marked uncalibrated."
        ),
    )
    parser.add_argument("--config", default=None, help="Topology diagnostics YAML config.")
    parser.add_argument("--features", default=None, help="Input Betlas feature CSV.")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional OOF prediction CSV. If absent, uncalibrated rule-softmax weights are used.",
    )
    parser.add_argument("--no-predictions", action="store_true", help="Disable prediction CSV loading.")
    parser.add_argument(
        "--model",
        default=None,
        help="Prediction model to use when --predictions contains multiple models. Use an empty value to skip filtering.",
    )
    parser.add_argument("--out", default=None, help="Output readout CSV. Stdout is status text, not CSV.")
    parser.add_argument("--manifest", default=None, help="Output run manifest JSON.")
    parser.add_argument("--no-manifest", action="store_true", help="Do not write a run manifest.")
    if fixed_mode is None:
        parser.add_argument(
            "--mode",
            choices=sorted(MODE_COLUMNS),
            default="all",
            help="Write all diagnostics or one readout subset.",
        )
    parser.add_argument("--neighbors", type=int, default=None, help="Nearest neighbors for boundary-region similarity.")
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Append readout columns to the original feature table instead of writing a compact readout table.",
    )
    args = parser.parse_args(argv)
    model = args.model if args.model is not None and str(args.model).strip() else None
    predictions = None if args.no_predictions else args.predictions
    mode = fixed_mode or args.mode
    result = run_topology_diagnostics(
        features_csv=args.features,
        predictions_csv=predictions,
        model=model,
        out_csv=args.out,
        mode=mode,
        k_neighbors=args.neighbors,
        include_features=args.include_features,
        config_path=args.config,
        manifest_path=args.manifest,
        write_manifest=not args.no_manifest,
        predictions_required=args.predictions is not None and not args.no_predictions,
    )
    print(f"Wrote {len(result.diagnostics)} {mode} topology diagnostic rows to {result.output_csv}")
    if result.manifest_path is not None:
        print(f"Wrote topology diagnostic manifest to {result.manifest_path}")
