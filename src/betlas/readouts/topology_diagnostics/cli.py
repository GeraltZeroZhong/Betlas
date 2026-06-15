from __future__ import annotations

import argparse

from .config import cfg_get, load_config
from .core import MODE_COLUMNS, run_topology_diagnostics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="betlas readout topology-diagnostics",
        description=(
            "Compute topology ambiguity, continuous fold-organization scores, "
            "and mixed-topology flags from a Betlas feature table."
        ),
    )
    parser.add_argument("--config", default=None, help="Topology diagnostics YAML config.")
    parser.add_argument("--features", default=None, help="Input Betlas feature CSV.")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional OOF prediction CSV. If absent, grammar-rule probabilities are used.",
    )
    parser.add_argument("--no-predictions", action="store_true", help="Disable prediction CSV loading.")
    parser.add_argument(
        "--model",
        default=None,
        help="Prediction model to use when --predictions contains multiple models. Use an empty value to skip filtering.",
    )
    parser.add_argument("--out", default=None, help="Output readout CSV.")
    parser.add_argument("--manifest", default=None, help="Output run manifest JSON.")
    parser.add_argument("--no-manifest", action="store_true", help="Do not write a run manifest.")
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
    config = load_config(args.config)
    model = args.model if args.model is not None and str(args.model).strip() else None
    predictions = None if args.no_predictions else (args.predictions or cfg_get(config, "io.predictions_csv", None))
    result = run_topology_diagnostics(
        features_csv=args.features,
        predictions_csv=predictions,
        model=model,
        out_csv=args.out,
        mode=args.mode,
        k_neighbors=args.neighbors,
        include_features=args.include_features,
        config_path=args.config,
        manifest_path=args.manifest,
        write_manifest=not args.no_manifest,
    )
    print(f"Wrote {len(result.diagnostics)} {args.mode} topology diagnostic rows to {result.output_csv}")
    if result.manifest_path is not None:
        print(f"Wrote topology diagnostic manifest to {result.manifest_path}")
