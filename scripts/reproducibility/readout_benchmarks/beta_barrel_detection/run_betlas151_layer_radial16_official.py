#!/usr/bin/env python
"""Run Betlas 151 beta-barrel detection readouts on the Betlas-Beta benchmark."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betlas.assets import resolve_asset_path  # noqa: E402
from scripts.reproducibility.readouts.beta_barrel_detection.betlas_readout import (  # noqa: E402
    DEFAULT_BETLAS_BETA_ROOT,
    DEFAULT_ESMC_NPZ,
    DEFAULT_OUT_DIR,
    ReadoutPaths,
    display_path,
    run_official_detection_readout,
)

DEFAULT_ASSET_ID = "betlas-beta-barrel-detection-official-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Required local inputs without --asset-id/--download-assets:\n"
            "  --feature-columns feature_columns_151.csv\n"
            "  --cohort-csv benchmark_cohort.csv, --features-csv betlas_151_chain_features.csv\n"
            "  --layer-values-csv layer_radial16_feature_values.csv and --layer-manifest-csv layer_radial16_feature_manifest.csv\n"
            "  --esmc-npz esmc_mean_embeddings_aligned.npz for the ESM-C branch\n\n"
            "Outputs: official_summary.csv, official_fold_metrics.csv, official_per_record_predictions.csv, "
            "external_baseline_comparison.csv, metadata.json, summary.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help=(
            "Optional Betlas asset id used to resolve the fixed cohort, Betlas 151 "
            "features, LayerRadial16 tables, and ESM-C cache from the local asset cache."
        ),
    )
    parser.add_argument(
        "--asset-cache-dir",
        type=Path,
        default=None,
        help="Optional Betlas asset cache directory. Defaults to BETLAS_ASSET_DIR.",
    )
    parser.add_argument(
        "--download-assets",
        action="store_true",
        help=(
            "Download and verify selected asset files before running. Requires a released payload "
            "or a BETLAS_ASSET_BASE_URL-compatible mirror."
        ),
    )
    parser.add_argument("--betlas-beta-root", type=Path, default=DEFAULT_BETLAS_BETA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=None,
        help=(
            "Ordered Betlas 151 feature-column CSV. Required unless --asset-id resolves "
            "feature_columns_151.csv from the asset cache."
        ),
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=None,
        help="Optional benchmark cohort CSV. If omitted, the bundled manifest builder is used.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help="Optional precomputed Betlas 151 feature table aligned to --cohort-csv.",
    )
    parser.add_argument(
        "--chain-results-csv",
        type=Path,
        default=None,
        help="Optional detector chain-results CSV used to build LayerRadial16.",
    )
    parser.add_argument(
        "--layer-values-csv",
        type=Path,
        default=None,
        help="Optional precomputed LayerRadial16 feature-values CSV aligned to --cohort-csv.",
    )
    parser.add_argument(
        "--layer-manifest-csv",
        type=Path,
        default=None,
        help="Optional LayerRadial16 feature manifest for --layer-values-csv.",
    )
    parser.add_argument(
        "--esmc-npz",
        type=Path,
        default=None,
        help=(
            "Optional detection-benchmark ESM-C mean-embedding cache. "
            "Defaults to <out-dir>/esmc_mean_embeddings_aligned.npz."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def _main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    asset_id = args.asset_id or (DEFAULT_ASSET_ID if args.download_assets else None)
    asset_cache_dir = args.asset_cache_dir.expanduser().resolve() if args.asset_cache_dir else None
    feature_columns = args.feature_columns.expanduser().resolve() if args.feature_columns else None
    esmc_npz = args.esmc_npz.expanduser().resolve() if args.esmc_npz else out_dir / DEFAULT_ESMC_NPZ.name
    if asset_id:
        def asset_path(filename: str) -> Path:
            return resolve_asset_path(
                asset_id,
                filename,
                cache_dir=asset_cache_dir,
                download=bool(args.download_assets),
            ).resolve()

        if args.feature_columns is None:
            feature_columns = asset_path("feature_columns_151.csv")
        if args.cohort_csv is None:
            args.cohort_csv = asset_path("benchmark_cohort.csv")
        if args.features_csv is None:
            args.features_csv = asset_path("betlas_151_chain_features.csv")
        if args.layer_values_csv is None:
            args.layer_values_csv = asset_path("layer_radial16_feature_values.csv")
        if args.layer_manifest_csv is None:
            args.layer_manifest_csv = asset_path("layer_radial16_feature_manifest.csv")
        if args.esmc_npz is None:
            esmc_npz = asset_path("esmc_mean_embeddings_aligned.npz")
    paths = ReadoutPaths(
        betlas_beta_root=args.betlas_beta_root.expanduser().resolve(),
        out_dir=out_dir,
        feature_columns=feature_columns,
        esmc_npz=esmc_npz,
        cohort_csv=args.cohort_csv.expanduser().resolve() if args.cohort_csv else None,
        features_csv=args.features_csv.expanduser().resolve() if args.features_csv else None,
        chain_results_csv=args.chain_results_csv.expanduser().resolve() if args.chain_results_csv else None,
        layer_values_csv=args.layer_values_csv.expanduser().resolve() if args.layer_values_csv else None,
        layer_manifest_csv=args.layer_manifest_csv.expanduser().resolve() if args.layer_manifest_csv else None,
        asset_id=asset_id or "",
        asset_cache_dir=asset_cache_dir if asset_id else None,
    )
    metadata = run_official_detection_readout(
        paths=paths,
        workers=int(args.workers),
        force_extract=bool(args.force_extract),
        n_splits=int(args.splits),
        iterations=int(args.iterations),
        seed=int(args.seed),
        jobs=int(args.jobs),
        pca_dim=int(args.pca_dim),
    )
    print(f"Wrote Betlas beta-barrel detection readout to {display_path(paths.out_dir)}")
    print(
        "Rows: "
        f"{metadata['metric_rows']} metric / {metadata['cohort_rows']} total; "
        f"Betlas parse-ok {metadata['betlas_151_parse_ok']}/{metadata['cohort_rows']}"
    )
    if metadata.get("skipped"):
        for item in metadata["skipped"]:
            print(f"Skipped {item['display_label']}: {item['skip_reason']}")
    return 0


def main() -> int:
    try:
        return _main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
