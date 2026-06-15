from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from betlas.workflows import (  # noqa: E402
    FullPipelineConfig,
    load_full_pipeline_config,
    run_full_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Betlas dataset, feature, and benchmark workflow.")
    parser.add_argument("--config", default=None, help="YAML config for FullPipelineConfig values.")
    parser.add_argument("--cath-dir", default=FullPipelineConfig.cath_dir)
    parser.add_argument("--mmcif-dir", default=FullPipelineConfig.mmcif_dir)
    parser.add_argument("--target-per-class", type=int, default=FullPipelineConfig.target_per_class)
    parser.add_argument("--include-putative", action="store_true")
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--max-per-pdb", type=int, default=FullPipelineConfig.max_per_pdb)
    parser.add_argument("--initial-max-per-s35", type=int, default=FullPipelineConfig.initial_max_per_s35)
    parser.add_argument("--max-s35-cap", type=int, default=FullPipelineConfig.max_s35_cap)
    parser.add_argument("--workers", type=int, default=FullPipelineConfig.workers)
    parser.add_argument("--download-workers", type=int, default=FullPipelineConfig.download_workers)
    parser.add_argument("--seed", type=int, default=FullPipelineConfig.seed)
    parser.add_argument("--splits", type=int, default=FullPipelineConfig.splits)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--benchmark-config", default=FullPipelineConfig.benchmark_config)
    parser.add_argument("--out-dir", default=FullPipelineConfig.out_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_full_pipeline_config(args.config)
    result = run_full_pipeline(
        FullPipelineConfig(
            cath_dir=args.cath_dir if args.cath_dir != FullPipelineConfig.cath_dir else base.cath_dir,
            mmcif_dir=args.mmcif_dir if args.mmcif_dir != FullPipelineConfig.mmcif_dir else base.mmcif_dir,
            target_per_class=args.target_per_class
            if args.target_per_class != FullPipelineConfig.target_per_class
            else base.target_per_class,
            include_putative=args.include_putative or base.include_putative,
            all_eligible=args.all_eligible or base.all_eligible,
            max_per_pdb=args.max_per_pdb if args.max_per_pdb != FullPipelineConfig.max_per_pdb else base.max_per_pdb,
            initial_max_per_s35=args.initial_max_per_s35
            if args.initial_max_per_s35 != FullPipelineConfig.initial_max_per_s35
            else base.initial_max_per_s35,
            max_s35_cap=args.max_s35_cap if args.max_s35_cap != FullPipelineConfig.max_s35_cap else base.max_s35_cap,
            workers=args.workers if args.workers != FullPipelineConfig.workers else base.workers,
            download_workers=args.download_workers
            if args.download_workers != FullPipelineConfig.download_workers
            else base.download_workers,
            seed=args.seed if args.seed != FullPipelineConfig.seed else base.seed,
            splits=args.splits if args.splits != FullPipelineConfig.splits else base.splits,
            force_download=args.force_download or base.force_download,
            benchmark_config=args.benchmark_config
            if args.benchmark_config != FullPipelineConfig.benchmark_config
            else base.benchmark_config,
            out_dir=args.out_dir if args.out_dir != FullPipelineConfig.out_dir else base.out_dir,
        )
    )
    print(f"Wrote labels to {result.labels_csv}")
    print(f"Wrote features to {result.features_csv}")
    print(f"Wrote benchmark outputs to {result.benchmark_dir}")
    print(f"Wrote full-pipeline manifest to {result.manifest_path}")


if __name__ == "__main__":
    main()
