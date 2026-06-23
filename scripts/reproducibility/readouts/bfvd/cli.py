from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct path execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from scripts.reproducibility.readouts.bfvd.core import run_bfvd_scan
else:
    from .core import run_bfvd_scan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reproducibility.readouts.bfvd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run a quality-aware Betlas grammar scan over BFVD PDB models and "
            "write a viral beta-fold topology audit table."
        ),
        epilog=(
            "Examples:\n"
            "  python -m scripts.reproducibility.readouts.bfvd --bfvd-dir path/to/BFVD --metadata bfvd_metadata.tsv --out-dir runs/bfvd_scan\n"
            "  PYTHONPATH=.:src python scripts/reproducibility/readouts/bfvd/cli.py --bfvd-dir path/to/BFVD --limit 100 --workers 4\n\n"
            "Outputs: feature rows, topology audit rows, scan funnel rows, and an optional run manifest.\n"
            "This is a repository reproducibility workflow, not a stable Betlas Python API."
        ),
    )
    parser.add_argument("--bfvd-dir", default="data/external/betlas_beta/BFVD")
    parser.add_argument("--metadata", default=None, help="BFVD bfvd_metadata.tsv path.")
    parser.add_argument(
        "--taxonomy",
        default=None,
        help="Optional BFVD bfvd_taxid_rank_scientificname_lineage.tsv path.",
    )
    parser.add_argument(
        "--foldseek-context",
        default=None,
        help="Optional nearest-known Foldseek context table to join by BFVD model id.",
    )
    parser.add_argument("--out-dir", default="runs/readouts/bfvd_viral_beta_fold_scan")
    parser.add_argument("--limit", type=int, default=None, help="Limit models after metadata QC.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Sample a reproducible subset after metadata QC instead of taking the first rows.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-pending-tasks",
        type=int,
        default=None,
        help="Maximum submitted-but-uncollected worker tasks; defaults to 4 * workers.",
    )
    parser.add_argument("--dssp-bin", default="mkdssp")
    parser.add_argument(
        "--dssp-cache-dir",
        default=None,
        help="Optional cache directory containing or receiving classic DSSP files.",
    )
    parser.add_argument(
        "--write-dssp-cache",
        action="store_true",
        help="Write missing DSSP files into --dssp-cache-dir while scanning.",
    )
    parser.add_argument("--min-avg-plddt", type=float, default=70.0)
    parser.add_argument("--min-ptm", type=float, default=0.45)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=1500)
    parser.add_argument("--max-domain-like-length", type=int, default=700)
    parser.add_argument("--min-beta-strands", type=int, default=4)
    parser.add_argument("--min-beta-content", type=float, default=0.18)
    parser.add_argument("--max-pca-elongation", type=float, default=5.0)
    parser.add_argument("--max-low-confidence-tail-fraction", type=float, default=0.25)
    parser.add_argument(
        "--allow-split",
        action="store_true",
        help="Keep BFVD entries marked as split sequence fragments in metadata QC.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write a run manifest JSON.",
    )
    args = parser.parse_args(argv)

    result = run_bfvd_scan(
        bfvd_dir=args.bfvd_dir,
        metadata_tsv=args.metadata,
        taxonomy_tsv=args.taxonomy,
        foldseek_context=args.foldseek_context,
        out_dir=args.out_dir,
        limit=args.limit,
        sample_seed=args.sample_seed,
        workers=args.workers,
        max_pending_tasks=args.max_pending_tasks,
        dssp_bin=args.dssp_bin,
        dssp_cache_dir=args.dssp_cache_dir,
        write_dssp_cache=args.write_dssp_cache,
        min_avg_plddt=args.min_avg_plddt,
        min_ptm=args.min_ptm,
        min_length=args.min_length,
        max_length=args.max_length,
        max_domain_like_length=args.max_domain_like_length,
        min_beta_strands=args.min_beta_strands,
        min_beta_content=args.min_beta_content,
        max_pca_elongation=args.max_pca_elongation,
        max_low_confidence_tail_fraction=args.max_low_confidence_tail_fraction,
        allow_split=args.allow_split,
        write_manifest=not args.no_manifest,
    )
    print(f"Wrote BFVD feature rows to {result.features_csv}")
    print(f"Wrote BFVD topology audit table to {result.audit_csv}")
    print(f"Wrote BFVD scan funnel to {result.funnel_csv}")
    if result.manifest_path is not None:
        print(f"Wrote BFVD scan manifest to {result.manifest_path}")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover - direct path execution
    main()
