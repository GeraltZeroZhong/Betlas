from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .constants import (
    DEFAULT_CATH_DIR,
    DEFAULT_FEATURES_CSV,
    DEFAULT_LABELS_CSV,
    DEFAULT_MMCIF_DIR,
    DEFAULT_RUN_DIR,
)
from .external_baselines import run_external_baselines
from .features.extract import assert_no_diagnostic_label_leakage, extract_feature_row
from .io.cath import build_cath_dataset, domain_from_row
from .io.rcsb import download_mmcifs, mmcif_path_for
from .io.snapshots import enrich_feature_table_with_mmcif_provenance, write_source_snapshot
from .ml.ablations import run_ablation_suite
from .ml.benchmark import run_grouped_benchmark
from .provenance import build_data_manifest, build_run_manifest, file_state, write_json
from .publication import run_publication_evidence
from .readouts import list_readouts, run_readout
from .readouts.beta_barrel_staves.publication import write_publication_gold_dataset
from .readouts.beta_barrel_staves.schema import write_beta_barrel_staves_validation_report
from .science_report import repair_feature_table_metadata, run_science_report


def _extract_one(row: dict[str, Any], mmcif_dir: str) -> dict[str, Any]:
    domain = domain_from_row(row)
    mmcif_path = mmcif_path_for(domain.pdb_id, Path(mmcif_dir))
    feature_row = extract_feature_row(domain, mmcif_path)
    mmcif_state = file_state(mmcif_path)
    feature_row["source_mmcif_path"] = mmcif_state["path"]
    feature_row["source_mmcif_sha256"] = mmcif_state["sha256"]
    feature_row["source_mmcif_size"] = mmcif_state["size"]
    feature_row["source_mmcif_exists"] = int(bool(mmcif_state["exists"]))
    assert_no_diagnostic_label_leakage(feature_row)
    return feature_row


def build_dataset_command(args: argparse.Namespace) -> None:
    df = build_cath_dataset(
        cath_dir=Path(args.cath_dir),
        target_per_class=args.target_per_class,
        seed=args.seed,
        include_putative=args.include_putative,
        all_eligible=args.all_eligible,
        max_per_pdb=args.max_per_pdb,
        initial_max_per_s35=args.initial_max_per_s35,
        max_s35_cap=args.max_s35_cap,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    cath_inputs = {
        path.name: file_state(path)
        for path in sorted(Path(args.cath_dir).glob("*.gz"))
        if path.is_file()
    }
    write_json(
        out.with_suffix(f"{out.suffix}.manifest.json"),
        build_run_manifest(
            command="betlas build-dataset",
            parameters={
                "cath_dir": args.cath_dir,
                "target_per_class": args.target_per_class,
                "seed": args.seed,
                "include_putative": args.include_putative,
                "all_eligible": args.all_eligible,
                "max_per_pdb": args.max_per_pdb,
                "initial_max_per_s35": args.initial_max_per_s35,
                "max_s35_cap": args.max_s35_cap,
            },
            outputs={"labels_csv": out},
            metrics={"rows": int(len(df))},
            extra={"cath_inputs": cath_inputs},
        ),
    )
    counts = df["fold_label_final"].value_counts().sort_index()
    print(f"Wrote {len(df)} labels to {out}")
    print(counts.to_string())


def extract_features_command(args: argparse.Namespace) -> None:
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    mmcif_dir = Path(args.mmcif_dir)
    pdb_ids = labels["pdb_id"].astype(str).str.lower().tolist()
    download_results = download_mmcifs(
        pdb_ids,
        mmcif_dir,
        workers=args.download_workers,
        force=args.force_download,
    )

    rows = labels.to_dict(orient="records")
    feature_rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for row in tqdm(rows, desc="Extracting features"):
            feature_rows.append(_extract_one(row, str(mmcif_dir)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_extract_one, row, str(mmcif_dir)) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                feature_rows.append(future.result())

    feature_df = pd.DataFrame(feature_rows)
    label_order = {str(row["record_id"]): index for index, row in enumerate(rows)}
    feature_df["_order"] = feature_df["record_id"].astype(str).map(label_order)
    feature_df = feature_df.sort_values("_order").drop(columns=["_order"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out, index=False)
    ok = pd.to_numeric(feature_df.get("cz_parse_ok", 0), errors="coerce").fillna(0).astype(int)
    mmcif_inputs = {
        pdb_id: file_state(mmcif_path_for(pdb_id, mmcif_dir))
        for pdb_id in sorted({str(pdb_id).strip().lower() for pdb_id in pdb_ids if str(pdb_id).strip()})
    }
    write_json(
        out.with_suffix(f"{out.suffix}.manifest.json"),
        build_run_manifest(
            command="betlas extract-features",
            parameters={
                "labels": args.labels,
                "mmcif_dir": args.mmcif_dir,
                "workers": args.workers,
                "download_workers": args.download_workers,
                "force_download": args.force_download,
            },
            inputs={"labels_csv": args.labels},
            outputs={"features_csv": out},
            metrics={
                "rows": int(len(feature_df)),
                "parse_ok_rows": int(ok.sum()),
                "unique_pdb_ids": int(len(set(pdb_ids))),
            },
            extra={"download_results": download_results, "mmcif_inputs": mmcif_inputs},
        ),
    )
    print(f"Wrote {len(feature_df)} feature rows to {out}")
    print(f"Parse-ok rows: {int(ok.sum())}/{len(feature_df)}")
    print(feature_df["fold_label_final"].value_counts().sort_index().to_string())


def benchmark_command(args: argparse.Namespace) -> None:
    result = run_grouped_benchmark(
        Path(args.features),
        Path(args.out_dir),
        n_splits=args.splits,
        random_state=args.seed,
        config_path=args.config,
    )
    print(f"Wrote benchmark outputs to {args.out_dir}")
    print(result.metrics_summary.to_string(index=False))


def ablate_command(args: argparse.Namespace) -> None:
    result = run_ablation_suite(
        Path(args.features),
        Path(args.out_dir),
        n_splits=args.splits,
        random_state=args.seed,
        max_group_combo_size=args.max_group_combo_size,
        config_path=args.config,
    )
    print(f"Wrote ablation outputs to {args.out_dir}")
    baseline = result.all_results[result.all_results["ablation_type"] == "baseline"]
    print(baseline[["name", "macro_f1", "boundary_macro_f1", "sandwich_jelly_macro_f1"]].to_string(index=False))
    drop_groups = result.all_results[result.all_results["ablation_type"] == "drop_group_single"]
    cols = ["name", "delta_macro_f1", "delta_boundary_macro_f1", "delta_sandwich_jelly_macro_f1"]
    print(drop_groups.sort_values("delta_boundary_macro_f1").head(12)[cols].to_string(index=False))


def list_readouts_command(_args: argparse.Namespace) -> None:
    for spec in list_readouts():
        print(f"{spec.name}\t{spec.summary}")


def _parse_named_paths(items: list[str] | None) -> dict[str, str]:
    paths = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep:
            name = Path(item).name or "artifact"
            value = item
        paths[name] = value
    return paths


def data_manifest_command(args: argparse.Namespace) -> None:
    roots = _parse_named_paths(args.root or ["processed=data/processed", "readouts=data/readouts"])
    artifacts = _parse_named_paths(args.file)
    manifest = build_data_manifest(
        name=args.name,
        roots=roots,
        artifacts=artifacts,
        patterns=args.pattern,
        max_file_bytes=args.max_file_bytes,
    )
    write_json(args.out, manifest)
    file_count = sum(len(root["files"]) for root in manifest["roots"].values())
    artifact_count = len(manifest["artifacts"])
    print(f"Wrote data manifest for {file_count} scanned files and {artifact_count} named artifacts to {args.out}")


def source_snapshot_command(args: argparse.Namespace) -> None:
    path = write_source_snapshot(
        args.out,
        cath_dir=args.cath_dir,
        mmcif_dir=args.mmcif_dir,
        labels_csv=args.labels,
    )
    print(f"Wrote source snapshot to {path}")


def enrich_feature_provenance_command(args: argparse.Namespace) -> None:
    df = enrich_feature_table_with_mmcif_provenance(
        args.features,
        args.out,
        mmcif_dir=args.mmcif_dir,
    )
    missing = int((pd.to_numeric(df["source_mmcif_exists"], errors="coerce").fillna(0) == 0).sum())
    write_json(
        Path(args.out).with_suffix(f"{Path(args.out).suffix}.manifest.json"),
        build_run_manifest(
            command="betlas enrich-feature-provenance",
            parameters={
                "features": args.features,
                "mmcif_dir": args.mmcif_dir,
                "out": args.out,
            },
            inputs={"features_csv": args.features},
            outputs={"features_csv": args.out},
            metrics={"rows": int(len(df)), "missing_mmcif_rows": missing},
        ),
    )
    print(f"Wrote {len(df)} feature rows with source hashes to {args.out}")
    print(f"Rows with missing mmCIF: {missing}")


def validate_staves_data_command(args: argparse.Namespace) -> None:
    path = write_beta_barrel_staves_validation_report(
        args.csv,
        args.out,
        schema_path=args.schema,
        repo_root=args.repo_root,
    )
    report = json.loads(Path(path).read_text(encoding="utf-8"))["extra"]["validation"]
    issue_count = int(report.get("issue_count", 0))
    print(f"Wrote beta-barrel-staves validation report to {path}")
    print(f"Issues: {issue_count}")
    if args.strict and issue_count:
        raise SystemExit(1)


def curate_staves_publication_set_command(args: argparse.Namespace) -> None:
    output = write_publication_gold_dataset(
        args.source,
        args.out,
        manifest_path=args.manifest,
    )
    print(f"Wrote publication-eligible beta-barrel-staves gold dataset to {output}")


def repair_feature_table_command(args: argparse.Namespace) -> None:
    repaired = repair_feature_table_metadata(args.features, args.labels, args.out)
    write_json(
        Path(args.out).with_suffix(f"{Path(args.out).suffix}.manifest.json"),
        build_run_manifest(
            command="betlas repair-feature-table",
            parameters={"features": args.features, "labels": args.labels, "out": args.out},
            inputs={"features_csv": args.features, "labels_csv": args.labels},
            outputs={"features_csv": args.out},
            metrics={"rows": int(len(repaired))},
        ),
    )
    print(f"Wrote repaired feature table to {args.out}")


def science_report_command(args: argparse.Namespace) -> None:
    result = run_science_report(
        features_csv=args.features,
        labels_csv=args.labels,
        topology_csv=args.topology,
        benchmark_dir=args.benchmark_dir,
        source_snapshot=args.source_snapshot,
        staves_readouts_csv=args.staves_readouts_full,
        out_dir=args.out_dir,
    )
    print(f"Wrote scientific audit report to {result.out_dir}")
    print(f"Wrote manifest to {result.manifest_path}")


def publication_evidence_command(args: argparse.Namespace) -> None:
    result = run_publication_evidence(args.config)
    print(f"Wrote publication evidence outputs to {result.out_dir}")
    print(f"Wrote manifest to {result.manifest_path}")
    print("Subset files:")
    for name, path in result.subset_paths.items():
        print(f"  {name}: {path}")


def external_baselines_command(args: argparse.Namespace) -> None:
    result = run_external_baselines(args.config)
    print(f"Wrote external baseline outputs to {result.out_dir}")
    print(f"Wrote external baseline figures to {result.figure_dir}")
    print(f"Wrote manifest to {result.manifest_path}")


def readout_command(args: argparse.Namespace) -> None:
    readout_args = list(args.readout_args)
    if getattr(args, "readout_help", False):
        readout_args.insert(0, "--help")
    run_readout(args.readout_name, readout_args)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if (
        len(raw_argv) >= 2
        and raw_argv[0] == "readout"
        and raw_argv[1] != "list"
        and not raw_argv[1].startswith("-")
    ):
        run_readout(raw_argv[1], raw_argv[2:])
        return

    parser = argparse.ArgumentParser(prog="betlas")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset", help="Build a CATH-derived seven-class label table.")
    build.add_argument("--cath-dir", default=str(DEFAULT_CATH_DIR))
    build.add_argument("--target-per-class", type=int, default=220)
    build.add_argument("--seed", type=int, default=13)
    build.add_argument("--include-putative", action="store_true")
    build.add_argument(
        "--all-eligible",
        action="store_true",
        default=True,
        help="Emit every eligible CATH-derived row instead of balanced sampling.",
    )
    build.add_argument(
        "--sample-balanced",
        action="store_false",
        dest="all_eligible",
        help="Use the balanced target-per-class sampler instead of the full eligible table.",
    )
    build.add_argument("--max-per-pdb", type=int, default=4)
    build.add_argument("--initial-max-per-s35", type=int, default=1)
    build.add_argument("--max-s35-cap", type=int, default=64)
    build.add_argument("--out", default=str(DEFAULT_LABELS_CSV))
    build.set_defaults(func=build_dataset_command)

    extract = sub.add_parser("extract-features", help="Download mmCIF files and extract Betlas features.")
    extract.add_argument("--labels", default=str(DEFAULT_LABELS_CSV))
    extract.add_argument("--mmcif-dir", default=str(DEFAULT_MMCIF_DIR))
    extract.add_argument("--out", default=str(DEFAULT_FEATURES_CSV))
    extract.add_argument("--workers", type=int, default=1)
    extract.add_argument("--download-workers", type=int, default=8)
    extract.add_argument("--force-download", action="store_true")
    extract.set_defaults(func=extract_features_command)

    bench = sub.add_parser("benchmark", help="Run grouped benchmark with transparent and learned models.")
    bench.add_argument("--features", default=str(DEFAULT_FEATURES_CSV))
    bench.add_argument("--out-dir", default=str(DEFAULT_RUN_DIR))
    bench.add_argument("--splits", type=int, default=5)
    bench.add_argument("--seed", type=int, default=13)
    bench.add_argument("--config", default=None, help="Benchmark YAML config.")
    bench.set_defaults(func=benchmark_command)

    ablate = sub.add_parser("ablate", help="Run grouped feature ablations for geometry grammars.")
    ablate.add_argument("--features", default=str(DEFAULT_FEATURES_CSV))
    ablate.add_argument("--out-dir", default="runs/betlas_full_ablations")
    ablate.add_argument("--splits", type=int, default=5)
    ablate.add_argument("--seed", type=int, default=13)
    ablate.add_argument("--max-group-combo-size", type=int, default=3)
    ablate.add_argument("--config", default=None, help="Ablation YAML config.")
    ablate.set_defaults(func=ablate_command)

    manifest = sub.add_parser("data-manifest", help="Write a manifest for input data artifacts.")
    manifest.add_argument("--name", default="betlas-data")
    manifest.add_argument(
        "--root",
        action="append",
        default=None,
        help="Root to scan, optionally name=path. Repeatable.",
    )
    manifest.add_argument(
        "--file",
        action="append",
        default=None,
        help="Named artifact to fingerprint, optionally name=path. Repeatable.",
    )
    manifest.add_argument("--pattern", action="append", default=["**/*"])
    manifest.add_argument("--max-file-bytes", type=int, default=None)
    manifest.add_argument("--out", default="data/manifest.json")
    manifest.set_defaults(func=data_manifest_command)

    snapshot = sub.add_parser("source-snapshot", help="Write CATH/RCSB source snapshot manifest.")
    snapshot.add_argument("--cath-dir", default=str(DEFAULT_CATH_DIR))
    snapshot.add_argument("--mmcif-dir", default=str(DEFAULT_MMCIF_DIR))
    snapshot.add_argument("--labels", default=str(DEFAULT_LABELS_CSV))
    snapshot.add_argument("--out", default="data/source_snapshot.json")
    snapshot.set_defaults(func=source_snapshot_command)

    enrich = sub.add_parser("enrich-feature-provenance", help="Add row-level mmCIF hashes to a feature table.")
    enrich.add_argument("--features", default=str(DEFAULT_FEATURES_CSV))
    enrich.add_argument("--mmcif-dir", default=str(DEFAULT_MMCIF_DIR))
    enrich.add_argument("--out", default=str(DEFAULT_FEATURES_CSV))
    enrich.set_defaults(func=enrich_feature_provenance_command)

    validate_staves = sub.add_parser(
        "validate-staves-data",
        help="Validate beta-barrel-staves processed data schema, hashes, and manifest paths.",
    )
    validate_staves.add_argument(
        "--csv",
        default="data/readouts/beta_barrel_staves/processed/beta_barrel_strand_dataset.csv",
    )
    validate_staves.add_argument(
        "--schema",
        default="data/readouts/beta_barrel_staves/schemas/beta_barrel_strand_dataset.schema.json",
    )
    validate_staves.add_argument("--repo-root", default=".")
    validate_staves.add_argument("--out", default="data/readouts/beta_barrel_staves/validation_report.json")
    validate_staves.add_argument("--strict", action="store_true")
    validate_staves.set_defaults(func=validate_staves_data_command)

    curate_staves = sub.add_parser(
        "curate-staves-publication-set",
        help="Write publication-eligible beta-barrel-staves gold/pass rows from a source table.",
    )
    curate_staves.add_argument(
        "--source",
        default="data/readouts/beta_barrel_staves/processed/beta_barrel_strand_dataset.csv",
    )
    curate_staves.add_argument(
        "--out",
        default="data/readouts/beta_barrel_staves/processed/beta_barrel_publication_gold.csv",
    )
    curate_staves.add_argument(
        "--manifest",
        default="data/readouts/beta_barrel_staves/processed/beta_barrel_publication_gold.manifest.json",
    )
    curate_staves.set_defaults(func=curate_staves_publication_set_command)

    repair_features = sub.add_parser(
        "repair-feature-table",
        help="Repair feature-table metadata from labels and clamp tiny numerical artifacts.",
    )
    repair_features.add_argument("--features", default=str(DEFAULT_FEATURES_CSV))
    repair_features.add_argument("--labels", default=str(DEFAULT_LABELS_CSV))
    repair_features.add_argument("--out", default=str(DEFAULT_FEATURES_CSV))
    repair_features.set_defaults(func=repair_feature_table_command)

    science = sub.add_parser("science-report", help="Write publication-oriented scientific audit tables.")
    science.add_argument("--features", default=str(DEFAULT_FEATURES_CSV))
    science.add_argument("--labels", default=str(DEFAULT_LABELS_CSV))
    science.add_argument("--topology", default="runs/readouts/topology_diagnostics_full/topology_diagnostics.csv")
    science.add_argument("--benchmark-dir", default=str(DEFAULT_RUN_DIR))
    science.add_argument("--source-snapshot", default="data/source_snapshot.json")
    science.add_argument(
        "--staves-readouts-full",
        default="data/readouts/beta_barrel_staves/processed/beta_barrel_readouts_full_508.csv",
    )
    science.add_argument("--out-dir", default="runs/science_report_full")
    science.set_defaults(func=science_report_command)

    evidence = sub.add_parser(
        "publication-evidence",
        help="Generate non-redundant sensitivity, confidence intervals, and case-study tables.",
    )
    evidence.add_argument("--config", default="configs/publication/evidence_full.yaml")
    evidence.set_defaults(func=publication_evidence_command)

    external = sub.add_parser(
        "external-baselines",
        help="Run Foldseek and ESM-C external baseline context analyses.",
    )
    external.add_argument("--config", default="configs/publication/external_baselines_full.yaml")
    external.set_defaults(func=external_baselines_command)

    readout = sub.add_parser("readout", help="Run or inspect Betlas secondary readouts.")
    readout_sub = readout.add_subparsers(dest="readout_command", required=True)
    readout_list = readout_sub.add_parser("list", help="List available readouts.")
    readout_list.set_defaults(func=list_readouts_command)

    for spec in list_readouts():
        readout_parser = readout_sub.add_parser(
            spec.name,
            help=spec.summary,
            add_help=False,
        )
        readout_parser.add_argument("-h", "--help", action="store_true", dest="readout_help")
        readout_parser.add_argument("readout_args", nargs=argparse.REMAINDER)
        readout_parser.set_defaults(func=readout_command, readout_name=spec.name)

    args = parser.parse_args(raw_argv)
    args.func(args)


if __name__ == "__main__":
    main()
