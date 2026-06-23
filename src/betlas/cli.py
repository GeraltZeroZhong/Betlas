from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .assets import (
    DEFAULT_ASSET_BASE_URL,
    AssetError,
    describe_asset,
    download_asset,
    list_assets,
    resolve_asset_path,
    resolve_esmc_weights,
    verify_asset,
)
from .constants import (
    DEFAULT_CATH_DIR,
    DEFAULT_FEATURES_CSV,
    DEFAULT_LABELS_CSV,
    DEFAULT_MMCIF_DIR,
    DEFAULT_RUN_DIR,
    FOLD_LABELS,
)
from .examples import copy_example, list_examples
from .features.extract import (
    assert_no_diagnostic_label_leakage,
    extract_feature_row,
    extract_structure_features,
)
from .features.rules import missing_or_invalid_rule_inputs
from .grammars import explain_fold_grammar, get_grammar, list_grammars
from .io.cath import build_cath_dataset, domain_from_row
from .io.mmcif import inspect_mmcif_chains
from .io.rcsb import download_mmcifs, mmcif_path_for
from .provenance import build_run_manifest, file_state, write_json
from .readouts import list_readouts, run_readout
from .schema import normalize_feature_columns
from .slicing import SliceConfig, slice_mmcif, summarize_slices
from .specs import list_feature_specs


def _require_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value).expanduser()
    if not str(path_value).strip():
        raise FileNotFoundError(f"{label} path is required")
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")
    return path


def _structure_failed_row(args: argparse.Namespace, structure_state: dict[str, Any], error: Exception) -> dict[str, Any]:
    structure_path = Path(args.structure)
    stem = structure_path.name.split(".", 1)[0].lower()
    chain_id = str(args.chain or "")
    return {
        "record_id": args.record_id or f"{stem}_{chain_id}",
        "pdb_id": (args.pdb_id or stem).lower(),
        "chain_id": chain_id,
        "domain_id": args.domain_id or f"{stem}_{chain_id}",
        "residue_ranges": args.residue_ranges or "",
        "source_mmcif_path": structure_state["path"],
        "source_mmcif_sha256": structure_state["sha256"],
        "source_mmcif_size": structure_state["size"],
        "source_mmcif_exists": int(bool(structure_state["exists"])),
        "betlas_parse_ok": 0,
        "betlas_warnings": "",
        "betlas_error": f"{type(error).__name__}: {error}",
    }


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
    if args.structure:
        if not args.chain:
            raise ValueError("--chain is required when --structure is used")
        _require_file(args.structure, label="structure")
        structure_state = file_state(Path(args.structure))
        try:
            feature_row = extract_structure_features(
                args.structure,
                chain_id=args.chain,
                residue_ranges=args.residue_ranges,
                record_id=args.record_id,
                domain_id=args.domain_id,
                pdb_id=args.pdb_id,
            )
        except Exception as exc:
            if not args.write_failed_row:
                raise
            feature_row = _structure_failed_row(args, structure_state, exc)
        feature_row["source_mmcif_path"] = structure_state["path"]
        feature_row["source_mmcif_sha256"] = structure_state["sha256"]
        feature_row["source_mmcif_size"] = structure_state["size"]
        feature_row["source_mmcif_exists"] = int(bool(structure_state["exists"]))
        assert_no_diagnostic_label_leakage(feature_row)
        feature_df = pd.DataFrame([feature_row])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(out, index=False)
        write_json(
            out.with_suffix(f"{out.suffix}.manifest.json"),
            build_run_manifest(
                command="betlas extract-features",
                parameters={
                    "structure": args.structure,
                    "chain": args.chain,
                    "residue_ranges": args.residue_ranges,
                    "record_id": args.record_id,
                    "domain_id": args.domain_id,
                    "pdb_id": args.pdb_id,
                    "write_failed_row": bool(args.write_failed_row),
                },
                inputs={"structure": args.structure},
                outputs={"features_csv": out},
                metrics={
                    "rows": int(len(feature_df)),
                    "parse_ok_rows": int(
                        pd.to_numeric(feature_df.get("betlas_parse_ok", 0), errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .sum()
                    ),
                },
                extra={"structure_file": structure_state},
            ),
        )
        print(f"Wrote {len(feature_df)} feature row to {out}")
        return

    labels_path = _require_file(args.labels, label="labels CSV")
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
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
    ok = pd.to_numeric(feature_df.get("betlas_parse_ok", 0), errors="coerce").fillna(0).astype(int)
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
    from .ml.benchmark import run_grouped_benchmark

    _require_file(args.features, label="feature CSV")
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
    from .ml.ablations import run_ablation_suite

    _require_file(args.features, label="feature CSV")
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
    print("name\tinput\toutput\trequires\tsurface")
    for spec in list_readouts():
        print(
            "\t".join(
                [
                    spec.name,
                    spec.input_protocol,
                    spec.output_protocol,
                    spec.requires or "none",
                    spec.surface,
                ]
            )
        )


def grammar_list_command(_args: argparse.Namespace) -> None:
    for spec in list_grammars():
        print(f"{spec.name}\t{spec.summary}")


def grammar_describe_command(args: argparse.Namespace) -> None:
    spec = get_grammar(args.name)
    if args.format == "json":
        print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        return
    print(f"Name: {spec.name}")
    print(f"Summary: {spec.summary}")
    print(f"Mathematical implementation: {spec.math_summary}")
    print("Inputs:")
    for item in spec.inputs:
        print(f"  - {item}")
    print("Outputs:")
    for item in spec.outputs:
        print(f"  - {item}")
    if spec.columns:
        print("Columns:")
        for column in spec.columns:
            print(f"  - {column}")
    if spec.column_prefixes:
        print("Column prefixes:")
        for prefix in spec.column_prefixes:
            print(f"  - {prefix}")
    resolved_columns = [feature.name for feature in list_feature_specs() if spec.matches_column(feature.name)]
    if resolved_columns:
        print(f"Resolved columns ({len(resolved_columns)}):")
        for column in resolved_columns:
            print(f"  - {column}")
    if spec.implementation_notes:
        print("Implementation notes:")
        for note in spec.implementation_notes:
            print(f"  - {note}")


def grammar_score_command(args: argparse.Namespace) -> None:
    features_path = _require_file(args.features, label="feature CSV")
    raw_features = pd.read_csv(features_path, dtype=str, keep_default_na=False)
    compatibility_columns = [str(column) for column in raw_features.columns if str(column).startswith("cz_")]
    if args.strict and compatibility_columns:
        raise ValueError(
            "grammar score strict validation requires canonical betlas_* columns; "
            f"found compatibility columns: {', '.join(compatibility_columns[:8])}. "
            "Use --no-strict only for compatibility scoring of older feature tables."
        )
    features = normalize_feature_columns(raw_features)
    strict_problems: list[str] = []
    parse_failures: list[str] = []
    if "betlas_parse_ok" in features.columns:
        parse_ok = pd.to_numeric(features["betlas_parse_ok"], errors="coerce").fillna(0).astype(int)
    elif args.strict:
        raise ValueError(
            "grammar score strict validation requires betlas_parse_ok=1 in the feature CSV. "
            "Use --no-strict only for compatibility scoring of older feature tables."
        )
    else:
        parse_ok = pd.Series([1] * len(features), index=features.index, dtype=int)
    if not args.allow_parse_fail and "betlas_parse_ok" in features.columns:
        for index, row in enumerate(features.to_dict(orient="records"), start=1):
            if int(parse_ok.iloc[index - 1]) != 1:
                row_id = row.get("record_id", f"row {index}")
                warning = row.get("betlas_warnings", "")
                error = row.get("betlas_error", "")
                detail = str(error or warning or "betlas_parse_ok is not 1")
                parse_failures.append(f"{row_id}: {detail}")
        if parse_failures:
            examples = "; ".join(parse_failures[:5])
            more = f"; plus {len(parse_failures) - 5} more row(s)" if len(parse_failures) > 5 else ""
            raise ValueError(
                "grammar score refused parse-failed feature rows: "
                f"{examples}{more}. Use --allow-parse-fail only to write status-only failed rows."
            )
    if args.strict:
        for index, row in enumerate(features.to_dict(orient="records"), start=1):
            if int(parse_ok.iloc[index - 1]) != 1:
                continue
            problems = missing_or_invalid_rule_inputs(row)
            if problems:
                details = ", ".join(f"{name}={reason}" for name, reason in sorted(problems.items())[:8])
                suffix = " ..." if len(problems) > 8 else ""
                row_id = row.get("record_id", f"row {index}")
                strict_problems.append(f"{row_id}: {details}{suffix}")
        if strict_problems:
            examples = "; ".join(strict_problems[:5])
            more = f"; plus {len(strict_problems) - 5} more row(s)" if len(strict_problems) > 5 else ""
            raise ValueError(
                "grammar score strict validation failed; required Betlas rule-score input "
                f"columns are missing or invalid: {examples}{more}. "
                "Use --no-strict only for exploratory scoring of incomplete feature tables."
            )
    rows: list[dict[str, Any]] = []
    passthrough_columns = [
        "record_id",
        "pdb_id",
        "chain_id",
        "domain_id",
        "source_mmcif_path",
        "source_mmcif_sha256",
        "source_mmcif_size",
        "source_mmcif_exists",
        "betlas_parse_ok",
        "betlas_error",
        "betlas_warnings",
    ]
    for index, row in enumerate(features.to_dict(orient="records")):
        out_row: dict[str, Any] = {
            column: row.get(column, "")
            for column in passthrough_columns
            if column in features.columns or column in {"record_id", "pdb_id", "chain_id", "domain_id"}
        }
        if int(parse_ok.iloc[index]) != 1:
            out_row["betlas_score_status"] = "parse_failed"
            rows.append(out_row)
            continue
        explanation = explain_fold_grammar(row, fold=args.fold, strict=args.strict)
        out_row["betlas_score_status"] = "ok"
        out_row.update(
            {
                "betlas_top_fold": explanation["top_fold"],
                "betlas_rule_margin": explanation["margin"],
            }
        )
        for label, score in explanation["scores"].items():
            out_row[f"betlas_rule_score_{label}"] = float(score)
        if args.fold is not None:
            out_row["requested_fold"] = args.fold
            out_row["requested_score"] = explanation["requested_score"]
            out_row["requested_rank"] = explanation["requested_rank"]
        rows.append(out_row)
    output = pd.DataFrame(rows)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        if "betlas_parse_ok" in features.columns:
            parse_ok_rows = int(
                pd.to_numeric(features["betlas_parse_ok"], errors="coerce")
                .fillna(0)
                .astype(int)
                .sum()
            )
        else:
            parse_ok_rows = int(len(output))
        write_json(
            path.with_suffix(f"{path.suffix}.manifest.json"),
            build_run_manifest(
                command="betlas grammar score",
                parameters={
                    "features": str(features_path),
                    "fold": args.fold,
                    "strict": bool(args.strict),
                    "allow_parse_fail": bool(args.allow_parse_fail),
                },
                inputs={"features_csv": features_path},
                outputs={"rule_scores_csv": path},
                metrics={
                    "rows": int(len(output)),
                    "parse_ok_rows": parse_ok_rows,
                },
                extra={"features_file": file_state(features_path)},
            ),
        )
        print(f"Wrote {len(output)} grammar score rows to {path}")
    else:
        print(output.to_csv(index=False), end="")


def readout_command(args: argparse.Namespace) -> None:
    readout_args = list(args.readout_args)
    if getattr(args, "readout_help", False):
        readout_args.insert(0, "--help")
    run_readout(args.readout_name, readout_args)


def assets_list_command(_args: argparse.Namespace) -> None:
    for asset_id in list_assets():
        manifest = describe_asset(asset_id)
        print(
            "\t".join(
                [
                    asset_id,
                    str(manifest.get("readout", "")),
                    str(manifest.get("profile", "")),
                    str(manifest.get("release_status", "")),
                    str(len(manifest.get("files", []))),
                ]
            )
        )


def assets_describe_command(args: argparse.Namespace) -> None:
    manifest = describe_asset(args.asset_id)
    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    print(f"Asset: {manifest['asset_id']}")
    print(f"Readout: {manifest.get('readout', '')}")
    print(f"Profile: {manifest.get('profile', '')}")
    release_status = str(manifest.get("release_status", ""))
    print(f"Release status: {release_status}")
    print(f"Bundle: {manifest.get('bundle', '')}")
    if release_status.lower() == "pending_release":
        print("Download base: explicit BETLAS_ASSET_BASE_URL or --base-url local mirror required")
    else:
        print(f"Default base URL: {DEFAULT_ASSET_BASE_URL}")
    print("Files:")
    for file_info in manifest.get("files", []):
        purpose = str(file_info.get("purpose", "")).strip()
        purpose_text = f", purpose={purpose}" if purpose else ""
        print(
            f"  - {file_info['filename']} "
            f"({file_info['byte_size']} bytes, sha256={file_info['sha256']}{purpose_text})"
        )


def assets_download_command(args: argparse.Namespace) -> None:
    paths = download_asset(
        args.asset_id,
        cache_dir=args.cache_dir,
        filenames=args.file,
        force=args.force,
        base_url=args.base_url,
    )
    for path in paths:
        print(path)


def assets_verify_command(args: argparse.Namespace) -> None:
    results = verify_asset(
        args.asset_id,
        cache_dir=args.cache_dir,
        filenames=args.file,
        strict=args.strict,
    )
    for filename, ok in results.items():
        print(f"{filename}\t{'ok' if ok else 'failed'}")
    if not all(results.values()):
        raise SystemExit(2)


def assets_path_command(args: argparse.Namespace) -> None:
    path = resolve_asset_path(
        args.asset_id,
        filename=args.file,
        cache_dir=args.cache_dir,
        download=args.download,
        force=args.force,
        base_url=args.base_url,
    )
    if args.must_exist and not path.exists():
        raise AssetError(f"asset path does not exist: {path}; use --download to fetch it first")
    print(path)


def assets_check_esmc_command(args: argparse.Namespace) -> None:
    path = resolve_esmc_weights(args.path, required=args.required)
    if path is None:
        print("ESM-C weights were not provided; set BETLAS_ESMC_WEIGHTS or pass --path when needed.")
        return
    print(path)


def examples_list_command(_args: argparse.Namespace) -> None:
    for name in list_examples():
        print(name)


def examples_copy_command(args: argparse.Namespace) -> None:
    path = copy_example(args.name, args.out_dir)
    print(path)


def chains_command(args: argparse.Namespace) -> None:
    structure = _require_file(args.path, label="structure")
    rows = inspect_mmcif_chains(structure)
    if args.format == "json":
        print(json.dumps({"structure": str(structure), "chains": rows}, indent=2, sort_keys=True))
        return
    print(
        "auth_chain_id\tlabel_chain_ids\tstandard_ca_residue_count\tinsertion_code_ca_count\t"
        "nonpolymer_atom_rows\tsheet_annotation_available\thelix_conf_annotation_available\tworkflow_hints"
    )
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["auth_chain_id"]),
                    ",".join(str(value) for value in row["label_chain_ids"]),
                    str(row["standard_ca_residue_count"]),
                    str(row["insertion_code_ca_count"]),
                    str(row["nonpolymer_atom_rows"]),
                    "yes" if row["sheet_annotation_available"] else "no",
                    "yes" if row["helix_conf_annotation_available"] else "no",
                    ",".join(str(value) for value in row["workflow_hints"]),
                ]
            )
        )


def structure_inspect_command(args: argparse.Namespace) -> None:
    chains_command(args)


def _slice_summary_payload(args: argparse.Namespace, bundle: Any) -> dict[str, Any]:
    source_state = file_state(Path(args.path))
    stem = Path(args.path).name.split(".", 1)[0].lower()
    return {
        "input": source_state["path"],
        "record_id": args.record_id or f"{stem}_{args.chain}",
        "pdb_id": (args.pdb_id or stem).lower(),
        "domain_id": args.domain_id or f"{stem}_{args.chain}",
        "chain_id": args.chain,
        "residue_ranges": args.residue_ranges or "",
        "source_mmcif_path": source_state["path"],
        "source_mmcif_sha256": source_state["sha256"],
        "source_mmcif_size": source_state["size"],
        "source_mmcif_exists": int(bool(source_state["exists"])),
        "status": "ok" if bundle.slices else "no_informative_slices",
        "parser_warnings": list(getattr(bundle, "warnings", ())),
        "axis_name": bundle.axis_name,
        "axis_score": float(bundle.axis_score),
        "axis_origin": list(bundle.axis_origin),
        "axis_direction": list(bundle.axis_direction),
        "slice_count": len(bundle.slices),
        "config": {
            "min_points_per_slice": int(args.min_points_per_slice),
            "target_bin_width": float(args.target_bin_width),
            "min_bins": int(args.min_bins),
            "max_bins": int(args.max_bins),
        },
        "summary": summarize_slices(bundle),
    }


def slice_command(args: argparse.Namespace) -> None:
    config = SliceConfig(
        min_points_per_slice=args.min_points_per_slice,
        target_bin_width=args.target_bin_width,
        min_bins=args.min_bins,
        max_bins=args.max_bins,
    )
    bundle = slice_mmcif(
        args.path,
        chain_id=args.chain,
        residue_ranges=args.residue_ranges,
        record_id=args.record_id,
        domain_id=args.domain_id,
        pdb_id=args.pdb_id,
        axis=args.axis,
        config=config,
    )
    payload = _slice_summary_payload(args, bundle)

    if args.out:
        slice_columns = [
            "record_id",
            "pdb_id",
            "domain_id",
            "chain_id",
            "residue_ranges",
            "source_mmcif_sha256",
            "axis_name",
            "min_points_per_slice",
            "target_bin_width",
            "min_bins",
            "max_bins",
            "status",
            "slice_index",
            "z_left",
            "z_right",
            "z_center",
            "point_count",
            "angular_coverage",
            "largest_gap_fraction",
        ]
        rows = [
            {
                "record_id": payload["record_id"],
                "pdb_id": payload["pdb_id"],
                "domain_id": payload["domain_id"],
                "chain_id": payload["chain_id"],
                "residue_ranges": payload["residue_ranges"],
                "source_mmcif_sha256": payload["source_mmcif_sha256"],
                "axis_name": payload["axis_name"],
                "min_points_per_slice": payload["config"]["min_points_per_slice"],
                "target_bin_width": payload["config"]["target_bin_width"],
                "min_bins": payload["config"]["min_bins"],
                "max_bins": payload["config"]["max_bins"],
                "status": payload["status"],
                "slice_index": item.index,
                "z_left": item.z_left,
                "z_right": item.z_right,
                "z_center": item.z_center,
                "point_count": item.point_count,
                "angular_coverage": item.angular_coverage,
                "largest_gap_fraction": item.largest_gap_fraction,
            }
            for item in bundle.slices
        ]
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=slice_columns).to_csv(path, index=False)
        print(f"Wrote {len(rows)} Betlas slice rows to {path}")

    if args.points_out:
        rows = []
        point_columns = [
            "record_id",
            "pdb_id",
            "domain_id",
            "chain_id",
            "residue_ranges",
            "source_mmcif_sha256",
            "axis_name",
            "min_points_per_slice",
            "target_bin_width",
            "min_bins",
            "max_bins",
            "status",
            "included",
            "exclusion_reason",
            "slice_index",
            "point_index",
            "x",
            "y",
            "z",
            "angle",
            "auth_seq_id",
            "label_seq_id",
            "insertion_code",
            "residue_name",
            "residue_uid",
            "strand_id",
            "sheet_id",
            "sheet_range_id",
        ]
        for point in bundle.points:
            rows.append(
                {
                    "record_id": payload["record_id"],
                    "pdb_id": payload["pdb_id"],
                    "domain_id": payload["domain_id"],
                    "chain_id": payload["chain_id"],
                    "residue_ranges": payload["residue_ranges"],
                    "source_mmcif_sha256": payload["source_mmcif_sha256"],
                    "axis_name": payload["axis_name"],
                    "min_points_per_slice": payload["config"]["min_points_per_slice"],
                    "target_bin_width": payload["config"]["target_bin_width"],
                    "min_bins": payload["config"]["min_bins"],
                    "max_bins": payload["config"]["max_bins"],
                    "status": payload["status"],
                    "included": int(bool(point.included)),
                    "exclusion_reason": point.exclusion_reason,
                    "slice_index": "" if point.slice_index is None else point.slice_index,
                    "point_index": point.point_index,
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                    "angle": point.angle,
                    "auth_seq_id": point.auth_seq_id,
                    "label_seq_id": point.label_seq_id,
                    "insertion_code": point.insertion_code,
                    "residue_name": point.residue_name,
                    "residue_uid": point.residue_uid,
                    "strand_id": point.strand_id,
                    "sheet_id": point.sheet_id,
                    "sheet_range_id": point.sheet_range_id,
                }
            )
        path = Path(args.points_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=point_columns).to_csv(path, index=False)
        print(f"Wrote {len(rows)} Betlas slice-point rows to {path}")

    if args.summary_out:
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote Betlas slice summary to {path}")

    if not args.out and not args.points_out and not args.summary_out:
        print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if (
        len(raw_argv) >= 2
        and raw_argv[0] == "readout"
        and raw_argv[1] != "list"
        and not raw_argv[1].startswith("-")
    ):
        try:
            run_readout(raw_argv[1], raw_argv[2:])
        except (AssetError, ValueError, FileNotFoundError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        return

    parser = argparse.ArgumentParser(
        prog="betlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Betlas command-line tools for beta-structure geometry features, grammar scores, readouts, benchmarks, and release assets.",
        epilog=(
            "Common workflows:\n"
            "  betlas examples copy mini --out-dir runs/examples\n"
            "  betlas extract-features --structure runs/examples/mini.cif --chain A --out runs/mini_features.csv\n"
            "  betlas grammar score --features runs/mini_features.csv --out runs/mini_rule_scores.csv\n"
            "  betlas slice structure.cif --chain A\n"
            "  betlas readout beta-barrel-detection structure.cif --out runs/detection.csv\n"
            "  betlas assets describe betlas-beta-barrel-detection-official-v1\n\n"
            "Use '<command> --help' for command-specific inputs, outputs, and examples."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-dataset",
        help="Build a CATH-derived seven-class label table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a Betlas label table from local CATH source files.",
        epilog=(
            "Examples:\n"
            "  betlas build-dataset --all-eligible --out runs/betlas_cath_labels.csv\n"
            "  betlas build-dataset --sample-balanced --target-per-class 220 --seed 13\n\n"
            "Output: CSV with record identifiers, CATH metadata, fold labels, and provenance columns."
        ),
    )
    build.add_argument("--cath-dir", default=str(DEFAULT_CATH_DIR), help="Directory containing CATH source files.")
    build.add_argument("--target-per-class", type=int, default=220, help="Balanced-sampler target rows per fold class.")
    build.add_argument("--seed", type=int, default=13, help="Random seed for deterministic sampling.")
    build.add_argument("--include-putative", action="store_true", help="Include putative CATH-derived rows when eligible.")
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
    build.add_argument("--max-per-pdb", type=int, default=4, help="Maximum sampled domains per PDB entry.")
    build.add_argument("--initial-max-per-s35", type=int, default=1, help="Initial maximum sampled domains per CATH S35 cluster.")
    build.add_argument("--max-s35-cap", type=int, default=64, help="Maximum relaxed cap per CATH S35 cluster.")
    build.add_argument("--out", default=str(DEFAULT_LABELS_CSV), help="Output label CSV path.")
    build.set_defaults(func=build_dataset_command)

    extract = sub.add_parser(
        "extract-features",
        help="Download mmCIF files and extract Betlas features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Write Betlas geometry grammar features from either a label table plus downloaded mmCIF files "
            "or one user-provided mmCIF chain."
        ),
        epilog=(
            "Examples:\n"
            "  betlas extract-features --labels runs/labels.csv --out runs/features.csv\n"
            "  betlas extract-features --labels runs/labels.csv --mmcif-dir cache/mmcif --workers 8\n\n"
            "  betlas extract-features --structure runs/examples/mini.cif --chain A --out runs/mini_features.csv\n"
            "  betlas grammar score --features runs/mini_features.csv\n\n"
            "Single-structure input currently accepts mmCIF files: .cif, .mmcif, .cif.gz, .mmcif.gz.\n"
            "Dataset construction and batch feature extraction may download CATH/RCSB data and can take time.\n"
            "Output: feature CSV with Betlas geometry columns, source mmCIF provenance, and parse-status columns."
        ),
    )
    extract.add_argument("--labels", default=str(DEFAULT_LABELS_CSV), help="Input label CSV from build-dataset or an equivalent table for batch mode.")
    extract.add_argument("--structure", default=None, help="Single mmCIF structure path for one-chain feature extraction.")
    extract.add_argument("--chain", default=None, help="Author chain id for --structure mode.")
    extract.add_argument("--residue-ranges", default="", help="Optional residue ranges for --structure mode, e.g. '10-180:A' or 'A:10-180'.")
    extract.add_argument("--record-id", default=None, help="Optional record id for --structure mode.")
    extract.add_argument("--domain-id", default=None, help="Optional domain id for --structure mode.")
    extract.add_argument("--pdb-id", default=None, help="Optional PDB id metadata for --structure mode.")
    extract.add_argument(
        "--write-failed-row",
        action="store_true",
        help="In --structure mode, write a status-only row instead of failing when parsing or beta-sheet extraction fails.",
    )
    extract.add_argument("--mmcif-dir", default=str(DEFAULT_MMCIF_DIR), help="Directory for downloaded or cached mmCIF files.")
    extract.add_argument("--out", default=str(DEFAULT_FEATURES_CSV), help="Output feature CSV path.")
    extract.add_argument("--workers", type=int, default=1, help="Feature-extraction worker processes.")
    extract.add_argument("--download-workers", type=int, default=8, help="Concurrent mmCIF download workers.")
    extract.add_argument("--force-download", action="store_true", help="Redownload mmCIF files even when cached files exist.")
    extract.set_defaults(func=extract_features_command)

    bench = sub.add_parser(
        "benchmark",
        help="Run grouped benchmark with transparent and learned models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate Betlas feature tables with grouped cross-validation and configured model families.",
        epilog=(
            "Examples:\n"
            "  betlas benchmark --features runs/features.csv --out-dir runs/benchmark\n"
            "  betlas benchmark --features runs/features.csv --splits 5 --seed 13 --config custom_benchmark.yaml\n\n"
            "Outputs: metrics summary, per-fold metrics, out-of-fold predictions, and run metadata under --out-dir."
        ),
    )
    bench.add_argument("--features", default=str(DEFAULT_FEATURES_CSV), help="Input Betlas feature CSV.")
    bench.add_argument("--out-dir", default=str(DEFAULT_RUN_DIR), help="Directory for benchmark outputs.")
    bench.add_argument("--splits", type=int, default=5, help="Grouped cross-validation split count.")
    bench.add_argument("--seed", type=int, default=13, help="Random seed for splits and compatible models.")
    bench.add_argument("--config", default=None, help="Optional benchmark YAML config.")
    bench.set_defaults(func=benchmark_command)

    ablate = sub.add_parser(
        "ablate",
        help="Run grouped feature ablations for geometry grammars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Measure feature-family and feature-block contributions using grouped benchmark splits.",
        epilog=(
            "Examples:\n"
            "  betlas ablate --features runs/features.csv --out-dir runs/ablations\n"
            "  betlas ablate --features runs/features.csv --max-group-combo-size 3 --config custom_ablation.yaml\n\n"
            "Outputs: baseline metrics, drop-group metrics, drop-feature metrics, and delta tables under --out-dir."
        ),
    )
    ablate.add_argument("--features", default=str(DEFAULT_FEATURES_CSV), help="Input Betlas feature CSV.")
    ablate.add_argument("--out-dir", default="runs/betlas_full_ablations", help="Directory for ablation outputs.")
    ablate.add_argument("--splits", type=int, default=5, help="Grouped cross-validation split count.")
    ablate.add_argument("--seed", type=int, default=13, help="Random seed for splits and compatible models.")
    ablate.add_argument("--max-group-combo-size", type=int, default=3, help="Largest feature-group combination size to evaluate.")
    ablate.add_argument("--config", default=None, help="Optional ablation YAML config.")
    ablate.set_defaults(func=ablate_command)

    grammar = sub.add_parser(
        "grammar",
        help="Inspect and score Betlas geometry grammars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="List geometry grammar families, describe their mathematical implementation, or score fold rules from a feature table.",
        epilog=(
            "Examples:\n"
            "  betlas grammar list\n"
            "  betlas grammar describe sheet_pair_packing\n"
            "  betlas grammar describe fold_rule_scores --format json\n"
            "  betlas grammar score --features runs/features.csv --out runs/rule_scores.csv"
            " --strict"
        ),
    )
    grammar_sub = grammar.add_subparsers(dest="grammar_command", required=True)
    grammar_list = grammar_sub.add_parser("list", help="List geometry grammar families.")
    grammar_list.set_defaults(func=grammar_list_command)
    grammar_describe = grammar_sub.add_parser("describe", help="Describe one geometry grammar, including inputs, outputs, and mathematical summary.")
    grammar_describe.add_argument("name", help="Grammar name from 'betlas grammar list'.")
    grammar_describe.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    grammar_describe.set_defaults(func=grammar_describe_command)
    grammar_score = grammar_sub.add_parser(
        "score",
        help="Score fold grammar rules from a feature table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compute transparent Betlas fold-rule scores for every row in a feature table.",
        epilog=(
            "Strict validation is enabled by default and requires the Betlas geometry columns consumed by "
            "the transparent rule formulas. Use --no-strict only for exploratory scoring of incomplete tables.\n"
            "Output columns include record identifiers, betlas_top_fold, betlas_rule_margin, "
            "and one betlas_rule_score_<fold> column per fold label."
        ),
    )
    grammar_score.add_argument("--features", default=str(DEFAULT_FEATURES_CSV), help="Input Betlas feature CSV.")
    grammar_score.add_argument("--fold", choices=list(FOLD_LABELS), default=None, help="Optional fold label to add requested-score and requested-rank columns.")
    grammar_score.add_argument("--out", default=None, help="Output CSV path. Defaults to stdout.")
    grammar_score.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Validate required rule-score input columns before scoring (default).",
    )
    grammar_score.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow missing or nonnumeric rule-score inputs to use the compatibility zero-fill behavior.",
    )
    grammar_score.add_argument(
        "--allow-parse-fail",
        action="store_true",
        help="Carry betlas_parse_ok != 1 rows through as status-only rows without fold calls or rule scores.",
    )
    grammar_score.set_defaults(func=grammar_score_command)

    chains = sub.add_parser(
        "chains",
        help="Inspect mmCIF author/label chains and annotation availability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Inspect chains in one mmCIF/mmCIF.gz structure before choosing --chain for Betlas workflows.",
        epilog=(
            "Examples:\n"
            "  betlas chains runs/examples/mini.cif\n"
            "  betlas chains runs/examples/mini.cif --format json\n\n"
            "Outputs: author chain id, label chain ids, standard CA residue counts, "
            "sheet/helix annotation availability, and supported workflow hints."
        ),
    )
    chains.add_argument("path", help="Input mmCIF or mmCIF.gz file.")
    chains.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    chains.set_defaults(func=chains_command)

    structure = sub.add_parser(
        "structure",
        help="Inspect structure metadata used by Betlas workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Structure utilities. Currently exposes chain inspection for mmCIF/mmCIF.gz files.",
    )
    structure_sub = structure.add_subparsers(dest="structure_command", required=True)
    structure_inspect = structure_sub.add_parser(
        "inspect",
        help="Inspect mmCIF author/label chains and annotation availability.",
    )
    structure_inspect.add_argument("path", help="Input mmCIF or mmCIF.gz file.")
    structure_inspect.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    structure_inspect.set_defaults(func=structure_inspect_command)

    slice_parser = sub.add_parser(
        "slice",
        help="Inspect Betlas axis-aligned beta-structure slices for one mmCIF chain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Project beta-structure points onto a Betlas axis, partition them into z-bins, "
            "and report the slice-closure summaries used by geometry grammars."
        ),
        epilog=(
            "Examples:\n"
            "  betlas slice 1abc.cif --chain A\n"
            "  betlas slice 1abc.cif --chain A --axis best --out runs/slices.csv --summary-out runs/slice_summary.json\n"
            "  betlas slice 1abc.cif --chain A --residue-ranges 10-180:A --points-out runs/slice_points.csv\n\n"
            "Output summary keys match the unprefixed values behind betlas_axis_best_* feature columns. "
            "The default axis is the same best-axis scoring rule used during Betlas feature extraction."
        ),
    )
    slice_parser.add_argument("path", help="Input mmCIF or mmCIF.gz file.")
    slice_parser.add_argument("--chain", required=True, help="Author chain id to slice.")
    slice_parser.add_argument("--residue-ranges", default="", help="Optional residue range string such as '10-180:A'.")
    slice_parser.add_argument("--axis", default="best", help="Axis hypothesis to use: best, strand_axis, point_pc1, point_pc2, or point_pc3.")
    slice_parser.add_argument("--record-id", default=None, help="Optional record id used in summary metadata.")
    slice_parser.add_argument("--domain-id", default=None, help="Optional domain id used in summary metadata.")
    slice_parser.add_argument("--pdb-id", default=None, help="Optional PDB id used in summary metadata.")
    slice_parser.add_argument("--min-points-per-slice", type=int, default=4, help="Minimum projected beta points required for an informative slice.")
    slice_parser.add_argument("--target-bin-width", type=float, default=4.0, help="Target z-bin width in Angstroms for axis-slice summaries.")
    slice_parser.add_argument("--min-bins", type=int, default=4, help="Minimum number of z-bins used for summary scoring.")
    slice_parser.add_argument("--max-bins", type=int, default=24, help="Maximum number of z-bins used for summary scoring.")
    slice_parser.add_argument("--out", default=None, help="Optional output CSV with one row per informative slice.")
    slice_parser.add_argument("--points-out", default=None, help="Optional residue-traceable output CSV with projected points assigned to informative slices.")
    slice_parser.add_argument("--summary-out", default=None, help="Optional output JSON with axis metadata and summary values.")
    slice_parser.set_defaults(func=slice_command)

    readout = sub.add_parser(
        "readout",
        help="Run or inspect Betlas secondary readouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="List or run Betlas secondary readouts for beta-barrel detection, stave counting, and topology diagnostics.",
        epilog="Use 'betlas readout <name> --help' for readout-specific inputs, outputs, and Hydra override examples.",
    )
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

    assets = sub.add_parser(
        "assets",
        help="Inspect, download, and verify Betlas release assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Use packaged Betlas manifests to locate, download, and verify release asset files.",
        epilog=(
            "Examples:\n"
            "  betlas assets list\n"
            "  betlas assets describe betlas-beta-barrel-detection-official-v1\n"
            "  BETLAS_ASSET_BASE_URL=/mirror/betlas-assets betlas assets download betlas-beta-barrel-detection-official-v1\n"
            "  betlas assets verify betlas-beta-barrel-detection-official-v1 --strict\n\n"
            "Pending-release bundles require BETLAS_ASSET_BASE_URL or --base-url pointing at a local mirror; "
            "BETLAS_ASSET_DIR or ~/.cache/betlas/assets for the local cache."
        ),
    )
    assets_sub = assets.add_subparsers(dest="assets_command", required=True)
    assets_list = assets_sub.add_parser("list", help="List available Betlas asset bundles.")
    assets_list.set_defaults(func=assets_list_command)
    assets_describe = assets_sub.add_parser("describe", help="Describe one Betlas asset bundle.")
    assets_describe.add_argument("asset_id", help="Asset id from 'betlas assets list'.")
    assets_describe.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    assets_describe.set_defaults(func=assets_describe_command)
    assets_download = assets_sub.add_parser(
        "download",
        help="Download a Betlas asset bundle or selected files into the local cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Download release asset files into the Betlas cache with atomic writes and SHA-256/size verification.",
        epilog=(
            "Examples:\n"
            "  BETLAS_ASSET_BASE_URL=/mirror/betlas-assets betlas assets download betlas-beta-barrel-staves-official-v1"
            "\n"
            "  betlas assets download betlas-beta-barrel-staves-official-v1 --base-url /mirror/betlas-assets --file betlas_151_chain_features.csv"
        ),
    )
    assets_download.add_argument("asset_id", help="Asset id from 'betlas assets list'.")
    assets_download.add_argument("--cache-dir", default=None, help="Asset cache directory. Defaults to BETLAS_ASSET_DIR or ~/.cache/betlas/assets.")
    assets_download.add_argument(
        "--base-url",
        default=None,
        help="Local mirror or released base URL. Pending-release assets require this or BETLAS_ASSET_BASE_URL.",
    )
    assets_download.add_argument("--file", action="append", default=None, help="Download one file from the bundle; repeat for multiple files.")
    assets_download.add_argument("--force", action="store_true", help="Replace existing cached files after re-downloading and verifying them.")
    assets_download.set_defaults(func=assets_download_command)
    assets_verify = assets_sub.add_parser(
        "verify",
        help="Verify cached Betlas asset files against manifest hashes.",
        description="Check cached files against the packaged manifest byte size and SHA-256 values.",
    )
    assets_verify.add_argument("asset_id", help="Asset id from 'betlas assets list'.")
    assets_verify.add_argument("--cache-dir", default=None, help="Asset cache directory. Defaults to BETLAS_ASSET_DIR or ~/.cache/betlas/assets.")
    assets_verify.add_argument("--file", action="append", default=None, help="Verify one file from the bundle; repeat for multiple files.")
    assets_verify.add_argument("--strict", action="store_true", help="Raise a detailed error when a file is missing or invalid.")
    assets_verify.set_defaults(func=assets_verify_command)
    assets_path = assets_sub.add_parser("path", help="Print the cache path for a Betlas asset bundle or file.")
    assets_path.add_argument("asset_id", help="Asset id from 'betlas assets list'.")
    assets_path.add_argument("--file", default=None, help="Return a path for one file in the bundle.")
    assets_path.add_argument("--cache-dir", default=None, help="Asset cache directory. Defaults to BETLAS_ASSET_DIR or ~/.cache/betlas/assets.")
    assets_path.add_argument("--download", action="store_true", help="Download and verify the asset before printing the path.")
    assets_path.add_argument("--must-exist", action="store_true", help="Fail if the resolved cache path does not exist.")
    assets_path.add_argument("--force", action="store_true", help="Replace existing cached files when used with --download.")
    assets_path.add_argument(
        "--base-url",
        default=None,
        help="Local mirror or released base URL. Pending-release assets require this or BETLAS_ASSET_BASE_URL.",
    )
    assets_path.set_defaults(func=assets_path_command)
    assets_esmc = assets_sub.add_parser(
        "check-esmc",
        help="Resolve a local ESM-C weights path without downloading third-party weights.",
        description=(
            "Resolve and validate a local ESM-C weights path. Betlas does not download "
            "or redistribute third-party ESM-C model weights."
        ),
    )
    assets_esmc.add_argument("--path", default=None, help="Local ESM-C weights path. Defaults to BETLAS_ESMC_WEIGHTS.")
    assets_esmc.add_argument("--required", action="store_true", help="Exit with an error if no ESM-C weights path is available.")
    assets_esmc.set_defaults(func=assets_check_esmc_command)

    examples = sub.add_parser(
        "examples",
        help="List or copy packaged Betlas example inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Inspect and copy tiny packaged Betlas examples for installed-package smoke tests.",
        epilog=(
            "Examples:\n"
            "  betlas examples list\n"
            "  betlas examples copy mini --out-dir runs/examples"
        ),
    )
    examples_sub = examples.add_subparsers(dest="examples_command", required=True)
    examples_list = examples_sub.add_parser("list", help="List packaged examples.")
    examples_list.set_defaults(func=examples_list_command)
    examples_copy = examples_sub.add_parser("copy", help="Copy a packaged example into a local directory.")
    examples_copy.add_argument("name", help="Example id from 'betlas examples list'.")
    examples_copy.add_argument("--out-dir", default=".", help="Destination directory for the copied example file.")
    examples_copy.set_defaults(func=examples_copy_command)

    args = parser.parse_args(raw_argv)
    try:
        args.func(args)
    except (AssetError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
