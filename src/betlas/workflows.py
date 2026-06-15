from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from .constants import DEFAULT_CATH_DIR, DEFAULT_MMCIF_DIR
from .features.extract import assert_no_diagnostic_label_leakage, extract_feature_row
from .io.cath import build_cath_dataset, domain_from_row
from .io.rcsb import download_mmcifs, mmcif_path_for
from .ml.benchmark import BenchmarkResult, run_grouped_benchmark
from .provenance import build_run_manifest, file_state, write_json


@dataclass(frozen=True)
class FullPipelineConfig:
    cath_dir: str = str(DEFAULT_CATH_DIR)
    mmcif_dir: str = str(DEFAULT_MMCIF_DIR)
    target_per_class: int = 220
    include_putative: bool = False
    all_eligible: bool = True
    max_per_pdb: int = 4
    initial_max_per_s35: int = 1
    max_s35_cap: int = 64
    workers: int = 8
    download_workers: int = 32
    seed: int = 13
    splits: int = 5
    force_download: bool = False
    benchmark_config: str | None = None
    out_dir: str = "runs/betlas_full"


@dataclass(frozen=True)
class FullPipelineResult:
    labels_csv: Path
    features_csv: Path
    benchmark_dir: Path
    manifest_path: Path
    benchmark: BenchmarkResult


def load_full_pipeline_config(config_path: str | Path | None = None) -> FullPipelineConfig:
    if config_path is None:
        return FullPipelineConfig()
    data = OmegaConf.to_container(OmegaConf.load(Path(config_path)), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"pipeline config must be a mapping: {config_path}")
    unknown = sorted(set(data) - set(FullPipelineConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown full-pipeline config keys in {config_path}: {unknown}")
    return FullPipelineConfig(**data)


def _extract_feature_row_for_worker(row: dict[str, Any], mmcif_dir: str) -> dict[str, Any]:
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


def extract_features_table(
    labels: pd.DataFrame,
    *,
    mmcif_dir: Path,
    workers: int,
) -> pd.DataFrame:
    rows = labels.to_dict(orient="records")
    feature_rows: list[dict[str, Any]] = []

    if workers <= 1:
        for row in tqdm(rows, desc="Extracting features"):
            feature_rows.append(_extract_feature_row_for_worker(row, str(mmcif_dir)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_feature_row_for_worker, row, str(mmcif_dir)) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                feature_rows.append(future.result())

    feature_df = pd.DataFrame(feature_rows)
    label_order = {str(row["record_id"]): index for index, row in enumerate(rows)}
    feature_df["_order"] = feature_df["record_id"].astype(str).map(label_order)
    return feature_df.sort_values("_order").drop(columns=["_order"])


def run_full_pipeline(config: FullPipelineConfig) -> FullPipelineResult:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_dir / "betlas_full_labels.csv"
    features_csv = out_dir / "betlas_full_features.csv"
    benchmark_dir = out_dir / "benchmark"
    manifest_path = out_dir / "full_pipeline_manifest.json"

    labels = build_cath_dataset(
        cath_dir=Path(config.cath_dir),
        target_per_class=config.target_per_class,
        seed=config.seed,
        include_putative=config.include_putative,
        all_eligible=config.all_eligible,
        max_per_pdb=config.max_per_pdb,
        initial_max_per_s35=config.initial_max_per_s35,
        max_s35_cap=config.max_s35_cap,
    )
    labels.to_csv(labels_csv, index=False)

    pdb_ids = labels["pdb_id"].astype(str).str.lower().tolist()
    download_mmcifs(
        pdb_ids,
        Path(config.mmcif_dir),
        workers=config.download_workers,
        force=config.force_download,
    )
    features = extract_features_table(
        labels,
        mmcif_dir=Path(config.mmcif_dir),
        workers=config.workers,
    )
    features.to_csv(features_csv, index=False)
    benchmark = run_grouped_benchmark(
        features_csv,
        benchmark_dir,
        n_splits=config.splits,
        random_state=config.seed,
        config_path=config.benchmark_config,
    )

    cath_inputs = {
        path.name: file_state(path)
        for path in sorted(Path(config.cath_dir).glob("*.gz"))
        if path.is_file()
    }
    manifest = build_run_manifest(
        command="betlas full-pipeline",
        parameters=asdict(config),
        inputs={"labels_csv": labels_csv, "features_csv": features_csv},
        outputs={
            "labels_csv": labels_csv,
            "features_csv": features_csv,
            "benchmark_metrics": benchmark_dir / "metrics_summary.csv",
            "benchmark_oof": benchmark_dir / "oof_predictions.csv",
        },
        metrics={
            "label_rows": int(len(labels)),
            "feature_rows": int(len(features)),
            "parse_ok_rows": int(pd.to_numeric(features.get("cz_parse_ok", 0), errors="coerce").fillna(0).sum()),
            "best_model": str(benchmark.metrics_summary.iloc[0]["model"]),
            "best_macro_f1": float(benchmark.metrics_summary.iloc[0]["macro_f1"]),
        },
        extra={"cath_inputs": cath_inputs},
    )
    write_json(manifest_path, manifest)
    return FullPipelineResult(labels_csv, features_csv, benchmark_dir, manifest_path, benchmark)
