from __future__ import annotations

import hashlib
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from ..constants import FOLD_LABELS
from ..io.mmcif import _selected_ca_records, read_mmcif_dict_for_geometry
from ..models import DomainCandidate, ResidueRecord
from ..provenance import build_run_manifest, file_state, write_json
from .figures import _plot_context_figures

BOUNDARY_LABELS = ("beta_barrel", "beta_sandwich", "jelly_roll")
SANDWICH_JELLY_LABELS = ("beta_sandwich", "jelly_roll")

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


@dataclass(frozen=True)
class ExternalBaselineResult:
    out_dir: Path
    figure_dir: Path
    outputs: dict[str, Path]
    manifest_path: Path


def _cfg_get(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _sequence_from_residues(residues: list[ResidueRecord]) -> str:
    return "".join(AA3_TO_1.get(residue.residue_name, "X") for residue in residues)


def _pdb_atom_line(serial: int, residue: ResidueRecord, residue_index: int) -> str:
    x, y, z = residue.coord_ca
    resname = residue.residue_name if residue.residue_name != "MSE" else "MET"
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} A{residue_index:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def _write_domain_pdb(path: Path, residues: list[ResidueRecord]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for serial, residue in enumerate(residues, start=1):
            handle.write(_pdb_atom_line(serial, residue, serial))
        handle.write("TER\nEND\n")


def _domain_from_row(row: pd.Series) -> DomainCandidate:
    return DomainCandidate.from_mapping(row.to_dict())


def prepare_domains(
    subset_csv: Path,
    out_dir: Path,
    *,
    reuse: bool,
    min_residues: int,
) -> tuple[pd.DataFrame, Path, Path]:
    structure_dir = out_dir / "domain_pdb"
    structure_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "domain_metadata.csv"
    fasta_path = out_dir / "domain_sequences.fasta"
    if reuse and metadata_path.exists() and fasta_path.exists():
        metadata = _read_table(metadata_path)
        if len(metadata) and all(Path(path).exists() for path in metadata["domain_pdb_path"]):
            return metadata, structure_dir, fasta_path

    subset = _read_table(subset_csv)
    rows: list[dict[str, Any]] = []
    fasta_lines: list[str] = []
    for _, row_series in tqdm(subset.iterrows(), total=len(subset), desc="Preparing domains"):
        domain = _domain_from_row(row_series)
        mmcif_path = Path(str(row_series.get("source_mmcif_path", "")))
        out_pdb = structure_dir / f"{domain.record_id}.pdb"
        status = "ok"
        error = ""
        sequence = ""
        residue_count = 0
        try:
            mmcif = read_mmcif_dict_for_geometry(mmcif_path, use_cache=True)
            residues, _ = _selected_ca_records(mmcif, domain)
            sequence = _sequence_from_residues(residues)
            residue_count = len(residues)
            if residue_count < min_residues:
                status = "too_short"
            else:
                _write_domain_pdb(out_pdb, residues)
                fasta_lines.append(f">{domain.record_id}\n{sequence}\n")
        except Exception as exc:  # pragma: no cover - data-dependent parse guard
            status = "error"
            error = str(exc)
        rows.append(
            {
                "record_id": domain.record_id,
                "pdb_id": domain.pdb_id,
                "chain_id": domain.chain_id,
                "domain_id": domain.domain_id,
                "residue_ranges": domain.residue_ranges,
                "fold_label_final": domain.fold_label_final,
                "cath_code": domain.cath_code,
                "cath_topology_code": domain.cath_topology_code,
                "cath_name": domain.cath_name,
                "group": str(row_series.get("cath_s35_cluster_id", "")),
                "sequence": sequence,
                "sequence_length": residue_count,
                "domain_pdb_path": str(out_pdb) if status == "ok" else "",
                "source_mmcif_path": str(mmcif_path),
                "status": status,
                "error": error,
            }
        )
    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_path, index=False)
    fasta_path.write_text("".join(fasta_lines), encoding="ascii")
    return metadata, structure_dir, fasta_path


def _load_fold_assignments(path: Path) -> pd.DataFrame:
    return _read_table(path)[["record_id", "fold", "group"]].drop_duplicates("record_id")


def _metric_rows(
    predictions: pd.DataFrame,
    *,
    model: str,
    dataset: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    valid = predictions[predictions["pred_label"].astype(str) != ""].copy()
    y_true = valid["true_label"].astype(str).to_numpy()
    y_pred = valid["pred_label"].astype(str).to_numpy()
    top2_acc = float(
        np.mean(
            [
                truth in str(top2).split(";")
                for truth, top2 in zip(valid["true_label"], valid.get("top2_labels", ""), strict=False)
            ]
        )
    ) if len(valid) and "top2_labels" in valid else float("nan")
    rows = [
        {
            "dataset": dataset,
            "model": model,
            "n": int(len(valid)),
            "coverage": float(len(valid) / max(1, len(predictions))),
            "metric": "accuracy",
            "value": float(accuracy_score(y_true, y_pred)) if len(valid) else float("nan"),
        },
        {
            "dataset": dataset,
            "model": model,
            "n": int(len(valid)),
            "coverage": float(len(valid) / max(1, len(predictions))),
            "metric": "macro_f1",
            "value": float(f1_score(y_true, y_pred, labels=list(FOLD_LABELS), average="macro", zero_division=0))
            if len(valid)
            else float("nan"),
        },
        {
            "dataset": dataset,
            "model": model,
            "n": int(len(valid)),
            "coverage": float(len(valid) / max(1, len(predictions))),
            "metric": "boundary_macro_f1",
            "value": float(
                f1_score(y_true, y_pred, labels=list(BOUNDARY_LABELS), average="macro", zero_division=0)
            )
            if len(valid)
            else float("nan"),
        },
        {
            "dataset": dataset,
            "model": model,
            "n": int(len(valid)),
            "coverage": float(len(valid) / max(1, len(predictions))),
            "metric": "sandwich_jelly_macro_f1",
            "value": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(SANDWICH_JELLY_LABELS),
                    average="macro",
                    zero_division=0,
                )
            )
            if len(valid)
            else float("nan"),
        },
        {
            "dataset": dataset,
            "model": model,
            "n": int(len(valid)),
            "coverage": float(len(valid) / max(1, len(predictions))),
            "metric": "top2_accuracy",
            "value": top2_acc,
        },
    ]
    if extra:
        for row in rows:
            row.update(extra)
    return rows


def _normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norm, 1e-12)


def run_esmc_embeddings(
    metadata: pd.DataFrame,
    out_dir: Path,
    *,
    weights_path: Path,
    device: str,
    precision: str,
    reuse: bool,
) -> tuple[np.ndarray, pd.DataFrame, Path, Path]:
    emb_path = out_dir / "esmc_mean_embeddings.npy"
    meta_path = out_dir / "esmc_embedding_metadata.csv"
    if reuse and emb_path.exists() and meta_path.exists():
        return np.load(emb_path), _read_table(meta_path), emb_path, meta_path

    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein
    from esm.tokenization import EsmSequenceTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmSequenceTokenizer()
    model = ESMC(tokenizer=tokenizer, d_model=1152, n_layers=36, n_heads=18).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {key.replace("module.", "").replace("model.", ""): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=False)
    if precision == "fp16" and str(device).startswith("cuda"):
        model = model.half()
    elif precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported ESM-C precision: {precision}")
    model.eval()

    ok = metadata[metadata["status"] == "ok"].copy().reset_index(drop=True)
    embeddings: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in tqdm(ok.itertuples(index=False), total=len(ok), desc="ESM-C embeddings"):
            sequence = str(row.sequence)[:1022]
            protein = ESMProtein(sequence=sequence)
            tokenized = model.encode(protein).sequence.unsqueeze(0).to(device)
            out = model(tokenized)
            residue = out.embeddings[0, 1:-1].detach().float().cpu()
            mean = residue.mean(dim=0).numpy().astype(np.float32)
            embeddings.append(mean)
            rows.append(
                {
                    "embedding_index": len(embeddings) - 1,
                    "record_id": row.record_id,
                    "fold_label_final": row.fold_label_final,
                    "sequence_length": int(row.sequence_length),
                    "embedded_length": int(len(sequence)),
                    "truncated": int(len(str(row.sequence)) > len(sequence)),
                }
            )
    matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 1152), dtype=np.float32)
    emb_meta = pd.DataFrame(rows)
    np.save(emb_path, matrix)
    emb_meta.to_csv(meta_path, index=False)
    return matrix, emb_meta, emb_path, meta_path


def run_embedding_knn(
    embeddings: np.ndarray,
    emb_meta: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    out_dir: Path,
    *,
    k_values: list[int],
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    meta = emb_meta.merge(fold_assignments, on="record_id", how="inner")
    if "embedding_index" in meta:
        order = pd.to_numeric(meta["embedding_index"], errors="raise").astype(int).to_numpy()
    else:
        order = meta.index.to_numpy()
    x = _normalise_embeddings(embeddings[order])
    labels = meta["fold_label_final"].astype(str).to_numpy()
    folds = meta["fold"].astype(str).to_numpy()
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)

    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for k in k_values:
        rows = []
        for i in range(len(meta)):
            train_mask = folds != folds[i]
            candidate_idx = np.flatnonzero(train_mask)
            scores = sim[i, candidate_idx]
            top = candidate_idx[np.argsort(scores)[::-1][:k]]
            top_labels = labels[top]
            counts = pd.Series(top_labels).value_counts()
            pred_label = str(counts.index[0]) if len(counts) else ""
            top2 = ";".join(str(label) for label in counts.index[:2])
            rows.append(
                {
                    "model": f"esmc_cosine_knn_k{k}",
                    "record_id": meta.iloc[i]["record_id"],
                    "fold": folds[i],
                    "group": meta.iloc[i]["group"],
                    "true_label": labels[i],
                    "pred_label": pred_label,
                    "top2_labels": top2,
                    "nearest_record_id": meta.iloc[top[0]]["record_id"] if len(top) else "",
                    "nearest_similarity": float(sim[i, top[0]]) if len(top) else float("nan"),
                }
            )
        pred = pd.DataFrame(rows)
        prediction_parts.append(pred)
        metric_rows.extend(_metric_rows(pred, model=f"esmc_cosine_knn_k{k}", dataset=dataset, extra={"k": k}))

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(out_dir / "esmc_knn_predictions.csv", index=False)
    metrics.to_csv(out_dir / "esmc_knn_metrics.csv", index=False)
    return predictions, metrics, sim


def _run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")


def run_foldseek_search(
    structure_dir: Path,
    out_dir: Path,
    *,
    foldseek_bin: Path,
    threads: int,
    sensitivity: float,
    max_seqs: int,
    reuse: bool,
) -> Path:
    hits_path = out_dir / "foldseek_all_vs_all.tsv"
    if reuse and hits_path.exists() and hits_path.stat().st_size > 0:
        return hits_path
    tmp_dir = out_dir / "foldseek_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(foldseek_bin),
        "easy-search",
        str(structure_dir),
        str(structure_dir),
        str(hits_path),
        str(tmp_dir),
        "--format-output",
        "query,target,evalue,bits,alntmscore,qtmscore,ttmscore,prob",
        "--max-seqs",
        str(max_seqs),
        "-s",
        str(sensitivity),
        "--threads",
        str(threads),
        "-v",
        "1",
    ]
    _run_command(command, out_dir / "foldseek.log")
    return hits_path


def run_foldseek_nn(
    hits_path: Path,
    metadata: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    out_dir: Path,
    *,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = metadata[metadata["status"] == "ok"].drop(columns=["group"], errors="ignore")
    meta = meta.merge(fold_assignments, on="record_id", how="inner")
    label_by_id = dict(zip(meta["record_id"], meta["fold_label_final"], strict=False))
    fold_by_id = dict(zip(meta["record_id"], meta["fold"].astype(str), strict=False))
    group_by_id = dict(zip(meta["record_id"], meta["group"], strict=False))
    hits = pd.read_csv(
        hits_path,
        sep="\t",
        names=["query", "target", "evalue", "bits", "alntmscore", "qtmscore", "ttmscore", "prob"],
        dtype=str,
    )
    for column in ["bits", "alntmscore", "qtmscore", "ttmscore", "prob"]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits = hits[hits["query"].isin(label_by_id) & hits["target"].isin(label_by_id)].copy()
    hits = hits[hits["query"] != hits["target"]].copy()
    hits = hits.sort_values(["query", "bits", "alntmscore", "prob"], ascending=[True, False, False, False])

    rows = []
    hits_by_query = {str(query): part for query, part in hits.groupby("query", sort=False)}
    for record_id in meta["record_id"]:
        part = hits_by_query.get(str(record_id), hits.iloc[0:0])
        test_fold = fold_by_id[record_id]
        candidates = part[part["target"].map(fold_by_id) != test_fold]
        candidates = candidates[candidates["target"].map(group_by_id) != group_by_id[record_id]]
        if len(candidates):
            best = candidates.iloc[0]
            distinct_labels: list[str] = []
            for target in candidates["target"]:
                label = label_by_id[str(target)]
                if label not in distinct_labels:
                    distinct_labels.append(label)
                if len(distinct_labels) == 2:
                    break
            rows.append(
                {
                    "model": "foldseek_nearest_neighbor",
                    "record_id": record_id,
                    "fold": test_fold,
                    "group": group_by_id[record_id],
                    "true_label": label_by_id[record_id],
                    "pred_label": label_by_id[str(best["target"])],
                    "top2_labels": ";".join(distinct_labels),
                    "nearest_record_id": best["target"],
                    "nearest_bits": best["bits"],
                    "nearest_alntmscore": best["alntmscore"],
                    "nearest_qtmscore": best["qtmscore"],
                    "nearest_ttmscore": best["ttmscore"],
                    "nearest_prob": best["prob"],
                }
            )
        else:
            rows.append(
                {
                    "model": "foldseek_nearest_neighbor",
                    "record_id": record_id,
                    "fold": test_fold,
                    "group": group_by_id[record_id],
                    "true_label": label_by_id[record_id],
                    "pred_label": "",
                    "top2_labels": "",
                    "nearest_record_id": "",
                }
            )
    predictions = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        _metric_rows(predictions, model="foldseek_nearest_neighbor", dataset=dataset)
    )
    predictions.to_csv(out_dir / "foldseek_nn_predictions.csv", index=False)
    metrics.to_csv(out_dir / "foldseek_nn_metrics.csv", index=False)
    return predictions, metrics


def run_external_baselines(config_path: str | Path) -> ExternalBaselineResult:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("external baseline config must be a mapping")
    out_dir = _as_path(_cfg_get(config, "outputs.out_dir", "runs/external_baselines_full"))
    figure_dir = _as_path(_cfg_get(config, "outputs.figure_dir", "outputs/external_baseline_context_figures"))
    deferred_figure_dir = _as_path(
        _cfg_get(config, "outputs.deferred_figure_dir", figure_dir.parent / "model_reliability_ablation_deferred")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    deferred_figure_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(_cfg_get(config, "dataset.name", "nr_s35_homology_representative"))
    subset_csv = _as_path(_cfg_get(config, "dataset.subset_csv", "data/processed/publication_evidence/nr_s35_homology_representative.csv"))
    fold_assignments_csv = _as_path(
        _cfg_get(
            config,
            "dataset.fold_assignments",
            "runs/publication_evidence_full/nonredundant_benchmarks/nr_s35_homology_representative/fold_assignments.csv",
        )
    )
    reuse = bool(_cfg_get(config, "reuse_existing", True))

    metadata, structure_dir, fasta_path = prepare_domains(
        subset_csv,
        out_dir,
        reuse=reuse,
        min_residues=int(_cfg_get(config, "domain.min_residues", 20)),
    )
    fold_assignments = _load_fold_assignments(fold_assignments_csv)

    embeddings, emb_meta, emb_path, emb_meta_path = run_esmc_embeddings(
        metadata,
        out_dir,
        weights_path=_as_path(_cfg_get(config, "esmc.weights", "data/external/esmc_weights/esmc_600m_2024_12_v0.pth")),
        device=str(_cfg_get(config, "esmc.device", "auto")),
        precision=str(_cfg_get(config, "esmc.precision", "fp32")),
        reuse=reuse,
    )
    _esmc_predictions, esmc_metrics, sim = run_embedding_knn(
        embeddings,
        emb_meta,
        fold_assignments,
        out_dir,
        k_values=[int(k) for k in _cfg_get(config, "esmc.knn_k", [1, 5, 10])],
        dataset=dataset,
    )
    np.save(out_dir / "esmc_cosine_similarity.npy", sim.astype(np.float32))

    foldseek_hits = run_foldseek_search(
        structure_dir,
        out_dir,
        foldseek_bin=_as_path(_cfg_get(config, "foldseek.binary", "foldseek")),
        threads=int(_cfg_get(config, "foldseek.threads", max(1, os.cpu_count() or 1))),
        sensitivity=float(_cfg_get(config, "foldseek.sensitivity", 9.5)),
        max_seqs=int(_cfg_get(config, "foldseek.max_seqs", 300)),
        reuse=reuse,
    )
    _foldseek_predictions, foldseek_metrics = run_foldseek_nn(
        foldseek_hits,
        metadata,
        fold_assignments,
        out_dir,
        dataset=dataset,
    )

    metrics = pd.concat([esmc_metrics, foldseek_metrics], ignore_index=True)
    metrics.to_csv(out_dir / "external_baseline_metrics.csv", index=False)

    benchmark_dir = fold_assignments_csv.parent
    benchmark_ci_csv = _as_path(
        _cfg_get(
            config,
            "figures.benchmark_ci",
            benchmark_dir.parent.parent / f"confidence_intervals_benchmark_{dataset}.csv",
        )
    )
    ablation_ci_csv = _as_path(
        _cfg_get(
            config,
            "figures.ablation_ci",
            "runs/publication_evidence_full/selected_ablation_oof/ablation_delta_confidence_intervals.csv",
        )
    )
    figure_outputs = _plot_context_figures(
        embeddings,
        emb_meta,
        foldseek_hits,
        figure_dir,
        fold_assignments=fold_assignments,
        benchmark_dir=benchmark_dir,
        benchmark_ci_csv=benchmark_ci_csv,
        ablation_ci_csv=ablation_ci_csv,
        external_dir=out_dir,
        dataset=dataset,
        seed=int(_cfg_get(config, "figures.seed", 13)),
        max_per_label=int(_cfg_get(config, "figures.max_per_label", 80)),
        network_k=int(_cfg_get(config, "figures.network_k", 5)),
        scatter_pairs=int(_cfg_get(config, "figures.scatter_pairs", 50000)),
        deferred_figure_dir=deferred_figure_dir,
    )

    outputs = {
        "domain_metadata": out_dir / "domain_metadata.csv",
        "domain_sequences": fasta_path,
        "esmc_embeddings": emb_path,
        "esmc_embedding_metadata": emb_meta_path,
        "esmc_knn_predictions": out_dir / "esmc_knn_predictions.csv",
        "esmc_knn_metrics": out_dir / "esmc_knn_metrics.csv",
        "esmc_cosine_similarity": out_dir / "esmc_cosine_similarity.npy",
        "foldseek_hits": foldseek_hits,
        "foldseek_nn_predictions": out_dir / "foldseek_nn_predictions.csv",
        "foldseek_nn_metrics": out_dir / "foldseek_nn_metrics.csv",
        "external_baseline_metrics": out_dir / "external_baseline_metrics.csv",
        **figure_outputs,
    }
    manifest_path = out_dir / "external_baseline_manifest.json"
    weights = _as_path(_cfg_get(config, "esmc.weights", ""))
    foldseek_bin = _as_path(_cfg_get(config, "foldseek.binary", ""))
    write_json(
        manifest_path,
        build_run_manifest(
            command="betlas external-baselines",
            parameters=config,
            inputs={
                "config": config_path,
                "subset_csv": subset_csv,
                "fold_assignments": fold_assignments_csv,
                "ablation_ci": ablation_ci_csv,
                "esmc_weights": weights,
                "foldseek_binary": foldseek_bin,
            },
            outputs=outputs,
            metrics={
                "dataset": dataset,
                "input_rows": int(len(metadata)),
                "prepared_domains": int((metadata["status"] == "ok").sum()),
                "esmc_embedding_rows": int(len(emb_meta)),
                "metrics_rows": int(len(metrics)),
            },
            extra={
                "esmc_weights_sha256": file_state(weights).get("sha256", ""),
                "foldseek_binary_sha256": file_state(foldseek_bin).get("sha256", ""),
                "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            },
        ),
    )
    outputs["manifest"] = manifest_path
    return ExternalBaselineResult(out_dir=out_dir, figure_dir=figure_dir, outputs=outputs, manifest_path=manifest_path)
